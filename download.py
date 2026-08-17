"""
LDCT-and-projection-data Downloader (Specific Patients Edition)
================================================================
Downloads DICOM data from NBIA for specific hardcoded patients:
  - Validates Full+Low dose availability 
  - Parallel fast downloads with resume support  
  - tqdm progress bars and CSV report
"""

import json
import time
import zipfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nbiatoolkit import NBIAClient
from tqdm import tqdm

from config import (
    DATA_DIR, TEST_DIR,
    DOWNLOAD_WORKERS, COLLECTION, DOWNLOAD_TIMEOUT,
    CHUNK_SIZE, NBIA_API_URL,
    EXPECTED_TEST, EXPECTED_VAL, EXPECTED_TRAIN,
)


# ═══════════════════════════════════════════
# SESSION (thread-local)
# ═══════════════════════════════════════════
def make_session():
    """Create a requests Session with retry logic."""
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    return s


_tl = threading.local()


def get_session():
    if not hasattr(_tl, "s"):
        _tl.s = make_session()
    return _tl.s


def log(msg):
    tqdm.write(msg)


# ═══════════════════════════════════════════
# STEP 1 — LOAD SERIES + SIZE INFO
# ═══════════════════════════════════════════
def fetch_series_info():
    print("\U0001f50d  Fetching series list from NBIA \u2026", flush=True)

    with NBIAClient() as client:
        all_series = client.getSeries(Collection=COLLECTION)

    print(f"\U0001f4cb  Total series received: {len(all_series)}", flush=True)

    patient_map = {}
    for s in all_series:
        desc = (s.get("SeriesDescription") or "").lower()
        pid = (s.get("PatientID") or "").strip()

        if "full dose images" in desc:
            dose = "Full"
        elif "low dose images" in desc:
            dose = "Low"
        else:
            continue

        raw_size = s.get("FileSize") or s.get("TotalSizeInBytes") or 0
        try:
            size_mb = float(raw_size) / (1024 * 1024)
        except Exception:
            size_mb = 0.0

        patient_map.setdefault(pid, {})
        if dose not in patient_map[pid]:
            patient_map[pid][dose] = {"uid": s["SeriesInstanceUID"], "size_mb": size_mb}

    return patient_map

# ═══════════════════════════════════════════
# STEP 2 — FILTER & DISTRIBUTE
# ═══════════════════════════════════════════
def select_patients(patient_map):
    """
    Filter patients with both doses available and distribute the exact 100 specified patients:
      - 90 patients (70 Train + 20 Val) to 'dataset/'
      - 10 patients to 'test/'
    """
    complete = {
        pid: data for pid, data in patient_map.items()
        if "Full" in data and "Low" in data
    }
    
    def get_total_size(pid):
        if pid in complete:
            return complete[pid]["Full"]["size_mb"] + complete[pid]["Low"]["size_mb"]
        return 0.0

    dataset_pids = sorted(list(EXPECTED_TRAIN | EXPECTED_VAL))
    test_pids = sorted(list(EXPECTED_TEST))
    selected_pids = dataset_pids + test_pids

    pid_to_dest_folder = {}
    for pid in dataset_pids:
        pid_to_dest_folder[pid] = Path(DATA_DIR)  # dataset (train + val)
    for pid in test_pids:
        pid_to_dest_folder[pid] = Path(TEST_DIR)  # test

    train_chest = [p for p in EXPECTED_TRAIN if p.startswith('C')]
    train_abdo = [p for p in EXPECTED_TRAIN if p.startswith('L')]
    val_chest = [p for p in EXPECTED_VAL if p.startswith('C')]
    val_abdo = [p for p in EXPECTED_VAL if p.startswith('L')]
    test_chest = [p for p in EXPECTED_TEST if p.startswith('C')]
    test_abdo = [p for p in EXPECTED_TEST if p.startswith('L')]

    print(f"\u2705  Explicit Distribution Strategy Status:")
    print(f"    - Dataset (train)   : {len(train_chest)} Chest + {len(train_abdo)} Abdomen = {len(EXPECTED_TRAIN)} Patients.")
    print(f"    - Dataset (val)     : {len(val_chest)} Chest + {len(val_abdo)} Abdomen = {len(EXPECTED_VAL)} Patients.")
    print(f"    - Testing (test)    : {len(test_chest)} Chest + {len(test_abdo)} Abdomen = {len(EXPECTED_TEST)} Patients.")

    total_est_mb = sum(get_total_size(p) for p in selected_pids)
    print(f"\n\U0001f4ca Total Target: {len(selected_pids)} patients (~{total_est_mb:.0f} MB estimated)\n", flush=True)

    return complete, selected_pids, pid_to_dest_folder
# ═══════════════════════════════════════════
# STEP 3 — RESUME DETECTION
# ═══════════════════════════════════════════
def is_series_complete(folder: Path) -> bool:
    if not folder.exists():
        return False
    if (folder / "data.zip").exists():
        return False
    return len(list(folder.rglob("*.dcm"))) > 0


def check_resume(selected_pids, pid_to_dest_folder):
    """
    Check disk for already-downloaded patients.
    """
    def patient_disk_status(pid):
        target_dir = pid_to_dest_folder[pid]
        p = target_dir / pid
        ok_f = is_series_complete(p / "Full_Dose")
        ok_l = is_series_complete(p / "Low_Dose")
        if ok_f and ok_l:
            return "done"
        if ok_f or ok_l:
            return "partial"
        return "none"

    already_done = [p for p in selected_pids if patient_disk_status(p) == "done"]
    partial_done = [p for p in selected_pids if patient_disk_status(p) == "partial"]
    need_download = [p for p in selected_pids if patient_disk_status(p) == "none"]

    print("\U0001f4ca  Resume check:")
    print(f"  \u2705  Already complete : {len(already_done)}")
    print(f"  \U0001f504  Partial (retry)  : {len(partial_done)}")
    print(f"  \u2b07   Not started      : {len(need_download)}\n", flush=True)

    to_download = partial_done + need_download

    # Load previous progress
    prev_results = []
    if Path("progress.json").exists():
        try:
            with open("progress.json") as f:
                prev_results = json.load(f)
            done_set = set(already_done)
            prev_results = [r for r in prev_results if r["PatientID"] in done_set]
            print(f"  \U0001f4c2  Loaded {len(prev_results)} previous results\n", flush=True)
        except Exception:
            prev_results = []

    return already_done, to_download, prev_results


# ═══════════════════════════════════════════
# STEP 4 — DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════
def download_series(uid, folder, label, global_bar):
    """Download and extract a single DICOM series ZIP from NBIA."""
    folder.mkdir(parents=True, exist_ok=True)
    zip_path = folder / "data.zip"
    session = get_session()

    for attempt in range(6):
        try:
            resp = session.get(
                NBIA_API_URL,
                params={"SeriesInstanceUID": uid, "format": "zip"},
                stream=True,
                timeout=DOWNLOAD_TIMEOUT,
            )
            resp.raise_for_status()

            total_bytes = 0
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        n = len(chunk)
                        total_bytes += n
                        global_bar.update(n)

            if total_bytes < 1000:
                raise ValueError(f"Too small ({total_bytes} B)")

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(folder)

            zip_path.unlink(missing_ok=True)
            return True, total_bytes / (1024 * 1024)

        except zipfile.BadZipFile:
            log(f"  \u26a0  Bad zip {uid[:20]} (try {attempt + 1})")
            zip_path.unlink(missing_ok=True)
            time.sleep(3 * (attempt + 1))
        except Exception as e:
            log(f"  \u26a0  Error {uid[:20]} (try {attempt + 1}): {str(e)[:80]}")
            zip_path.unlink(missing_ok=True)
            time.sleep(3 * (attempt + 1))

    return False, 0.0


def download_patient(pid, complete, pid_to_dest_folder, global_bar):
    """Download both Full Dose and Low Dose for a single patient."""
    data = complete[pid]
    target_dir = pid_to_dest_folder[pid]
    p_dir = target_dir / pid
    full_folder = p_dir / "Full_Dose"
    low_folder = p_dir / "Low_Dose"
    full_already = is_series_complete(full_folder)
    low_already = is_series_complete(low_folder)
    ptype = "Chest" if pid[0].upper() == "C" else "Abdomen"
    est_mb = data["Full"]["size_mb"] + data["Low"]["size_mb"]

    if full_already and low_already:
        actual_mb = sum(f.stat().st_size for f in p_dir.rglob("*.dcm")) / (1024 * 1024)
        log(f"\u23ed  [{pid}]  skipped")
        return {
            "PatientID": pid, "Type": ptype, "Destination": target_dir.name,
            "Estimated_MB": round(est_mb, 2), "Downloaded_MB": round(actual_mb, 2),
            "Full_Dose_MB": "skipped", "Low_Dose_MB": "skipped",
            "Avg_Speed_MBps": "-", "status": "success",
            "full_ok": True, "low_ok": True,
        }

    log(f"\u2b07  [{pid}]  (~{est_mb:.0f} MB) [{ptype}]")
    t0 = time.time()

    if full_already:
        ok_full = True
        mb_full = sum(f.stat().st_size for f in full_folder.rglob("*.dcm")) / (1024 * 1024)
    else:
        ok_full, mb_full = download_series(data["Full"]["uid"], full_folder, f"{pid}/Full", global_bar)

    if low_already:
        ok_low = True
        mb_low = sum(f.stat().st_size for f in low_folder.rglob("*.dcm")) / (1024 * 1024)
    else:
        ok_low, mb_low = download_series(data["Low"]["uid"], low_folder, f"{pid}/Low", global_bar)

    elapsed = max(time.time() - t0, 0.1)
    total_dl = mb_full + mb_low
    speed = total_dl / elapsed

    status = "success" if (ok_full and ok_low) else ("partial" if (ok_full or ok_low) else "failed")
    icon = {"success": "\u2705", "partial": "\U0001f504", "failed": "\u274c"}[status]
    log(f"{icon}  [{pid}]  {status}  {total_dl:.1f} MB  @ {speed:.2f} MB/s")

    return {
        "PatientID": pid, "Type": ptype, "Destination": target_dir.name,
        "Estimated_MB": round(est_mb, 2), "Downloaded_MB": round(total_dl, 2),
        "Full_Dose_MB": round(mb_full, 2), "Low_Dose_MB": round(mb_low, 2),
        "Avg_Speed_MBps": round(speed, 2), "status": status,
        "full_ok": ok_full, "low_ok": ok_low,
    }


# ═══════════════════════════════════════════
# STEP 5 — PARALLEL DOWNLOAD + REPORT
# ═══════════════════════════════════════════
def run_downloads(to_download, complete, pid_to_dest_folder, prev_results):
    """Execute parallel downloads and generate a CSV report."""
    def _total_size(pid):
        d = complete[pid]
        return d["Full"]["size_mb"] + d["Low"]["size_mb"]

    remaining_mb = sum(_total_size(p) for p in to_download)
    print(f"\U0001f680  Will download {len(to_download)} patients  (~{remaining_mb:.0f} MB)\n", flush=True)

    global_bar = tqdm(
        total=int(remaining_mb * 1024 * 1024),
        unit="B", unit_scale=True, unit_divisor=1024,
        desc="\U0001f4e5 Downloading Specific Patients", colour="cyan", leave=True, dynamic_ncols=True,
    )

    results = list(prev_results)
    start = time.time()

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(download_patient, pid, complete, pid_to_dest_folder, global_bar): pid
            for pid in to_download
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            with open("progress.json", "w") as f:
                json.dump(results, f, indent=2)

    global_bar.close()

    # ── CSV Report ──
    df = pd.DataFrame(results)
    cols = [
        "PatientID", "Type", "Destination", "Estimated_MB", "Downloaded_MB",
        "Full_Dose_MB", "Low_Dose_MB", "Avg_Speed_MBps", "status", "full_ok", "low_ok",
    ]
    df = df.reindex(columns=cols)
    df = df.sort_values(["Destination", "Type", "Downloaded_MB"])
    df.to_csv("download_report.csv", index=False)

    # ── Summary ──
    total_time = time.time() - start
    overall_mb = df["Downloaded_MB"].apply(lambda x: x if isinstance(x, (int, float)) else 0).sum()
    overall_speed = overall_mb / max(total_time, 1)

    print("\n" + "=" * 55)
    print(f"\U0001f389  DONE in {total_time / 60:.1f} minutes")
    print(f"\u26a1  Average speed : {overall_speed:.2f} MB/s")
    print(f"\U0001f4e6  Total downloaded : {overall_mb:.0f} MB")
    print(f"\U0001f4ca  Report  : download_report.csv\n")


# ═══════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════
def main():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEST_DIR).mkdir(parents=True, exist_ok=True) 
    patient_map = fetch_series_info()
    complete, selected_pids, pid_to_dest_folder = select_patients(patient_map)

    if not selected_pids:
        print("\u274c  No target patients found on NBIA or lacking criteria. Exiting.")
        return

    already_done, to_download, prev_results = check_resume(selected_pids, pid_to_dest_folder)

    if not to_download:
        print("\U0001f389  All target patients are already downloaded!")
        return

    run_downloads(to_download, complete, pid_to_dest_folder, prev_results)

if __name__ == "__main__":
    main()
