"""Phase 1 -- Tissue-resolved HU audit (the Audit arm of the study).

Quantifies, per model and per test patient, HOW the HU error is structured --
before any new loss or head is introduced. Every later arm (B..G) must be
justified by what this audit reveals on the baselines.

WHAT IS MEASURED (per patient, in physical HU)
----------------------------------------------
1. SoftBias_<tissue>    : Gaussian-weighted mean error inside each tissue
                          bin. Soft membership avoids the hard-bin edge
                          instability observed in earlier work.
2. HardBias_<tissue>    : classic hard-interval bias, kept for direct
                          comparability with previous results.
3. MSE decomposition    : per bin, the exact conditional decomposition
                          MSE_k = b_k^2 + Var(error | bin k), reported as
                          MSE_<t>, BiasSq_<t>, Var_<t>.
4. Calibration line     : weighted least-squares fit of per-bin mean
                          HU_pred against per-bin mean HU_ref.
                          Ideal: slope alpha = 1, intercept beta = 0.
5. Threshold crossings  : fraction of pixels whose classification against
                          fixed HU thresholds flips between reference and
                          prediction (threshold-crossing sensitivity
                          analysis; NOT a clinical endpoint claim).
6. Identity curve       : mean HU_pred per fine HU_ref bin, exported to CSV
                          for identity plots (HU_pred vs HU_ref).

Usage
-----
HU_RANGE_PRESET=benchmark python hu_audit.py \\
    --test-dir test --runs-root runs --output hu_audit \\
    --archs redcnn,resnet --include-input

# Arm D: also audit trunk+head (loads <heads_root>/<arch>/best_head.pt and
# adds a \"<Model> + Head\" row next to the bare trunk):
HU_RANGE_PRESET=benchmark python hu_audit.py \\
    --test-dir test --runs-root runs --heads-root runs_armD \\
    --archs redcnn --include-input --output hu_audit_d
"""

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config as cfg
from benchmark_data import denormalize_to_pixel, standardize_hu
from calibration_head import load_head
from evaluate_image import ARCH_MAP, get_test_set, load_checkpoint
from utils import (
    load_dicom_tensor, setup_reproducibility, get_device,
    sort_by_instance_number,
)


# Fixed physical tissue intervals (HU), identical to previous work for
# comparability. Soft membership is a Gaussian around each bin center.
TISSUE_BINS = (
    ("AirLung", -1024.0, -500.0),
    ("FatLow",   -500.0, -200.0),
    ("Soft",     -200.0,  200.0),
    ("Dense",     200.0,  600.0),
    ("Bone",      600.0, 1900.0),
)
SOFT_SIGMA_FRACTION = 0.25   # sigma_k = 0.25 * bin width
THRESHOLDS_HU = (-950.0, 0.0, 100.0, 130.0)
IDENTITY_BIN_WIDTH = 25.0    # fine bins for the identity curve


def soft_bin_params():
    """(name, center, sigma) for every tissue bin."""
    return [
        (name, 0.5 * (lo + hi), SOFT_SIGMA_FRACTION * (hi - lo))
        for name, lo, hi in TISSUE_BINS
    ]


def make_model_forward(model, head=None):
    """LDCT slice in physical HU -> denoised slice in physical HU.

    If `head` is given (arm D+), the calibration head is applied to the
    trunk output in the standardized domain before denormalization.
    """
    @torch.no_grad()
    def forward(low_hu):
        x = standardize_hu(low_hu).unsqueeze(0).unsqueeze(0)
        z = model(x)
        if head is not None:
            z = head(z)
        pred_px = denormalize_to_pixel(z.squeeze())
        pred_px = pred_px.clamp(0.0, cfg.EVAL_DATA_RANGE)
        return pred_px - cfg.HU_OFFSET
    return forward


@torch.no_grad()
def input_forward(low_hu):
    """Audit the raw LDCT input itself (no-denoising reference row)."""
    return low_hu.clamp(cfg.A_MIN, cfg.A_MAX)


@torch.no_grad()
def audit_patient(pid: str, patient_dir: Path, forward_fn, device):
    low  = sort_by_instance_number(glob(str(patient_dir / "Low_Dose"  / "*.dcm")))
    full = sort_by_instance_number(glob(str(patient_dir / "Full_Dose" / "*.dcm")))
    if len(low) != len(full):
        raise RuntimeError(f"[{pid}] slice mismatch: {len(low)} vs {len(full)}")

    body = "Chest" if pid.upper().startswith("C") else "Abdomen"
    soft = soft_bin_params()

    acc  = {name: {"w": 0.0, "we": 0.0, "we2": 0.0, "wref": 0.0, "wpred": 0.0}
            for name, _, _ in soft}
    hard = {name: {"n": 0, "e": 0.0} for name, _, _ in TISSUE_BINS}
    thr  = {t: {"ref_pos": 0, "pred_pos": 0, "disagree": 0, "n": 0}
            for t in THRESHOLDS_HU}

    n_id = int((cfg.A_MAX - cfg.A_MIN) / IDENTITY_BIN_WIDTH)
    id_count = np.zeros(n_id, dtype=np.float64)
    id_sum   = np.zeros(n_id, dtype=np.float64)

    for low_path, full_path in tqdm(
        zip(low, full), total=len(low), desc=f"  {pid}", leave=False
    ):
        low_hu  = load_dicom_tensor(low_path).to(device)
        full_hu = load_dicom_tensor(full_path).to(device).clamp(cfg.A_MIN, cfg.A_MAX)
        pred_hu = forward_fn(low_hu)
        err = pred_hu - full_hu

        for name, c, s in soft:
            w = torch.exp(-0.5 * ((full_hu - c) / s) ** 2)
            acc[name]["w"]     += float(w.sum())
            acc[name]["we"]    += float((w * err).sum())
            acc[name]["we2"]   += float((w * err * err).sum())
            acc[name]["wref"]  += float((w * full_hu).sum())
            acc[name]["wpred"] += float((w * pred_hu).sum())

        for name, lo_b, hi_b in TISSUE_BINS:
            mask = (full_hu >= lo_b) & (full_hu < hi_b)
            n = int(mask.sum())
            if n:
                hard[name]["n"] += n
                hard[name]["e"] += float(err[mask].sum())

        for t in THRESHOLDS_HU:
            ref_pos  = full_hu > t
            pred_pos = pred_hu > t
            thr[t]["ref_pos"]  += int(ref_pos.sum())
            thr[t]["pred_pos"] += int(pred_pos.sum())
            thr[t]["disagree"] += int((ref_pos ^ pred_pos).sum())
            thr[t]["n"]        += int(ref_pos.numel())

        ref_np  = full_hu.detach().cpu().numpy().ravel()
        pred_np = pred_hu.detach().cpu().numpy().ravel()
        idx = np.clip(
            ((ref_np - cfg.A_MIN) // IDENTITY_BIN_WIDTH).astype(np.int64),
            0, n_id - 1,
        )
        id_count += np.bincount(idx, minlength=n_id)
        id_sum   += np.bincount(idx, weights=pred_np, minlength=n_id)

    row = {"PatientID": pid, "BodyType": body, "NumSlices": len(low)}
    mu_ref, mu_pred, mass = [], [], []
    for name, _, _ in soft:
        a = acc[name]
        if a["w"] > 0:
            b   = a["we"]  / a["w"]
            mse = a["we2"] / a["w"]
            row[f"SoftBias_{name}"] = b
            row[f"MSE_{name}"]      = mse
            row[f"BiasSq_{name}"]   = b * b
            row[f"Var_{name}"]      = mse - b * b
            mu_ref.append(a["wref"] / a["w"])
            mu_pred.append(a["wpred"] / a["w"])
            mass.append(a["w"])
        else:
            for key in ("SoftBias", "MSE", "BiasSq", "Var"):
                row[f"{key}_{name}"] = float("nan")

    for name, _, _ in TISSUE_BINS:
        h = hard[name]
        row[f"HardBias_{name}"] = h["e"] / h["n"] if h["n"] else float("nan")

    if len(mu_ref) >= 2:
        alpha, beta = np.polyfit(
            np.asarray(mu_ref), np.asarray(mu_pred), 1,
            w=np.sqrt(np.asarray(mass)),
        )
        row["CalibSlope_alpha"]    = float(alpha)
        row["CalibIntercept_beta"] = float(beta)
    else:
        row["CalibSlope_alpha"]    = float("nan")
        row["CalibIntercept_beta"] = float("nan")

    for t in THRESHOLDS_HU:
        d = thr[t]
        tag = f"{int(t)}HU"
        row[f"ThrDisagree_{tag}_pct"] = 100.0 * d["disagree"] / max(1, d["n"])
        row[f"ThrRefPos_{tag}_pct"]   = 100.0 * d["ref_pos"]  / max(1, d["n"])
        row[f"ThrPredPos_{tag}_pct"]  = 100.0 * d["pred_pos"] / max(1, d["n"])

    return row, id_count, id_sum


def print_summary(all_dfs: dict):
    metrics = ["SoftBias_AirLung", "SoftBias_Soft", "SoftBias_Bone",
               "CalibSlope_alpha", "CalibIntercept_beta",
               "ThrDisagree_130HU_pct"]
    ideal = {"SoftBias_AirLung": "0 HU", "SoftBias_Soft": "0 HU",
             "SoftBias_Bone": "0 HU", "CalibSlope_alpha": "1.000",
             "CalibIntercept_beta": "0.00", "ThrDisagree_130HU_pct": "0 %"}
    print("\n" + "=" * 118)
    print("  TISSUE-RESOLVED HU AUDIT (Phase 1)")
    print("=" * 118)
    header    = f"  {'Model':<24}" + "".join(f"{m:>18}" for m in metrics)
    ideal_row = f"  {'(ideal)':<24}" + "".join(f"{ideal[m]:>18}" for m in metrics)
    for body in ["Chest", "Abdomen", "Overall"]:
        print(f"\n  [{body.upper()}]")
        print(header)
        print(ideal_row)
        print("  " + "-" * 114)
        for label, df in all_dfs.items():
            sub = df if body == "Overall" else df[df["BodyType"] == body]
            if sub.empty:
                continue
            r = sub[metrics].mean()
            print(f"  {label:<24}" + "".join(f"{r[m]:>18.4f}" for m in metrics))
        print("  " + "-" * 114)
    print("\n  Per-bin decomposition (MSE_k = BiasSq_k + Var_k) is in the CSVs.")
    print("=" * 118)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--test-dir",  default=cfg.TEST_DIR)
    p.add_argument("--output",    default="hu_audit")
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--archs", default="redcnn,resnet")
    p.add_argument("--include-input", action="store_true",
                   help="Also audit the raw LDCT input (no denoising).")
    p.add_argument("--heads-root", default=None,
                   help="Arm D: root dir with <arch>/best_head.pt "
                        "calibration heads. Adds a '<Model> + Head' target "
                        "next to each bare trunk.")
    args = p.parse_args()

    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Run with HU_RANGE_PRESET=benchmark.")

    setup_reproducibility()
    device   = get_device()
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    test_ids = get_test_set(args.split)
    test_patients = sorted([
        d for d in Path(args.test_dir).iterdir()
        if d.is_dir()
        and d.name in test_ids
        and (d / "Low_Dose").exists()
        and (d / "Full_Dose").exists()
    ])
    if not test_patients:
        raise RuntimeError(
            f"No test patients found in '{args.test_dir}' "
            f"matching the {args.split} split.\n"
            f"Expected IDs: {sorted(test_ids)}"
        )

    print(f"Split        : {args.split} ({len(test_patients)} patients found)")
    print(f"Test patients: {[d.name for d in test_patients]}")

    targets = []
    if args.include_input:
        targets.append(("input", "LDCT input", input_forward))
    for arch in [a.strip() for a in args.archs.split(",") if a.strip()]:
        ckpt = Path(args.runs_root) / arch / "best_model.pt"
        if not ckpt.exists():
            print(f"  Skipping {arch}: {ckpt} not found")
            continue
        model = load_checkpoint(str(ckpt), arch, device)
        label = ARCH_MAP.get(arch, arch)
        targets.append((arch, label, make_model_forward(model)))
        if args.heads_root:
            head_ckpt = Path(args.heads_root) / arch / "best_head.pt"
            if head_ckpt.exists():
                head = load_head(str(head_ckpt), device)
                targets.append((f"{arch}_head", f"{label} + Head",
                                make_model_forward(model, head)))
            else:
                print(f"  No head for {arch}: {head_ckpt} not found")

    if not targets:
        print("Nothing to audit. Train baselines first or pass --include-input.")
        return

    n_id = int((cfg.A_MAX - cfg.A_MIN) / IDENTITY_BIN_WIDTH)
    centers = cfg.A_MIN + (np.arange(n_id) + 0.5) * IDENTITY_BIN_WIDTH

    all_dfs: dict = {}
    for key, label, forward_fn in targets:
        print(f"\nAuditing {label} ...")
        rows = []
        id_count = np.zeros(n_id, dtype=np.float64)
        id_sum   = np.zeros(n_id, dtype=np.float64)
        for d in test_patients:
            row, c, s = audit_patient(d.name, d, forward_fn, device)
            rows.append(row)
            id_count += c
            id_sum   += s
        df = pd.DataFrame(rows)
        df["Model"] = label
        all_dfs[label] = df
        df.to_csv(out_path / f"{key}_hu_audit.csv", index=False)

        mean_pred = np.divide(
            id_sum, id_count,
            out=np.full(n_id, np.nan), where=id_count > 0,
        )
        pd.DataFrame({
            "BinCenter_HU": centers,
            "PixelCount":   id_count,
            "MeanPred_HU":  mean_pred,
        }).to_csv(out_path / f"{key}_identity_curve.csv", index=False)

    print_summary(all_dfs)

    summary_path = out_path / "hu_audit_comparison.csv"
    pd.concat(all_dfs.values(), ignore_index=True).to_csv(summary_path, index=False)
    print(f"\nFull report -> {summary_path}")


if __name__ == "__main__":
    main()
