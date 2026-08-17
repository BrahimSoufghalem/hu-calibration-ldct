"""Evaluate trained models on the held-out test set (image-quality metrics).

Computes per-patient PSNR / SSIM (clinically windowed), RMSE (HU) and VIF at
full 512x512 resolution, plus deltas against the LDCT input baseline.
For the tissue-resolved HU audit (per-bin bias, calibration line, MSE
decomposition, threshold-crossing sensitivity), see hu_audit.py.

Usage
-----
# Full 100-patient split (default, 10 test patients):
    HU_RANGE_PRESET=benchmark python evaluate_image.py \\
        --test-dir test --runs-root runs --output eval_results

# 20-patient experiment split (5 test patients):
    HU_RANGE_PRESET=benchmark python evaluate_image.py \\
        --test-dir test --runs-root runs_20p --output eval_20p --split 20p
"""

import argparse
from glob import glob
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import config as cfg
from benchmark_data import (
    BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD,
    denormalize_to_pixel, standardize_hu,
)
from metrics import (
    compute_psnr_windowed, compute_ssim_windowed,
    compute_rmse_hu, compute_vif_hu,
)
from models import ARCH_CHOICES, build_bare_model
from twenty_patient_split import TEST_20P
from utils import (
    load_dicom_tensor, setup_reproducibility, get_device,
    sort_by_instance_number,
)


ARCH_MAP = {
    "redcnn": "RED-CNN",
    "resnet": "ResNet",
}


def get_test_set(split: str) -> set:
    if split == "20p":
        return TEST_20P
    return cfg.EXPECTED_TEST


def load_checkpoint(path: str, arch: str, device):
    state = torch.load(path, map_location=device, weights_only=False)
    meta  = state.get("meta", {}) if isinstance(state, dict) else {}

    if abs(float(meta.get("pixel_mean", BENCHMARK_PIXEL_MEAN)) - BENCHMARK_PIXEL_MEAN) > 1e-6:
        raise RuntimeError(f"[{arch}] pixel_mean mismatch in checkpoint")
    if abs(float(meta.get("pixel_std",  BENCHMARK_PIXEL_STD))  - BENCHMARK_PIXEL_STD)  > 1e-6:
        raise RuntimeError(f"[{arch}] pixel_std mismatch in checkpoint")

    model = build_bare_model(arch).to(device)
    weights = state.get("model_state_dict", state)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate_patient(pid: str, patient_dir: Path, model, device) -> dict:
    low  = sort_by_instance_number(glob(str(patient_dir / "Low_Dose"  / "*.dcm")))
    full = sort_by_instance_number(glob(str(patient_dir / "Full_Dose" / "*.dcm")))
    if len(low) != len(full):
        raise RuntimeError(f"[{pid}] slice mismatch: {len(low)} vs {len(full)}")

    body   = "Chest" if pid.upper().startswith("C") else "Abdomen"
    keys   = ("psnr", "ssim", "rmse", "vif",
              "base_psnr", "base_ssim", "base_rmse", "base_vif")
    scores = {k: [] for k in keys}

    for low_path, full_path in tqdm(
        zip(low, full), total=len(low), desc=f"  {pid}", leave=False
    ):
        low_hu  = load_dicom_tensor(low_path).to(device)
        full_hu = load_dicom_tensor(full_path).to(device)

        x       = standardize_hu(low_hu).unsqueeze(0).unsqueeze(0)
        pred_z  = model(x)

        pred_px = denormalize_to_pixel(pred_z.squeeze()).clamp(0.0, cfg.EVAL_DATA_RANGE)
        full_px = (full_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE)
        low_px  = (low_hu  + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE)

        scores["psnr"].append(compute_psnr_windowed(pred_px, full_px, body))
        scores["ssim"].append(compute_ssim_windowed(pred_px, full_px, body))
        scores["rmse"].append(compute_rmse_hu(pred_px, full_px))
        scores["vif"].append(compute_vif_hu(pred_px, full_px))
        scores["base_psnr"].append(compute_psnr_windowed(low_px, full_px, body))
        scores["base_ssim"].append(compute_ssim_windowed(low_px, full_px, body))
        scores["base_rmse"].append(compute_rmse_hu(low_px, full_px))
        scores["base_vif"].append(compute_vif_hu(low_px, full_px))

    avg = lambda k: sum(scores[k]) / max(1, len(scores[k]))
    return {
        "PatientID":     pid,
        "BodyType":      body,
        "NumSlices":     len(low),
        "PSNR":          avg("psnr"),
        "SSIM":          avg("ssim"),
        "RMSE_HU":       avg("rmse"),
        "VIF":           avg("vif"),
        "Baseline_PSNR": avg("base_psnr"),
        "Baseline_VIF":  avg("base_vif"),
        "Delta_PSNR":    avg("psnr") - avg("base_psnr"),
        "Delta_SSIM":    avg("ssim") - avg("base_ssim"),
        "Delta_VIF":     avg("vif")  - avg("base_vif"),
    }


def print_comparison(all_dfs: dict, split: str):
    n_label = "20" if split == "20p" else "100"
    print("\n" + "=" * 72)
    print(f"  {n_label}-PATIENT-SPLIT COMPARISON")
    print("  (benchmark mean/std | same data | same protocol)")
    print("=" * 72)
    metrics = ["PSNR", "SSIM", "RMSE_HU", "VIF"]
    header  = f"  {'Model':<24}" + "".join(f"{m:>11}" for m in metrics)
    for body in ["Chest", "Abdomen", "Overall"]:
        print(f"\n  [{body.upper()}]")
        print(header)
        print("  " + "-" * 68)
        for arch, df in all_dfs.items():
            sub = df if body == "Overall" else df[df["BodyType"] == body]
            if sub.empty:
                continue
            row   = sub[metrics].mean()
            label = ARCH_MAP.get(arch, arch)
            print(f"  {label:<24}" + "".join(f"{row[m]:>11.4f}" for m in metrics))
        print("  " + "-" * 68)
    print("=" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--test-dir",  default=cfg.TEST_DIR)
    p.add_argument("--output",    default="eval_results")
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--archs", default="redcnn,resnet",
                   help="Comma-separated architectures to evaluate. "
                        f"Available: {', '.join(ARCH_CHOICES)}.")
    args = p.parse_args()

    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError(
            "Run with HU_RANGE_PRESET=benchmark.\n"
            "Example: HU_RANGE_PRESET=benchmark python evaluate_image.py ..."
        )

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

    all_dfs: dict = {}
    for arch in [a.strip() for a in args.archs.split(",") if a.strip()]:
        ckpt = Path(args.runs_root) / arch / "best_model.pt"
        if not ckpt.exists():
            print(f"  Skipping {arch}: {ckpt} not found")
            continue
        print(f"\nEvaluating {ARCH_MAP.get(arch, arch)} ...")
        try:
            model = load_checkpoint(str(ckpt), arch, device)
        except Exception as e:
            print(f"  ERROR loading {arch}: {e}")
            continue

        rows = [evaluate_patient(d.name, d, model, device) for d in test_patients]
        df   = pd.DataFrame(rows)
        df["Model"] = ARCH_MAP.get(arch, arch)
        all_dfs[arch] = df
        df.to_csv(out_path / f"{arch}_results.csv", index=False)

    if not all_dfs:
        print("No checkpoints found. Train first.")
        return

    print_comparison(all_dfs, args.split)

    summary_path = out_path / "comparison.csv"
    pd.concat(all_dfs.values(), ignore_index=True).to_csv(summary_path, index=False)
    print(f"\nFull report -> {summary_path}")


if __name__ == "__main__":
    main()
