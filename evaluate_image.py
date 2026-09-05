"""Evaluate trained models on the held-out test set (image-quality metrics).

Computes per-patient PSNR / SSIM (clinically windowed), RMSE (HU) and VIF at
full 512x512 resolution, plus deltas against the LDCT input baseline.
When ``--heads-root`` is supplied, the same pass also evaluates the selected
calibration-head checkpoint and reports deltas against the bare trunk
(Cycle 00). Full-slice heads receive context from the uncropped low-dose slice.
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

# Post-hoc full-slice head (Cycle 00 trunk vs selected checkpoint):
    HU_RANGE_PRESET=benchmark python evaluate_image.py \\
        --test-dir test --runs-root runs \\
        --heads-root runs_armE_full_slice --archs redcnn \\
        --output eval_armE_full_slice
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
from calibration_head import (
    ContextCalibrationHead, SpatialGatedCalibrationHead, load_head,
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


def _head_checkpoint_meta(path: Path, arch: str, trunk_path: Path,
                          split: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    checkpoint_arch = meta.get("architecture")
    if checkpoint_arch is not None and checkpoint_arch != arch:
        raise RuntimeError(
            f"Head architecture mismatch: expected {arch}, got {checkpoint_arch}")
    if bool(meta.get("joint", False)):
        raise RuntimeError(
            "evaluate_image.py compares a post-hoc head with its frozen trunk "
            "(Cycle 00); joint head checkpoints require their jointly trained "
            "model checkpoint and do not have this baseline.")
    checkpoint_split = meta.get("split")
    if checkpoint_split is None:
        raise RuntimeError("Head checkpoint does not record its data split")
    if checkpoint_split != split:
        raise RuntimeError(
            f"Head split mismatch: --split is {split}, checkpoint used "
            f"{checkpoint_split}")
    recorded_trunk = meta.get("trunk_checkpoint")
    if not recorded_trunk:
        raise RuntimeError(
            "Head checkpoint does not record the frozen trunk checkpoint")
    recorded_trunk_path = Path(recorded_trunk).expanduser().resolve()
    actual_trunk_path = trunk_path.expanduser().resolve()
    if recorded_trunk_path != actual_trunk_path:
        raise RuntimeError(
            "Head/trunk checkpoint mismatch: head was trained on "
            f"'{recorded_trunk_path}', but evaluation loaded "
            f"'{actual_trunk_path}'")
    if abs(float(meta.get("pixel_mean", BENCHMARK_PIXEL_MEAN))
           - BENCHMARK_PIXEL_MEAN) > 1e-6:
        raise RuntimeError("Head pixel_mean mismatch")
    if abs(float(meta.get("pixel_std", BENCHMARK_PIXEL_STD))
           - BENCHMARK_PIXEL_STD) > 1e-6:
        raise RuntimeError("Head pixel_std mismatch")
    return {
        "HeadSeed": meta.get("seed"),
        "HeadIteration": payload.get("iteration"),
        "HeadSelectBy": payload.get("select_by", meta.get("select_by")),
    }


def _apply_calibration_head(head, x: torch.Tensor, z: torch.Tensor,
                            body: str) -> torch.Tensor:
    """Apply a loaded head with the context source used during training."""
    if not isinstance(head, ContextCalibrationHead):
        return head(z)

    if head.full_slice_context:
        context = head.inferred_context(x)
    elif head.oracle:
        context = head.oracle_context_from_bodies(z, [body] * z.shape[0])
    else:
        context = head.inferred_context(z)
    if isinstance(head, SpatialGatedCalibrationHead):
        return z + head.correction(z, context=context, source=x)
    return z + head.correction(z, context=context)


def _metric_values(pred_px: torch.Tensor, full_px: torch.Tensor,
                   body: str) -> dict:
    return {
        "PSNR": compute_psnr_windowed(pred_px, full_px, body),
        "SSIM": compute_ssim_windowed(pred_px, full_px, body),
        "RMSE_HU": compute_rmse_hu(pred_px, full_px),
        "VIF": compute_vif_hu(pred_px, full_px),
    }


def _mean_metrics(scores: list[dict]) -> dict:
    return {
        metric: sum(row[metric] for row in scores) / max(1, len(scores))
        for metric in ("PSNR", "SSIM", "RMSE_HU", "VIF")
    }


def _patient_row(pid: str, body: str, num_slices: int, stage: str,
                 metrics: dict, input_metrics: dict,
                 trunk_metrics: dict) -> dict:
    row = {
        "PatientID": pid,
        "BodyType": body,
        "NumSlices": num_slices,
        "EvaluationStage": stage,
        **metrics,
    }
    for metric in ("PSNR", "SSIM", "RMSE_HU", "VIF"):
        row[f"Baseline_{metric}"] = input_metrics[metric]
        row[f"Delta_vs_Input_{metric}"] = metrics[metric] - input_metrics[metric]
        row[f"Delta_vs_Trunk_{metric}"] = metrics[metric] - trunk_metrics[metric]

    # Preserve the old delta columns; they are deltas against the LDCT input.
    row["Delta_PSNR"] = row["Delta_vs_Input_PSNR"]
    row["Delta_SSIM"] = row["Delta_vs_Input_SSIM"]
    row["Delta_RMSE_HU"] = row["Delta_vs_Input_RMSE_HU"]
    row["Delta_VIF"] = row["Delta_vs_Input_VIF"]
    return row


@torch.no_grad()
def evaluate_patient(pid: str, patient_dir: Path, model, device,
                     head=None) -> dict:
    low  = sort_by_instance_number(glob(str(patient_dir / "Low_Dose"  / "*.dcm")))
    full = sort_by_instance_number(glob(str(patient_dir / "Full_Dose" / "*.dcm")))
    if len(low) != len(full):
        raise RuntimeError(f"[{pid}] slice mismatch: {len(low)} vs {len(full)}")
    if not low:
        raise RuntimeError(f"[{pid}] no paired DICOM slices found")

    body   = "Chest" if pid.upper().startswith("C") else "Abdomen"
    scores = {"input": [], "trunk": []}
    if head is not None:
        scores["head"] = []

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

        scores["input"].append(_metric_values(low_px, full_px, body))
        scores["trunk"].append(_metric_values(pred_px, full_px, body))
        if head is not None:
            head_z = _apply_calibration_head(head, x, pred_z, body)
            head_px = denormalize_to_pixel(head_z.squeeze()).clamp(
                0.0, cfg.EVAL_DATA_RANGE)
            scores["head"].append(_metric_values(head_px, full_px, body))

    means = {stage: _mean_metrics(values) for stage, values in scores.items()}
    rows = {
        "trunk": _patient_row(
            pid, body, len(low),
            ("Cycle 00 (frozen trunk)" if head is not None
             else "Trunk checkpoint"),
            means["trunk"], means["input"], means["trunk"]),
    }
    if head is not None:
        rows["head"] = _patient_row(
            pid, body, len(low), "Selected head checkpoint",
            means["head"], means["input"], means["trunk"])
    return rows


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
        for key, df in all_dfs.items():
            sub = df if body == "Overall" else df[df["BodyType"] == body]
            if sub.empty:
                continue
            row   = sub[metrics].mean()
            label = str(sub["Model"].iloc[0]) if "Model" in sub else key
            print(f"  {label:<24}" + "".join(f"{row[m]:>11.4f}" for m in metrics))
        print("  " + "-" * 68)
    print("=" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--heads-root", default=None,
                   help="Optional root containing <arch>/best_head.pt. "
                        "Evaluates Cycle 00 and the selected calibration head "
                        "with the context mode stored in its checkpoint.")
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

        head = None
        head_ckpt = None
        head_meta = {}
        context_mode = None
        if args.heads_root:
            head_ckpt = Path(args.heads_root) / arch / "best_head.pt"
            if head_ckpt.exists():
                try:
                    head_meta = _head_checkpoint_meta(
                        head_ckpt, arch, ckpt, args.split)
                    head = load_head(str(head_ckpt), device)
                except Exception as e:
                    print(f"  ERROR loading head {head_ckpt}: {e}")
                    continue
                context_mode = (
                    "spatial-full-slice"
                    if isinstance(head, SpatialGatedCalibrationHead)
                    else "full-slice" if getattr(head, "full_slice_context", False)
                    else "oracle" if getattr(head, "oracle", False)
                    else "inferred" if isinstance(head, ContextCalibrationHead)
                    else "intensity"
                )
                print(f"  Head context : {context_mode}")
            else:
                raise FileNotFoundError(
                    f"--heads-root requested a paired comparison, but no "
                    f"head exists for {arch}: {head_ckpt}")

        patient_results = [
            evaluate_patient(d.name, d, model, device, head=head)
            for d in test_patients
        ]
        label = ARCH_MAP.get(arch, arch)

        trunk_df = pd.DataFrame([result["trunk"] for result in patient_results])
        trunk_df["Model"] = f"{label} (Cycle 00)" if head is not None else label
        trunk_df["TrunkCheckpoint"] = str(ckpt)
        all_dfs[arch] = trunk_df
        trunk_df.to_csv(out_path / f"{arch}_results.csv", index=False)

        if head is not None:
            head_df = pd.DataFrame([result["head"] for result in patient_results])
            head_df["Model"] = f"{label} + Head"
            head_df["TrunkCheckpoint"] = str(ckpt)
            head_df["HeadCheckpoint"] = str(head_ckpt)
            head_df["HeadContextMode"] = context_mode
            for key, value in head_meta.items():
                head_df[key] = value
            all_dfs[f"{arch}_head"] = head_df
            head_df.to_csv(out_path / f"{arch}_head_results.csv", index=False)

    if not all_dfs:
        print("No checkpoints found. Train first.")
        return

    print_comparison(all_dfs, args.split)

    summary_path = out_path / "comparison.csv"
    pd.concat(all_dfs.values(), ignore_index=True).to_csv(summary_path, index=False)
    print(f"\nFull report -> {summary_path}")


if __name__ == "__main__":
    main()
