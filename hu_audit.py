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
                           analysis; NOT a clinical endpoint claim). Exports
                           directional error prevalence plus class-conditional
                           false-positive and false-negative rates explicitly.
6. Identity curve       : mean HU_pred per fine HU_ref bin, exported to CSV
                           for identity plots (HU_pred vs HU_ref).
7. Head threshold diagnostic (when --heads-root is used): paired trunk/head
                           crossing direction, distance from 130 HU before the
                           head, and the actual per-slice T(130)-130 correction.

Usage
-----
HU_RANGE_PRESET=benchmark python hu_audit.py \\
    --test-dir test --runs-root runs --output hu_audit \\
    --archs redcnn,resnet --include-input

# Arms D/E: also audit trunk+head (loads <heads_root>/<arch>/best_head.pt
# and adds a \"<Model> + Head\" row next to the bare trunk). Oracle heads
# (arm E v2 diagnostic) are handled automatically: the ground-truth body
# type of each test patient is fed to the head.
HU_RANGE_PRESET=benchmark python hu_audit.py \\
    --test-dir test --runs-root runs --heads-root runs_armD \\
    --archs redcnn --include-input --output hu_audit_d
"""

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config as cfg
from benchmark_data import (
    BENCHMARK_PIXEL_STD, denormalize_to_pixel, standardize_hu,
)
from calibration_head import (
    ContextCalibrationHead, SpatialGatedCalibrationHead, load_head,
)
from evaluate_image import (
    ARCH_MAP, _head_checkpoint_meta, get_test_set, load_checkpoint,
)
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
HEAD_DIAGNOSTIC_THRESHOLD_HU = 130.0


def soft_bin_params():
    """(name, center, sigma) for every tissue bin."""
    return [
        (name, 0.5 * (lo + hi), SOFT_SIGMA_FRACTION * (hi - lo))
        for name, lo, hi in TISSUE_BINS
    ]


def _head_context(head, x, z, body):
    if not isinstance(head, ContextCalibrationHead):
        return None
    if head.full_slice_context:
        return head.inferred_context(x)
    if head.oracle:
        return head.oracle_context_from_bodies(z, [body] * z.shape[0])
    return head.inferred_context(z)


def _apply_head(head, x, z, body):
    context = _head_context(head, x, z, body)
    if isinstance(head, SpatialGatedCalibrationHead):
        return z + head.correction(z, context=context, source=x), context
    if isinstance(head, ContextCalibrationHead):
        return z + head.correction(z, context=context), context
    return head(z), None


def _threshold_counts(ref_hu, pred_hu, threshold):
    ref_pos = ref_hu > threshold
    pred_pos = pred_hu > threshold
    false_pos = (~ref_pos) & pred_pos
    false_neg = ref_pos & (~pred_pos)
    return {
        "ref_pos": int(ref_pos.sum()),
        "pred_pos": int(pred_pos.sum()),
        "false_pos": int(false_pos.sum()),
        "false_neg": int(false_neg.sum()),
        "disagree": int((false_pos | false_neg).sum()),
        "n": int(ref_pos.numel()),
    }


def _threshold_metrics(counts):
    n = max(1, counts["n"])
    ref_pos = counts["ref_pos"]
    ref_neg = counts["n"] - ref_pos
    return {
        "Disagree_pct": 100.0 * counts["disagree"] / n,
        "RefPos_pct": 100.0 * ref_pos / n,
        "PredPos_pct": 100.0 * counts["pred_pos"] / n,
        "FalsePosPrevalence_pct": 100.0 * counts["false_pos"] / n,
        "FalseNegPrevalence_pct": 100.0 * counts["false_neg"] / n,
        "FalsePositiveRate_pct": (100.0 * counts["false_pos"] / ref_neg
                                  if ref_neg else float("nan")),
        "FalseNegativeRate_pct": (100.0 * counts["false_neg"] / ref_pos
                                  if ref_pos else float("nan")),
    }


def _threshold_counts_grid(ref_hu, pred_hu, thresholds):
    """Exact crossing counts for a sorted threshold grid in one vectorized pass."""
    output_thresholds = np.asarray(thresholds, dtype=np.float64)
    if output_thresholds.ndim != 1 or output_thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional grid")
    if np.any(np.diff(output_thresholds) <= 0):
        raise ValueError("thresholds must be strictly increasing")

    ref = ref_hu.detach().cpu().numpy().reshape(-1)
    pred = pred_hu.detach().cpu().numpy().reshape(-1)
    thresholds = output_thresholds.astype(ref.dtype)
    n = ref.size
    ref_pos = n - np.searchsorted(np.sort(ref), thresholds, side="right")
    pred_pos = n - np.searchsorted(np.sort(pred), thresholds, side="right")

    def interval_counts(low, high):
        starts = np.searchsorted(thresholds, low, side="left")
        stops = np.searchsorted(thresholds, high, side="left")
        delta = (np.bincount(starts, minlength=thresholds.size + 1)
                 - np.bincount(stops, minlength=thresholds.size + 1))
        return np.cumsum(delta)[:thresholds.size]

    upward = pred > ref
    downward = ref > pred
    false_pos = interval_counts(ref[upward], pred[upward])
    false_neg = interval_counts(pred[downward], ref[downward])
    return {
        float(threshold): {
            "ref_pos": int(ref_pos[i]),
            "pred_pos": int(pred_pos[i]),
            "false_pos": int(false_pos[i]),
            "false_neg": int(false_neg[i]),
            "disagree": int(false_pos[i] + false_neg[i]),
            "n": int(n),
        }
        for i, threshold in enumerate(output_thresholds)
    }


def _threshold_tag(threshold):
    return f"{float(threshold):.12g}HU"


def make_model_forward(model, head=None, body=None):
    """LDCT slice in physical HU -> denoised slice in physical HU.

    If `head` is given (arms D/E), the calibration head is applied to the
    trunk output in the standardized domain before denormalization. For
    oracle heads, set `head.oracle_body` before calling (done per patient
    in main()). Full-slice-context heads derive their detached context from
    the same uncropped low-dose slice supplied to the frozen trunk.
    """
    @torch.no_grad()
    def forward(low_hu):
        x = standardize_hu(low_hu).unsqueeze(0).unsqueeze(0)
        z = model(x)
        if head is not None:
            if body is None:
                body_name = getattr(head, "oracle_body", None)
                if getattr(head, "oracle", False) and body_name is None:
                    raise RuntimeError("Oracle head requires a body type")
            else:
                body_name = body
            z, _ = _apply_head(head, x, z, body_name)
        pred_px = denormalize_to_pixel(z.squeeze())
        pred_px = pred_px.clamp(0.0, cfg.EVAL_DATA_RANGE)
        return pred_px - cfg.HU_OFFSET
    return forward


@torch.no_grad()
def input_forward(low_hu):
    """Audit the raw LDCT input itself (no-denoising reference row)."""
    return low_hu.clamp(cfg.A_MIN, cfg.A_MAX)


_DISTANCE_BANDS_HU = (1.0, 2.0, 5.0, 10.0, 20.0)


def _mask_distance_stats(mask, distance_hu):
    count = int(mask.sum())
    stats = {
        "Count": count,
        "MeanDistance_HU": (float(distance_hu[mask].sum()) / count
                            if count else float("nan")),
    }
    for band in _DISTANCE_BANDS_HU:
        tag = f"Within{int(band)}HU_pct"
        stats[tag] = (100.0 * int((mask & (distance_hu <= band)).sum()) / count
                      if count else float("nan"))
    return stats


def _prefixed_stats(row, prefix, stats):
    for key, value in stats.items():
        row[f"{prefix}_{key}"] = value


def _paired_threshold_stats(ref_hu, trunk_hu, head_hu, threshold):
    ref_pos = ref_hu > threshold
    trunk_pos = trunk_hu > threshold
    head_pos = head_hu > threshold
    trunk_wrong = ref_pos ^ trunk_pos
    head_wrong = ref_pos ^ head_pos
    masks = {
        "FlipUp": (~trunk_pos) & head_pos,
        "FlipDown": trunk_pos & (~head_pos),
        "NewDisagree": head_wrong & (~trunk_wrong),
        "ResolvedDisagree": trunk_wrong & (~head_wrong),
    }
    distance = (trunk_hu - threshold).abs()
    return {name: _mask_distance_stats(mask, distance)
            for name, mask in masks.items()}, masks, distance


def _accumulate_event_stats(total, stats, mask, distance):
    total["count"] += stats["Count"]
    if not stats["Count"]:
        return
    total["distance_sum"] += float(distance[mask].sum())
    for band in _DISTANCE_BANDS_HU:
        total["bands"][band] += int((mask & (distance <= band)).sum())


def _finalize_event_stats(total, total_pixels):
    count = total["count"]
    row = {
        "Count": count,
        "pct": 100.0 * count / total_pixels,
        "MeanDistance_HU": (total["distance_sum"] / count
                            if count else float("nan")),
    }
    for band in _DISTANCE_BANDS_HU:
        row[f"Within{int(band)}HU_pct"] = (
            100.0 * total["bands"][band] / count
            if count else float("nan"))
    return row


def _correction_at_threshold_hu(head, context, threshold, device, dtype):
    z_threshold = standardize_hu(torch.tensor(
        threshold, device=device, dtype=dtype)).reshape(1, 1)
    if isinstance(head, SpatialGatedCalibrationHead):
        raise ValueError(
            "spatial threshold correction requires a real image neighborhood")
    elif isinstance(head, ContextCalibrationHead):
        corr_z = head.correction(z_threshold, context=context)
    else:
        corr_z = head.correction(z_threshold)
    return float(corr_z.detach().reshape(-1)[0] * BENCHMARK_PIXEL_STD)


@torch.no_grad()
def audit_head_threshold_patient(pid, patient_dir, model, head, device,
                                 threshold=HEAD_DIAGNOSTIC_THRESHOLD_HU):
    """Paired trunk/head diagnosis for one fixed physical-HU threshold."""
    low = sort_by_instance_number(glob(str(patient_dir / "Low_Dose" / "*.dcm")))
    full = sort_by_instance_number(glob(str(patient_dir / "Full_Dose" / "*.dcm")))
    if len(low) != len(full):
        raise RuntimeError(f"[{pid}] slice mismatch: {len(low)} vs {len(full)}")
    if not low:
        raise RuntimeError(f"[{pid}] no paired DICOM slices found")

    body = "Chest" if pid.upper().startswith("C") else "Abdomen"
    names = ("FlipUp", "FlipDown", "NewDisagree", "ResolvedDisagree")
    totals = {name: {"count": 0, "distance_sum": 0.0,
                     "bands": {band: 0 for band in _DISTANCE_BANDS_HU}}
              for name in names}
    correction_130 = []
    correction_130_support = []
    slice_rows = []
    total_pixels = 0

    for slice_index, (low_path, full_path) in enumerate(zip(low, full)):
        low_hu = load_dicom_tensor(low_path).to(device)
        full_hu = load_dicom_tensor(full_path).to(device).clamp(cfg.A_MIN, cfg.A_MAX)
        x = standardize_hu(low_hu).unsqueeze(0).unsqueeze(0)
        trunk_z = model(x)
        head_z, context = _apply_head(head, x, trunk_z, body)
        trunk_hu = (denormalize_to_pixel(trunk_z.squeeze()) - cfg.HU_OFFSET).clamp(
            cfg.A_MIN, cfg.A_MAX)
        head_hu = (denormalize_to_pixel(head_z.squeeze()) - cfg.HU_OFFSET).clamp(
            cfg.A_MIN, cfg.A_MAX)

        stats_by_name, masks, distance = _paired_threshold_stats(
            full_hu, trunk_hu, head_hu, threshold)
        n_pixels = int(full_hu.numel())
        total_pixels += n_pixels

        if isinstance(head, SpatialGatedCalibrationHead):
            correction_map = head_hu - trunk_hu
            near_threshold = (trunk_hu - threshold).abs() <= 0.5
            if near_threshold.any():
                support = int(near_threshold.sum())
                corr_hu = float(correction_map[near_threshold].mean())
            else:
                support = 0
                corr_hu = float("nan")
        else:
            support = n_pixels
            corr_hu = _correction_at_threshold_hu(
                head, context, threshold, device, trunk_z.dtype)
        correction_130.append(corr_hu)
        correction_130_support.append(support)

        slice_row = {
            "PatientID": pid,
            "BodyType": body,
            "SliceIndex": slice_index,
            "LowDosePath": str(low_path),
            "Threshold_HU": threshold,
            "NumPixels": n_pixels,
            "CorrectionAt130_HU": corr_hu,
            "CorrectionSupportAt130_Count": support,
        }
        for name, mask in masks.items():
            stats = stats_by_name[name]
            _prefixed_stats(slice_row, name, stats)
            _accumulate_event_stats(totals[name], stats, mask, distance)
        slice_rows.append(slice_row)

    support_total = sum(correction_130_support)
    weighted_correction = (
        sum(value * support for value, support in zip(
            correction_130, correction_130_support) if support)
        / support_total if support_total else float("nan"))
    finite_corrections = [value for value, support in zip(
        correction_130, correction_130_support) if support]
    patient_row = {
        "PatientID": pid,
        "BodyType": body,
        "NumSlices": len(low),
        "Threshold_HU": threshold,
        "NumPixels": total_pixels,
        "CorrectionSupportAt130_Count": support_total,
        "MeanCorrectionAt130_HU": weighted_correction,
        "MinCorrectionAt130_HU": (min(finite_corrections)
                                   if finite_corrections else float("nan")),
        "MaxCorrectionAt130_HU": (max(finite_corrections)
                                   if finite_corrections else float("nan")),
    }
    for name, values in totals.items():
        _prefixed_stats(
            patient_row, name, _finalize_event_stats(values, total_pixels))
    return patient_row, slice_rows


@torch.no_grad()
def audit_patient(pid: str, patient_dir: Path, forward_fn, device,
                  thresholds_hu=THRESHOLDS_HU):
    low  = sort_by_instance_number(glob(str(patient_dir / "Low_Dose"  / "*.dcm")))
    full = sort_by_instance_number(glob(str(patient_dir / "Full_Dose" / "*.dcm")))
    if len(low) != len(full):
        raise RuntimeError(f"[{pid}] slice mismatch: {len(low)} vs {len(full)}")

    body = "Chest" if pid.upper().startswith("C") else "Abdomen"
    soft = soft_bin_params()

    acc  = {name: {"w": 0.0, "we": 0.0, "we2": 0.0, "wref": 0.0, "wpred": 0.0}
            for name, _, _ in soft}
    hard = {name: {"n": 0, "e": 0.0} for name, _, _ in TISSUE_BINS}
    thr  = {t: {"ref_pos": 0, "pred_pos": 0, "false_pos": 0,
                "false_neg": 0, "disagree": 0, "n": 0}
            for t in thresholds_hu}

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

        threshold_counts = _threshold_counts_grid(
            full_hu, pred_hu, thresholds_hu)
        for t in thresholds_hu:
            counts = threshold_counts[float(t)]
            for key, value in counts.items():
                thr[t][key] += value

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

    for t in thresholds_hu:
        d = thr[t]
        tag = _threshold_tag(t)
        for metric, value in _threshold_metrics(d).items():
            row[f"Thr{metric.replace('_pct', '')}_{tag}_pct"] = value

    return row, id_count, id_sum


def print_summary(all_dfs: dict):
    metrics = ["SoftBias_AirLung", "SoftBias_Soft", "SoftBias_Bone",
               "CalibSlope_alpha", "CalibIntercept_beta"]
    if all("ThrDisagree_130HU_pct" in df.columns for df in all_dfs.values()):
        metrics.append("ThrDisagree_130HU_pct")
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
                   help="Arms D/E: root dir with <arch>/best_head.pt "
                        "calibration heads. Adds a '<Model> + Head' target "
                         "next to each bare trunk.")
    p.add_argument("--threshold-grid", default=None, metavar="MIN,MAX,STEP",
                   help="Audit an inclusive dense HU threshold grid instead "
                        "of the legacy -950,0,100,130 thresholds.")
    args = p.parse_args()

    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Run with HU_RANGE_PRESET=benchmark.")

    thresholds_hu = THRESHOLDS_HU
    if args.threshold_grid is not None:
        try:
            threshold_min, threshold_max, threshold_step = (
                float(value) for value in args.threshold_grid.split(","))
        except ValueError as error:
            raise ValueError("--threshold-grid must be MIN,MAX,STEP") from error
        if threshold_min >= threshold_max or threshold_step <= 0.0:
            raise ValueError("--threshold-grid requires MIN < MAX and STEP > 0")
        intervals = (threshold_max - threshold_min) / threshold_step
        rounded_intervals = round(intervals)
        if not np.isclose(intervals, rounded_intervals, rtol=0.0, atol=1e-9):
            raise ValueError(
                "--threshold-grid STEP must divide MAX-MIN for an inclusive grid")
        thresholds_hu = tuple(np.linspace(
            threshold_min, threshold_max, rounded_intervals + 1))

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

    # Targets: (key, label, forward_fn, head_or_None). The head reference
    # is kept so oracle heads can receive the ground-truth body type per
    # patient.
    targets = []
    head_diagnostics = []
    if args.include_input:
        targets.append(("input", "LDCT input", input_forward, None))
    for arch in [a.strip() for a in args.archs.split(",") if a.strip()]:
        ckpt = Path(args.runs_root) / arch / "best_model.pt"
        if not ckpt.exists():
            print(f"  Skipping {arch}: {ckpt} not found")
            continue
        model = load_checkpoint(str(ckpt), arch, device)
        label = ARCH_MAP.get(arch, arch)
        targets.append((arch, label, make_model_forward(model), None))
        if args.heads_root:
            head_ckpt = Path(args.heads_root) / arch / "best_head.pt"
            if head_ckpt.exists():
                _head_checkpoint_meta(head_ckpt, arch, ckpt, args.split)
                head = load_head(str(head_ckpt), device)
                if getattr(head, "oracle", False):
                    print(f"  {arch}: ORACLE head detected -- ground-truth "
                          "body type will be fed per patient.")
                targets.append((f"{arch}_head", f"{label} + Head",
                                 make_model_forward(model, head), head))
                head_diagnostics.append((arch, model, head))
            else:
                print(f"  No head for {arch}: {head_ckpt} not found")

    if not targets:
        print("Nothing to audit. Train baselines first or pass --include-input.")
        return


    manifest = {
        "arguments": vars(args),
        "effective_thresholds_hu": [float(t) for t in thresholds_hu],
        "targets": [key for key, _, _, _ in targets],
        "trunk_checkpoints": {
            arch: str((Path(args.runs_root) / arch / "best_model.pt").resolve())
            for arch in [a.strip() for a in args.archs.split(",") if a.strip()]
            if (Path(args.runs_root) / arch / "best_model.pt").exists()
        },
        "head_checkpoints": {
            arch: str((Path(args.heads_root) / arch / "best_head.pt").resolve())
            for arch in [a.strip() for a in args.archs.split(",") if a.strip()]
            if args.heads_root
            and (Path(args.heads_root) / arch / "best_head.pt").exists()
        },
    }
    with open(out_path / "audit_manifest.json", "w", encoding="ascii") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    n_id = int((cfg.A_MAX - cfg.A_MIN) / IDENTITY_BIN_WIDTH)
    centers = cfg.A_MIN + (np.arange(n_id) + 0.5) * IDENTITY_BIN_WIDTH

    all_dfs: dict = {}
    for key, label, forward_fn, head in targets:
        print(f"\nAuditing {label} ...")
        rows = []
        id_count = np.zeros(n_id, dtype=np.float64)
        id_sum   = np.zeros(n_id, dtype=np.float64)
        for d in test_patients:
            if head is not None and getattr(head, "oracle", False):
                head.oracle_body = ("Chest" if d.name.upper().startswith("C")
                                    else "Abdomen")
            row, c, s = audit_patient(
                d.name, d, forward_fn, device, thresholds_hu=thresholds_hu)
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

    for arch, model, head in head_diagnostics:
        print(f"\nDiagnosing {ARCH_MAP.get(arch, arch)} + Head at "
              f"{HEAD_DIAGNOSTIC_THRESHOLD_HU:.0f} HU ...")
        patient_rows = []
        slice_rows = []
        for patient_dir in test_patients:
            patient_row, patient_slices = audit_head_threshold_patient(
                patient_dir.name, patient_dir, model, head, device)
            patient_rows.append(patient_row)
            slice_rows.extend(patient_slices)
        pd.DataFrame(patient_rows).to_csv(
            out_path / f"{arch}_head_threshold_130_patient.csv", index=False)
        pd.DataFrame(slice_rows).to_csv(
            out_path / f"{arch}_head_threshold_130_slice.csv", index=False)

    print_summary(all_dfs)

    summary_path = out_path / "hu_audit_comparison.csv"
    pd.concat(all_dfs.values(), ignore_index=True).to_csv(summary_path, index=False)
    print(f"\nFull report -> {summary_path}")


if __name__ == "__main__":
    main()
