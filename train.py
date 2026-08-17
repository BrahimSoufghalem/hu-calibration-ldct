"""Matched-budget trainer for the HU-calibration LDCT denoising study.

Trains RED-CNN or ResNet with the study's HU-fidelity losses (hu_losses.py):

  Arm A (baseline)  : pure MSE -- official ldct-benchmark protocol.
  Arm B (+L_HU)     : --hu-weight W       adds W * L_HU (MAE in HU units,
                      expressed in standardized units; the transparent
                      control arm -- see hu_losses.hu_mae_loss).
  Arm C (+L_HU-Cal) : --hucal-weight W    adds W * (L_SoftBias
                      + lambda_s*|alpha-1| + lambda_b*|beta|) with Gaussian
                      soft tissue bins.

Protocol identical to the previous study: hyperparameters follow the
official configs/redcnn.yaml of github.com/eeulig/ldct-benchmark and the
checkpoint criterion (--select-by bench_ssim) replicates the paper's
save_checkpoint(to_optimize=\"SSIM\"): best overall validation SSIM computed
WITHOUT clinical windowing. MATCHED training budget: 30k iterations for
every arm.

Mixed precision (--amp)
-----------------------
Optional and OFF by default (the benchmark protocol is pure fp32). If you
enable it, enable it for EVERY arm so internal comparisons stay matched.
Safety guarantees of this implementation:
  - only the network FORWARD runs under autocast; the model output is
    upcast to fp32 before any loss, so all loss reductions (including the
    soft-bin weighted sums and the alpha/beta regression of L_HU-Cal) are
    computed in full precision;
  - fp16 uses torch.amp.GradScaler (state saved in the checkpoint, so
    --resume is loss-scale correct); bf16 needs no scaler;
  - VALIDATION always runs in fp32 so the best-checkpoint selection
    numerics are identical with and without --amp.

Usage
-----
# Arm A -- baseline (pure MSE):
    HU_RANGE_PRESET=benchmark python train.py --arch redcnn \\
        --data-dir dataset --split 100p \\
        --max-iterations 30000 --iterations-before-val 1000 \\
        --batch-size 73 --patch-size 128 --val-patch-size 128 \\
        --lr 9.583417460320728e-05 --lr-schedule constant \\
        --select-by bench_ssim

# Arm B -- + L_HU control:
    ... --hu-weight 0.2

# Arm C -- + L_HU-Cal (soft-bin bias + calibration-line penalties):
    ... --hucal-weight 0.2 \\
        --hucal-slope-lambda 0.1 --hucal-intercept-lambda 0.01

# Arm C with per-bin weights (order AirLung,FatLow,Soft,Dense,Bone):
    ... --hucal-weight 0.2 --hucal-bin-weights 1,2,1,2,2

# Mixed-precision training (use for ALL arms if used at all):
    ... --amp                # fp16 + GradScaler (default dtype)
    ... --amp --amp-dtype bf16   # Ampere+ GPUs; no scaler needed

# Pilot mode (config screening only -- ranking, NOT reportable numbers):
    ... --train-patients 8 --val-patients 4 --max-iterations 8000

# Multi-seed runs for reporting mean +/- std:
    ... --seed 1 --output-root runs_seed1

# Resume after interruption:
    ... --resume
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as _sk_ssim
from tqdm import tqdm

import config as cfg
from benchmark_data import (
    BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD,
    denormalize_to_pixel, prepare_benchmark_data,
)
from hu_losses import HUCalLoss, hu_mae_loss, BIN_NAMES
from metrics import (
    compute_psnr_windowed, compute_ssim_windowed, compute_rmse_hu,
    compute_vif_hu,
)
from models import ARCH_CHOICES, build_benchmark_model
from twenty_patient_split import TRAIN_20P, VAL_20P
from utils import setup_reproducibility, get_device, get_state_dict

_SELECT_CHOICES = ("ssim", "psnr", "vif", "chest_ssim", "chest_vif",
                   "bench_ssim")
_AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(
        description="Matched-budget trainer for the HU-calibration study")
    p.add_argument("--arch", required=True, choices=list(ARCH_CHOICES))
    p.add_argument("--data-dir", default=cfg.DATA_DIR)
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--max-iterations",        type=int,   default=30_000)
    p.add_argument("--iterations-before-val", type=int,   default=1_000)
    p.add_argument("--batch-size",            type=int,   default=73)
    p.add_argument("--patch-size",            type=int,   default=128)
    p.add_argument("--val-patch-size",        type=int,   default=128)
    p.add_argument("--lr",                    type=float, default=9.583417460320728e-05)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"],
                   default="constant")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--num-workers",           type=int,   default=2)
    p.add_argument("--cache-rate",            type=float, default=1.0)
    p.add_argument("--output-root",           default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")

    # Mixed precision
    p.add_argument("--amp", action="store_true",
                   help="Mixed-precision TRAINING (autocast forward only; "
                        "all loss reductions stay fp32; validation stays "
                        "fp32). OFF by default because the ldct-benchmark "
                        "protocol is pure fp32. If enabled, enable it for "
                        "ALL arms so comparisons stay matched. CUDA only.")
    p.add_argument("--amp-dtype", choices=list(_AMP_DTYPES), default="fp16",
                   help="Autocast dtype for --amp. 'fp16' uses a GradScaler "
                        "(state is checkpointed, resume-safe); 'bf16' needs "
                        "no scaler but requires Ampere or newer GPUs.")

    # Pilot mode
    p.add_argument("--train-patients", type=int, default=None, metavar="N",
                   help="PILOT MODE: train on only N patients (deterministic "
                        "chest/abdomen-balanced subset). Config ranking "
                        "only; NOT reportable numbers.")
    p.add_argument("--val-patients", type=int, default=None, metavar="N",
                   help="PILOT MODE: validate on only N patients.")

    # Checkpoint selection
    p.add_argument("--select-by", choices=list(_SELECT_CHOICES),
                   default="bench_ssim",
                   help="Validation metric for best_model.pt. 'bench_ssim' "
                        "replicates the ldct-benchmark paper criterion "
                        "exactly (overall unwindowed SSIM, data_range=2924).")
    p.add_argument("--val-vif", action="store_true")

    # Study losses (arms B and C)
    p.add_argument("--hu-weight", type=float, default=0.0, metavar="W",
                   help="Arm B: weight of L_HU (MAE in HU units, expressed "
                        "in standardized units so W ~ 0.1-0.5 is a natural "
                        "range). 0 = off.")
    p.add_argument("--hucal-weight", type=float, default=0.0, metavar="W",
                   help="Arm C: weight of L_HU-Cal (soft-bin bias + "
                        "calibration-line penalties, standardized units). "
                        "0 = off.")
    p.add_argument("--hucal-slope-lambda", type=float, default=0.1,
                   metavar="L", help="lambda_s for |alpha - 1| inside "
                                     "L_HU-Cal.")
    p.add_argument("--hucal-intercept-lambda", type=float, default=0.01,
                   metavar="L", help="lambda_b for |beta| inside L_HU-Cal.")
    p.add_argument("--hucal-bin-weights", type=str, default=None,
                   metavar="W1,W2,W3,W4,W5",
                   help="Per-bin weights for L_SoftBias, order "
                        f"{','.join(BIN_NAMES)}. Default uniform.")
    return p.parse_args()


def compute_loss(pred, target, hu_weight=0.0, hucal=None, hucal_weight=0.0):
    loss = F.mse_loss(pred, target)
    if hu_weight > 0.0:
        loss = loss + float(hu_weight) * hu_mae_loss(pred, target)
    if hucal is not None and hucal_weight > 0.0:
        loss = loss + float(hucal_weight) * hucal(pred, target)
    return loss


def apply_split(split):
    if split == "20p":
        cfg.EXPECTED_TRAIN = TRAIN_20P
        cfg.EXPECTED_VAL   = VAL_20P
        return len(TRAIN_20P), len(VAL_20P)
    return len(cfg.EXPECTED_TRAIN), len(cfg.EXPECTED_VAL)


@torch.no_grad()
def validate(model, loader, device, with_vif=False):
    """Overall AND per-region (Chest/Abdomen) validation metrics.

    Always runs in fp32 (even with --amp) so best-checkpoint selection is
    numerically identical across precision settings. 'bench_ssim' is the
    exact ldct-benchmark paper metric: overall SSIM WITHOUT clinical
    windowing (skimage, data_range=2924, denormalized clipped images).
    """
    model.eval()
    sums    = dict(mse=0.0, psnr=0.0, ssim=0.0, rmse=0.0,
                   baseline_psnr=0.0, vif=0.0, bench_ssim=0.0)
    region  = {
        "Chest":   dict(psnr=0.0, ssim=0.0, vif=0.0, n=0),
        "Abdomen": dict(psnr=0.0, ssim=0.0, vif=0.0, n=0),
    }
    batches = samples = 0
    for batch in tqdm(loader, desc="  Val", leave=False, dynamic_ncols=True):
        x    = batch["image"].to(device, non_blocking=True)
        y    = batch["label"].to(device, non_blocking=True)
        pred = model(x)
        sums["mse"] += float(F.mse_loss(pred, y))
        batches += 1
        pred_px = denormalize_to_pixel(pred).clamp(0.0, cfg.EVAL_DATA_RANGE)
        y_px    = denormalize_to_pixel(y).clamp(0.0, cfg.EVAL_DATA_RANGE)
        x_px    = denormalize_to_pixel(x).clamp(0.0, cfg.EVAL_DATA_RANGE)
        body = batch.get("body_type", ["Abdomen"] * pred.shape[0])
        for i in range(pred.shape[0]):
            bt = "Chest" if str(body[i]).lower().startswith("c") else "Abdomen"
            ps = compute_psnr_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            ss = compute_ssim_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            vf = compute_vif_hu(pred_px[i].squeeze(), y_px[i].squeeze()) if with_vif else 0.0
            t_np = y_px[i].squeeze().detach().cpu().numpy()
            p_np = pred_px[i].squeeze().detach().cpu().numpy()
            bss  = float(_sk_ssim(t_np, p_np,
                                  data_range=float(cfg.EVAL_DATA_RANGE)))
            sums["psnr"]          += ps
            sums["ssim"]          += ss
            sums["vif"]           += vf
            sums["bench_ssim"]    += bss
            sums["baseline_psnr"] += compute_psnr_windowed(x_px[i].squeeze(), y_px[i].squeeze(), bt)
            sums["rmse"]          += compute_rmse_hu(pred_px[i].squeeze(), y_px[i].squeeze())
            r = region[bt]
            r["psnr"] += ps
            r["ssim"] += ss
            r["vif"]  += vf
            r["n"]    += 1
            samples += 1
    n_b, n_s = max(1, batches), max(1, samples)
    out = {
        "mse":        sums["mse"]  / n_b,
        "psnr":       sums["psnr"] / n_s,
        "dpsnr":      (sums["psnr"] - sums["baseline_psnr"]) / n_s,
        "ssim":       sums["ssim"] / n_s,
        "rmse":       sums["rmse"] / n_s,
        "vif":        sums["vif"]  / n_s,
        "bench_ssim": sums["bench_ssim"] / n_s,
    }
    for name, r in region.items():
        key = name.lower()
        n   = max(1, r["n"])
        out[f"{key}_psnr"] = r["psnr"] / n
        out[f"{key}_ssim"] = r["ssim"] / n
        out[f"{key}_vif"]  = r["vif"]  / n
        out[f"{key}_n"]    = r["n"]
    return out


def selection_score(val, select_by):
    if select_by == "ssim":
        return val["ssim"]
    if select_by == "psnr":
        return val["psnr"]
    if select_by == "vif":
        return val["vif"]
    if select_by == "bench_ssim":
        return val["bench_ssim"]
    if select_by == "chest_ssim":
        return val["chest_ssim"] if val["chest_n"] > 0 else val["ssim"]
    if select_by == "chest_vif":
        return val["chest_vif"] if val["chest_n"] > 0 else val["vif"]
    raise ValueError(f"Unknown --select-by: {select_by}")


def lr_at(iteration, base_lr, min_lr, max_iter, schedule):
    """LR as a pure function of the iteration counter (resume-safe)."""
    if schedule != "cosine":
        return base_lr
    t = min(1.0, max(0.0, iteration / max(1, max_iter)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def train_cycle(model, loader, optimizer, device, iteration, max_iter,
                hu_weight=0.0, hucal=None, hucal_weight=0.0,
                base_lr=1e-4, min_lr=1e-6, lr_schedule="constant",
                use_amp=False, amp_dtype=torch.float16, scaler=None):
    model.train()
    total = count = 0.0
    bar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in bar:
        if iteration >= max_iter:
            break
        lr_now = lr_at(iteration, base_lr, min_lr, max_iter, lr_schedule)
        for g in optimizer.param_groups:
            g["lr"] = lr_now
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # Autocast covers the network forward only. The output is upcast to
        # fp32 before the loss so every reduction (MSE, L_HU, soft-bin sums
        # and the alpha/beta regression of L_HU-Cal) runs in full precision.
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            pred = model(x)
        loss = compute_loss(pred.float(), y.float(), hu_weight=hu_weight,
                            hucal=hucal, hucal_weight=hucal_weight)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        iteration += 1
        total += float(loss.detach())
        count += 1
        bar.set_postfix(iter=iteration, loss=f"{loss.item():.6f}",
                        lr=f"{lr_now:.2e}")
    return iteration, total / max(1, count)


def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Set HU_RANGE_PRESET=benchmark.")
    if args.hu_weight < 0.0:
        raise ValueError("--hu-weight must be >= 0")
    if args.hucal_weight < 0.0:
        raise ValueError("--hucal-weight must be >= 0")
    if args.hu_weight > 0.0 and args.hucal_weight > 0.0:
        raise ValueError(
            "Arms B and C are separate: use either --hu-weight or "
            "--hucal-weight, not both (the study design has no B+C arm).")
    hucal_bin_weights = None
    if args.hucal_bin_weights is not None:
        if args.hucal_weight <= 0.0:
            raise ValueError("--hucal-bin-weights requires --hucal-weight > 0")
        try:
            hucal_bin_weights = [float(v)
                                 for v in args.hucal_bin_weights.split(",")]
        except ValueError:
            raise ValueError(
                "--hucal-bin-weights must be comma-separated numbers")
    if args.train_patients is not None and args.train_patients < 1:
        raise ValueError("--train-patients must be >= 1")
    if args.val_patients is not None and args.val_patients < 1:
        raise ValueError("--val-patients must be >= 1")

    n_train, n_val = apply_split(args.split)
    if args.train_patients is not None:
        n_train = min(n_train, args.train_patients)
    if args.val_patients is not None:
        n_val = min(n_val, args.val_patients)
    pilot = args.train_patients is not None or args.val_patients is not None

    cfg.SEED = int(args.seed)
    setup_reproducibility(args.seed)
    device    = get_device()
    out_dir   = os.path.join(args.output_root, args.arch)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    use_amp = bool(args.amp)
    amp_dtype = _AMP_DTYPES[args.amp_dtype]
    if use_amp and device.type != "cuda":
        print("  WARNING: --amp requested but no CUDA device found; "
              "training in fp32.")
        use_amp = False
    if use_amp and amp_dtype is torch.bfloat16 \
            and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "--amp-dtype bf16 requested but this GPU does not support "
            "bfloat16; use --amp-dtype fp16.")
    # GradScaler is only needed for fp16 (bf16 has fp32-like range).
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype is torch.float16)

    with_vif = bool(args.val_vif or args.select_by in ("vif", "chest_vif"))

    hucal = None
    if args.hucal_weight > 0.0:
        hucal = HUCalLoss(
            slope_lambda=args.hucal_slope_lambda,
            intercept_lambda=args.hucal_intercept_lambda,
            bin_weights=hucal_bin_weights,
        )

    if args.hu_weight > 0.0:
        arm = "B (+L_HU)"
        loss_desc = f"MSE + {args.hu_weight}*L_HU"
    elif args.hucal_weight > 0.0:
        arm = "C (+L_HU-Cal)"
        loss_desc = (f"MSE + {args.hucal_weight}*(L_SoftBias"
                     f" + {args.hucal_slope_lambda}*|a-1|"
                     f" + {args.hucal_intercept_lambda}*|b|)")
        if hucal_bin_weights is not None:
            loss_desc += ("  [bin weights "
                          + ",".join(str(w) for w in hucal_bin_weights) + "]")
    else:
        arm = "A (baseline)"
        loss_desc = "1.00*MSE"

    print(f"\n{'='*68}")
    print(f"  arch={args.arch.upper()} | split={args.split} | seed={args.seed}")
    print(f"  Study arm      : {arm}")
    print(f"  Train patients : {n_train}  |  Val patients: {n_val}")
    print(f"  Data dir       : {args.data_dir}")
    print(f"  Output         : {out_dir}")
    print(f"  Loss           : {loss_desc}")
    print(f"  Precision      : "
          + (f"AMP {args.amp_dtype} (forward only; fp32 losses/val"
             + (", GradScaler" if scaler.is_enabled() else "")
             + ")" if use_amp else "fp32 (benchmark protocol)"))
    print(f"  LR schedule    : {args.lr_schedule} (lr={args.lr:.2e}"
          + (f" -> {args.min_lr:.2e}" if args.lr_schedule == "cosine" else "")
          + ")")
    print(f"  Select best by : {args.select_by}"
          + (" [ldct-benchmark paper criterion]"
             if args.select_by == "bench_ssim" else "")
          + (" (+VIF in val)" if with_vif else ""))
    if pilot:
        print("  *** PILOT MODE : reduced patient subset. Config "
              "ranking ONLY -- retrain the winner on the full split "
              "before reporting. ***")
    print(f"{'='*68}\n")

    model     = build_benchmark_model(args.arch, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999))

    iteration  = 0
    best_score = -float("inf")
    if args.resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler.is_enabled() and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        iteration  = int(ckpt.get("iteration", 0))
        old_select = ckpt.get("select_by", "ssim")
        if old_select == args.select_by:
            best_score = float(ckpt.get("score", -float("inf")))
        else:
            print(f"  NOTE: checkpoint used --select-by {old_select}; "
                  f"resetting best score for {args.select_by}.")
        print(f"  Resumed at iter {iteration} | "
              f"best {args.select_by} {best_score:.5f}")
    elif args.resume:
        print(f"  --resume: no checkpoint at {ckpt_path}, starting fresh.")

    if iteration >= args.max_iterations:
        print(f"  Training already complete "
              f"({iteration}/{args.max_iterations}).")
        return

    train_loader, val_loader = prepare_benchmark_data(
        in_dir=args.data_dir,
        train_patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        iterations_before_val=args.iterations_before_val,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        max_train_patients=args.train_patients,
        max_val_patients=args.val_patients,
    )

    print(f"Loss      : {loss_desc}")
    print(f"Optimizer : Adam(lr={args.lr:.2e}, schedule={args.lr_schedule})")

    start = time.time()
    cycle = iteration // args.iterations_before_val

    while iteration < args.max_iterations:
        cycle += 1
        t0 = time.time()
        iteration, train_loss = train_cycle(
            model, train_loader, optimizer, device,
            iteration, args.max_iterations,
            hu_weight=args.hu_weight,
            hucal=hucal,
            hucal_weight=args.hucal_weight,
            base_lr=args.lr,
            min_lr=args.min_lr,
            lr_schedule=args.lr_schedule,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val   = validate(model, val_loader, device, with_vif=with_vif)
        score = selection_score(val, args.select_by)

        meta = {
            "architecture":    args.arch,
            "split":           args.split,
            "seed":            int(args.seed),
            "study_arm":       arm,
            "hu_weight":       args.hu_weight,
            "hucal_weight":    args.hucal_weight,
            "hucal_slope_lambda":     args.hucal_slope_lambda,
            "hucal_intercept_lambda": args.hucal_intercept_lambda,
            "hucal_bin_weights":      hucal_bin_weights,
            "amp":             use_amp,
            "amp_dtype":       args.amp_dtype if use_amp else None,
            "lr_schedule":     args.lr_schedule,
            "min_lr":          args.min_lr,
            "select_by":       args.select_by,
            "normalization":   "benchmark_meanstd",
            "pixel_mean":      BENCHMARK_PIXEL_MEAN,
            "pixel_std":       BENCHMARK_PIXEL_STD,
            "pixel_domain":    "HU+1024",
            "hu_preset":       cfg.HU_RANGE_PRESET,
            "eval_data_range": cfg.EVAL_DATA_RANGE,
            "loss":            loss_desc,
            "input_mode":      "2d",
            "n_train_patients": n_train,
            "n_val_patients":   n_val,
            "max_train_patients": args.train_patients,
            "max_val_patients":   args.val_patients,
            "pilot_mode":      pilot,
        }
        payload = {
            "model_state_dict": get_state_dict(model),
            "meta":      meta,
            "iteration": iteration,
            "ssim":      val["ssim"],
            "psnr":      val["psnr"],
            "val_mse":   val["mse"],
            "score":     score,
            "select_by": args.select_by,
            "val_detail": {k: v for k, v in val.items()},
        }
        torch.save(payload, os.path.join(out_dir, "last_model.pt"))
        if score > best_score:
            best_score = score
            torch.save(payload, os.path.join(out_dir, "best_model.pt"))
        torch.save({**payload,
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": (scaler.state_dict()
                                     if scaler.is_enabled() else None)},
                   ckpt_path)

        elapsed = time.time() - t0
        region_str = (
            f"C-PSNR {val['chest_psnr']:.2f} C-SSIM {val['chest_ssim']:.4f} | "
            f"A-PSNR {val['abdomen_psnr']:.2f} A-SSIM {val['abdomen_ssim']:.4f}"
            if val["chest_n"] > 0 and val["abdomen_n"] > 0 else ""
        )
        vif_str = (
            f" | VIF {val['vif']:.4f}"
            + (f" (C {val['chest_vif']:.4f})" if val["chest_n"] > 0 else "")
            if with_vif else ""
        )
        print(
            f"Cycle {cycle:02d} | Iter {iteration:06d}/{args.max_iterations} | "
            f"Loss {train_loss:.6f} | Val MSE {val['mse']:.6f} | "
            f"PSNR {val['psnr']:.3f} | dPSNR {val['dpsnr']:+.3f} | "
            f"SSIM {val['ssim']:.5f} | bSSIM {val['bench_ssim']:.5f} | "
            f"RMSE {val['rmse']:.2f}"
            f"{vif_str} | {args.select_by} {score:.5f}"
            f"{(' | ' + region_str) if region_str else ''} | "
            f"{elapsed:.1f}s"
        )

    total = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"\nDone [{args.arch.upper()}] in {total} | "
          f"best {args.select_by}={best_score:.5f}")
    print(f"Checkpoint -> {os.path.join(out_dir, 'best_model.pt')}")

    if pilot:
        print("\nPILOT MODE reminder: these numbers are for config "
              "ranking only. Retrain the winning config on the full "
              "split with the study budget before reporting.")


if __name__ == "__main__":
    main()
