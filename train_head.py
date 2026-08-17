"""Arm D -- post-hoc calibration head trained on a FROZEN denoising trunk.

The trunk (arm A baseline, or any trained checkpoint) is loaded and frozen;
only the ~1k-parameter CalibrationHead is optimized. Objective:

    L = w_mse * MSE(T(f(x)), y) + w_cal * L_HU-Cal(T(f(x)), y)
        [+ lambda_w * |T(z_water) - z_water|^2]

Because the head is bounded (<= delta_hu), certified monotonic, and
identity-initialized, it cannot destroy the trunk's output -- it can only
reshape the intensity mapping within a +/- delta_hu envelope.

Budget: deliberately tiny (default 3k iterations); the head is a 1D curve.
This is post-hoc calibration, not denoiser training, so the matched-budget
rule of arms A/B/C does not apply -- but the budget is documented in meta.

Usage
-----
# Arm D on the arm-A RED-CNN baseline:
    HU_RANGE_PRESET=benchmark python train_head.py --arch redcnn \\
        --data-dir dataset --split 100p \\
        --trunk-checkpoint runs/redcnn/best_model.pt \\
        --output-root runs_armD

# With the water-anchor constraint (tested, not assumed):
    ... --water-anchor-lambda 0.1

# Audit trunk vs trunk+head afterwards:
    HU_RANGE_PRESET=benchmark python hu_audit.py --test-dir test \\
        --runs-root runs --heads-root runs_armD --archs redcnn \\
        --include-input --output hu_audit_d
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as cfg
from benchmark_data import (
    BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD, prepare_benchmark_data,
)
from calibration_head import CalibrationHead, save_head
from evaluate_image import load_checkpoint
from hu_losses import HUCalLoss, BIN_NAMES
from models import ARCH_CHOICES
from train import apply_split, validate, selection_score
from utils import setup_reproducibility, get_device

_SELECT_CHOICES = ("val_loss", "bench_ssim", "ssim", "psnr")


class TrunkWithHead(torch.nn.Module):
    """Frozen trunk -> trainable head. Trunk runs under no_grad."""

    def __init__(self, trunk, head):
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x):
        with torch.no_grad():
            z = self.trunk(x)
        return self.head(z)


def parse_args():
    p = argparse.ArgumentParser(
        description="Arm D: post-hoc calibration head on a frozen trunk")
    p.add_argument("--arch", required=True, choices=list(ARCH_CHOICES))
    p.add_argument("--trunk-checkpoint", default=None,
                   help="Path to the frozen trunk checkpoint "
                        "(default: runs/<arch>/best_model.pt).")
    p.add_argument("--data-dir", default=cfg.DATA_DIR)
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--max-iterations",        type=int,   default=3_000)
    p.add_argument("--iterations-before-val", type=int,   default=500)
    p.add_argument("--batch-size",            type=int,   default=73)
    p.add_argument("--patch-size",            type=int,   default=128)
    p.add_argument("--val-patch-size",        type=int,   default=128)
    p.add_argument("--lr",                    type=float, default=1e-3)
    p.add_argument("--num-workers",           type=int,   default=2)
    p.add_argument("--cache-rate",            type=float, default=1.0)
    p.add_argument("--output-root",           default="runs_armD")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-patients", type=int, default=None)
    p.add_argument("--val-patients",   type=int, default=None)

    # Head architecture / constraints
    p.add_argument("--hidden",   type=int,   default=32)
    p.add_argument("--delta-hu", type=float, default=80.0,
                   help="Hard bound on the correction magnitude in HU.")
    p.add_argument("--kappa",    type=float, default=0.9,
                   help="Monotonicity margin: T' >= 1 - kappa > 0.")

    # Objective
    p.add_argument("--mse-weight",   type=float, default=1.0)
    p.add_argument("--hucal-weight", type=float, default=1.0)
    p.add_argument("--hucal-slope-lambda",     type=float, default=0.1)
    p.add_argument("--hucal-intercept-lambda", type=float, default=0.01)
    p.add_argument("--hucal-bin-weights", type=str, default=None,
                   metavar="W1,W2,W3,W4,W5",
                   help=f"Per-bin weights, order {','.join(BIN_NAMES)}.")
    p.add_argument("--water-anchor-lambda", type=float, default=0.0,
                   help="lambda_w for |T(0 HU)|^2 (0 = anchor off).")

    p.add_argument("--select-by", choices=list(_SELECT_CHOICES),
                   default="val_loss",
                   help="'val_loss' selects the head that best optimizes "
                        "the calibration objective on validation; global "
                        "image metrics are reported alongside either way.")
    return p.parse_args()


def compute_objective(pred, target, head, args, hucal):
    loss = args.mse_weight * F.mse_loss(pred, target)
    if args.hucal_weight > 0.0:
        loss = loss + args.hucal_weight * hucal(pred, target)
    if args.water_anchor_lambda > 0.0:
        loss = loss + args.water_anchor_lambda * head.water_anchor_penalty()
    return loss


@torch.no_grad()
def validation_objective(wrapped, loader, device, head, args, hucal):
    wrapped.eval()
    total = count = 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        total += float(compute_objective(wrapped(x), y, head, args, hucal))
        count += 1
    return total / max(1, count)


def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Set HU_RANGE_PRESET=benchmark.")

    trunk_ckpt = args.trunk_checkpoint or os.path.join(
        "runs", args.arch, "best_model.pt")
    if not os.path.exists(trunk_ckpt):
        raise FileNotFoundError(f"Trunk checkpoint not found: {trunk_ckpt}")

    hucal_bin_weights = None
    if args.hucal_bin_weights is not None:
        hucal_bin_weights = [float(v)
                             for v in args.hucal_bin_weights.split(",")]

    n_train, n_val = apply_split(args.split)
    if args.train_patients is not None:
        n_train = min(n_train, args.train_patients)
    if args.val_patients is not None:
        n_val = min(n_val, args.val_patients)

    cfg.SEED = int(args.seed)
    setup_reproducibility(args.seed)
    device  = get_device()
    out_dir = os.path.join(args.output_root, args.arch)
    os.makedirs(out_dir, exist_ok=True)

    trunk = load_checkpoint(trunk_ckpt, args.arch, device)
    trunk.eval()
    for prm in trunk.parameters():
        prm.requires_grad_(False)

    head = CalibrationHead(hidden=args.hidden, delta_hu=args.delta_hu,
                           kappa=args.kappa).to(device)
    wrapped = TrunkWithHead(trunk, head)

    hucal = HUCalLoss(
        slope_lambda=args.hucal_slope_lambda,
        intercept_lambda=args.hucal_intercept_lambda,
        bin_weights=hucal_bin_weights,
    )

    n_params = sum(prm.numel() for prm in head.parameters())
    loss_desc = (f"{args.mse_weight}*MSE + {args.hucal_weight}*(L_SoftBias"
                 f" + {args.hucal_slope_lambda}*|a-1|"
                 f" + {args.hucal_intercept_lambda}*|b|)")
    if args.water_anchor_lambda > 0.0:
        loss_desc += f" + {args.water_anchor_lambda}*WaterAnchor"

    print(f"\n{'='*68}")
    print(f"  ARM D — post-hoc calibration head (frozen trunk)")
    print(f"  arch={args.arch.upper()} | split={args.split} | seed={args.seed}")
    print(f"  Trunk          : {trunk_ckpt} (FROZEN)")
    print(f"  Head           : hidden={args.hidden}, |corr|<={args.delta_hu} HU, "
          f"T'>={1.0 - args.kappa:.2f} ({n_params} params)")
    print(f"  Objective      : {loss_desc}")
    print(f"  Budget         : {args.max_iterations} iterations")
    print(f"  Select best by : {args.select_by}")
    print(f"{'='*68}\n")

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

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999))

    iteration = 0
    best_score = -float("inf")
    start = time.time()
    cycle = 0

    while iteration < args.max_iterations:
        cycle += 1
        t0 = time.time()
        head.train()
        total = count = 0.0
        bar = tqdm(train_loader, desc="  Train", leave=False,
                   dynamic_ncols=True)
        for batch in bar:
            if iteration >= args.max_iterations:
                break
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_objective(wrapped(x), y, head, args, hucal)
            loss.backward()
            optimizer.step()
            iteration += 1
            total += float(loss.detach())
            count += 1
            bar.set_postfix(iter=iteration, loss=f"{loss.item():.6f}")
        train_loss = total / max(1, count)

        val = validate(wrapped, val_loader, device)
        val_obj = validation_objective(wrapped, val_loader, device, head,
                                       args, hucal)
        score = (-val_obj if args.select_by == "val_loss"
                 else selection_score(val, args.select_by))

        meta = {
            "architecture":    args.arch,
            "study_arm":       "D (post-hoc head, frozen trunk)",
            "trunk_checkpoint": trunk_ckpt,
            "split":           args.split,
            "seed":            int(args.seed),
            "head_hidden":     args.hidden,
            "delta_hu":        args.delta_hu,
            "kappa":           args.kappa,
            "mse_weight":      args.mse_weight,
            "hucal_weight":    args.hucal_weight,
            "hucal_slope_lambda":     args.hucal_slope_lambda,
            "hucal_intercept_lambda": args.hucal_intercept_lambda,
            "hucal_bin_weights":      hucal_bin_weights,
            "water_anchor_lambda":    args.water_anchor_lambda,
            "budget_iterations": args.max_iterations,
            "select_by":       args.select_by,
            "normalization":   "benchmark_meanstd",
            "pixel_mean":      BENCHMARK_PIXEL_MEAN,
            "pixel_std":       BENCHMARK_PIXEL_STD,
            "pixel_domain":    "HU+1024",
            "hu_preset":       cfg.HU_RANGE_PRESET,
            "eval_data_range": cfg.EVAL_DATA_RANGE,
            "loss":            loss_desc,
        }
        extra = {"meta": meta, "iteration": iteration, "score": score,
                 "select_by": args.select_by, "val_objective": val_obj,
                 "val_detail": {k: v for k, v in val.items()}}
        save_head(head, os.path.join(out_dir, "last_head.pt"), extra)
        if score > best_score:
            best_score = score
            save_head(head, os.path.join(out_dir, "best_head.pt"), extra)

        print(
            f"Cycle {cycle:02d} | Iter {iteration:05d}/{args.max_iterations} | "
            f"Loss {train_loss:.6f} | ValObj {val_obj:.6f} | "
            f"PSNR {val['psnr']:.3f} | SSIM {val['ssim']:.5f} | "
            f"bSSIM {val['bench_ssim']:.5f} | RMSE {val['rmse']:.2f} | "
            f"{args.select_by} {score:.6f} | {time.time() - t0:.1f}s"
        )

    # Report the learned transfer curve at tissue-bin centers.
    head.eval()
    hu_in, hu_out = head.transfer_curve()
    corr = (hu_out - hu_in).detach().cpu().numpy()
    hu_np = hu_in.detach().cpu().numpy()
    print("\nLearned transfer-curve correction T(HU) - HU:")
    for name, center in (("AirLung", -762), ("FatLow", -350), ("Soft", 0),
                         ("Dense", 400), ("Bone", 1250)):
        i = int(abs(hu_np - center).argmin())
        print(f"  {name:<8} ({center:>5} HU): {corr[i]:+7.2f} HU")
    i0 = int(abs(hu_np - 0.0).argmin())
    print(f"  Water anchor |T(0)|: {abs(corr[i0]):.2f} HU")
    print(f"  Max |correction|   : {abs(corr).max():.2f} HU "
          f"(bound: {args.delta_hu} HU)")

    total_t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"\nDone [ARM D / {args.arch.upper()}] in {total_t} | "
          f"best {args.select_by}={best_score:.6f}")
    print(f"Head checkpoint -> {os.path.join(out_dir, 'best_head.pt')}")


if __name__ == "__main__":
    main()
