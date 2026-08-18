"""Arms D & E -- constrained calibration heads on top of a denoising trunk.

Arm D : intensity-only head (1D transfer curve), post-hoc on a FROZEN trunk.
Arm E : context-conditioned head -- same constrained curve, but g(z, c) is
        conditioned on a DETACHED per-image context vector (mean, std, and
        the 5 soft tissue-bin occupancy fractions). Motivated directly by
        the arm C/D finding: HU bias is anatomy-dependent, so one global
        loss/curve cannot fix chest and abdomen simultaneously. Includes
        the anti-collapse centering penalty (per-image mean correction -> 0)
        so the head reshapes intensities, never a per-image DC shift.

Modes
-----
- post-hoc (default) : trunk FROZEN, head trained alone (tiny budget,
                       directly comparable D vs E-posthoc).
- --joint            : trunk trained FROM SCRATCH together with the head
                       (use the full matched budget: --max-iterations 30000
                       --iterations-before-val 1000 --select-by bench_ssim).
                       With --hucal-weight > 0 this is arm F (losses+head).
                       Saves best_model.pt (trunk) alongside best_head.pt so
                       hu_audit.py can audit the pair with
                       --runs-root <out> --heads-root <out>.

Objective
---------
    L = w_mse*MSE + w_cal*L_HU-Cal + lambda_c*Center(corr)
        [+ lambda_w*WaterAnchor]

Usage
-----
# Arm D (intensity head, frozen arm-A trunk):
    HU_RANGE_PRESET=benchmark python train_head.py --arch redcnn \\
        --data-dir dataset --split 100p --output-root runs_armD

# Arm E post-hoc (context head, frozen arm-A trunk; direct D comparison):
    HU_RANGE_PRESET=benchmark python train_head.py --arch redcnn \\
        --data-dir dataset --split 100p --head-type context \\
        --output-root runs_armE

# Arm E joint (context head + trunk from scratch, matched budget):
    HU_RANGE_PRESET=benchmark python train_head.py --arch redcnn \\
        --data-dir dataset --split 100p --head-type context --joint \\
        --max-iterations 30000 --iterations-before-val 1000 \\
        --select-by bench_ssim --output-root runs_armE_joint

# Audit post-hoc heads (trunk rows from runs/, head rows from runs_armE/):
    HU_RANGE_PRESET=benchmark python hu_audit.py --test-dir test \\
        --runs-root runs --heads-root runs_armE --archs redcnn \\
        --include-input --output hu_audit_e
# Audit a joint run: --runs-root runs_armE_joint --heads-root runs_armE_joint
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
from calibration_head import (
    CalibrationHead, ContextCalibrationHead, save_head,
)
from evaluate_image import load_checkpoint
from hu_losses import HUCalLoss, BIN_NAMES
from models import ARCH_CHOICES, build_benchmark_model
from train import apply_split, validate, selection_score
from utils import setup_reproducibility, get_device, get_state_dict

_SELECT_CHOICES = ("val_loss", "bench_ssim", "ssim", "psnr")


class TrunkWithHead(torch.nn.Module):
    """Trunk (no_grad) -> head. Used for metric validation only."""

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
        description="Arms D/E: constrained calibration heads")
    p.add_argument("--arch", required=True, choices=list(ARCH_CHOICES))
    p.add_argument("--head-type", choices=["intensity", "context"],
                   default="intensity",
                   help="'intensity' = arm D (1D curve); "
                        "'context' = arm E (per-image conditioned curve).")
    p.add_argument("--joint", action="store_true",
                   help="Train the trunk FROM SCRATCH jointly with the head "
                        "(full budget). Default: post-hoc on a frozen trunk.")
    p.add_argument("--trunk-checkpoint", default=None,
                   help="Frozen trunk checkpoint for post-hoc mode "
                        "(default: runs/<arch>/best_model.pt).")
    p.add_argument("--data-dir", default=cfg.DATA_DIR)
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--max-iterations",        type=int,   default=3_000)
    p.add_argument("--iterations-before-val", type=int,   default=500)
    p.add_argument("--batch-size",            type=int,   default=73)
    p.add_argument("--patch-size",            type=int,   default=128)
    p.add_argument("--val-patch-size",        type=int,   default=128)
    p.add_argument("--head-lr", type=float, default=1e-3,
                   help="Adam LR for the head parameters.")
    p.add_argument("--lr", type=float, default=9.583417460320728e-05,
                   help="Adam LR for the trunk (joint mode only).")
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
    p.add_argument("--center-lambda", type=float, default=0.1,
                   help="Anti-collapse centering penalty (context head "
                        "only): per-image mean correction -> 0.")
    p.add_argument("--water-anchor-lambda", type=float, default=0.0,
                   help="lambda_w for |T(0 HU)|^2 (0 = anchor off).")

    p.add_argument("--select-by", choices=list(_SELECT_CHOICES),
                   default="val_loss",
                   help="'val_loss' selects by the calibration objective on "
                        "validation; global metrics are reported alongside. "
                        "For --joint runs, bench_ssim is recommended to "
                        "match the A/B/C selection protocol.")
    return p.parse_args()


def build_head(args, device):
    cls = (ContextCalibrationHead if args.head_type == "context"
           else CalibrationHead)
    return cls(hidden=args.hidden, delta_hu=args.delta_hu,
               kappa=args.kappa).to(device)


def objective(z, corr, target, head, args, hucal):
    pred = z + corr
    loss = args.mse_weight * F.mse_loss(pred, target)
    if args.hucal_weight > 0.0:
        loss = loss + args.hucal_weight * hucal(pred, target)
    if args.center_lambda > 0.0 and isinstance(head, ContextCalibrationHead):
        loss = loss + args.center_lambda \
            * ContextCalibrationHead.centering_penalty(corr)
    if args.water_anchor_lambda > 0.0:
        if isinstance(head, ContextCalibrationHead):
            anchor = head.water_anchor_penalty(z)
        else:
            anchor = head.water_anchor_penalty()
        loss = loss + args.water_anchor_lambda * anchor
    return loss


@torch.no_grad()
def validation_objective(trunk, head, loader, device, args, hucal):
    trunk.eval()
    head.eval()
    total = count = 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        z = trunk(x)
        corr = head.correction(z)
        total += float(objective(z, corr, y, head, args, hucal))
        count += 1
    return total / max(1, count)


def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Set HU_RANGE_PRESET=benchmark.")
    if args.joint and args.max_iterations <= 5_000:
        print("  WARNING: --joint with a tiny budget "
              f"({args.max_iterations} iters). For reportable joint runs "
              "use the matched budget (30000).")

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

    # Trunk: frozen checkpoint (post-hoc) or fresh (joint).
    if args.joint:
        trunk = build_benchmark_model(args.arch, device)
        trunk_ckpt = None
    else:
        trunk_ckpt = args.trunk_checkpoint or os.path.join(
            "runs", args.arch, "best_model.pt")
        if not os.path.exists(trunk_ckpt):
            raise FileNotFoundError(
                f"Trunk checkpoint not found: {trunk_ckpt}")
        trunk = load_checkpoint(trunk_ckpt, args.arch, device)
        trunk.eval()
        for prm in trunk.parameters():
            prm.requires_grad_(False)

    head = build_head(args, device)
    wrapped = TrunkWithHead(trunk, head)

    hucal = HUCalLoss(
        slope_lambda=args.hucal_slope_lambda,
        intercept_lambda=args.hucal_intercept_lambda,
        bin_weights=hucal_bin_weights,
    )

    if args.head_type == "context":
        if args.joint:
            arm = ("F (joint context head + L_HU-Cal)"
                   if args.hucal_weight > 0.0 else "E (joint context head)")
        else:
            arm = "E-posthoc (context head, frozen trunk)"
    else:
        arm = ("joint intensity head (non-study configuration)"
               if args.joint else "D (post-hoc head, frozen trunk)")

    n_params = sum(prm.numel() for prm in head.parameters())
    loss_desc = (f"{args.mse_weight}*MSE + {args.hucal_weight}*(L_SoftBias"
                 f" + {args.hucal_slope_lambda}*|a-1|"
                 f" + {args.hucal_intercept_lambda}*|b|)")
    if args.head_type == "context" and args.center_lambda > 0.0:
        loss_desc += f" + {args.center_lambda}*Center"
    if args.water_anchor_lambda > 0.0:
        loss_desc += f" + {args.water_anchor_lambda}*WaterAnchor"

    print(f"\n{'='*68}")
    print(f"  CALIBRATION HEAD TRAINING — study arm: {arm}")
    print(f"  arch={args.arch.upper()} | split={args.split} | seed={args.seed}")
    print(f"  Trunk          : "
          + ("FROM SCRATCH (joint)" if args.joint
             else f"{trunk_ckpt} (FROZEN)"))
    print(f"  Head           : {args.head_type}, hidden={args.hidden}, "
          f"|corr|<={args.delta_hu} HU, T'>={1.0 - args.kappa:.2f} "
          f"({n_params} params)")
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

    groups = [{"params": head.parameters(), "lr": args.head_lr}]
    if args.joint:
        groups.append({"params": trunk.parameters(), "lr": args.lr})
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999))

    iteration = 0
    best_score = -float("inf")
    start = time.time()
    cycle = 0

    while iteration < args.max_iterations:
        cycle += 1
        t0 = time.time()
        head.train()
        if args.joint:
            trunk.train()
        total = count = 0.0
        bar = tqdm(train_loader, desc="  Train", leave=False,
                   dynamic_ncols=True)
        for batch in bar:
            if iteration >= args.max_iterations:
                break
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.joint:
                z = trunk(x)
            else:
                with torch.no_grad():
                    z = trunk(x)
            corr = head.correction(z)
            loss = objective(z, corr, y, head, args, hucal)
            loss.backward()
            optimizer.step()
            iteration += 1
            total += float(loss.detach())
            count += 1
            bar.set_postfix(iter=iteration, loss=f"{loss.item():.6f}")
        train_loss = total / max(1, count)

        val = validate(wrapped, val_loader, device)
        val_obj = validation_objective(trunk, head, val_loader, device,
                                       args, hucal)
        score = (-val_obj if args.select_by == "val_loss"
                 else selection_score(val, args.select_by))

        meta = {
            "architecture":    args.arch,
            "study_arm":       arm,
            "head_type":       args.head_type,
            "joint":           bool(args.joint),
            "trunk_checkpoint": trunk_ckpt,
            "split":           args.split,
            "seed":            int(args.seed),
            "head_hidden":     args.hidden,
            "delta_hu":        args.delta_hu,
            "kappa":           args.kappa,
            "head_lr":         args.head_lr,
            "trunk_lr":        args.lr if args.joint else None,
            "mse_weight":      args.mse_weight,
            "hucal_weight":    args.hucal_weight,
            "hucal_slope_lambda":     args.hucal_slope_lambda,
            "hucal_intercept_lambda": args.hucal_intercept_lambda,
            "hucal_bin_weights":      hucal_bin_weights,
            "center_lambda":          args.center_lambda,
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
        if args.joint:
            trunk_payload = {
                "model_state_dict": get_state_dict(trunk),
                "meta":      meta,
                "iteration": iteration,
                "ssim":      val["ssim"],
                "psnr":      val["psnr"],
                "val_mse":   val["mse"],
                "score":     score,
                "select_by": args.select_by,
                "val_detail": {k: v for k, v in val.items()},
            }
            torch.save(trunk_payload, os.path.join(out_dir, "last_model.pt"))
        if score > best_score:
            best_score = score
            save_head(head, os.path.join(out_dir, "best_head.pt"), extra)
            if args.joint:
                torch.save(trunk_payload,
                           os.path.join(out_dir, "best_model.pt"))

        print(
            f"Cycle {cycle:02d} | Iter {iteration:05d}/{args.max_iterations} | "
            f"Loss {train_loss:.6f} | ValObj {val_obj:.6f} | "
            f"PSNR {val['psnr']:.3f} | SSIM {val['ssim']:.5f} | "
            f"bSSIM {val['bench_ssim']:.5f} | RMSE {val['rmse']:.2f} | "
            f"{args.select_by} {score:.6f} | {time.time() - t0:.1f}s"
        )

    # ------------------------------------------------------------------
    # Final diagnostics
    # ------------------------------------------------------------------
    head.eval()
    if args.head_type == "intensity":
        hu_in, hu_out = head.transfer_curve()
        corr_hu = (hu_out - hu_in).detach().cpu().numpy()
        hu_np = hu_in.detach().cpu().numpy()
        print("\nLearned transfer-curve correction T(HU) - HU:")
        for name, center in (("AirLung", -762), ("FatLow", -350), ("Soft", 0),
                             ("Dense", 400), ("Bone", 1250)):
            i = int(abs(hu_np - center).argmin())
            print(f"  {name:<8} ({center:>5} HU): {corr_hu[i]:+7.2f} HU")
        i0 = int(abs(hu_np - 0.0).argmin())
        print(f"  Water anchor |T(0)|: {abs(corr_hu[i0]):.2f} HU")
        print(f"  Max |correction|   : {abs(corr_hu).max():.2f} HU "
              f"(bound: {args.delta_hu} HU)")
    else:
        # Show that the context pathway differentiates images: per-image
        # AirLung occupancy (chest proxy) vs applied correction stats.
        print("\nContext differentiation on one validation batch "
              "(AirLung occupancy is a chest proxy):")
        batch = next(iter(val_loader))
        x = batch["image"].to(device)[:8]
        with torch.no_grad():
            z = trunk(x)
            ctx = head.context(z)
            corr = head.correction(z, context=ctx)
        corr_hu = corr * BENCHMARK_PIXEL_STD
        for i in range(x.shape[0]):
            print(f"  img {i}: AirLung occ {float(ctx[i, 2]):.4f} | "
                  f"mean corr {float(corr_hu[i].mean()):+7.2f} HU | "
                  f"max |corr| {float(corr_hu[i].abs().max()):6.2f} HU")

    total_t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"\nDone [{arm} / {args.arch.upper()}] in {total_t} | "
          f"best {args.select_by}={best_score:.6f}")
    print(f"Head checkpoint -> {os.path.join(out_dir, 'best_head.pt')}")
    if args.joint:
        print(f"Trunk checkpoint -> {os.path.join(out_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
