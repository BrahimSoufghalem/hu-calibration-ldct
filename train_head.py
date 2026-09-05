"""Arms D & E -- constrained calibration heads on top of a denoising trunk.

Arm D : intensity-only head (1D transfer curve), post-hoc on a FROZEN trunk.
Arm E : context-conditioned head -- same constrained curve, but g(z, c) is
        conditioned on a DETACHED per-image context vector. Includes the
        anti-collapse centering penalty (per-image mean correction -> 0).

v2 changes (motivated by the E-patch/E-slice results)
-----------------------------------------------------
1. --hucal-reduction image (DEFAULT): L_HU-Cal is computed PER IMAGE and
   averaged. The original batch-pooled reduction (still available with
   --hucal-reduction batch; used by the original arm D/E runs) collapses a
   mixed chest/abdomen batch into one pooled bias target dominated by the
   chest, destroying gradient attribution for the context pathway --
   exactly the observed collapse of E onto D's global curve.
2. --context-oracle: diagnostic UPPER BOUND. APPENDS the ground-truth
   body-type one-hot (from batch['body_type']) to the inferred context, so
   the oracle sees a strict SUPERSET of what the inferred head sees.
   Oracle runs: post-hoc only, --select-by val_loss only. hu_audit.py
   audits oracle heads correctly (it sets head.oracle_body per patient).

v3 changes (motivated by an independent re-analysis of the D/E audit tables)
---------------------------------------------------------------------------
3. The oracle context is now ADDITIVE rather than REPLACING. Previously it
   zeroed the 7 inferred features and kept only a 2-dim one-hot, so the
   "upper bound" actually had LESS information than the arm it was meant to
   bound, on a different input support -- an oracle loss would have been
   uninterpretable in both directions.
4. Iteration-0 validation is logged. The head is identity at init, so cycle 00
   is exactly the frozen trunk inside this protocol -- the missing baseline
   for every "the head costs no image quality" claim.
5. The context diagnostic no longer prints only the first validation batch.
   That batch is a SINGLE chest patient (shuffle=False, patient-contiguous
   slices, EXPECTED_VAL starts with 10 chest patients), so the old printout
   could not distinguish a collapsed context from a working one. It now scans
   both anatomies and reports between-anatomy separation, plus an oracle
   counterfactual (same image, label flipped) that isolates the label effect.

Pre-registered endpoint (calibration_head.PRIMARY_ENDPOINT)
-----------------------------------------------------------
dChest/dAbd applied-correction ratio on Bone, computed from the audit tables:
    (bias_arm_chest - bias_A_chest) / (bias_arm_abd - bias_A_abd)
Anatomy-blind = 1.00 ; ideal = 9.53 . D=0.990, E-v1=0.993, E-slice=0.986,
E-v2=0.998 -- every existing arm is anatomy-blind. Not a loss term, not the
selection criterion. Secondary non-circular endpoint: ThrDisagree_130HU_pct.

Modes
-----
- post-hoc (default) : trunk FROZEN, head trained alone (tiny budget).
- --joint            : trunk trained FROM SCRATCH together with the head
                       (use the full matched budget: --max-iterations 30000
                       --iterations-before-val 1000 --select-by bench_ssim).
                       With --hucal-weight > 0 this is arm F (losses+head).

Objective
---------
    L = w_mse*MSE + w_cal*L_HU-Cal[per-image] + lambda_c*Center(corr)
        [+ lambda_w*WaterAnchor]

Usage
-----
# Arm E v2 (context head, inferred context, per-image L_HU-Cal):
    HU_RANGE_PRESET=benchmark python train_head.py --arch redcnn \\
        --data-dir dataset --split 100p --head-type context \\
        --output-root runs_armE_v2

# Oracle diagnostic (inferred context + ground-truth body one-hot).
# Keep every other flag identical to the E-v2 run above: the body label is
# the ONLY variable that may differ, or the comparison is uninterpretable.
    HU_RANGE_PRESET=benchmark python train_head.py --arch redcnn \\
        --data-dir dataset --split 100p --head-type context \\
        --context-oracle --output-root runs_armE_oracle

# E-full-slice-context: the only E-v2 change is where the seven context
# statistics are measured. They come from the paired uncropped low-dose slice;
# the trunk, patch, loss, reduction, and budget remain unchanged.
    HU_RANGE_PRESET=benchmark python train_head.py --arch redcnn \\
        --data-dir dataset --split 100p --head-type context \\
        --context-full-slice --output-root runs_armE_full_slice

# Arm D (intensity head; add --hucal-reduction batch to reproduce the
# original arm D run exactly):
    HU_RANGE_PRESET=benchmark python train_head.py --arch redcnn \\
        --data-dir dataset --split 100p --output-root runs_armD

# Audit (trunk rows from runs/, head rows from <heads-root>/):
    HU_RANGE_PRESET=benchmark python hu_audit.py --test-dir test \\
        --runs-root runs --heads-root runs_armE_v2 --archs redcnn \\
        --include-input --output hu_audit_e_v2
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
from hu_losses import HUCalLoss, BIN_NAMES, threshold_no_harm_loss
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
    p.add_argument("--context-oracle", action="store_true",
                   help="Diagnostic upper bound: append the ground-truth "
                        "body-type one-hot to inferred context. "
                        "Requires --head-type context, post-hoc mode, and "
                        "--select-by val_loss.")
    p.add_argument("--context-full-slice", action="store_true",
                   help="Infer context from the matching uncropped low-dose "
                        "slice while trunk/loss use the configured patch.")
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
    p.add_argument("--hucal-reduction", choices=["image", "batch"],
                   default="image",
                   help="'image' (default, v2): L_HU-Cal per image, then "
                        "averaged -- required for correct gradient "
                        "attribution with a context head. 'batch': original "
                        "pooled reduction (reproduces the first D/E runs).")
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
    p.add_argument("--threshold-no-harm-lambda", type=float, default=0.0,
                   help="Penalize soft threshold-disagreement regression "
                        "versus the trunk over sampled HU thresholds.")
    p.add_argument("--threshold-min-hu", type=float, default=-1000.0)
    p.add_argument("--threshold-max-hu", type=float, default=1500.0)
    p.add_argument("--threshold-samples", type=int, default=16,
                   help="Random thresholds per training batch; validation "
                        "uses an evenly spaced grid of the same size.")
    p.add_argument("--threshold-pixel-samples", type=int, default=65536,
                   help="Maximum pixels sampled per batch for the threshold "
                        "loss, to bound memory use.")
    p.add_argument("--threshold-temperature-hu", type=float, default=5.0,
                   help="Sigmoid temperature for differentiable crossings.")
    p.add_argument("--threshold-worst-weight", type=float, default=1.0,
                   help="Weight of worst-threshold CVaR in addition to mean "
                        "per-image threshold regression.")
    p.add_argument("--threshold-cvar-fraction", type=float, default=0.0,
                   help="Fraction of worst thresholds used by CVaR; 0 keeps "
                        "the legacy single-maximum reduction.")
    p.add_argument("--threshold-density-fraction", type=float, default=0.0,
                   help="Fraction of thresholds sampled from target HU density; "
                        "the remainder cover the HU range uniformly.")
    p.add_argument("--curve-identity-lambda", type=float, default=0.0,
                   help="Penalize mean squared correction over an HU grid.")
    p.add_argument("--curve-slope-lambda", type=float, default=0.0,
                   help="Penalize squared correction slope over an HU grid.")
    p.add_argument("--curve-grid-points", type=int, default=128)

    p.add_argument("--select-by", choices=list(_SELECT_CHOICES),
                   default="val_loss",
                   help="'val_loss' selects by the calibration objective on "
                        "validation; global metrics are reported alongside. "
                        "For --joint runs, bench_ssim is recommended to "
                        "match the A/B/C selection protocol.")
    return p.parse_args()


def build_head(args, device):
    if args.head_type == "context":
        return ContextCalibrationHead(
            hidden=args.hidden, delta_hu=args.delta_hu, kappa=args.kappa,
            oracle=args.context_oracle,
            full_slice_context=args.context_full_slice).to(device)
    return CalibrationHead(hidden=args.hidden, delta_hu=args.delta_hu,
                           kappa=args.kappa).to(device)


def batch_context(head, batch, z):
    """Explicit context for a batch: inferred features PLUS the ground-truth
    body one-hot when in oracle mode, otherwise None (the head infers its own
    statistics). The oracle context is a strict superset of the inferred one."""
    if isinstance(head, ContextCalibrationHead):
        inferred = batch.get("full_context")
        if inferred is None:
            inferred = head.inferred_context(z)
        else:
            inferred = inferred.to(z.device, non_blocking=True).detach()
        if head.oracle:
            onehot = head._body_one_hot(batch["body_type"], z.device, z.dtype)
            return torch.cat([inferred, onehot], dim=1)
        if "full_context" in batch:
            return inferred
    return None


def hucal_term(pred, target, hucal, reduction):
    """L_HU-Cal with per-image or batch-pooled bin statistics."""
    if reduction == "batch":
        return hucal(pred, target)
    b = pred.shape[0]
    total = 0.0
    for i in range(b):
        total = total + hucal(pred[i:i + 1], target[i:i + 1])
    return total / max(1, b)


def sampled_thresholds(args, reference: torch.Tensor,
                       training: bool) -> torch.Tensor:
    """Combine HU-range coverage with thresholds from the observed HU density."""
    if not training:
        return torch.linspace(args.threshold_min_hu, args.threshold_max_hu,
                              args.threshold_samples, device=reference.device,
                              dtype=reference.dtype)
    density_count = round(args.threshold_samples
                          * args.threshold_density_fraction)
    if 0.0 < args.threshold_density_fraction < 1.0:
        density_count = min(density_count, args.threshold_samples - 2)
    uniform_count = args.threshold_samples - density_count
    if uniform_count == 0:
        uniform = reference.new_empty(0)
    else:
        uniform = torch.rand(uniform_count, device=reference.device,
                             dtype=reference.dtype)
        uniform = args.threshold_min_hu + uniform * (
            args.threshold_max_hu - args.threshold_min_hu)

    if density_count == 0:
        return uniform
    reference_hu = (reference.detach().reshape(-1) * BENCHMARK_PIXEL_STD
                    + BENCHMARK_PIXEL_MEAN - cfg.HU_OFFSET)
    reference_hu = reference_hu[
        (reference_hu >= args.threshold_min_hu)
        & (reference_hu <= args.threshold_max_hu)]
    if reference_hu.numel() == 0:
        density = torch.linspace(args.threshold_min_hu, args.threshold_max_hu,
                                 density_count, device=reference.device,
                                 dtype=reference.dtype)
    else:
        indices = torch.randint(reference_hu.numel(), (density_count,),
                                device=reference.device)
        density = reference_hu[indices]
    return torch.cat([uniform, density]).sort().values


def sampled_threshold_pixels(pred, trunk, target, max_pixels, training):
    """Select aligned pixels per image, preserving patient-level reductions."""
    if pred.shape != trunk.shape or pred.shape != target.shape:
        raise ValueError("pred, trunk and target must have identical shapes")
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        trunk = trunk.unsqueeze(0)
        target = target.unsqueeze(0)
    batch = pred.shape[0]
    if max_pixels < batch:
        raise ValueError("max_pixels must allow at least one pixel per image")
    pred = pred.reshape(batch, -1)
    trunk = trunk.reshape(batch, -1)
    target = target.reshape(batch, -1)
    per_image = max(1, max_pixels // batch)
    if pred.shape[1] <= per_image:
        return pred, trunk, target
    if training:
        indices = torch.randint(pred.shape[1], (batch, per_image),
                                device=pred.device)
    else:
        one_image = torch.linspace(0, pred.shape[1] - 1, per_image,
                                   device=pred.device).long()
        indices = one_image.unsqueeze(0).expand(batch, -1)
    return (pred.gather(1, indices), trunk.gather(1, indices),
            target.gather(1, indices))


def curve_regularization(head, reference: torch.Tensor, args, ctx=None):
    """Return identity and slope penalties on the whole configured HU range."""
    hu = torch.linspace(args.threshold_min_hu, args.threshold_max_hu,
                        args.curve_grid_points, device=reference.device,
                        dtype=reference.dtype)
    z_grid = ((hu + cfg.HU_OFFSET - BENCHMARK_PIXEL_MEAN)
              / BENCHMARK_PIXEL_STD)
    if isinstance(head, ContextCalibrationHead):
        if ctx is None:
            raise ValueError("context is required for context-curve regularization")
        grid = z_grid.reshape(1, -1).expand(ctx.shape[0], -1)
        curve_corr = head.correction(grid, context=ctx)
    else:
        grid = z_grid.reshape(1, -1)
        curve_corr = head.correction(grid)

    identity = curve_corr.square().mean()
    dz = z_grid[1:] - z_grid[:-1]
    slope = ((curve_corr[..., 1:] - curve_corr[..., :-1]) / dz).square().mean()
    return identity, slope


def objective(z, corr, target, head, args, hucal, ctx=None, training=True):
    pred = z + corr
    loss = args.mse_weight * F.mse_loss(pred, target)
    if args.hucal_weight > 0.0:
        loss = loss + args.hucal_weight * hucal_term(
            pred, target, hucal, args.hucal_reduction)
    if args.center_lambda > 0.0 and isinstance(head, ContextCalibrationHead):
        loss = loss + args.center_lambda \
            * ContextCalibrationHead.centering_penalty(corr)
    if args.water_anchor_lambda > 0.0:
        if isinstance(head, ContextCalibrationHead):
            anchor = head.water_anchor_penalty(z, context=ctx)
        else:
            anchor = head.water_anchor_penalty()
        loss = loss + args.water_anchor_lambda * anchor
    if args.threshold_no_harm_lambda > 0.0:
        threshold_pred, threshold_trunk, threshold_target = \
            sampled_threshold_pixels(
                pred, z, target, args.threshold_pixel_samples, training)
        thresholds = sampled_thresholds(args, threshold_target, training)
        loss = loss + args.threshold_no_harm_lambda * threshold_no_harm_loss(
            threshold_pred, threshold_trunk, threshold_target, thresholds,
            temperature_hu=args.threshold_temperature_hu,
            worst_weight=args.threshold_worst_weight,
            cvar_fraction=args.threshold_cvar_fraction)
    if args.curve_identity_lambda > 0.0 or args.curve_slope_lambda > 0.0:
        identity, slope = curve_regularization(head, pred, args, ctx=ctx)
        loss = (loss + args.curve_identity_lambda * identity
                + args.curve_slope_lambda * slope)
    return loss


def compute_correction(head, batch, z):
    ctx = batch_context(head, batch, z)
    if isinstance(head, ContextCalibrationHead):
        return head.correction(z, context=ctx), ctx
    return head.correction(z), None


_BIN_CENTERS = (("AirLung", -762), ("FatLow", -350), ("Soft", 0),
                ("Dense", 400), ("Bone", 1250))


@torch.no_grad()
def _context_diagnostics(trunk, head, val_loader, device, args):
    """Anatomy-differentiation diagnostic for the context head.

    Sampling note: val_loader has shuffle=False and collect_files() emits all
    slices of one patient before the next, while EXPECTED_VAL starts with 10
    chest patients (~250 slices each). So the FIRST batch is a single chest
    patient. An earlier version of this diagnostic printed exactly that batch
    and concluded 'the context has collapsed' -- but 8 consecutive slices of
    one chest patient are SUPPOSED to share a context vector, so that printout
    could not distinguish a collapsed head from a working one.

    We therefore scan the whole validation set and keep chest and abdomen
    slices separately, then report the between-anatomy separation.
    """
    trunk.eval()
    head.eval()
    is_ctx = isinstance(head, ContextCalibrationHead)

    per_body = {"Chest": [], "Abdomen": []}
    ctx_rows = {"Chest": [], "Abdomen": []}
    for batch in val_loader:
        x = batch["image"].to(device, non_blocking=True)
        bodies = list(batch["body_type"])
        z = trunk(x)
        ctx = batch_context(head, batch, z) if is_ctx else None
        if is_ctx and ctx is None:
            ctx = head.context(z)
        corr = head.correction(z, context=ctx) * BENCHMARK_PIXEL_STD
        for i, b in enumerate(bodies):
            key = "Chest" if str(b).lower().startswith("c") else "Abdomen"
            if len(per_body[key]) < 200:
                per_body[key].append(float(corr[i].mean()))
                ctx_rows[key].append(ctx[i].detach().cpu())
        if min(len(per_body["Chest"]), len(per_body["Abdomen"])) >= 200:
            break

    print("\nContext differentiation over the validation set")
    print(f"  (scanned until 200 slices per anatomy; "
          f"chest {len(per_body['Chest'])}, "
          f"abdomen {len(per_body['Abdomen'])})")
    for key in ("Chest", "Abdomen"):
        v = per_body[key]
        if not v:
            print(f"  {key:<8}: none found")
            continue
        m = sum(v) / len(v)
        sd = (sum((a - m) ** 2 for a in v) / max(1, len(v) - 1)) ** 0.5
        print(f"  {key:<8}: mean corr {m:+7.2f} HU | sd {sd:5.2f} | "
              f"min {min(v):+7.2f} | max {max(v):+7.2f}")
    if per_body["Chest"] and per_body["Abdomen"]:
        mc = sum(per_body["Chest"]) / len(per_body["Chest"])
        ma = sum(per_body["Abdomen"]) / len(per_body["Abdomen"])
        print(f"  between-anatomy separation of mean corr: "
              f"{abs(mc - ma):.2f} HU")

    if not is_ctx:
        return

    # Per-anatomy transfer curves at a representative (median) context.
    print("\n  Transfer-curve correction at bin centers, per anatomy context:")
    curves = {}
    for key in ("Chest", "Abdomen"):
        if not ctx_rows[key]:
            continue
        stack = torch.stack(ctx_rows[key]).to(device)
        c1 = stack.median(dim=0).values
        hu_in, hu_out = head.transfer_curve(c1)
        ch = (hu_out - hu_in).detach().cpu()
        hn = hu_in.detach().cpu()
        vals = {n: float(ch[int((hn - c).abs().argmin())])
                for n, c in _BIN_CENTERS}
        curves[key] = vals
        print(f"    {key:<8}: " + "  ".join(
            f"{n}:{vals[n]:+7.2f}" for n, _ in _BIN_CENTERS))
    if len(curves) == 2:
        print("    " + "-" * 60)
        print("    sep     : " + "  ".join(
            f"{n}:{abs(curves['Chest'][n] - curves['Abdomen'][n]):+7.2f}"
            for n, _ in _BIN_CENTERS))

    # The decisive counterfactual: same image, label flipped. Isolates the
    # effect of the body label with the inferred features held fixed.
    if head.oracle and ctx_rows["Chest"] and ctx_rows["Abdomen"]:
        print("\n  ORACLE counterfactual -- same inferred features, "
              "body label flipped:")
        for key in ("Chest", "Abdomen"):
            stack = torch.stack(ctx_rows[key]).to(device)
            base = stack.median(dim=0).values.reshape(1, -1)
            out = {}
            for body in ("Chest", "Abdomen"):
                c1 = head.with_body_override(base, body)[0]
                hu_in, hu_out = head.transfer_curve(c1)
                ch = (hu_out - hu_in).detach().cpu()
                hn = hu_in.detach().cpu()
                out[body] = {n: float(ch[int((hn - c).abs().argmin())])
                             for n, c in _BIN_CENTERS}
            print(f"    real {key} slices, forced label:")
            for body in ("Chest", "Abdomen"):
                print(f"      as {body:<8}: " + "  ".join(
                    f"{n}:{out[body][n]:+7.2f}" for n, _ in _BIN_CENTERS))
            print(f"      delta     : " + "  ".join(
                f"{n}:{abs(out['Chest'][n] - out['Abdomen'][n]):+7.2f}"
                for n, _ in _BIN_CENTERS))
        print("\n  PRE-REGISTERED ENDPOINT (calibration_head.PRIMARY_ENDPOINT):")
        print("    dChest/dAbd applied-correction ratio on Bone, from the")
        print("    audit tables. Anatomy-blind = 1.00 ; ideal = 9.53 .")
        print("    E-v2 measured 0.998 -- i.e. no anatomy dependence at all.")


@torch.no_grad()
def validation_objective(trunk, head, loader, device, args, hucal):
    trunk.eval()
    head.eval()
    total = 0.0
    count = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        z = trunk(x)
        corr, ctx = compute_correction(head, batch, z)
        batch_size = x.shape[0]
        total += batch_size * float(objective(
            z, corr, y, head, args, hucal, ctx=ctx, training=False))
        count += batch_size
    return total / max(1, count)


def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Set HU_RANGE_PRESET=benchmark.")
    if args.context_oracle and args.context_full_slice:
        raise ValueError("--context-oracle and --context-full-slice are mutually exclusive")
    if args.context_full_slice:
        if args.head_type != "context":
            raise ValueError("--context-full-slice requires --head-type context")
        if args.joint:
            raise ValueError("--context-full-slice is post-hoc only")
        if args.select_by != "val_loss":
            raise ValueError("--context-full-slice requires --select-by val_loss")
    if args.context_oracle:
        if args.head_type != "context":
            raise ValueError("--context-oracle requires --head-type context")
        if args.joint:
            raise ValueError("--context-oracle is a post-hoc diagnostic; "
                             "do not combine with --joint")
        if args.select_by != "val_loss":
            raise ValueError("--context-oracle requires --select-by "
                             "val_loss (metric validation cannot pass "
                             "body-type labels through the model forward)")
    if args.joint and args.max_iterations <= 5_000:
        print("  WARNING: --joint with a tiny budget "
              f"({args.max_iterations} iters). For reportable joint runs "
              "use the matched budget (30000).")
    if args.threshold_samples < 2:
        raise ValueError("--threshold-samples must be >= 2")
    if args.threshold_pixel_samples < 1:
        raise ValueError("--threshold-pixel-samples must be >= 1")
    if args.curve_grid_points < 2:
        raise ValueError("--curve-grid-points must be >= 2")
    if args.threshold_min_hu >= args.threshold_max_hu:
        raise ValueError("--threshold-min-hu must be less than --threshold-max-hu")
    for name in ("threshold_no_harm_lambda", "threshold_worst_weight",
                 "curve_identity_lambda", "curve_slope_lambda"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.threshold_temperature_hu <= 0.0:
        raise ValueError("--threshold-temperature-hu must be > 0")
    if not (0.0 <= args.threshold_cvar_fraction <= 1.0):
        raise ValueError("--threshold-cvar-fraction must be in [0, 1]")
    if not (0.0 <= args.threshold_density_fraction <= 1.0):
        raise ValueError("--threshold-density-fraction must be in [0, 1]")

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
        if args.context_oracle:
            arm = "E-oracle (ground-truth context, frozen trunk)"
        elif args.context_full_slice:
            arm = "E-full-slice-context (inferred full-slice context, frozen trunk)"
        elif args.joint:
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
                 f" + {args.hucal_intercept_lambda}*|b|)"
                 f"[{args.hucal_reduction}]")
    if args.head_type == "context" and args.center_lambda > 0.0:
        loss_desc += f" + {args.center_lambda}*Center"
    if args.water_anchor_lambda > 0.0:
        loss_desc += f" + {args.water_anchor_lambda}*WaterAnchor"
    if args.threshold_no_harm_lambda > 0.0:
        loss_desc += (f" + {args.threshold_no_harm_lambda}*ThresholdNoHarm"
                      f"[{args.threshold_samples} in "
                      f"{args.threshold_min_hu:g}:{args.threshold_max_hu:g} HU, "
                      f"CVaR={args.threshold_cvar_fraction:g}, "
                      f"density={args.threshold_density_fraction:g}]")
    if args.curve_identity_lambda > 0.0:
        loss_desc += f" + {args.curve_identity_lambda}*CurveIdentity"
    if args.curve_slope_lambda > 0.0:
        loss_desc += f" + {args.curve_slope_lambda}*CurveSlope"

    print(f"\n{'='*68}")
    print(f"  CALIBRATION HEAD TRAINING \u2014 study arm: {arm}")
    print(f"  arch={args.arch.upper()} | split={args.split} | seed={args.seed}")
    print(f"  Trunk          : "
          + ("FROM SCRATCH (joint)" if args.joint
             else f"{trunk_ckpt} (FROZEN)"))
    print(f"  Head           : {args.head_type}"
          + (" [ORACLE]" if args.context_oracle else "")
          + (" [FULL-SLICE CONTEXT]" if args.context_full_slice else "")
          + f", hidden={args.hidden}, "
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
        include_full_slice_context=args.context_full_slice,
    )

    groups = [{"params": head.parameters(), "lr": args.head_lr}]
    if args.joint:
        groups.append({"params": trunk.parameters(), "lr": args.lr})
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999))

    iteration = 0
    best_score = -float("inf")
    start = time.time()
    cycle = 0

    # Iteration-0 baseline. The head is identity-initialized, so this row IS
    # the bare frozen trunk measured inside this exact protocol. Without it
    # there is no in-protocol reference for PSNR/SSIM/RMSE, and the claim
    # "the head does not cost image quality" cannot be checked at all.
    if not args.joint:
        val0 = None if (args.context_oracle or args.context_full_slice) \
            else validate(wrapped, val_loader, device)
        obj0 = validation_objective(trunk, head, val_loader, device,
                                    args, hucal)
        if val0 is not None:
            print(
                f"Cycle 00 | Iter 00000/{args.max_iterations} | "
                f"Loss {float('nan'):.6f} | ValObj {obj0:.6f} | "
                f"PSNR {val0['psnr']:.3f} | SSIM {val0['ssim']:.5f} | "
                f"bSSIM {val0['bench_ssim']:.5f} | RMSE {val0['rmse']:.2f} | "
                f"{args.select_by} {-obj0:.6f} | IDENTITY INIT = FROZEN TRUNK"
            )
        else:
            print(
                f"Cycle 00 | Iter 00000/{args.max_iterations} | "
                f"ValObj {obj0:.6f} | val_loss {-obj0:.6f} | "
                f"IDENTITY INIT = FROZEN TRUNK"
            )

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
            corr, ctx = compute_correction(head, batch, z)
            loss = objective(z, corr, y, head, args, hucal, ctx=ctx)
            loss.backward()
            optimizer.step()
            iteration += 1
            total += float(loss.detach())
            count += 1
            bar.set_postfix(iter=iteration, loss=f"{loss.item():.6f}")
        train_loss = total / max(1, count)

        # Oracle heads cannot run the generic metric validation (labels
        # cannot pass through model.forward); the objective is enough for
        # val_loss selection.
        val = None if (args.context_oracle or args.context_full_slice) \
            else validate(wrapped, val_loader, device)
        val_obj = validation_objective(trunk, head, val_loader, device,
                                       args, hucal)
        score = (-val_obj if args.select_by == "val_loss"
                 else selection_score(val, args.select_by))

        meta = {
            "architecture":    args.arch,
            "study_arm":       arm,
            "head_type":       args.head_type,
            "context_oracle":  bool(args.context_oracle),
            "context_full_slice": bool(args.context_full_slice),
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
            "hucal_reduction": args.hucal_reduction,
            "hucal_slope_lambda":     args.hucal_slope_lambda,
            "hucal_intercept_lambda": args.hucal_intercept_lambda,
            "hucal_bin_weights":      hucal_bin_weights,
            "center_lambda":          args.center_lambda,
            "water_anchor_lambda":    args.water_anchor_lambda,
            "threshold_no_harm_lambda": args.threshold_no_harm_lambda,
            "threshold_min_hu": args.threshold_min_hu,
            "threshold_max_hu": args.threshold_max_hu,
            "threshold_samples": args.threshold_samples,
            "threshold_pixel_samples": args.threshold_pixel_samples,
            "threshold_temperature_hu": args.threshold_temperature_hu,
            "threshold_worst_weight": args.threshold_worst_weight,
            "threshold_cvar_fraction": args.threshold_cvar_fraction,
            "threshold_density_fraction": args.threshold_density_fraction,
            "curve_identity_lambda": args.curve_identity_lambda,
            "curve_slope_lambda": args.curve_slope_lambda,
            "curve_grid_points": args.curve_grid_points,
            "budget_iterations": args.max_iterations,
            "select_by":       args.select_by,
            "normalization":   "benchmark_meanstd",
            "pixel_mean":      BENCHMARK_PIXEL_MEAN,
            "pixel_std":       BENCHMARK_PIXEL_STD,
            "pixel_domain":    "HU+1024",
            "hu_preset":       cfg.HU_RANGE_PRESET,
            "eval_data_range": cfg.EVAL_DATA_RANGE,
            "loss":            loss_desc,
            "training_args":   vars(args).copy(),
        }
        extra = {"meta": meta, "iteration": iteration, "score": score,
                 "select_by": args.select_by, "val_objective": val_obj,
                 "val_detail": ({k: v for k, v in val.items()}
                                if val is not None else {})}
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

        if val is not None:
            print(
                f"Cycle {cycle:02d} | Iter {iteration:05d}/"
                f"{args.max_iterations} | Loss {train_loss:.6f} | "
                f"ValObj {val_obj:.6f} | PSNR {val['psnr']:.3f} | "
                f"SSIM {val['ssim']:.5f} | bSSIM {val['bench_ssim']:.5f} | "
                f"RMSE {val['rmse']:.2f} | {args.select_by} {score:.6f} | "
                f"{time.time() - t0:.1f}s"
            )
        else:
            print(
                f"Cycle {cycle:02d} | Iter {iteration:05d}/"
                f"{args.max_iterations} | Loss {train_loss:.6f} | "
                f"ValObj {val_obj:.6f} | val_loss {score:.6f} | "
                f"{time.time() - t0:.1f}s"
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
        _context_diagnostics(trunk, head, val_loader, device, args)

    total_t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"\nDone [{arm} / {args.arch.upper()}] in {total_t} | "
          f"best {args.select_by}={best_score:.6f}")
    print(f"Head checkpoint -> {os.path.join(out_dir, 'best_head.pt')}")
    if args.joint:
        print(f"Trunk checkpoint -> {os.path.join(out_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
