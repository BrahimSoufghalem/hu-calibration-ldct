"""HU-fidelity losses for the study arms B and C.

All losses operate on tensors in the BENCHMARK-STANDARDIZED domain (the
network's native input/output domain: z = (HU + 1024 - MEAN) / STD) and
return values in standardized units so their magnitudes are directly
comparable with the standardized MSE base loss. Tissue-bin membership is
computed from the REFERENCE (NDCT) image in physical HU.

Arm B -- L_HU (control)
-----------------------
``hu_mae_loss`` is the mean absolute error in HU divided by STD, i.e.
exactly ``F.l1_loss`` in standardized units. This transparency is the
point of the control arm: naive "HU supervision" is a scaled L1 term,
sign-blind and pixel-averaged, so it cannot specifically target the
per-tissue bias structure measured in Phase 1.

Arm C -- L_HU-Cal
-----------------
``HUCalLoss`` = L_SoftBias + lambda_s * |alpha - 1| + lambda_b * |beta|

- L_SoftBias: squared mean error inside each tissue bin with GAUSSIAN soft
  membership computed on the reference HU (soft binning is mandatory: hard
  bins showed sign instability at bin edges in earlier experiments).
  sigma_k = 0.25 * bin width, identical to hu_audit.py.
- alpha/beta: differentiable mass-weighted least-squares line fitted to the
  per-bin (mean ref, mean pred) points of the current batch. Phase-1 note:
  the GLOBAL (alpha, beta) is near-perfect on baselines while per-bin biases
  reach +/-67 HU (opposite signs cancel in the regression), so the slope /
  intercept penalties are secondary regularizers -- L_SoftBias is the
  primary term.
- Bias and beta are expressed in standardized units (HU / STD); alpha is
  dimensionless. Bins whose soft mass in the batch is negligible are
  skipped (their gradient would be noise).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_data import BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD

HU_OFFSET = 1024.0

# Fixed physical tissue intervals (HU) -- identical to hu_audit.py.
TISSUE_BINS = (
    ("AirLung", -1024.0, -500.0),
    ("FatLow",   -500.0, -200.0),
    ("Soft",     -200.0,  200.0),
    ("Dense",     200.0,  600.0),
    ("Bone",      600.0, 1900.0),
)
SOFT_SIGMA_FRACTION = 0.25   # sigma_k = 0.25 * bin width (same as audit)
_MIN_SOFT_MASS = 32.0        # skip bins with negligible membership mass

BIN_NAMES = tuple(name for name, _, _ in TISSUE_BINS)


def to_hu(z: torch.Tensor) -> torch.Tensor:
    """Benchmark-standardized tensor -> physical HU."""
    return z * BENCHMARK_PIXEL_STD + (BENCHMARK_PIXEL_MEAN - HU_OFFSET)


def hu_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_HU (arm B): MAE in physical HU, expressed in standardized units.

    Identity: MAE_HU / STD == L1(pred_z, target_z). Kept as an explicit
    function so the control arm's definition is visible and testable.
    """
    return F.l1_loss(pred, target)


def threshold_no_harm_loss(pred: torch.Tensor, trunk: torch.Tensor,
                           target: torch.Tensor, thresholds_hu: torch.Tensor,
                           temperature_hu: float = 5.0,
                           worst_weight: float = 1.0,
                           cvar_fraction: float = 0.0) -> torch.Tensor:
    """Penalize threshold disagreement regressions relative to the trunk.

    Thresholds can span the full HU range rather than encoding one clinical
    cutoff. A sigmoid makes each threshold comparison differentiable. The
    Regressions are rectified per image and threshold before reduction, so one
    patient cannot hide harm to another. The mean protects the sampled range;
    CVaR over each image's worst thresholds prevents local harm from being
    hidden without relying on one noisy maximum.
    """
    if temperature_hu <= 0.0:
        raise ValueError("temperature_hu must be > 0")
    if worst_weight < 0.0:
        raise ValueError("worst_weight must be >= 0")
    if not (0.0 <= cvar_fraction <= 1.0):
        raise ValueError("cvar_fraction must be in [0, 1]")
    thresholds = thresholds_hu.reshape(-1)
    if thresholds.numel() == 0:
        raise ValueError("thresholds_hu must not be empty")

    if pred.shape != trunk.shape or pred.shape != target.shape:
        raise ValueError("pred, trunk and target must have identical shapes")
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        trunk = trunk.unsqueeze(0)
        target = target.unsqueeze(0)
    pred_hu = to_hu(pred).reshape(pred.shape[0], -1)
    trunk_hu = to_hu(trunk.detach()).reshape(pred.shape[0], -1)
    target_hu = to_hu(target.detach()).reshape(pred.shape[0], -1)
    regressions = []
    for threshold in thresholds:
        target_pos = (target_hu > threshold).to(dtype=pred_hu.dtype)
        pred_pos = torch.sigmoid((pred_hu - threshold) / temperature_hu)
        trunk_pos = torch.sigmoid((trunk_hu - threshold) / temperature_hu)
        head_disagree = (pred_pos - target_pos).abs().mean(dim=1)
        trunk_disagree = (trunk_pos - target_pos).abs().mean(dim=1)
        regressions.append(F.relu(head_disagree - trunk_disagree))

    regressions = torch.stack(regressions, dim=1)
    if cvar_fraction == 0.0:
        cvar = regressions.max(dim=1).values.mean()
    else:
        tail_size = math.ceil(regressions.shape[1] * cvar_fraction)
        cvar = regressions.topk(tail_size, dim=1).values.mean(dim=1).mean()
    return regressions.mean() + worst_weight * cvar


def soft_bin_weights(ref_hu: torch.Tensor):
    """Yield (name, weight-map) Gaussian soft memberships from reference HU."""
    for name, lo, hi in TISSUE_BINS:
        center = 0.5 * (lo + hi)
        sigma = SOFT_SIGMA_FRACTION * (hi - lo)
        yield name, torch.exp(-0.5 * ((ref_hu - center) / sigma) ** 2)


class HUCalLoss(nn.Module):
    """L_HU-Cal (arm C) = L_SoftBias + lambda_s*|alpha-1| + lambda_b*|beta|.

    Args:
        slope_lambda:     weight of |alpha - 1| (dimensionless slope).
        intercept_lambda: weight of |beta| (beta in standardized units).
        bin_weights:      optional per-bin weights for L_SoftBias, length 5,
                          order AirLung/FatLow/Soft/Dense/Bone. None =
                          uniform. Phase-1 suggests up-weighting
                          FatLow/Dense/Bone if the uniform version
                          underfits the chest bins.
    """

    def __init__(self, slope_lambda: float = 0.1,
                 intercept_lambda: float = 0.01,
                 bin_weights=None):
        super().__init__()
        if bin_weights is not None:
            if len(bin_weights) != len(TISSUE_BINS):
                raise ValueError(
                    f"bin_weights needs {len(TISSUE_BINS)} values "
                    f"({', '.join(BIN_NAMES)})")
            if any(w < 0 for w in bin_weights):
                raise ValueError("bin_weights must be >= 0")
            if all(w == 0 for w in bin_weights):
                raise ValueError("bin_weights must not be all zero")
        self.slope_lambda = float(slope_lambda)
        self.intercept_lambda = float(intercept_lambda)
        self.bin_weights = (list(map(float, bin_weights))
                            if bin_weights is not None
                            else [1.0] * len(TISSUE_BINS))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                return_parts: bool = False):
        ref_hu = to_hu(target).detach()
        err = pred - target                      # standardized units

        bias_sq_sum = pred.new_zeros(())
        weight_sum = 0.0
        mu_ref, mu_pred, mass = [], [], []

        for w_bin, (name, w) in zip(self.bin_weights, soft_bin_weights(ref_hu)):
            wsum = w.sum()
            if float(wsum) < _MIN_SOFT_MASS:
                continue
            bias = (w * err).sum() / wsum        # standardized units
            if w_bin > 0.0:
                bias_sq_sum = bias_sq_sum + w_bin * bias * bias
                weight_sum += w_bin
            mu_ref.append((w * target.detach()).sum() / wsum)
            mu_pred.append((w * pred).sum() / wsum)
            mass.append(wsum.detach())

        l_softbias = bias_sq_sum / max(1.0, weight_sum)

        # Differentiable mass-weighted least squares over the present bins.
        if len(mu_ref) >= 2:
            x = torch.stack(mu_ref)
            y = torch.stack(mu_pred)
            m = torch.stack(mass)
            m = m / m.sum()
            xm = (m * x).sum()
            ym = (m * y).sum()
            var = (m * (x - xm) ** 2).sum()
            cov = (m * (x - xm) * (y - ym)).sum()
            alpha = cov / var.clamp_min(1e-8)
            beta = ym - alpha * xm               # standardized units
            l_cal = (self.slope_lambda * (alpha - 1.0).abs()
                     + self.intercept_lambda * beta.abs())
        else:
            alpha = pred.new_ones(())
            beta = pred.new_zeros(())
            l_cal = pred.new_zeros(())

        total = l_softbias + l_cal
        if return_parts:
            return total, {
                "l_softbias": float(l_softbias.detach()),
                "l_cal": float(l_cal.detach()),
                "alpha": float(alpha.detach()),
                "beta_std": float(beta.detach()),
            }
        return total
