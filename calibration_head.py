"""Constrained plug-in HU calibration head (arm D; extended in arm E).

T(z) = z + s * tanh(g_scaled(z)), applied POINT-WISE to the denoised output
in the standardized intensity domain (z = (HU + 1024 - mean) / std).

Guarantees BY CONSTRUCTION (not by hoping the optimizer behaves):

1. Bounded     : |T(z) - z| <= s, with s = delta_hu / BENCHMARK_PIXEL_STD.
                 The head can never move any pixel by more than delta_hu HU
                 (default 80 HU -- chosen to cover the largest audited bias,
                 FatLow +67 HU chest, with margin).
2. Monotonic   : T'(z) = 1 + s * sech^2(.) * d/dz[g_scaled] >= 1 - kappa > 0.
                 g is rescaled at every forward by min(1, kappa / (s * L_ub)),
                 where L_ub = product of layer spectral norms (tanh slope <= 1)
                 is a certified upper bound on Lip(g). No HU ordering can
                 ever be inverted by the head.
3. Identity init: the last layer is zero-initialized, so g == 0 and
                 T == identity at step 0. Training can only move away from
                 identity where the calibration objective demands it.
4. Water anchor (optional, TESTED not assumed): penalty on |T(z_0HU) - z_0HU|
                 exposed as `water_anchor_penalty()`.

The arm-D head is intensity-only (a 1D transfer curve, ~1k params). The
context-conditioned variant (arm E) will condition g on image-level features
with an anti-collapse constraint.
"""

import torch
import torch.nn as nn

import config as cfg
from benchmark_data import BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD

# Standardized value of water (0 HU); the pixel domain is HU + 1024.
Z_WATER = (0.0 + cfg.HU_OFFSET - BENCHMARK_PIXEL_MEAN) / BENCHMARK_PIXEL_STD


class CalibrationHead(nn.Module):
    """Bounded, certified-monotonic, identity-initialized intensity map."""

    def __init__(self, hidden: int = 32, delta_hu: float = 80.0,
                 kappa: float = 0.9):
        super().__init__()
        if not (0.0 < kappa < 1.0):
            raise ValueError("kappa must be in (0, 1)")
        if delta_hu <= 0.0:
            raise ValueError("delta_hu must be > 0")
        self.hidden = int(hidden)
        self.delta_hu = float(delta_hu)
        self.kappa = float(kappa)
        # Max correction magnitude in standardized units.
        self.s = float(delta_hu) / BENCHMARK_PIXEL_STD
        self.fc1 = nn.Linear(1, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.fc3.weight)   # identity init: g == 0 at step 0
        nn.init.zeros_(self.fc3.bias)

    def config(self) -> dict:
        return {"hidden": self.hidden, "delta_hu": self.delta_hu,
                "kappa": self.kappa}

    def _lip_upper_bound(self) -> torch.Tensor:
        """Certified upper bound on Lip(g): product of spectral norms
        (tanh has slope <= 1). Differentiable, so training feels the
        constraint instead of colliding with it."""
        lip = None
        for fc in (self.fc1, self.fc2, self.fc3):
            n = torch.linalg.matrix_norm(fc.weight, ord=2)
            lip = n if lip is None else lip * n
        return lip

    def correction(self, z: torch.Tensor) -> torch.Tensor:
        """s * tanh(g_scaled(z)); same shape as z, standardized units."""
        h = z.reshape(-1, 1)
        h = torch.tanh(self.fc1(h))
        h = torch.tanh(self.fc2(h))
        g = self.fc3(h).reshape(z.shape)
        lip = self._lip_upper_bound()
        # Ensure s * Lip(tanh(g_scaled)) <= kappa < 1  =>  T strictly monotone.
        scale = torch.clamp(self.kappa / (self.s * lip + 1e-12), max=1.0)
        return self.s * torch.tanh(g * scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.correction(z)

    def water_anchor_penalty(self) -> torch.Tensor:
        """Squared correction at 0 HU (water). Add with a lambda to the
        training objective to TEST the water-anchor constraint."""
        z0 = torch.tensor([Z_WATER], device=self.fc1.weight.device,
                          dtype=self.fc1.weight.dtype)
        return (self.correction(z0) ** 2).sum()

    @torch.no_grad()
    def transfer_curve(self, hu_min: float = -1024.0, hu_max: float = 1900.0,
                       n: int = 1024):
        """(hu_in, hu_out) of the learned transfer curve in physical HU."""
        hu = torch.linspace(hu_min, hu_max, n, device=self.fc1.weight.device)
        z = (hu + cfg.HU_OFFSET - BENCHMARK_PIXEL_MEAN) / BENCHMARK_PIXEL_STD
        t = self.forward(z)
        hu_out = t * BENCHMARK_PIXEL_STD + BENCHMARK_PIXEL_MEAN - cfg.HU_OFFSET
        return hu, hu_out


def save_head(head: CalibrationHead, path: str, extra: dict | None = None):
    payload = {"head_state_dict": head.state_dict(),
               "head_config": head.config()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_head(path: str, device) -> CalibrationHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    head = CalibrationHead(**payload["head_config"]).to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    return head
