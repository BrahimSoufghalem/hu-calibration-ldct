"""Constrained plug-in HU calibration heads (arms D and E).

Both heads share the same guarantees BY CONSTRUCTION:

1. Bounded     : |T(z) - z| <= s, with s = delta_hu / BENCHMARK_PIXEL_STD
                 (default 80 HU -- covers the largest audited bias, FatLow
                 +67 HU chest, with margin).
2. Monotonic   : T'(z) >= 1 - kappa > 0, certified by rescaling g so that
                 s * Lip_z(g) <= kappa, where Lip_z is an upper bound from
                 the product of layer spectral norms (tanh slope <= 1).
3. Identity init: last layer zero-initialized -> T == identity at step 0.
4. Water anchor (optional, TESTED not assumed): penalty on |T(z_0HU)-z_0HU|.

Arm D -- CalibrationHead:        T(z)    = z + s * tanh(g(z))     (1D curve)
Arm E -- ContextCalibrationHead: T(z|c)  = z + s * tanh(g(z, c))  (per-image
         curve). c is a 7-dim per-image context vector: mean, std, and the
         5 Gaussian soft tissue-bin occupancy fractions. Motivated directly
         by the arm C/D finding: HU bias is anatomy-dependent (chest vs
         abdomen), so one global loss/curve cannot fix both regions;
         conditioning on occupancy (e.g. lung fraction) lets the head apply
         different curves to different anatomies.

Context-specific guarantees (arm E design locks):
- c is DETACHED (stop-gradient): the head cannot game the objective through
  the context statistics, and for the fixed per-image c the monotonicity
  certificate in z holds exactly (Lip bound uses the z input column only).
- Anti-collapse centering: `centering_penalty(corr)` penalizes the
  per-image MEAN correction, forcing the head to REDISTRIBUTE intensities
  (reshape the curve) instead of collapsing into a trivial per-image DC
  shift that games bias metrics.
"""

import torch
import torch.nn as nn

import config as cfg
from benchmark_data import BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD

# Standardized value of water (0 HU); the pixel domain is HU + 1024.
Z_WATER = (0.0 + cfg.HU_OFFSET - BENCHMARK_PIXEL_MEAN) / BENCHMARK_PIXEL_STD

# Same physical bins as hu_audit.py / hu_losses.py (kept local to avoid
# import coupling; do not change without changing them everywhere).
TISSUE_BINS_HU = (
    ("AirLung", -1024.0, -500.0),
    ("FatLow",   -500.0, -200.0),
    ("Soft",     -200.0,  200.0),
    ("Dense",     200.0,  600.0),
    ("Bone",      600.0, 1900.0),
)
SOFT_SIGMA_FRACTION = 0.25
CONTEXT_DIM = 2 + len(TISSUE_BINS_HU)   # mean, std, 5 bin occupancies


def _bins_std():
    """(center, sigma) of each tissue bin in standardized units."""
    out = []
    for _, lo, hi in TISSUE_BINS_HU:
        c_hu = 0.5 * (lo + hi)
        s_hu = SOFT_SIGMA_FRACTION * (hi - lo)
        out.append((
            (c_hu + cfg.HU_OFFSET - BENCHMARK_PIXEL_MEAN) / BENCHMARK_PIXEL_STD,
            s_hu / BENCHMARK_PIXEL_STD,
        ))
    return out


class CalibrationHead(nn.Module):
    """Arm D: bounded, certified-monotonic, identity-init 1D intensity map."""

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
        self.s = float(delta_hu) / BENCHMARK_PIXEL_STD
        self.fc1 = nn.Linear(1, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def config(self) -> dict:
        return {"type": "intensity", "hidden": self.hidden,
                "delta_hu": self.delta_hu, "kappa": self.kappa}

    def _lip_upper_bound(self) -> torch.Tensor:
        lip = None
        for fc in (self.fc1, self.fc2, self.fc3):
            n = torch.linalg.matrix_norm(fc.weight, ord=2)
            lip = n if lip is None else lip * n
        return lip

    def correction(self, z: torch.Tensor) -> torch.Tensor:
        h = z.reshape(-1, 1)
        h = torch.tanh(self.fc1(h))
        h = torch.tanh(self.fc2(h))
        g = self.fc3(h).reshape(z.shape)
        lip = self._lip_upper_bound()
        scale = torch.clamp(self.kappa / (self.s * lip + 1e-12), max=1.0)
        return self.s * torch.tanh(g * scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.correction(z)

    def water_anchor_penalty(self) -> torch.Tensor:
        z0 = torch.tensor([Z_WATER], device=self.fc1.weight.device,
                          dtype=self.fc1.weight.dtype)
        return (self.correction(z0) ** 2).sum()

    @torch.no_grad()
    def transfer_curve(self, hu_min: float = -1024.0, hu_max: float = 1900.0,
                       n: int = 1024):
        hu = torch.linspace(hu_min, hu_max, n, device=self.fc1.weight.device)
        z = (hu + cfg.HU_OFFSET - BENCHMARK_PIXEL_MEAN) / BENCHMARK_PIXEL_STD
        t = self.forward(z)
        hu_out = t * BENCHMARK_PIXEL_STD + BENCHMARK_PIXEL_MEAN - cfg.HU_OFFSET
        return hu, hu_out


class ContextCalibrationHead(nn.Module):
    """Arm E: the same constrained curve, conditioned on a detached
    per-image context vector (so chest-like and abdomen-like images can
    receive DIFFERENT transfer curves)."""

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
        self.s = float(delta_hu) / BENCHMARK_PIXEL_STD
        self.fc1 = nn.Linear(1 + CONTEXT_DIM, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.fc3.weight)   # identity init
        nn.init.zeros_(self.fc3.bias)
        self._bins = _bins_std()

    def config(self) -> dict:
        return {"type": "context", "hidden": self.hidden,
                "delta_hu": self.delta_hu, "kappa": self.kappa}

    def context(self, z: torch.Tensor) -> torch.Tensor:
        """(B, CONTEXT_DIM) detached per-image statistics of z."""
        b = z.shape[0]
        flat = z.detach().reshape(b, -1)
        feats = [flat.mean(dim=1), flat.std(dim=1)]
        for c, s in self._bins:
            feats.append(torch.exp(-0.5 * ((flat - c) / s) ** 2).mean(dim=1))
        return torch.stack(feats, dim=1)

    def _lip_z_upper_bound(self) -> torch.Tensor:
        """Certified Lip bound of g w.r.t. the z input ONLY (context is a
        per-image constant), so T(.|c) is monotone in z for every c."""
        lip = torch.linalg.matrix_norm(self.fc1.weight[:, :1], ord=2)
        lip = lip * torch.linalg.matrix_norm(self.fc2.weight, ord=2)
        lip = lip * torch.linalg.matrix_norm(self.fc3.weight, ord=2)
        return lip

    def correction(self, z: torch.Tensor,
                   context: torch.Tensor | None = None) -> torch.Tensor:
        b = z.shape[0]
        if context is None:
            context = self.context(z)
        flat = z.reshape(b, -1, 1)
        cexp = context.unsqueeze(1).expand(-1, flat.shape[1], -1)
        h = torch.cat([flat, cexp], dim=-1)
        h = torch.tanh(self.fc1(h))
        h = torch.tanh(self.fc2(h))
        g = self.fc3(h).reshape(z.shape)
        lip = self._lip_z_upper_bound()
        scale = torch.clamp(self.kappa / (self.s * lip + 1e-12), max=1.0)
        return self.s * torch.tanh(g * scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.correction(z)

    @staticmethod
    def centering_penalty(corr: torch.Tensor) -> torch.Tensor:
        """Anti-collapse: squared per-image MEAN correction. Keeps the head
        redistributive (curve reshaping), never a per-image DC shift."""
        b = corr.shape[0]
        return (corr.reshape(b, -1).mean(dim=1) ** 2).mean()

    def water_anchor_penalty(self, z_batch: torch.Tensor) -> torch.Tensor:
        """Mean squared correction at 0 HU under each image's context."""
        context = self.context(z_batch)
        z0 = torch.full((context.shape[0], 1), Z_WATER,
                        device=z_batch.device, dtype=z_batch.dtype)
        return (self.correction(z0, context=context) ** 2).mean()

    @torch.no_grad()
    def transfer_curve(self, context: torch.Tensor,
                       hu_min: float = -1024.0, hu_max: float = 1900.0,
                       n: int = 1024):
        """Per-context transfer curve in physical HU. `context` is one row
        (CONTEXT_DIM,) e.g. from self.context(z)[i]."""
        hu = torch.linspace(hu_min, hu_max, n, device=self.fc1.weight.device)
        z = ((hu + cfg.HU_OFFSET - BENCHMARK_PIXEL_MEAN)
             / BENCHMARK_PIXEL_STD).reshape(1, -1)
        corr = self.correction(z, context=context.reshape(1, -1))
        t = z + corr
        hu_out = (t * BENCHMARK_PIXEL_STD + BENCHMARK_PIXEL_MEAN
                  - cfg.HU_OFFSET).reshape(-1)
        return hu, hu_out


def save_head(head, path: str, extra: dict | None = None):
    payload = {"head_state_dict": head.state_dict(),
               "head_config": head.config()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_head(path: str, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    head_config = dict(payload["head_config"])
    head_type = head_config.pop("type", "intensity")
    cls = ContextCalibrationHead if head_type == "context" else CalibrationHead
    head = cls(**head_config).to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    return head
