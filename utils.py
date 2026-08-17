"""Shared utilities: reproducibility, device selection, DICOM I/O and
checkpoint/state-dict helpers."""

import os
import random

import numpy as np
import pydicom
import torch
import torch.nn as nn

import config as cfg


# ═══════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════
def setup_reproducibility(seed=None, deterministic=False):
    """Seed every RNG used by the project.

    ``deterministic=True`` also forces deterministic cuDNN kernels, which is
    slower but makes runs bit-for-bit reproducible.
    """
    seed = cfg.SEED if seed is None else int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    return seed


def get_device():
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════
# DICOM HELPERS
# ═══════════════════════════════════════════
def sort_by_instance_number(file_paths):
    """Sort DICOM paths by InstanceNumber (falls back to the filename)."""
    def key(path):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0
    return sorted(file_paths, key=lambda p: (key(p), os.path.basename(p)))


def load_dicom_tensor(path):
    """Read one DICOM file and return a float32 tensor in physical HU."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    return torch.from_numpy(arr)


# ═══════════════════════════════════════════
# STATE-DICT HELPERS (DataParallel-safe)
# ═══════════════════════════════════════════
def unwrap_model(model):
    """Return the underlying module for DataParallel / DDP wrappers."""
    return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model


def get_state_dict(model):
    """Always return an UNWRAPPED state dict (no `module.` prefixes)."""
    return unwrap_model(model).state_dict()


def load_state_into(model, state, strict=True):
    """Load a state dict saved either with or without a `module.` prefix."""
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    return unwrap_model(model).load_state_dict(cleaned, strict=strict)
