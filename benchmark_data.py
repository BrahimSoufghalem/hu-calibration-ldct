"""Benchmark-aligned 2D data pipeline.

Reproduces ldct-benchmark's global mean/std convention in the stored CT pixel
domain: pixel = HU + 1024. Training batches are patient-balanced with
replacement sampling; validation is deterministic and visits every slice.
"""

import os
import random
from collections import Counter
from glob import glob

import torch
from monai.data import CacheDataset, Dataset, DataLoader, PydicomReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandSpatialCropSamplesd,
    ResizeWithPadOrCropd, ToTensord,
)
from monai.utils import set_determinism
from torch.utils.data import WeightedRandomSampler

import config as cfg
from utils import sort_by_instance_number


# Exact constants in eeulig/ldct-benchmark ldctbench/data/info.yml.
BENCHMARK_PIXEL_MEAN = 481.45419786099086
BENCHMARK_PIXEL_STD = 502.18507379395044
BENCHMARK_PIXEL_OFFSET = 1024.0
_CONTEXT_BINS_HU = (
    (-1024.0, -500.0), (-500.0, -200.0), (-200.0, 200.0),
    (200.0, 600.0), (600.0, 1900.0),
)


def standardize_hu(hu):
    """Physical HU -> benchmark-standardized stored-pixel values, without clipping."""
    if not isinstance(hu, torch.Tensor):
        hu = torch.as_tensor(hu, dtype=torch.float32)
    pixel = hu.float() + BENCHMARK_PIXEL_OFFSET
    return (pixel - BENCHMARK_PIXEL_MEAN) / BENCHMARK_PIXEL_STD


def denormalize_to_pixel(z):
    """Benchmark-standardized tensor -> physical non-negative pixel/HU+1024 domain."""
    return z * BENCHMARK_PIXEL_STD + BENCHMARK_PIXEL_MEAN


class BenchmarkMeanStdd:
    """MONAI dictionary transform using the exact benchmark global statistics."""

    def __init__(self, keys=("image", "label")):
        self.keys = tuple(keys)

    def __call__(self, data):
        for key in self.keys:
            data[key] = standardize_hu(data[key])
        return data


class FullSliceContextd:
    """Store detached statistics from the uncropped standardized low-dose slice.

    This runs before the random crop, so every patch from a slice receives the
    same seven-element context without retaining an additional 512x512 tensor
    in CacheDataset memory.
    """

    def __call__(self, data):
        flat = data["image"].reshape(-1)
        features = [flat.mean(), flat.std()]
        for lo_hu, hi_hu in _CONTEXT_BINS_HU:
            center_hu = 0.5 * (lo_hu + hi_hu)
            sigma_hu = 0.25 * (hi_hu - lo_hu)
            center = standardize_hu(torch.tensor(center_hu))
            sigma = sigma_hu / BENCHMARK_PIXEL_STD
            features.append(torch.exp(-0.5 * ((flat - center) / sigma) ** 2).mean())
        data["full_context"] = torch.stack(features).detach()
        return data


def _benchmark_reader():
    """Match pydicom.pixel_array orientation used by ldct-benchmark.

    MONAI's PydicomReader applies RescaleSlope/Intercept correctly, but its
    default ``swap_ij=True`` transposes the spatial axes. The benchmark and our
    full-resolution evaluator use pydicom arrays without that swap. Keeping the
    default here would train on transposed anatomy and test on unswapped anatomy.
    """
    return PydicomReader(swap_ij=False)


def _train_transform(patch_size, include_full_slice_context=False):
    keys = ["image", "label"]
    transforms = [
        LoadImaged(keys=keys, reader=_benchmark_reader()),
        EnsureChannelFirstd(keys=keys),
        BenchmarkMeanStdd(keys=keys),
    ]
    if include_full_slice_context:
        transforms.append(FullSliceContextd())
    transforms.extend([
        RandSpatialCropSamplesd(
            keys=["image", "label"], roi_size=(patch_size, patch_size),
            num_samples=1,
        ),
        ToTensord(keys=keys + (["full_context"] if include_full_slice_context else [])),
    ])
    return Compose(transforms)


def _val_transform(patch_size, include_full_slice_context=False):
    keys = ["image", "label"]
    transforms = [
        LoadImaged(keys=keys, reader=_benchmark_reader()),
        EnsureChannelFirstd(keys=keys),
        BenchmarkMeanStdd(keys=keys),
    ]
    if include_full_slice_context:
        transforms.append(FullSliceContextd())
    transforms.extend([
        ResizeWithPadOrCropd(
            keys=["image", "label"], spatial_size=(patch_size, patch_size),
        ),
        ToTensord(keys=keys + (["full_context"] if include_full_slice_context else [])),
    ])
    return Compose(transforms)


def _limit_patients(patients, max_n, label):
    """Deterministic chest/abdomen-balanced subset of a sorted patient list.

    PILOT MODE helper: alternates chest ('C*') and abdomen ('L*') patients in
    sorted order until ``max_n`` are picked, so a small subset keeps both body
    regions represented. Deterministic: same input list -> same subset, which
    keeps different pilot configs comparable.
    """
    if max_n is None or int(max_n) <= 0 or int(max_n) >= len(patients):
        return patients
    max_n = int(max_n)
    chest = [p for p in patients if p.lower().startswith("c")]
    abd = [p for p in patients if not p.lower().startswith("c")]
    picked = []
    i = 0
    while len(picked) < max_n and (i < len(chest) or i < len(abd)):
        if i < len(chest):
            picked.append(chest[i])
        if len(picked) < max_n and i < len(abd):
            picked.append(abd[i])
        i += 1
    picked = sorted(picked)
    print(f"PILOT: limiting {label} patients to {len(picked)}/{len(patients)}: "
          f"{picked}")
    return picked


def collect_files(patient_list, in_dir=cfg.DATA_DIR):
    """One record per slice: paired Low_Dose / Full_Dose DICOM paths."""
    files = []
    for patient in patient_list:
        low_dir = os.path.join(in_dir, patient, "Low_Dose")
        full_dir = os.path.join(in_dir, patient, "Full_Dose")
        low_imgs = sort_by_instance_number(glob(os.path.join(low_dir, "*.dcm")))
        full_imgs = sort_by_instance_number(glob(os.path.join(full_dir, "*.dcm")))
        assert len(low_imgs) == len(full_imgs), \
            f"Mismatch for patient {patient}: {len(low_imgs)} vs {len(full_imgs)}"
        for i in range(len(low_imgs)):
            files.append({
                "image": low_imgs[i],
                "label": full_imgs[i],
                "patient": patient,
                "body_type": "Chest" if patient.lower().startswith("c") else "Abdomen",
            })
    return files


def prepare_benchmark_data(
    in_dir=cfg.DATA_DIR,
    train_patch_size=64,
    val_patch_size=128,
    train_batch_size=64,
    val_batch_size=64,
    iterations_before_val=1000,
    num_workers=cfg.NUM_WORKERS,
    cache=True,
    cache_rate=1.0,
    max_train_patients=None,
    max_val_patients=None,
    include_full_slice_context=False,
):
    """Create patient-balanced train batches and deterministic validation crops.

    Each training cycle contains exactly ``iterations_before_val`` batches, as
    in ldct-benchmark. Every patient has equal total sampling mass regardless of
    slice count. Validation remains deterministic and visits every validation
    slice.

    ``max_train_patients`` / ``max_val_patients`` enable PILOT MODE: a
    deterministic chest/abdomen-balanced subset for fast config screening.
    Pilot results rank configurations; they are NOT reportable numbers.
    """
    set_determinism(seed=cfg.SEED)
    random.seed(cfg.SEED)

    all_patients = sorted([
        p for p in os.listdir(in_dir)
        if os.path.isdir(os.path.join(in_dir, p))
    ])
    train_patients = [p for p in all_patients if p in cfg.EXPECTED_TRAIN]
    val_patients = [p for p in all_patients if p in cfg.EXPECTED_VAL]
    if not train_patients or not val_patients:
        raise RuntimeError(
            "The benchmark-aligned pipeline requires the explicit patient "
            "split from config.py; train or validation patients were not found."
        )
    train_patients = _limit_patients(train_patients, max_train_patients, "train")
    val_patients = _limit_patients(val_patients, max_val_patients, "val")

    train_files = collect_files(train_patients, in_dir)
    val_files = collect_files(val_patients, in_dir)
    counts = Counter(item["patient"] for item in train_files)
    weights = torch.tensor(
        [1.0 / counts[item["patient"]] for item in train_files],
        dtype=torch.double,
    )
    samples_per_cycle = int(train_batch_size) * int(iterations_before_val)
    sampler_generator = torch.Generator().manual_seed(cfg.SEED)
    sampler = WeightedRandomSampler(
        weights, num_samples=samples_per_cycle, replacement=True,
        generator=sampler_generator,
    )

    train_transform = _train_transform(int(train_patch_size), include_full_slice_context)
    val_transform = _val_transform(int(val_patch_size), include_full_slice_context)
    cache_rate = float(min(max(cache_rate, 0.0), 1.0))
    if cache and cache_rate > 0:
        train_ds = CacheDataset(
            train_files, train_transform, cache_rate=cache_rate,
        )
        val_ds = CacheDataset(
            val_files, val_transform, cache_rate=cache_rate,
        )
    else:
        train_ds = Dataset(train_files, train_transform)
        val_ds = Dataset(val_files, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=num_workers > 0,
    )

    chest_train = sum(p.lower().startswith("c") for p in train_patients)
    chest_val = sum(p.lower().startswith("c") for p in val_patients)
    print("\nBenchmark-aligned data:")
    print(f"Train patients : {len(train_patients)} ({chest_train} chest, {len(train_patients)-chest_train} abdomen)")
    print(f"Val patients   : {len(val_patients)} ({chest_val} chest, {len(val_patients)-chest_val} abdomen)")
    print(f"Train slices   : {len(train_files)} | patient-balanced replacement sampling")
    print(f"Val slices     : {len(val_files)} | deterministic center crop")
    print(f"Patches        : train {train_patch_size} | val {val_patch_size}")
    if include_full_slice_context:
        print("Context        : 7 statistics from full low-dose slice before crop")
    print(f"Train cycle    : {iterations_before_val} iterations x batch {train_batch_size}")
    print("DICOM orientation: PydicomReader(swap_ij=False), aligned with benchmark/evaluation")
    print(
        "Standardization : (HU + 1024 - "
        f"{BENCHMARK_PIXEL_MEAN:.12f}) / {BENCHMARK_PIXEL_STD:.12f}"
    )
    return train_loader, val_loader
