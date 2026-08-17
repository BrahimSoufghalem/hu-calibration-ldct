"""Central configuration: paths, HU convention, evaluation constants and
patient splits.

The HU convention reproduces eeulig/ldct-benchmark EXACTLY:
A_MAX + HU_OFFSET = 2924, the DATA_RANGE constant in
ldctbench/evaluate/utils.py. All training and evaluation scripts in this
repository assume this single convention.
"""

import os

# ═══════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════
DATA_DIR = "dataset"            # train + validation patients
TEST_DIR = "test"               # held-out test patients
EVAL_OUTPUT_DIR = "eval_results"
OUTPUT_ROOT = "runs"

# ═══════════════════════════════════════════
# REPRODUCIBILITY / DATA LOADING
# ═══════════════════════════════════════════
# SEED may be overridden at runtime (train.py --seed) BEFORE the data
# pipeline is constructed; benchmark_data.py reads it at call time.
SEED = 0
NUM_WORKERS = 8

# ═══════════════════════════════════════════
# HU CONVENTION (ldct-benchmark)
# ═══════════════════════════════════════════
# Kept as an environment variable for protocol transparency: run commands
# state the convention explicitly, so every run log records it.
HU_RANGE_PRESET = os.environ.get("HU_RANGE_PRESET", "benchmark").strip().lower()
if HU_RANGE_PRESET != "benchmark":
    raise ValueError(
        "This repository supports only HU_RANGE_PRESET=benchmark "
        f"(the ldct-benchmark convention), got '{HU_RANGE_PRESET}'."
    )

A_MIN = -1024.0
A_MAX = 1900.0                  # A_MAX + HU_OFFSET == 2924 == ldctbench DATA_RANGE

HU_OFFSET = 1024.0              # HU -> non-negative stored-pixel domain
HU_OFFSET_MAX = A_MAX + HU_OFFSET
EVAL_DATA_RANGE = HU_OFFSET_MAX  # 2924

# Clinical diagnostic windows as (center, width) in the HU+1024 domain.
# These match CW["C"] and CW["L"] in ldctbench/evaluate/utils.py exactly.
CLINICAL_WINDOWS = {
    "Chest": (HU_OFFSET - 600, 1500),   # lung window:        C=-600 HU, W=1500 HU
    "Abdomen": (HU_OFFSET + 50, 400),   # soft-tissue window: C=  50 HU, W= 400 HU
}

# ═══════════════════════════════════════════
# EXPLICIT PATIENT SPLITS (100 patients, Mayo LDCT-and-Projection-data)
# ═══════════════════════════════════════════
EXPECTED_TEST = {
    'C121', 'C249', 'C170', 'C135', 'C280', 'L241', 'L107', 'L006', 'L221', 'L220'
}

EXPECTED_VAL = {
    'C202', 'C219', 'C227', 'C258', 'C067', 'C295', 'C190', 'C232', 'C052', 'C107',
    'L033', 'L187', 'L123', 'L058', 'L212', 'L077', 'L179', 'L014', 'L186', 'L193'
}

EXPECTED_TRAIN = {
    'C095', 'C261', 'C296', 'C218', 'C224', 'C267', 'C099', 'C030', 'C241', 'C162',
    'C268', 'C128', 'C252', 'C234', 'C130', 'C246', 'C124', 'C077', 'C002', 'C021',
    'C203', 'C111', 'C179', 'C012', 'C081', 'C004', 'C120', 'C193', 'C166', 'C257',
    'C160', 'C016', 'C027', 'C050', 'C158', 'L081', 'L248', 'L203', 'L219', 'L210',
    'L277', 'L057', 'L229', 'L131', 'L114', 'L004', 'L237', 'L148', 'L145', 'L116',
    'L150', 'L110', 'L232', 'L134', 'L056', 'L075', 'L209', 'L019', 'L064', 'L299',
    'L160', 'L049', 'L072', 'L071', 'L273', 'L175', 'L178', 'L125', 'L266', 'L170'
}

# ═══════════════════════════════════════════
# DOWNLOADER CONFIG (download.py)
# ═══════════════════════════════════════════
DOWNLOAD_WORKERS = 6
COLLECTION = "LDCT-and-projection-data"
DOWNLOAD_TIMEOUT = 300
CHUNK_SIZE = 1 * 1024 * 1024   # 1 MB
NBIA_API_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
