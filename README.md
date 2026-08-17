# Quantitative HU Fidelity in Deep LDCT Denoising

**A Tissue-Resolved Audit and a Constrained Plug-in Calibration Head**

Deep LDCT denoisers are usually judged by PSNR/SSIM, which are blind to
systematic, HU-dependent quantitative errors. This project (1) **audits** how
the HU error of standard denoisers is structured across tissue types, and
(2) introduces **HU-calibration losses** and a **constrained plug-in
calibration head** to reduce it — evaluated entirely in physical Hounsfield
Units.

## Status

- [x] Repo bootstrap — shared infrastructure imported from
  [physics-faithful-ldct-denoising](https://github.com/BrahimSoufghalem/physics-faithful-ldct-denoising)
- [x] **Phase 1 — Audit** — completed on MSE baselines (RED-CNN + ResNet,
  10 test patients)
- [ ] **Phase 2 — HU losses (arms B, C)** ← current step (`hu_losses.py`, `train.py`)
- [ ] Phase 3 — Post-hoc calibration head (arm D)
- [ ] Phase 4 — Context-conditioned head (E), full (F), matched-weight control (G)

Staged philosophy: **Audit → Loss → Post-hoc Head → Context → Adaptive only if
needed.** Every arm must be justified by the results of the previous one.

### Phase-1 findings (why arms B–D are justified)

- The HU distortion is **architecture-independent**: RED-CNN and ResNet show
  nearly identical identity-deviation curves (bias-sign agreement in 43/50
  patient×bin cells; 10/10 in FatLow/Dense/Bone).
- It is **chest-dominated and zig-zag shaped**: soft-bin bias ≈ −17 HU
  (AirLung), **+67** (FatLow), −3 (Soft), **−52** (Dense), **−54** (Bone)
  for RED-CNN; ResNet is slightly worse. Consistent with over-smoothing
  pulling everything toward soft tissue.
- The **global calibration line is blind**: α ≈ 1.01, β ≈ −4 HU while
  per-bin biases reach ±67 HU (opposite signs cancel in the regression)
  → per-bin `L_SoftBias` is the primary term of `L_HU-Cal`.
- Bias explains up to **27.6 % of per-bin MSE** (chest FatLow) → a
  calibration head has a real theoretical ceiling; variance dominates
  elsewhere → training-time losses (arms B/C) target the remainder.

## Experiment arms

| Arm | Configuration |
| --- | ------------- |
| A | Baseline (MSE) |
| B | + `L_HU` (control: is plain HU supervision enough?) |
| C | + `L_HU-Cal` = soft-bin bias + slope/intercept penalty (α→1, β→0) |
| D | A + post-hoc calibration head `T(x) = x + δ·tanh(g(x))`, frozen trunk |
| E | Joint context-conditioned head + anti-collapse constraint |
| F | Full: C + head |
| G | Matched-weight control |

**Trunks:** RED-CNN and ResNet (both mandatory for every arm). Seeds 0/1/2.
Head guarantees by construction: bounded correction, monotonicity,
near-identity init; water anchor `T(0) ≈ 0` is a *tested* constraint, not an
assumption.

## Repository layout

| File | Role |
| ---- | ---- |
| `config.py` | HU convention (ldct-benchmark exact), patient splits, constants |
| `download.py` | Mayo *LDCT-and-Projection-data* downloader (NBIA) |
| `benchmark_data.py` | Benchmark-aligned MONAI data pipeline |
| `models/` | RED-CNN, ResNet trunks (exact ldct-benchmark copies) |
| `metrics.py` | RMSE (HU), clinically-windowed PSNR/SSIM, VIF |
| `hu_losses.py` | **Arms B/C** — `L_HU` control and `L_HU-Cal` (soft-bin bias + α/β) |
| `train.py` | Matched-budget trainer (arms A, B, C) |
| `evaluate_image.py` | Standard image-quality evaluation |
| `hu_audit.py` | **Phase 1** — tissue-resolved HU audit |
| `twenty_patient_split.py` | Balanced 20-patient pilot split |

## Quickstart

```bash
pip install -r requirements.txt
python download.py   # fetch the Mayo LDCT patients from NBIA

# Phase 1 — audit any checkpoint (and the raw LDCT input):
HU_RANGE_PRESET=benchmark python hu_audit.py \
    --test-dir test --runs-root runs \
    --archs redcnn,resnet --include-input

# Phase 2 — train the arms (matched 30k-iteration budget):
# Arm A (baseline):
HU_RANGE_PRESET=benchmark python train.py --arch redcnn \
    --data-dir dataset --split 100p --select-by bench_ssim
# Arm B (+L_HU control):
#   ... --hu-weight 0.2 --output-root runs_armB
# Arm C (+L_HU-Cal):
#   ... --hucal-weight 0.2 --output-root runs_armC
```

`hu_audit.py` reports, per patient and per tissue bin (AirLung / FatLow /
Soft / Dense / Bone, Gaussian soft membership):

- soft-bin and hard-bin HU bias,
- the exact conditional decomposition `MSE_k = b_k² + Var(error | bin k)`,
- the calibration line (slope **α**, intercept **β**; ideal 1 / 0),
- threshold-crossing sensitivity (fraction of pixels flipping across fixed HU
  thresholds — a sensitivity analysis, **not** a clinical endpoint claim),
- identity-curve data (`HU_pred` vs `HU_ref`) exported to CSV for plotting.

## Protocol

Matched training budget across all arms (identical iterations, optimizer,
data pipeline and seeds), Mayo *LDCT-and-Projection-data* (explicit
100-patient split in `config.py`; 20-patient pilot split available),
per-anatomy reporting (chest / abdomen), evaluation in physical HU.

## License

MIT
