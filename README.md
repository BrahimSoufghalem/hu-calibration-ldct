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
- [x] **Phase 2 — HU losses (arms B, C)** — naive HU L1 did not remove bias;
  explicit calibration loss improved chest bias but overshot in abdomen
- [x] **Phase 3 — Post-hoc calibration head (arm D)** — bounded and monotone,
  but one global curve remained anatomy-blind
- [x] **Phase 4 — Context-conditioned head (arm E)** — patch context remained
  blind; full-slice image-derived context restored anatomy-sensitive correction
- [x] **E-full-slice replication** — three head seeds on RED-CNN and ResNet
- [ ] Context-aware quality evaluation and isolated loss ablation
- [ ] Full combination (arm F) and matched-weight control (arm G)

Staged philosophy: **Audit → Loss → Post-hoc Head → Context → Adaptive only if
needed.** Every arm must be justified by the results of the previous one.

### Current findings

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
- Patch-derived context heads converged to the analytical ceiling of an
  anatomy-blind correction. On Bone, the predicted residuals were about
  −24.32 / +24.32 HU (chest / abdomen), versus −24.97 / +23.73 measured.
- Computing the same seven context statistics from the uncropped low-dose
  slice broke this ceiling. Across three head seeds, Bone SoftBias changed by
  +43.44 HU for RED-CNN chest and +43.75 HU for ResNet chest. Non-circular
  HardBias Bone improved consistently as well.
- The gain has a measured cost: `ThrDisagree_130HU_pct` increased for both
  anatomies and both architectures. Image-quality preservation is therefore
  not assumed; it must be measured explicitly against the frozen trunk.

## Experiment arms

| Arm | Configuration |
| --- | ------------- |
| A | Baseline (MSE) |
| B | + `L_HU` (control: is plain HU supervision enough?) |
| C | + `L_HU-Cal` = soft-bin bias + slope/intercept penalty (α→1, β→0) |
| D | A + post-hoc calibration head `T(x) = x + δ·tanh(g(x))`, frozen trunk |
| E | Post-hoc context-conditioned head; patch and full-slice variants |
| F | Full: C + head |
| G | Matched-weight control |

**Trunks:** RED-CNN and ResNet. E-full-slice uses three repeated head fits per
architecture over one fixed trunk checkpoint; these are head seeds, not
independent trunk replications.
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
| `evaluate_image.py` | Full-resolution PSNR/SSIM/RMSE_HU/VIF; optional paired Cycle-00 vs selected-head evaluation |
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

# Post-hoc E-full-slice head:
HU_RANGE_PRESET=benchmark python train_head.py --arch redcnn \
    --data-dir dataset --split 100p --head-type context \
    --context-full-slice --output-root runs_armE_full_slice

# Threshold-agnostic no-harm v2 exploratory follow-up:
HU_RANGE_PRESET=benchmark python train_head.py --arch redcnn \
    --data-dir dataset --split 100p --head-type context \
    --context-full-slice --output-root runs_armE_no_harm_v2 \
    --threshold-no-harm-lambda 5.0 --threshold-samples 32 \
    --threshold-pixel-samples 131072 \
    --threshold-min-hu -1000 --threshold-max-hu 1500 \
    --threshold-temperature-hu 5 --threshold-worst-weight 1 \
    --threshold-cvar-fraction 0.2 --threshold-density-fraction 0.5 \
    --curve-identity-lambda 0.005 --curve-slope-lambda 0.001

# Tissue-resolved audit of the frozen trunk and selected head:
HU_RANGE_PRESET=benchmark python hu_audit.py --test-dir test \
    --runs-root runs --heads-root runs_armE_full_slice \
    --archs redcnn --include-input --output hu_audit_e_full_slice

# Audit a dense set of held-out thresholds after training:
HU_RANGE_PRESET=benchmark python hu_audit.py --test-dir test \
    --runs-root runs --heads-root runs_armE_no_harm_v2 --archs redcnn \
    --threshold-grid=-1000,1500,5 --output hu_audit_e_no_harm_v2
```

The no-harm v2 configuration still trains only the post-hoc head; the selected
trunk checkpoint remains frozen. It rectifies regression separately for every
image and threshold, so improvement in one patient cannot hide harm in another.
Half the thresholds cover the HU range uniformly and half follow the observed
target-HU density. CVaR emphasizes the worst 20% of thresholds per image without
using one noisy maximum. Validation uses deterministic grid/quantile thresholds.
The weaker curve penalties discourage broad corrections without suppressing the
anatomy-sensitive Bone correction as strongly as the first no-harm run. These
remain starting values for an isolated ablation, not validated hyperparameters.
Run RED-CNN and ResNet independently, then audit the dense grid and all existing
quality/bias endpoints before accepting the configuration. New options do not
change historical commands unless the no-harm weight is enabled.

`hu_audit.py` reports, per patient and per tissue bin (AirLung / FatLow /
Soft / Dense / Bone, Gaussian soft membership):

- soft-bin and hard-bin HU bias,
- the exact conditional decomposition `MSE_k = b_k² + Var(error | bin k)`,
- the calibration line (slope **α**, intercept **β**; ideal 1 / 0),
- threshold-crossing sensitivity (fraction of pixels flipping across fixed HU
  thresholds — a sensitivity analysis, **not** a clinical endpoint claim),
- identity-curve data (`HU_pred` vs `HU_ref`) exported to CSV for plotting.

## Context-aware image-quality evaluation

`evaluate_image.py` can evaluate a post-hoc calibration head and its frozen
trunk in the same full-resolution test pass. This closes the previous gap in
which generic validation could not pass explicit full-slice context.

```bash
HU_RANGE_PRESET=benchmark python evaluate_image.py \
    --test-dir test \
    --runs-root runs \
    --heads-root runs_armE_full_slice \
    --archs redcnn \
    --split 100p \
    --output eval_armE_full_slice
```

The evaluator:

- reports clinically windowed PSNR/SSIM, physical-HU RMSE, and VIF per patient;
- computes full-slice context from the uncropped standardized low-dose input,
  matching training and `hu_audit.py`;
- supports intensity, inferred-context, full-slice-context, and oracle heads;
- writes the frozen trunk as `Cycle 00` and the selected head separately;
- reports deltas against both the raw LDCT input and the frozen trunk;
- validates architecture, normalization constants, data split, and exact trunk
  checkpoint provenance before accepting a head;
- rejects joint heads because they do not share the post-hoc Cycle-00 baseline;
- fails if `--heads-root` is requested but a paired head checkpoint is missing.

Outputs are:

- `<arch>_results.csv`: frozen-trunk / Cycle-00 patient metrics;
- `<arch>_head_results.csv`: selected-head metrics and deltas versus the trunk;
- `comparison.csv`: all evaluated stages and architectures.

For `Delta_vs_Trunk_RMSE_HU`, a negative value is an improvement. For PSNR,
SSIM, and VIF, a positive value is an improvement. Quality results must not be
claimed until this evaluator has been run on the actual test data and selected
checkpoints.

## Protocol

Matched training budget across all arms (identical iterations, optimizer,
data pipeline and seeds), Mayo *LDCT-and-Projection-data* (explicit
100-patient split in `config.py`; 20-patient pilot split available),
per-anatomy reporting (chest / abdomen), evaluation in physical HU.

## License

MIT
