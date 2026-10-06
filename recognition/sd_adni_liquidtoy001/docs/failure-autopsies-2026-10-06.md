# Qualitative failure autopsies and risk probes — 6 October 2026

Private panels: outputs/calibration-20261006/F1-background.png through F5-repeat.png.
Each panel shows generated / nearest full-training / nearest full-validation image.
Neighbors were selected at grayscale64 by pixel RMSE; displayed at256. Identities
remain in ignored private-cases.json. These are research observations, not medical
interpretations. Three observed failure modes and two unconfirmed risk probes:

| Case | Exact sample | Observation | Hypothesis, alternatives and next control |
| --- | --- | --- | --- |
| F1 background | adapted AD37220, CFG7.5 | Broad gray background absent from displayed real neighbors; RMSE0.3576/0.3590 | Background mismatch strongly contributes to full-frame error. Compare background/foreground error separately; color prior or sampling instability are unconfirmed causes. Do not hide it by cropping before reporting. |
| F2 color | adapted NC37217, CFG7.5 | Cyan patch in nominally grayscale MRI output; RMSE0.0823/0.0665 | Grayscale metric discards this defect. Measure RGB channel disagreement over all64 outputs and compare model-wide; natural-image prior is a hypothesis, not a proven cause. |
| F3 nearest risk probe | adapted NC37215 | Minimum training NN distance0.0702; displayed neighbor is not pixel-identical and fine structures differ | Proximity alone is not memorization. Real validation-to-training distances span0.0484–0.0982, but this calibration cannot rule out partial memorization. Feature/multiscale neighbor analysis remains pending. |
| F4 blur | baseline AD37200 | Fine boundaries are visibly smoothed, yet RMSE0.0378/0.0382 is very low | Low resolution, architecture capacity and reconstruction objective can each contribute. Their causal contributions are not isolated. Compare at matched resolution with edge/foreground metrics and original-scale visual review. |
| F5 repetition probe | adapted AD37201 and full fixed grid | Similar gross shape across seeds; anatomy/domain homogeneity can also explain it | Real matched pairwise mean0.1524–0.1612 versus adapted0.1821 does not support calling collapse based only on this metric. Background variation can inflate it. Need foreground/feature-space coverage and matched slice-position controls. |

Calibration job635718 completed43s,exit0:0. Training queries exclude every image
from their own patient, not only the exact self image.102 patients per reference
subset,33AD/69NC. Real validation->full train mean0.07310; train->other-patient full
train mean0.07630. Real validation->matched train mean0.09185. Adapted->full train
mean0.11219; baseline0.05652. No significance test or privacy guarantee is claimed.
Only three cases are confirmed visual failures; F3/F5 must not be described as
confirmed failures merely to fill the report's3–5-case quota. Student review and
further controlled validation remain pending.

## 2026-10-07 — Appearance reference calibration

CPU job636215 COMPLETED in7s, exit0:0. All192 generated images and102 validation
images passed recorded SHA256 checks; downloaded report script hash matches local
source. No test images read. Local refactoring reproduced all nine earlier
aggregate means within1e-12. Ruff and shell syntax checks passed.

| Group | n | Mean color range | Mean outer-border intensity | Mean edge difference |
| --- | ---: | ---: | ---: | ---: |
| Original SD | 64 | 0.344538 | 0.439551 | 0.060737 |
| Adapted SD | 64 | 0.020839 | 0.059221 | 0.022961 |
| Simple VAE | 64 | 0 | 0.000077 | 0.007563 |
| Real validation | 102 | 0 | 0.000000100 | 0.017551 |

AD and NC separately show the same edge ordering. Validation edge means are
0.017292/0.017674; VAE0.007623/0.007502; adapted0.023217/0.022704.
The results support the smoothing observation, not anatomical-quality rankings.
Native128 vs256, slice position, model architecture and losses remain confounded.
Single-channel real/VAE outputs are grayscale by construction, not a learned win.
Border is a spatial proxy, affected by padding; no segmentation was performed.

F1 border0.411542 versus real-reference maximum0.000004085 corroborates the
background mismatch. F2 color0.021035 confirms non-grayscale output, but does not
localize the cyan patch or establish its cause. F4 edge0.005648 is below the real
reference range0.010177-0.025855, consistent with smoothing; not a diagnostic
threshold. F3/F5 remain unconfirmed risk probes. No retraining decision yet.

Next controlled step: compare the existing fixed-seed CFG/resolution probes with
these same descriptive measures, then choose any new sampling ablation. Learned
features, foreground segmentation, interpolation and final test evaluation remain
unperformed. See appearance-protocol.md and appearance-result-2026-10-07.json.

## 2026-10-07 — Retrospective sampling appearance comparison

Measured all16 images from job632530. SHA256 verification and complete factorial
coverage passed. Each within-resolution CFG pair shares its initial-noise hash.
This is the original unaligned scheduler-precision run; do not merge it with the
later aligned cohort as though precision were controlled across runs.

| Resolution | CFG | Mean color range | Mean border | Mean edge |
| --- | ---: | ---: | ---: | ---: |
| 256 | 1 | 0.010626 | 0.178816 | 0.030590 |
| 256 | 7.5 | 0.016531 | 0.004558 | 0.022557 |
| 512 | 1 | 0.007888 | 0.398652 | 0.041371 |
| 512 | 7.5 | 0.009361 | 0.051831 | 0.021844 |

All eight CFG pairs decrease border and edge measures. Seven increase color
range. This refines the visual statement that higher guidance reduces artifacts:
it improves the border proxy here but does not consistently improve grayscale
fidelity. Edge decrease is not itself anatomical improvement. There are only two
independent seeds; AD/NC share noise, so these are not eight independent replicates.
Across resolutions noise tensors differ in shape; equal seed is not equal latent.
512 does not eliminate border mismatch in these samples. No optimal CFG, causal
training defect, or need for retraining is established.

Next bounded experiment should hold aligned precision, checkpoint, DDIM30 and
256 resolution fixed, use all32 existing cohort seeds and both labels, and vary
CFG only (e.g.3,5,7.5). Reuse7.5 outputs only after confirming identical pipeline
and noise contracts. Retain every image and report paired changes, including
known failure seeds37220/37217. This is a proposed experiment, not submitted.
No test images used; no new GPU job launched in this analysis.
