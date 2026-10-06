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
