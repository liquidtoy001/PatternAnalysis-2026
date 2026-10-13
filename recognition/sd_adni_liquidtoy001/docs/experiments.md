# Experiment Ledger

No model experiment has completed yet.

Pre-experiment audit: [Rangpur/ADNI, 2026-10-02](rangpur-audit-2026-10-02.md), CPU job 620988, completed with exit 0:0. This is a dataset/resource inspection, not a model experiment. The report also records unsuccessful preliminary shell probes.

Data preparation: [status](data-preparation-status.md). CPU job 623310 completed with exit 0:0 in 40 seconds. Full decode, source coverage, patient separation, deterministic allocation, sampled loader checks and negative controls passed. Aggregate split counts are recorded in the linked report.

For each real run, record:

- Run ID, purpose/hypothesis, and date/time with timezone.
- Git commit and any uncommitted changes; exact command and configuration.
- Dataset/split manifest version and patient separation checks.
- Model identifier/revision, trainable components, seed, and preprocessing.
- Environment versions, node/GPU, job ID, allocated host resources.
- Training steps, batch/accumulation settings, precision, and evaluation protocol.
- Measured loss/metrics, GPU/host memory, timing, and checkpoint size.
- Exit status, artifact paths, recovery verification, failures, and interpretation.

Keep failed and inconclusive runs. Distinguish measurements from estimates and proposed experiments. Store bulk logs and weights outside Git; reference them with stable run identifiers and checksums where appropriate.

## VAE diagnostic queued — 2 October 2026

- Protocol: [pre-experiment design](vae-protocol.md).
- CPU environment/download job **623826**: last checked PENDING (Resources).
- GPU reconstruction job **623828**: submitted with `afterok:623826` and
  `--kill-on-invalid-dep=yes`; ten-minute limit, one GPU, two CPUs.
- If setup fails, reconstruction must not run. Neither job trains a model.
- Local syntax compilation and Ruff lint/format checks passed. This does not
  establish package compatibility, download success or GPU acceptance.
- Source: `requirements-vae.txt`, `tools/download_vae.py`, `tools/check_vae.py`,
  and their two Slurm launchers. Uses the existing accepted private manifest.
- Remote project: `~/comp3710-sd-audit/project`.
- Setup logs: `logs/vae_setup_623826.{out,err}`; full dependency snapshot:
  `logs/vae_environment.txt`; pinned model hashes: `models/sd15-vae/provenance.json`.
- Reconstruction logs: `logs/vae_check_623828.{out,err}`; completion record:
  `outputs/vae-623828/results.json`; triptychs in the same ignored directory.
- Retrieve both exit statuses and inspect numerical results and images before
  reporting success. No reconstruction metric is claimed while queued.

## Dependency failure and retry — 3 October 2026

Job 623826 FAILED with exit 1:0 after 10 seconds. Pip rejected the requested
huggingface-hub 2.1.1 because diffusers 0.40.0 requires >=1.23.0,<2.0.
The initial pins were incompatible; no VAE results were produced. Dependent GPU
job 623828 was CANCELLED without running, as intended by the dependency guard.

Verified the version constraint from PyPI package metadata and changed only
huggingface-hub to 1.33.0. CPU setup retry **625832** and dependent GPU diagnostic
**625833** were submitted. Last observed: setup pending resources, diagnostic
waiting on successful setup. Real environment resolution and model execution
remain unverified until those jobs complete. New outputs use their new job IDs;
failed-run records are retained.

### Setup retry accepted

Job **625832 COMPLETED, exit 0:0, 1 minute 19 seconds**. Package installation,
`pip check`, pinned VAE download and CPU model loading all passed. The downloaded
weights occupy 334,643,276 bytes. See `vae-model-provenance.json` for revision and
file hashes and `vae-environment-2026-10-03.txt` for resolved package versions.
GPU reconstruction job **625833** remains pending; no reconstruction figures or
quality metrics are available yet. This supersedes the retry's pending-setup
snapshot above. No SD training has been run.

## VAE reconstruction accepted — 4 October 2026

After about 29 hours pending in comp3710, existing job 625833 was moved with
`scontrol update JobId=625833 Partition=a100-test`. Eligibility and the test
partition's 20-minute QoS limit were checked first. No duplicate was submitted.
It COMPLETED with exit 0:0 in 33 seconds on a100-b, NVIDIA A100-PCIE-40GB.

All 16 records (8 preselected slices at two resolutions) and corresponding
triptychs were retrieved into ignored `outputs/vae-625833/`. Finite checks passed;
the held-out project test split was not evaluated. Original source directory
names may say test because the project uses a new patient-grouped manifest.

| Resolution | Mean per-image PSNR, 8 images (dB) | Peak torch allocated GiB |
| --- | ---: | ---: |
| 256 | 27.0383 | 0.5218 |
| 512 | 35.3320 | 1.0866 |

PSNR is measured against each resolution's own preprocessed input after decoder
clipping. These numbers are not a common-native-resolution comparison or SD
training metrics. Upsampling to 512 does not create new acquired detail. GPU
allocation is VAE-only, not total device use or future training memory. Timing
includes a cold first inference and is not a warmed throughput benchmark.

Visual inspection of validation-1-AD at both sizes found preserved gross shape,
more visible fine-detail smoothing at 256, and smaller displayed error at 512.
This is one inspected pair, not a medical or population-wide quality judgment.
The raw decoder clipping fractions span roughly 47–73%; before finalizing the
protocol, measure unclipped error and distinguish background undershoot from
foreground overshoot. These fractions alone do not establish severe distortion.
Next: refine that diagnostic and prepare bounded training/save/resume validation;
no optimizer update has yet been run.

## Short training and restart acceptance — 4 October 2026

Protocol was recorded before execution in `smoke-training-protocol.md`.
CPU job 629842 completed in 2m28s, exit 0:0: dry-run resolution, installation,
pip check, pinned component download, CPU load and LoRA attachment passed.
VAE diagnostic 629851 completed in 12s, exit 0:0. Intensity-mask (>0.02) foreground
clipping averaged 0.97%/0.21% at 256/512; background clipping averaged 66.9%/76.3%
but correction magnitude was approximately 0.004. Raw/clipped MSE is retained
in `vae-clipping-2026-10-04.json`. This mask is not anatomical segmentation.

GPU job **629852 completed in 55 seconds, exit 0:0**, in a100-test. Real pretrained
SD1.5 LoRA updates ran as continuous8, interrupted4 and resumed4 in three Python
processes. Final adapter maximum difference was **0.0**. Optimizer, scheduler,
CPU/CUDA RNG and input/noise/timestep/loss traces matched within documented
floating tolerances. Corrupt-contract and changed-adapter negative controls were
rejected. No decreasing-loss or image-quality claim is inferred.

Trainable parameters: 797,184. Peak torch allocated bytes: 6,109,799,424 (~5.69 GiB).
Continuous checkpoint: 9,884,717 bytes (~9.43 MiB), includes optimizer and RNG.
Measured eight-update/save region: 4.376s; full three-process job: 55s. Do not use
this tiny run as a production throughput estimate. Frozen source weights are
loaded from the pinned fp16 files into float32; bfloat16 autocast is used for UNet
compute. Diffusers emitted dtype-casting advisory warnings; no non-finite output
or acceptance failure occurred. The constructor dtype cleanup can be considered
with future trainer changes, without altering this accepted historical run.

All raw output JSONs are retained in ignored local `outputs/`; model manifests,
environment snapshot and aggregate acceptance are in docs. No medical images or
patient-level manifests are uploaded to the website or public repository.
The HTML is updated with these measured states. Formal baseline/adaptation,
generation evaluation, and signal-driven checkpoint recovery remain outstanding.

## Paired generation pilot submitted — 4 October 2026

Job629919: a100-test, one GPU,20-minute wall-time cap. Protocol is in
pilot-protocol.md. Selected32 training patients (16AD/16NC),200 updates, fixed
8-image DDIM grids at steps0/100/200. Checkpoint100 resumes in a new process.
Submission is not success; results are pending. Held-out test data stays sealed.

### Pilot629919 result

COMPLETED, exit0:0,2m15s on A100. Retrieved both checkpoints, reports and all24
images into ignored local outputs/pilot-629919. Verified steps100/200, resumed
flag, identical contracts, preserved first100 traces,32 unique training indices,
finite loss/gradients, eight records per stage, PNG hashes, and identical paired
initial-noise hashes across all three stages. Training peak allocation5.69GiB;
final checkpoint9,912,045bytes. Aggregate: pilot-acceptance-2026-10-04.json.

Visual inspection of all three uncurated grids: step0 is strongly colored and
non-MRI-like; steps100/200 become predominantly grayscale with brain-like textures.
However, multiple disconnected structures, implausible geometry and collage-like
composition persist. AD/NC images sharing a seed are visually very similar.
This does not demonstrate anatomical validity or learned disease conditioning.
These are observed failure modes, not verified causal diagnoses. Sampling uses
256 pixels and guidance1, so these observations do not characterize every possible
base-model inference setting. Do not attribute changes solely to generalization
or infer privacy; nearest-neighbor auditing has not been done.

Next: investigate the sampling/settings contribution and training-slice composition,
then implement the simple baseline and broader evaluation before extending training.
No automatic long training was submitted.

## Full-split long experiment queued — 4 October 2026

User requested prioritizing the longer training queue to preserve analysis/rework
 time. Recorded full-training-protocol.md before submission. Frozen remote source:
~/comp3710-sd-audit/runs/full20k-20261004, with source-sha256.txt; shared model/data
and environment symlinks. No queued source will be edited in place.

Gate629930 runs actual continuous8 versus4+4 state comparison in a100-test.
Long job629931 uses afterok:629930, comp3710 account/partition,1GPU,2CPUs,4h cap.
Initial state: gate RUNNING; long PENDING(Dependency). Fresh base weights, all
training-split slices eligible,20000 updates, complete checkpoints and fixed
sampling every2000 updates. Approximately0.94 slice epochs, not proof of convergence.
Queue position and completion time are unknown. No validation/test evaluation.

Gate629930 COMPLETED in46s, exit0:0. Full state comparison passed with final
adapter max difference0.0; corruption negative controls rejected. Acceptance is
saved in full-gate-2026-10-04.json. Long job629931 dependency is satisfied and its
latest queue reason is Priority, not a failed test or missing GPU request.

## Long run629931 completed — verified5 October2026

Slurm COMPLETED, exit0:0, elapsed1h21m39s,20000 steps. Retrieved all ten
stage directories (approximately140MiB) including full checkpoints to ignored
outputs/full-629931. Verified report step counts, checkpoint byte sizes, finite
loss/gradient traces, matching contracts and all resumed trace prefixes, and all
88 generated PNG hashes with matching initial noise across stages. This is artifact
and trace verification, not a new continuous20000-step equivalence test.

Inspected the final eight-image uncurated grid: brain-like textures persist, but
fragmented/collage structures and background block artifacts remain. Matched-seed
AD/NC images look very similar. Generation quality is not accepted. Causes are
not established. Next priority is inference parity against the official pipeline
and controlled resolution/guidance comparisons, followed by training-data visual
inspection and nearest-neighbor audit. Do not simply infer that more steps will
resolve the artifacts. No new long job submitted during this verification.

## Inference audit submitted — 5 October 2026

Job632530 in a100-test is RUNNING on a100-a. Protocol recorded before execution
in inference-audit-protocol.md. No optimizer updates and no held-out data.
The official pipeline is compared against four historical quantized images and
then used for16 fixed factorial samples. A mismatch is diagnostic evidence, not
a reason to silently loosen tolerances. In particular, the historical loop casts
UNet output tofloat32 before the scheduler; the official mixed-precision path may
retainbf16 there. If outputs differ, distinguish arithmetic precision from sampler
logic before attributing generation artifacts to an implementation error.

### Inference audit632530 result

COMPLETED, exit0:0,1m57s. Retrieved all16 images and four uncurated grids.
Verified PNG hashes and identical initial noise across guidance settings within
each resolution. Four historical pairs do not match exactly: mean absolute pixel
differences0.93–2.29 on0–255 scale, maximum46–105. A follow-up632535 aligns only
the UNet-output float32 scheduler boundary and repeats the four historical pairs;
this distinguishes the suspected precision effect without changing trained weights.

Visual inspection of all four grids: guidance7.5 removes much of the collage and
background artifacts in these fixed examples at256 and512. Guidance1 at512 is
heavily textured/fragmented; merely increasing resolution does not fix it.
The256/guidance7.5 seeds look highly similar. These tiny selected-in-advance grids
cannot establish distribution coverage, anatomical validity, disease specificity
or privacy. Do not claim guidance7.5 is optimal or512 is universally better.

Follow-up632535 COMPLETED, exit0:0,26s. After matching the float32 scheduler-input
boundary, all four historical outputs match the official pipeline pixel-for-pixel
(max difference0), with identical historical initial-noise hashes. PNG hashes
verified locally. This strongly supports sampler consistency on the tested cases;
it does not prove all prompts/settings or intermediate latent equality. The prior
small mismatch is explained by the controlled precision-boundary change. No weight
update was needed. The large collage failures remain reproducible with guidance1,
while guidance7.5 changes sample appearance substantially on this small fixed set.
Next: broader predeclared seeds, nearest training/validation neighbors, diversity,
and a separately trained simple baseline. No further long training submitted.

## Cohort audit and baseline queued — 5 October 2026

Protocol: cohort-baseline-protocol.md, recorded before execution. GPU632557
produces64 base and64 adapted SD outputs at256/guidance7.5, with predeclared
seeds37200–37231. CPU632558 depends on it and performs private full/matched
train/validation nearest-neighbor and diversity diagnostics. Test remains sealed.

Baseline gate632563 was cancelled while pending to replace it with an immutable
source checkout after lint cleanup. Replacement gate632569 and dependent5-epoch
baseline632570 use ~/comp3710-sd-audit/runs/baseline128-20261005 with recorded
source hashes. Basic conditional conv VAE,128 native pixels,64 latent dimensions,
no pretrained weights. Epoch checkpoints support bounded loss on interruption;
no emergency signal-save claimed. Gate tests actual updates and exact save/reload
inference, not full interrupted-training trajectory equivalence. Submission and
static checks are not successful model or metric results.

## Cohort and baseline results verified — 6 October 2026

632557 COMPLETED6m08s;632558 COMPLETED53s;632569 COMPLETED8s;632570 COMPLETED3m02s;
all exit0:0. Retrieved128 SD images,64 baseline images and all five baseline
checkpoints. Verified192 image hashes, SD initial-noise pairing, five epochs each
covering21280 training samples, finite objectives and final checkpoint SHA256.
Supplementary baseline CPU audit635709 completed51s, exit0:0, using the same
previously recorded metric/reference protocol. Private neighbors remain ignored.

| Model | Matched train NN mean RMSE | Matched validation NN mean RMSE | Generated pairwise mean RMSE |
| --- | ---: | ---: | ---: |
| Base SD | 0.493368 | 0.495440 | 0.442523 |
| Adapted SD | 0.130497 | 0.131289 | 0.182098 |
| Small VAE | 0.077038 | 0.077355 | 0.117431 |

References:102 patients per split (33AD/69NC), one deterministic slice each.
All distances are grayscale64 pixel RMSE. Exact-copy reference controls passed.
The full-reference adapted-training minimum distance0.07021 is nonzero, but this
cannot exclude approximate memorization. Similar matched train/validation means
are a limited observation, not a privacy guarantee or statistical equivalence.

Visual review of all three uncurated grids: base SD produces colorful/illustrative
or collage-like images far from the target appearance; adapted SD produces much
sharper, similar brain-like structures with residual colored/background artifacts;
small VAE generates blurred structures. The blurred baseline wins low-resolution
RMSE. This is evidence that this metric alone is insufficient to rank fine-detail
quality, not evidence that SD is categorically better. Pairwise variability of
invalid base outputs is not useful anatomical diversity. Neither disease-label
control nor anatomical validity is established by these grids.

Baseline1268961 parameters;5 epochs=106400 training-image exposures, compared with
SD20000 updates at batch1. Baseline native128 versus SD256 further limits direct
fairness. Training VAE objective declined224.81->89.90, but this is not held-out
validation quality. Next: real-vs-real distance calibration, foreground/feature
metrics and private nearest-neighbor visual panels, then3–5 representative failure
analyses. Keep the test split untouched until final evaluation protocol is fixed.

## 2026-10-06 — Real-reference calibration and case inspection

CPU job635718 completed43s, exit0:0. Matched queries contain102 patients per
split (33AD/69NC), one deterministic slice each. Training queries exclude all
images from the same patient. Grayscale64 RMSE: real validation to full training
mean0.073101; real training to other-patient full training mean0.076297; real
validation to matched training mean0.091846. Real matched-set pairwise means:
train0.161217, validation0.152429. These are context, not privacy thresholds.

Five private panels inspected: gray background, colored artifact, and simple-VAE
blur are observed failures; minimum-distance and repeated-shape examples are
unconfirmed risk probes. Causes remain hypotheses requiring controlled checks.
See calibration-2026-10-06.json and failure-autopsies-2026-10-06.md. No test-set
evaluation performed. Next: foreground/color/edge and feature-space diagnostics.

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

## 2026-10-07 — CFG sweep submitted

Job636273: a100-test, oneGPU,20-minute limit. Fixed checkpoint,256/DDIM30/eta0,
aligned precision,32 seeds and2 labels, CFG3/5/7.5 (192 outputs). See
cfg-sweep-protocol.md. Regenerate7.5 for subsequent historical parity checking.
Local Ruff/format/syntax checks and remote shell syntax passed. A login-node
--help import stalled in I/O; stopped that exact invocation before submission.
Model import/execution remains to be verified by the scheduled job. No results
or best setting claimed at submission; test data stays sealed.

## 2026-10-07 — CFG sweep acceptance and visual review

Job636273 completed9m06s, exit0:0. Verified all192 PNG hashes and64 matched-noise
triplets. All64 regenerated CFG7.5 PNG hashes equal historical cohort632557.
Scheduler contracts match after sorting only the metadata list of defaulted keys.
Full three64-image grids and six individual F1/F2 outputs were visually reviewed.

| CFG | Mean color | Mean border | Mean edge |
| --- | ---: | ---: | ---: |
| 3 | 0.021165 | 0.204864 | 0.022431 |
| 5 | 0.017498 | 0.109868 | 0.023333 |
| 7.5 | 0.020839 | 0.059221 | 0.022961 |

Paired result:3->5 lowers border in64/64 samples,5->7.5 in63/64. In contrast,
3->5 lowers color in only23/64 samples (11/32 seed blocks); its lower mean is not
majority improvement.5->7.5 increases color in45/64 samples, with23/32 seed blocks
increasing. AD/NC share noise, so block counts avoid pretending64 independent seeds.
No inferential significance test or anatomical-quality ranking is claimed.

F1 AD37220: border0.57695/0.52643/0.41154 at CFG3/5/7.5. Gray background remains
in all three. CFG3 also has visibly textured background; higher guidance suppresses
some texture while leaving the brightness mismatch. F2 NC37217: color range
0.00769/0.00854/0.02104; cyan patch is conspicuous at7.5 and not conspicuous at3/5.
The observed dependence is on guidance within this fixed setup; it does not identify
the learned mechanism or establish that retraining would fix it.

The grids retain similar gross orientations/shapes and paired AD/NC resemblance.
No label-conditioned disease fidelity or mode-collapse conclusion follows from
visual resemblance. Existing F3/F5 remain unconfirmed risk probes.

Decision: retain7.5 as the existing comparison reference, with5 as a documented
tradeoff, not a newly proven optimum. Stop expanding CFG search for now. Prioritize
remaining required evidence: feature-space coverage/label sensitivity, interpolation,
and student check-off/implementation scope. Do not replace formal held-out validation
with selection on these32 diagnostic seeds. Test remains sealed. No new job submitted.

Reproduce with tools/audit_cfg_sweep.py; outputs stay private, aggregate evidence
is in docs/cfg-sweep-result-2026-10-07.json. Ruff passed on the real-data audit tool.

## 2026-10-07 — Interpolation submitted

Job639671 submitted: two noise-seed pairs, five linear mixing coefficients,
two fixed prompts,20 total images; CFG7.5/256/DDIM30 with aligned precision.
See interpolation-protocol.md for lower interior variance and endpoint checks.
Ruff, Python syntax and remote shell syntax passed. No result acceptance yet.
Private HTML updated with completed appearance/CFG evidence; interpolation remains
unverified there. No training or test-set evaluation initiated.

## 2026-10-07 — Initial-noise interpolation accepted

Job639671 COMPLETED1m05s,exit0:0. All20 PNG hashes verified; all eight endpoint
noise hashes and PNG hashes exactly match the accepted historical CFG7.5 cohort.
Complete four paths inspected at alpha0,.25,.5,.75,1. Reproduce integrity checks
with tools/audit_interpolation.py (Ruff passed).

Contours, local texture and intensity visibly vary along the four sampled paths.
Five positions cannot establish continuous perceptual smoothness or anatomical
validity. NC37216->37217 ends with the known cyan artifact; interpolation does not
repair it. Gross orientation remains similar across the displayed paths. This is
not evidence of disease-state interpolation, semantic coverage, or non-memorization.

Endpoint noise norms are63.74-64.51; midpoint norms45.26/45.69. Linear mixing
therefore changes noise magnitude as well as direction. Interior appearance cannot
be attributed to direction alone. AD/NC use the same two noise paths; there are
two independent endpoint pairs, not four independent noise experiments.

The narrow initial-noise interpolation diagnostic is complete; feature-space
coverage, label fidelity, and final test evaluation remain unperformed. Keep these
requirements separate. Next priority is a documented feature-space protocol with
real validation references, rather than further cosmetic CFG search. No new GPU
job submitted in this acceptance step. Student review remains pending.

## 2026-10-07 — Frozen VAE feature diagnostic

Job639765 COMPLETED19s,exit0:0.396 images ->256 features. Source-code hash matches
report; downloaded feature and private-row SHA256 match remote files. Reference
selection and every input image hash verified during extraction. No test images.

| Query/reference | Mean VAE-feature RMSE |
| --- | ---: |
| Validation ->102 matched training references | 0.899889 |
| Validation ->64 label-count-matched training references | 0.953590 |
| Validation ->64 original SD samples | 6.260788 |
| Validation ->64 adapted SD samples | 1.265907 |
| Validation ->64 simple VAE samples | 2.193512 |

This is a finite-sample reverse nearest-distance diagnostic, not a coverage percent.
Adapted SD is closer than the simple VAE by this directional measure but remains
farther than the real-reference context. Forward generated->validation means:
base7.327543, adapted1.926602, baseline1.871396. Forward and reverse directions
answer different questions; do not claim SD wins every measure. Class-stratified
results are preserved in feature-result-2026-10-07.json.

Joint t-SNE (396 points,perplexity30,seed3710,PCA init,1000 iterations) was rendered
and inspected. Baseline forms a separate region; base SD is also separated; adapted
samples overlap parts of the real projection and also extend away from it. These
are visualization observations, not numeric overlap/coverage evidence. AD/NC
colors for generated samples are prompt labels, not verified disease labels.

SD VAE is shared with the generator, trained for compression rather than clinical
assessment; pooled latents retain alignment/background/domain bias. The comparison
is supplementary and cannot establish anatomical quality, privacy, or AD control.
Plot and private feature arrays remain in outputs/features-639765. No patient
identifiers are included in the aggregate report or plot labels.

PDF2.4 checked: initial-noise interpolation alone does not complete the requested
synthetic/training-image interpolation. SSIM with an explicit pairing protocol and
encoded real/synthetic interpolation still remain; final test is sealed. Do not
label the entire evaluation complete. Student interpretation pending.

## 2026-10-07 — Explicit SSIM and encoded real/synthetic paths

Job639777 completed13s,exit0:0; exported204 preprocessed private real references
and two five-frame VAE-code interpolation paths, with original endpoints shown
separately. Extractor code hash verified. SSIM computed locally using pinned
scikit-image0.25.2; actual-reference identity control returned1. Ruff passed.

| Model | Fixed-reference mean SSIM | Feature-nearest mean SSIM | Nearest fraction>0.6 |
| --- | ---: | ---: | ---: |
| Original SD | 0.088068 | 0.087000 | 0/64 |
| Adapted SD | 0.350366 | 0.376754 | 23/64 |
| Simple VAE | 0.672548 | 0.726595 | 64/64 |

Real validation->same-label training feature-neighbor mean0.709904 (102 queries).
Under BOTH stated pairings, adapted SD's mean falls short of the course's0.6
reference target and below the simple VAE. Do not present a pass or silently alter
pairing to obtain a higher score. Native resolution, smoothing, black background,
alignment and arbitrary correspondence limit interpretation; low SD scores and
its visibly sharper outputs must both be reported. Reference selection used VAE
features before SSIM, not SSIM maximization. No ground-truth counterpart exists
for unconditional outputs. Scores are not clinical judgments or privacy guarantees.

Both encoded paths inspected: NC37215 and AD37220 to their same-label feature
nearest TRAIN reference. Intermediate frames show blurred/blended textures, with
structure changing between endpoints; no claim of anatomical continuity. VAE
endpoint generated reconstruction MSE0.000604/0.000546, training reconstruction
MSE0.004103/0.003971. Endpoints therefore reconstruct inputs rather than exactly
reproducing their pixels. This is posterior-mean interpolation in the frozen VAE,
not reverse-diffusion inversion. It does not disprove memorization.

Original/reference images and per-image SSIM stay private. Both interpolation
scope and remaining deficiencies should be discussed at check-off. Next: reconcile
course requirement coverage and prepare the one-page feasibility discussion;
prioritize reviewer-approved changes over open-ended parameter searches. Test
set remains sealed; no new training submitted on the strength of these scores.

## 2026-10-12 — E1 full-UNet fine-tuning: implementation and setup

Protocol fixed before any job ran: [e1-protocol.md](e1-protocol.md), commit de6f30e.
Code: `modules.py`, `train.py`, `predict.py` (commits 824bca7, 9b09bab, d0c2475);
`dataset.py` now verifies SHA-256 per slice when asked (693df5a). 17 CPU unit tests
pass with tiny random models, including equality of the guided DDIM sampler with
diffusers' StableDiffusionPipeline latents. These test mechanics, not image quality.

Remote run directory `~/comp3710-sd-audit/runs/e1-20261012`, symlinking the shared
`.venv-train`, `data` and `models`; `source-sha256.txt` records commit de6f30e and
the hash of each copied file, all equal to the committed blobs.

CPU job 651837 added matplotlib 3.10.9 to `.venv-train` for `train.py`'s loss
curves: COMPLETED, 2 min 07 s, exit 0:0, `pip check` clean, environment frozen to
`logs/e1_environment_651837.txt`. Diffusers 0.40.0 and PEFT 0.21.2 unchanged.

### S1 probe, job 651838

`a100-test`, node a100-a, NVIDIA A100-PCIE-40GB: COMPLETED in 3 min 05 s, exit 0:0.
200 full-UNet steps at batch 16 on the first 3,200 training slices (859,520,964
trainable parameters), 256 fixed validation slices. Steady throughput 94 images/s
after the first window (28 images/s while cuDNN benchmarked); peak PyTorch
allocation 23.24 GiB. Encoding 3,456 slices with SHA-256 checks took 35 s.
Validation loss after 200 steps: 0.0701 live, 0.0704 EMA; training-window loss
fell from 0.0847 (steps 1-50) to 0.0681 (steps 151-200), still in warm-up (lr 4e-6).

The 1.72 GB float16 EMA file was written to job scratch and reloaded by
`predict.py` for two seeds per class. Its printed statistics (channel spread
0.005, border mean 0.0004-0.0006) come from four images after 200 steps and are a
pipeline check only, not an E1 result. Visual check of `samples/epoch-01.png`:
black background and grayscale brain structure, while the pre-training grid
(`epoch-00.png`, base SD at CFG 7.5) shows colourful non-MRI images. The eight
probe samples share a similar pose, so the S3 audit must examine diversity.

At 94 images/s the protocol's 10 epochs need about 38 min of training; the
pre-registered epoch count is kept rather than extended. Main run: job 651851.

### S2 scheduling change

Job 651851 (4 h request, 200 min budget, as in the protocol) waited on priority
with an estimated start of 22:08. With S1's measured throughput the run needs
about 50 min including setup, validation and sampling, so it was cancelled
before starting and resubmitted unchanged except for scheduling: job 651860,
`--time=01:30:00`, `--time-budget-minutes 75`. The training recipe, epochs and
acceptance criteria are as recorded in e1-protocol.md.

### S2 main run, job 651860

`comp3710`, node a100-6, NVIDIA A100-PCIE-40GB: started 2026-10-12 03:36,
COMPLETED in 45 min 29 s, exit 0:0, stop reason `completed`. 10 epochs x 1,330
steps = 13,300 steps at batch 16 over all 21,280 training slices, every slice
hash-checked during encoding (133 s). 93.1 images/s over 2,286 s of training;
peak PyTorch allocation 23.82 GiB. EMA weights saved at epochs 5 and 10
(1.72 GB each, `runs/e1-20261012/outputs/e1-651860/weights`); home then had
5.0 GB free.

| Epoch | Train loss | Val loss, live | Val loss, EMA |
| ---: | ---: | ---: | ---: |
| 1 | 0.06282 | 0.05820 | 0.05755 |
| 2 | 0.05834 | 0.05675 | 0.05619 |
| 3 | 0.05579 | 0.05633 | 0.05552 |
| 5 | 0.05500 | 0.05555 | 0.05487 |
| 7 | 0.05349 | 0.05537 | 0.05458 |
| 9 | 0.05206 | 0.05534 | 0.05452 |
| 10 | 0.05264 | 0.05530 | 0.05457 |

Validation is 1,024 fixed slices with fixed noise and timesteps, plain noise
and true prompts. EMA validation loss falls to a minimum at epoch 9 and rises
once at epoch 10 (+0.00006), so the two-consecutive-rise stop condition was not
met and epoch 10 remains the primary weights. From epoch 7 the training loss
keeps falling while validation loss is flat: a widening gap that makes the
memorisation audit in S3 more important, not less.

Visual review of the per-epoch fixed-seed grids (8 images each, CFG 7.5):
from epoch 1 every sample has a black background and grayscale brain anatomy
in the same orientation as the real slices. Remaining defects at epoch 10:
frayed, speckled brain outlines; faint blue tints along some ventricle and
cortex edges; and one dark, low-contrast, ghosted sample (NC seed 37101) that
looked normal at epoch 5. Eight images are not the evaluation; S3 is.
