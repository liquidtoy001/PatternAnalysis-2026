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
