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
