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
