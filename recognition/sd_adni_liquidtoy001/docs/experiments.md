# Experiment Ledger

No experiments have been run for this project.

Pre-experiment audit: [Rangpur/ADNI, 2026-10-02](rangpur-audit-2026-10-02.md), CPU job 620988, completed with exit 0:0. This is a dataset/resource inspection, not a model experiment. The report also records unsuccessful preliminary shell probes.

Data preparation: [status](data-preparation-status.md). Local data-contract tests pass; CPU job 623310 is pending. No model metrics or completed split results are available.

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
