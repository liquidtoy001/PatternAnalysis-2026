# Data Preparation Status — 2 October 2026

## Implemented

- `dataset.py`: full-manifest validation, split-specific lazy loading, aspect-preserving square padding, bilinear resizing, replicated grayscale channels, float32 tensors in [-1,1].
- `tools/prepare_data.py`: metadata-backed patient grouping, deterministic class-stratified 70/15/15 allocation, full JPEG decode and SHA-256 checks, private CSV manifest and aggregate JSON report.
- `tools/check_data.py`: checks real source coverage, deterministic grouping, sampled training/validation tensors and collation, plus rejection of intentionally corrupted copies of the real manifest.
- `tools/prepare_data.sbatch`: bounded CPU-only preparation/validation without dependency installation or model downloads.

The [protocol](data-protocol.md) was recorded before running preparation. The split is provisional pending confirmation of the course evaluation requirements.

## Completed validation

Seven local unit tests passed using deliberately synthetic tiny grayscale images and manifest fixtures. They verify geometry and normalization, patient/content overlap rejection, path traversal and label-path mismatch rejection, resolution checks, and class-stratified counts/order independence. These are contract tests, not evidence of clinical quality or successful full-data processing. PyTorch 2.13.0+cu132 and Pillow 12.3.0 were used locally. No packages were installed.

Ruff check and formatting validation passed on the new Python files. The loader uses PyTorch and Pillow; torchvision is not required for pixel conversion.

## Remote acceptance pending

Immediate CPU allocation 623309 was cancelled when resources were unavailable. Batch job **623310**, `sd-data-check`, was submitted to the CPU partition with one CPU and a five-minute running-time limit. Last observed state: **PENDING (Resources)**. Another user-owned Transformer backup job was also pending and was not modified.

No actual new split counts, image-decoding success, or real-data loader acceptance are claimed yet. Retrieve and inspect the following only after the job completes successfully:

```text
~/comp3710-sd-audit/project/data/preparation.json
~/comp3710-sd-audit/project/data/validation.json
~/comp3710-sd-audit/project/data/manifest-seed3710.csv
~/comp3710-sd-audit/project/data_check_623310.out
~/comp3710-sd-audit/project/data_check_623310.err
```

The CSV contains patient-level information and remains outside public Git history. Code writes only small project files and the manifest; it does not copy MRI images, modify source data, install dependencies, clean home, or use a GPU. The job is not an automatic ongoing monitor.

## Reproduction

From this project directory on a machine with PyTorch and Pillow:

```bash
python -m unittest discover -s tests -v
```

On Rangpur, upload `dataset.py` and `tools/` to an isolated project directory, enter it, and submit `sbatch tools/prepare_data.sbatch`. The generator refuses to overwrite an existing manifest. Do not resubmit a completed preparation without first deciding whether to reuse the verified manifest or create a separate run directory. Check `sacct` and both JSON outputs, not merely log existence.

After real-data acceptance, address durable model/checkpoint storage and run a VAE reconstruction check on training/validation samples. No GPU experiment is ready or queued yet.
