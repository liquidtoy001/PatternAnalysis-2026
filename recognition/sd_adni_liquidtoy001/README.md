# Stable Diffusion Adaptation for ADNI Brain MRI

COMP3710 final report project by liquidtoy001.

**Status: initial read-only Rangpur/data audit complete. No model training or evaluation has been completed.** The supplied train/test directories share 216 patients; a patient-independent project split is required. Home has approximately 2.20 GiB free, so persistent model/checkpoint storage must be resolved before training.

This project will investigate whether adapting a pretrained latent diffusion model to ADNI brain MRI improves structural fidelity and distribution coverage without increasing evidence of training-image memorization. The intended output is a research prototype and an evidence-based assessment, not a clinically validated generator.

## Proposed approach

Use the course ADNI dataset at `/home/groups/comp3710/ADNI` on Rangpur. Audit patient identifiers before defining patient-independent training, validation, and test splits. Check reconstruction through the pretrained VAE before training the denoising model. Compare an implemented simple generative baseline, the unadapted diffusion model, and the adapted model using the same evaluation protocol.

The initial candidate is an SD 1.x-scale model with a frozen VAE and text encoder. LoRA is a candidate for the first training smoke test; its suitability as the main assessed implementation and the permitted pretrained-library scope must be clarified with the tutor. The exact checkpoint, revision, conditioning scheme, and final adaptation method are not yet selected.

## Project records

- [Project plan and feasibility review outline](docs/project-plan.md)
- [Experiment ledger](docs/experiments.md)
- [AI assistance record](AI_USAGE.md)
- [Local development and single-PR workflow](docs/version-control.md)
- [Rangpur and ADNI audit, 2 October 2026](docs/rangpur-audit-2026-10-02.md)
- [Provisional patient split and preprocessing protocol](docs/data-protocol.md)
- [Data preparation implementation and validation status](docs/data-preparation-status.md)

## Planned implementation

`dataset.py` now implements manifest validation and image loading, with seven passing local contract tests. Full-data CPU preparation/acceptance passed in job 623310 (40 seconds, exit 0:0). `modules.py`, `train.py`, and `predict.py` will be added when they contain working functionality. Environment versions, commands, architecture explanations, figures, and measured results will be documented as they become available.

An eight-step LoRA engineering diagnostic passed continuous-versus-restarted acceptance (job 629852). A general-purpose training workflow and quality evaluation are not available yet. A bounded VAE reconstruction diagnostic is now available; see [the protocol](docs/vae-protocol.md).

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with interpreting the supplied assessment, discussing model and compute choices, drafting project documentation and Git configuration, and implementing/executing a read-only dataset audit. The assistant checked the local assessment text, repository state, remote resource information, and metadata-backed split overlap. The audit produced aggregate observations; no training or model-evaluation results exist yet.

The student must review these documents and verify all subsequent code, equations, metrics, and experimental claims. The [AI assistance record](AI_USAGE.md) will track assistance and verification as development proceeds; student verification will only be recorded once performed.

## Submission and version control

Development takes place on `topic-recognition` in the student's fork. The eventual pull request targets `shakes76/PatternAnalysis-2026:topic-recognition`. Commit changes at genuine milestones, using explicit file paths when staging. Keep datasets, model weights, credentials, and bulk run artifacts outside Git; include selected report figures and compact measured result tables deliberately.

The assessment also requires a midpoint feasibility check-off, a PDF export of this report submitted to Turnitin, and a later response to teaching-team code review. On 2 October 2026 the student supplied a course-profile excerpt showing 19–30 October 2026 and a tutor clarification that check-off shares the final-report deadline and can be completed earlier when prepared. This supersedes the earlier tentative 26 October date. The exact Phase 1 submission timestamp must still be checked in the assessment submission page; the quoted answer does not explicitly specify that timestamp.

## Sources

- Course assessment: `COMP3710_Report_2026_draft_v2.1(1).pdf`, especially Sections 2.4, 3–6 and Appendix A. The supplied document is marked Draft.
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).

Code adaptations will be attributed in the relevant docstrings, this report, and the eventual PR description.
