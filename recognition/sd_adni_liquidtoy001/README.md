# Stable Diffusion Adaptation for ADNI Brain MRI

COMP3710 final report project by liquidtoy001.

**Status: planning only. No dataset audit, model training, or evaluation has been completed.**

This project will investigate whether adapting a pretrained latent diffusion model to ADNI brain MRI improves structural fidelity and distribution coverage without increasing evidence of training-image memorization. The intended output is a research prototype and an evidence-based assessment, not a clinically validated generator.

## Proposed approach

Use the course ADNI dataset at `/home/groups/comp3710/ADNI` on Rangpur. Audit patient identifiers before defining patient-independent training, validation, and test splits. Check reconstruction through the pretrained VAE before training the denoising model. Compare an implemented simple generative baseline, the unadapted diffusion model, and the adapted model using the same evaluation protocol.

The initial candidate is an SD 1.x-scale model with a frozen VAE and text encoder. LoRA is a candidate for the first training smoke test; its suitability as the main assessed implementation and the permitted pretrained-library scope must be clarified with the tutor. The exact checkpoint, revision, conditioning scheme, and final adaptation method are not yet selected.

## Project records

- [Project plan and feasibility review outline](docs/project-plan.md)
- [Experiment ledger](docs/experiments.md)
- [AI assistance record](AI_USAGE.md)
- [Local development and single-PR workflow](docs/version-control.md)

## Planned implementation

The assessed implementation will include `modules.py`, `dataset.py`, `train.py`, and `predict.py`. These files will be added when they contain working functionality. Environment versions, commands, architecture explanations, figures, and measured results will be documented as they become available.

No executable training workflow is available yet.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with interpreting the supplied assessment, discussing model and compute choices, and drafting this initial project documentation and Git configuration. The assistant checked the local assessment text and repository state. No training code or experimental results have been generated at this stage.

The student must review these documents and verify all subsequent code, equations, metrics, and experimental claims. The [AI assistance record](AI_USAGE.md) will track assistance and verification as development proceeds; student verification will only be recorded once performed.

## Submission and version control

Development takes place on `topic-recognition` in the student's fork. The eventual pull request targets `shakes76/PatternAnalysis-2026:topic-recognition`. Commit changes at genuine milestones, using explicit file paths when staging. Keep datasets, model weights, credentials, and bulk run artifacts outside Git; include selected report figures and compact measured result tables deliberately.

The assessment also requires a midpoint feasibility check-off, a PDF export of this report submitted to Turnitin, and a later response to teaching-team code review. On 2 October 2026 the student supplied a course-profile excerpt showing 19–30 October 2026 and a tutor clarification that check-off shares the final-report deadline and can be completed earlier when prepared. This supersedes the earlier tentative 26 October date. The exact Phase 1 submission timestamp must still be checked in the assessment submission page; the quoted answer does not explicitly specify that timestamp.

## Sources

- Course assessment: `COMP3710_Report_2026_draft_v2.1(1).pdf`, especially Sections 2.4, 3–6 and Appendix A. The supplied document is marked Draft.
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).

Code adaptations will be attributed in the relevant docstrings, this report, and the eventual PR description.
