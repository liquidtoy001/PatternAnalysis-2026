"""Build the one-page evidence-based discussion draft; tutor check-off is pending."""

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate


PROBE = (
    "E1 probe (job 651838, 12 Oct): 200 full-UNet steps at batch 16 ran at 94 images/s with "
    "23.2 GiB peak allocation on one A100 40GB; EMA weights were saved and reloaded for generation. "
    "The 10-epoch main run (job 651851) is about 45 minutes of training."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("E:/Documents/COMP3710/finalreport/checkoff"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "feasibility-review-2026-10-12.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=(210 * mm, 297 * mm),
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )
    body = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9.2,
        leading=12,
        spaceAfter=5,
        alignment=TA_LEFT,
    )
    heading = ParagraphStyle(
        "heading",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=13,
        textColor=colors.HexColor("#135866"),
        spaceBefore=7,
    )
    title = ParagraphStyle(
        "title", parent=heading, fontSize=16, leading=19, spaceBefore=0
    )
    story = []
    PROBE_SENTENCE = PROBE

    def add(text, style=body):
        story.append(Paragraph(text, style))

    add("COMP3710 | Feasibility review", title)
    add(
        "ADNI synthetic MRI with Stable Diffusion fine-tuning | Project: sd_adni_liquidtoy001"
    )
    add(
        "<b>Discussion draft - 12 October 2026 (revises 7 October). Student review and tutor check-off pending.</b><br/>"
        "Experiments below have already run; this is a current review, not a retrospective claim of prior approval."
    )
    add("1. User need, scope and acceptance criteria", heading)
    add(
        "A research engineer needs to assess whether synthetic 2D MRI cohorts offer useful fidelity and coverage "
        "without reproducing training patients. This prototype compares models and diagnoses limitations; it does "
        "not establish clinical utility, anonymization or safe public release."
    )
    add(
        "<b>Five testable criteria:</b> (1) Zero known cross-split patient/scan/file-hash overlap; "
        "(2) finite training and verified save/resume; (3) compare original SD, LoRA-adapted SD (E0), fully "
        "fine-tuned SD (E1) and an implemented simple VAE under one documented SSIM/reference protocol, "
        "reporting the course's 0.6 target honestly; "
        "(4) provide uncurated grids, feature distances, t-SNE, nearest-neighbor and interpolation diagnostics, "
        "and 3-5 observed failure cases; (5) run on one A100 40GB within job limits with measured peak memory "
        "and durable checkpoints. Criteria 1-2 have evidence; E0 evaluation exists but its SD fidelity is below target."
    )
    add("2. Model choice and course concepts", heading)
    add(
        "ADNI: 30,520 slices, 680 subjects. Filename numbers are scan IDs: the original train/test folders share "
        "no scan, but metadata subject IDs show 216 subjects in both (re-verified 12 Oct), so the project splits "
        "by subject: 476 train / 102 validation / 102 test. SD1.5 latent diffusion is the Hard route at 256 px "
        "with noise-prediction MSE and frozen VAE/text encoder. E0 trained rank 4 attention LoRA (797,184 parameters). "
        "E1 fine-tunes the full UNet with 10% empty-prompt dropout, offset noise and EMA weights. "
        "Concepts: latent compression, forward noising, denoising, conditioning, classifier-free guidance. "
        "Reference: Rombach et al., <i>High-Resolution Image Synthesis with Latent Diffusion Models</i> (CVPR 2022). "
        "Library scope needs tutor confirmation; Hard eligibility is not a mark guarantee."
    )
    add("3. Measured feasibility and unresolved quality", heading)
    add(
        "E0 20,000 updates: 1 h 21 m 39 s; peak PyTorch allocation 5.69 GiB. Eight-step continuous/resumed "
        "comparison matched exactly; this does not prove a continuous 20k equivalence. Simple conditional VAE: "
        "128 px, 1,268,961 parameters, five epochs in 3 m 02 s. Its 106,400 image exposures exceed E0's 20,000, "
        "so the comparison is exploratory, not budget-matched."
    )
    add(
        "Mean SSIM (fixed same-label reference / VAE-feature nearest): original SD 0.088/0.087; "
        "adapted SD 0.350/0.377; simple VAE 0.673/0.727. <b>Adapted SD does not meet 0.6 under either protocol.</b> "
        "No unique ground-truth target exists for a generated image; nearest selection is optimistic and "
        "background/alignment/smoothing affect scores. Gray backgrounds, color artifacts and baseline blur "
        "are observed failures. Shared-VAE feature distances and t-SNE are supplementary, not independent clinical evidence."
    )
    add(PROBE_SENTENCE)
    add("4. Risks, budget, next experiment and fallback", heading)
    add(
        "Next experiment (E1, approved 12 Oct): each change targets an observed E0 failure. Offset noise targets the grey "
        "background, prompt dropout the colour artifacts that grow with guidance, full-UNet training the photographic "
        "texture, and 10 epochs at batch 16 the under-training (E0 saw 0.94 epochs). Criteria fixed before running, "
        "validation data only: nearest SSIM mean >= 0.6, border intensity <= 0.01, colour range below E0, and "
        "generated-to-training nearest RMSE >= 0.0731 (real-to-real). Budget: about 3, at most 5 A100-hours. "
        "Risks: full fine-tuning may memorize training slices, which the nearest-neighbor audit tests; the changes "
        "are bundled, so a LoRA ablation follows if time allows. Test images stay sealed until the protocol is frozen. "
        "Fallback: report E0 and E1 against the same criteria, including any negative result."
    )
    add("Discussion points / completion record", heading)
    add(
        "Confirm (a) whether the 2.6 clarification (libraries may load models; training loop and evaluation must be "
        "own code) also applies to 2.4, (b) acceptable SSIM correspondence/target, (c) the subject-level re-split. "
        "Student must verify and defend this draft; Codex and Claude Code assisted code, experiments and wording. "
        "Consultation date: __________  Tutor/record: __________"
    )
    add(
        "Sources: task sheet v2.2 sections 2.4 and 3; docs/experiments.md, docs/e1-protocol.md and versioned results. "
        "The review must later be included in README; this PDF is not evidence of completed check-off."
    )
    doc.build(story)
    print(path)


if __name__ == "__main__":
    main()
