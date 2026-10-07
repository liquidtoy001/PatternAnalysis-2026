"""Build the one-page evidence-based discussion draft; tutor check-off is pending."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate


def main():
    output = Path("E:/Documents/COMP3710/finalreport/checkoff")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "feasibility-review-2026-10-07.pdf"
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

    def add(text, style=body):
        story.append(Paragraph(text, style))

    add("COMP3710 | Feasibility review", title)
    add("ADNI synthetic MRI with Stable Diffusion LoRA | Project: sd_adni_liquidtoy001")
    add(
        "<b>Discussion draft - 7 October 2026. Student review and tutor check-off pending.</b><br/>"
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
        "(2) finite training and verified save/resume; (3) compare original SD, adapted SD and an implemented "
        "simple VAE under documented SSIM/reference rules, reporting the course's 0.6 target honestly; "
        "(4) provide uncurated grids, feature distances, t-SNE, nearest-neighbor and interpolation diagnostics, "
        "and 3-5 observed failure cases; (5) run on one A100 40GB within job limits with measured peak memory "
        "and durable checkpoints. Criteria 1-2 have evidence; evaluation exists but SD fidelity remains below target."
    )
    add("2. Model choice and course concepts", heading)
    add(
        "ADNI: 30,520 slices, 680 patients. Patient split: 476 train / 102 validation / 102 test; "
        "the original folders shared 216 patients. SD1.5 is the Hard architecture route, adapted at 256 with "
        "rank 4 attention LoRA (797,184 trainable parameters), frozen VAE/text encoder/base UNet, and noise-prediction MSE. "
        "Concepts: latent compression, forward noising, denoising, conditioning and low-rank adaptation. "
        "Reference: Rombach et al., <i>High-Resolution Image Synthesis with Latent Diffusion Models</i> (CVPR 2022). "
        "Pretrained-library/LoRA scope needs tutor confirmation; Hard eligibility is not a mark guarantee."
    )
    add("3. Measured feasibility and unresolved quality", heading)
    add(
        "SD 20,000 updates: 1 h 21 m 39 s; peak PyTorch allocation 5.69 GiB. Eight-step continuous/resumed "
        "comparison matched exactly; this does not prove a continuous 20k equivalence. Simple conditional VAE: "
        "128 px, 1,268,961 parameters, five epochs in 3 m 02 s. Its 106,400 image exposures exceed SD's20,000, "
        "so the comparison is exploratory, not budget-matched."
    )
    add(
        "Mean SSIM (fixed same-label reference / VAE-feature nearest): original SD 0.088/0.087; "
        "adapted SD 0.350/0.377; simple VAE 0.673/0.727. <b>Adapted SD does not meet 0.6 under either protocol.</b> "
        "No unique ground-truth target exists for a generated image; nearest selection is optimistic and "
        "background/alignment/smoothing affect scores. Gray backgrounds, color artifacts and baseline blur "
        "are observed failures. Shared-VAE feature distances and t-SNE are supplementary, not independent clinical evidence."
    )
    add("4. Risks, budget, next experiment and fallback", heading)
    add(
        "Risks: unclear LoRA assessment scope, anatomical artifacts, weak label fidelity, possible approximate "
        "memorization and metric bias. Test images remain sealed; patient outputs remain private. "
        "Main SD+baseline training consumed about 85 GPU-minutes; diagnostics are additional. No further long "
        "training is scheduled. After feedback, propose one bounded matched-resolution/budget comparison "
        "(maximum 2 A100-hours, checkpointed), then freeze evaluation before opening test data. "
        "Fallback: retain the reproducible baseline and report negative SD findings; any architecture change "
        "requires agreement on scope/difficulty."
    )
    add("Discussion points / completion record", heading)
    add(
        "Confirm (a) LoRA/library implementation scope, (b) acceptable SSIM correspondence/target, "
        "(c) whether VAE-code real/synthetic interpolation is sufficient. Student must verify and defend this draft; "
        "Codex assisted code, experiments and wording. Consultation date: __________  Tutor/record: __________"
    )
    add(
        "Sources: task sheet sections 2.4 and3; project docs/experiments.md and versioned aggregate results. "
        "The review must later be included in README; this PDF is not evidence of completed check-off."
    )
    doc.build(story)
    print(path)


if __name__ == "__main__":
    main()
