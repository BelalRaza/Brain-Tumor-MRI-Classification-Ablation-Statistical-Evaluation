#!/usr/bin/env python3
"""Build the 12-slide M4/M5 presentation (paper/M4_M5_presentation.pptx) from real results."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "M4_M5_presentation.pptx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
RED = RGBColor(0xC0, 0x50, 0x4D)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xEE, 0xF3, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def add_slide(title: str, num: int, subtitle: str | None = None):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(11.5), Inches(0.75))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    if subtitle:
        st = s.shapes.add_textbox(Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.45))
        q = st.text_frame.paragraphs[0]
        q.text = subtitle
        q.font.size = Pt(15)
        q.font.italic = True
        q.font.color.rgb = GREY
    ft = s.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.35))
    fp = ft.text_frame.paragraphs[0]
    fp.text = f"Brain Tumor CE-MRI · M4 & M5 · benchmark study, no clinical claims        {num} / 12"
    fp.font.size = Pt(10)
    fp.font.color.rgb = GREY
    fp.alignment = PP_ALIGN.RIGHT
    return s


def bullets(slide, items, left, top, width, height, size=16):
    """items: list of str or (str, level) or (str, level, color)."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        if isinstance(it, str):
            text, level, color = it, 0, None
        else:
            text, level, color = (list(it) + [None])[:3]
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = ("• " if level == 0 else "– ") + text
        p.level = level
        p.font.size = Pt(size - 2 * level)
        p.space_after = Pt(6)
        if color is not None:
            p.font.color.rgb = color
            p.font.bold = True
    return tb


def table(slide, rows, left, top, width, height, col_widths=None, size=12, bold_rows=(), header_fill=NAVY):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, left, top, width, height)
    t = shp.table
    if col_widths:
        total = sum(col_widths)
        for i, w in enumerate(col_widths):
            t.columns[i].width = Emu(int(width * w / total))
    for r in range(nr):
        for c in range(nc):
            cell = t.cell(r, c)
            cell.text = str(rows[r][c])
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(size)
            cell.margin_left = cell.margin_right = Inches(0.05)
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
                para.font.color.rgb = WHITE
                para.font.bold = True
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
                if r in bold_rows:
                    para.font.bold = True
                    para.font.color.rgb = GREEN
    return t


def picture(slide, path, left, top, width=None, height=None):
    return slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def callout(slide, text, left, top, width, height, color=RED, size=15):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = color
    box.line.fill.background()
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER


# ---------------------------------------------------------------- 1. Title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
tb = s.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.4))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Stress-Testing EfficientNetB2 for Brain Tumor MRI Classification"
p.font.size = Pt(38)
p.font.bold = True
p.font.color.rgb = WHITE
p2 = tf.add_paragraph()
p2.text = "Patient-wise validation · Class-balanced focal loss · Deduplicated external test (BRISC 2025)"
p2.font.size = Pt(20)
p2.font.color.rgb = RGBColor(0xC8, 0xD6, 0xE5)
tb2 = s.shapes.add_textbox(Inches(0.8), Inches(4.6), Inches(11.7), Inches(1.8))
tf2 = tb2.text_frame
for i, line in enumerate([
    "Milestones M4 (implementation + ablations) and M5 (failure analysis + paper)",
    "Base paper: Zulfiqar et al., Biomed. Signal Process. Control 84 (2023) — Figshare CE-MRI, 3 classes",
    "Vaishnav Verma · CV Research-Reproduction Project · No clinical claims",
]):
    q = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
    q.text = line
    q.font.size = Pt(16)
    q.font.color.rgb = WHITE

# ---------------------------------------------------------------- 2. Recap / problem
s = add_slide("Where M3 left us: the leakage gap", 2, "Same model, same data — only the split changed")
picture(s, FIG / "slice_vs_patient_gap.png", Inches(0.4), Inches(1.6), width=Inches(6.8))
bullets(s, [
    "Figshare CE-MRI: 3,064 slices from only 233 patients (~13 slices / patient)",
    "Slice-wise split puts neighbouring slices of the same patient in train AND test → memorisation",
    "Patient-wise test: accuracy 98.21% → 92.50%, meningioma recall 94.37% → 78.47%",
    "Even with a patient-wise test, the checkpoint was picked with validation_split=0.1 on augmented copies → val accuracy 99.44% (leaky)",
    ("Meningioma is the weak class: 19/144 → pituitary, 12/144 → glioma", 0, RED),
], Inches(7.4), Inches(1.7), Inches(5.6), Inches(5.0), size=16)

# ---------------------------------------------------------------- 3. Hypothesis triage
s = add_slide("Hypotheses from M3 — what we tested and why", 3, "Budget: two training runs → one experimental variable")
table(s, [
    ["#", "Hypothesis", "Role in M4", "Cost", "Decision"],
    ["H1", "Pick best model on NEW patients (patient-grouped val)", "Protocol fix (both runs)", "0 extra runs", "✔ Done"],
    ["H2", "Extra attention to meningioma (class-balanced focal loss)", "THE one experimental change", "1 run", "✔ Main test"],
    ["H3", "Fix randomness (seeds, hashes, config logs)", "Reproducibility support", "Unit test only", "✔ Done"],
    ["H4", "Test on a different dataset (BRISC 2025)", "Evaluation only", "Inference only", "✔ Done"],
    ["H5", "Scanner-style intensity augmentation", "Would need a 3rd run", "1+ run", "✘ Future work"],
], Inches(0.5), Inches(1.7), Inches(12.3), Inches(3.2), col_widths=[0.6, 5, 3, 1.8, 1.6], size=14)
callout(s, "Run A and Run B differ ONLY in the loss function — split, seed, backbone, optimiser, augmentation files and epoch budget are identical",
        Inches(1.2), Inches(5.4), Inches(10.9), Inches(1.0), color=NAVY, size=16)

# ---------------------------------------------------------------- 4. Pre-registered predictions
s = add_slide("Predictions written BEFORE training (results/predictions.md)", 4, "Handout rule 1: predict first — being wrong is fine")
table(s, [
    ["Hypothesis", "Prediction", "Mechanism"],
    ["H2 · meningioma recall", "+3 to +8 pp vs CE", "Fewest source slices (484 vs 1,024) + hardest class"],
    ["H2 · macro recall", "+1 to +3 pp", "Gain concentrated in meningioma"],
    ["H2 · accuracy", "within ±1 pp (may dip)", "Some glioma/pituitary pushed toward meningioma"],
    ["H2 · macro AUC", "stable (±0.01)", "Focal changes calibration more than ranking"],
    ["H1 · patient val", "close to test (≤ ~3 pp), far below 99.4%", "Val patients unseen in training"],
    ["H4 · BRISC", "accuracy drops 10–25 pp; meningioma worst", "Different scanners, JPEG, mixed planes"],
    ["H3 · determinism", "same seed → identical first-epoch loss (CPU)", "Seeded RNGs + op determinism"],
], Inches(0.5), Inches(1.7), Inches(12.3), Inches(4.6), col_widths=[2.6, 4.2, 5.5], size=14)
bullets(s, ["Falsification rule: H2 is NOT supported if meningioma recall gains < 1 pp or the bootstrap CIs overlap almost completely"],
        Inches(0.5), Inches(6.45), Inches(12.3), Inches(0.6), size=14)

# ---------------------------------------------------------------- 5. Design
s = add_slide("Experimental design", 5, "Patient-grouped splits with zero overlap (asserted by unit tests)")
table(s, [
    ["Split", "Glioma", "Meningioma", "Pituitary", "Total"],
    ["Train (source slices)", "1,024 (61 pts)", "484 (56 pts)", "637 (42 pts)", "2,145 (159 pts)"],
    ["Train (augmented files)", "3,072", "2,903", "3,185", "9,160"],
    ["Val (patient-grouped)", "170 (10 pts)", "80 (9 pts)", "109 (7 pts)", "359 (26 pts)"],
    ["Test (patient-wise, untouched)", "232 (18 pts)", "144 (17 pts)", "184 (13 pts)", "560 (48 pts)"],
], Inches(0.5), Inches(1.65), Inches(7.6), Inches(2.4), col_widths=[3, 1.8, 1.8, 1.8, 2], size=13)
bullets(s, [
    "Run A: categorical cross-entropy",
    "Run B: class-balanced focal loss",
    ("FL = −Σ αc · yc · (1 − pc)^γ · log pc,  γ = 2", 1),
    ("αc from PRE-augmentation counts (Cui et al., β = 0.999)", 1),
    ("α = [0.744, 1.243, 1.012] → meningioma 1.67× glioma", 1),
    "Why pre-aug counts? Offline aug already equalises files; 2,903 meningioma files come from only 484 slices",
], Inches(8.3), Inches(1.6), Inches(4.8), Inches(4.6), size=15)
bullets(s, [
    "Shared: EfficientNetB2 (ImageNet), 240 px, Adam 1e-3, batch 32, seed 42, max 25 epochs",
    "LR ×0.3 after 3 epochs without val gain; early stop after 7; checkpoint = best patient-val macro-F1 (tie-break meningioma recall)",
    "Test set touched once, after training · 1,000× bootstrap CIs resampling PATIENTS",
], Inches(0.5), Inches(4.3), Inches(7.6), Inches(2.6), size=14)

# ---------------------------------------------------------------- 6. Compute + reproducibility
s = add_slide("Implementation & compute (H3 reproducibility)", 6, "Both runs in parallel on AWS SageMaker, every epoch saved to S3")
table(s, [
    ["", "Run A (CE)", "Run B (CB-Focal)"],
    ["Instance", "ml.g5.xlarge (A10G 24 GB)", "ml.g5.2xlarge (A10G 24 GB)"],
    ["Speed", "~26 s / epoch", "~26 s / epoch"],
    ["Epochs run → best", "11 → epoch 4", "21 → epoch 14"],
    ["Billable time", "767 s (12.8 min)", "974 s (16.2 min)"],
    ["Approx. cost", "≈ $0.30", "≈ $0.41"],
], Inches(0.5), Inches(1.7), Inches(6.6), Inches(3.0), col_widths=[2.2, 3, 3], size=14)
bullets(s, [
    "Saved every epoch: epoch_XX / best / last weights, train_state.json, history.csv → auto-resume on restart",
    "All AWS resources deleted afterwards (0 jobs, 0 buckets)",
    "Local M5 Air estimate for the same work: ~6–9 h",
], Inches(0.5), Inches(4.95), Inches(6.6), Inches(2.0), size=14)
bullets(s, [
    "12/12 unit tests pass (tests/test_m4.py):",
    ("zero patient overlap across 233 patients", 1),
    ("every augmented file maps to a training patient", 1),
    ("focal(γ=0, α=1) == cross-entropy", 1),
    ("metrics match scikit-learn; bootstrap CI narrows with n", 1),
    ("pHash robust to JPEG / resize", 1),
    ("same seed → identical first-epoch loss (CPU)", 1),
    "config.json per run: args, seed, data SHA-256, library versions, instance",
    ("GPU kernels are not bit-exact → reported as a limitation", 0, RED),
], Inches(7.4), Inches(1.7), Inches(5.6), Inches(5.2), size=15)

# ---------------------------------------------------------------- 7. Training curves
s = add_slide("Training dynamics", 7, "CE peaks early; CB-Focal keeps improving after the LR drop")
picture(s, FIG / "training_curves.png", Inches(0.4), Inches(1.6), width=Inches(12.5))
bullets(s, [
    "Run A: train accuracy > 99.6% by epoch 6; patient-val macro-F1 peaks at epoch 4 (93.6%), then declines → early stop",
    "Run B: patient-val macro-F1 rises to 94.6% at epoch 14 (val meningioma recall 96.3%) → early stop at epoch 21",
], Inches(0.5), Inches(5.95), Inches(12.3), Inches(1.0), size=15)

# ---------------------------------------------------------------- 8. Main results
s = add_slide("Results — Figshare patient-wise test (48 patients, 560 slices)", 8, "95% CIs from 1,000 patient-level bootstrap resamples")
table(s, [
    ["Model", "Accuracy", "Macro-F1", "Macro recall", "Meningioma recall", "Macro AUC"],
    ["M2 baseline (CE, leaky val)", "92.50 [87.6–96.1]", "91.56", "91.21", "78.47 [64.0–88.2]", "0.989"],
    ["Run A (CE, patient val)", "86.43 [78.0–93.0]", "84.93", "84.87", "66.67 [50.7–81.9]", "0.982"],
    ["Run B (CB-Focal, patient val)", "95.89 [93.1–98.1]", "95.50", "95.47", "90.28 [83.2–96.2]", "0.997"],
], Inches(0.5), Inches(1.65), Inches(12.3), Inches(1.9), col_widths=[3.4, 2.2, 1.4, 1.6, 2.4, 1.3], size=14, bold_rows=(3,))
picture(s, FIG / "per_class_recall_ci.png", Inches(0.4), Inches(3.75), width=Inches(7.9))
bullets(s, [
    ("Gain, not loss", 0, GREEN),
    "Meningioma recall: +23.6 pp vs Run A, +11.8 pp vs M2 (130/144 vs 96/144)",
    "Accuracy +9.5 pp vs Run A; pituitary stays 100%, glioma 96.1%",
    "Flip analysis: B fixes 59 slices A got wrong, breaks 6 → net +53",
    ("Caveat: single seed; Run A < M2, so the conservative gain is B vs M2", 0, RED),
], Inches(8.5), Inches(3.75), Inches(4.6), Inches(3.2), size=14)

# ---------------------------------------------------------------- 9. External validation
s = add_slide("H4 — external test on BRISC 2025", 9, "Glioma / meningioma / pituitary test slices; no_tumor excluded; evaluation only")
table(s, [
    ["pHash dedupe vs 3,064 Figshare slices", "Raw", "Dropped (dH ≤ 5)", "Kept"],
    ["Glioma", "254", "64 (25.2%)", "190"],
    ["Meningioma", "306", "27 (8.8%)", "279"],
    ["Pituitary", "300", "67 (22.3%)", "233"],
    ["Total", "860", "158 (18.4%)", "702"],
], Inches(0.5), Inches(1.65), Inches(6.2), Inches(2.4), col_widths=[3.4, 1, 2, 1], size=14)
bullets(s, [
    "124 of the 158 were exact hash matches (dH = 0) → BRISC re-uses Figshare slices",
    "Without the dedupe, the 'external' test would leak training data",
    "No patient IDs in BRISC → image-level CIs only",
], Inches(0.5), Inches(4.25), Inches(6.2), Inches(2.6), size=15)
table(s, [
    ["Model (702 slices)", "Accuracy", "Meningioma recall", "Macro AUC"],
    ["M2 baseline", "92.88", "83.87", "0.996"],
    ["Run A (CE)", "88.32 [85.9–90.7]", "73.12 [67.7–78.3]", "0.992"],
    ["Run B (CB-Focal)", "95.87 [94.3–97.3]", "92.11 [89.2–95.4]", "0.999"],
], Inches(7.0), Inches(1.65), Inches(5.9), Inches(2.0), col_widths=[2.4, 2.2, 2.2, 1.2], size=13, bold_rows=(3,))
bullets(s, [
    "Meningioma gain transfers: 73.1% → 92.1%",
    "Run A's main error on BRISC: 51/279 meningioma → pituitary (Run B: 13)",
    ("Predicted a 10–25 pp drop → did NOT happen (≈0 pp)", 0, RED),
    ("Likely: residual near-duplicates beyond dH = 5 and similar acquisition", 1),
], Inches(7.0), Inches(3.9), Inches(5.9), Inches(3.0), size=15)

# ---------------------------------------------------------------- 10. Predictions vs reality
s = add_slide("Predictions vs reality", 10, "Where we were right, wrong, and why")
table(s, [
    ["Hypothesis", "Predicted", "Observed", "Verdict"],
    ["H2 meningioma recall", "+3 to +8 pp", "+23.6 pp vs A / +11.8 pp vs M2", "Supported (bigger than expected)"],
    ["H2 accuracy", "±1 pp, may dip", "+9.5 pp vs A / +3.4 pp vs M2", "Wrong direction (better)"],
    ["H2 macro AUC", "stable ±0.01", "0.982 → 0.997", "Slightly better than predicted"],
    ["H1 patient val ≈ test", "≤ ~3 pp gap", "Run B 0.9 pp; Run A 8.7 pp", "Partly (B yes, A no)"],
    ["H4 BRISC drop", "−10 to −25 pp", "≈ 0 pp (95.9 vs 95.9)", "Falsified"],
    ["H4 meningioma worst", "yes", "yes, for all three models", "Supported"],
    ["H3 determinism (CPU)", "identical loss", "identical loss", "Supported"],
], Inches(0.5), Inches(1.65), Inches(12.3), Inches(4.0), col_widths=[2.8, 2.2, 4.2, 3.1], size=14)
bullets(s, [
    "Why the gain beat our prediction: CE saturates fast (loss → 0 by epoch 6), so honest early stopping froze Run A at epoch 4. Focal loss keeps gradients on hard boundary slices, so learning continued to epoch 14",
], Inches(0.5), Inches(5.85), Inches(12.3), Inches(1.1), size=15)

# ---------------------------------------------------------------- 11. Failure analysis
s = add_slide("M5 failure analysis", 11, "Which classes get confused, which patients fail, and where the model looks")
picture(s, FIG / "confusion_comparison.png", Inches(0.3), Inches(1.55), height=Inches(5.4))
picture(s, FIG / "failure_gallery_gradcam.png", Inches(6.6), Inches(1.55), height=Inches(3.6))
bullets(s, [
    "Main error: meningioma → pituitary (A: 44, B: 13 on Figshare); parasellar meningiomas sit next to the sella",
    "Run B errors cluster: 23 errors in 12 of 48 patients; 6 patients hold 17 of them",
    "Grad-CAM: on the remaining errors, attention often sits on bright sellar/vessel regions, not the lesion (qualitative only)",
], Inches(9.0), Inches(1.6), Inches(4.2), Inches(5.3), size=13)

# ---------------------------------------------------------------- 12. Limitations & conclusion
s = add_slide("Limitations & conclusion", 12)
bullets(s, [
    ("Limitations", 0, RED),
    "Single split and single seed (2-run budget); CIs only partly capture run-to-run variance",
    "Run A scored below M2 → part of the A→B gap may be CE instability",
    "Keras 3.4 (SageMaker) vs 3.15 (local): input normalisation differs vs M2 (not between A and B)",
    "GPU kernels are not bit-exact; 2D slices only; no_tumor excluded",
    "BRISC lacks patient IDs; pHash dH ≤ 5 may miss re-cropped duplicates",
    "No clinical claims",
], Inches(0.5), Inches(1.3), Inches(6.2), Inches(5.6), size=15)
bullets(s, [
    ("Conclusion", 0, GREEN),
    "Slice-wise splits hide a 5.7 pp accuracy / 15.9 pp meningioma gap",
    "Patient-grouped validation gives an honest model-selection signal (H1)",
    "CB-Focal (γ = 2): meningioma recall 90.3% on patients the model never saw, 92.1% on deduplicated BRISC (H2, H4)",
    "18.4% of BRISC test slices duplicate Figshare → always dedupe before 'external' validation",
    ("Next steps", 0, NAVY),
    "3 seeds × 5 patient folds; test H5 (scanner augmentation); 3D multi-site data",
], Inches(6.9), Inches(1.3), Inches(6.2), Inches(5.6), size=15)

prs.save(OUT)
print("Saved", OUT, "slides:", len(prs.slides))
