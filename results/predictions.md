# M4 Pre-registered Predictions

Written **2026-11-03, before any M4 training run was started**. Do not edit after the runs start;
the actual numbers are filled in only in `results/summary.md` / `results/ablation_table.md`.

## Setup being compared

| | Run A (baseline) | Run B (hypothesis H2) |
|---|---|---|
| Loss | categorical cross-entropy | class-balanced focal loss, gamma = 2, alpha from pre-augmentation train counts (effective number, beta = 0.999) |
| Everything else | EfficientNetB2 ImageNet, 240 px, Adam 1e-3, batch 32, max 25 epochs, seed 42, identical patient-wise train/val/test, identical augmented files, LR x0.3 on val-macro-F1 plateau (patience 3), early stop (patience 7) | identical |
| Checkpoint rule (H1) | best macro-F1 on a **patient-grouped** validation set (tie-break meningioma recall) | identical |

Protocol fixes applied to both runs (not experimental variables):
- **H1** patient-grouped validation for checkpoint selection (old pipeline used `validation_split=0.1` on augmented copies of training patients -> leaky val_accuracy 0.994 vs 0.925 test).
- **H3** global seeding, config + manifest hashes logged per run.
- **H4** evaluation-only external test on BRISC 2025 (glioma/meningioma/pituitary; no_tumor dropped; near-duplicates of Figshare removed by pHash).

## Prediction 1 — H2 on the Figshare patient-wise test set (Run B minus Run A)

| Metric | Predicted change | Reason |
|---|---|---|
| Meningioma recall | **+3 to +8 pp** | meningioma has the fewest source slices (564 vs 1194 glioma) and is the hardest class (78.5% recall in old baseline); focal + class alpha shift gradient towards it |
| Macro recall | +1 to +3 pp | gains concentrated in meningioma |
| Accuracy | within +/-1 pp (may dip) | some glioma/pituitary slices pushed towards meningioma |
| Macro one-vs-rest AUC | stable (+/-0.01) | focal loss changes calibration/threshold more than ranking |
| Glioma & pituitary precision | small drop possible | more predictions to meningioma -> meningioma precision may drop, not others; uncertain |

Falsification: if meningioma recall improves by < 1 pp or the bootstrap 95% CIs of the two runs overlap almost
completely, H2 is **not supported** at this sample size.

## Prediction 2 — H1 (checkpoint selection)

- Patient-grouped validation macro-F1 will be **far below** the old leaky val_accuracy (0.994), and close to
  the test score (within ~3 pp), i.e. it becomes an honest model-selection signal.

## Prediction 3 — H4 external validation on BRISC (both models)

| Metric | Prediction |
|---|---|
| Accuracy drop vs Figshare test | **10 to 25 pp** (different scanners/sites, JPEG, mixed planes, different crop) |
| Worst class | meningioma |
| Run B vs Run A on BRISC | meningioma recall gain partly carries over (smaller than on Figshare); overall gap to Figshare remains large |
| Near-duplicates removed | a non-zero number (BRISC is curated partly from public sources overlapping Figshare) |

## Prediction 4 — H3 (reproducibility)

- On CPU with op determinism, two runs with the same seed give **identical** first-epoch loss.
- On GPU (SageMaker) bit-exactness is not guaranteed; we report it as a limitation, not a result.
