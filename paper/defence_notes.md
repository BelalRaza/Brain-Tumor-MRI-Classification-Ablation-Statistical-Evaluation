# M5 Oral Defence Preparation Notes

## 1. 60-Second Elevator Pitch
- **Base paper reproduced:** Zulfiqar et al. (*Biomedical Signal Processing and Control*, 2023) — EfficientNetB2 transfer learning on the Figshare CE-MRI 3-class brain tumor dataset (3,064 slices from 233 patients: glioma, meningioma, pituitary).
- **Reproduction & protocol finding (M2 → H1):** Under the authors' slice-wise 80:20 split, our EfficientNetB2 reached **98.21% accuracy / 97.77% macro-recall** (within 0.65 pp of the paper's 98.86%). However, when we enforced a **patient-grouped split** (zero patient overlap across splits), test accuracy dropped by **5.71 pp to 92.50%** and **meningioma recall fell by 15.90 pp (from 94.37% to 78.47%)**, while validation accuracy on the paper's `validation_split=0.1` over augmented training slices stayed artificially high at **99.44%**.
- **Protocol fix (H1 + H3):** We carved a **patient-grouped validation set** (26 patients, 359 slices) out of the training patients, selected checkpoints on patient-grouped validation macro-F1 (tie-break meningioma recall), and logged deterministic seeds + SHA-256 manifest hashes.
- **Hypothesis tested (H2 — our ONE experimental variable):** Holding split, seed, architecture, optimizer, and augmented training files identical, we replaced categorical cross-entropy (Run A) with **class-balanced focal loss** ($\gamma=2.0$, effective-number $\alpha_c$ computed from pre-augmentation class counts: $[0.744, 1.243, 1.012]$ for glioma, meningioma, pituitary; Run B).
  - **Result on Figshare patient-wise test ($n=560$, 48 patients):** Meningioma recall rose from **66.67% (Run A)** and **78.47% (M2 baseline)** to **90.28% (95% CI 83.25–96.18%)**; overall accuracy reached **95.89% (95% CI 93.05–98.14%)**, macro-F1 **95.50%**, and macro OvR AUC **0.9974** (fixing 59 errors of Run A while breaking only 6).
- **External validation (H4):** We tested all models zero-shot on **BRISC 2025** (3 overlapping tumor classes). Using a 64-bit DCT perceptual hash (pHash, Hamming distance $\le 5$), we discovered and removed **158 near-duplicate slices (18.4%)** out of 860 BRISC test slices that overlapped with Figshare, leaving **702 clean external slices**. On those 702 deduplicated slices, Run B achieved **95.87% accuracy** and **92.11% meningioma recall** (vs 73.12% for Run A and 83.87% for M2).

---

## 2. Likely Examiner Questions & Answers

### Q1: Why did your slice-wise reproduction get 98.21% vs the paper's 98.86%, and why did patient-wise drop to 92.50%?
- **Slice-wise gap (0.65 pp):** Expected stochastic variation from random initialization, hardware kernels, and exact `ImageDataGenerator` realizations; our 98.21% is well within standard reproduction tolerance.
- **Patient-wise drop (98.21% → 92.50%, meningioma 94.37% → 78.47%):** Figshare has 3,064 slices across only 233 patients (~13 slices/patient). Adjacent axial/coronal/sagittal slices of the same patient share skull geometry, contrast timing, and tumor appearance. Slice-wise splitting places near-identical neighbor slices in both train and test, measuring *patient memorization* rather than cross-patient generalization.

### Q2: Why did you compute class-balanced weights ($\alpha_c$) from *pre-augmentation* counts instead of post-augmentation counts?
- The paper's offline augmentation (`aug_factors = {glioma: 2, meningioma: 5, pituitary: 4}`, writing $n+1$ images per source slice) already equalizes file counts on disk (~3,072 glioma, ~2,903 meningioma, ~3,185 pituitary).
- However, 2,903 augmented meningioma slices come from only **484 unique source slices (56 patients)**, whereas glioma has **1,024 unique source slices (61 patients)**. Effective diversity is still imbalanced by $>2\times$. Computing Cui et al. effective-number weights ($\beta=0.999$) on the pre-augmentation counts gives $\alpha = [0.744, 1.243, 1.012]$ (normalised to mean 1), giving meningioma $1.67\times$ the gradient weight of glioma while $(1-p_t)^2$ focuses gradients on hard boundary slices.

### Q3: Why did you only run 2 training jobs instead of a full multi-fold grid?
- To isolate a **single experimental variable** (loss function: CE vs CB-Focal) under identical conditions on cloud GPU hardware (`ml.g5` NVIDIA A10G) within budget, and to pair every point estimate with **1,000-iteration patient-grouped bootstrap 95% confidence intervals** (resampling the 48 test patients rather than treating correlated slices as independent). We explicitly report single-split / single-seed evaluation as a study limitation.

### Q4: Why did you run pHash deduplication on BRISC 2025?
- Public brain MRI benchmarks frequently re-package slices from older collections (especially Figshare CE-MRI). Without deduplication, "external validation" is contaminated by test leakage. Our 64-bit DCT pHash check against all 3,064 Figshare slices caught **158 near-duplicates (18.4% of the 860 3-class BRISC test set)** at Hamming distance $\le 5$. Removing them makes the external test genuinely independent at the image level.

### Q5: Why are there no clinical claims in your paper?
- Benchmark 2D slice classifiers operate on pre-selected tumor slices with brain-contour cropping, omit clinical history and multi-parametric 3D volumes (T1, T1c, T2, FLAIR), and exclude normal/non-tumor or rare pathologies in the 3-class setting. High benchmark AUC or recall does not translate to clinical diagnostic validity.
