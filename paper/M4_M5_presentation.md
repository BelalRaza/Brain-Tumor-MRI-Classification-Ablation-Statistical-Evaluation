# M4 & M5 Research Presentation (12 Slides)

**Generated PowerPoint Deck:** [M4_M5_presentation.pptx](file:///Users/vaishnavverma/Downloads/cv_research/implementation/paper/M4_M5_presentation.pptx)  
**Generator Script:** [make_presentation.py](file:///Users/vaishnavverma/Downloads/cv_research/implementation/paper/make_presentation.py)

---

## Slide 1 — Title & Project Overview
**Title:** Patient-Aware Evaluation and Class-Balanced Focal Loss for Brain Tumor CE-MRI Classification  
**Subtitle:** Milestones M4 (Implementation & Ablations) & M5 (Failure Analysis, Paper & Defence) — Reproduction & Extension of Talukder et al. (2023)

- **Target Paper:** Talukder et al., *IEEE JBHI* 2023 — EfficientNetB2 transfer learning on Figshare CE-MRI ($3{,}064$ slices, 233 patients)
- **Classes:** Glioma ($1{,}426$ slices / 89 patients), Meningioma ($708$ slices / 82 patients), Pituitary ($930$ slices / 62 patients)
- **Milestones Completed:**
  - **M4:** Pre-registered hypotheses, patient-grouped splits, CB-Focal loss ($\gamma=2.0$), AWS SageMaker GPU runs (`A_ce`, `B_cbfocal`), pHash-deduplicated BRISC 2025 external evaluation
  - **M5:** Patient-level error concentration, slice flip analysis, Grad-CAM failure modes, IEEE conference paper (`main.tex`), oral defence pack

---

## Slide 2 — Problem Statement: Why 99.64% Was an Illusion
**Subtitle:** Auditing Data Leakage and Minority-Class Sensitivity Collapse (M1–M3 Recap)

| Split Regime | Overall Accuracy | Macro-F1 | Meningioma Recall (Minority) |
|---|---|---|---|
| **Paper Reported (Talukder et al., 2023)** | 99.64% | 99.60% | ~99.3% |
| **Our M2 Frozen Baseline (Slice-Wise Leaky Split)** | 98.21% | 98.08% | **94.37%** |
| **Our M2 Frozen Baseline (Patient-Wise Disjoint Split)** | **92.50%** | **91.56%** | **78.47%** (113 / 144) |
| **Leakage Inflation Gap ($\Delta$)** | **$-5.71$ pp** | **$-6.52$ pp** | **$-15.90$ pp** |

- **Root Cause 1 (Data Leakage):** Random slice-level splitting places adjacent 2D slices from the same 3D patient volume into both train and test sets — memorizing patient anatomy instead of tumor morphology.
- **Root Cause 2 (Class Imbalance & Boundary Overlap):** Meningioma ($708$ slices) is half as frequent as glioma ($1{,}426$) and shares extrasellar/sellar borders with pituitary tumors.

---

## Slide 3 — Pre-Registered Hypotheses & M4 Experimental Design
**Subtitle:** Falsifiable Predictions Locked Before Training (`results/predictions.md`)

- **H1 — Loss Function under Patient-Grouped Splits (Run B vs. Run A):**
  - *Hypothesis:* Replacing Cross-Entropy (CE) with Class-Balanced Focal Loss ($\gamma=2.0$, effective-number weights $\alpha=[0.744, 1.243, 1.012]$) will increase patient-wise meningioma recall by $\ge 2.5$ pp without reducing overall accuracy by $> 0.5$ pp.
- **H2 — Patient-Grouped Validation Protocol (Run A vs. M2):**
  - *Hypothesis:* Enforcing patient-disjoint validation (`GroupKFold` on `patient_id`) removes anatomical shortcut selection and changes early-stopping dynamics compared to random 10% slice validation.
- **H3 — Cross-Dataset External Generalization (BRISC 2025):**
  - *Hypothesis:* After removing cross-benchmark duplicates via 64-bit DCT perceptual hashing ($d_H \le 5$), CB-Focal will retain higher meningioma sensitivity than CE on unseen external scans.
- **H5 — Dropped Before Training:**
  - Removed patient majority-vote aggregation before training so every primary metric reflects true slice-level performance on patient-disjoint cohorts with patient-clustered bootstrap 95% CIs.

---

## Slide 4 — M4 Methodology & Engineering Architecture
**Subtitle:** Architecture, Mathematical Formulation, and Reproducibility (`test_m4.py`: 12/12 Passing)

- **Backbone & Reconstruction Head (`model.py`):**
  - ImageNet-pretrained `EfficientNetB2` ($260 \times 260 \times 3$, $8.37\text{M}$ params; $7.70\text{M}$ trainable)
  - Custom head: `GlobalAveragePooling2D` $\to$ `BatchNorm` $\to$ `Dense(256, ReLU)` $\to$ `Dropout(0.45)` $\to$ `Dense(128, ReLU)` $\to$ `Dropout(0.30)` $\to$ `Dense(3, Softmax)`
- **Class-Balanced Focal Loss (`losses.py`, Cui et al. CVPR 2019 + Lin et al. ICCV 2017):**
  $$\mathcal{L}_{\text{CB-Focal}}(y, \mathbf{p}) = -\sum_{c=1}^{C} \alpha_c \, y_c \, (1 - p_c)^{\gamma} \log(\text{clip}(p_c, 10^{-7}, 1)), \quad \alpha_c \propto \frac{1 - \beta}{1 - \beta^{n_c}}$$
  - With $\beta=0.9999$ on $N_{\text{train}} = [934, 456, 606]$: normalized $\alpha = [0.7444, 1.2433, 1.0123]$, $\gamma = 2.0$
- **Strict Patient-Disjoint Split (`splits.py`):**
  - 185 train/val patients split via 8-fold `GroupKFold`: 161 train patients ($1{,}996$ slices), 24 val patients ($282$ slices), 48 held-out test patients ($560$ slices) — **0 patient overlap verified**

---

## Slide 5 — AWS SageMaker Training & Optimization Dynamics
**Subtitle:** Parallel GPU Training on `ml.g5.xlarge` / `ml.g5.2xlarge` (NVIDIA A10G 24 GB)

![Training Curves](file:///Users/vaishnavverma/Downloads/cv_research/implementation/results/figures/training_curves.png)

- **Run A (Cross-Entropy + Patient-Grouped Val):**
  - Completed in `767 s` on `ml.g5.xlarge`
  - Best `val_macro_f1 = 0.8712` at **Epoch 4**; early-stopped at Epoch 11 (patience 7)
  - Standard CE rapidly overfits to dominant glioma/pituitary features; validation loss diverges from `0.58` to `2.08`
- **Run B (CB-Focal $\gamma=2.0$ + Patient-Grouped Val):**
  - Completed in `974 s` on `ml.g5.2xlarge`
  - Best `val_macro_f1 = 0.9032` at **Epoch 14**; early-stopped at Epoch 21
  - Modulating factor $(1-p_c)^2$ suppresses easy background gradients, preventing early overfitting and stabilizing validation loss (`0.0888`)

---

## Slide 6 — Figshare Patient-Wise Test Results ($n=560$ Slices, 48 Patients)
**Subtitle:** Evaluated on Patient-Disjoint Test Set with 1,000-Iteration Patient-Clustered Bootstrap 95% CIs

| Model Variant | Overall Acc [95% CI] | Macro-F1 [95% CI] | Macro-Recall | Glioma Recall | Meningioma Recall [95% CI] | Pituitary Recall | Macro AUC |
|---|---|---|---|---|---|---|---|
| **M2 Frozen Baseline** *(CE, leaky slice val)* | 92.50% `[87.6, 96.1]` | 91.56% `[86.0, 95.6]` | 91.21% | 97.48% | 78.47% `[64.0, 88.2]` | 97.67% | 0.9890 |
| **Run A (`A_ce`)** *(CE, patient val)* | 86.43% `[78.1, 93.0]` | 84.93% `[75.9, 92.2]` | 84.87% | 97.48% | 66.67% `[50.7, 81.9]` | 90.48% | 0.9819 |
| **Run B (`B_cbfocal`)** *(CB-Focal, patient val)* | **95.89%** `[93.1, 98.1]` | **95.50%** `[92.4, 98.0]` | **95.47%** | **97.90%** | **90.28%** `[83.3, 96.2]` | **98.21%** | **0.9974** |

- **Why Run A Dropped Below M2 ($86.43\%$ vs. $92.50\%$):**
  - M2's random 10% slice validation leaked patient anatomy into checkpoint selection (peaking at Epoch 13). Under honest patient-grouped validation, standard CE stopped at Epoch 4 before resolving the meningioma/pituitary boundary.
- **How Run B Solved Both ($95.89\%$ Acc, $90.28\%$ Meningioma Recall):**
  - CB-Focal boosted meningioma recall by **$+23.61\text{ pp}$ over Run A** ($96 \to 130 / 144$ slices) and **$+11.81\text{ pp}$ over M2**, confirming **H1**.

---

## Slide 7 — Per-Class Sensitivity & Confusion Matrix Comparison
**Subtitle:** Eliminating Meningioma $\to$ Pituitary Misclassification Without Sacrificing Majority Classes

![Per-Class Recall](file:///Users/vaishnavverma/Downloads/cv_research/implementation/results/figures/per_class_recall_ci.png)
![Confusion Matrices](file:///Users/vaishnavverma/Downloads/cv_research/implementation/results/figures/confusion_comparison.png)

- **Meningioma Error Reduction (144 Test Slices):**
  - **M2 Baseline:** 113 correct, 20 predicted as glioma, 11 predicted as pituitary (`78.47%` recall)
  - **Run A (CE):** 96 correct, 4 predicted as glioma, **44 predicted as pituitary** (`66.67%` recall)
  - **Run B (CB-Focal):** **130 correct**, 1 predicted as glioma, **only 13 predicted as pituitary** (`90.28%` recall)

---

## Slide 8 — External Validation on BRISC 2025 & pHash Deduplication
**Subtitle:** Detecting Cross-Benchmark Contamination & Testing Zero-Shot Generalization

![ROC Curves](file:///Users/vaishnavverma/Downloads/cv_research/implementation/results/figures/roc_curves.png)

- **pHash Deduplication Audit (`phash.py`, 64-bit DCT Hash, $d_H \le 5$):**
  - Audited 860 3-class BRISC test slices against all $3{,}064$ Figshare slices
  - Discovered and removed **158 near-duplicate slices (18.4% contamination!)**: 124 exact ($d_H=0$), 32 ($d_H=2$), 2 ($d_H=4$) -> **702 clean external slices**

| Model Variant | BRISC Clean Acc ($n=702$) | Macro-F1 | Glioma Recall | Meningioma Recall [95% CI] | Pituitary Recall | Macro AUC |
|---|---|---|---|---|---|---|
| **M2 Frozen Baseline** | 92.88% `[90.9, 94.6]` | 92.95% | 94.88% | 83.87% `[79.7, 88.2]` | 100.00% | 0.9949 |
| **Run A (`A_ce`)** | 88.32% `[85.9, 90.7]` | 88.41% | 96.46% | 73.12% `[67.7, 78.3]` | 96.45% | 0.9854 |
| **Run B (`B_cbfocal`)** | **95.87%** `[94.3, 97.3]` | **95.92%** | **96.46%** | **92.11%** `[88.8, 95.1]` | **99.41%** | **0.9986** |

- **Zero Generalization Gap:** Run B achieves **95.87% accuracy** on deduplicated BRISC vs. **95.89%** on Figshare (**H3 confirmed**).

---

## Slide 9 — M5 Failure Analysis: Slice Flips & Patient Concentration
**Subtitle:** Where Do the Remaining 23 Errors Occur Across the 48 Held-Out Test Patients?

- **Slice-Level Flip Analysis (Run A $\to$ Run B on $n=560$ Test Slices):**
  - Both Correct: **478 slices** (85.36%) | Both Wrong: **17 slices** (3.04%)
  - **Fixed by CB-Focal (A wrong $\to$ B right): 59 slices** (10.54%) — primarily parasellar meningiomas previously misclassified as pituitary
  - **Broken by CB-Focal (A right $\to$ B wrong): 6 slices** (1.07%) — net gain of **$+53$ slices ($+9.46\text{ pp}$)**
- **Patient-Level Error Concentration (48 Test Patients in Run B):**
  - **36 / 48 patients (75.0%) achieved 100% slice-level accuracy (0 errors)**
  - All 23 errors in Run B are concentrated in just **12 patients**, with **17 / 23 errors (73.9%) occurring in 6 borderline patients**:

| Patient ID | True Class | Total Slices | Run A Errors | Run B Errors | Primary Anatomical Failure Mechanism |
|---|---|---|---|---|---|
| `1034` | Meningioma | 5 | 4 (80.0%) | **5 (100.0%)** | Midline sellar/suprasellar mass mimicking pituitary adenoma |
| `1826` | Meningioma | 9 | 7 (77.8%) | **4 (44.4%)** | Parasellar dural lesion bordering pituitary fossa |
| `1259` | Meningioma | 10 | 8 (80.0%) | **2 (20.0%)** | Low-contrast peripheral slices at cranial base |
| `1129` | Glioma | 12 | 2 (16.7%) | **2 (16.7%)** | Rim-enhancing frontal glioma resembling dural meningioma |
| `1452` | Pituitary | 11 | 2 (18.2%) | **2 (18.2%)** | Superiorly extended macroadenoma mimicking meningioma |
| `970` | Meningioma | 7 | 7 (100.0%) | **2 (28.6%)** | Basal meningioma; CB-Focal rescued 5 of 7 slices |

---

## Slide 10 — M5 Qualitative Failure Analysis: Grad-CAM Saliency
**Subtitle:** Visualising Conv-Layer Attention (`top_activation`) on High-Confidence Misclassifications

![Grad-CAM Failure Gallery](file:///Users/vaishnavverma/Downloads/cv_research/implementation/results/figures/failure_gallery_gradcam.png)

- **Dominant Failure Mode 1 — Sellar/Suprasellar Meningioma $\to$ Pituitary (13 / 14 Meningioma Errors):**
  - In patients `1034` and `1826`, meningiomas arise near the tuberculum sellae / sphenoid wing. Grad-CAM reveals the network attends to the **sellar anatomical location** and bright contrast enhancement rather than the dural tail Attachment.
- **Dominant Failure Mode 2 — Rim-Enhancing Glioma $\to$ Meningioma (5 / 5 Glioma Errors):**
  - Cortical high-grade gliomas abutting the meninges draw focal activation along the outer cortical rim, mimicking extra-axial dural masses.
- **Takeaway:** Remaining errors are anatomically ambiguous 2D slices lacking 3D volumetric continuity.

---

## Slide 11 — Limitations, Ethical Guardrails & Future Work
**Subtitle:** Honest Scientific Reporting in Compliance with Medical AI Evaluation Standards

- **Methodological & Dataset Limitations:**
  1. **2D Slice Independence vs. 3D Volumetric Context:** Evaluating slices independently loses inter-slice continuity across axial, coronal, and sagittal planes that neuroradiologists use to trace dural tails or stalk displacement.
  2. **BRISC Missing Patient Identifiers:** BRISC 2025 distributes individual `.jpg` slices without `patient_id` metadata; while pHash ($d_H \le 5$) eliminates Figshare overlap, external CIs are slice-bootstrapped rather than patient-clustered.
  3. **3-Class Tumor Scope:** Restricted to glioma, meningioma, and pituitary tumors on T1-weighted CE-MRI; excludes healthy (`no_tumor`) brains, metastases, and multi-parametric sequences (T2, FLAIR, DWI).
- **Strict Non-Clinical Disclaimer:**
  - **No clinical deployment claims are made.** High retrospective benchmark accuracy does not equate to clinical readiness or prospective diagnostic validity across multi-vendor 1.5T/3T MRI scanners.

---

## Slide 12 — Summary of Contributions & Deliverable Readiness
**Subtitle:** Complete M4 & M5 Package Verified and Ready for Submission / Oral Defence

- **Key Scientific Takeaways:**
  1. **Exposed the Leakage Illusion:** Showed slice-wise splitting inflates accuracy by $+5.71\text{ pp}$ and meningioma sensitivity by $+15.90\text{ pp}$.
  2. **Diagnosed Patient-Grouped Validation Instability:** Standard CE early-stops prematurely at Epoch 4 (`86.43%` acc) due to majority-class overfitting.
  3. **Resolved Minority Sensitivity via CB-Focal Loss:** Achieved **`95.89%` accuracy** and **`90.28%` meningioma recall** ($+23.61\text{ pp}$ over Run A, $+11.81\text{ pp}$ over M2).
  4. **Contamination-Free External Generalization:** Discovered **18.4% duplicate leakage** in BRISC 2025 via pHash and proved **`95.87%` clean external accuracy**.
- **Completed Deliverables & Cloud Hygiene:**
  - **Code & Tests:** Modular `brain_tumor` package + `12/12` passing unit tests (`tests/test_m4.py`)
  - **Paper & Presentation:** 6-page IEEE paper (`paper/main.tex`), `overleaf_bundle.zip`, `defence_notes.md`, `M4_M5_presentation.pptx`
  - **AWS Cost Safety:** All SageMaker jobs completed, weights synced locally, and S3 bucket (`s3://sagemaker-us-east-1-948127566077`) permanently deleted (`$0.00` ongoing cost).
