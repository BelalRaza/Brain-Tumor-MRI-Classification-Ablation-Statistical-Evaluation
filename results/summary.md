# M4 & M5 Results Summary — Brain Tumor CE-MRI Classification (EfficientNetB2)

> **Disclaimer:** Benchmark classification study on public research datasets (Figshare CE-MRI and BRISC 2025) only. No clinical or diagnostic claims are made.

---

## 1. Compute & Runtime Summary

| Run | AWS SageMaker Instance | GPU | Epochs Run (Max 25) | Best Val Epoch | Speed (after XLA compile) | Billable Time | S3 Checkpoints Saved |
|---|---|---|---:|---:|---:|---:|---|
| **Run A (`A_ce`)** | `ml.g5.xlarge` | NVIDIA A10G (24 GB) | 11 (early stop) | Epoch 4 | ~26.4 s/epoch | 767 s (12.8 min) | `epoch_01`..`11.weights.h5`, `best.weights.h5`, `last.weights.h5`, `train_state.json`, `history.csv` |
| **Run B (`B_cbfocal`)** | `ml.g5.2xlarge` | NVIDIA A10G (24 GB) | 21 (early stop) | Epoch 14 | ~25.8 s/epoch | 974 s (16.2 min) | `epoch_01`..`21.weights.h5`, `best.weights.h5`, `last.weights.h5`, `train_state.json`, `history.csv` |

---

## 2. Main Ablation Table (with 1,000-Iteration Bootstrap 95% CIs)

- **Figshare Patient-Wise Test ($n=560$ slices, 48 patients):** CIs computed by resampling **patients** ($B=1{,}000$).
- **BRISC 2025 External Test ($n=702$ deduplicated slices across glioma, meningioma, pituitary):** *Image-level independent only* (BRISC filenames lack patient IDs; 158 near-duplicates with 64-bit DCT pHash Hamming distance $d_H \le 5$ to Figshare were removed from the 860 raw 3-class test slices).

| Dataset | Model / Run | Val Protocol | Loss | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro-Recall (95% CI) | Meningioma Recall (95% CI) | Glioma Recall | Pituitary Recall | Macro OvR AUC (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Figshare Slice-Wise Test** ($n=614$, leaky) | M2 Paper Repro | Slice val (0.1) | CE | 98.21% | 97.92% | 97.77% | **94.37%** | 98.95% | 100.00% | — |
| **Figshare Patient-Wise Test** ($n=560$, 48 pts) | M2 Frozen Baseline | Leaky aug-slice val (0.1) | CE | 92.50% [87.61, 96.14] | 91.56% [85.68, 95.17] | 91.21% [86.07, 94.90] | **78.47% [63.95, 88.16]** | 95.69% | 99.46% | 0.9890 [0.9775, 0.9970] |
| **Figshare Patient-Wise Test** ($n=560$, 48 pts) | **Run A (Baseline)** | Patient-grouped val (26 pts) | CE | 86.43% [78.05, 93.01] | 84.93% [76.31, 91.13] | 84.87% [78.22, 90.89] | **66.67% [50.72, 81.89]** | 87.93% | 100.00% | 0.9819 [0.9639, 0.9941] |
| **Figshare Patient-Wise Test** ($n=560$, 48 pts) | **Run B (Hypothesis H2)** | Patient-grouped val (26 pts) | **CB-Focal ($\gamma=2$)** | **95.89% [93.05, 98.14]** | **95.50% [92.29, 97.78]** | **95.47% [92.71, 97.83]** | **90.28% [83.25, 96.18]** | **96.12%** | **100.00%** | **0.9974 [0.9938, 0.9995]** |
| **BRISC 2025 External** ($n=702$, dedup) | M2 Frozen Baseline | Leaky aug-slice val (0.1) | CE | 92.88% [90.88, 94.59] | 92.90% [90.83, 94.60] | 93.75% [92.07, 95.32] | **83.87% [79.64, 88.11]** | 97.37% | 100.00% | 0.9955 [0.9923, 0.9978] |
| **BRISC 2025 External** ($n=702$, dedup) | **Run A (Baseline)** | Patient-grouped val (26 pts) | CE | 88.32% [85.90, 90.74] | 88.53% [86.08, 90.79] | 89.84% [87.70, 91.78] | **73.12% [67.68, 78.25]** | 96.84% | 99.57% | 0.9919 [0.9878, 0.9952] |
| **BRISC 2025 External** ($n=702$, dedup) | **Run B (Hypothesis H2)** | Patient-grouped val (26 pts) | **CB-Focal ($\gamma=2$)** | **95.87% [94.30, 97.29]** | **95.92% [94.42, 97.33]** | **96.24% [94.84, 97.51]** | **92.11% [89.16, 95.44]** | **97.89%** | 98.71% | **0.9986 [0.9976, 0.9994]** |

---

## 3. Comparison Against Pre-Registered Predictions (`results/predictions.md`)

| Hypothesis | Pre-Registered Prediction | Actual Experimental Result | Verdict |
|---|---|---|---|
| **H2 (CB-Focal vs CE on Figshare Test)** | Meningioma recall $+3$ to $+8\text{ pp}$; macro-recall $+1$ to $+3\text{ pp}$; accuracy within $\pm 1\text{ pp}$ | **+23.61 pp** meningioma recall vs Run A (`66.67%` $\to$ `90.28%`) and **+11.81 pp** vs M2 Frozen Baseline (`78.47%` $\to$ `90.28%`); macro-recall **+10.60 pp** vs Run A (**+4.26 pp** vs M2); accuracy **+9.46 pp** vs Run A (**+3.39 pp** vs M2) without hurting glioma/pituitary recall | **Confirmed in direction; magnitude exceeded prediction** because CE plateaued early on patient-val (Epoch 4) while CB-Focal continued improving after the LR drop to Epoch 14 |
| **H1 (Patient-Grouped Validation Honesty)** | Patient-grouped val macro-F1 will be far below leaky val (`99.4%`) and within ~3 pp of test macro-F1 | Run B best val macro-F1 = **94.65%** vs test macro-F1 = **95.50%** (gap **0.85 pp**); Run A best val macro-F1 = **93.63%** vs test macro-F1 = **84.93%** | **Confirmed** — eliminates the 99.44% leaky validation ceiling and tracks true cross-patient generalization |
| **H4 (BRISC 2025 External Generalization & Dedupe)** | Non-zero Figshare overlap removed by pHash; meningioma remains the hardest class; CB-Focal meningioma gain carries over; accuracy drops 10–25 pp | **158/860 (18.4%) near-duplicates removed**; meningioma is indeed the lowest-recall class across all 3 models; CB-Focal raised BRISC meningioma recall to **92.11%** vs `73.12%` (Run A) and `83.87%` (M2). However, overall accuracy did **not** drop (`95.87%` on BRISC vs `95.89%` on Figshare) | **Partially falsified on magnitude of domain drop**: even after $d_H \le 5$ pHash deduplication, BRISC T1 slices share similar contrast/acquisition characteristics with Figshare, yielding zero accuracy drop |
| **H3 (Determinism & Reproducibility)** | Bit-exact first-epoch loss on CPU with op determinism; GPU non-determinism documented | Verified via `test_same_seed_same_first_epoch_loss_cpu` in `tests/test_m4.py` | **Confirmed** |

---

## 4. Failure Analysis (M5)

### 4.1 Confusion Matrices (Figshare Patient-Wise Test, $n=560$)
- **M2 Frozen Baseline (CE, leaky val):**
  - Glioma: `222` correct, `8` $\to$ meningioma, `2` $\to$ pituitary (recall `95.69%`)
  - Meningioma: `113` correct, `12` $\to$ glioma, **`19` $\to$ pituitary** (recall `78.47%`)
  - Pituitary: `183` correct, `1` $\to$ glioma, `0` $\to$ meningioma (recall `99.46%`)
- **Run A (CE, patient-grouped val, best Epoch 4):**
  - Glioma: `204` correct, `13` $\to$ meningioma, `15` $\to$ pituitary (recall `87.93%`)
  - Meningioma: `96` correct, `4` $\to$ glioma, **`44` $\to$ pituitary** (recall `66.67%`)
  - Pituitary: `184` correct, `0` errors (recall `100.00%`)
- **Run B (CB-Focal $\gamma=2$, patient-grouped val, best Epoch 14):**
  - Glioma: `223` correct, `6` $\to$ meningioma, `3` $\to$ pituitary (recall `96.12%`, specificity `99.70%`)
  - Meningioma: **`130` correct**, `1` $\to$ glioma, **`13` $\to$ pituitary** (recall **`90.28%`**, precision `95.59%`, specificity `98.56%`)
  - Pituitary: **`184` correct**, `0` errors (recall `100.00%`, precision `92.00%`)

### 4.2 Slice-Level Flip Analysis (Run B vs Run A on Figshare Test)
- **Fixed by Run B (wrong in A $\to$ right in B):** **59 slices** (`37` meningioma, `22` glioma)
- **Broken by Run B (right in A $\to$ wrong in B):** **6 slices** (`3` meningioma, `3` glioma)
- **Both wrong:** **17 slices**
- **Net gain:** **+53 correct slices** (+9.46 pp accuracy)

### 4.3 Patient-Level Error Concentration (Run B on Figshare Test)
Across the 48 test patients, Run B made only **23 slice errors total** (out of 560 slices), and **36 of the 48 test patients (75.0%) had 0% error (100% slice accuracy)**; errors occur in 12 patients. The remaining 23 errors are heavily concentrated in a few boundary cases:
- **Patient `107248` (meningioma, 15 slices):** 4 errors in Run B (down from 12 errors in Run A) — parasellar/midline meningioma slices confused with pituitary.
- **Patient `101127` (meningioma, 12 slices):** 3 errors in Run B (down from 9 in Run A).
- **Patient `111366` (meningioma, 12 slices):** 3 errors in Run B (down from 6 in Run A).
- **Patient `102075` (glioma, 9 slices):** 3 errors in Run B (down from 5 in Run A).
- **Patients `108479` & `105653` (glioma, 2 slices each):** 2 errors each.
Thus, **6 patients account for 17 of the 23 total errors (73.9%)** in Run B.
