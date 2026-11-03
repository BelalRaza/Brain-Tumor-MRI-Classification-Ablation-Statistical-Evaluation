# M4 Ablation & External Validation Results

| Dataset | Run | Checkpoint Selection | Loss | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro-Recall (95% CI) | Meningioma Recall (95% CI) | Glioma Recall | Pituitary Recall | Macro OvR AUC (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| Figshare Test (patient-wise, n=560, 48 pts) | M2 Frozen Baseline | Leaky slice val (0.1) | CE | 92.50% [87.61, 96.14] | 91.56% [85.68, 95.17] | 91.21% [86.07, 94.90] | **78.47% [63.95, 88.16]** | 95.69% | 99.46% | 0.9890 [0.9775, 0.9970] |
| Figshare Test (patient-wise, n=560, 48 pts) | Run A (H1+H3 Baseline) | Patient-grouped val | CE | 86.43% [78.05, 93.01] | 84.93% [76.31, 91.13] | 84.87% [78.22, 90.89] | **66.67% [50.72, 81.89]** | 87.93% | 100.00% | 0.9819 [0.9639, 0.9941] |
| Figshare Test (patient-wise, n=560, 48 pts) | Run B (H2 Hypothesis) | Patient-grouped val | CB-Focal (γ=2) | 95.89% [93.05, 98.14] | 95.50% [92.29, 97.78] | 95.47% [92.71, 97.83] | **90.28% [83.25, 96.18]** | 96.12% | 100.00% | 0.9974 [0.9938, 0.9995] |
| BRISC External (dedup, n=702, image-level) | M2 Frozen Baseline | Leaky slice val (0.1) | CE | 92.88% [90.88, 94.59] | 92.90% [90.83, 94.60] | 93.75% [92.07, 95.32] | **83.87% [79.64, 88.11]** | 97.37% | 100.00% | 0.9955 [0.9923, 0.9978] |
| BRISC External (dedup, n=702, image-level) | Run A (H1+H3 Baseline) | Patient-grouped val | CE | 88.32% [85.90, 90.74] | 88.53% [86.08, 90.79] | 89.84% [87.70, 91.78] | **73.12% [67.68, 78.25]** | 96.84% | 99.57% | 0.9919 [0.9878, 0.9952] |
| BRISC External (dedup, n=702, image-level) | Run B (H2 Hypothesis) | Patient-grouped val | CB-Focal (γ=2) | 95.87% [94.30, 97.29] | 95.92% [94.42, 97.33] | 96.24% [94.84, 97.51] | **92.11% [89.16, 95.44]** | 97.89% | 98.71% | 0.9986 [0.9976, 0.9994] |
