# Multi-class brain tumor classification (EfficientNets)

Reimplementation of Zulfiqar, Bajwa, and Mehmood, *Biomedical Signal Processing and Control* 84 (2023) 104777.

The pipeline follows the paper and the authors’ notebooks: Figshare CE-MRI → brain-contour crop → 80:20 split → class-wise train-only augmentation → EfficientNetB0–B4 fine-tuning.

Reported EfficientNetB2 test scores to reproduce: **98.86%** accuracy, **98.65%** precision, **98.77%** recall/sensitivity, **98.71%** F1.

## Setup

Use a TensorFlow-supported Python (3.10–3.11). The Figshare zips are already under `dataset/Figshare File 1512427/`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# 1. Unzip .mat files, crop brain contours, 80:20 split, augment train
PYTHONPATH=. python scripts/prepare_data.py

# 2. Train the paper’s best backbone (EfficientNetB2, 50 epochs)
PYTHONPATH=. python scripts/train.py --backbone B2

# Optional: B0–B4
PYTHONPATH=. python scripts/train.py --backbone B0

# 3. Evaluate a saved checkpoint
PYTHONPATH=. python scripts/evaluate.py --backbone B2

# 4. Grad-CAM on a test PNG
PYTHONPATH=. python scripts/visualize_gradcam.py --backbone B2 \
  --image data/augmented/test/glioma/<file>.png

# 5. Optional rigorous patient-wise split (keeps the paper split intact)
PYTHONPATH=. python scripts/prepare_data.py --skip-extract --patient-wise

# Train/evaluate against the patient-wise split
PYTHONPATH=. python scripts/train.py --backbone B2 \
  --data-root data/patient_augmented
PYTHONPATH=. python scripts/evaluate.py --backbone B2 \
  --test-dir data/patient_augmented/test

# Generate publication-style curves, confusion matrix, per-class scores,
# and a three-row Grad-CAM figure from the latest saved checkpoint
PYTHONPATH=. python scripts/generate_report_plots.py --backbone B2 \
  --test-dir data/patient_augmented/test
```

`--freeze-bn` turns on the paper’s §5.1 note (keep BatchNorm frozen). The public notebooks did not freeze BN; omit the flag to match those runs.

The preparation command recreates `data/split/` on each run so stale files
cannot change the deterministic 80:20 split. Augmentation still writes only
the generated training images and copies the untouched test split.
The `--patient-wise` command additionally creates `data/patient_split/` and
`data/patient_augmented/`; all slices from a patient stay in one partition.

## What is implemented

| Paper / notebook setting | This repo |
|---|---|
| Figshare T1-CE, 3064 slices, 3 classes | `scripts/prepare_data.py` |
| Label map 1/2/3 → meningioma/glioma/pituitary | `src/brain_tumor/config.py` |
| Training labels 0/1/2 = glioma/meningioma/pituitary | same order as `CLASS_NAMES` |
| Brain crop: gray, blur (5,5), thresh 45, erode/dilate ×2, extrema | `src/brain_tumor/crop.py` |
| Resize 240×240×3 | after crop |
| 80:20 stratified split, seed 3 | `split_train_test` |
| Patient-wise 80:20 split, seed 3 | `split_patient_train_test` |
| Train-only aug: rotate 10°, shift 0.1, h/v flip; glioma×3, meningioma×6, pituitary×5 written images | `src/brain_tumor/augment.py` |
| Shuffle seed 101 | `prepare_arrays` |
| EfficientNet ImageNet + GAP + Dropout 0.5 + softmax 3 | `build_efficientnet` |
| drop_connect_rate 0.2 | constructor arg (Keras default) |
| Adam 0.001, categorical cross-entropy, batch 32, 50 epochs, val_split 0.1 | `scripts/train.py` |
| ReduceLROnPlateau val_acc, factor 0.3, patience 5, min_delta 0.001 | `make_callbacks` |
| ModelCheckpoint on val_accuracy | `outputs/models/` |
| Accuracy, precision, recall/sensitivity, F1, specificity, confusion matrix | `src/brain_tumor/metrics.py` |
| Grad-CAM on `top_activation` | `scripts/visualize_gradcam.py` |

## Layout

```
dataset/Figshare File 1512427/   # original zips
data/raw|cropped|split|augmented|patient_split|patient_augmented/
src/brain_tumor/
scripts/prepare_data.py train.py evaluate.py visualize_gradcam.py
outputs/models|metrics|plots|logs/   # .h5 checkpoints (Keras 2.15)
```

## Assumptions (paper gaps)

- The default `data/augmented/` split is slice-wise for paper reproduction.
  Use `--patient-wise` and `data/patient_augmented/` for leakage-resistant
  evaluation.
- Crop at native resolution, then resize to 240 (notebooks), not resize-then-crop (paper wording).
- Keras `ImageDataGenerator.flow` writes `n+1` images per source file; originals are not copied into the augmented train set.
- No global TF seed; only split seed 3 and shuffle seed 101 from the notebooks.

## M4 & M5: Patient-Grouped Validation, CB-Focal Ablation, and External Evaluation (BRISC 2025)

All M4/M5 additions are additive and leave the frozen M2 checkpoint (`outputs/models/efficientnetb2.h5`) untouched.

```bash
# 1. Run M4 unit tests (zero patient overlap, focal loss, pHash, bootstrap CI, CPU determinism)
MPLCONFIGDIR=/tmp/.mpl KERAS_HOME=/tmp/.keras PYTHONPATH=. .venv/bin/python -c "
import tests.test_m4 as t
for name in sorted(dir(t)):
    if name.startswith('test_'): getattr(t, name)()
print('ALL TESTS PASSED')
"

# 2. Build patient-grouped train/val/test manifests + compressed dataset (data/m4/m4_data.npz)
PYTHONPATH=. .venv/bin/python scripts/build_m4_dataset.py

# 3. Prepare BRISC 2025 external test set (3 classes, crop+240px, 64-bit pHash dedupe vs Figshare)
PYTHONPATH=. .venv/bin/python scripts/prepare_brisc.py

# 4a. Launch Run A (CE on ml.g5.xlarge) and Run B (CB-Focal γ=2 on ml.g5.2xlarge) in parallel on SageMaker
#     Every epoch is saved to /opt/ml/checkpoints and synced to S3 with automatic resume support.
PYTHONPATH=. .venv/bin/python scripts/launch_sagemaker_m4.py --epochs 25 --seed 42

# 4b. Check SageMaker status and sync checkpoints/metrics down to runs/{A_ce,B_cbfocal}/
PYTHONPATH=. .venv/bin/python scripts/sync_sagemaker_m4.py --sync --exclude-epoch-weights

# 4c. Or run locally with per-epoch checkpointing and automatic resume:
PYTHONPATH=. .venv/bin/python scripts/train_m4.py --loss ce    --run-name A_ce      --epochs 25 --seed 42
PYTHONPATH=. .venv/bin/python scripts/train_m4.py --loss focal --run-name B_cbfocal --epochs 25 --seed 42 --gamma 2.0

# 5. Evaluate Frozen M2, Run A, and Run B on Figshare patient-wise test and BRISC deduplicated test
for tag_model in "frozen_m2:outputs/models/efficientnetb2.h5" "A_ce:runs/A_ce" "B_cbfocal:runs/B_cbfocal"; do
  tag="${tag_model%%:*}"; model="${tag_model#*:}"
  for ds in figshare_test brisc; do
    PYTHONPATH=. .venv/bin/python scripts/eval_m4.py --model "$model" --dataset "$ds" --tag "$tag"
  done
done

# 6. Generate M5 ablation tables, bootstrap CIs, flip analysis, patient error concentration, and Grad-CAM gallery
PYTHONPATH=. .venv/bin/python scripts/failure_analysis.py
```
