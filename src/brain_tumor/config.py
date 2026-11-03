from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Figshare CE-MRI: 1=meningioma, 2=glioma, 3=pituitary tumor
FIGSHARE_LABEL_TO_NAME = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary tumor",
}

# Paper encoding after split/augmentation: 0=glioma, 1=meningioma, 2=pituitary
CLASS_NAMES = ["glioma", "meningioma", "pituitary tumor"]

IMAGE_SIZE = 240
N_CLASSES = 3
CHANNELS = 3

# Table 3 / authors' Keras notebook
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3  # paper 0.001; Keras Adam default. Ignore the "(1e-2)" typo.
LOSS = "categorical_crossentropy"
DROPOUT_RATE = 0.5
DROP_CONNECT_RATE = 0.2
VALIDATION_SPLIT = 0.1
LR_FACTOR = 0.3
LR_PATIENCE = 5
LR_MIN_DELTA = 0.001

# Authors' Crop_Brain_Contours.ipynb
CROP_BLUR_KERNEL = (5, 5)
CROP_THRESH = 45
CROP_ERODE_ITERS = 2
CROP_DILATE_ITERS = 2

# Authors' Data_Augmentation.ipynb
# flow() uses `if i > n_generated_samples: break`, so n+1 images are written per source.
# That reproduces Table 2: 1140*3=3420, 566*6=3396 (~3395), 744*5=3720 (~3719).
AUG_FACTORS = {
    "glioma": 2,
    "meningioma": 5,
    "pituitary tumor": 4,
}
AUG_ROTATION = 10
AUG_WIDTH_SHIFT = 0.1
AUG_HEIGHT_SHIFT = 0.1
AUG_FILL_MODE = "nearest"

# split-folders notebook used seed=3; training notebook shuffled with 101
SPLIT_SEED = 3
SHUFFLE_SEED = 101
TEST_SIZE = 0.2

ZIP_DIR = ROOT / "dataset" / "Figshare File 1512427"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CROPPED_DIR = DATA_DIR / "cropped"
METADATA_PATH = CROPPED_DIR / "metadata.json"
SPLIT_DIR = DATA_DIR / "split"
AUGMENTED_DIR = DATA_DIR / "augmented"
PATIENT_SPLIT_DIR = DATA_DIR / "patient_split"
PATIENT_AUGMENTED_DIR = DATA_DIR / "patient_augmented"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

BACKBONES = ("B0", "B1", "B2", "B3", "B4")
DEFAULT_BACKBONE = "B2"
