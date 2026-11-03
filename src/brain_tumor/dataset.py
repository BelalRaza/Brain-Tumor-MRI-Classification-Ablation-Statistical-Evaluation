from pathlib import Path

import cv2
import numpy as np
from sklearn.utils import shuffle as sklearn_shuffle
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from .config import CLASS_NAMES, IMAGE_SIZE, N_CLASSES, SHUFFLE_SEED


def load_split_directory(root: Path, image_size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        folder = root / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        files = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        for path in tqdm(files, desc=f"Load {root.name}/{class_name}"):
            image = cv2.imread(str(path))
            if image is None:
                continue
            if image.shape[0] != image_size or image.shape[1] != image_size:
                image = cv2.resize(image, (image_size, image_size))
            images.append(image)
            labels.append(class_index)
    x = np.asarray(images, dtype=np.uint8)
    y = np.asarray(labels, dtype=np.int32)
    return x, y


def prepare_arrays(
    train_dir: Path,
    test_dir: Path,
    shuffle_seed: int = SHUFFLE_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train = load_split_directory(train_dir)
    x_test, y_test = load_split_directory(test_dir)
    x_train, y_train = sklearn_shuffle(x_train, y_train, random_state=shuffle_seed)
    x_test, y_test = sklearn_shuffle(x_test, y_test, random_state=shuffle_seed)
    y_train_cat = to_categorical(y_train, num_classes=N_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=N_CLASSES)
    return x_train, y_train_cat, x_test, y_test_cat
