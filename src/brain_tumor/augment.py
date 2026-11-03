from pathlib import Path

import cv2

from .config import (
    AUG_FACTORS,
    AUG_FILL_MODE,
    AUG_HEIGHT_SHIFT,
    AUG_ROTATION,
    AUG_WIDTH_SHIFT,
    CLASS_NAMES,
)


def make_augmenter():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        rotation_range=AUG_ROTATION,
        width_shift_range=AUG_WIDTH_SHIFT,
        height_shift_range=AUG_HEIGHT_SHIFT,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode=AUG_FILL_MODE,
    )


def augment_class_dir(src_dir: Path, dst_dir: Path, n_generated_samples: int) -> int:
    """Write n_generated_samples+1 augmented PNGs per source image (authors' loop)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    gen = make_augmenter()
    written = 0
    for image_path in sorted(src_dir.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = image.reshape((1,) + image.shape)
        save_prefix = f"aug_{image_path.stem}"
        i = 0
        for _ in gen.flow(
            x=image,
            batch_size=1,
            save_to_dir=str(dst_dir),
            save_prefix=save_prefix,
            save_format="png",
        ):
            i += 1
            written += 1
            if i > n_generated_samples:
                break
    return written


def augment_train_set(train_dir: Path, output_train_dir: Path) -> dict[str, int]:
    counts = {}
    for class_name in CLASS_NAMES:
        n = AUG_FACTORS[class_name]
        counts[class_name] = augment_class_dir(
            train_dir / class_name,
            output_train_dir / class_name,
            n_generated_samples=n,
        )
    return counts
