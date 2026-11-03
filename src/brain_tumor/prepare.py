import shutil
import zipfile
import json
from pathlib import Path

import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import (
    CLASS_NAMES,
    CROPPED_DIR,
    FIGSHARE_LABEL_TO_NAME,
    IMAGE_SIZE,
    METADATA_PATH,
    PATIENT_SPLIT_DIR,
    RAW_DIR,
    SPLIT_DIR,
    SPLIT_SEED,
    TEST_SIZE,
    ZIP_DIR,
)
from .crop import crop_brain_contour
from .mat_io import load_figshare_mat


def extract_zips(zip_dir: Path = ZIP_DIR, dest: Path = RAW_DIR) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    zips = sorted(zip_dir.glob("brainTumorDataPublic_*.zip"))
    if not zips:
        raise FileNotFoundError(f"No Figshare zip files in {zip_dir}")
    for archive in zips:
        print(f"Extracting {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)


def convert_and_crop(raw_dir: Path = RAW_DIR, cropped_dir: Path = CROPPED_DIR) -> dict[str, int]:
    for name in CLASS_NAMES:
        (cropped_dir / name).mkdir(parents=True, exist_ok=True)

    mat_files = sorted(raw_dir.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files under {raw_dir}")

    counts = {name: 0 for name in CLASS_NAMES}
    metadata = {}
    for mat_path in tqdm(mat_files, desc="Crop + convert"):
        image, label, pid = load_figshare_mat(mat_path)
        class_name = FIGSHARE_LABEL_TO_NAME[int(label)]
        cropped = crop_brain_contour(image)
        resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
        out_path = cropped_dir / class_name / f"{mat_path.stem}.png"
        cv2.imwrite(str(out_path), resized)
        metadata[out_path.name] = {"class_name": class_name, "patient_id": pid}
        counts[class_name] += 1
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return counts


def split_train_test(
    cropped_dir: Path = CROPPED_DIR,
    split_dir: Path = SPLIT_DIR,
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> dict[str, dict[str, int]]:
    """Stratified 80:20 split. Paper Table 1; seed from authors' split-folders notebook."""
    if split_dir.exists():
        shutil.rmtree(split_dir)
    summary = {name: {"train": 0, "test": 0} for name in CLASS_NAMES}
    for class_name in CLASS_NAMES:
        files = sorted((cropped_dir / class_name).glob("*.png"))
        if not files:
            raise FileNotFoundError(f"No cropped images in {cropped_dir / class_name}")
        train_files, test_files = train_test_split(
            files,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        for split_name, split_files in (("train", train_files), ("test", test_files)):
            dest = split_dir / split_name / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                shutil.copy2(src, dest / src.name)
            summary[class_name][split_name] = len(split_files)
    return summary


def split_patient_train_test(
    cropped_dir: Path = CROPPED_DIR,
    split_dir: Path = PATIENT_SPLIT_DIR,
    metadata_path: Path = METADATA_PATH,
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> dict[str, dict[str, int]]:
    """Split by patient so no patient's slices occur in both partitions."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file {metadata_path}; run conversion before patient split"
        )
    metadata = json.loads(metadata_path.read_text())
    if split_dir.exists():
        shutil.rmtree(split_dir)

    summary = {name: {"train": 0, "test": 0} for name in CLASS_NAMES}
    for class_name in CLASS_NAMES:
        class_metadata = {
            filename: item
            for filename, item in metadata.items()
            if item["class_name"] == class_name
        }
        patients = sorted({item["patient_id"] for item in class_metadata.values()})
        if not patients:
            raise FileNotFoundError(f"No metadata for class {class_name}")
        train_patients, test_patients = train_test_split(
            patients,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        patient_split = {
            patient: "train" for patient in train_patients
        } | {patient: "test" for patient in test_patients}
        for filename, item in class_metadata.items():
            source = cropped_dir / class_name / filename
            split_name = patient_split[item["patient_id"]]
            destination = split_dir / split_name / class_name / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            summary[class_name][split_name] += 1
    return summary
