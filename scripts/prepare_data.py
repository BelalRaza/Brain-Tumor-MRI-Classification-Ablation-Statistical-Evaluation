#!/usr/bin/env python3
"""Extract Figshare .mat files, crop brain contours, 80:20 split, augment train only."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain_tumor.augment import augment_train_set
from src.brain_tumor.config import (
    AUGMENTED_DIR,
    CLASS_NAMES,
    CROPPED_DIR,
    PATIENT_AUGMENTED_DIR,
    PATIENT_SPLIT_DIR,
    RAW_DIR,
    SPLIT_DIR,
)
from src.brain_tumor.prepare import (
    convert_and_crop,
    extract_zips,
    split_patient_train_test,
    split_train_test,
)


def copy_test_split(split_dir: Path, dest_root: Path) -> None:
    src = split_dir / "test"
    dst = dest_root / "test"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-crop", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument(
        "--patient-wise",
        action="store_true",
        help="Also create patient_split/ and patient_augmented/ without changing the paper split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_extract:
        extract_zips()
        print(f"Raw .mat files: {len(list(RAW_DIR.rglob('*.mat')))}")

    if not args.skip_crop:
        counts = convert_and_crop()
        print("Cropped class counts:", counts)

    if not args.skip_split:
        summary = split_train_test()
        print("80:20 split:", summary)

    if not args.skip_augment:
        if AUGMENTED_DIR.exists():
            shutil.rmtree(AUGMENTED_DIR)
        train_counts = augment_train_set(SPLIT_DIR / "train", AUGMENTED_DIR / "train")
        copy_test_split(SPLIT_DIR, AUGMENTED_DIR)
        print("Augmented train counts:", train_counts)
        for split in ("train", "test"):
            for name in CLASS_NAMES:
                n = len(list((AUGMENTED_DIR / split / name).glob("*.png")))
                print(f"  {split}/{name}: {n}")

    if args.patient_wise:
        patient_summary = split_patient_train_test()
        print("Patient-wise split:", patient_summary)
        if PATIENT_AUGMENTED_DIR.exists():
            shutil.rmtree(PATIENT_AUGMENTED_DIR)
        train_counts = augment_train_set(
            PATIENT_SPLIT_DIR / "train",
            PATIENT_AUGMENTED_DIR / "train",
        )
        copy_test_split(PATIENT_SPLIT_DIR, PATIENT_AUGMENTED_DIR)
        print("Patient-wise augmented train counts:", train_counts)


if __name__ == "__main__":
    main()
