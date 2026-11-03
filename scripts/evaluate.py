#!/usr/bin/env python3
"""Evaluate a saved EfficientNet checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain_tumor.config import AUGMENTED_DIR, CLASS_NAMES, DEFAULT_BACKBONE, MODELS_DIR, OUTPUT_DIR, PLOTS_DIR
from src.brain_tumor.dataset import load_split_directory
from src.brain_tumor.metrics import dump_metrics, evaluate_predictions, save_confusion_matrix
from src.brain_tumor.model import build_efficientnet, compile_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=AUGMENTED_DIR / "test")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional tag inserted into metric and confusion-matrix filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model or (MODELS_DIR / f"efficientnet{args.backbone.lower()}.h5")
    data_dir = args.test_dir if args.split == "test" else args.test_dir.parent / "train"
    x, y = load_split_directory(data_dir)
    try:
        model = load_model(model_path)
    except (TypeError, ValueError):
        model = build_efficientnet(args.backbone)
        compile_model(model)
        model.load_weights(model_path)
    pred = model.predict(x, verbose=1).argmax(axis=1)
    summary = evaluate_predictions(y, pred)
    print(f"{args.split} accuracy: {summary['accuracy']*100:.2f}%")
    for name in CLASS_NAMES:
        row = summary["per_class"][name]
        print(
            f"  {name:18s} P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} Spec={row['specificity']:.4f}"
        )
    tag = f"_{args.output_tag}" if args.output_tag else ""
    stem = f"efficientnet{args.backbone.lower()}{tag}_{args.split}"
    dump_metrics(summary, OUTPUT_DIR / "metrics" / f"{stem}.json")
    save_confusion_matrix(
        np.array(summary["confusion_matrix"]),
        PLOTS_DIR / f"{stem}_confusion.png",
        title=f"EfficientNet{args.backbone} {args.split}",
    )


if __name__ == "__main__":
    main()
