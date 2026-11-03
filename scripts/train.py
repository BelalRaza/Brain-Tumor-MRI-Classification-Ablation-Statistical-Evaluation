#!/usr/bin/env python3
"""Train EfficientNetB0–B4 on the prepared Figshare split (paper Table 3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain_tumor.config import (
    AUGMENTED_DIR,
    BACKBONES,
    BATCH_SIZE,
    DEFAULT_BACKBONE,
    EPOCHS,
    OUTPUT_DIR,
    PLOTS_DIR,
)
from src.brain_tumor.dataset import prepare_arrays
from src.brain_tumor.metrics import dump_metrics, evaluate_predictions, save_confusion_matrix, save_history_plots
from src.brain_tumor.model import build_efficientnet, compile_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=BACKBONES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a saved Keras model checkpoint.",
    )
    parser.add_argument(
        "--initial-epoch",
        type=int,
        default=0,
        help="Zero-based epoch index to resume from, e.g. 5 after Epoch 5.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the Adam learning rate, useful when fine-tuning a checkpoint.",
    )
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="Freeze BatchNorm during fine-tuning (paper §5.1 observation).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=AUGMENTED_DIR,
        help="Root with train/ and test/ class folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.data_root / "train"
    test_dir = args.data_root / "test"
    x_train, y_train, x_test, y_test = prepare_arrays(train_dir, test_dir)
    print("x_train", x_train.shape, "y_train", y_train.shape)
    print("x_test", x_test.shape, "y_test", y_test.shape)

    if args.resume:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        model = build_efficientnet(args.backbone, freeze_bn=args.freeze_bn)
        compile_model(model)
        model.load_weights(args.resume)
    else:
        model = build_efficientnet(args.backbone, freeze_bn=args.freeze_bn)
        compile_model(model, learning_rate=args.learning_rate or 1e-3)
    if args.resume and args.learning_rate is not None:
        compile_model(model, learning_rate=args.learning_rate)
    history, ckpt = train_model(
        model,
        x_train,
        y_train,
        variant=args.backbone,
        epochs=args.epochs,
        initial_epoch=args.initial_epoch,
        batch_size=args.batch_size,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_history_plots(history, PLOTS_DIR / f"efficientnet{args.backbone.lower()}_history.png")

    best = load_model(ckpt)
    pred = best.predict(x_test, verbose=1)
    y_pred = pred.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    summary = evaluate_predictions(y_true, y_pred)
    print(
        f"Test accuracy={summary['accuracy']*100:.2f}%  "
        f"precision={summary['precision']*100:.2f}%  "
        f"recall={summary['recall']*100:.2f}%  "
        f"F1={summary['f1']*100:.2f}%"
    )
    dump_metrics(summary, OUTPUT_DIR / "metrics" / f"efficientnet{args.backbone.lower()}.json")
    save_confusion_matrix(
        np.array(summary["confusion_matrix"]),
        PLOTS_DIR / f"efficientnet{args.backbone.lower()}_confusion.png",
        title=f"EfficientNet{args.backbone} confusion matrix",
    )


if __name__ == "__main__":
    main()
