#!/usr/bin/env python3
"""Generate training, per-class, confusion-matrix, and Grad-CAM report figures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from src.brain_tumor.config import CLASS_NAMES, IMAGE_SIZE, MODELS_DIR, OUTPUT_DIR, PLOTS_DIR
from src.brain_tumor.dataset import load_split_directory
from src.brain_tumor.gradcam import gradcam_heatmap, overlay_gradcam
from src.brain_tumor.metrics import (
    dump_metrics,
    evaluate_predictions,
    save_confusion_matrix,
)
from src.brain_tumor.model import build_efficientnet, compile_model


def parse_history(log_path: Path) -> dict[str, list[float]]:
    pattern = re.compile(
        r"accuracy:\s+(?P<accuracy>[0-9.]+).*?loss:\s+(?P<loss>[0-9.]+)"
        r".*?val_accuracy:\s+(?P<val_accuracy>[0-9.]+).*?"
        r"val_loss:\s+(?P<val_loss>[0-9.]+)"
    )
    history = {key: [] for key in ("accuracy", "loss", "val_accuracy", "val_loss")}
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            for key in history:
                history[key].append(float(match.group(key)))
    if not history["accuracy"]:
        raise ValueError(f"No completed epochs found in {log_path}")
    return history


def load_checkpoint(path: Path, backbone: str):
    try:
        return load_model(path)
    except (TypeError, ValueError):
        model = build_efficientnet(backbone)
        compile_model(model)
        model.load_weights(path)
        return model


def save_history_plot(history: dict[str, list[float]], output: Path) -> None:
    epochs = np.arange(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["accuracy"], label="Training accuracy")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("(a) Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].plot(epochs, history["loss"], label="Training loss")
    axes[1].plot(epochs, history["val_loss"], label="Validation loss")
    axes[1].set_title("(b) Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_per_class_plot(summary: dict, output: Path) -> None:
    metrics = ("precision", "sensitivity", "f1", "specificity")
    values = np.array(
        [[summary["per_class"][name][metric] for metric in metrics] for name in CLASS_NAMES]
    )
    x = np.arange(len(CLASS_NAMES))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, metric in enumerate(metrics):
        ax.bar(x + (index - 1.5) * width, values[:, index], width, label=metric.title())
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_gradcam_grid(model, test_dir: Path, output: Path) -> None:
    fig, axes = plt.subplots(3, len(CLASS_NAMES), figsize=(14, 10))
    row_titles = ("Attention map", "Original image", "Grad-CAM")
    for column, class_name in enumerate(CLASS_NAMES):
        image_path = sorted((test_dir / class_name).glob("*.png"))[0]
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        batch = np.expand_dims(image.astype("float32"), axis=0)
        predictions = model.predict(batch, verbose=0)[0]
        heatmap = gradcam_heatmap(batch, model, "top_activation")
        overlay = overlay_gradcam(image, heatmap)

        axes[0, column].imshow(heatmap, cmap="jet")
        axes[1, column].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[2, column].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, column].set_title(class_name)
        for row in range(3):
            axes[row, column].axis("off")
        axes[2, column].set_xlabel(
            f"Predicted: {CLASS_NAMES[int(np.argmax(predictions))]}",
            labelpad=8,
        )
    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=11)
    fig.suptitle("EfficientNetB2 Grad-CAM visualization", y=0.98)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="B2")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=Path("data/patient_augmented/test"))
    parser.add_argument("--log", type=Path, default=Path("outputs/logs/train-live.log"))
    args = parser.parse_args()

    model_path = args.model or MODELS_DIR / f"efficientnet{args.backbone.lower()}.h5"
    model = load_checkpoint(model_path, args.backbone)
    x_test, y_test = load_split_directory(args.test_dir)
    predictions = model.predict(x_test, verbose=1).argmax(axis=1)
    y_true = y_test
    summary = evaluate_predictions(y_true, predictions)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    dump_metrics(summary, OUTPUT_DIR / "metrics" / f"efficientnet{args.backbone.lower()}_test.json")
    save_confusion_matrix(
        np.asarray(summary["confusion_matrix"]),
        PLOTS_DIR / f"efficientnet{args.backbone.lower()}_test_confusion.png",
        title=f"EfficientNet{args.backbone} patient-wise test confusion matrix",
    )
    save_history_plot(
        parse_history(args.log),
        PLOTS_DIR / f"efficientnet{args.backbone.lower()}_accuracy_loss.png",
    )
    save_per_class_plot(
        summary,
        PLOTS_DIR / f"efficientnet{args.backbone.lower()}_per_class_scores.png",
    )
    save_gradcam_grid(
        model,
        args.test_dir,
        PLOTS_DIR / f"efficientnet{args.backbone.lower()}_gradcam_grid.png",
    )
    print(f"Test accuracy: {summary['accuracy'] * 100:.2f}%")
    print(f"Completed epochs in log: {len(parse_history(args.log)['accuracy'])}")


if __name__ == "__main__":
    main()
