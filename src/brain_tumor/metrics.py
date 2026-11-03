from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .config import CLASS_NAMES, PLOTS_DIR


def class_specificity(cm: np.ndarray) -> np.ndarray:
    specificities = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        specificities.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return np.asarray(specificities)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    spec = class_specificity(cm)
    accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    class_rows = {}
    for i, name in enumerate(CLASS_NAMES):
        class_rows[name] = {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "sensitivity": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "specificity": float(spec[i]),
            "support": int(report[name]["support"]),
        }
    summary = {
        "accuracy": accuracy,
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "sensitivity": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "specificity": float(spec.mean()),
        "per_class": class_rows,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return summary


def save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str = "Confusion matrix") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    tick = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(tick, CLASS_NAMES)
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_history_plots(history, out_path: Path) -> None:
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(epochs_range, acc, label="Train Set")
    axes[0].plot(epochs_range, val_acc, label="Val Set")
    axes[0].legend(loc="best")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy")

    axes[1].plot(epochs_range, loss, label="Train Set")
    axes[1].plot(epochs_range, val_loss, label="Val Set")
    axes[1].legend(loc="best")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Model Loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def dump_metrics(summary: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(summary)
    serializable.pop("classification_report", None)
    out_path.write_text(json.dumps(serializable, indent=2))
