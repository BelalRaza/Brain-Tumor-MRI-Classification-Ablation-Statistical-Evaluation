#!/usr/bin/env python3
"""Grad-CAM overlays for the fine-tuned EfficientNet (paper §5.3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain_tumor.config import CLASS_NAMES, DEFAULT_BACKBONE, IMAGE_SIZE, MODELS_DIR, PLOTS_DIR
from src.brain_tumor.gradcam import gradcam_heatmap, overlay_gradcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--last-conv", default="top_activation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model or (MODELS_DIR / f"efficientnet{args.backbone.lower()}.h5")
    model = load_model(model_path)
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    batch = np.expand_dims(image, axis=0)
    preds = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    heatmap = gradcam_heatmap(batch.astype("float32"), model, args.last_conv)
    overlay = overlay_gradcam(image, heatmap)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Grad-CAM")
    axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Predicted: {CLASS_NAMES[pred_idx]}\nconf={preds[pred_idx]:.3f}"
    )
    axes[2].imshow(heatmap, cmap="jet")
    axes[2].set_title("Attention map")
    for ax in axes:
        ax.axis("off")
    out = PLOTS_DIR / f"gradcam_{args.image.stem}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
