"""Loss functions for the M4 ablation (H2). Container-safe (numpy + TensorFlow only)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

_register = tf.keras.utils.register_keras_serializable


def class_balanced_alpha(counts, beta: float = 0.999) -> np.ndarray:
    """Cui et al. (CVPR 2019) effective-number class weights, normalised to mean 1.

    counts must be the *pre-augmentation* per-class training counts; the offline
    augmentation already equalises file counts, so post-augmentation weights would be ~uniform.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("class counts must be positive")
    effective = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    weights = 1.0 / effective
    return (weights / weights.mean()).astype(np.float32)


@_register(package="brain_tumor")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """FL = - sum_c alpha_c * y_c * (1 - p_c)^gamma * log(p_c).

    With gamma = 0 and alpha = 1 this equals categorical cross-entropy.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, name: str = "categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = None if alpha is None else [float(a) for a in np.asarray(alpha).ravel()]
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -y_true * tf.math.log(y_pred)
        modulating = tf.pow(1.0 - y_pred, self.gamma)
        loss = modulating * ce
        if self.alpha is not None:
            loss = loss * tf.constant(self.alpha, dtype=y_pred.dtype)
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config


def make_loss(kind: str, train_counts=None, gamma: float = 2.0, beta: float = 0.999):
    """Return (loss_object_or_name, info_dict)."""
    if kind == "ce":
        return "categorical_crossentropy", {"loss": "ce"}
    if kind == "focal":
        if train_counts is None:
            raise ValueError("focal loss needs pre-augmentation train_counts for class balancing")
        alpha = class_balanced_alpha(train_counts, beta=beta)
        info = {
            "loss": "cb_focal",
            "gamma": gamma,
            "beta": beta,
            "alpha": alpha.tolist(),
            "alpha_counts": [int(c) for c in train_counts],
        }
        return CategoricalFocalLoss(alpha=alpha, gamma=gamma), info
    raise ValueError(f"unknown loss {kind!r}")
