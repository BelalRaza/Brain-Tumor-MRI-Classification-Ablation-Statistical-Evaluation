"""Numpy-only metrics (container-safe): confusion matrix, P/R/F1/specificity, OvR AUC, bootstrap CIs."""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ["glioma", "meningioma", "pituitary tumor"]


def confusion(y_true, y_pred, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def binary_auc(scores, positives) -> float:
    """Rank-based (Mann-Whitney) ROC-AUC with tie handling. NaN if one class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positives, dtype=bool)
    n_pos = positives.sum()
    n_neg = positives.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    n = scores.size
    while i < n:  # average ranks of ties
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_summary(y_true, probs, class_names=CLASS_NAMES, labels_present=None) -> dict:
    """Full metric dict. labels_present restricts macro averages to a subset of classes."""
    y_true = np.asarray(y_true, int)
    probs = np.asarray(probs, dtype=np.float64)
    n = len(class_names)
    y_pred = probs.argmax(axis=1)
    cm = confusion(y_true, y_pred, n)
    total = cm.sum()
    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "f1": float(f1),
            "specificity": float(spec),
            "auc_ovr": binary_auc(probs[:, i], y_true == i),
            "support": int(cm[i, :].sum()),
        }
    keep = list(range(n)) if labels_present is None else list(labels_present)
    keep_names = [class_names[i] for i in keep]

    def macro(key):
        vals = [per_class[k][key] for k in keep_names]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n": int(total),
        "accuracy": float(np.trace(cm) / total) if total else 0.0,
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "macro_specificity": macro("specificity"),
        "macro_auc_ovr": macro("auc_ovr"),
        "meningioma_recall": per_class["meningioma"]["recall"] if "meningioma" in per_class else None,
        "classes_evaluated": keep_names,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


BOOT_KEYS = ("accuracy", "macro_recall", "macro_f1", "macro_auc_ovr", "meningioma_recall")


def bootstrap_ci(
    y_true,
    probs,
    groups=None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    keys=BOOT_KEYS,
    labels_present=None,
) -> dict:
    """Percentile bootstrap CIs. If groups (patient ids) are given, resample groups, not images."""
    y_true = np.asarray(y_true, int)
    probs = np.asarray(probs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        s = classification_summary(y_true[idx], probs[idx], labels_present=labels_present)
        for k in keys:
            samples[k].append(s[k])
    out = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        lo, hi = (np.quantile(arr, [alpha / 2, 1 - alpha / 2]) if arr.size else (np.nan, np.nan))
        out[k] = {"lo": float(lo), "hi": float(hi)}
    out["unit"] = "patient" if len(uniq) < len(y_true) else "image"
    out["n_units"] = int(len(uniq))
    out["n_boot"] = n_boot
    return out
