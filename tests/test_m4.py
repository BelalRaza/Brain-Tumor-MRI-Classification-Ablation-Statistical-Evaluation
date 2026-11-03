"""M4 unit tests: patient leakage, losses, metrics/bootstrap, pHash, determinism (CPU)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # allow running directly via `python tests/test_m4.py` without pytest installed
    class _MiniPytest:
        class mark:
            @staticmethod
            def skipif(cond, reason=""):
                def dec(fn):
                    return (lambda: None) if cond else fn
                return dec

        @staticmethod
        def importorskip(mod):
            return __import__(mod)

        @staticmethod
        def raises(exc):
            class _Ctx:
                def __enter__(self):
                    return self
                def __exit__(self, et, ev, tb):
                    if et is None:
                        raise AssertionError(f"Expected {exc} to be raised")
                    return issubclass(et, exc)
            return _Ctx()

    pytest = _MiniPytest()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain_tumor.config import CLASS_NAMES, METADATA_PATH  # noqa: E402
from src.brain_tumor.m4_metrics import binary_auc, bootstrap_ci, classification_summary  # noqa: E402
from src.brain_tumor.splits import (  # noqa: E402
    PatientLeakageError,
    assert_no_patient_overlap,
    build_manifests,
    source_stem,
)

HAVE_DATA = METADATA_PATH.exists()


# ------------------------------------------------------------------ splits (H1)


def test_source_stem():
    assert source_stem("aug_1841_0_6798.png") == "1841"
    assert source_stem("1841.png") == "1841"


def test_overlap_assertion_raises():
    with pytest.raises(PatientLeakageError):
        assert_no_patient_overlap({"train": [{"patient_id": "1"}], "test": [{"patient_id": "1"}]})


@pytest.mark.skipif(not HAVE_DATA, reason="prepared Figshare data not present")
def test_real_manifests_have_no_patient_overlap():
    splits = build_manifests()
    pats = {k: {r["patient_id"] for r in v} for k, v in splits.items()}
    train_p = pats["train"] | pats["train_source"]
    assert not (train_p & pats["val"])
    assert not (train_p & pats["test"])
    assert not (pats["val"] & pats["test"])
    assert len(train_p | pats["val"] | pats["test"]) == 233
    # every class present in val
    assert {r["class_name"] for r in splits["val"]} == set(CLASS_NAMES)


# ------------------------------------------------------------------ losses (H2)


def test_focal_gamma0_alpha1_equals_ce():
    tf = pytest.importorskip("tensorflow")
    from src.brain_tumor.losses import CategoricalFocalLoss

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(16, 3)).astype("float32")
    p = tf.nn.softmax(logits)
    y = np.eye(3, dtype="float32")[rng.integers(0, 3, 16)]
    fl = CategoricalFocalLoss(alpha=None, gamma=0.0)(y, p).numpy()
    ce = tf.keras.losses.CategoricalCrossentropy()(y, p).numpy()
    assert np.isclose(fl, ce, atol=1e-5)


def test_focal_downweights_easy_examples():
    pytest.importorskip("tensorflow")
    from src.brain_tumor.losses import CategoricalFocalLoss

    y = np.array([[1, 0, 0]], dtype="float32")
    easy = np.array([[0.95, 0.03, 0.02]], dtype="float32")
    hard = np.array([[0.30, 0.40, 0.30]], dtype="float32")
    f = CategoricalFocalLoss(gamma=2.0)
    ratio_focal = float(f(y, hard)) / float(f(y, easy))
    ratio_ce = np.log(0.30) / np.log(0.95)
    assert ratio_focal > ratio_ce  # hard examples weigh relatively more


def test_class_balanced_alpha_favours_meningioma():
    pytest.importorskip("tensorflow")
    from src.brain_tumor.losses import class_balanced_alpha

    a = class_balanced_alpha([1024, 484, 637])
    assert np.isclose(a.mean(), 1.0)
    assert a[1] == a.max() and a[0] == a.min()


def test_focal_serialization_roundtrip():
    pytest.importorskip("tensorflow")
    from src.brain_tumor.losses import CategoricalFocalLoss

    f = CategoricalFocalLoss(alpha=[0.5, 1.5, 1.0], gamma=2.0)
    g = CategoricalFocalLoss.from_config(f.get_config())
    assert g.alpha == f.alpha and g.gamma == f.gamma


# ------------------------------------------------------------------ metrics


def test_metrics_match_sklearn():
    from sklearn.metrics import f1_score, recall_score, roc_auc_score

    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    probs = rng.dirichlet(np.ones(3), 300)
    probs[np.arange(300), y] += 0.5
    probs /= probs.sum(1, keepdims=True)
    s = classification_summary(y, probs)
    pred = probs.argmax(1)
    assert np.isclose(s["macro_f1"], f1_score(y, pred, average="macro"))
    assert np.isclose(s["meningioma_recall"], recall_score(y, pred, labels=[1], average="macro"))
    assert np.isclose(s["macro_auc_ovr"], roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    assert np.isclose(binary_auc([1, 1, 2, 3], [0, 1, 0, 1]), roc_auc_score([0, 1, 0, 1], [1, 1, 2, 3]))


def test_bootstrap_ci_contains_point_and_narrows():
    rng = np.random.default_rng(2)

    def make(n):
        y = rng.integers(0, 3, n)
        probs = rng.dirichlet(np.ones(3), n)
        probs[np.arange(n), y] += 0.8
        return y, probs / probs.sum(1, keepdims=True)

    y1, p1 = make(120)
    y2, p2 = make(2000)
    ci1 = bootstrap_ci(y1, p1, n_boot=300)
    ci2 = bootstrap_ci(y2, p2, n_boot=300)
    point = classification_summary(y1, p1)["accuracy"]
    assert ci1["accuracy"]["lo"] <= point <= ci1["accuracy"]["hi"]
    w1 = ci1["accuracy"]["hi"] - ci1["accuracy"]["lo"]
    w2 = ci2["accuracy"]["hi"] - ci2["accuracy"]["lo"]
    assert w2 < w1


def test_bootstrap_resamples_patients():
    y = np.repeat([0, 1, 2], 10)
    probs = np.eye(3)[y]
    groups = np.repeat(np.arange(6), 5)
    ci = bootstrap_ci(y, probs, groups=groups, n_boot=50)
    assert ci["unit"] == "patient" and ci["n_units"] == 6


# ------------------------------------------------------------------ pHash (H4)


def test_phash_properties():
    cv2 = pytest.importorskip("cv2")
    from src.brain_tumor.phash import hamming, phash

    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur((rng.random((240, 240)) * 255).astype("uint8"), (31, 31), 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    other = cv2.GaussianBlur((rng.random((240, 240)) * 255).astype("uint8"), (31, 31), 0)
    other = cv2.normalize(other, None, 0, 255, cv2.NORM_MINMAX)
    ok, enc = cv2.imencode(".jpg", cv2.resize(img, (300, 300)), [cv2.IMWRITE_JPEG_QUALITY, 70])
    recompressed = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    assert hamming(phash(img), phash(img)) == 0
    assert hamming(phash(img), phash(recompressed)) <= 5
    assert hamming(phash(img), phash(other)) > 10


# ------------------------------------------------------------------ determinism (H3)

_DET = r"""
import sys; sys.path.insert(0, %r)
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import numpy as np
from src.brain_tumor.repro import set_global_seed
set_global_seed(123)
tf.config.experimental.enable_op_determinism()
inp = tf.keras.Input((32, 32, 3))
x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
x = tf.keras.layers.DepthwiseConv2D(3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
out = tf.keras.layers.Dense(3, activation="softmax")(x)
m = tf.keras.Model(inp, out)
m.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy")
rng = np.random.default_rng(0)
X = rng.random((64, 32, 32, 3)).astype("float32"); Y = np.eye(3)[rng.integers(0, 3, 64)]
h = m.fit(X, Y, epochs=1, batch_size=16, shuffle=True, verbose=0)
print(repr(h.history["loss"][0]))
"""


def test_same_seed_same_first_epoch_loss_cpu():
    pytest.importorskip("tensorflow")
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONHASHSEED="0")
    runs = [
        subprocess.run([sys.executable, "-c", _DET % str(ROOT)], capture_output=True, text=True, env=env, timeout=300)
        for _ in range(2)
    ]
    for r in runs:
        assert r.returncode == 0, r.stderr[-2000:]
    losses = [r.stdout.strip().splitlines()[-1] for r in runs]
    assert losses[0] == losses[1], losses
