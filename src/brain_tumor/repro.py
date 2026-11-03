"""Reproducibility helpers (H3): global seeding, determinism probe, run-config logging.

Container-safe: depends only on the standard library, numpy and TensorFlow.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and TensorFlow RNGs."""
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:  # pragma: no cover - very old TF
        pass


_PROBE = r"""
import numpy as np, tensorflow as tf
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
inp = tf.keras.Input((32, 32, 3))
x = tf.keras.layers.Conv2D(8, 3, padding="same")(inp)
x = tf.keras.layers.DepthwiseConv2D(3, padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
out = tf.keras.layers.Dense(3, activation="softmax")(x)
m = tf.keras.Model(inp, out)
m.compile("adam", "categorical_crossentropy")
m.fit(np.random.rand(16, 32, 32, 3).astype("float32"), np.eye(3)[np.arange(16) % 3], epochs=1, verbose=0)
print("DETERMINISM_OK")
"""


def probe_op_determinism(timeout: int = 300) -> bool:
    """Check in a subprocess whether op determinism works on this device.

    Some GPU kernels have no deterministic implementation and raise at run time;
    probing in a child process avoids crashing (or permanently configuring) the main run.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return False
    return out.returncode == 0 and "DETERMINISM_OK" in out.stdout


def enable_determinism(mode: str = "auto") -> str:
    """mode: 'on' | 'off' | 'auto' (probe first). Returns the effective status string."""
    import tensorflow as tf

    if mode == "off":
        return "off"
    if mode == "auto" and not probe_op_determinism():
        return "unavailable (probe failed; running non-deterministic kernels)"
    tf.config.experimental.enable_op_determinism()
    return "on"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def library_versions() -> dict:
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import tensorflow as tf

        versions["tensorflow"] = tf.__version__
        versions["keras"] = getattr(tf.keras, "__version__", None)
        try:
            import keras

            versions["keras"] = keras.__version__
        except Exception:
            pass
        versions["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        pass
    return versions


def write_run_config(run_dir: Path, args: dict, extra: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": args,
        "platform": platform.platform(),
        "versions": library_versions(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
    }
    if extra:
        config.update(extra)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    return config
