from pathlib import Path

import numpy as np


def _minmax_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    lo, hi = image.min(), image.max()
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = 255.0 * (image - lo) / (hi - lo)
    return scaled.astype(np.uint8)


def load_figshare_mat(path: Path) -> tuple[np.ndarray, int, str]:
    """Load one Figshare .mat slice. Returns (uint8 image, label int, patient id)."""
    path = Path(path)
    try:
        from scipy.io import loadmat

        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        cjdata = mat["cjdata"]
        image = np.asarray(cjdata.image)
        label = int(np.asarray(cjdata.label).squeeze())
        pid = str(np.asarray(cjdata.PID).squeeze())
    except Exception:
        import h5py

        with h5py.File(path, "r") as handle:
            group = handle["cjdata"]
            image = np.array(group["image"]).T
            label = int(np.array(group["label"]).squeeze())
            pid_raw = np.array(group["PID"]).squeeze()
            if pid_raw.dtype == np.uint16 or np.issubdtype(pid_raw.dtype, np.integer):
                pid = "".join(chr(int(x)) for x in np.atleast_1d(pid_raw))
            else:
                pid = str(pid_raw)

    image = _minmax_uint8(np.squeeze(image))
    return image, label, pid
