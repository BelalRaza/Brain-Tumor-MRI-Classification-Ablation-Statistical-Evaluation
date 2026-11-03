"""DCT perceptual hash (pHash) for near-duplicate detection between datasets (H4).

Equivalent in spirit to ``imagehash.phash`` (32x32 grayscale -> 2D DCT -> top-left 8x8 ->
median threshold -> 64-bit hash) without the extra dependency.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn


def phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """Return a boolean array of hash_size**2 bits."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = hash_size * highfreq_factor
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)
    dct = dctn(small, norm="ortho")[:hash_size, :hash_size]
    return (dct > np.median(dct)).ravel()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def pack(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.astype(np.uint8))


def min_hamming_to_set(query_bits: np.ndarray, ref_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """query (Q,64) bool, ref (R,64) bool -> (min distance per query, argmin index)."""
    q = query_bits.astype(np.int16)
    r = ref_bits.astype(np.int16)
    # hamming = sum(q xor r) = sum(q) + sum(r) - 2 q.r
    dist = q.sum(1)[:, None] + r.sum(1)[None, :] - 2 * (q @ r.T)
    idx = dist.argmin(1)
    return dist[np.arange(len(q)), idx], idx
