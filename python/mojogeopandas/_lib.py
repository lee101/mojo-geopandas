"""ctypes bridge for the Mojo broad-phase geometry kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-geopandas.so")
I = ctypes.c_int64
P = ctypes.c_void_p

_SIGNATURES = {
    "mgpd_bbox_intersects": ([P, P, P, I, I], None),
    "mgpd_bbox_contains": ([P, P, P, I, I], None),
    "mgpd_bbox_distance2": ([P, P, P, I, I], None),
    "mgpd_bbox_pair_counts": ([P, P, P, I, I, ctypes.c_bool], None),
    "mgpd_bbox_fill_pairs": ([P, P, P, P, P, I, I, ctypes.c_bool], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(["bash", script], capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes, fn.restype = argtypes, restype
    return _library


def _bounds(bounds: np.ndarray, name: str) -> np.ndarray:
    """Validate and convert bounds to the kernel's contiguous column layout."""
    array = np.asarray(bounds)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (n, 4), not {array.shape}")
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64, not {array.dtype}")
    return np.ascontiguousarray(array.T)


def bbox_mask(left: np.ndarray, right: np.ndarray, relation: str) -> np.ndarray:
    """Return a row-major broad-phase mask for two ``(n, 4)`` bounds arrays."""
    left = _bounds(left, "left")
    right = _bounds(right, "right")
    nleft, nright = left.shape[1], right.shape[1]
    mask = np.empty((nleft, nright), dtype=np.uint8)
    if relation == "intersects":
        fn = lib().mgpd_bbox_intersects
    elif relation == "contains":
        fn = lib().mgpd_bbox_contains
    else:
        raise ValueError(f"unknown AABB relation: {relation}")
    # Mojo pointers are non-nullable; NumPy may expose address zero for empties.
    if nleft and nright:
        fn(left.ctypes.data_as(P), right.ctypes.data_as(P), mask.ctypes.data_as(P), nleft, nright)
    return mask.view(bool)


def bbox_distance2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return squared minimum AABB distances, a lower bound on geometry distance."""
    left = _bounds(left, "left")
    right = _bounds(right, "right")
    nleft, nright = left.shape[1], right.shape[1]
    distances = np.empty((nleft, nright), dtype=np.float64)
    if nleft and nright:
        lib().mgpd_bbox_distance2(
            left.ctypes.data_as(P), right.ctypes.data_as(P), distances.ctypes.data_as(P), nleft, nright
        )
    return distances


def bbox_pairs(left: np.ndarray, right: np.ndarray, relation: str) -> tuple[np.ndarray, np.ndarray]:
    """Return row-major pairs from the broad phase without a dense mask."""
    left = _bounds(left, "left")
    right = _bounds(right, "right")
    nleft, nright = left.shape[1], right.shape[1]
    if relation not in {"intersects", "contains"}:
        raise ValueError(f"unknown AABB relation: {relation}")
    if not nleft or not nright:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    contains = relation == "contains"
    counts = np.empty(nleft, dtype=np.int64)
    library = lib()
    library.mgpd_bbox_pair_counts(
        left.ctypes.data_as(P), right.ctypes.data_as(P), counts.ctypes.data_as(P),
        nleft, nright, contains,
    )
    offsets = np.empty(nleft, dtype=np.int64)
    offsets[0] = 0
    if nleft > 1:
        np.cumsum(counts[:-1], out=offsets[1:])
    size = int(offsets[-1] + counts[-1])
    left_indices = np.empty(size, dtype=np.int64)
    right_indices = np.empty(size, dtype=np.int64)
    if size:
        library.mgpd_bbox_fill_pairs(
            left.ctypes.data_as(P), right.ctypes.data_as(P), offsets.ctypes.data_as(P),
            left_indices.ctypes.data_as(P), right_indices.ctypes.data_as(P),
            nleft, nright, contains,
        )
    return left_indices, right_indices
