from __future__ import annotations

import numpy as np


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if len(pts) == 0:
        return np.asarray([], dtype=float)
    if len(pts) == 1:
        return np.asarray([0.0], dtype=float)
    deltas = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def polyline_length(points: np.ndarray) -> float:
    lengths = cumulative_lengths(points)
    if len(lengths) == 0:
        return 0.0
    return float(lengths[-1])


def remove_duplicate_neighbors(points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts.copy()
    keep = [0]
    for index in range(1, len(pts)):
        if float(np.linalg.norm(pts[index] - pts[keep[-1]])) > tolerance:
            keep.append(index)
    return pts[np.asarray(keep, dtype=int)]
