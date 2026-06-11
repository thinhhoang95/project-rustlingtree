from __future__ import annotations

import numpy as np

from vlm_ppe.geo.polyline import cumulative_lengths, remove_duplicate_neighbors


def arc_length_resample_points(points: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    pts = remove_duplicate_neighbors(np.asarray(points, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if len(pts) < 2:
        raise ValueError("at least two distinct points are required")

    s = cumulative_lengths(pts)
    total = float(s[-1])
    if total <= 0.0:
        raise ValueError("polyline length must be positive")

    target_s = np.linspace(0.0, total, int(n_points))
    x = np.interp(target_s, s, pts[:, 0])
    y = np.interp(target_s, s, pts[:, 1])
    return np.column_stack([x, y]), target_s
