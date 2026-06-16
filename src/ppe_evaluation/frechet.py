from __future__ import annotations

import numpy as np


def as_points(value: object) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("trajectory points must have shape (n_points, 2)")
    if len(points) == 0:
        raise ValueError("trajectory must contain at least one point")
    return points


def discrete_frechet_distance(points_a: object, points_b: object) -> float:
    """Return the discrete Frechet distance between two 2D trajectories."""
    a = as_points(points_a)
    b = as_points(points_b)
    distances = np.linalg.norm(a[:, np.newaxis, :] - b[np.newaxis, :, :], axis=2)
    ca = np.empty(distances.shape, dtype=float)

    ca[0, 0] = distances[0, 0]
    for i in range(1, len(a)):
        ca[i, 0] = max(ca[i - 1, 0], distances[i, 0])
    for j in range(1, len(b)):
        ca[0, j] = max(ca[0, j - 1], distances[0, j])
    for i in range(1, len(a)):
        for j in range(1, len(b)):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                distances[i, j],
            )
    return float(ca[-1, -1])
