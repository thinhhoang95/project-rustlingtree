"""Small, deterministic polyline primitives used by :mod:`hailmary`.

The simulator uses metres internally.  Functions in this module deliberately
return owned, read-only ``float64`` arrays so offline artifacts cannot acquire
mutable aliases accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def readonly_float64(
    values: object,
    *,
    name: str = "array",
    ndim: int | None = None,
) -> np.ndarray:
    """Return a finite, C-contiguous, owned and read-only ``float64`` array."""

    array = np.array(values, dtype=np.float64, order="C", copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _points(values: object, *, name: str = "points_m", minimum: int = 0) -> np.ndarray:
    points = readonly_float64(values, name=name, ndim=2)
    if points.shape[1:] != (2,):
        raise ValueError(f"{name} must have shape (n, 2)")
    if len(points) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} points")
    return points


def remove_duplicate_neighbors(points_m: object, *, tolerance_m: float = 1.0e-9) -> np.ndarray:
    """Remove only consecutive duplicate points while preserving endpoints."""

    if not np.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("tolerance_m must be finite and nonnegative")
    points = _points(points_m)
    if len(points) <= 1:
        return points
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], deltas > float(tolerance_m)))
    return readonly_float64(points[keep], name="deduplicated points", ndim=2)


def cumulative_lengths_m(points_m: object) -> np.ndarray:
    """Cumulative Euclidean arc length for a local-coordinate polyline."""

    points = _points(points_m)
    if len(points) == 0:
        return readonly_float64([], name="cumulative lengths", ndim=1)
    if len(points) == 1:
        return readonly_float64([0.0], name="cumulative lengths", ndim=1)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return readonly_float64(
        np.concatenate(([0.0], np.cumsum(segment_lengths, dtype=np.float64))),
        name="cumulative lengths",
        ndim=1,
    )


def polyline_length_m(points_m: object) -> float:
    lengths = cumulative_lengths_m(points_m)
    return 0.0 if len(lengths) == 0 else float(lengths[-1])


@dataclass(frozen=True)
class ResampledPolyline:
    """Polyline sampled at equal fractions of its own arc length."""

    points_m: np.ndarray
    distance_m: np.ndarray
    progress: np.ndarray

    def __post_init__(self) -> None:
        points = _points(self.points_m, minimum=2)
        distance = readonly_float64(self.distance_m, name="distance_m", ndim=1)
        progress = readonly_float64(self.progress, name="progress", ndim=1)
        if len(points) != len(distance) or len(points) != len(progress):
            raise ValueError("resampled arrays must have equal lengths")
        if distance[0] != 0.0 or np.any(np.diff(distance) <= 0.0):
            raise ValueError("distance_m must start at zero and be strictly increasing")
        if not np.isclose(progress[0], 0.0) or not np.isclose(progress[-1], 1.0):
            raise ValueError("progress must span [0, 1]")
        if np.any(np.diff(progress) <= 0.0):
            raise ValueError("progress must be strictly increasing")
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "progress", progress)


def resample_polyline(points_m: object, n_points: int = 128) -> ResampledPolyline:
    """Resample a polyline at equally spaced arc-length fractions."""

    if isinstance(n_points, bool) or int(n_points) != n_points or n_points < 2:
        raise ValueError("n_points must be an integer at least 2")
    points = remove_duplicate_neighbors(points_m)
    if len(points) < 2:
        raise ValueError("at least two distinct points are required")
    source_s = cumulative_lengths_m(points)
    total_m = float(source_s[-1])
    if total_m <= 0.0:
        raise ValueError("polyline length must be positive")
    target_s = np.linspace(0.0, total_m, int(n_points), dtype=np.float64)
    sampled = np.column_stack(
        (
            np.interp(target_s, source_s, points[:, 0]),
            np.interp(target_s, source_s, points[:, 1]),
        )
    )
    # np.interp is exact at the endpoints in normal use; assigning them makes
    # that contract explicit even across NumPy versions.
    sampled[0] = points[0]
    sampled[-1] = points[-1]
    return ResampledPolyline(
        points_m=sampled,
        distance_m=target_s,
        progress=np.linspace(0.0, 1.0, int(n_points), dtype=np.float64),
    )


def orient_upstream_to_threshold(
    points_m: object,
    *,
    threshold_xy_m: Iterable[float] = (0.0, 0.0),
) -> np.ndarray:
    """Orient a track so the runway-threshold end is last.

    The endpoint closer to ``threshold_xy_m`` is treated as the threshold end.
    Equal-distance ties retain input order, a deterministic and auditable rule.
    """

    points = _points(points_m, minimum=2)
    threshold = readonly_float64(tuple(threshold_xy_m), name="threshold_xy_m", ndim=1)
    if threshold.shape != (2,):
        raise ValueError("threshold_xy_m must have shape (2,)")
    first_distance = float(np.linalg.norm(points[0] - threshold))
    last_distance = float(np.linalg.norm(points[-1] - threshold))
    oriented = points[::-1] if first_distance < last_distance else points
    return readonly_float64(oriented, name="oriented points", ndim=2)


def clip_to_terminal_radius(
    points_m: object,
    *,
    radius_m: float,
    threshold_xy_m: Iterable[float] = (0.0, 0.0),
) -> np.ndarray:
    """Clip an arrival at its first outside-to-inside radius crossing."""

    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    threshold = readonly_float64(tuple(threshold_xy_m), name="threshold_xy_m", ndim=1)
    if threshold.shape != (2,):
        raise ValueError("threshold_xy_m must have shape (2,)")
    points = orient_upstream_to_threshold(points_m, threshold_xy_m=threshold)
    relative = points - threshold
    distances = np.linalg.norm(relative, axis=1)
    for index in range(len(points) - 1):
        if distances[index] < radius_m or distances[index + 1] > radius_m:
            continue
        p0 = relative[index]
        delta = relative[index + 1] - p0
        a = float(np.dot(delta, delta))
        if a <= 0.0:
            continue
        b = 2.0 * float(np.dot(p0, delta))
        c = float(np.dot(p0, p0) - radius_m * radius_m)
        discriminant = max(0.0, b * b - 4.0 * a * c)
        roots = sorted(((-b - np.sqrt(discriminant)) / (2.0 * a), (-b + np.sqrt(discriminant)) / (2.0 * a)))
        fractions = [value for value in roots if -1.0e-12 <= value <= 1.0 + 1.0e-12]
        if not fractions:
            continue
        fraction = float(np.clip(fractions[-1], 0.0, 1.0))
        crossing = points[index] + fraction * (points[index + 1] - points[index])
        clipped = np.vstack((crossing, points[index + 1 :]))
        return remove_duplicate_neighbors(clipped)
    raise ValueError("track does not contain an outside-to-inside terminal-radius crossing")


def align_terminal_track(
    points_m: object,
    *,
    terminal_radius_m: float,
    threshold_capture_radius_m: float,
    threshold_xy_m: Iterable[float] = (0.0, 0.0),
) -> np.ndarray:
    """Clip, orient, and attach the exact runway threshold endpoint."""

    if not np.isfinite(threshold_capture_radius_m) or threshold_capture_radius_m <= 0.0:
        raise ValueError("threshold_capture_radius_m must be finite and positive")
    threshold = readonly_float64(tuple(threshold_xy_m), name="threshold_xy_m", ndim=1)
    clipped = clip_to_terminal_radius(
        points_m,
        radius_m=terminal_radius_m,
        threshold_xy_m=threshold,
    )
    endpoint_distance = float(np.linalg.norm(clipped[-1] - threshold))
    if endpoint_distance > threshold_capture_radius_m:
        raise ValueError("track does not terminate within the threshold capture radius")
    if endpoint_distance > 1.0e-9:
        clipped = np.vstack((clipped, threshold))
    return remove_duplicate_neighbors(clipped)


def prepare_track_for_clustering(
    points_m: object,
    *,
    terminal_radius_m: float,
    threshold_capture_radius_m: float,
    n_points: int = 128,
    threshold_xy_m: Iterable[float] = (0.0, 0.0),
) -> ResampledPolyline:
    aligned = align_terminal_track(
        points_m,
        terminal_radius_m=terminal_radius_m,
        threshold_capture_radius_m=threshold_capture_radius_m,
        threshold_xy_m=threshold_xy_m,
    )
    return resample_polyline(aligned, n_points=n_points)


@dataclass(frozen=True)
class ExecutablePolyline:
    """Threshold-to-upstream geometry indexed by remaining distance ``s_m``."""

    points_m: np.ndarray
    s_m: np.ndarray

    def __post_init__(self) -> None:
        points = _points(self.points_m, minimum=2)
        station = readonly_float64(self.s_m, name="s_m", ndim=1)
        if len(points) != len(station):
            raise ValueError("points_m and s_m must have equal lengths")
        if station[0] != 0.0 or np.any(np.diff(station) <= 0.0):
            raise ValueError("s_m must start at zero and increase upstream")
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "s_m", station)


def to_executable_station_order(upstream_to_threshold_points_m: object) -> ExecutablePolyline:
    """Convert clustering order into ``s=0`` threshold-first artifact order."""

    points = _points(upstream_to_threshold_points_m, minimum=2)
    executable = readonly_float64(points[::-1], name="executable points", ndim=2)
    return ExecutablePolyline(points_m=executable, s_m=cumulative_lengths_m(executable))


# Concise compatibility aliases for callers familiar with the PPE helpers.
cumulative_lengths = cumulative_lengths_m
polyline_length = polyline_length_m


def arc_length_resample_points(points: object, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    sampled = resample_polyline(points, n_points)
    return sampled.points_m, sampled.distance_m
