from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class CompressionBreakpoints:
    lateral_indices: np.ndarray
    altitude_indices: np.ndarray
    minimal_indices: np.ndarray


def project_latlon_to_local_m(
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project latitude/longitude to a local equirectangular meter frame."""
    if len(latitudes_deg) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    lat0_rad = np.radians(float(np.nanmean(latitudes_deg)))
    lon0_rad = np.radians(float(np.nanmean(longitudes_deg)))
    lat_rad = np.radians(latitudes_deg.astype(float))
    lon_rad = np.radians(longitudes_deg.astype(float))

    x = EARTH_RADIUS_M * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = EARTH_RADIUS_M * (lat_rad - lat0_rad)
    return x, y


def _rdp_spatial_recursive(points: np.ndarray, start: int, end: int, epsilon: float, keep: set[int]) -> None:
    if end <= start + 1:
        return

    segment = points[end] - points[start]
    candidates = points[start + 1 : end]
    if len(candidates) == 0:
        return

    segment_norm = float(np.linalg.norm(segment))
    if segment_norm == 0.0:
        distances = np.linalg.norm(candidates - points[start], axis=1)
    else:
        candidate_vectors = candidates - points[start]
        cross_products = segment[0] * candidate_vectors[:, 1] - segment[1] * candidate_vectors[:, 0]
        distances = np.abs(cross_products / segment_norm)

    relative_index = int(np.argmax(distances))
    max_distance = float(distances[relative_index])
    if max_distance <= epsilon:
        return

    index = start + 1 + relative_index
    keep.add(index)
    _rdp_spatial_recursive(points, start, index, epsilon, keep)
    _rdp_spatial_recursive(points, index, end, epsilon, keep)


def douglas_peucker_spatial_indices(
    x: np.ndarray,
    y: np.ndarray,
    epsilon_m: float,
) -> np.ndarray:
    """Return retained point indices for a 2D trajectory in meter coordinates."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) <= 2:
        return np.arange(len(x), dtype=int)

    points = np.column_stack([x.astype(float), y.astype(float)])
    keep = {0, len(points) - 1}
    _rdp_spatial_recursive(points, 0, len(points) - 1, float(epsilon_m), keep)
    return np.array(sorted(keep), dtype=int)


def _interpolated_value_at(
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    end: int,
    target_indices: np.ndarray,
) -> np.ndarray:
    x0 = float(x[start])
    x1 = float(x[end])
    y0 = float(y[start])
    y1 = float(y[end])
    if x1 == x0:
        return np.full(len(target_indices), y0, dtype=float)

    fraction = (x[target_indices].astype(float) - x0) / (x1 - x0)
    return y0 + fraction * (y1 - y0)


def _rdp_series_recursive(
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    end: int,
    epsilon: float,
    keep: set[int],
) -> None:
    if end <= start + 1:
        return

    candidate_indices = np.arange(start + 1, end, dtype=int)
    if len(candidate_indices) == 0:
        return

    interpolated = _interpolated_value_at(x, y, start, end, candidate_indices)
    errors = np.abs(y[candidate_indices].astype(float) - interpolated)
    relative_index = int(np.argmax(errors))
    max_error = float(errors[relative_index])
    if max_error <= epsilon:
        return

    index = int(candidate_indices[relative_index])
    keep.add(index)
    _rdp_series_recursive(x, y, start, index, epsilon, keep)
    _rdp_series_recursive(x, y, index, end, epsilon, keep)


def douglas_peucker_series_indices(
    x: np.ndarray,
    y: np.ndarray,
    epsilon_y: float,
) -> np.ndarray:
    """Return retained indices for y as a piecewise-linear function of x."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) <= 2:
        return np.arange(len(x), dtype=int)

    keep = {0, len(x) - 1}
    _rdp_series_recursive(x.astype(float), y.astype(float), 0, len(x) - 1, float(epsilon_y), keep)
    return np.array(sorted(keep), dtype=int)


def compress_breakpoints(
    times: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    geoaltitudes: np.ndarray,
    lateral_tolerance_m: float,
    altitude_tolerance_m: float,
) -> CompressionBreakpoints:
    """Compress lateral and altitude dimensions, then union their breakpoints."""
    if not (len(times) == len(latitudes) == len(longitudes) == len(geoaltitudes)):
        raise ValueError("times, latitudes, longitudes, and geoaltitudes must have the same length")

    if len(times) == 0:
        empty = np.array([], dtype=int)
        return CompressionBreakpoints(empty, empty, empty)

    x_m, y_m = project_latlon_to_local_m(latitudes.astype(float), longitudes.astype(float))
    lateral_indices = douglas_peucker_spatial_indices(x_m, y_m, lateral_tolerance_m)
    altitude_indices = douglas_peucker_series_indices(times.astype(float), geoaltitudes.astype(float), altitude_tolerance_m)
    minimal_indices = np.array(sorted(set(lateral_indices.tolist()) | set(altitude_indices.tolist())), dtype=int)

    return CompressionBreakpoints(
        lateral_indices=lateral_indices,
        altitude_indices=altitude_indices,
        minimal_indices=minimal_indices,
    )
