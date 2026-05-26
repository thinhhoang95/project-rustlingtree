from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class LocalProjection:
    origin_lat_deg: float
    origin_lon_deg: float

    def project(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lat = np.radians(np.asarray(lat_deg, dtype=float))
        lon = np.radians(np.asarray(lon_deg, dtype=float))
        lat0 = np.radians(self.origin_lat_deg)
        lon0 = np.radians(self.origin_lon_deg)
        x_m = EARTH_RADIUS_M * np.cos(lat0) * (lon - lon0)
        y_m = EARTH_RADIUS_M * (lat - lat0)
        return x_m, y_m

    def unproject(self, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lat0 = np.radians(self.origin_lat_deg)
        lon0 = np.radians(self.origin_lon_deg)
        lat = lat0 + np.asarray(y_m, dtype=float) / EARTH_RADIUS_M
        lon = lon0 + np.asarray(x_m, dtype=float) / (EARTH_RADIUS_M * np.cos(lat0))
        return np.degrees(lat), np.degrees(lon)


def projection_from_latlon(lat_deg: np.ndarray, lon_deg: np.ndarray) -> LocalProjection:
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    if not finite.any():
        raise ValueError("cannot build local projection from empty lat/lon data")
    return LocalProjection(
        origin_lat_deg=float(np.median(lat[finite])),
        origin_lon_deg=float(np.median(lon[finite])),
    )


def cumulative_distance_m(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    if x.size == 0:
        return np.array([], dtype=float)
    segment = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate(([0.0], np.cumsum(segment)))


def resample_polyline_by_fraction(
    x_m: np.ndarray,
    y_m: np.ndarray,
    station_fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    stations = np.asarray(station_fractions, dtype=float)
    if x.size != y.size:
        raise ValueError("x_m and y_m must have the same length")
    if x.size < 2:
        raise ValueError("at least two points are required to resample a trajectory")

    distance = cumulative_distance_m(x, y)
    total = float(distance[-1])
    if total <= 0.0:
        raise ValueError("trajectory must have positive path length")

    normalized = distance / total
    unique_station, unique_indices = np.unique(normalized, return_index=True)
    if unique_station.size < 2:
        raise ValueError("trajectory must contain at least two unique stations")
    return (
        np.interp(stations, unique_station, x[unique_indices]),
        np.interp(stations, unique_station, y[unique_indices]),
    )


def reference_tangent_normal(reference_xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_xy_m, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("reference_xy_m must have shape M x 2")
    if reference.shape[0] < 2:
        raise ValueError("reference path must contain at least two stations")

    dx = np.gradient(reference[:, 0])
    dy = np.gradient(reference[:, 1])
    norm = np.hypot(dx, dy)
    fallback = norm <= 1e-12
    if fallback.any():
        norm[fallback] = 1.0
    tangent = np.column_stack((dx / norm, dy / norm))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return tangent, normal


def normal_deviation_m(samples_xy_m: np.ndarray, reference_xy_m: np.ndarray, normals_xy: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples_xy_m, dtype=float)
    reference = np.asarray(reference_xy_m, dtype=float)
    normals = np.asarray(normals_xy, dtype=float)
    if samples.shape != reference.shape or reference.shape != normals.shape:
        raise ValueError("samples, reference, and normals must share shape M x 2")
    return np.sum((samples - reference) * normals, axis=1)
