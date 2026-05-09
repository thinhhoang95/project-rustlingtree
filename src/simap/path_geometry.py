from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def _wrap_angle_rad(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def _mean_angle_rad(a_rad: float, b_rad: float) -> float:
    return float(np.arctan2(np.sin(a_rad) + np.sin(b_rad), np.cos(a_rad) + np.cos(b_rad)))


@dataclass(frozen=True)
class ReferencePath:
    origin_lat_deg: float
    origin_lon_deg: float
    waypoint_lat_deg: np.ndarray
    waypoint_lon_deg: np.ndarray
    s_from_start_m: np.ndarray
    s_m: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    track_rad: np.ndarray
    curvature_inv_m: np.ndarray
    total_length_m: float

    def __post_init__(self) -> None:
        waypoint_arrays = (self.waypoint_lat_deg, self.waypoint_lon_deg)
        waypoint_lengths = {len(np.asarray(array)) for array in waypoint_arrays}
        if len(waypoint_lengths) != 1:
            raise ValueError("waypoint arrays must have the same length")
        if not waypoint_lengths or min(waypoint_lengths) < 2:
            raise ValueError("ReferencePath requires at least two waypoints")

        arrays = (
            self.s_from_start_m,
            self.s_m,
            self.east_m,
            self.north_m,
            self.lat_deg,
            self.lon_deg,
            self.track_rad,
            self.curvature_inv_m,
        )
        lengths = {len(np.asarray(array)) for array in arrays}
        if len(lengths) != 1:
            raise ValueError("reference-path arrays must have the same length")
        if not lengths or min(lengths) < 2:
            raise ValueError("ReferencePath requires at least two samples")
        if np.any(np.diff(self.s_from_start_m) <= 0.0):
            raise ValueError("s_from_start_m must be strictly increasing")
        if np.any(np.diff(self.s_m) >= 0.0):
            raise ValueError("s_m must be strictly decreasing")

    @classmethod
    def from_geographic(
        cls,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        *,
        samples_per_segment: int = 48,
    ) -> "ReferencePath":
        lat = np.asarray(lat_deg, dtype=float)
        lon = np.asarray(lon_deg, dtype=float)
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("geographic coordinates must be one-dimensional")
        if len(lat) != len(lon):
            raise ValueError("lat_deg and lon_deg must have the same length")
        if len(lat) < 2:
            raise ValueError("ReferencePath requires at least two waypoints")

        origin_lat_deg = float(lat[-1])
        origin_lon_deg = float(lon[-1])
        lat0_rad = np.deg2rad(origin_lat_deg)
        east_m = EARTH_RADIUS_M * np.cos(lat0_rad) * np.deg2rad(lon - origin_lon_deg)
        north_m = EARTH_RADIUS_M * np.deg2rad(lat - origin_lat_deg)

        chord_m = np.hypot(np.diff(east_m), np.diff(north_m))
        if np.any(chord_m <= 0.0):
            raise ValueError("waypoints must be unique and ordered")
        waypoint_s_from_start_m = np.concatenate(([0.0], np.cumsum(chord_m)))

        segment_track_rad = np.arctan2(np.diff(north_m), np.diff(east_m))
        waypoint_track_rad = np.empty(len(lat), dtype=float)
        waypoint_track_rad[0] = float(segment_track_rad[0])
        waypoint_track_rad[-1] = float(segment_track_rad[-1])
        for index in range(1, len(lat) - 1):
            waypoint_track_rad[index] = _mean_angle_rad(float(segment_track_rad[index - 1]), float(segment_track_rad[index]))

        total_length_m = float(waypoint_s_from_start_m[-1])
        sample_count = max((len(lat) - 1) * samples_per_segment + 1, len(lat))
        s_from_start_m = np.linspace(0.0, total_length_m, sample_count, dtype=float)
        east_sample = np.interp(s_from_start_m, waypoint_s_from_start_m, east_m)
        north_sample = np.interp(s_from_start_m, waypoint_s_from_start_m, north_m)
        track_sample = np.interp(s_from_start_m, waypoint_s_from_start_m, np.unwrap(waypoint_track_rad))
        curvature_inv_m = np.gradient(track_sample, s_from_start_m, edge_order=1)
        track_rad = track_sample
        s_m = total_length_m - s_from_start_m
        lat_sample = origin_lat_deg + np.rad2deg(north_sample / EARTH_RADIUS_M)
        lon_sample = origin_lon_deg + np.rad2deg(east_sample / (EARTH_RADIUS_M * np.cos(lat0_rad)))

        return cls(
            origin_lat_deg=origin_lat_deg,
            origin_lon_deg=origin_lon_deg,
            waypoint_lat_deg=lat,
            waypoint_lon_deg=lon,
            s_from_start_m=s_from_start_m,
            s_m=s_m,
            east_m=east_sample,
            north_m=north_sample,
            lat_deg=lat_sample,
            lon_deg=lon_sample,
            track_rad=track_rad,
            curvature_inv_m=curvature_inv_m,
            total_length_m=total_length_m,
        )

    def _interp_for_s(self, values: np.ndarray, s_m: float) -> float:
        s = float(np.clip(s_m, 0.0, self.total_length_m))
        return float(np.interp(s, self.s_m[::-1], values[::-1]))

    def _interp_many_for_s(self, values: np.ndarray, s_m: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s_m, dtype=float), 0.0, self.total_length_m)
        return np.asarray(np.interp(s, self.s_m[::-1], values[::-1]), dtype=float)

    def position_ne(self, s_m: float) -> tuple[float, float]:
        return self._interp_for_s(self.east_m, s_m), self._interp_for_s(self.north_m, s_m)

    def position_ne_many(self, s_m: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                self._interp_many_for_s(self.east_m, s_m),
                self._interp_many_for_s(self.north_m, s_m),
            ]
        )

    def project_s_m(self, east_m: float, north_m: float) -> float:
        """Return the remaining path distance for the closest point on the path."""
        point = np.asarray([float(east_m), float(north_m)], dtype=float)
        starts = np.column_stack([self.east_m[:-1], self.north_m[:-1]])
        ends = np.column_stack([self.east_m[1:], self.north_m[1:]])
        segments = ends - starts
        segment_lengths_sq = np.einsum("ij,ij->i", segments, segments)
        with np.errstate(divide="ignore", invalid="ignore"):
            fractions = np.einsum("ij,ij->i", point - starts, segments) / segment_lengths_sq
        fractions = np.clip(np.nan_to_num(fractions, nan=0.0), 0.0, 1.0)
        closest = starts + fractions[:, np.newaxis] * segments
        distances_sq = np.einsum("ij,ij->i", closest - point, closest - point)
        index = int(np.argmin(distances_sq))
        s_from_start = float(
            self.s_from_start_m[index]
            + fractions[index] * (self.s_from_start_m[index + 1] - self.s_from_start_m[index])
        )
        return float(np.clip(self.total_length_m - s_from_start, 0.0, self.total_length_m))

    def latlon(self, s_m: float) -> tuple[float, float]:
        return self._interp_for_s(self.lat_deg, s_m), self._interp_for_s(self.lon_deg, s_m)

    def track_angle_rad(self, s_m: float) -> float:
        return _wrap_angle_rad(self._interp_for_s(self.track_rad, s_m))

    def track_angle_rad_many(self, s_m: np.ndarray) -> np.ndarray:
        track = self._interp_many_for_s(self.track_rad, s_m)
        return np.arctan2(np.sin(track), np.cos(track))

    def curvature(self, s_m: float) -> float:
        return self._interp_for_s(self.curvature_inv_m, s_m)

    def curvature_many(self, s_m: np.ndarray) -> np.ndarray:
        return self._interp_many_for_s(self.curvature_inv_m, s_m)

    def tangent_hat(self, s_m: float) -> np.ndarray:
        track_rad = self.track_angle_rad(s_m)
        return np.asarray([np.cos(track_rad), np.sin(track_rad)], dtype=float)

    def tangent_hat_many(self, s_m: np.ndarray) -> np.ndarray:
        track_rad = self.track_angle_rad_many(s_m)
        return np.column_stack([np.cos(track_rad), np.sin(track_rad)])

    def normal_hat(self, s_m: float) -> np.ndarray:
        tangent = self.tangent_hat(s_m)
        return np.asarray([-tangent[1], tangent[0]], dtype=float)

    def normal_hat_many(self, s_m: np.ndarray) -> np.ndarray:
        tangent = self.tangent_hat_many(s_m)
        return np.column_stack([-tangent[:, 1], tangent[:, 0]])

    def latlon_from_ne(self, east_m: float, north_m: float) -> tuple[float, float]:
        lat0_rad = np.deg2rad(self.origin_lat_deg)
        lat_deg = self.origin_lat_deg + np.rad2deg(north_m / EARTH_RADIUS_M)
        lon_deg = self.origin_lon_deg + np.rad2deg(east_m / (EARTH_RADIUS_M * np.cos(lat0_rad)))
        return float(lat_deg), float(lon_deg)

    def latlon_from_ne_many(self, east_m: np.ndarray, north_m: np.ndarray) -> np.ndarray:
        lat0_rad = np.deg2rad(self.origin_lat_deg)
        lat_deg = self.origin_lat_deg + np.rad2deg(np.asarray(north_m, dtype=float) / EARTH_RADIUS_M)
        lon_deg = self.origin_lon_deg + np.rad2deg(np.asarray(east_m, dtype=float) / (EARTH_RADIUS_M * np.cos(lat0_rad)))
        return np.column_stack([lat_deg, lon_deg])
