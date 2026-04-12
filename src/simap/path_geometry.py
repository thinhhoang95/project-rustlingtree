from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_interp_spline

EARTH_RADIUS_M = 6_371_000.0


def _wrap_angle_rad(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


@dataclass(frozen=True)
class ReferencePath:
    origin_lat_deg: float
    origin_lon_deg: float
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
        u = np.concatenate(([0.0], np.cumsum(chord_m)))

        if len(lat) == 2:
            sample_count = max(samples_per_segment + 1, 2)
            u_sample = np.linspace(u[0], u[-1], sample_count, dtype=float)
            east_sample = np.interp(u_sample, u, east_m)
            north_sample = np.interp(u_sample, u, north_m)
            dx_du = np.gradient(east_sample, u_sample, edge_order=1)
            dy_du = np.gradient(north_sample, u_sample, edge_order=1)
            ddx_du = np.zeros_like(dx_du)
            ddy_du = np.zeros_like(dy_du)
        else:
            k = min(3, len(lat) - 1)
            east_spline = make_interp_spline(u, east_m, k=k)
            north_spline = make_interp_spline(u, north_m, k=k)
            sample_count = max((len(lat) - 1) * samples_per_segment + 1, len(lat))
            u_sample = np.linspace(u[0], u[-1], sample_count, dtype=float)
            east_sample = east_spline(u_sample)
            north_sample = north_spline(u_sample)
            dx_du = east_spline(u_sample, 1)
            dy_du = north_spline(u_sample, 1)
            if k >= 2:
                ddx_du = east_spline(u_sample, 2)
                ddy_du = north_spline(u_sample, 2)
            else:
                ddx_du = np.zeros_like(dx_du)
                ddy_du = np.zeros_like(dy_du)

        segment_m = np.hypot(np.diff(east_sample), np.diff(north_sample))
        s_from_start_m = np.concatenate(([0.0], np.cumsum(segment_m)))
        keep = np.concatenate(([True], np.diff(s_from_start_m) > 1e-6))
        east_sample = east_sample[keep]
        north_sample = north_sample[keep]
        s_from_start_m = s_from_start_m[keep]
        dx_du = dx_du[keep]
        dy_du = dy_du[keep]
        ddx_du = ddx_du[keep]
        ddy_du = ddy_du[keep]

        speed_du = np.clip(dx_du**2 + dy_du**2, 1e-9, None)
        track_rad = np.unwrap(np.arctan2(dy_du, dx_du))
        curvature_inv_m = (dx_du * ddy_du - dy_du * ddx_du) / np.power(speed_du, 1.5)
        total_length_m = float(s_from_start_m[-1])
        s_m = total_length_m - s_from_start_m
        lat_sample = origin_lat_deg + np.rad2deg(north_sample / EARTH_RADIUS_M)
        lon_sample = origin_lon_deg + np.rad2deg(east_sample / (EARTH_RADIUS_M * np.cos(lat0_rad)))

        return cls(
            origin_lat_deg=origin_lat_deg,
            origin_lon_deg=origin_lon_deg,
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

    def position_ne(self, s_m: float) -> tuple[float, float]:
        return self._interp_for_s(self.east_m, s_m), self._interp_for_s(self.north_m, s_m)

    def latlon(self, s_m: float) -> tuple[float, float]:
        return self._interp_for_s(self.lat_deg, s_m), self._interp_for_s(self.lon_deg, s_m)

    def track_angle_rad(self, s_m: float) -> float:
        return _wrap_angle_rad(self._interp_for_s(self.track_rad, s_m))

    def curvature(self, s_m: float) -> float:
        return self._interp_for_s(self.curvature_inv_m, s_m)

    def tangent_hat(self, s_m: float) -> np.ndarray:
        track_rad = self.track_angle_rad(s_m)
        return np.asarray([np.cos(track_rad), np.sin(track_rad)], dtype=float)

    def normal_hat(self, s_m: float) -> np.ndarray:
        tangent = self.tangent_hat(s_m)
        return np.asarray([-tangent[1], tangent[0]], dtype=float)

    def latlon_from_ne(self, east_m: float, north_m: float) -> tuple[float, float]:
        lat0_rad = np.deg2rad(self.origin_lat_deg)
        lat_deg = self.origin_lat_deg + np.rad2deg(north_m / EARTH_RADIUS_M)
        lon_deg = self.origin_lon_deg + np.rad2deg(east_m / (EARTH_RADIUS_M * np.cos(lat0_rad)))
        return float(lat_deg), float(lon_deg)
