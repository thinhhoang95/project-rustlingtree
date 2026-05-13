from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
# A modest nominal bank keeps the generated fly-by path trackable in approach
# modes after the roll loop and bank limits are applied.
_DEFAULT_FLYBY_BANK_RAD = float(np.deg2rad(12.0))
_DEFAULT_FLYBY_SPEED_MPS = 230.0 * 0.514444
_MIN_FLYBY_TURN_RAD = float(np.deg2rad(3.0))
_MAX_FLYBY_TURN_RAD = float(np.deg2rad(165.0))
_MAX_FLYBY_LEG_FRACTION = 0.45
_MAX_FLYBY_LEAD_M = 7.0 * 1_852.0


def _wrap_angle_rad(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def _left_normal(vector: np.ndarray) -> np.ndarray:
    return np.asarray([-vector[1], vector[0]], dtype=float)


def _append_line(
    samples: list[np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    *,
    target_spacing_m: float,
) -> None:
    length_m = float(np.linalg.norm(end - start))
    if length_m <= 1e-6:
        return
    count = max(2, int(np.ceil(length_m / target_spacing_m)) + 1)
    for fraction in np.linspace(0.0, 1.0, count):
        point = start + float(fraction) * (end - start)
        if samples and np.linalg.norm(point - samples[-1]) <= 1e-6:
            continue
        samples.append(point)


def _append_arc(
    samples: list[np.ndarray],
    center: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    turn_sign: float,
    radius_m: float,
    target_spacing_m: float,
) -> None:
    start_angle = float(np.arctan2(start[1] - center[1], start[0] - center[0]))
    end_angle = float(np.arctan2(end[1] - center[1], end[0] - center[0]))
    if turn_sign > 0.0:
        while end_angle <= start_angle:
            end_angle += 2.0 * np.pi
    else:
        while end_angle >= start_angle:
            end_angle -= 2.0 * np.pi
    sweep_rad = abs(end_angle - start_angle)
    count = max(3, int(np.ceil(radius_m * sweep_rad / target_spacing_m)) + 1)
    for angle in np.linspace(start_angle, end_angle, count):
        point = center + radius_m * np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
        if samples and np.linalg.norm(point - samples[-1]) <= 1e-6:
            continue
        samples.append(point)


def _flyby_turn_radius_m() -> float:
    return float(_DEFAULT_FLYBY_SPEED_MPS**2 / (9.80665 * np.tan(_DEFAULT_FLYBY_BANK_RAD)))


def _build_flyby_samples(points_ne: np.ndarray, *, samples_per_segment: int) -> np.ndarray:
    chord_m = np.hypot(np.diff(points_ne[:, 0]), np.diff(points_ne[:, 1]))
    total_chord_m = float(np.sum(chord_m))
    target_count = max((len(points_ne) - 1) * samples_per_segment, len(points_ne) - 1)
    target_spacing_m = float(np.clip(total_chord_m / target_count, 60.0, 250.0))
    nominal_radius_m = _flyby_turn_radius_m()

    turn_starts: list[np.ndarray | None] = [None] * len(points_ne)
    turn_ends: list[np.ndarray | None] = [None] * len(points_ne)
    turn_centers: list[np.ndarray | None] = [None] * len(points_ne)
    turn_signs: list[float] = [0.0] * len(points_ne)
    turn_radii: list[float] = [0.0] * len(points_ne)

    for index in range(1, len(points_ne) - 1):
        previous_point = points_ne[index - 1]
        waypoint = points_ne[index]
        next_point = points_ne[index + 1]
        inbound = waypoint - previous_point
        outbound = next_point - waypoint
        inbound_length_m = float(np.linalg.norm(inbound))
        outbound_length_m = float(np.linalg.norm(outbound))
        if inbound_length_m <= 0.0 or outbound_length_m <= 0.0:
            continue
        inbound_hat = inbound / inbound_length_m
        outbound_hat = outbound / outbound_length_m
        turn_cross = float(inbound_hat[0] * outbound_hat[1] - inbound_hat[1] * outbound_hat[0])
        turn_dot = float(np.clip(np.dot(inbound_hat, outbound_hat), -1.0, 1.0))
        turn_angle_rad = abs(float(np.arctan2(turn_cross, turn_dot)))
        if turn_angle_rad < _MIN_FLYBY_TURN_RAD or turn_angle_rad > _MAX_FLYBY_TURN_RAD:
            continue

        requested_lead_m = nominal_radius_m * float(np.tan(0.5 * turn_angle_rad))
        max_lead_m = min(
            _MAX_FLYBY_LEAD_M,
            _MAX_FLYBY_LEG_FRACTION * inbound_length_m,
            _MAX_FLYBY_LEG_FRACTION * outbound_length_m,
        )
        lead_m = float(min(requested_lead_m, max_lead_m))
        if lead_m <= 1.0:
            continue

        radius_m = lead_m / float(np.tan(0.5 * turn_angle_rad))
        turn_sign = 1.0 if turn_cross > 0.0 else -1.0
        turn_start = waypoint - lead_m * inbound_hat
        turn_end = waypoint + lead_m * outbound_hat
        center = turn_start + turn_sign * radius_m * _left_normal(inbound_hat)

        turn_starts[index] = turn_start
        turn_ends[index] = turn_end
        turn_centers[index] = center
        turn_signs[index] = turn_sign
        turn_radii[index] = radius_m

    samples: list[np.ndarray] = [points_ne[0]]
    line_start = points_ne[0]
    for index in range(1, len(points_ne) - 1):
        turn_start = turn_starts[index]
        turn_end = turn_ends[index]
        turn_center = turn_centers[index]
        if turn_start is None or turn_end is None or turn_center is None:
            line_end = points_ne[index]
            _append_line(samples, line_start, line_end, target_spacing_m=target_spacing_m)
            line_start = line_end
            continue

        _append_line(samples, line_start, turn_start, target_spacing_m=target_spacing_m)
        _append_arc(
            samples,
            turn_center,
            turn_start,
            turn_end,
            turn_sign=turn_signs[index],
            radius_m=turn_radii[index],
            target_spacing_m=target_spacing_m,
        )
        line_start = turn_end

    _append_line(samples, line_start, points_ne[-1], target_spacing_m=target_spacing_m)
    return np.asarray(samples, dtype=float)


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
        waypoint_ne = np.column_stack([east_m, north_m])

        chord_m = np.hypot(np.diff(east_m), np.diff(north_m))
        if np.any(chord_m <= 0.0):
            raise ValueError("waypoints must be unique and ordered")
        route_length_m = float(np.sum(chord_m))
        path_ne = _build_flyby_samples(waypoint_ne, samples_per_segment=samples_per_segment)
        east_sample = path_ne[:, 0]
        north_sample = path_ne[:, 1]
        sample_chord_m = np.hypot(np.diff(east_sample), np.diff(north_sample))
        physical_s_from_start_m = np.concatenate(([0.0], np.cumsum(sample_chord_m)))
        physical_length_m = float(physical_s_from_start_m[-1])
        if physical_length_m <= 0.0:
            raise ValueError("reference path must have positive length")
        s_from_start_m = physical_s_from_start_m * (route_length_m / physical_length_m)
        total_length_m = route_length_m

        segment_track_rad = np.arctan2(np.diff(north_sample), np.diff(east_sample))
        track_sample = np.empty(len(east_sample), dtype=float)
        track_sample[0] = float(segment_track_rad[0])
        track_sample[-1] = float(segment_track_rad[-1])
        if len(track_sample) > 2:
            unwrapped_segment_track = np.unwrap(segment_track_rad)
            track_sample[1:-1] = 0.5 * (unwrapped_segment_track[:-1] + unwrapped_segment_track[1:])
        curvature_inv_m = np.gradient(track_sample, s_from_start_m, edge_order=1)
        track_rad = np.unwrap(track_sample)
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
