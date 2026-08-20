from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_TIME_NAMES = ("relative_elapsed_time_s", "elapsed_time_s", "t_s", "time_s")


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    elapsed_time_s: float
    s_m: float
    east_m: float | None = None
    north_m: float | None = None
    lat_deg: float | None = None
    lon_deg: float | None = None
    altitude_m: float | None = None
    cas_mps: float | None = None
    tas_mps: float | None = None
    ground_speed_mps: float | None = None


@dataclass(frozen=True, slots=True)
class MonotoneTrajectory:
    """Validated time-order view of a trajectory variant.

    Variants may be stored either in flight-time order (decreasing ``s_m``) or
    on the canonical Hailmary station grid (increasing ``s_m`` with decreasing
    elapsed time).  This view normalizes both representations to increasing
    elapsed time and decreasing remaining distance.
    """

    variant: object
    elapsed_time_s: np.ndarray
    s_m: np.ndarray
    source_indices: np.ndarray

    @classmethod
    def from_variant(cls, variant: object) -> "MonotoneTrajectory":
        raw_s = _required_array(variant, ("s_m",))
        raw_time = _required_array(variant, _TIME_NAMES)
        if raw_s.ndim != 1 or raw_time.ndim != 1:
            raise ValueError("trajectory station/time arrays must be one-dimensional")
        if len(raw_s) != len(raw_time) or len(raw_s) < 2:
            raise ValueError("trajectory station/time arrays must have the same length of at least two")
        if not np.isfinite(raw_s).all() or not np.isfinite(raw_time).all():
            raise ValueError("trajectory station/time arrays must be finite")

        time_delta = np.diff(raw_time)
        if np.all(time_delta > 0.0):
            source_indices = np.arange(len(raw_time), dtype=np.int64)
        elif np.all(time_delta < 0.0):
            source_indices = np.arange(len(raw_time) - 1, -1, -1, dtype=np.int64)
        else:
            raise ValueError("trajectory elapsed-time coordinates must be strictly monotone")

        ordered_time = np.asarray(raw_time[source_indices], dtype=np.float64)
        ordered_time = ordered_time - float(ordered_time[0])
        ordered_s = np.asarray(raw_s[source_indices], dtype=np.float64)
        if np.any(np.diff(ordered_s) >= 0.0):
            raise ValueError("remaining distance s_m must strictly decrease as elapsed time increases")
        if float(ordered_s[-1]) < -1e-9:
            raise ValueError("trajectory remaining distance cannot be negative")

        for array in (ordered_time, ordered_s, source_indices):
            array.setflags(write=False)
        return cls(
            variant=variant,
            elapsed_time_s=ordered_time,
            s_m=ordered_s,
            source_indices=source_indices,
        )

    @property
    def duration_s(self) -> float:
        return float(self.elapsed_time_s[-1])

    @property
    def upstream_s_m(self) -> float:
        return float(self.s_m[0])

    @property
    def downstream_s_m(self) -> float:
        return float(self.s_m[-1])

    def elapsed_at_station(self, s_m: float, *, clip: bool = False) -> float:
        station = float(s_m)
        if not np.isfinite(station):
            raise ValueError("s_m must be finite")
        lower, upper = self.downstream_s_m, self.upstream_s_m
        if not clip and not lower - 1e-9 <= station <= upper + 1e-9:
            raise ValueError(f"station {station} lies outside trajectory range [{lower}, {upper}]")
        station = float(np.clip(station, lower, upper))
        return float(np.interp(station, self.s_m[::-1], self.elapsed_time_s[::-1]))

    def station_at_elapsed(self, elapsed_time_s: float, *, clip: bool = True) -> float:
        elapsed = float(elapsed_time_s)
        if not np.isfinite(elapsed):
            raise ValueError("elapsed_time_s must be finite")
        if not clip and not -1e-9 <= elapsed <= self.duration_s + 1e-9:
            raise ValueError(f"elapsed time {elapsed} lies outside [0, {self.duration_s}]")
        elapsed = float(np.clip(elapsed, 0.0, self.duration_s))
        return float(np.interp(elapsed, self.elapsed_time_s, self.s_m))

    def value_at_elapsed(
        self,
        names: tuple[str, ...],
        elapsed_time_s: float,
        *,
        required: bool = False,
    ) -> float | None:
        values = _optional_array(self.variant, names)
        if values is None:
            if required:
                raise ValueError(f"trajectory is missing one of fields {names!r}")
            return None
        if values.ndim != 1 or len(values) != len(self.source_indices):
            raise ValueError(f"trajectory field {names!r} must match the station/time grid")
        ordered = np.asarray(values[self.source_indices], dtype=np.float64)
        if not np.isfinite(ordered).all():
            raise ValueError(f"trajectory field {names!r} must be finite")
        elapsed = float(np.clip(float(elapsed_time_s), 0.0, self.duration_s))
        return float(np.interp(elapsed, self.elapsed_time_s, ordered))

    def sample(
        self,
        elapsed_time_s: float,
        *,
        clip: bool = True,
    ) -> TrajectorySample:
        elapsed = float(elapsed_time_s)
        if not np.isfinite(elapsed):
            raise ValueError("elapsed_time_s must be finite")
        if not clip and not -1e-9 <= elapsed <= self.duration_s + 1e-9:
            raise ValueError(
                f"elapsed time {elapsed} lies outside [0, {self.duration_s}]"
            )
        elapsed = float(np.clip(elapsed, 0.0, self.duration_s))
        return TrajectorySample(
            elapsed_time_s=elapsed,
            s_m=self.station_at_elapsed(elapsed),
            east_m=self.value_at_elapsed(("east_m",), elapsed),
            north_m=self.value_at_elapsed(("north_m",), elapsed),
            lat_deg=self.value_at_elapsed(("lat_deg",), elapsed),
            lon_deg=self.value_at_elapsed(("lon_deg",), elapsed),
            altitude_m=self.value_at_elapsed(("altitude_m", "h_m", "geoaltitude_m"), elapsed),
            cas_mps=self.value_at_elapsed(("cas_mps", "v_cas_mps"), elapsed),
            tas_mps=self.value_at_elapsed(("tas_mps", "v_tas_mps"), elapsed),
            ground_speed_mps=self.value_at_elapsed(("ground_speed_mps",), elapsed),
        )


def elapsed_time_at_station(variant: object, s_m: float, *, clip: bool = False) -> float:
    return MonotoneTrajectory.from_variant(variant).elapsed_at_station(s_m, clip=clip)


def station_at_elapsed_time(variant: object, elapsed_time_s: float, *, clip: bool = True) -> float:
    return MonotoneTrajectory.from_variant(variant).station_at_elapsed(elapsed_time_s, clip=clip)


def trajectory_duration_s(variant: object) -> float:
    return MonotoneTrajectory.from_variant(variant).duration_s


def _required_array(variant: object, names: tuple[str, ...]) -> np.ndarray:
    value = _get_value(variant, names)
    if value is None:
        raise ValueError(f"trajectory variant is missing one of fields {names!r}")
    return np.asarray(value, dtype=np.float64)


def _optional_array(variant: object, names: tuple[str, ...]) -> np.ndarray | None:
    value = _get_value(variant, names)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _get_value(variant: object, names: tuple[str, ...]) -> Any | None:
    if isinstance(variant, Mapping):
        for name in names:
            if name in variant:
                return variant[name]
    for name in names:
        if hasattr(variant, name):
            return getattr(variant, name)
    return None


__all__ = [
    "MonotoneTrajectory",
    "TrajectorySample",
    "elapsed_time_at_station",
    "station_at_elapsed_time",
    "trajectory_duration_s",
]
