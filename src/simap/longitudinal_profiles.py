from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import km_to_m


@dataclass(frozen=True)
class ScalarProfile:
    s_m: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        s = np.asarray(self.s_m, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if s.ndim != 1 or y.ndim != 1:
            raise ValueError("ScalarProfile expects one-dimensional arrays")
        if len(s) != len(y):
            raise ValueError("s_m and y must have the same length")
        if len(s) < 2:
            raise ValueError("ScalarProfile requires at least two nodes")
        if np.any(np.diff(s) <= 0.0):
            raise ValueError("s_m must be strictly increasing")
        object.__setattr__(self, "s_m", s)
        object.__setattr__(self, "y", y)

    def value(self, s_m: float) -> float:
        s = float(np.clip(s_m, self.s_m[0], self.s_m[-1]))
        return float(np.interp(s, self.s_m, self.y))

    def values(self, s_m: np.ndarray) -> np.ndarray:
        points = np.asarray(s_m, dtype=float)
        clipped = np.clip(points, self.s_m[0], self.s_m[-1])
        return np.interp(clipped, self.s_m, self.y)

    def slope(self, s_m: float) -> float:
        s = float(np.clip(s_m, self.s_m[0], self.s_m[-1]))
        index = int(np.searchsorted(self.s_m, s))
        index = max(1, min(index, len(self.s_m) - 1))
        ds = self.s_m[index] - self.s_m[index - 1]
        dy = self.y[index] - self.y[index - 1]
        return float(dy / ds)


@dataclass(frozen=True)
class ConstraintEnvelope:
    s_m: np.ndarray
    h_lower_m: np.ndarray
    h_upper_m: np.ndarray
    cas_lower_mps: np.ndarray
    cas_upper_mps: np.ndarray
    gamma_lower_rad: np.ndarray | None = None
    gamma_upper_rad: np.ndarray | None = None
    thrust_lower_n: np.ndarray | None = None
    thrust_upper_n: np.ndarray | None = None
    cl_max: np.ndarray | None = None

    def __post_init__(self) -> None:
        s = np.asarray(self.s_m, dtype=float)
        if s.ndim != 1 or len(s) < 2:
            raise ValueError("ConstraintEnvelope requires a one-dimensional distance grid")
        if np.any(np.diff(s) <= 0.0):
            raise ValueError("ConstraintEnvelope distances must be strictly increasing")
        object.__setattr__(self, "s_m", s)
        required = (
            "h_lower_m",
            "h_upper_m",
            "cas_lower_mps",
            "cas_upper_mps",
        )
        optional = (
            "gamma_lower_rad",
            "gamma_upper_rad",
            "thrust_lower_n",
            "thrust_upper_n",
            "cl_max",
        )
        for name in required:
            object.__setattr__(self, name, self._coerce(name, getattr(self, name), required=True))
        for name in optional:
            object.__setattr__(self, name, self._coerce(name, getattr(self, name), required=False))
        if np.any(self.h_lower_m > self.h_upper_m):
            raise ValueError("altitude lower bounds must not exceed upper bounds")
        if np.any(self.cas_lower_mps > self.cas_upper_mps):
            raise ValueError("CAS lower bounds must not exceed upper bounds")
        if self.gamma_lower_rad is not None and self.gamma_upper_rad is not None:
            if np.any(self.gamma_lower_rad > self.gamma_upper_rad):
                raise ValueError("gamma lower bounds must not exceed upper bounds")
        if self.thrust_lower_n is not None and self.thrust_upper_n is not None:
            if np.any(self.thrust_lower_n > self.thrust_upper_n):
                raise ValueError("thrust lower bounds must not exceed upper bounds")

    def _coerce(self, name: str, values: np.ndarray | None, *, required: bool) -> np.ndarray | None:
        if values is None:
            if required:
                raise ValueError(f"{name} is required")
            return None
        arr = np.asarray(values, dtype=float)
        if arr.shape != self.s_m.shape:
            raise ValueError(f"{name} must have the same shape as s_m")
        return arr

    @classmethod
    def from_profiles(
        cls,
        *,
        altitude_lower: ScalarProfile,
        altitude_upper: ScalarProfile,
        cas_lower: ScalarProfile,
        cas_upper: ScalarProfile,
        gamma_lower: ScalarProfile | None = None,
        gamma_upper: ScalarProfile | None = None,
        thrust_lower: ScalarProfile | None = None,
        thrust_upper: ScalarProfile | None = None,
        cl_max: ScalarProfile | None = None,
    ) -> "ConstraintEnvelope":
        profiles = [altitude_lower, altitude_upper, cas_lower, cas_upper]
        for optional in (gamma_lower, gamma_upper, thrust_lower, thrust_upper, cl_max):
            if optional is not None:
                profiles.append(optional)
        s_m = np.unique(np.concatenate([profile.s_m for profile in profiles]))
        return cls(
            s_m=s_m,
            h_lower_m=altitude_lower.values(s_m),
            h_upper_m=altitude_upper.values(s_m),
            cas_lower_mps=cas_lower.values(s_m),
            cas_upper_mps=cas_upper.values(s_m),
            gamma_lower_rad=None if gamma_lower is None else gamma_lower.values(s_m),
            gamma_upper_rad=None if gamma_upper is None else gamma_upper.values(s_m),
            thrust_lower_n=None if thrust_lower is None else thrust_lower.values(s_m),
            thrust_upper_n=None if thrust_upper is None else thrust_upper.values(s_m),
            cl_max=None if cl_max is None else cl_max.values(s_m),
        )

    def _interp(self, values: np.ndarray | None, s_m: float, *, fallback: float | None = None) -> float | None:
        if values is None:
            return fallback
        s = float(np.clip(s_m, self.s_m[0], self.s_m[-1]))
        return float(np.interp(s, self.s_m, values))

    def h_bounds(self, s_m: float) -> tuple[float, float]:
        return self._interp(self.h_lower_m, s_m), self._interp(self.h_upper_m, s_m)

    def cas_bounds(self, s_m: float) -> tuple[float, float]:
        return self._interp(self.cas_lower_mps, s_m), self._interp(self.cas_upper_mps, s_m)

    def gamma_bounds(self, s_m: float) -> tuple[float | None, float | None]:
        return self._interp(self.gamma_lower_rad, s_m), self._interp(self.gamma_upper_rad, s_m)

    def thrust_bounds(self, s_m: float) -> tuple[float | None, float | None]:
        return self._interp(self.thrust_lower_n, s_m), self._interp(self.thrust_upper_n, s_m)

    def cl_upper(self, s_m: float) -> float | None:
        return self._interp(self.cl_max, s_m)


def build_speed_schedule_from_wrap(
    wrap,
    s_nodes_km: tuple[float, ...] = (0.0, 8.0, 30.0, 60.0),
) -> ScalarProfile:
    from .openap_adapter import wrap_default

    v_des_mps = wrap_default(wrap, "descent_const_vcas")
    v_final_mps = wrap_default(wrap, "finalapp_vcas")
    v_land_mps = wrap_default(wrap, "landing_speed")

    s_km = np.asarray(s_nodes_km, dtype=float)
    v_mps = np.asarray([v_land_mps, v_final_mps, v_des_mps, v_des_mps], dtype=float)
    if len(s_km) != len(v_mps):
        raise ValueError("s_nodes_km must provide four schedule anchors")
    return ScalarProfile(s_m=km_to_m(s_km), y=v_mps)
