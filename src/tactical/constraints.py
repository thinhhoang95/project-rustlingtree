from __future__ import annotations

import numpy as np

from simap import ConstraintEnvelope, ScalarProfile, build_speed_schedule_from_wrap
from simap.units import ft_to_m, kts_to_mps

from .models import AltitudeConstraint


def _merge_altitude_constraints(
    *,
    s_nodes_m: list[float],
    lower_m: list[float],
    upper_m: list[float],
    constraint_s_m: dict[str, float],
    constraints: tuple[AltitudeConstraint, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_by_s: dict[float, tuple[float | None, float | None]] = {}
    for constraint in constraints:
        identifier = constraint.fix_identifier.upper()
        if identifier not in constraint_s_m:
            continue
        s_m = float(constraint_s_m[identifier])
        existing_lower, existing_upper = values_by_s.get(s_m, (None, None))
        next_lower = None if constraint.lower_ft is None else ft_to_m(constraint.lower_ft)
        next_upper = None if constraint.upper_ft is None else ft_to_m(constraint.upper_ft)
        if existing_lower is not None and next_lower is not None:
            next_lower = max(existing_lower, next_lower)
        elif next_lower is None:
            next_lower = existing_lower
        if existing_upper is not None and next_upper is not None:
            next_upper = min(existing_upper, next_upper)
        elif next_upper is None:
            next_upper = existing_upper
        values_by_s[s_m] = (next_lower, next_upper)

    if not values_by_s:
        return np.asarray(s_nodes_m, dtype=float), np.asarray(lower_m, dtype=float), np.asarray(upper_m, dtype=float)

    all_s = np.unique(np.concatenate([np.asarray(s_nodes_m, dtype=float), np.asarray(list(values_by_s), dtype=float)]))
    base_lower = np.interp(all_s, np.asarray(s_nodes_m, dtype=float), np.asarray(lower_m, dtype=float))
    base_upper = np.interp(all_s, np.asarray(s_nodes_m, dtype=float), np.asarray(upper_m, dtype=float))
    for s_m, (constraint_lower, constraint_upper) in values_by_s.items():
        idx = int(np.where(np.isclose(all_s, s_m))[0][0])
        if constraint_lower is not None:
            base_lower[idx] = max(base_lower[idx], constraint_lower)
        if constraint_upper is not None:
            base_upper[idx] = min(base_upper[idx], constraint_upper)
        if base_lower[idx] > base_upper[idx]:
            raise ValueError(f"incompatible altitude bounds at s={s_m:.1f} m")
    return all_s, base_lower, base_upper


def build_tactical_constraint_envelope(
    *,
    total_path_length_m: float,
    threshold_altitude_m: float,
    upstream_altitude_m: float,
    threshold_cas_mps: float,
    upstream_cas_kts: float,
    openap_wrap,
    altitude_constraints: tuple[AltitudeConstraint, ...] = (),
    constraint_s_m: dict[str, float] | None = None,
) -> ConstraintEnvelope:
    max_s_m = float(max(total_path_length_m, 60_000.0))
    altitude_capture_s_m = min(5_000.0, max_s_m)
    s_alt = [0.0, altitude_capture_s_m, max_s_m]
    altitude_lower = [threshold_altitude_m, threshold_altitude_m, threshold_altitude_m]
    altitude_upper = [
        threshold_altitude_m + ft_to_m(500.0),
        upstream_altitude_m + ft_to_m(2_000.0),
        upstream_altitude_m + ft_to_m(2_000.0),
    ]
    if altitude_constraints:
        s_alt_arr, altitude_lower_arr, altitude_upper_arr = _merge_altitude_constraints(
            s_nodes_m=s_alt,
            lower_m=altitude_lower,
            upper_m=altitude_upper,
            constraint_s_m={} if constraint_s_m is None else constraint_s_m,
            constraints=altitude_constraints,
        )
    else:
        s_alt_arr = np.asarray(s_alt, dtype=float)
        altitude_lower_arr = np.asarray(altitude_lower, dtype=float)
        altitude_upper_arr = np.asarray(altitude_upper, dtype=float)

    speed_schedule = build_speed_schedule_from_wrap(openap_wrap)
    upstream_cas_mps = kts_to_mps(upstream_cas_kts)
    cas_s = np.unique(np.concatenate([speed_schedule.s_m, np.asarray([max_s_m], dtype=float)]))
    scheduled_cas = speed_schedule.values(cas_s)
    cas_lower = np.maximum(scheduled_cas - kts_to_mps(12.0), threshold_cas_mps)
    cas_upper = np.maximum(scheduled_cas, upstream_cas_mps)

    return ConstraintEnvelope.from_profiles(
        altitude_lower=ScalarProfile(s_m=s_alt_arr, y=altitude_lower_arr),
        altitude_upper=ScalarProfile(s_m=s_alt_arr, y=altitude_upper_arr),
        cas_lower=ScalarProfile(s_m=cas_s, y=cas_lower),
        cas_upper=ScalarProfile(s_m=cas_s, y=cas_upper),
        gamma_lower=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([-np.deg2rad(5.5), -np.deg2rad(0.25)], dtype=float),
        ),
        gamma_upper=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([-np.deg2rad(1.5), np.deg2rad(1.0)], dtype=float),
        ),
    )
