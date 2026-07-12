"""Historical speed reconstruction and executable profile helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.ndimage import median_filter
from sklearn.isotonic import IsotonicRegression  # pyright: ignore[reportMissingImports]

from hailmary.config import MPS_PER_KNOT
from hailmary.errors import ArtifactValidationError


def finite_difference_ground_speed(
    time_s: np.ndarray,
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    max_gap_s: float = 180.0,
    minimum_speed_mps: float = 20.0,
    maximum_speed_mps: float = 180.0,
) -> np.ndarray:
    """Estimate point speeds from displacement while rejecting bad intervals."""

    time = np.asarray(time_s, dtype=float)
    east = np.asarray(east_m, dtype=float)
    north = np.asarray(north_m, dtype=float)
    if time.ndim != 1 or len(time) < 2 or len(east) != len(time) or len(north) != len(time):
        raise ArtifactValidationError("time/east/north must be equal-length one-dimensional arrays")
    dt = np.diff(time)
    distance = np.hypot(np.diff(east), np.diff(north))
    valid = (dt > 0.0) & (dt <= max_gap_s)
    interval = np.full(len(dt), np.nan, dtype=float)
    interval[valid] = distance[valid] / dt[valid]
    interval[(interval < minimum_speed_mps) | (interval > maximum_speed_mps)] = np.nan
    valid_values = interval[np.isfinite(interval)]
    if len(valid_values) == 0:
        raise ArtifactValidationError("track has no valid finite-difference speed intervals")

    median = float(np.median(valid_values))
    mad = float(np.median(np.abs(valid_values - median)))
    if mad > 1e-9:
        robust_z = np.abs(interval - median) / (1.4826 * mad)
        interval[robust_z > 5.0] = np.nan
    good = np.flatnonzero(np.isfinite(interval))
    interval = np.interp(np.arange(len(interval), dtype=float), good.astype(float), interval[good])

    point_speed = np.empty(len(time), dtype=float)
    point_speed[0] = interval[0]
    point_speed[-1] = interval[-1]
    if len(time) > 2:
        point_speed[1:-1] = 0.5 * (interval[:-1] + interval[1:])
    return point_speed


def robust_smooth(values: np.ndarray, *, window: int = 7) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ArtifactValidationError("smooth input must be a finite one-dimensional profile")
    size = min(max(1, int(window)), len(array) if len(array) % 2 == 1 else len(array) - 1)
    size = max(1, size)
    return np.asarray(median_filter(array, size=size, mode="nearest"), dtype=float)


def monotone_command_profile(s_m: np.ndarray, cas_mps: np.ndarray) -> np.ndarray:
    """Make command speed non-increasing in the downstream flight direction.

    ``s_m`` is ascending threshold-to-upstream, so this is an increasing
    isotonic fit as a function of station.
    """

    stations = np.asarray(s_m, dtype=float)
    cas = np.asarray(cas_mps, dtype=float)
    if len(stations) != len(cas) or len(stations) < 2 or np.any(np.diff(stations) <= 0.0):
        raise ArtifactValidationError("station/profile arrays are invalid for isotonic reconstruction")
    fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(stations, cas)
    return np.asarray(fitted, dtype=float)


def tas_to_cas(tas_mps: np.ndarray, altitude_m: np.ndarray) -> np.ndarray:
    from simap import aero

    tas = np.asarray(tas_mps, dtype=float)
    altitude = np.asarray(altitude_m, dtype=float)
    if len(tas) != len(altitude):
        raise ArtifactValidationError("TAS and altitude arrays must have equal length")
    return np.asarray(
        [aero.tas2cas(float(speed), float(height)) for speed, height in zip(tas, altitude, strict=True)],
        dtype=float,
    )


def cas_to_tas(cas_mps: np.ndarray, altitude_m: np.ndarray) -> np.ndarray:
    from simap import aero

    cas = np.asarray(cas_mps, dtype=float)
    altitude = np.asarray(altitude_m, dtype=float)
    if len(cas) != len(altitude):
        raise ArtifactValidationError("CAS and altitude arrays must have equal length")
    return np.asarray(
        [aero.cas2tas(float(speed), float(height)) for speed, height in zip(cas, altitude, strict=True)],
        dtype=float,
    )


def cas_envelope(
    s_m: np.ndarray,
    altitude_m: np.ndarray,
    reference_cas_mps: np.ndarray,
    *,
    bounds_at_station: Callable[[float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stations = np.asarray(s_m, dtype=float)
    altitude = np.asarray(altitude_m, dtype=float)
    reference = np.asarray(reference_cas_mps, dtype=float)
    if not (len(stations) == len(altitude) == len(reference)):
        raise ArtifactValidationError("station, altitude, and CAS arrays must have equal length")
    if bounds_at_station is None:
        lower = np.maximum(35.0, np.minimum(reference - 25.0 * MPS_PER_KNOT, reference * 0.80))
        upper = np.maximum(reference, 250.0 * MPS_PER_KNOT)
    else:
        bounds = np.asarray([bounds_at_station(float(station)) for station in stations], dtype=float)
        lower = bounds[:, 0]
        upper = bounds[:, 1]
    below_ten_thousand = altitude < 10_000.0 * 0.3048
    upper = np.where(below_ten_thousand, np.minimum(upper, 250.0 * MPS_PER_KNOT), upper)
    if np.any(~np.isfinite(lower)) or np.any(lower <= 0.0):
        raise ArtifactValidationError("lower CAS envelope is invalid")
    if np.any(upper < lower):
        raise ArtifactValidationError("upper CAS envelope is below the lower envelope")
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def clamp_reference_to_envelope(
    reference_cas_mps: np.ndarray,
    lower_cas_mps: np.ndarray,
    upper_cas_mps: np.ndarray,
    *,
    max_excursion_kts: float,
    max_clamped_fraction: float,
) -> tuple[np.ndarray, float, float]:
    reference = np.asarray(reference_cas_mps, dtype=float)
    lower = np.asarray(lower_cas_mps, dtype=float)
    upper = np.asarray(upper_cas_mps, dtype=float)
    clipped = np.clip(reference, lower, upper)
    excursions = np.abs(clipped - reference)
    max_excursion = float(np.max(excursions, initial=0.0))
    fraction = float(np.count_nonzero(excursions > 1e-9) / max(1, len(reference)))
    if max_excursion > max_excursion_kts * MPS_PER_KNOT + 1e-9:
        worst_index = int(np.argmax(excursions))
        direction = "below" if reference[worst_index] < lower[worst_index] else "above"
        raise ArtifactValidationError(
            "historical CAS exceeds the envelope by more than the clamp budget "
            f"(max_excursion_kts={max_excursion / MPS_PER_KNOT:.3f}, "
            f"allowed_kts={max_excursion_kts:.3f}, direction={direction}, "
            f"station_index={worst_index})"
        )
    if fraction > max_clamped_fraction + 1e-12:
        raise ArtifactValidationError(
            "historical CAS requires clamping at too many stations "
            f"(clamped_fraction={fraction:.6f}, allowed_fraction={max_clamped_fraction:.6f})"
        )
    return clipped, max_excursion, fraction


def integrate_elapsed_time(s_m: np.ndarray, ground_speed_mps: np.ndarray) -> np.ndarray:
    stations = np.asarray(s_m, dtype=float)
    speed = np.asarray(ground_speed_mps, dtype=float)
    if len(stations) != len(speed) or len(stations) < 2:
        raise ArtifactValidationError("station and ground-speed profiles are incompatible")
    if np.any(np.diff(stations) <= 0.0) or np.any(speed <= 0.0):
        raise ArtifactValidationError("station must increase and ground speed must be positive")
    elapsed = np.zeros(len(stations), dtype=float)
    for index in range(len(stations) - 2, -1, -1):
        elapsed[index] = elapsed[index + 1] + (stations[index + 1] - stations[index]) / max(
            0.5 * (speed[index + 1] + speed[index]),
            1e-9,
        )
    return elapsed
