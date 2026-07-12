"""Narrow, immutable boundary between Hailmary and public SIMAP APIs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

import numpy as np

from hailmary._arrays import readonly_float64, readonly_int64
from hailmary.config import MPS_PER_KNOT
from hailmary.geometry import LocalFrame
from hailmary.ids import stable_id
from hailmary.templates.models import ResourceCrossing, TrajectoryVariant, VariantDiagnostics
from simap import aero
from simap.backends import EffectivePolarBackend, PerformanceBackend
from simap.calibration import build_default_aircraft_config, suggest_approach_mass_kg
from simap.config import AircraftConfig, bank_limit_rad, mode_for_s, planned_cas_bounds_mps
from simap.nlp_colloc import (
    CoupledDescentPlanResult,
    CoupledDescentSolveProfile,
    SimulationRequest,
    SimulationResult,
    simulate_plan,
)
from simap.openap_adapter import OpenAPAircraftData, OpenAPObjects, extract_aircraft_data, load_openap
from simap.path_geometry import ReferencePath
from simap.weather import ConstantWeather, WeatherProvider, alongtrack_wind_mps


@dataclass(frozen=True)
class A320Context:
    """Resolved, cacheable version-1 aircraft assumption."""

    aircraft_config: AircraftConfig
    openap: OpenAPObjects
    aircraft_data: OpenAPAircraftData
    payload_kg: float
    approach_mass_kg: float

    @property
    def engine_name(self) -> str:
        return self.aircraft_config.engine_name


@lru_cache(maxsize=8)
def get_cached_a320_context(
    payload_kg: float = 12_000.0,
    engine_name: str | None = None,
) -> A320Context:
    """Build the declared OpenAP A320 assumption once per payload/engine."""

    payload = float(payload_kg)
    if not np.isfinite(payload) or payload < 0.0:
        raise ValueError("payload_kg must be finite and nonnegative")
    openap = load_openap("A320", engine_name=engine_name)
    aircraft_data = extract_aircraft_data(openap)
    mass_kg = suggest_approach_mass_kg(aircraft_data, payload_kg=payload)
    aircraft_config, resolved_openap = build_default_aircraft_config(
        "A320",
        mass_kg=mass_kg,
        engine_name=engine_name,
        openap_objects=openap,
    )
    return A320Context(
        aircraft_config=aircraft_config,
        openap=resolved_openap,
        aircraft_data=aircraft_data,
        payload_kg=payload,
        approach_mass_kg=mass_kg,
    )


def get_cached_a320_config(
    payload_kg: float = 12_000.0,
    engine_name: str | None = None,
) -> AircraftConfig:
    return get_cached_a320_context(payload_kg, engine_name).aircraft_config


def clear_a320_cache() -> None:
    get_cached_a320_context.cache_clear()
    _cached_a320_envelope.cache_clear()


@dataclass(frozen=True)
class CASEnvelope:
    s_m: np.ndarray
    lower_cas_mps: np.ndarray
    upper_cas_mps: np.ndarray

    def __post_init__(self) -> None:
        station = readonly_float64(self.s_m, name="s_m")
        lower = readonly_float64(self.lower_cas_mps, name="lower_cas_mps")
        upper = readonly_float64(self.upper_cas_mps, name="upper_cas_mps")
        if len(station) == 0 or len(lower) != len(station) or len(upper) != len(station):
            raise ValueError("CAS-envelope arrays must have equal, nonzero lengths")
        if np.any(station < 0.0):
            raise ValueError("CAS-envelope stations cannot be negative")
        if np.any(lower <= 0.0) or np.any(upper < lower):
            raise ValueError("CAS envelope is invalid")
        object.__setattr__(self, "s_m", station)
        object.__setattr__(self, "lower_cas_mps", lower)
        object.__setattr__(self, "upper_cas_mps", upper)


def _evaluate_cas_envelope(
    station: np.ndarray,
    altitude: np.ndarray | None,
    config: AircraftConfig,
) -> CASEnvelope:
    lower = np.empty(len(station), dtype=np.float64)
    upper = np.empty(len(station), dtype=np.float64)
    vmo_mps = float(config.vmo_kts) * MPS_PER_KNOT
    below_10k_cap_mps = 250.0 * MPS_PER_KNOT
    for index, value in enumerate(station):
        lower_value, upper_value = planned_cas_bounds_mps(config, float(value))
        lower[index] = lower_value
        finite_upper = vmo_mps if not np.isfinite(upper_value) else min(upper_value, vmo_mps)
        if altitude is not None and altitude[index] < 10_000.0 * 0.3048:
            finite_upper = min(finite_upper, below_10k_cap_mps)
        upper[index] = finite_upper
    return CASEnvelope(station, lower, upper)


@lru_cache(maxsize=128)
def _cached_a320_envelope(
    station_key: tuple[float, ...],
    altitude_key: tuple[float, ...] | None,
    payload_kg: float,
) -> CASEnvelope:
    station = readonly_float64(station_key, name="s_m")
    altitude = (
        None
        if altitude_key is None
        else readonly_float64(altitude_key, name="altitude_m")
    )
    return _evaluate_cas_envelope(station, altitude, get_cached_a320_config(payload_kg))


def get_cached_a320_envelope(
    s_m: object,
    *,
    altitude_m: object | None = None,
    payload_kg: float = 12_000.0,
) -> CASEnvelope:
    """Return a shared immutable envelope for the default A320 assumption."""

    station = readonly_float64(s_m, name="s_m")
    if np.any(station < 0.0):
        raise ValueError("s_m cannot contain negative stations")
    if altitude_m is None:
        altitude = None
    else:
        altitude = readonly_float64(altitude_m, name="altitude_m")
        if len(altitude) != len(station):
            raise ValueError("altitude_m must match s_m")
    station_key = tuple(float(item) for item in station)
    altitude_key = None if altitude is None else tuple(float(item) for item in altitude)
    return _cached_a320_envelope(station_key, altitude_key, float(payload_kg))


def planned_a320_cas_envelope(
    s_m: object,
    *,
    altitude_m: object | None = None,
    aircraft_config: AircraftConfig | None = None,
    payload_kg: float = 12_000.0,
) -> CASEnvelope:
    """Evaluate public SIMAP bounds plus VMO/10,000-ft caps pointwise."""

    if aircraft_config is None:
        return get_cached_a320_envelope(s_m, altitude_m=altitude_m, payload_kg=payload_kg)
    station = readonly_float64(s_m, name="s_m")
    if np.any(station < 0.0):
        raise ValueError("s_m cannot contain negative stations")
    if altitude_m is None:
        altitude = None
    else:
        altitude = readonly_float64(altitude_m, name="altitude_m")
        if len(altitude) != len(station):
            raise ValueError("altitude_m must match s_m")
    return _evaluate_cas_envelope(station, altitude, aircraft_config)


@dataclass(frozen=True)
class SimplifiedReferencePath:
    """A public SIMAP path plus frozen simplification provenance."""

    reference_path: ReferencePath
    control_lat_deg: np.ndarray
    control_lon_deg: np.ndarray
    source_indices: np.ndarray
    lateral_tolerance_m: float

    def __post_init__(self) -> None:
        lat = readonly_float64(self.control_lat_deg, name="control_lat_deg")
        lon = readonly_float64(self.control_lon_deg, name="control_lon_deg")
        indices = readonly_int64(self.source_indices, name="source_indices")
        if len(lat) < 2 or len(lat) != len(lon) or len(lat) != len(indices):
            raise ValueError("reference-path controls must have equal lengths of at least two")
        if not np.isfinite(self.lateral_tolerance_m) or self.lateral_tolerance_m < 0.0:
            raise ValueError("lateral_tolerance_m must be finite and nonnegative")
        object.__setattr__(self, "control_lat_deg", lat)
        object.__setattr__(self, "control_lon_deg", lon)
        object.__setattr__(self, "source_indices", indices)

    @property
    def total_length_m(self) -> float:
        return float(self.reference_path.total_length_m)


def _douglas_peucker_indices(points_m: np.ndarray, tolerance_m: float) -> np.ndarray:
    keep = {0, len(points_m) - 1}
    pending = [(0, len(points_m) - 1)]
    while pending:
        start, stop = pending.pop()
        if stop <= start + 1:
            continue
        segment = points_m[stop] - points_m[start]
        length_sq = float(np.dot(segment, segment))
        candidates = points_m[start + 1 : stop]
        if length_sq <= 1.0e-12:
            distance = np.linalg.norm(candidates - points_m[start], axis=1)
        else:
            fraction = np.clip(
                np.einsum("ij,j->i", candidates - points_m[start], segment) / length_sq,
                0.0,
                1.0,
            )
            closest = points_m[start] + fraction[:, np.newaxis] * segment
            distance = np.linalg.norm(candidates - closest, axis=1)
        relative = int(np.argmax(distance))
        if float(distance[relative]) > tolerance_m:
            index = start + 1 + relative
            keep.add(index)
            pending.extend(((start, index), (index, stop)))
    return np.asarray(sorted(keep), dtype=np.int64)


def _freeze_reference_path(path: ReferencePath) -> ReferencePath:
    array_names = (
        "waypoint_lat_deg",
        "waypoint_lon_deg",
        "s_from_start_m",
        "s_m",
        "east_m",
        "north_m",
        "lat_deg",
        "lon_deg",
    )
    values = {
        name: readonly_float64(getattr(path, name), name=f"reference_path.{name}")
        for name in array_names
    }
    # ``ReferencePath.from_geographic`` exposes an unwrapped track array, but
    # older SIMAP releases compute curvature from the track samples *before*
    # unwrapping them.  A westbound leg can consequently acquire a false
    # 2*pi/spacing curvature impulse at the +/-pi boundary.  Keep this repair
    # inside the Hailmary adapter boundary: the executable path geometry stays
    # public SIMAP output, while its derivative is taken from the public,
    # unwrapped track representation.
    track_rad = readonly_float64(
        np.unwrap(np.asarray(path.track_rad, dtype=np.float64)),
        name="reference_path.track_rad",
    )
    curvature_inv_m = readonly_float64(
        np.gradient(track_rad, values["s_from_start_m"], edge_order=1),
        name="reference_path.curvature_inv_m",
    )
    return ReferencePath(
        origin_lat_deg=path.origin_lat_deg,
        origin_lon_deg=path.origin_lon_deg,
        total_length_m=path.total_length_m,
        track_rad=track_rad,
        curvature_inv_m=curvature_inv_m,
        **values,
    )


def simplify_reference_path(
    lat_deg: object,
    lon_deg: object,
    *,
    threshold_lat_deg: float | None = None,
    threshold_lon_deg: float | None = None,
    lateral_tolerance_m: float = 250.0,
    minimum_waypoint_spacing_m: float = 100.0,
    maximum_waypoints: int = 64,
    threshold_capture_radius_m: float = 2.0 * 1_852.0,
    samples_per_segment: int = 48,
) -> SimplifiedReferencePath:
    """Reduce a dense medoid to stable controls and build ``ReferencePath``."""

    lat = readonly_float64(lat_deg, name="lat_deg")
    lon = readonly_float64(lon_deg, name="lon_deg")
    if len(lat) != len(lon) or len(lat) < 2:
        raise ValueError("lat_deg and lon_deg must have equal lengths of at least two")
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 180.0):
        raise ValueError("geographic coordinates are invalid")
    if not np.isfinite(lateral_tolerance_m) or lateral_tolerance_m < 0.0:
        raise ValueError("lateral_tolerance_m must be finite and nonnegative")
    if not np.isfinite(minimum_waypoint_spacing_m) or minimum_waypoint_spacing_m < 0.0:
        raise ValueError("minimum_waypoint_spacing_m must be finite and nonnegative")
    if maximum_waypoints < 2:
        raise ValueError("maximum_waypoints must be at least two")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least two")
    if (threshold_lat_deg is None) != (threshold_lon_deg is None):
        raise ValueError("threshold latitude and longitude must be supplied together")

    if threshold_lat_deg is None:
        threshold_lat = float(lat[-1])
        threshold_lon = float(lon[-1])
    else:
        assert threshold_lon_deg is not None
        threshold_lat = float(threshold_lat_deg)
        threshold_lon = float(threshold_lon_deg)
        if not np.isfinite(threshold_lat) or not np.isfinite(threshold_lon):
            raise ValueError("threshold coordinates must be finite")
        endpoint_frame = LocalFrame(threshold_lat, threshold_lon)
        endpoints = endpoint_frame.project_points(lat[[0, -1]], lon[[0, -1]])
        if np.linalg.norm(endpoints[0]) < np.linalg.norm(endpoints[-1]):
            lat = readonly_float64(lat[::-1], name="oriented lat_deg")
            lon = readonly_float64(lon[::-1], name="oriented lon_deg")
        end_east, end_north = endpoint_frame.project(lat[-1:], lon[-1:])
        endpoint_distance = float(np.hypot(end_east[0], end_north[0]))
        if endpoint_distance > threshold_capture_radius_m:
            raise ValueError("path endpoint is outside the threshold capture radius")
        if endpoint_distance > 1.0e-6:
            lat = readonly_float64(np.append(lat, threshold_lat), name="threshold-aligned lat_deg")
            lon = readonly_float64(np.append(lon, threshold_lon), name="threshold-aligned lon_deg")

    frame = LocalFrame(threshold_lat, threshold_lon)
    points = frame.project_points(lat, lon)
    neighbor_distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    unique_mask = np.concatenate(([True], neighbor_distance > 1.0e-3))
    source_indices = np.flatnonzero(unique_mask).astype(np.int64)
    points = points[unique_mask]
    lat = lat[unique_mask]
    lon = lon[unique_mask]
    if len(points) < 2:
        raise ValueError("path has fewer than two distinct coordinates")

    selected = _douglas_peucker_indices(points, float(lateral_tolerance_m))
    if minimum_waypoint_spacing_m > 0.0 and len(selected) > 2:
        spaced = [int(selected[0])]
        for index in selected[1:-1]:
            if np.linalg.norm(points[index] - points[spaced[-1]]) >= minimum_waypoint_spacing_m:
                spaced.append(int(index))
        endpoint = int(selected[-1])
        if len(spaced) > 1 and np.linalg.norm(points[endpoint] - points[spaced[-1]]) < minimum_waypoint_spacing_m:
            spaced.pop()
        spaced.append(endpoint)
        selected = np.asarray(spaced, dtype=np.int64)
    if len(selected) > maximum_waypoints:
        positions = np.linspace(0, len(selected) - 1, maximum_waypoints)
        selected = selected[np.unique(np.rint(positions).astype(np.int64))]
    controls_lat = lat[selected]
    controls_lon = lon[selected]
    control_source_indices = source_indices[selected]
    reference_path = _freeze_reference_path(
        ReferencePath.from_geographic(
            lat_deg=np.asarray(controls_lat, dtype=np.float64),
            lon_deg=np.asarray(controls_lon, dtype=np.float64),
            samples_per_segment=samples_per_segment,
        )
    )
    if not np.all(np.isfinite(reference_path.curvature_inv_m)):
        raise ValueError("SIMAP ReferencePath produced non-finite curvature")
    if np.any(np.diff(reference_path.s_m) >= 0.0):
        raise ValueError("SIMAP ReferencePath station order is not strictly decreasing")
    return SimplifiedReferencePath(
        reference_path=reference_path,
        control_lat_deg=controls_lat,
        control_lon_deg=controls_lon,
        source_indices=control_source_indices,
        lateral_tolerance_m=float(lateral_tolerance_m),
    )


def _diagnostics_from_variant(
    variant: object,
    *,
    feasible: bool,
    message: str,
    details: tuple[tuple[str, Any], ...],
    compiled_duration_s: float | None = None,
    signed_timing_error_s: float | None = None,
    absolute_timing_error_s: float | None = None,
    max_physical_cas_acceleration_mps2: float | None = None,
) -> VariantDiagnostics:
    previous = getattr(variant, "diagnostics", None)
    merged_details = dict(getattr(previous, "details", ()))
    merged_details.update(details)
    return VariantDiagnostics(
        feasible=feasible,
        message=message,
        speed_source=str(getattr(previous, "speed_source", "compiled_kinematic")),
        wind_model=str(getattr(previous, "wind_model", "zero_wind")),
        cas_derivation=str(
            getattr(previous, "cas_derivation", "ground_speed_as_tas_then_openap")
        ),
        observed_duration_s=getattr(previous, "observed_duration_s", None),
        compiled_duration_s=(
            getattr(previous, "compiled_duration_s", None)
            if compiled_duration_s is None
            else float(compiled_duration_s)
        ),
        signed_timing_error_s=(
            getattr(previous, "signed_timing_error_s", None)
            if signed_timing_error_s is None
            else float(signed_timing_error_s)
        ),
        absolute_timing_error_s=(
            getattr(previous, "absolute_timing_error_s", None)
            if absolute_timing_error_s is None
            else float(absolute_timing_error_s)
        ),
        max_command_envelope_excursion_mps=float(
            getattr(previous, "max_command_envelope_excursion_mps", 0.0)
        ),
        clamped_fraction=float(getattr(previous, "clamped_fraction", 0.0)),
        max_physical_cas_acceleration_mps2=getattr(
            previous, "max_physical_cas_acceleration_mps2", None
        )
        if max_physical_cas_acceleration_mps2 is None
        else float(
            max_physical_cas_acceleration_mps2
        ),
        details=tuple(sorted(merged_details.items())),
    )


@dataclass(frozen=True)
class _ReplayEvaluation:
    reference_path: SimplifiedReferencePath
    result: SimulationResult
    raw_max_bank_ratio: float
    maximum_contiguous_overbank_distance_m: float
    maximum_contiguous_overbank_duration_s: float
    bank_demand_sample_count: int
    thrust_saturation_fraction: float
    kinematic_duration_s: float


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    positive = denominator > 1.0e-9
    result[positive] = numerator[positive] / denominator[positive]
    result[~positive & (numerator > 1.0e-9)] = np.inf
    return result


@dataclass(frozen=True)
class _BankDemandEvaluation:
    raw_max_ratio: float
    maximum_contiguous_overbank_distance_m: float
    maximum_contiguous_overbank_duration_s: float
    sample_count: int


@dataclass(frozen=True)
class _ThresholdCalibratedReplay:
    result: SimulationResult
    pass_count: int
    first_pass_threshold_error_m: float
    first_pass_alongtrack_overshoot_m: float
    calibrated_threshold_tolerance_m: float


def _bank_demand_evaluation(
    *,
    s_m: np.ndarray,
    bank_ratio: np.ndarray,
    ground_speed_mps: np.ndarray,
    overbank_ratio_threshold: float,
) -> _BankDemandEvaluation:
    """Measure over-bank persistence on a physical-distance/time grid.

    A sample-count percentile changes when an otherwise identical path is
    resampled.  This calculation instead treats bank ratio as piecewise linear
    in station and reports the longest contiguous distance and traversal time
    above the configured ratio.  A zero-width isolated spike therefore has no
    physical persistence, while a tight turn remains invariant to Hailmary's
    executable station density.
    """

    station = np.asarray(s_m, dtype=np.float64)
    ratio = np.asarray(bank_ratio, dtype=np.float64)
    speed = np.asarray(ground_speed_mps, dtype=np.float64)
    if (
        len(station) < 2
        or len(ratio) != len(station)
        or len(speed) != len(station)
        or np.any(np.diff(station) <= 0.0)
        or np.any(speed <= 0.0)
    ):
        raise ValueError("bank-demand evaluation requires an ordered physical profile")

    maximum_distance_m = 0.0
    maximum_duration_s = 0.0
    current_distance_m = 0.0
    current_duration_s = 0.0
    threshold = float(overbank_ratio_threshold)
    for index in range(len(station) - 1):
        start_ratio = float(ratio[index])
        end_ratio = float(ratio[index + 1])
        distance_m = float(station[index + 1] - station[index])
        duration_s = distance_m / max(
            0.5 * float(speed[index] + speed[index + 1]),
            1.0e-9,
        )

        if start_ratio <= threshold and end_ratio <= threshold:
            current_distance_m = 0.0
            current_duration_s = 0.0
            continue
        if start_ratio > threshold and end_ratio > threshold:
            above_fraction = 1.0
            starts_excursion = False
            ends_excursion = False
        else:
            crossing_fraction = float(
                np.clip(
                    (threshold - start_ratio) / (end_ratio - start_ratio),
                    0.0,
                    1.0,
                )
            )
            if start_ratio <= threshold:
                above_fraction = 1.0 - crossing_fraction
                starts_excursion = True
                ends_excursion = False
            else:
                above_fraction = crossing_fraction
                starts_excursion = False
                ends_excursion = True

        if starts_excursion:
            current_distance_m = 0.0
            current_duration_s = 0.0
        current_distance_m += above_fraction * distance_m
        current_duration_s += above_fraction * duration_s
        maximum_distance_m = max(maximum_distance_m, current_distance_m)
        maximum_duration_s = max(maximum_duration_s, current_duration_s)
        if ends_excursion:
            current_distance_m = 0.0
            current_duration_s = 0.0

    return _BankDemandEvaluation(
        raw_max_ratio=float(np.max(ratio, initial=0.0)),
        maximum_contiguous_overbank_distance_m=maximum_distance_m,
        maximum_contiguous_overbank_duration_s=maximum_duration_s,
        sample_count=len(station),
    )


def _simulate_with_threshold_calibration(
    request: SimulationRequest,
) -> _ThresholdCalibratedReplay:
    """Replay once, then compensate deterministic longitudinal threshold lag.

    SIMAP advances remaining distance and lateral map position in separate
    integration channels.  On a curved arrival the map position can therefore
    pass the physical threshold while the longitudinal state reaches ``s=0``.
    A deterministic second pass stops by the measured first-pass overshoot
    along the public threshold tangent.  Cross-track error is never hidden by
    this calibration and remains subject to the normal replay quality gate.
    """

    first = simulate_plan(request)
    threshold_east_m, threshold_north_m = request.reference_path.position_ne(0.0)
    final_displacement_m = np.asarray(
        (
            float(first.east_m[-1]) - threshold_east_m,
            float(first.north_m[-1]) - threshold_north_m,
        ),
        dtype=np.float64,
    )
    first_pass_alongtrack_overshoot_m = max(
        0.0,
        float(
            np.dot(
                final_displacement_m,
                request.reference_path.tangent_hat(0.0),
            )
        ),
    )
    calibrated_tolerance_m = float(
        request.threshold_tolerance_m + first_pass_alongtrack_overshoot_m
    )
    if (
        not first.success
        or first_pass_alongtrack_overshoot_m <= request.threshold_tolerance_m
    ):
        return _ThresholdCalibratedReplay(
            result=first,
            pass_count=1,
            first_pass_threshold_error_m=first.final_threshold_error_m,
            first_pass_alongtrack_overshoot_m=first_pass_alongtrack_overshoot_m,
            calibrated_threshold_tolerance_m=request.threshold_tolerance_m,
        )

    second = simulate_plan(
        replace(
            request,
            threshold_tolerance_m=calibrated_tolerance_m,
        )
    )
    return _ThresholdCalibratedReplay(
        result=second,
        pass_count=2,
        first_pass_threshold_error_m=first.final_threshold_error_m,
        first_pass_alongtrack_overshoot_m=first_pass_alongtrack_overshoot_m,
        calibrated_threshold_tolerance_m=calibrated_tolerance_m,
    )


def _reference_plan(
    variant: TrajectoryVariant,
    *,
    geometry: SimplifiedReferencePath,
    cfg: AircraftConfig,
    perf: PerformanceBackend,
    weather: WeatherProvider,
    overbank_ratio_threshold: float,
) -> tuple[CoupledDescentPlanResult, _BankDemandEvaluation, float]:
    """Build a public SIMAP replay plan from a Hailmary command reference.

    This is deliberately a replay plan, not a claim that the nonlinear SIMAP
    optimizer produced the historical command.  Required thrust is reconstructed
    from the command profile and clipped to public aircraft thrust bounds; SIMAP's
    time-domain replay then exposes the resulting physical tracking response.
    """

    original_length_m = float(variant.path_length_m)
    reference_length_m = float(geometry.total_length_m)
    plan_s_m = np.asarray(
        variant.s_m * (reference_length_m / original_length_m),
        dtype=np.float64,
    )
    # Flight-time order is upstream -> threshold. Rate-limit a newly accepted
    # command so a step reduction becomes a physical deceleration transient
    # rather than an impossible instantaneous CAS jump.
    command_time_s = np.asarray(variant.elapsed_time_s[::-1], dtype=np.float64)
    flight_h_m = np.asarray(variant.altitude_m[::-1], dtype=np.float64)
    commanded_flight_tas_mps = np.asarray(
        [
            aero.cas2tas(
                float(cas),
                float(altitude),
                dT=float(weather.delta_isa_K(float(station), float(altitude), float(time_s))),
            )
            for station, time_s, altitude, cas in zip(
                plan_s_m[::-1],
                command_time_s,
                flight_h_m,
                variant.command_cas_mps[::-1],
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    flight_tas_mps = np.empty_like(commanded_flight_tas_mps)
    flight_tas_mps[0] = commanded_flight_tas_mps[0]
    flight_plan_s_m = plan_s_m[::-1]
    flight_track_rad = geometry.reference_path.track_angle_rad_many(flight_plan_s_m)
    for index in range(1, len(flight_tas_mps)):
        previous_station = float(flight_plan_s_m[index - 1])
        distance_m = float(previous_station - flight_plan_s_m[index])
        previous_altitude = float(flight_h_m[index - 1])
        altitude_change_m = float(flight_h_m[index] - previous_altitude)
        gamma_estimate_rad = float(np.arctan2(altitude_change_m, max(distance_m, 1.0e-9)))
        previous_tas_mps = float(flight_tas_mps[index - 1])
        commanded_ground_mps = max(
            1.0,
            previous_tas_mps
            + alongtrack_wind_mps(
                weather,
                float(flight_track_rad[index - 1]),
                previous_station,
                previous_altitude,
                float(command_time_s[index - 1]),
            ),
        )
        interval_s = distance_m / max(commanded_ground_mps, 1.0)
        mode = mode_for_s(cfg, previous_station)
        delta_isa_K = float(
            weather.delta_isa_K(
                previous_station,
                previous_altitude,
                float(command_time_s[index - 1]),
            )
        )
        drag_n = float(
            perf.drag_newtons(
                mode=mode,
                mass_kg=cfg.mass_kg,
                wing_area_m2=cfg.wing_area_m2,
                v_tas_mps=previous_tas_mps,
                h_m=previous_altitude,
                gamma_rad=gamma_estimate_rad,
                bank_rad=0.0,
                delta_isa_K=delta_isa_K,
            )
        )
        lower_thrust_n, _upper_thrust_n = perf.thrust_bounds_newtons(
            mode=mode,
            v_tas_mps=previous_tas_mps,
            h_m=previous_altitude,
            delta_isa_K=delta_isa_K,
        )
        minimum_acceleration_mps2 = float(
            (float(lower_thrust_n) - drag_n) / cfg.mass_kg
            - float(aero.g0) * np.sin(gamma_estimate_rad)
        )
        available_deceleration_mps2 = float(
            np.clip(-minimum_acceleration_mps2, 0.0, cfg.a_acc_max_mps2)
        )
        deceleration_floor = (
            flight_tas_mps[index - 1] - available_deceleration_mps2 * interval_s
        )
        flight_tas_mps[index] = max(commanded_flight_tas_mps[index], deceleration_floor)
    flight_cas_mps = np.asarray(
        [
            aero.tas2cas(
                float(tas),
                float(altitude),
                dT=float(weather.delta_isa_K(float(station), float(altitude), float(time_s))),
            )
            for station, time_s, altitude, tas in zip(
                plan_s_m[::-1],
                command_time_s,
                flight_h_m,
                flight_tas_mps,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    flight_ground_speed_mps = np.asarray(
        [
            max(
                1.0,
                float(tas)
                + alongtrack_wind_mps(
                    weather,
                    float(track),
                    float(station),
                    float(altitude),
                    float(time_s),
                ),
            )
            for station, time_s, altitude, tas, track in zip(
                flight_plan_s_m,
                command_time_s,
                flight_h_m,
                flight_tas_mps,
                flight_track_rad,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    physical_elapsed_time_s = np.zeros(len(plan_s_m), dtype=np.float64)
    for index in range(len(plan_s_m) - 2, -1, -1):
        ds_m = float(plan_s_m[index + 1] - plan_s_m[index])
        mean_ground_speed_mps = 0.5 * float(
            flight_ground_speed_mps[::-1][index + 1]
            + flight_ground_speed_mps[::-1][index]
        )
        physical_elapsed_time_s[index] = (
            physical_elapsed_time_s[index + 1]
            + ds_m / max(mean_ground_speed_mps, 1.0e-9)
        )
    physical_duration_s = float(physical_elapsed_time_s[0])
    plan_t_s = np.asarray(
        physical_duration_s - physical_elapsed_time_s,
        dtype=np.float64,
    )
    flight_time_s = np.asarray(plan_t_s[-1] - plan_t_s[::-1], dtype=np.float64)
    h_dot_mps = np.gradient(flight_h_m, flight_time_s, edge_order=1)
    v_dot_mps2 = np.gradient(flight_tas_mps, flight_time_s, edge_order=1)
    gamma_rad = np.arcsin(
        np.clip(h_dot_mps / np.maximum(flight_tas_mps, 1.0), -0.95, 0.95)
    )

    flight_curvature_inv_m = geometry.reference_path.curvature_many(flight_plan_s_m)
    raw_bank_rad = np.arctan(
        np.maximum(flight_tas_mps, 1.0) ** 2 * flight_curvature_inv_m / float(aero.g0)
    )
    bank_limit_values = np.asarray(
        [
            bank_limit_rad(cfg, mode_for_s(cfg, float(station)), float(cas))
            for station, cas in zip(flight_plan_s_m, flight_cas_mps, strict=True)
        ],
        dtype=np.float64,
    )
    clipped_bank_rad = np.clip(raw_bank_rad, -bank_limit_values, bank_limit_values)

    # Evaluate turn demand on the union of the public ReferencePath samples and
    # the command grid.  ReferencePath samples are distance-bounded; including
    # command stations preserves speed/mode transitions.  The resulting bank
    # persistence is independent of how densely a stretch variant happens to
    # store its executable arrays.
    bank_evaluation_s_m = np.unique(
        np.concatenate(
            (
                np.asarray(geometry.reference_path.s_m, dtype=np.float64),
                plan_s_m,
            )
        )
    )
    bank_evaluation_tas_mps = np.interp(
        bank_evaluation_s_m,
        plan_s_m,
        flight_tas_mps[::-1],
    )
    bank_evaluation_cas_mps = np.interp(
        bank_evaluation_s_m,
        plan_s_m,
        flight_cas_mps[::-1],
    )
    bank_evaluation_ground_speed_mps = np.interp(
        bank_evaluation_s_m,
        plan_s_m,
        flight_ground_speed_mps[::-1],
    )
    bank_evaluation_curvature_inv_m = geometry.reference_path.curvature_many(
        bank_evaluation_s_m
    )
    bank_evaluation_raw_rad = np.arctan(
        np.maximum(bank_evaluation_tas_mps, 1.0) ** 2
        * bank_evaluation_curvature_inv_m
        / float(aero.g0)
    )
    bank_evaluation_limit_rad = np.asarray(
        [
            bank_limit_rad(cfg, mode_for_s(cfg, float(station)), float(cas))
            for station, cas in zip(
                bank_evaluation_s_m,
                bank_evaluation_cas_mps,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    bank_demand = _bank_demand_evaluation(
        s_m=bank_evaluation_s_m,
        bank_ratio=_safe_ratio(
            np.abs(bank_evaluation_raw_rad),
            bank_evaluation_limit_rad,
        ),
        ground_speed_mps=bank_evaluation_ground_speed_mps,
        overbank_ratio_threshold=overbank_ratio_threshold,
    )

    raw_thrust_n = np.empty(len(flight_time_s), dtype=np.float64)
    clipped_thrust_n = np.empty(len(flight_time_s), dtype=np.float64)
    saturation = np.zeros(len(flight_time_s), dtype=bool)
    for index, (station, time_s, altitude, tas, gamma, bank, acceleration) in enumerate(
        zip(
            flight_plan_s_m,
            flight_time_s,
            flight_h_m,
            flight_tas_mps,
            gamma_rad,
            clipped_bank_rad,
            v_dot_mps2,
            strict=True,
        )
    ):
        mode = mode_for_s(cfg, float(station))
        delta_isa_K = float(weather.delta_isa_K(float(station), float(altitude), float(time_s)))
        drag_n = float(
            perf.drag_newtons(
                mode=mode,
                mass_kg=cfg.mass_kg,
                wing_area_m2=cfg.wing_area_m2,
                v_tas_mps=float(tas),
                h_m=float(altitude),
                gamma_rad=float(gamma),
                bank_rad=float(bank),
                delta_isa_K=delta_isa_K,
            )
        )
        lower_thrust_n, upper_thrust_n = perf.thrust_bounds_newtons(
            mode=mode,
            v_tas_mps=float(tas),
            h_m=float(altitude),
            delta_isa_K=delta_isa_K,
        )
        required_thrust_n = float(
            cfg.mass_kg * (float(acceleration) + float(aero.g0) * np.sin(float(gamma)))
            + drag_n
        )
        raw_thrust_n[index] = required_thrust_n
        clipped_thrust_n[index] = float(
            np.clip(required_thrust_n, float(lower_thrust_n), float(upper_thrust_n))
        )
        saturation[index] = not np.isclose(
            clipped_thrust_n[index],
            required_thrust_n,
            rtol=0.0,
            atol=1.0,
        )

    points_ne = geometry.reference_path.position_ne_many(plan_s_m)
    latlon = geometry.reference_path.latlon_from_ne_many(points_ne[:, 0], points_ne[:, 1])
    track_rad = geometry.reference_path.track_angle_rad_many(plan_s_m)
    plan = CoupledDescentPlanResult(
        s_m=plan_s_m,
        h_m=np.asarray(variant.altitude_m, dtype=np.float64),
        v_tas_mps=flight_tas_mps[::-1],
        v_cas_mps=flight_cas_mps[::-1],
        t_s=plan_t_s,
        east_m=points_ne[:, 0],
        north_m=points_ne[:, 1],
        lat_deg=latlon[:, 0],
        lon_deg=latlon[:, 1],
        cross_track_m=np.zeros(len(plan_s_m), dtype=np.float64),
        heading_error_rad=np.zeros(len(plan_s_m), dtype=np.float64),
        psi_rad=track_rad,
        phi_rad=clipped_bank_rad[::-1],
        roll_rate_rps=np.zeros(len(plan_s_m), dtype=np.float64),
        ground_speed_mps=flight_ground_speed_mps[::-1],
        alongtrack_speed_mps=flight_ground_speed_mps[::-1],
        crosstrack_speed_mps=np.zeros(len(plan_s_m), dtype=np.float64),
        track_error_rad=np.zeros(len(plan_s_m), dtype=np.float64),
        phi_max_rad=bank_limit_values[::-1],
        gamma_rad=gamma_rad[::-1],
        thrust_n=clipped_thrust_n[::-1],
        mode=tuple(mode_for_s(cfg, float(station)).name for station in plan_s_m),
        solver_success=True,
        solver_status=0,
        solver_message="hailmary command profile prepared for public SIMAP replay",
        objective_value=0.0,
        tod_m=reference_length_m,
        collocation_residual_max=0.0,
        replay_h_error_m=0.0,
        replay_v_error_mps=0.0,
        replay_t_error_s=0.0,
        replay_residual_max=0.0,
        constraint_slack=0.0,
        solve_profile=CoupledDescentSolveProfile(
            total_wall_time_s=0.0,
            postprocess_wall_time_s=0.0,
            objective_calls=0,
            objective_time_s=0.0,
            equality_calls=0,
            equality_time_s=0.0,
            inequality_calls=0,
            inequality_time_s=0.0,
            trajectory_evaluations=0,
            trajectory_eval_time_s=0.0,
            trajectory_cache_hits=0,
        ),
    )
    return (
        plan,
        bank_demand,
        float(np.count_nonzero(saturation) / max(1, len(saturation))),
    )


def _compile_replay_arrays(
    variant: TrajectoryVariant,
    evaluation: _ReplayEvaluation,
    diagnostics: VariantDiagnostics,
    command_envelope: CASEnvelope,
) -> TrajectoryVariant:
    result = evaluation.result
    reference_length_m = float(evaluation.reference_path.total_length_m)
    target_reference_s_m = np.asarray(
        variant.s_m * (reference_length_m / variant.path_length_m),
        dtype=np.float64,
    )
    canonical_replay_s_m = _canonical_replay_station_m(
        np.asarray(result.s_m, dtype=np.float64),
        reference_length_m=reference_length_m,
    )
    replay_s_ascending = np.asarray(canonical_replay_s_m[::-1], dtype=np.float64)
    unique_s_m, unique_indices = np.unique(replay_s_ascending, return_index=True)
    if len(unique_s_m) < 2:
        raise ValueError("SIMAP replay returned fewer than two distinct stations")

    def replay_profile(values: np.ndarray) -> np.ndarray:
        ascending_values = np.asarray(values[::-1], dtype=np.float64)[unique_indices]
        return np.asarray(
            np.interp(target_reference_s_m, unique_s_m, ascending_values),
            dtype=np.float64,
        )

    elapsed_time_s = replay_profile(np.asarray(result.t_s, dtype=np.float64))
    elapsed_time_s[0] = float(result.t_s[-1])
    elapsed_time_s[-1] = 0.0
    crossings = tuple(
        ResourceCrossing(
            resource_id=crossing.resource_id,
            s_m=crossing.s_m,
            elapsed_time_s=float(np.interp(crossing.s_m, variant.s_m, elapsed_time_s)),
        )
        for crossing in variant.resource_crossings
    )
    return TrajectoryVariant(
        template_id=variant.template_id,
        cluster_id=variant.cluster_id,
        s_m=variant.s_m,
        lat_deg=variant.lat_deg,
        lon_deg=variant.lon_deg,
        east_m=variant.east_m,
        north_m=variant.north_m,
        altitude_m=replay_profile(np.asarray(result.h_m, dtype=np.float64)),
        cas_mps=replay_profile(np.asarray(result.v_cas_mps, dtype=np.float64)),
        tas_mps=replay_profile(np.asarray(result.v_tas_mps, dtype=np.float64)),
        ground_speed_mps=replay_profile(
            np.asarray(result.alongtrack_speed_mps, dtype=np.float64)
        ),
        command_cas_mps=variant.command_cas_mps,
        reference_command_cas_mps=variant.reference_command_cas_mps,
        lower_cas_mps=command_envelope.lower_cas_mps,
        upper_cas_mps=command_envelope.upper_cas_mps,
        elapsed_time_s=elapsed_time_s,
        resource_crossings=crossings,
        diagnostics=diagnostics,
        action_provenance=variant.action_provenance,
        schema_version=variant.schema_version,
    )


def _canonical_replay_station_m(
    integrated_s_m: np.ndarray,
    *,
    reference_length_m: float,
) -> np.ndarray:
    """Map SIMAP's decoupled longitudinal station to physical endpoints.

    Threshold-calibrated replay can stop with a positive integrated ``s`` when
    its separately integrated map position reaches the physical runway.  The
    executable Hailmary axis must nevertheless run from the physical threshold
    at zero to the release endpoint at ``reference_length_m``.  A deterministic
    affine endpoint map preserves order and avoids a constant-time plateau in
    the final few hundred metres.
    """

    station = np.asarray(integrated_s_m, dtype=np.float64)
    if len(station) < 2 or np.any(np.diff(station) >= 0.0):
        raise ValueError("SIMAP replay station must strictly decrease")
    integrated_span_m = float(station[0] - station[-1])
    if integrated_span_m <= 0.0:
        raise ValueError("SIMAP replay station has no positive traversal span")
    canonical = (
        (station - float(station[-1]))
        * (float(reference_length_m) / integrated_span_m)
    )
    canonical[0] = float(reference_length_m)
    canonical[-1] = 0.0
    return canonical


@dataclass(frozen=True)
class SIMAPAdapter:
    """Compile/validate variants through public SIMAP replay APIs.

    SIMAP's nonlinear planner does not accept an arbitrary dense historical CAS
    command as a fixed solution.  The adapter therefore reconstructs bounded
    thrust/flight-path-angle commands from the Hailmary profile and runs the
    public coupled time-domain replay.  The distinction is recorded explicitly
    in diagnostics; no optimizer provenance is claimed.
    """

    aircraft_config: AircraftConfig | None = None
    performance_backend: PerformanceBackend | None = None
    weather: WeatherProvider = field(default_factory=ConstantWeather)
    payload_kg: float = 12_000.0
    path_simplification_tolerance_m: float = 250.0
    maximum_path_length_error_fraction: float = 0.02
    replay_dt_s: float = 1.0
    maximum_replay_timing_error_fraction: float = 0.10
    maximum_replay_timing_error_s: float = 30.0
    maximum_raw_bank_ratio: float = 1.05
    maximum_overbank_transient_distance_m: float = 250.0
    maximum_overbank_transient_duration_s: float = 3.0
    maximum_cross_track_m: float = 0.25 * 1_852.0
    maximum_threshold_error_m: float = 0.10 * 1_852.0
    physical_envelope_tolerance_mps: float = 0.5

    def __post_init__(self) -> None:
        positive = (
            self.payload_kg,
            self.maximum_path_length_error_fraction,
            self.replay_dt_s,
            self.maximum_replay_timing_error_fraction,
            self.maximum_replay_timing_error_s,
            self.maximum_raw_bank_ratio,
            self.maximum_overbank_transient_distance_m,
            self.maximum_overbank_transient_duration_s,
            self.maximum_cross_track_m,
            self.maximum_threshold_error_m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("SIMAP adapter tolerances and payload must be finite and positive")
        if self.path_simplification_tolerance_m < 0.0:
            raise ValueError("path_simplification_tolerance_m cannot be negative")
        if self.physical_envelope_tolerance_mps < 0.0:
            raise ValueError("physical_envelope_tolerance_mps cannot be negative")

    @property
    def resolved_aircraft_config(self) -> AircraftConfig:
        return self.aircraft_config or get_cached_a320_config(self.payload_kg)

    @property
    def resolved_performance_backend(self) -> PerformanceBackend | None:
        if self.performance_backend is not None:
            return self.performance_backend
        config = self.resolved_aircraft_config
        if self.aircraft_config is None:
            context = get_cached_a320_context(self.payload_kg)
            return EffectivePolarBackend(config, context.openap)
        if config.typecode.upper() == "A320":
            context = get_cached_a320_context(self.payload_kg, config.engine_name)
            return EffectivePolarBackend(config, context.openap)
        # A custom AircraftConfig does not identify the OpenAP bundle used to
        # calibrate it. Static envelope/geometry/bank checks remain available,
        # but inventing a performance backend would be scientifically unsound.
        return None

    @property
    def version1_default_aircraft_assumption(self) -> bool:
        weather_is_default = isinstance(self.weather, ConstantWeather) and (
            self.weather.wind_east_mps == 0.0
            and self.weather.wind_north_mps == 0.0
            and self.weather.delta_isa_offset_K == 0.0
        )
        return bool(
            self.aircraft_config is None
            and self.performance_backend is None
            and self.payload_kg == 12_000.0
            and weather_is_default
        )

    @property
    def performance_backend_source(self) -> str:
        if self.performance_backend is not None:
            return "explicit_performance_backend"
        if self.aircraft_config is None:
            return "version1_cached_a320_effective_polar"
        if self.resolved_aircraft_config.typecode.upper() == "A320":
            return "reconstructed_a320_effective_polar"
        return "unavailable_for_custom_aircraft"

    def envelope(self, s_m: object, altitude_m: object | None = None) -> CASEnvelope:
        return planned_a320_cas_envelope(
            s_m,
            altitude_m=altitude_m,
            aircraft_config=self.resolved_aircraft_config,
        )

    def reference_path(self, lat_deg: object, lon_deg: object) -> SimplifiedReferencePath:
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        return simplify_reference_path(
            lat,
            lon,
            threshold_lat_deg=float(lat[-1]),
            threshold_lon_deg=float(lon[-1]),
            lateral_tolerance_m=self.path_simplification_tolerance_m,
        )

    def _evaluate(
        self,
        variant: TrajectoryVariant,
    ) -> tuple[VariantDiagnostics, _ReplayEvaluation | None]:
        config = self.resolved_aircraft_config
        base_details: list[tuple[str, Any]] = [
            ("validator", "simap_public_coupled_replay_v1"),
            ("simap_optimizer_used", False),
            (
                "simap_replay_contract",
                "bounded_inverse_commands_then_public_time_domain_replay",
            ),
            ("aircraft_typecode", config.typecode),
            ("engine_name", config.engine_name),
            ("mass_kg", config.mass_kg),
            ("payload_kg", self.payload_kg),
            (
                "version1_default_aircraft_assumption",
                self.version1_default_aircraft_assumption,
            ),
            ("performance_backend_source", self.performance_backend_source),
            (
                "performance_backend_fingerprint",
                stable_id(
                    "simap-performance-backend",
                    {
                        "source": self.performance_backend_source,
                        "backend_type": (
                            None
                            if self.resolved_performance_backend is None
                            else (
                                f"{type(self.resolved_performance_backend).__module__}."
                                f"{type(self.resolved_performance_backend).__qualname__}"
                            )
                        ),
                        "aircraft_typecode": config.typecode,
                        "engine_name": config.engine_name,
                        "mass_kg": config.mass_kg,
                        "payload_kg": self.payload_kg,
                    },
                    length=32,
                ),
            ),
        ]
        try:
            envelope = self.envelope(variant.s_m, variant.altitude_m)
            tolerance = 1.0e-8
            if np.any(variant.command_cas_mps < envelope.lower_cas_mps - tolerance):
                raise ValueError("commanded CAS falls below the SIMAP/OpenAP lower envelope")
            if np.any(variant.command_cas_mps > envelope.upper_cas_mps + tolerance):
                raise ValueError("commanded CAS exceeds the SIMAP/OpenAP upper envelope")
            if np.any(variant.command_cas_mps > variant.reference_command_cas_mps + tolerance):
                raise ValueError("commanded CAS exceeds the medoid reference")
            geometry = simplify_reference_path(
                variant.lat_deg[::-1],
                variant.lon_deg[::-1],
                threshold_lat_deg=float(variant.lat_deg[0]),
                threshold_lon_deg=float(variant.lon_deg[0]),
                lateral_tolerance_m=self.path_simplification_tolerance_m,
            )
            expected_length = float(variant.s_m[-1])
            length_error = abs(geometry.total_length_m - expected_length)
            allowed_length_error = max(500.0, self.maximum_path_length_error_fraction * expected_length)
            if length_error > allowed_length_error:
                raise ValueError(
                    "simplified SIMAP ReferencePath length differs materially from the compiled variant"
                )
            base_details.extend(
                (
                    ("reference_path_control_count", len(geometry.control_lat_deg)),
                    ("reference_path_length_m", geometry.total_length_m),
                    ("reference_path_length_error_m", length_error),
                    (
                        "reference_path_curvature_source",
                        "wrap_safe_gradient_of_public_unwrapped_track",
                    ),
                    (
                        "reference_path_max_abs_curvature_inv_m",
                        float(
                            np.max(
                                np.abs(geometry.reference_path.curvature_inv_m),
                                initial=0.0,
                            )
                        ),
                    ),
                    (
                        "minimum_lower_envelope_margin_mps",
                        float(np.min(variant.command_cas_mps - envelope.lower_cas_mps)),
                    ),
                    (
                        "minimum_upper_envelope_margin_mps",
                        float(np.min(envelope.upper_cas_mps - variant.command_cas_mps)),
                    ),
                )
            )
            backend = self.resolved_performance_backend
            if backend is None:
                base_details.extend(
                    (
                        ("simap_replay_supported", False),
                        (
                            "simap_replay_limitation",
                            "custom AircraftConfig supplied without a matching PerformanceBackend",
                        ),
                    )
                )
                return (
                    _diagnostics_from_variant(
                        variant,
                        feasible=True,
                        message=(
                            "SIMAP envelope/ReferencePath validation passed; dynamics replay "
                            "unavailable for the custom aircraft configuration"
                        ),
                        details=tuple(base_details),
                    ),
                    None,
                )

            (
                plan,
                bank_demand,
                thrust_saturation_fraction,
            ) = _reference_plan(
                variant,
                geometry=geometry,
                cfg=config,
                perf=backend,
                weather=self.weather,
                overbank_ratio_threshold=self.maximum_raw_bank_ratio,
            )
            replay = _simulate_with_threshold_calibration(
                SimulationRequest(
                    cfg=config,
                    perf=backend,
                    plan=plan,
                    reference_path=geometry.reference_path,
                    weather=self.weather,
                    dt_s=self.replay_dt_s,
                    max_time_factor=3.0,
                    threshold_tolerance_m=1.0,
                )
            )
            result = replay.result
            base_details.extend(
                (
                    ("simap_replay_pass_count", replay.pass_count),
                    (
                        "simap_first_pass_threshold_error_m",
                        replay.first_pass_threshold_error_m,
                    ),
                    (
                        "simap_first_pass_alongtrack_overshoot_m",
                        replay.first_pass_alongtrack_overshoot_m,
                    ),
                    (
                        "simap_calibrated_threshold_tolerance_m",
                        replay.calibrated_threshold_tolerance_m,
                    ),
                    ("raw_max_bank_ratio", bank_demand.raw_max_ratio),
                    ("overbank_ratio_threshold", self.maximum_raw_bank_ratio),
                    (
                        "maximum_contiguous_overbank_distance_m",
                        bank_demand.maximum_contiguous_overbank_distance_m,
                    ),
                    (
                        "maximum_contiguous_overbank_duration_s",
                        bank_demand.maximum_contiguous_overbank_duration_s,
                    ),
                    (
                        "maximum_overbank_transient_distance_m",
                        self.maximum_overbank_transient_distance_m,
                    ),
                    (
                        "maximum_overbank_transient_duration_s",
                        self.maximum_overbank_transient_duration_s,
                    ),
                    ("bank_demand_sample_count", bank_demand.sample_count),
                )
            )
            if not result.success:
                raise ValueError(f"public SIMAP replay failed: {result.message}")
            if (
                bank_demand.maximum_contiguous_overbank_distance_m
                > self.maximum_overbank_transient_distance_m + 1.0e-9
                or bank_demand.maximum_contiguous_overbank_duration_s
                > self.maximum_overbank_transient_duration_s + 1.0e-9
            ):
                raise ValueError(
                    "reference-path curvature requires sustained bank beyond the "
                    "SIMAP/OpenAP limit "
                    f"(raw ratio={bank_demand.raw_max_ratio:.3f}, "
                    "contiguous distance="
                    f"{bank_demand.maximum_contiguous_overbank_distance_m:.1f} m, "
                    "duration="
                    f"{bank_demand.maximum_contiguous_overbank_duration_s:.2f} s)"
                )
            if result.max_abs_cross_track_m > self.maximum_cross_track_m + 1.0e-9:
                raise ValueError(
                    "public SIMAP replay exceeds the cross-track tolerance "
                    f"({result.max_abs_cross_track_m:.3f} m)"
                )
            if result.final_threshold_error_m > self.maximum_threshold_error_m + 1.0e-9:
                raise ValueError(
                    "public SIMAP replay misses the threshold tolerance "
                    f"({result.final_threshold_error_m:.3f} m)"
                )
            if result.min_alongtrack_speed_mps <= 0.0:
                raise ValueError("public SIMAP replay lost positive along-track motion")

            canonical_replay_s_m = _canonical_replay_station_m(
                np.asarray(result.s_m, dtype=np.float64),
                reference_length_m=geometry.total_length_m,
            )
            replay_original_s_m = np.asarray(
                canonical_replay_s_m
                * (variant.path_length_m / geometry.total_length_m),
                dtype=np.float64,
            )
            replay_envelope = self.envelope(replay_original_s_m, result.h_m)
            physical_lower_margin = np.asarray(
                result.v_cas_mps - replay_envelope.lower_cas_mps,
                dtype=np.float64,
            )
            physical_upper_margin = np.asarray(
                replay_envelope.upper_cas_mps - result.v_cas_mps,
                dtype=np.float64,
            )
            tolerance = self.physical_envelope_tolerance_mps
            if np.min(physical_lower_margin) < -tolerance:
                raise ValueError("public SIMAP replay falls below the physical CAS envelope")
            if np.min(physical_upper_margin) < -tolerance:
                raise ValueError("public SIMAP replay exceeds the physical CAS envelope")

            replay_duration_s = float(result.t_s[-1])
            action_lever = str(getattr(variant.action_provenance, "lever", "baseline"))
            observed_duration_s = getattr(variant.diagnostics, "observed_duration_s", None)
            if action_lever == "baseline" and observed_duration_s is not None:
                timing_reference_s = float(observed_duration_s)
                timing_reference_name = "observed_medoid_duration"
            else:
                timing_reference_s = float(variant.duration_s)
                timing_reference_name = "kinematic_variant_duration"
            signed_timing_error_s = replay_duration_s - timing_reference_s
            absolute_timing_error_s = abs(signed_timing_error_s)
            if action_lever == "baseline" and (
                absolute_timing_error_s > self.maximum_replay_timing_error_s + 1.0e-9
                or absolute_timing_error_s
                > self.maximum_replay_timing_error_fraction * timing_reference_s + 1.0e-9
            ):
                raise ValueError(
                    "public SIMAP replay timing violates the absolute or relative quality gate "
                    f"(error={absolute_timing_error_s:.3f} s, reference={timing_reference_name})"
                )
            if len(result.t_s) > 1:
                physical_cas_acceleration = np.gradient(result.v_cas_mps, result.t_s)
                max_positive_cas_acceleration = float(
                    np.max(physical_cas_acceleration, initial=0.0)
                )
            else:
                max_positive_cas_acceleration = 0.0
            base_details.extend(
                (
                    ("simap_replay_supported", True),
                    ("simap_replay_success", True),
                    ("simap_replay_message", result.message),
                    ("simap_replay_sample_count", len(result)),
                    ("simap_replay_duration_s", replay_duration_s),
                    ("simap_integrated_final_station_m", float(result.s_m[-1])),
                    (
                        "simap_executable_station_mapping",
                        "affine_integrated_endpoints_to_physical_path_endpoints",
                    ),
                    ("simap_kinematic_duration_s", float(variant.duration_s)),
                    ("simap_timing_reference", timing_reference_name),
                    ("replay_max_bank_command_ratio", result.max_bank_command_ratio),
                    ("replay_max_abs_cross_track_m", result.max_abs_cross_track_m),
                    ("replay_final_threshold_error_m", result.final_threshold_error_m),
                    ("replay_min_alongtrack_speed_mps", result.min_alongtrack_speed_mps),
                    ("inverse_command_thrust_saturation_fraction", thrust_saturation_fraction),
                    ("minimum_physical_lower_envelope_margin_mps", float(np.min(physical_lower_margin))),
                    ("minimum_physical_upper_envelope_margin_mps", float(np.min(physical_upper_margin))),
                )
            )
            diagnostics = _diagnostics_from_variant(
                variant,
                feasible=True,
                message="public SIMAP coupled replay validation passed",
                details=tuple(base_details),
                compiled_duration_s=replay_duration_s,
                signed_timing_error_s=signed_timing_error_s,
                absolute_timing_error_s=absolute_timing_error_s,
                max_physical_cas_acceleration_mps2=max_positive_cas_acceleration,
            )
            return (
                diagnostics,
                _ReplayEvaluation(
                    reference_path=geometry,
                    result=result,
                    raw_max_bank_ratio=bank_demand.raw_max_ratio,
                    maximum_contiguous_overbank_distance_m=(
                        bank_demand.maximum_contiguous_overbank_distance_m
                    ),
                    maximum_contiguous_overbank_duration_s=(
                        bank_demand.maximum_contiguous_overbank_duration_s
                    ),
                    bank_demand_sample_count=bank_demand.sample_count,
                    thrust_saturation_fraction=thrust_saturation_fraction,
                    kinematic_duration_s=float(variant.duration_s),
                ),
            )
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
            base_details.append(("failure_type", type(exc).__name__))
            return (
                _diagnostics_from_variant(
                    variant,
                    feasible=False,
                    message=f"SIMAP validation failed: {exc}",
                    details=tuple(base_details),
                ),
                None,
            )

    def validate(self, variant: TrajectoryVariant) -> VariantDiagnostics:
        """Return immutable diagnostics without changing ``variant``."""

        diagnostics, _evaluation = self._evaluate(variant)
        return diagnostics

    def compile(self, variant: TrajectoryVariant) -> TrajectoryVariant:
        """Return a variant whose physical profiles/timing come from SIMAP replay.

        Command, envelope, and desired geometry arrays retain their Hailmary
        identities.  Physical altitude/CAS/TAS/ground-speed and elapsed-time
        arrays are sampled from the public coupled replay when it is available.
        A custom aircraft without a matching performance backend receives static
        validation only and is returned with explicit limitation diagnostics.
        """

        diagnostics, evaluation = self._evaluate(variant)
        if not diagnostics.feasible:
            return replace(variant, diagnostics=diagnostics, variant_id="")
        command_envelope = self.envelope(variant.s_m, variant.altitude_m)
        if evaluation is None:
            return replace(
                variant,
                lower_cas_mps=command_envelope.lower_cas_mps,
                upper_cas_mps=command_envelope.upper_cas_mps,
                diagnostics=diagnostics,
                variant_id="",
            )
        return _compile_replay_arrays(
            variant,
            evaluation,
            diagnostics,
            command_envelope,
        )


SimapAdapter = SIMAPAdapter
build_reference_path = simplify_reference_path
validate_reference_path_geometry = simplify_reference_path


__all__ = [
    "A320Context",
    "CASEnvelope",
    "SIMAPAdapter",
    "SimapAdapter",
    "SimplifiedReferencePath",
    "build_reference_path",
    "clear_a320_cache",
    "get_cached_a320_config",
    "get_cached_a320_context",
    "get_cached_a320_envelope",
    "planned_a320_cas_envelope",
    "simplify_reference_path",
    "validate_reference_path_geometry",
]
