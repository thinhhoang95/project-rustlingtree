"""Compile a historical medoid track into an immutable kinematic template."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol

import numpy as np
from pyproj import CRS, Transformer  # pyright: ignore[reportMissingImports]

from hailmary._arrays import readonly_float64, strictly_increasing, validate_same_length
from hailmary.config import M_PER_NM, TemplateConfig
from hailmary.errors import ArtifactValidationError
from hailmary.ids import stable_id
from hailmary.templates.models import (
    ActionStation,
    ClusterTemplate,
    TrajectoryVariant,
    VariantDiagnostics,
)
from hailmary.templates.speed import (
    cas_envelope,
    cas_to_tas,
    clamp_reference_to_envelope,
    finite_difference_ground_speed,
    integrate_elapsed_time,
    monotone_command_profile,
    robust_smooth,
    tas_to_cas,
)


class VariantValidator(Protocol):
    def validate(self, variant: TrajectoryVariant) -> VariantDiagnostics: ...


@dataclass(frozen=True)
class MedoidTrack:
    flight_id: str
    time_s: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    altitude_m: np.ndarray
    ground_speed_mps: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("time_s", "lat_deg", "lon_deg", "altitude_m"):
            object.__setattr__(self, name, readonly_float64(getattr(self, name), name=name))
        if not self.flight_id:
            raise ArtifactValidationError("medoid flight ID cannot be empty")
        n = len(self.time_s)
        if n < 2:
            raise ArtifactValidationError("a medoid track needs at least two samples")
        validate_same_length(n, lat_deg=self.lat_deg, lon_deg=self.lon_deg, altitude_m=self.altitude_m)
        if self.ground_speed_mps is not None:
            ground_speed = readonly_float64(
                self.ground_speed_mps,
                name="ground_speed_mps",
            )
            validate_same_length(n, ground_speed_mps=ground_speed)
            if np.any(ground_speed <= 0.0):
                raise ArtifactValidationError("medoid ground speed must be positive")
            object.__setattr__(self, "ground_speed_mps", ground_speed)
        if not strictly_increasing(self.time_s):
            raise ArtifactValidationError("medoid timestamps must strictly increase")
        if np.any(np.abs(self.lat_deg) > 90.0) or np.any(np.abs(self.lon_deg) > 180.0):
            raise ArtifactValidationError("medoid geographic coordinates are invalid")


def _project_track(track: MedoidTrack) -> tuple[np.ndarray, np.ndarray]:
    origin_lat = float(track.lat_deg[-1])
    origin_lon = float(track.lon_deg[-1])
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    east, north = transformer.transform(track.lon_deg, track.lat_deg)
    return np.asarray(east, dtype=float), np.asarray(north, dtype=float)


def _deduplicate_track(
    track: MedoidTrack,
    east_m: np.ndarray,
    north_m: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    distance = np.hypot(np.diff(east_m), np.diff(north_m))
    keep = np.concatenate(([True], distance > 1e-3))
    if np.count_nonzero(keep) < 2:
        raise ArtifactValidationError("medoid path has fewer than two distinct coordinates")
    return (
        track.time_s[keep],
        track.lat_deg[keep],
        track.lon_deg[keep],
        track.altitude_m[keep],
        east_m[keep],
        north_m[keep],
        None if track.ground_speed_mps is None else track.ground_speed_mps[keep],
    )


def select_action_stations(
    s_m: np.ndarray,
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    count: int,
    commitment_gate_m: float,
    kind: str,
) -> tuple[ActionStation, ...]:
    stations = np.asarray(s_m, dtype=float)
    eligible = np.flatnonzero(stations >= commitment_gate_m - 1e-9)
    if len(eligible) < count + 1:
        raise ArtifactValidationError(f"path/grid cannot provide {count} distinct {kind} locations")
    entry_s = float(stations[-1])
    gate_s = float(max(commitment_gate_m, stations[eligible[0]]))
    targets = [entry_s - (index / (count + 1.0)) * (entry_s - gate_s) for index in range(1, count + 1)]
    chosen: list[int] = []
    available = set(int(index) for index in eligible)
    for target in targets:
        if not available:
            raise ArtifactValidationError(f"cannot deduplicate {kind} locations")
        index = min(available, key=lambda candidate: (abs(float(stations[candidate]) - target), -candidate))
        chosen.append(index)
        available.remove(index)
    chosen.sort(reverse=True)
    if len(set(chosen)) != count:
        raise ArtifactValidationError(f"{kind} locations are not distinct")
    return tuple(
        ActionStation(
            entry_order=entry_order,
            kind=kind,  # type: ignore[arg-type]
            grid_index=grid_index,
            s_m=float(stations[grid_index]),
            east_m=float(east_m[grid_index]),
            north_m=float(north_m[grid_index]),
        )
        for entry_order, grid_index in enumerate(chosen)
    )


@dataclass(frozen=True)
class TemplateCompiler:
    config: TemplateConfig = TemplateConfig()
    station_count: int = 512
    aircraft_config: object | None = None
    validator: VariantValidator | None = None

    def compile(
        self,
        track: MedoidTrack,
        *,
        cluster_id: str,
        member_count: int,
        dataset_id: str,
        airport_id: str,
        runway_id: str,
        dispersion_m: float = 0.0,
        threshold_resource_id: str | None = None,
    ) -> ClusterTemplate:
        if self.station_count < self.config.speed_action_count + self.config.path_stretch_count + 2:
            raise ArtifactValidationError("compiler station grid is too small for action locations")
        east, north = _project_track(track)
        time, lat, lon, altitude, east, north, supplied_ground = _deduplicate_track(
            track,
            east,
            north,
        )
        segment = np.hypot(np.diff(east), np.diff(north))
        progress = np.concatenate(([0.0], np.cumsum(segment)))
        total_length = float(progress[-1])
        gate_m = self.config.commitment_gate_nm * M_PER_NM
        if total_length <= gate_m:
            raise ArtifactValidationError("medoid path does not extend upstream of the commitment gate")

        remaining = total_length - progress
        if np.any(np.diff(remaining) >= 0.0):
            raise ArtifactValidationError("medoid geometry is not strictly ordered upstream to threshold")
        original_ground = (
            finite_difference_ground_speed(time, east, north)
            if supplied_ground is None
            else supplied_ground
        )

        dense_s = np.linspace(0.0, total_length, self.station_count, dtype=float)
        source_s = remaining[::-1]
        dense_lat = np.interp(dense_s, source_s, lat[::-1])
        dense_lon = np.interp(dense_s, source_s, lon[::-1])
        dense_altitude = np.maximum(0.0, np.interp(dense_s, source_s, altitude[::-1]))
        dense_east = np.interp(dense_s, source_s, east[::-1])
        dense_north = np.interp(dense_s, source_s, north[::-1])
        historical_ground = robust_smooth(np.interp(dense_s, source_s, original_ground[::-1]))
        historical_cas = tas_to_cas(historical_ground, dense_altitude)
        isotonic_cas = monotone_command_profile(dense_s, historical_cas)

        bounds_function: Callable[[float], tuple[float, float]] | None = None
        if self.aircraft_config is not None:
            from simap.config import planned_cas_bounds_mps

            aircraft_config = self.aircraft_config

            def aircraft_bounds(station: float) -> tuple[float, float]:
                return planned_cas_bounds_mps(aircraft_config, station)  # type: ignore[arg-type]

            bounds_function = aircraft_bounds
        lower, upper = cas_envelope(
            dense_s,
            dense_altitude,
            isotonic_cas,
            bounds_at_station=bounds_function,
        )
        command, max_excursion, clamped_fraction = clamp_reference_to_envelope(
            isotonic_cas,
            lower,
            upper,
            max_excursion_kts=self.config.max_historical_clamp_kts,
            max_clamped_fraction=self.config.max_clamped_fraction,
        )
        if np.any(np.diff(command) < -1e-8):
            raise ArtifactValidationError(
                "envelope clamping made the command increase in the downstream flight direction"
            )
        compiled_tas = cas_to_tas(command, dense_altitude)
        compiled_ground = compiled_tas.copy()
        elapsed = integrate_elapsed_time(dense_s, compiled_ground)
        observed_duration = float(time[-1] - time[0])
        compiled_duration = float(elapsed[0])
        signed_error = compiled_duration - observed_duration
        absolute_error = abs(signed_error)
        if self.validator is None and (
            absolute_error > self.config.max_timing_error_s + 1e-9
            or absolute_error > self.config.max_timing_error_fraction * observed_duration + 1e-9
        ):
            raise ArtifactValidationError(
                "compiled 50-NM traversal time violates both the relative and absolute quality gates"
            )

        logical_template_id = stable_id(
            "template-key",
            {
                "dataset_id": dataset_id,
                "airport_id": airport_id,
                "runway_id": runway_id,
                "cluster_id": cluster_id,
                "medoid_flight_id": track.flight_id,
            },
            length=32,
        )
        resource_id = threshold_resource_id or f"{airport_id}:{runway_id}:threshold"
        diagnostics = VariantDiagnostics(
            feasible=True,
            message="compiled kinematic profile; SIMAP validation pending" if self.validator is None else "pending validator",
            speed_source="finite_difference_ground_speed",
            wind_model="zero_wind",
            cas_derivation="ground_speed_as_tas_then_openap",
            observed_duration_s=observed_duration,
            compiled_duration_s=compiled_duration,
            signed_timing_error_s=signed_error,
            absolute_timing_error_s=absolute_error,
            max_command_envelope_excursion_mps=max_excursion,
            clamped_fraction=clamped_fraction,
            details=(("aircraft_typecode", self.config.aircraft_typecode), ("payload_kg", self.config.payload_kg)),
        )
        baseline = TrajectoryVariant(
            template_id=logical_template_id,
            cluster_id=cluster_id,
            s_m=dense_s,
            lat_deg=dense_lat,
            lon_deg=dense_lon,
            east_m=dense_east,
            north_m=dense_north,
            altitude_m=dense_altitude,
            cas_mps=command,
            tas_mps=compiled_tas,
            ground_speed_mps=compiled_ground,
            command_cas_mps=command,
            reference_command_cas_mps=command,
            lower_cas_mps=lower,
            upper_cas_mps=upper,
            elapsed_time_s=elapsed,
            resource_crossings=(),
            diagnostics=diagnostics,
        )
        # Rebuild once so the canonical threshold crossing is embedded.
        baseline = TrajectoryVariant.from_kinematic_profile(
            template_id=logical_template_id,
            cluster_id=cluster_id,
            s_m=dense_s,
            lat_deg=dense_lat,
            lon_deg=dense_lon,
            east_m=dense_east,
            north_m=dense_north,
            altitude_m=dense_altitude,
            cas_mps=command,
            tas_mps=compiled_tas,
            ground_speed_mps=compiled_ground,
            command_cas_mps=command,
            reference_command_cas_mps=command,
            lower_cas_mps=lower,
            upper_cas_mps=upper,
            threshold_resource_id=resource_id,
            diagnostics=diagnostics,
        )
        if self.validator is not None:
            compile_variant = getattr(self.validator, "compile", None)
            if callable(compile_variant):
                compiled = compile_variant(baseline)
                if not isinstance(compiled, TrajectoryVariant):
                    raise TypeError("variant compiler must return TrajectoryVariant")
                validated = compiled.diagnostics
            else:
                validated = self.validator.validate(baseline)
                compiled = TrajectoryVariant(
                    **{**baseline.__dict__, "diagnostics": validated, "variant_id": ""}
                )
            if not validated.feasible:
                raise ArtifactValidationError(f"SIMAP validation failed: {validated.message}")
            baseline = compiled
            validated_duration = validated.compiled_duration_s
            if validated_duration is not None:
                validated_error = abs(float(validated_duration) - observed_duration)
                if (
                    validated_error > self.config.max_timing_error_s + 1e-9
                    or validated_error
                    > self.config.max_timing_error_fraction * observed_duration + 1e-9
                ):
                    raise ArtifactValidationError(
                        "SIMAP-compiled traversal time violates the relative or absolute quality gate"
                    )

        speed_stations = select_action_stations(
            dense_s,
            dense_east,
            dense_north,
            count=self.config.speed_action_count,
            commitment_gate_m=gate_m,
            kind="speed",
        )
        stretch_stations = select_action_stations(
            dense_s,
            dense_east,
            dense_north,
            count=self.config.path_stretch_count,
            commitment_gate_m=gate_m,
            kind="path_stretch",
        )
        return ClusterTemplate(
            cluster_id=cluster_id,
            medoid_flight_id=track.flight_id,
            member_count=member_count,
            baseline_variant=baseline,
            speed_action_stations=speed_stations,
            path_stretch_stations=stretch_stations,
            dataset_id=dataset_id,
            airport_id=airport_id,
            runway_id=runway_id,
            dispersion_m=dispersion_m,
            provenance=(
                ("speed_source", "finite_difference_ground_speed"),
                ("wind_model", "zero_wind"),
                ("cas_derivation", "ground_speed_as_tas_then_openap"),
                ("aircraft_typecode", self.config.aircraft_typecode),
                ("payload_kg", self.config.payload_kg),
            ),
        )


def compile_kinematic_template(track: MedoidTrack, **kwargs: object) -> ClusterTemplate:
    """Convenience entry point using the version-1 compiler defaults."""

    return TemplateCompiler().compile(track, **kwargs)  # type: ignore[arg-type]
