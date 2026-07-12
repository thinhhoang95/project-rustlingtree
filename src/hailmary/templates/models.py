"""Immutable artifact models consumed by the hot simulator."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from hailmary._arrays import (
    readonly_float64,
    strictly_decreasing,
    strictly_increasing,
    validate_same_length,
)
from hailmary.errors import ArtifactValidationError
from hailmary.ids import canonical_data, content_hash, stable_id

StationKind = Literal["speed", "path_stretch"]


def _freeze_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ArtifactValidationError("artifact metadata cannot contain NaN or infinity")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_metadata_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        frozen = tuple(_freeze_metadata_value(item) for item in value)
        return tuple(sorted(frozen, key=repr))
    if isinstance(value, np.ndarray):
        return _freeze_metadata_value(value.tolist())
    raise ArtifactValidationError(f"unsupported mutable artifact metadata value: {type(value).__qualname__}")


def _freeze_metadata_pairs(values: Any, *, name: str) -> tuple[tuple[str, Any], ...]:
    pairs = tuple((str(key), _freeze_metadata_value(value)) for key, value in values)
    if len({key for key, _ in pairs}) != len(pairs):
        raise ArtifactValidationError(f"{name} contains duplicate keys")
    return pairs


@dataclass(frozen=True, order=True)
class ResourceCrossing:
    resource_id: str
    s_m: float
    elapsed_time_s: float

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ArtifactValidationError("resource_id cannot be empty")
        if not np.isfinite(self.s_m) or self.s_m < 0.0:
            raise ArtifactValidationError("resource crossing station must be finite and non-negative")
        if not np.isfinite(self.elapsed_time_s) or self.elapsed_time_s < 0.0:
            raise ArtifactValidationError("resource crossing time must be finite and non-negative")


@dataclass(frozen=True)
class VariantDiagnostics:
    feasible: bool = True
    message: str = "ok"
    speed_source: str = "compiled_kinematic"
    wind_model: str = "zero_wind"
    cas_derivation: str = "ground_speed_as_tas_then_openap"
    observed_duration_s: float | None = None
    compiled_duration_s: float | None = None
    signed_timing_error_s: float | None = None
    absolute_timing_error_s: float | None = None
    max_command_envelope_excursion_mps: float = 0.0
    clamped_fraction: float = 0.0
    max_physical_cas_acceleration_mps2: float | None = None
    details: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _freeze_metadata_pairs(self.details, name="diagnostics details"))


@dataclass(frozen=True)
class ActionProvenance:
    lever: str = "baseline"
    band: str = "baseline"
    parent_variant_id: str | None = None
    anchor_station_index: int | None = None
    added_distance_m: float = 0.0
    speed_reduction_mps: float = 0.0
    realization_metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.added_distance_m < 0.0 or self.speed_reduction_mps < 0.0:
            raise ArtifactValidationError("action-provenance magnitudes cannot be negative")
        object.__setattr__(
            self,
            "realization_metadata",
            _freeze_metadata_pairs(self.realization_metadata, name="action realization metadata"),
        )


@dataclass(frozen=True)
class TrajectoryVariant:
    """A precompiled path/profile indexed by remaining distance.

    Arrays are ordered threshold-to-upstream: ``s_m`` strictly increases while
    ``elapsed_time_s`` strictly decreases. At release the aircraft starts at
    the last array element and flies toward index zero.
    """

    template_id: str
    cluster_id: str
    s_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    altitude_m: np.ndarray
    cas_mps: np.ndarray
    tas_mps: np.ndarray
    ground_speed_mps: np.ndarray
    command_cas_mps: np.ndarray
    reference_command_cas_mps: np.ndarray
    lower_cas_mps: np.ndarray
    upper_cas_mps: np.ndarray
    elapsed_time_s: np.ndarray
    resource_crossings: tuple[ResourceCrossing, ...] = ()
    diagnostics: VariantDiagnostics = field(default_factory=VariantDiagnostics)
    action_provenance: ActionProvenance = field(default_factory=ActionProvenance)
    schema_version: str = "hailmary.trajectory_variant.v1"
    variant_id: str = ""

    def __post_init__(self) -> None:
        array_names = (
            "s_m",
            "lat_deg",
            "lon_deg",
            "east_m",
            "north_m",
            "altitude_m",
            "cas_mps",
            "tas_mps",
            "ground_speed_mps",
            "command_cas_mps",
            "reference_command_cas_mps",
            "lower_cas_mps",
            "upper_cas_mps",
            "elapsed_time_s",
        )
        for name in array_names:
            object.__setattr__(self, name, readonly_float64(getattr(self, name), name=name))

        n = len(self.s_m)
        if n < 2:
            raise ArtifactValidationError("trajectory variants require at least two stations")
        validate_same_length(n, **{name: getattr(self, name) for name in array_names if name != "s_m"})
        if abs(float(self.s_m[0])) > 1e-6 or not strictly_increasing(self.s_m):
            raise ArtifactValidationError("s_m must start at zero and strictly increase upstream")
        if abs(float(self.elapsed_time_s[-1])) > 1e-6 or not strictly_decreasing(self.elapsed_time_s):
            raise ArtifactValidationError("elapsed_time_s must strictly decrease to zero at the upstream endpoint")
        if np.any(self.ground_speed_mps <= 0.0) or np.any(self.tas_mps <= 0.0) or np.any(self.cas_mps <= 0.0):
            raise ArtifactValidationError("CAS, TAS, and ground speed must be positive")
        if np.any(self.lower_cas_mps <= 0.0) or np.any(self.upper_cas_mps < self.lower_cas_mps):
            raise ArtifactValidationError("invalid CAS envelope")
        tolerance = 1e-8
        if np.any(self.command_cas_mps < self.lower_cas_mps - tolerance):
            raise ArtifactValidationError("commanded CAS is below the lower envelope")
        if np.any(self.command_cas_mps > self.upper_cas_mps + tolerance):
            raise ArtifactValidationError("commanded CAS is above the upper envelope")
        if np.any(self.command_cas_mps > self.reference_command_cas_mps + tolerance):
            raise ArtifactValidationError("an intervention command cannot exceed the medoid reference")
        if np.any(np.diff(self.command_cas_mps) < -tolerance):
            raise ArtifactValidationError("commanded CAS must not increase in the downstream flight direction")
        if np.any(np.diff(self.reference_command_cas_mps) < -tolerance):
            raise ArtifactValidationError("reference CAS must not increase in the downstream flight direction")
        if not self.template_id or not self.cluster_id:
            raise ArtifactValidationError("template_id and cluster_id cannot be empty")

        crossing_ids: set[str] = set()
        for crossing in self.resource_crossings:
            if crossing.resource_id in crossing_ids:
                raise ArtifactValidationError(f"duplicate resource crossing: {crossing.resource_id}")
            crossing_ids.add(crossing.resource_id)
            if crossing.s_m > float(self.s_m[-1]) + 1e-6:
                raise ArtifactValidationError(f"resource {crossing.resource_id} is outside the trajectory")
            expected = float(np.interp(crossing.s_m, self.s_m, self.elapsed_time_s))
            if abs(expected - crossing.elapsed_time_s) > 1e-5:
                raise ArtifactValidationError(f"resource {crossing.resource_id} time is inconsistent with the trajectory")

        computed = stable_id("variant", self._hash_payload(), length=32)
        if self.variant_id and self.variant_id != computed:
            raise ArtifactValidationError("variant_id does not match trajectory content")
        object.__setattr__(self, "variant_id", computed)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "cluster_id": self.cluster_id,
            "s_m": self.s_m,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "east_m": self.east_m,
            "north_m": self.north_m,
            "altitude_m": self.altitude_m,
            "cas_mps": self.cas_mps,
            "tas_mps": self.tas_mps,
            "ground_speed_mps": self.ground_speed_mps,
            "command_cas_mps": self.command_cas_mps,
            "reference_command_cas_mps": self.reference_command_cas_mps,
            "lower_cas_mps": self.lower_cas_mps,
            "upper_cas_mps": self.upper_cas_mps,
            "elapsed_time_s": self.elapsed_time_s,
            "resource_crossings": self.resource_crossings,
            "diagnostics": self.diagnostics,
            "action_provenance": self.action_provenance,
        }

    @property
    def content_hash(self) -> str:
        return self.variant_id.removeprefix("variant_")

    @property
    def duration_s(self) -> float:
        return float(self.elapsed_time_s[0])

    @property
    def path_length_m(self) -> float:
        return float(self.s_m[-1])

    @property
    def threshold_resource_id(self) -> str | None:
        for crossing in self.resource_crossings:
            if abs(crossing.s_m) <= 1e-6:
                return crossing.resource_id
        return None

    def resource(self, resource_id: str) -> ResourceCrossing:
        for crossing in self.resource_crossings:
            if crossing.resource_id == resource_id:
                return crossing
        raise KeyError(resource_id)

    def to_dict(self) -> dict[str, Any]:
        diagnostics = {
            item.name: canonical_data(getattr(self.diagnostics, item.name))
            for item in fields(self.diagnostics)
        }
        provenance = {
            item.name: canonical_data(getattr(self.action_provenance, item.name))
            for item in fields(self.action_provenance)
        }
        return {
            "schema_version": self.schema_version,
            "variant_id": self.variant_id,
            "template_id": self.template_id,
            "cluster_id": self.cluster_id,
            **{
                name: getattr(self, name).tolist()
                for name in (
                    "s_m",
                    "lat_deg",
                    "lon_deg",
                    "east_m",
                    "north_m",
                    "altitude_m",
                    "cas_mps",
                    "tas_mps",
                    "ground_speed_mps",
                    "command_cas_mps",
                    "reference_command_cas_mps",
                    "lower_cas_mps",
                    "upper_cas_mps",
                    "elapsed_time_s",
                )
            },
            "resource_crossings": [
                {
                    "resource_id": crossing.resource_id,
                    "s_m": crossing.s_m,
                    "elapsed_time_s": crossing.elapsed_time_s,
                }
                for crossing in self.resource_crossings
            ],
            "diagnostics": diagnostics,
            "action_provenance": provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryVariant":
        diagnostics_payload = dict(payload.get("diagnostics", {}))
        diagnostics_payload["details"] = tuple(
            (str(item[0]), item[1]) for item in diagnostics_payload.get("details", ())
        )
        provenance_payload = dict(payload.get("action_provenance", {}))
        provenance_payload["realization_metadata"] = tuple(
            (str(item[0]), item[1])
            for item in provenance_payload.get("realization_metadata", ())
        )
        return cls(
            template_id=str(payload["template_id"]),
            cluster_id=str(payload["cluster_id"]),
            s_m=np.asarray(payload["s_m"], dtype=np.float64),
            lat_deg=np.asarray(payload["lat_deg"], dtype=np.float64),
            lon_deg=np.asarray(payload["lon_deg"], dtype=np.float64),
            east_m=np.asarray(payload["east_m"], dtype=np.float64),
            north_m=np.asarray(payload["north_m"], dtype=np.float64),
            altitude_m=np.asarray(payload["altitude_m"], dtype=np.float64),
            cas_mps=np.asarray(payload["cas_mps"], dtype=np.float64),
            tas_mps=np.asarray(payload["tas_mps"], dtype=np.float64),
            ground_speed_mps=np.asarray(payload["ground_speed_mps"], dtype=np.float64),
            command_cas_mps=np.asarray(payload["command_cas_mps"], dtype=np.float64),
            reference_command_cas_mps=np.asarray(
                payload["reference_command_cas_mps"], dtype=np.float64
            ),
            lower_cas_mps=np.asarray(payload["lower_cas_mps"], dtype=np.float64),
            upper_cas_mps=np.asarray(payload["upper_cas_mps"], dtype=np.float64),
            elapsed_time_s=np.asarray(payload["elapsed_time_s"], dtype=np.float64),
            resource_crossings=tuple(
                ResourceCrossing(
                    resource_id=str(item["resource_id"]),
                    s_m=float(item["s_m"]),
                    elapsed_time_s=float(item["elapsed_time_s"]),
                )
                for item in payload.get("resource_crossings", ())
            ),
            diagnostics=VariantDiagnostics(**diagnostics_payload),
            action_provenance=ActionProvenance(**provenance_payload),
            schema_version=str(payload.get("schema_version", "hailmary.trajectory_variant.v1")),
            variant_id=str(payload.get("variant_id", "")),
        )

    @classmethod
    def from_kinematic_profile(
        cls,
        *,
        template_id: str,
        cluster_id: str,
        s_m: np.ndarray,
        east_m: np.ndarray,
        north_m: np.ndarray,
        altitude_m: np.ndarray,
        cas_mps: np.ndarray,
        lower_cas_mps: np.ndarray,
        upper_cas_mps: np.ndarray,
        lat_deg: np.ndarray | None = None,
        lon_deg: np.ndarray | None = None,
        tas_mps: np.ndarray | None = None,
        ground_speed_mps: np.ndarray | None = None,
        command_cas_mps: np.ndarray | None = None,
        reference_command_cas_mps: np.ndarray | None = None,
        threshold_resource_id: str = "runway_threshold",
        resource_stations_m: tuple[tuple[str, float], ...] = (),
        diagnostics: VariantDiagnostics | None = None,
        action_provenance: ActionProvenance | None = None,
    ) -> "TrajectoryVariant":
        stations = np.asarray(s_m, dtype=float)
        speeds = np.asarray(ground_speed_mps if ground_speed_mps is not None else tas_mps if tas_mps is not None else cas_mps, dtype=float)
        if len(stations) != len(speeds) or np.any(speeds <= 0.0):
            raise ArtifactValidationError("positive ground speed is required at every station")
        elapsed = np.zeros(len(stations), dtype=float)
        for index in range(len(stations) - 2, -1, -1):
            ds = float(stations[index + 1] - stations[index])
            mean_speed = 0.5 * float(speeds[index + 1] + speeds[index])
            elapsed[index] = elapsed[index + 1] + ds / max(mean_speed, 1e-9)
        crossing_pairs = ((threshold_resource_id, 0.0), *resource_stations_m)
        crossings = tuple(
            ResourceCrossing(
                resource_id=resource_id,
                s_m=float(station),
                elapsed_time_s=float(np.interp(station, stations, elapsed)),
            )
            for resource_id, station in crossing_pairs
        )
        n = len(stations)
        cas = np.asarray(cas_mps, dtype=float)
        command = np.asarray(command_cas_mps if command_cas_mps is not None else cas, dtype=float)
        reference = np.asarray(
            reference_command_cas_mps if reference_command_cas_mps is not None else command,
            dtype=float,
        )
        return cls(
            template_id=template_id,
            cluster_id=cluster_id,
            s_m=stations,
            lat_deg=np.zeros(n, dtype=float) if lat_deg is None else lat_deg,
            lon_deg=np.zeros(n, dtype=float) if lon_deg is None else lon_deg,
            east_m=east_m,
            north_m=north_m,
            altitude_m=altitude_m,
            cas_mps=cas,
            tas_mps=cas if tas_mps is None else tas_mps,
            ground_speed_mps=speeds,
            command_cas_mps=command,
            reference_command_cas_mps=reference,
            lower_cas_mps=lower_cas_mps,
            upper_cas_mps=upper_cas_mps,
            elapsed_time_s=elapsed,
            resource_crossings=crossings,
            diagnostics=diagnostics or VariantDiagnostics(compiled_duration_s=float(elapsed[0])),
            action_provenance=action_provenance or ActionProvenance(),
        )


@dataclass(frozen=True, order=True)
class ActionStation:
    entry_order: int
    kind: StationKind
    grid_index: int
    s_m: float
    east_m: float
    north_m: float

    def __post_init__(self) -> None:
        if self.entry_order < 0 or self.grid_index < 0:
            raise ArtifactValidationError("action station indices cannot be negative")
        if not all(np.isfinite(value) for value in (self.s_m, self.east_m, self.north_m)):
            raise ArtifactValidationError("action station coordinates must be finite")


@dataclass(frozen=True)
class ClusterTemplate:
    cluster_id: str
    medoid_flight_id: str
    member_count: int
    baseline_variant: TrajectoryVariant
    speed_action_stations: tuple[ActionStation, ...]
    path_stretch_stations: tuple[ActionStation, ...]
    dataset_id: str = "synthetic"
    airport_id: str = "UNKNOWN"
    runway_id: str = "UNKNOWN"
    dispersion_m: float = 0.0
    expected_speed_station_count: int = 16
    expected_path_station_count: int = 8
    schema_version: str = "hailmary.cluster_template.v1"
    provenance: tuple[tuple[str, Any], ...] = ()
    template_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _freeze_metadata_pairs(self.provenance, name="template provenance"))
        if not self.cluster_id or not self.medoid_flight_id:
            raise ArtifactValidationError("cluster and medoid IDs cannot be empty")
        if self.member_count < 1:
            raise ArtifactValidationError("member_count must be positive")
        if self.baseline_variant.cluster_id != self.cluster_id:
            raise ArtifactValidationError("baseline variant cluster does not match template")
        if len(self.speed_action_stations) != self.expected_speed_station_count:
            raise ArtifactValidationError(
                f"template requires {self.expected_speed_station_count} distinct speed stations"
            )
        if len(self.path_stretch_stations) != self.expected_path_station_count:
            raise ArtifactValidationError(
                f"template requires {self.expected_path_station_count} distinct path-stretch stations"
            )
        self._validate_stations(self.speed_action_stations, kind="speed")
        self._validate_stations(self.path_stretch_stations, kind="path_stretch")

        computed = stable_id(
            "template",
            {
                "schema_version": self.schema_version,
                "dataset_id": self.dataset_id,
                "airport_id": self.airport_id,
                "runway_id": self.runway_id,
                "cluster_id": self.cluster_id,
                "medoid_flight_id": self.medoid_flight_id,
                "member_count": self.member_count,
                "dispersion_m": self.dispersion_m,
                "baseline_variant_id": self.baseline_variant.variant_id,
                "speed_action_stations": self.speed_action_stations,
                "path_stretch_stations": self.path_stretch_stations,
                "provenance": self.provenance,
            },
            length=32,
        )
        if self.template_id and self.template_id != computed:
            raise ArtifactValidationError("template_id does not match template content")
        object.__setattr__(self, "template_id", computed)

    def _validate_stations(self, stations: tuple[ActionStation, ...], *, kind: StationKind) -> None:
        if any(station.kind != kind for station in stations):
            raise ArtifactValidationError(f"{kind} station list contains another location type")
        if len({station.grid_index for station in stations}) != len(stations):
            raise ArtifactValidationError(f"{kind} action stations must be distinct")
        if tuple(station.entry_order for station in stations) != tuple(range(len(stations))):
            raise ArtifactValidationError(f"{kind} stations must preserve entry-to-final order")
        station_s = tuple(station.s_m for station in stations)
        if any(next_s >= current_s for current_s, next_s in zip(station_s, station_s[1:])):
            raise ArtifactValidationError(f"{kind} stations must move downstream in entry order")
        for station in stations:
            if station.grid_index >= len(self.baseline_variant.s_m):
                raise ArtifactValidationError(f"{kind} station grid index is outside the baseline variant")
            if abs(float(self.baseline_variant.s_m[station.grid_index]) - station.s_m) > 1e-6:
                raise ArtifactValidationError(f"{kind} station does not match its baseline grid index")

    @property
    def content_hash(self) -> str:
        return content_hash(self, namespace="cluster-template")

    def to_dict(self) -> dict[str, Any]:
        def station_payload(station: ActionStation) -> dict[str, Any]:
            return {
                "entry_order": station.entry_order,
                "kind": station.kind,
                "grid_index": station.grid_index,
                "s_m": station.s_m,
                "east_m": station.east_m,
                "north_m": station.north_m,
            }

        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "cluster_id": self.cluster_id,
            "medoid_flight_id": self.medoid_flight_id,
            "member_count": self.member_count,
            "dataset_id": self.dataset_id,
            "airport_id": self.airport_id,
            "runway_id": self.runway_id,
            "dispersion_m": self.dispersion_m,
            "expected_speed_station_count": self.expected_speed_station_count,
            "expected_path_station_count": self.expected_path_station_count,
            "provenance": canonical_data(self.provenance),
            "baseline_variant": self.baseline_variant.to_dict(),
            "speed_action_stations": [station_payload(item) for item in self.speed_action_stations],
            "path_stretch_stations": [station_payload(item) for item in self.path_stretch_stations],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClusterTemplate":
        def stations(name: str) -> tuple[ActionStation, ...]:
            return tuple(
                ActionStation(
                    entry_order=int(item["entry_order"]),
                    kind=str(item["kind"]),  # type: ignore[arg-type]
                    grid_index=int(item["grid_index"]),
                    s_m=float(item["s_m"]),
                    east_m=float(item["east_m"]),
                    north_m=float(item["north_m"]),
                )
                for item in payload[name]
            )

        return cls(
            cluster_id=str(payload["cluster_id"]),
            medoid_flight_id=str(payload["medoid_flight_id"]),
            member_count=int(payload["member_count"]),
            baseline_variant=TrajectoryVariant.from_dict(payload["baseline_variant"]),
            speed_action_stations=stations("speed_action_stations"),
            path_stretch_stations=stations("path_stretch_stations"),
            dataset_id=str(payload.get("dataset_id", "synthetic")),
            airport_id=str(payload.get("airport_id", "UNKNOWN")),
            runway_id=str(payload.get("runway_id", "UNKNOWN")),
            dispersion_m=float(payload.get("dispersion_m", 0.0)),
            expected_speed_station_count=int(payload.get("expected_speed_station_count", 16)),
            expected_path_station_count=int(payload.get("expected_path_station_count", 8)),
            schema_version=str(payload.get("schema_version", "hailmary.cluster_template.v1")),
            provenance=tuple((str(item[0]), item[1]) for item in payload.get("provenance", ())),
            template_id=str(payload.get("template_id", "")),
        )
