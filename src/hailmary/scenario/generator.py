"""Generic deterministic scenario materialization.

Traffic-window construction lives in :mod:`hailmary.scenario.traffic`. The
controlled-factorial generator was intentionally removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Iterable, Mapping

import numpy as np

from hailmary.scenario.models import (
    ActionStationDefinition,
    FlightDefinition,
    MaterializedExogenousEvent,
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
    SegmentTraversalDefinition,
)


@dataclass(frozen=True, slots=True)
class FlightGenerationSpec:
    flight_id: str
    baseline_variant_id: str
    observed_release_time_s: float
    cluster_id: str = ""
    callsign: str = ""
    icao24: str = ""
    runway: str = ""
    release_offset_s: float | None = None
    action_stations: tuple[ActionStationDefinition, ...] = ()
    resource_crossings: tuple[ResourceCrossingDefinition, ...] = ()
    segment_traversals: tuple[SegmentTraversalDefinition, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.flight_id or not self.baseline_variant_id:
            raise ValueError("flight and baseline-variant identities must be non-empty")
        if not math.isfinite(self.observed_release_time_s):
            raise ValueError("observed_release_time_s must be finite")
        if self.release_offset_s is not None and not math.isfinite(self.release_offset_s):
            raise ValueError("release_offset_s must be finite when supplied")

    @classmethod
    def from_template(
        cls,
        *,
        flight_id: str,
        observed_release_time_s: float,
        template: object,
        callsign: str = "",
        icao24: str = "",
        runway: str = "",
        release_offset_s: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FlightGenerationSpec":
        baseline = getattr(template, "baseline_variant", None)
        variant_id = getattr(baseline, "variant_id", "")
        cluster_id = str(getattr(template, "cluster_id", ""))
        if baseline is None or not variant_id or not cluster_id:
            raise TypeError("template must expose cluster_id and baseline_variant.variant_id")
        raw_stations = (
            *tuple(getattr(template, "speed_action_stations", ())),
            *tuple(getattr(template, "path_stretch_stations", ())),
        )
        stations = tuple(
            ActionStationDefinition(
                station_index=int(getattr(station, "entry_order")),
                s_m=float(getattr(station, "s_m")),
                station_type=str(getattr(station, "kind")),
            )
            for station in raw_stations
        )
        return cls(
            flight_id=flight_id,
            baseline_variant_id=str(variant_id),
            observed_release_time_s=float(observed_release_time_s),
            cluster_id=cluster_id,
            callsign=callsign,
            icao24=icao24,
            runway=runway or str(getattr(template, "runway_id", "")),
            release_offset_s=release_offset_s,
            action_stations=stations,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class ScenarioGenerator:
    """Deterministically materialize a scenario from release specifications."""

    master_seed: int

    def named_rng(self, stream_name: str) -> np.random.Generator:
        if not stream_name:
            raise ValueError("stream_name must be non-empty")
        digest = hashlib.sha256(f"{self.master_seed}:{stream_name}".encode("utf-8")).digest()
        return np.random.default_rng(int.from_bytes(digest[:16], "big"))

    def generate(
        self,
        *,
        scenario_id: str,
        flight_specs: Iterable[FlightGenerationSpec],
        variants: Iterable[object],
        resources: Iterable[ResourceDefinition],
        exogenous_events: Iterable[MaterializedExogenousEvent] = (),
        release_jitter_s: float = 0.0,
        weather: object | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ScenarioDefinition:
        jitter = float(release_jitter_s)
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError("release_jitter_s must be finite and non-negative")
        flights: list[FlightDefinition] = []
        for spec in sorted(flight_specs, key=lambda item: item.flight_id):
            if spec.release_offset_s is not None:
                release_offset_s = float(spec.release_offset_s)
            elif jitter > 0.0:
                release_offset_s = float(
                    self.named_rng(f"release:{spec.flight_id}").uniform(-jitter, jitter)
                )
            else:
                release_offset_s = 0.0
            flights.append(
                FlightDefinition(
                    flight_id=spec.flight_id,
                    release_time_s=float(spec.observed_release_time_s + release_offset_s),
                    baseline_variant_id=spec.baseline_variant_id,
                    cluster_id=spec.cluster_id,
                    callsign=spec.callsign,
                    icao24=spec.icao24,
                    runway=spec.runway,
                    observed_release_time_s=float(spec.observed_release_time_s),
                    release_offset_s=release_offset_s,
                    action_stations=spec.action_stations,
                    resource_crossings=spec.resource_crossings,
                    segment_traversals=spec.segment_traversals,
                    metadata=spec.metadata,
                )
            )
        return ScenarioDefinition(
            scenario_id=scenario_id,
            seed=int(self.master_seed),
            flights=tuple(flights),
            resources=tuple(sorted(resources, key=lambda item: item.resource_id)),
            variants=tuple(variants),
            exogenous_events=tuple(
                sorted(exogenous_events, key=lambda item: (item.time_s, item.event_id))
            ),
            weather=weather,
            metadata={} if metadata is None else metadata,
        )


def build_scenario_definition(
    *,
    scenario_id: str,
    seed: int,
    flights: Iterable[FlightDefinition],
    resources: Iterable[ResourceDefinition],
    variants: Iterable[object],
    exogenous_events: Iterable[MaterializedExogenousEvent] = (),
    weather: object | None = None,
    metadata: Mapping[str, Any] | None = None,
    schema_version: str = "hailmary.scenario.v2",
) -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id=scenario_id,
        seed=int(seed),
        flights=tuple(sorted(flights, key=lambda item: item.flight_id)),
        resources=tuple(sorted(resources, key=lambda item: item.resource_id)),
        variants=tuple(variants),
        exogenous_events=tuple(
            sorted(exogenous_events, key=lambda item: (item.time_s, item.event_id))
        ),
        weather=weather,
        metadata={} if metadata is None else metadata,
        schema_version=schema_version,
    )


__all__ = ["FlightGenerationSpec", "ScenarioGenerator", "build_scenario_definition"]
