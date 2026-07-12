"""Read-only schedule protocol adapter for existing evaluator consumers.

Despite the compatibility-oriented module name, this file imports no mutable
scenario-manager implementation or HTTP model.  It only projects canonical
Hailmary definitions/states into fresh schedule dictionaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from hailmary.config import MPS_PER_KNOT
from hailmary.scenario.models import ScenarioDefinition, trajectory_variant_id
from hailmary.simulator.interpolation import MonotoneTrajectory
from hailmary.simulator.state import FlightLifecycle, SimulationState


def _variant_array(variant: object, *names: str) -> np.ndarray:
    for name in names:
        if hasattr(variant, name):
            value = np.asarray(getattr(variant, name), dtype=np.float64)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError(f"trajectory field {name!r} must be a finite vector")
            return value
    raise ValueError(f"trajectory is missing one of fields {names!r}")


def _diagnostics_payload(variant: object) -> dict[str, Any]:
    diagnostics = getattr(variant, "diagnostics", None)
    if diagnostics is None:
        return {"success": True, "message": "canonical Hailmary trajectory"}
    payload = asdict(diagnostics) if hasattr(diagnostics, "__dataclass_fields__") else {}
    payload["success"] = bool(payload.pop("feasible", True))
    details = payload.get("details", ())
    payload["details"] = dict(details)
    return payload


@dataclass(frozen=True, init=False)
class HailmaryScheduleView:
    """Implement ``arrival_schedule()`` over an immutable definition/state."""

    definition: ScenarioDefinition
    state: SimulationState | None
    include_completed: bool

    def __init__(
        self,
        source: ScenarioDefinition | SimulationState | object,
        *,
        include_completed: bool = True,
    ) -> None:
        if isinstance(source, SimulationState):
            definition = source.definition
            state: SimulationState | None = source
        elif isinstance(source, ScenarioDefinition):
            definition = source
            state = None
        else:
            candidate = getattr(source, "state", None)
            if not isinstance(candidate, SimulationState):
                raise TypeError("source must be a ScenarioDefinition, SimulationState, or Simulator")
            definition = candidate.definition
            state = candidate
        object.__setattr__(self, "definition", definition)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "include_completed", bool(include_completed))

    @property
    def state_id(self) -> str | None:
        return None if self.state is None else self.state.state_id

    def _runtime(self, flight_id: str) -> tuple[str, float, FlightLifecycle | None]:
        definition_flight = self.definition.flight(flight_id)
        if self.state is None:
            return definition_flight.baseline_variant_id, definition_flight.release_time_s, None
        dynamic = self.state.flight(flight_id)
        return (
            dynamic.current_variant_id,
            dynamic.trajectory_clock_origin_s,
            dynamic.lifecycle,
        )

    def _arrival(self, flight_id: str) -> dict[str, Any] | None:
        flight = self.definition.flight(flight_id)
        variant_id, release_time_s, lifecycle = self._runtime(flight_id)
        if lifecycle is FlightLifecycle.COMPLETED and not self.include_completed:
            return None
        variant = self.definition.variant(variant_id)
        trajectory = MonotoneTrajectory.from_variant(variant)
        order = trajectory.source_indices
        lat = _variant_array(variant, "lat_deg")[order]
        lon = _variant_array(variant, "lon_deg")[order]
        altitude = _variant_array(variant, "altitude_m", "h_m", "geoaltitude_m")[order]
        cas = _variant_array(variant, "cas_mps", "v_cas_mps")[order]
        if any(len(values) != len(order) for values in (lat, lon, altitude, cas)):
            raise ValueError("trajectory schedule fields must share one canonical grid")
        absolute_time = release_time_s + trajectory.elapsed_time_s
        points = [
            [float(time_s), float(lat_deg), float(lon_deg), float(altitude_m), 3]
            for time_s, lat_deg, lon_deg, altitude_m in zip(
                absolute_time,
                lat,
                lon,
                altitude,
                strict=True,
            )
        ]
        action_provenance = getattr(variant, "action_provenance", None)
        lever = str(getattr(action_provenance, "lever", "baseline"))
        route_type = "base-route" if lever == "baseline" else lever.replace("_", "-")
        payload: dict[str, Any] = {
            "scenario_id": self.definition.scenario_id,
            "state_id": self.state_id,
            "flight_id": flight.flight_id,
            "callsign": flight.callsign or flight.flight_id,
            "icao24": flight.icao24,
            "runway": flight.runway,
            "cluster_id": flight.cluster_id or str(getattr(variant, "cluster_id", "")),
            "variant_id": trajectory_variant_id(variant),
            "route_type": route_type,
            "time_at_first_fix": float(absolute_time[0]),
            "time_at_last_event": float(absolute_time[-1]),
            "first_time": float(absolute_time[0]),
            "last_time": float(absolute_time[-1]),
            "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
            "breakpoint_mask_bits": {"lateral": 1, "altitude": 2},
            "points": points,
            "raw_point_count": len(points),
            "compressed_point_count": len(points),
            "lateral_tolerance_m": 0.0,
            "altitude_tolerance_m": 0.0,
            "cas_profile": {
                "columns": ["time", "cas_kts"],
                "units": {"cas_kts": "kt"},
                "source": "hailmary_canonical_variant",
                "points": [
                    [float(time_s), float(cas_mps / MPS_PER_KNOT)]
                    for time_s, cas_mps in zip(absolute_time, cas, strict=True)
                ],
            },
            "simulation": _diagnostics_payload(variant),
        }
        if lifecycle is not None:
            payload["lifecycle"] = lifecycle.value
        return payload

    def arrival_schedule(self) -> list[dict[str, Any]]:
        """Return a fresh, threshold-ETA-ordered schedule on every call."""

        arrivals = [
            arrival
            for flight in self.definition.flights
            if (arrival := self._arrival(flight.flight_id)) is not None
        ]
        arrivals.sort(key=lambda item: (float(item["time_at_last_event"]), str(item["flight_id"])))
        return arrivals

    def departure_schedule(self) -> list[dict[str, Any]]:
        return []


ScenarioManagerScheduleAdapter = HailmaryScheduleView
HailmaryArrivalScheduleAdapter = HailmaryScheduleView


__all__ = [
    "HailmaryArrivalScheduleAdapter",
    "HailmaryScheduleView",
    "ScenarioManagerScheduleAdapter",
]
