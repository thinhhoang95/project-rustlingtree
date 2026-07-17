from __future__ import annotations

from dataclasses import InitVar, dataclass, field, replace
from enum import StrEnum
import json
import math
from typing import Any, Mapping

import numpy as np

from hailmary.errors import StaleActionError
from hailmary.scenario.models import (
    FrozenPayload,
    ScenarioDefinition,
    freeze_payload,
    thaw_payload,
)
from hailmary.simulator.events import ScheduledEvent
from hailmary.simulator.hashing import dynamic_content_hash, provenance_state_id


SIMULATION_SNAPSHOT_VERSION = "hailmary.simulation-snapshot.v2"
LEGACY_SIMULATION_SNAPSHOT_VERSION = "hailmary.simulation-snapshot.v1"


class StaleStateError(StaleActionError):
    pass


class FlightLifecycle(StrEnum):
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class FlightDynamic:
    flight_id: str
    lifecycle: FlightLifecycle
    current_variant_id: str
    release_time_s: float
    trajectory_origin_time_s: float | None = None
    crossed_action_station_indices: tuple[int, ...] = ()
    crossed_action_station_keys: tuple[tuple[str, int], ...] = ()
    crossed_resource_ids: tuple[str, ...] = ()
    predicted_resource_crossing_times: tuple[tuple[str, float], ...] = ()
    action_history: tuple[str, ...] = ()
    speed_action_count: int = 0
    path_stretch_count: int = 0
    action_station_fraction_cap: float | None = None
    intervention_budget_fraction_cap: float | None = None
    intercept_gate_active: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "lifecycle", FlightLifecycle(self.lifecycle))
        if not self.flight_id or not self.current_variant_id:
            raise ValueError("flight dynamic identity and variant must be non-empty")
        if not math.isfinite(self.release_time_s):
            raise ValueError("flight release_time_s must be finite")
        origin = (
            self.release_time_s
            if self.trajectory_origin_time_s is None
            else float(self.trajectory_origin_time_s)
        )
        if not math.isfinite(origin):
            raise ValueError("flight trajectory_origin_time_s must be finite")
        object.__setattr__(self, "trajectory_origin_time_s", origin)
        object.__setattr__(
            self,
            "crossed_action_station_indices",
            tuple(int(index) for index in self.crossed_action_station_indices),
        )
        object.__setattr__(
            self,
            "crossed_action_station_keys",
            tuple(
                (str(station_type), int(index))
                for station_type, index in self.crossed_action_station_keys
            ),
        )
        object.__setattr__(
            self,
            "crossed_resource_ids",
            tuple(str(item) for item in self.crossed_resource_ids),
        )
        object.__setattr__(
            self,
            "predicted_resource_crossing_times",
            tuple(
                (str(resource_id), float(time_s))
                for resource_id, time_s in self.predicted_resource_crossing_times
            ),
        )
        object.__setattr__(
            self, "action_history", tuple(str(item) for item in self.action_history)
        )
        if self.speed_action_count < 0 or self.path_stretch_count < 0:
            raise ValueError("action counters must be non-negative")
        for name in ("action_station_fraction_cap", "intervention_budget_fraction_cap"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must lie in [0, 1] when supplied")
        object.__setattr__(
            self, "intercept_gate_active", bool(self.intercept_gate_active)
        )
        if len(set(self.crossed_action_station_indices)) != len(
            self.crossed_action_station_indices
        ):
            raise ValueError("crossed action-station indices must be unique")
        if len(set(self.crossed_action_station_keys)) != len(
            self.crossed_action_station_keys
        ):
            raise ValueError("crossed action-station keys must be unique")
        if len(set(self.crossed_resource_ids)) != len(self.crossed_resource_ids):
            raise ValueError("crossed resource IDs must be unique")
        for _, time_s in self.predicted_resource_crossing_times:
            if not math.isfinite(time_s):
                raise ValueError("predicted crossing times must be finite")

    @property
    def station_cursor(self) -> int:
        if not self.crossed_action_station_indices:
            return -1
        return max(self.crossed_action_station_indices)

    @property
    def trajectory_clock_origin_s(self) -> float:
        """Absolute time corresponding to elapsed time zero on the live variant."""

        assert self.trajectory_origin_time_s is not None
        return self.trajectory_origin_time_s

    def predicted_resource_time(self, resource_id: str) -> float | None:
        for candidate, time_s in self.predicted_resource_crossing_times:
            if candidate == resource_id:
                return time_s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "flight_id": self.flight_id,
            "lifecycle": self.lifecycle.value,
            "current_variant_id": self.current_variant_id,
            "release_time_s": self.release_time_s,
            "trajectory_origin_time_s": self.trajectory_origin_time_s,
            "crossed_action_station_indices": list(self.crossed_action_station_indices),
            "crossed_action_station_keys": [
                list(item) for item in self.crossed_action_station_keys
            ],
            "crossed_resource_ids": list(self.crossed_resource_ids),
            "predicted_resource_crossing_times": [
                list(item) for item in self.predicted_resource_crossing_times
            ],
            "action_history": list(self.action_history),
            "speed_action_count": self.speed_action_count,
            "path_stretch_count": self.path_stretch_count,
            "action_station_fraction_cap": self.action_station_fraction_cap,
            "intervention_budget_fraction_cap": self.intervention_budget_fraction_cap,
            "intercept_gate_active": self.intercept_gate_active,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FlightDynamic":
        return cls(
            flight_id=str(payload["flight_id"]),
            lifecycle=FlightLifecycle(str(payload["lifecycle"])),
            current_variant_id=str(payload["current_variant_id"]),
            release_time_s=float(payload["release_time_s"]),
            trajectory_origin_time_s=float(
                payload.get("trajectory_origin_time_s", payload["release_time_s"])
            ),
            crossed_action_station_indices=tuple(
                int(item) for item in payload.get("crossed_action_station_indices", [])
            ),
            crossed_action_station_keys=tuple(
                (str(item[0]), int(item[1]))
                for item in payload.get("crossed_action_station_keys", [])
            ),
            crossed_resource_ids=tuple(
                str(item) for item in payload.get("crossed_resource_ids", [])
            ),
            predicted_resource_crossing_times=tuple(
                (str(item[0]), float(item[1]))
                for item in payload.get("predicted_resource_crossing_times", [])
            ),
            action_history=tuple(
                str(item) for item in payload.get("action_history", [])
            ),
            speed_action_count=int(payload.get("speed_action_count", 0)),
            path_stretch_count=int(payload.get("path_stretch_count", 0)),
            action_station_fraction_cap=(
                None
                if payload.get("action_station_fraction_cap") is None
                else float(payload["action_station_fraction_cap"])
            ),
            intervention_budget_fraction_cap=(
                None
                if payload.get("intervention_budget_fraction_cap") is None
                else float(payload["intervention_budget_fraction_cap"])
            ),
            intercept_gate_active=bool(payload.get("intercept_gate_active", False)),
        )


@dataclass(frozen=True, slots=True)
class SimulationState:
    definition: ScenarioDefinition
    state_id: str
    parent_state_id: str | None
    branch_label: str
    sim_time_s: float
    version: int
    event_sequence: int
    flights: tuple[FlightDynamic, ...]
    event_heap: tuple[ScheduledEvent, ...]
    rng_state_json: str
    metrics: tuple[tuple[str, float], ...] = ()
    action_log: tuple[FrozenPayload, ...] = ()
    exogenous_state: FrozenPayload = ()
    exogenous_event_log: tuple[FrozenPayload, ...] = ()
    decision_epoch_index: int = 0
    _precomputed_dynamic_content_hash: InitVar[str | None] = None
    _dynamic_content_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _precomputed_dynamic_content_hash: str | None) -> None:
        if not self.branch_label:
            raise ValueError("branch_label must be non-empty")
        if not math.isfinite(self.sim_time_s):
            raise ValueError("sim_time_s must be finite")
        if self.version < 0 or self.event_sequence < 0 or self.decision_epoch_index < 0:
            raise ValueError("state counters must be non-negative")
        object.__setattr__(self, "flights", tuple(self.flights))
        object.__setattr__(self, "event_heap", tuple(self.event_heap))
        object.__setattr__(
            self,
            "metrics",
            tuple((str(key), float(value)) for key, value in self.metrics),
        )
        object.__setattr__(
            self, "action_log", tuple(freeze_payload(item) for item in self.action_log)
        )
        object.__setattr__(
            self, "exogenous_state", freeze_payload(self.exogenous_state)
        )
        object.__setattr__(
            self,
            "exogenous_event_log",
            tuple(freeze_payload(item) for item in self.exogenous_event_log),
        )
        _validate_rng_state_json(self.rng_state_json)
        if len({flight.flight_id for flight in self.flights}) != len(self.flights):
            raise ValueError("state flight IDs must be unique")
        if {flight.flight_id for flight in self.flights} != {
            flight.flight_id for flight in self.definition.flights
        }:
            raise ValueError(
                "state flights must exactly match scenario-definition flights"
            )
        resolved_hash = (
            dynamic_content_hash(self)
            if _precomputed_dynamic_content_hash is None
            else str(_precomputed_dynamic_content_hash)
        )
        if not resolved_hash:
            raise ValueError("precomputed dynamic-content hash cannot be empty")
        object.__setattr__(self, "_dynamic_content_hash", resolved_hash)

    @property
    def dynamic_content_hash(self) -> str:
        return self._dynamic_content_hash

    @property
    def state_version(self) -> int:
        return self.version

    @property
    def rng_state(self) -> dict[str, Any]:
        return json.loads(self.rng_state_json)

    @property
    def metrics_dict(self) -> dict[str, float]:
        return dict(self.metrics)

    @property
    def action_log_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(thaw_payload(item) for item in self.action_log)

    @property
    def exogenous_state_dict(self) -> dict[str, Any]:
        return thaw_payload(self.exogenous_state)

    @property
    def exogenous_event_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(thaw_payload(item) for item in self.exogenous_event_log)

    def flight(self, flight_id: str) -> FlightDynamic:
        for flight in self.flights:
            if flight.flight_id == flight_id:
                return flight
        raise KeyError(f"unknown flight {flight_id!r}")

    def assert_fresh(
        self,
        *,
        expected_version: int | None = None,
        expected_state_id: str | None = None,
    ) -> None:
        if expected_version is not None and expected_version != self.version:
            raise StaleStateError(
                f"stale simulation state version {expected_version}; current version is {self.version}"
            )
        if expected_state_id is not None and expected_state_id != self.state_id:
            raise StaleStateError(
                f"stale simulation state ID {expected_state_id!r}; current ID is {self.state_id!r}"
            )

    def to_snapshot(
        self,
        *,
        runtime_configuration_hash: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": SIMULATION_SNAPSHOT_VERSION,
            "runtime_configuration_hash": runtime_configuration_hash,
            "definition_hash": self.definition.definition_hash,
            "state_id": self.state_id,
            "parent_state_id": self.parent_state_id,
            "branch_label": self.branch_label,
            "sim_time_s": self.sim_time_s,
            "version": self.version,
            "event_sequence": self.event_sequence,
            "flights": [flight.to_dict() for flight in self.flights],
            "event_heap": [event.to_dict() for event in self.event_heap],
            "rng_state": self.rng_state,
            "metrics": [[key, value] for key, value in self.metrics],
            "action_log": list(self.action_log_records),
            "exogenous_state": self.exogenous_state_dict,
            "exogenous_event_log": list(self.exogenous_event_records),
            "decision_epoch_index": self.decision_epoch_index,
            "dynamic_content_hash": self.dynamic_content_hash,
        }

    @classmethod
    def from_snapshot(
        cls, definition: ScenarioDefinition, payload: Mapping[str, Any]
    ) -> "SimulationState":
        schema_version = payload.get("schema_version")
        if schema_version not in {
            LEGACY_SIMULATION_SNAPSHOT_VERSION,
            SIMULATION_SNAPSHOT_VERSION,
        }:
            raise ValueError("unsupported simulation snapshot schema version")
        if schema_version == SIMULATION_SNAPSHOT_VERSION:
            if "runtime_configuration_hash" not in payload:
                raise ValueError(
                    "simulation snapshot v2 requires runtime_configuration_hash"
                )
            runtime_hash = payload["runtime_configuration_hash"]
            if runtime_hash is not None and (
                type(runtime_hash) is not str
                or not runtime_hash
                or runtime_hash != runtime_hash.strip()
            ):
                raise ValueError(
                    "snapshot runtime_configuration_hash must be null or a non-empty exact string"
                )
        elif "runtime_configuration_hash" in payload:
            raise ValueError(
                "legacy simulation snapshots cannot contain runtime_configuration_hash"
            )
        if str(payload.get("definition_hash", "")) != definition.definition_hash:
            raise ValueError(
                "snapshot definition hash does not match supplied scenario definition"
            )
        state = cls(
            definition=definition,
            state_id=str(payload["state_id"]),
            parent_state_id=(
                None
                if payload.get("parent_state_id") is None
                else str(payload.get("parent_state_id"))
            ),
            branch_label=str(payload["branch_label"]),
            sim_time_s=float(payload["sim_time_s"]),
            version=int(payload["version"]),
            event_sequence=int(payload["event_sequence"]),
            flights=tuple(FlightDynamic.from_dict(item) for item in payload["flights"]),
            event_heap=tuple(
                ScheduledEvent.from_dict(item) for item in payload["event_heap"]
            ),
            rng_state_json=_canonical_rng_state_json(payload["rng_state"]),
            metrics=tuple(
                (str(item[0]), float(item[1])) for item in payload.get("metrics", [])
            ),
            action_log=tuple(
                freeze_payload(item) for item in payload.get("action_log", [])
            ),
            exogenous_state=freeze_payload(payload.get("exogenous_state", {})),
            exogenous_event_log=tuple(
                freeze_payload(item) for item in payload.get("exogenous_event_log", [])
            ),
            decision_epoch_index=int(payload.get("decision_epoch_index", 0)),
        )
        expected_hash = payload.get("dynamic_content_hash")
        if (
            expected_hash is not None
            and str(expected_hash) != state.dynamic_content_hash
        ):
            raise ValueError(
                "snapshot dynamic-content hash does not match reconstructed state"
            )
        return state


def initial_rng_state_json(seed: int) -> str:
    return _canonical_rng_state_json(np.random.default_rng(seed).bit_generator.state)


def rng_from_state_json(rng_state_json: str) -> np.random.Generator:
    state = json.loads(rng_state_json)
    bit_generator_name = str(state.get("bit_generator", "PCG64"))
    bit_generator_type = getattr(np.random, bit_generator_name, None)
    if bit_generator_type is None:
        raise ValueError(f"unsupported NumPy bit generator {bit_generator_name!r}")
    bit_generator = bit_generator_type()
    bit_generator.state = state
    return np.random.Generator(bit_generator)


def rng_state_json(generator: np.random.Generator) -> str:
    return _canonical_rng_state_json(generator.bit_generator.state)


def make_initial_state(
    definition: ScenarioDefinition,
    *,
    flights: tuple[FlightDynamic, ...],
    event_heap: tuple[ScheduledEvent, ...],
    event_sequence: int,
) -> SimulationState:
    sim_time_s = min((event.time_s for event in event_heap), default=0.0)
    pending = SimulationState(
        definition=definition,
        state_id="",
        parent_state_id=None,
        branch_label="root",
        sim_time_s=float(sim_time_s),
        version=0,
        event_sequence=event_sequence,
        flights=flights,
        event_heap=event_heap,
        rng_state_json=initial_rng_state_json(definition.seed),
    )
    state_id = provenance_state_id(
        pending.dynamic_content_hash,
        parent_state_id=None,
        branch_label="root",
        transition="initial",
        lineage_version=0,
    )
    return replace(
        pending,
        state_id=state_id,
        _precomputed_dynamic_content_hash=pending.dynamic_content_hash,
    )


def evolve_state(
    previous: SimulationState,
    *,
    transition: str,
    version_increment: int = 1,
    **changes: Any,
) -> SimulationState:
    version = previous.version + int(version_increment)
    pending = replace(
        previous,
        state_id="",
        parent_state_id=previous.state_id,
        version=version,
        **changes,
    )
    state_id = provenance_state_id(
        pending.dynamic_content_hash,
        parent_state_id=previous.state_id,
        branch_label=pending.branch_label,
        transition=transition,
        lineage_version=version,
    )
    return replace(
        pending,
        state_id=state_id,
        _precomputed_dynamic_content_hash=pending.dynamic_content_hash,
    )


def fork_state(parent: SimulationState, *, label: str) -> SimulationState:
    if not label:
        raise ValueError("fork label must be non-empty")
    # Reconstruct each branch-local collection.  The ScenarioDefinition and its
    # immutable trajectory arrays remain shared by identity.
    pending = SimulationState(
        definition=parent.definition,
        state_id="",
        parent_state_id=parent.state_id,
        branch_label=str(label),
        sim_time_s=parent.sim_time_s,
        version=parent.version,
        event_sequence=parent.event_sequence,
        flights=parent.flights,
        event_heap=tuple(list(parent.event_heap)),
        rng_state_json=str(parent.rng_state_json),
        metrics=parent.metrics,
        action_log=parent.action_log,
        exogenous_state=parent.exogenous_state,
        exogenous_event_log=parent.exogenous_event_log,
        decision_epoch_index=parent.decision_epoch_index,
        _precomputed_dynamic_content_hash=parent.dynamic_content_hash,
    )
    state_id = provenance_state_id(
        pending.dynamic_content_hash,
        parent_state_id=parent.state_id,
        branch_label=str(label),
        transition="fork",
        lineage_version=parent.version,
    )
    return replace(
        pending,
        state_id=state_id,
        _precomputed_dynamic_content_hash=pending.dynamic_content_hash,
    )


def _canonical_rng_state_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_rng_state_json(value: str) -> None:
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or "bit_generator" not in parsed:
        raise ValueError("rng_state_json must encode a NumPy bit-generator state")


__all__ = [
    "FlightDynamic",
    "FlightLifecycle",
    "LEGACY_SIMULATION_SNAPSHOT_VERSION",
    "SIMULATION_SNAPSHOT_VERSION",
    "SimulationState",
    "StaleStateError",
    "evolve_state",
    "fork_state",
    "initial_rng_state_json",
    "make_initial_state",
    "rng_from_state_json",
    "rng_state_json",
]
