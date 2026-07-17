from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
from typing import Any, Callable, Mapping

import numpy as np

from hailmary.errors import SimulationError
from hailmary.scenario.models import (
    FlightDefinition,
    ScenarioDefinition,
    freeze_payload,
    trajectory_variant_id,
)
from hailmary.simulator.events import (
    DecisionEpoch,
    EventBatchResult,
    EventKind,
    ScheduledEvent,
)
from hailmary.simulator.interpolation import MonotoneTrajectory, TrajectorySample
from hailmary.simulator.state import (
    FlightDynamic,
    FlightLifecycle,
    LEGACY_SIMULATION_SNAPSHOT_VERSION,
    SIMULATION_SNAPSHOT_VERSION,
    SimulationState,
    evolve_state,
    fork_state,
    make_initial_state,
    rng_from_state_json,
    rng_state_json,
)


_FLIGHT_FUTURE_EVENT_KINDS = frozenset(
    {
        EventKind.ACTION_STATION_CROSSED,
        EventKind.RESOURCE_CROSSED,
        EventKind.FLIGHT_COMPLETED,
    }
)


@dataclass(frozen=True, slots=True)
class PolicyDecisionContext:
    simulator: "Simulator"
    state: SimulationState
    decision_epoch: DecisionEpoch
    event_batch: EventBatchResult


def _runtime_configuration_hash(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError("runtime_configuration_hash must be a string or None")
    resolved = value.strip()
    if not resolved:
        raise ValueError("runtime_configuration_hash cannot be blank")
    if resolved != value:
        raise ValueError("runtime_configuration_hash must be an exact string")
    return resolved


class Simulator:
    """Mutable driver around immutable, lineage-tracked simulation states."""

    def __init__(
        self,
        definition: ScenarioDefinition | None = None,
        *,
        state: SimulationState | None = None,
        action_applier: Callable[["Simulator", Any], Any] | None = None,
        runtime_configuration_hash: str | None = None,
    ) -> None:
        if (definition is None) == (state is None):
            raise ValueError("provide exactly one of definition or state")
        resolved_runtime_hash = _runtime_configuration_hash(runtime_configuration_hash)
        if action_applier is not None and resolved_runtime_hash is None:
            raise ValueError(
                "action_applier requires a non-empty runtime_configuration_hash"
            )
        self.state = initial_state(definition) if state is None else state
        self._action_applier = action_applier
        self._runtime_configuration_hash = resolved_runtime_hash
        self._fork_origin_dynamic_content_hash: str | None = None
        self.last_run_batches: tuple[EventBatchResult, ...] = ()
        self.last_action_result: Any | None = None

    @classmethod
    def from_state(
        cls,
        state: SimulationState,
        *,
        action_applier: Callable[["Simulator", Any], Any] | None = None,
        runtime_configuration_hash: str | None = None,
    ) -> "Simulator":
        return cls(
            state=state,
            action_applier=action_applier,
            runtime_configuration_hash=runtime_configuration_hash,
        )

    @classmethod
    def resume(
        cls,
        definition: ScenarioDefinition,
        snapshot: Mapping[str, Any],
        *,
        action_applier: Callable[["Simulator", Any], Any] | None = None,
        runtime_configuration_hash: str | None = None,
    ) -> "Simulator":
        resolved_runtime_hash = _runtime_configuration_hash(runtime_configuration_hash)
        schema_version = snapshot.get("schema_version")
        if schema_version == SIMULATION_SNAPSHOT_VERSION:
            if "runtime_configuration_hash" not in snapshot:
                raise ValueError(
                    "simulation snapshot v2 requires runtime_configuration_hash"
                )
            snapshot_runtime_hash = _runtime_configuration_hash(
                snapshot["runtime_configuration_hash"]
            )
            if snapshot_runtime_hash != resolved_runtime_hash:
                raise ValueError(
                    "snapshot runtime_configuration_hash does not match the resume runtime"
                )
        elif schema_version == LEGACY_SIMULATION_SNAPSHOT_VERSION:
            if resolved_runtime_hash is not None:
                raise ValueError(
                    "legacy simulation snapshot cannot prove the configured resume runtime"
                )
        else:
            raise ValueError("unsupported simulation snapshot schema version")
        return cls(
            state=SimulationState.from_snapshot(definition, snapshot),
            action_applier=action_applier,
            runtime_configuration_hash=resolved_runtime_hash,
        )

    @property
    def definition(self) -> ScenarioDefinition:
        return self.state.definition

    @property
    def action_applier(self) -> Callable[["Simulator", Any], Any] | None:
        return self._action_applier

    @property
    def runtime_configuration_hash(self) -> str | None:
        return self._runtime_configuration_hash

    @property
    def fork_origin_dynamic_content_hash(self) -> str | None:
        return self._fork_origin_dynamic_content_hash

    @property
    def dynamic_content_hash(self) -> str:
        return self.state.dynamic_content_hash

    @property
    def next_event_time_s(self) -> float | None:
        if not self.state.event_heap:
            return None
        return float(self.state.event_heap[0].time_s)

    def snapshot(self) -> dict[str, Any]:
        return self.state.to_snapshot(
            runtime_configuration_hash=self._runtime_configuration_hash
        )

    def fork(self, *, label: str) -> "Simulator":
        child = Simulator.from_state(
            fork_state(self.state, label=label),
            action_applier=self._action_applier,
            runtime_configuration_hash=self._runtime_configuration_hash,
        )
        child._fork_origin_dynamic_content_hash = self.dynamic_content_hash
        return child

    def assert_fresh(
        self,
        *,
        expected_version: int | None = None,
        expected_state_id: str | None = None,
    ) -> None:
        self.state.assert_fresh(
            expected_version=expected_version,
            expected_state_id=expected_state_id,
        )

    def sample_flight(
        self, flight_id: str, *, at_time_s: float | None = None
    ) -> TrajectorySample:
        dynamic = self.state.flight(flight_id)
        absolute_time_s = (
            self.state.sim_time_s if at_time_s is None else float(at_time_s)
        )
        trajectory = MonotoneTrajectory.from_variant(
            self.definition.variant(dynamic.current_variant_id)
        )
        elapsed = absolute_time_s - dynamic.trajectory_clock_origin_s
        return trajectory.sample(elapsed)

    def advance_next(self) -> EventBatchResult | None:
        if not self.state.event_heap:
            return None

        before = self.state
        heap = list(before.event_heap)
        heapq.heapify(heap)
        time_s = float(heap[0].time_s)
        events: list[ScheduledEvent] = []
        while heap and float(heap[0].time_s) == time_s:
            events.append(heapq.heappop(heap))
        events.sort(key=lambda event: event.sort_key)

        by_flight = {flight.flight_id: flight for flight in before.flights}
        exogenous_state = before.exogenous_state_dict
        exogenous_log = list(before.exogenous_event_log)
        metrics = before.metrics_dict
        next_sequence = before.event_sequence
        invalidated_current_flights: set[str] = set()
        applied_events: list[ScheduledEvent] = []
        for event in events:
            if event.kind is EventKind.EXOGENOUS_DISTURBANCE:
                payload = event.payload_dict
                updates = payload.get("state_updates", {})
                if not isinstance(updates, Mapping):
                    raise ValueError("exogenous state_updates must be a mapping")
                exogenous_state.update(
                    {str(key): value for key, value in updates.items()}
                )
                metric_deltas = payload.get("metric_deltas", {})
                if not isinstance(metric_deltas, Mapping):
                    raise ValueError("exogenous metric_deltas must be a mapping")
                for name, raw_delta in metric_deltas.items():
                    delta = float(raw_delta)
                    if not math.isfinite(delta):
                        raise ValueError("exogenous metric deltas must be finite")
                    metrics[str(name)] = metrics.get(str(name), 0.0) + delta
                exogenous_log.append(
                    freeze_payload(
                        {
                            "event_id": event.event_id,
                            "time_s": event.time_s,
                            "stream_name": payload.get("stream_name", "exogenous"),
                            "payload": payload,
                        }
                    )
                )
                raw_shift = payload.get("flight_time_shift_s")
                if raw_shift is not None:
                    shift_s = float(raw_shift)
                    if not math.isfinite(shift_s) or shift_s < 0.0:
                        raise ValueError(
                            "exogenous flight_time_shift_s must be finite and non-negative"
                        )
                    target_flight_id = str(
                        payload.get("target_flight_id", event.flight_id)
                    )
                    if not target_flight_id:
                        raise ValueError(
                            "flight-time disturbances require target_flight_id"
                        )
                    try:
                        target = by_flight[target_flight_id]
                    except KeyError as exc:
                        raise ValueError(
                            f"exogenous disturbance references unknown flight {target_flight_id!r}"
                        ) from exc
                    if target.lifecycle is FlightLifecycle.COMPLETED:
                        raise ValueError("cannot time-shift a completed flight")
                    if shift_s > 0.0:
                        lifecycle_before_shift = target.lifecycle
                        target = replace(
                            target,
                            release_time_s=(
                                target.release_time_s + shift_s
                                if target.lifecycle is FlightLifecycle.SCHEDULED
                                else target.release_time_s
                            ),
                            trajectory_origin_time_s=target.trajectory_clock_origin_s
                            + shift_s,
                        )
                        if lifecycle_before_shift is FlightLifecycle.ACTIVE:
                            pending_to_shift = [
                                pending
                                for pending in heap
                                if pending.flight_id == target_flight_id
                                and pending.kind in _FLIGHT_FUTURE_EVENT_KINDS
                            ]
                            if target_flight_id not in invalidated_current_flights:
                                pending_to_shift.extend(
                                    pending
                                    for pending in events
                                    if pending.flight_id == target_flight_id
                                    and pending.kind in _FLIGHT_FUTURE_EVENT_KINDS
                                )
                            heap = [
                                pending
                                for pending in heap
                                if not (
                                    pending.flight_id == target_flight_id
                                    and pending.kind in _FLIGHT_FUTURE_EVENT_KINDS
                                )
                            ]
                            predicted = dict(target.predicted_resource_crossing_times)
                            for pending in sorted(
                                pending_to_shift,
                                key=lambda item: item.sort_key,
                            ):
                                shifted = replace(
                                    pending,
                                    time_s=pending.time_s + shift_s,
                                    insertion_sequence=next_sequence,
                                )
                                next_sequence += 1
                                heap.append(shifted)
                                if shifted.kind is EventKind.RESOURCE_CROSSED:
                                    predicted[shifted.resource_id] = shifted.time_s
                            target = replace(
                                target,
                                predicted_resource_crossing_times=tuple(
                                    sorted(predicted.items())
                                ),
                            )
                        else:
                            heap = [
                                pending
                                for pending in heap
                                if not (
                                    pending.flight_id == target_flight_id
                                    and pending.kind
                                    in (
                                        _FLIGHT_FUTURE_EVENT_KINDS
                                        | {EventKind.FLIGHT_RELEASED}
                                    )
                                )
                            ]
                            definition_flight = before.definition.flight(
                                target_flight_id
                            )
                            future, next_sequence, predicted = _flight_events(
                                definition=before.definition,
                                flight=definition_flight,
                                dynamic=target,
                                start_sequence=next_sequence,
                                include_release=True,
                                after_time_s=time_s,
                            )
                            heap.extend(future)
                            target = replace(
                                target, predicted_resource_crossing_times=predicted
                            )
                        by_flight[target_flight_id] = target
                        invalidated_current_flights.add(target_flight_id)
                commitment_controls = payload.get("commitment_controls")
                if commitment_controls is not None:
                    if not isinstance(commitment_controls, Mapping):
                        raise ValueError(
                            "exogenous commitment_controls must be a mapping"
                        )
                    target_flight_id = str(
                        commitment_controls.get(
                            "target_flight_id",
                            payload.get("target_flight_id", event.flight_id),
                        )
                    )
                    if not target_flight_id:
                        raise ValueError("commitment controls require target_flight_id")
                    try:
                        target = by_flight[target_flight_id]
                    except KeyError as exc:
                        raise ValueError(
                            f"commitment controls reference unknown flight {target_flight_id!r}"
                        ) from exc
                    station_cap = float(
                        commitment_controls["action_station_fraction_cap"]
                    )
                    budget_cap = float(
                        commitment_controls["intervention_budget_fraction_cap"]
                    )
                    gate_value = float(commitment_controls["intercept_gate_flag"])
                    if not all(
                        math.isfinite(value)
                        for value in (station_cap, budget_cap, gate_value)
                    ):
                        raise ValueError("commitment control values must be finite")
                    if not 0.0 <= station_cap <= 1.0 or not 0.0 <= budget_cap <= 1.0:
                        raise ValueError("commitment freedom caps must lie in [0, 1]")
                    if gate_value not in (0.0, 1.0):
                        raise ValueError("intercept_gate_flag must be zero or one")
                    by_flight[target_flight_id] = replace(
                        target,
                        action_station_fraction_cap=station_cap,
                        intervention_budget_fraction_cap=budget_cap,
                        intercept_gate_active=bool(gate_value),
                    )
                applied_events.append(event)
                continue
            if event.flight_id in invalidated_current_flights:
                # The exogenous event at this timestamp rescheduled the old
                # physical event before it could be applied.
                continue
            if event.flight_id not in by_flight:
                raise RuntimeError(
                    f"event {event.event_id!r} references unknown flight {event.flight_id!r}"
                )
            dynamic = by_flight[event.flight_id]
            if event.kind is EventKind.FLIGHT_RELEASED:
                if dynamic.lifecycle is not FlightLifecycle.SCHEDULED:
                    raise RuntimeError(
                        f"flight {event.flight_id!r} was released more than once"
                    )
                dynamic = replace(dynamic, lifecycle=FlightLifecycle.ACTIVE)
            elif event.kind is EventKind.ACTION_STATION_CROSSED:
                _require_active(dynamic, event)
                station_type = str(event.payload_dict.get("station_type", "speed"))
                station_key = (station_type, event.station_index)
                keys = dynamic.crossed_action_station_keys
                indices = dynamic.crossed_action_station_indices
                if station_key not in keys:
                    keys = (*keys, station_key)
                if event.station_index not in indices:
                    indices = (*indices, event.station_index)
                dynamic = replace(
                    dynamic,
                    crossed_action_station_keys=keys,
                    crossed_action_station_indices=indices,
                )
            elif event.kind is EventKind.RESOURCE_CROSSED:
                _require_active(dynamic, event)
                crossed = dynamic.crossed_resource_ids
                if event.resource_id not in crossed:
                    crossed = (*crossed, event.resource_id)
                dynamic = replace(dynamic, crossed_resource_ids=crossed)
            elif event.kind is EventKind.FLIGHT_COMPLETED:
                _require_active(dynamic, event)
                dynamic = replace(dynamic, lifecycle=FlightLifecycle.COMPLETED)
            by_flight[event.flight_id] = dynamic
            applied_events.append(event)

        heapq.heapify(heap)

        trigger_ids = tuple(
            event.event_id
            for event in applied_events
            if event.kind.value in before.definition.decision_trigger_kinds
        )
        decision_epoch_index = before.decision_epoch_index + (1 if trigger_ids else 0)
        after = evolve_state(
            before,
            transition=f"event-batch:{time_s:.9f}",
            sim_time_s=time_s,
            event_heap=tuple(heap),
            event_sequence=next_sequence,
            flights=tuple(
                sorted(by_flight.values(), key=lambda flight: flight.flight_id)
            ),
            metrics=tuple(sorted(metrics.items())),
            exogenous_state=freeze_payload(exogenous_state),
            exogenous_event_log=tuple(exogenous_log),
            decision_epoch_index=decision_epoch_index,
        )
        self.state = after
        decision = (
            DecisionEpoch(
                epoch_index=decision_epoch_index,
                time_s=time_s,
                state_version=after.version,
                state_id=after.state_id,
                trigger_event_ids=trigger_ids,
            )
            if trigger_ids
            else None
        )
        return EventBatchResult(
            time_s=time_s,
            events=tuple(applied_events),
            decision_epoch=decision,
            state_id_before=before.state_id,
            state_id_after=after.state_id,
        )

    def advance_until(
        self,
        horizon_s: float,
        *,
        policy: Any | None = None,
    ) -> tuple[EventBatchResult, ...]:
        horizon = float(horizon_s)
        if not math.isfinite(horizon):
            raise ValueError("horizon_s must be finite")
        if horizon < self.state.sim_time_s - 1e-9:
            raise ValueError("cannot run backward in simulation time")

        batches: list[EventBatchResult] = []
        while self.state.event_heap and self.state.event_heap[0].time_s <= horizon:
            batch = self.advance_next()
            if batch is not None:
                batches.append(batch)
                if policy is not None and batch.decision_epoch is not None:
                    selector = getattr(policy, "select_action", None)
                    if not callable(selector):
                        raise TypeError(
                            "rollout policy must implement select_action(context)"
                        )
                    action = selector(
                        PolicyDecisionContext(
                            simulator=self,
                            state=self.state,
                            decision_epoch=batch.decision_epoch,
                            event_batch=batch,
                        )
                    )
                    if action is not None:
                        self.apply(action)
        if horizon > self.state.sim_time_s:
            self.state = evolve_state(
                self.state,
                transition=f"advance-time:{horizon:.9f}",
                sim_time_s=horizon,
            )
        return tuple(batches)

    def run_until(self, horizon_s: float, *, policy: Any | None = None) -> "Simulator":
        """Advance to ``horizon_s`` and return this branch for rollout chaining."""

        self.last_run_batches = self.advance_until(horizon_s, policy=policy)
        return self

    def run(self) -> tuple[EventBatchResult, ...]:
        batches: list[EventBatchResult] = []
        while self.state.event_heap:
            batch = self.advance_next()
            if batch is not None:
                batches.append(batch)
        return tuple(batches)

    def apply(self, action: Any) -> "Simulator":
        """Apply a branch-local action through the configured realization layer.

        With no realization layer, only the canonical ``no_op/no_op`` action
        is an identity. Otherwise every action is delegated so eligibility and
        audit logging stay consistent. Audit objects are retained as
        ``last_action_result`` while returned state/simulator objects are
        adopted.
        """

        if isinstance(action, Mapping):
            lever = action.get("lever")
            band = action.get("band")
        else:
            lever = getattr(action, "lever", None)
            band = getattr(action, "band", None)
        lever_value = getattr(lever, "value", lever)
        band_value = getattr(band, "value", band)
        if self._action_applier is None:
            if lever_value == "no_op" and band_value == "no_op":
                return self
            raise SimulationError(
                "no action realization layer is configured for this simulator"
            )
        result = self._action_applier(self, action)
        self.last_action_result = result
        if isinstance(result, SimulationState):
            self.state = result
        elif isinstance(result, Simulator) and result is not self:
            self.state = result.state
        return self

    apply_action = apply

    def install_variant(
        self,
        variant: object,
        *,
        expected_version: int | None = None,
        expected_state_id: str | None = None,
    ) -> SimulationState:
        """Install an immutable compiled variant in this branch's definition.

        Existing arrays remain shared. A fork receives a new frozen definition
        tuple, so installing a cache-miss realization can never mutate its
        parent or sibling.
        """

        self.assert_fresh(
            expected_version=expected_version, expected_state_id=expected_state_id
        )
        variant_id = trajectory_variant_id(variant)
        if variant_id in self.definition.variant_ids:
            return self.state
        definition = replace(
            self.definition, variants=(*self.definition.variants, variant)
        )
        self.state = evolve_state(
            self.state,
            transition=f"install-variant:{variant_id}",
            definition=definition,
        )
        return self.state

    def replace_flight_variant(
        self,
        flight_id: str,
        variant_id: str,
        *,
        action_id: str = "",
        action_lever: str = "",
        splice_s_m: float | None = None,
        station_mapping_m: tuple[tuple[float, float], ...] | None = None,
        expected_version: int | None = None,
        expected_state_id: str | None = None,
    ) -> SimulationState:
        """Splice an active flight onto a new variant without rewriting history.

        ``station_mapping_m`` maps stations on the current variant to their
        physical-progress counterparts on the replacement.  It is identity
        for a speed-only profile and a monotone arc-length mapping for a
        dogleg.  Pending station/resource events are transformed from their
        live payloads rather than regenerated from baseline definitions.
        """

        self.assert_fresh(
            expected_version=expected_version, expected_state_id=expected_state_id
        )
        before = self.state
        dynamic = before.flight(flight_id)
        if dynamic.lifecycle is not FlightLifecycle.ACTIVE:
            raise ValueError(
                "a trajectory variant can only be replaced for an active flight"
            )
        current_variant = self.definition.variant(dynamic.current_variant_id)
        replacement_variant = self.definition.variant(variant_id)
        current_trajectory = MonotoneTrajectory.from_variant(current_variant)
        replacement_trajectory = MonotoneTrajectory.from_variant(replacement_variant)

        current_elapsed_s = before.sim_time_s - dynamic.trajectory_clock_origin_s
        if not -1e-8 <= current_elapsed_s <= current_trajectory.duration_s + 1e-8:
            raise ValueError("active flight time lies outside its current trajectory")
        current_station_s_m = current_trajectory.station_at_elapsed(current_elapsed_s)
        requested_splice_s_m = (
            current_station_s_m if splice_s_m is None else float(splice_s_m)
        )
        station_tolerance_m = max(1e-5, current_trajectory.upstream_s_m * 1e-10)
        if abs(requested_splice_s_m - current_station_s_m) > station_tolerance_m:
            raise ValueError(
                "variant replacement splice does not match the aircraft's current station"
            )
        mapped_splice_s_m = _mapped_station_s_m(
            requested_splice_s_m,
            station_mapping_m,
            current_trajectory=current_trajectory,
            replacement_trajectory=replacement_trajectory,
        )
        replacement_elapsed_s = replacement_trajectory.elapsed_at_station(
            mapped_splice_s_m
        )
        replacement_origin_s = before.sim_time_s - replacement_elapsed_s
        _assert_splice_position_continuity(
            current_trajectory.sample(current_elapsed_s),
            replacement_trajectory.sample(replacement_elapsed_s),
        )

        pending_for_flight = sorted(
            (
                event
                for event in before.event_heap
                if event.flight_id == flight_id
                and event.kind in _FLIGHT_FUTURE_EVENT_KINDS
            ),
            key=lambda event: event.sort_key,
        )
        remaining = [
            event
            for event in before.event_heap
            if not (
                event.flight_id == flight_id
                and event.kind in _FLIGHT_FUTURE_EVENT_KINDS
            )
        ]
        sequence = before.event_sequence
        predicted = dict(dynamic.predicted_resource_crossing_times)
        future_events: list[ScheduledEvent] = []
        for event in pending_for_flight:
            payload = event.payload_dict
            if event.kind in {
                EventKind.ACTION_STATION_CROSSED,
                EventKind.RESOURCE_CROSSED,
            }:
                if "s_m" not in payload:
                    raise ValueError(
                        f"pending event {event.event_id!r} has no physical station"
                    )
                mapped_station = _mapped_station_s_m(
                    float(payload["s_m"]),
                    station_mapping_m,
                    current_trajectory=current_trajectory,
                    replacement_trajectory=replacement_trajectory,
                )
                event_time_s = (
                    replacement_origin_s
                    + replacement_trajectory.elapsed_at_station(mapped_station)
                )
                payload["s_m"] = mapped_station
                if event.kind is EventKind.RESOURCE_CROSSED:
                    predicted[event.resource_id] = event_time_s
            else:
                event_time_s = replacement_origin_s + replacement_trajectory.duration_s
            if event_time_s <= before.sim_time_s + 1e-9:
                raise ValueError(
                    f"replacement maps pending event {event.event_id!r} to the present or past"
                )
            future_events.append(
                replace(
                    event,
                    time_s=event_time_s,
                    insertion_sequence=sequence,
                    payload=payload,
                )
            )
            sequence += 1

        updated_dynamic = replace(
            dynamic,
            current_variant_id=variant_id,
            trajectory_origin_time_s=replacement_origin_s,
            action_history=(
                dynamic.action_history
                if not action_id
                else (*dynamic.action_history, action_id)
            ),
            speed_action_count=dynamic.speed_action_count
            + (1 if action_lever == "speed" else 0),
            path_stretch_count=dynamic.path_stretch_count
            + (1 if action_lever == "path_stretch" else 0),
            predicted_resource_crossing_times=tuple(sorted(predicted.items())),
        )
        remaining.extend(future_events)
        heapq.heapify(remaining)
        flights = tuple(
            updated_dynamic if item.flight_id == flight_id else item
            for item in before.flights
        )
        action_log = before.action_log
        if action_id:
            action_log = (
                *action_log,
                freeze_payload(
                    {
                        "action_id": action_id,
                        "flight_id": flight_id,
                        "from_variant_id": dynamic.current_variant_id,
                        "to_variant_id": variant_id,
                        "lever": action_lever,
                        "time_s": before.sim_time_s,
                        "from_trajectory_origin_time_s": dynamic.trajectory_clock_origin_s,
                        "to_trajectory_origin_time_s": replacement_origin_s,
                        "from_splice_s_m": requested_splice_s_m,
                        "to_splice_s_m": mapped_splice_s_m,
                    }
                ),
            )
        self.state = evolve_state(
            before,
            transition=f"replace-variant:{flight_id}:{variant_id}",
            flights=flights,
            event_heap=tuple(remaining),
            event_sequence=sequence,
            action_log=action_log,
        )
        return self.state

    def schedule_event(
        self,
        *,
        time_s: float,
        kind: EventKind | str,
        event_id: str,
        flight_id: str = "",
        station_index: int = -1,
        resource_id: str = "",
        payload: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        expected_state_id: str | None = None,
    ) -> ScheduledEvent:
        self.assert_fresh(
            expected_version=expected_version, expected_state_id=expected_state_id
        )
        event_time = float(time_s)
        if event_time < self.state.sim_time_s - 1e-9:
            raise ValueError("cannot schedule an event in the past")
        if any(event.event_id == event_id for event in self.state.event_heap):
            raise ValueError(f"pending event_id {event_id!r} already exists")
        event = ScheduledEvent(
            time_s=event_time,
            kind=EventKind(kind),
            event_id=event_id,
            insertion_sequence=self.state.event_sequence,
            flight_id=flight_id,
            station_index=station_index,
            resource_id=resource_id,
            payload={} if payload is None else payload,
        )
        heap = list(self.state.event_heap)
        heapq.heappush(heap, event)
        self.state = evolve_state(
            self.state,
            transition=f"schedule-event:{event_id}",
            event_heap=tuple(heap),
            event_sequence=self.state.event_sequence + 1,
        )
        return event

    def add_metric(
        self,
        name: str,
        delta: float,
        *,
        expected_version: int | None = None,
    ) -> float:
        self.assert_fresh(expected_version=expected_version)
        if not name:
            raise ValueError("metric name must be non-empty")
        increment = float(delta)
        if not math.isfinite(increment):
            raise ValueError("metric delta must be finite")
        metrics = self.state.metrics_dict
        metrics[name] = metrics.get(name, 0.0) + increment
        self.state = evolve_state(
            self.state,
            transition=f"metric:{name}",
            metrics=tuple(sorted(metrics.items())),
        )
        return metrics[name]

    def record_action(
        self,
        record: Mapping[str, Any],
        *,
        expected_version: int | None = None,
    ) -> SimulationState:
        self.assert_fresh(expected_version=expected_version)
        self.state = evolve_state(
            self.state,
            transition="record-action",
            action_log=(*self.state.action_log, freeze_payload(record)),
        )
        return self.state

    def random_uniform(
        self,
        *,
        low: float = 0.0,
        high: float = 1.0,
        expected_version: int | None = None,
    ) -> float:
        self.assert_fresh(expected_version=expected_version)
        generator = rng_from_state_json(self.state.rng_state_json)
        value = float(generator.uniform(float(low), float(high)))
        self.state = evolve_state(
            self.state,
            transition="rng-uniform",
            rng_state_json=rng_state_json(generator),
        )
        return value


def initial_state(definition: ScenarioDefinition | None) -> SimulationState:
    if definition is None:
        raise ValueError("definition is required")
    events: list[ScheduledEvent] = []
    dynamics: list[FlightDynamic] = []
    sequence = 0

    for disturbance in sorted(
        definition.exogenous_events, key=lambda item: (item.time_s, item.event_id)
    ):
        events.append(
            ScheduledEvent(
                time_s=disturbance.time_s,
                kind=EventKind.EXOGENOUS_DISTURBANCE,
                event_id=disturbance.event_id,
                insertion_sequence=sequence,
                payload={
                    **disturbance.payload_dict,
                    "stream_name": disturbance.stream_name,
                },
            )
        )
        sequence += 1

    for flight in sorted(definition.flights, key=lambda item: item.flight_id):
        dynamic = FlightDynamic(
            flight_id=flight.flight_id,
            lifecycle=FlightLifecycle.SCHEDULED,
            current_variant_id=flight.baseline_variant_id,
            release_time_s=flight.release_time_s,
            trajectory_origin_time_s=flight.release_time_s,
        )
        flight_events, sequence, predicted = _flight_events(
            definition=definition,
            flight=flight,
            dynamic=dynamic,
            start_sequence=sequence,
            include_release=True,
            after_time_s=-math.inf,
        )
        events.extend(flight_events)
        dynamics.append(replace(dynamic, predicted_resource_crossing_times=predicted))

    heapq.heapify(events)
    return make_initial_state(
        definition,
        flights=tuple(sorted(dynamics, key=lambda item: item.flight_id)),
        event_heap=tuple(events),
        event_sequence=sequence,
    )


def _flight_events(
    *,
    definition: ScenarioDefinition,
    flight: FlightDefinition,
    dynamic: FlightDynamic,
    start_sequence: int,
    include_release: bool,
    after_time_s: float,
) -> tuple[list[ScheduledEvent], int, tuple[tuple[str, float], ...]]:
    variant = definition.variant(dynamic.current_variant_id)
    trajectory = MonotoneTrajectory.from_variant(variant)
    events: list[ScheduledEvent] = []
    sequence = int(start_sequence)

    def append(event: ScheduledEvent) -> None:
        nonlocal sequence
        events.append(event)
        sequence += 1

    if include_release and dynamic.release_time_s >= after_time_s - 1e-9:
        append(
            ScheduledEvent(
                time_s=dynamic.release_time_s,
                kind=EventKind.FLIGHT_RELEASED,
                event_id=f"{definition.scenario_id}:{flight.flight_id}:release",
                insertion_sequence=sequence,
                flight_id=flight.flight_id,
            )
        )

    crossed_station_keys = set(dynamic.crossed_action_station_keys)
    for station in sorted(
        flight.action_stations, key=lambda item: (item.station_type, item.station_index)
    ):
        key = (station.station_type, station.station_index)
        if key in crossed_station_keys:
            continue
        crossing_time_s = (
            dynamic.trajectory_clock_origin_s
            + trajectory.elapsed_at_station(station.s_m)
        )
        if crossing_time_s <= after_time_s + 1e-9:
            continue
        append(
            ScheduledEvent(
                time_s=crossing_time_s,
                kind=EventKind.ACTION_STATION_CROSSED,
                event_id=(
                    f"{definition.scenario_id}:{flight.flight_id}:"
                    f"action-station:{station.station_type}:{station.station_index}"
                ),
                insertion_sequence=sequence,
                flight_id=flight.flight_id,
                station_index=station.station_index,
                payload={"s_m": station.s_m, "station_type": station.station_type},
            )
        )

    predicted: list[tuple[str, float]] = []
    crossed_resources = set(dynamic.crossed_resource_ids)
    crossings = _resource_crossings(flight, variant)
    for crossing in sorted(
        crossings,
        key=lambda item: (item.resource_id, item.station_index),
    ):
        definition.resource(crossing.resource_id)  # validate resource ownership
        crossing_time_s = (
            dynamic.trajectory_clock_origin_s
            + trajectory.elapsed_at_station(crossing.s_m)
        )
        predicted.append((crossing.resource_id, crossing_time_s))
        if (
            crossing.resource_id in crossed_resources
            or crossing_time_s <= after_time_s + 1e-9
        ):
            continue
        append(
            ScheduledEvent(
                time_s=crossing_time_s,
                kind=EventKind.RESOURCE_CROSSED,
                event_id=(
                    f"{definition.scenario_id}:{flight.flight_id}:"
                    f"resource:{crossing.resource_id}:{crossing.station_index}"
                ),
                insertion_sequence=sequence,
                flight_id=flight.flight_id,
                station_index=crossing.station_index,
                resource_id=crossing.resource_id,
                payload={"s_m": crossing.s_m},
            )
        )

    completion_time_s = dynamic.trajectory_clock_origin_s + trajectory.duration_s
    if completion_time_s > after_time_s + 1e-9:
        append(
            ScheduledEvent(
                time_s=completion_time_s,
                kind=EventKind.FLIGHT_COMPLETED,
                event_id=f"{definition.scenario_id}:{flight.flight_id}:completed",
                insertion_sequence=sequence,
                flight_id=flight.flight_id,
            )
        )
    elif dynamic.lifecycle is not FlightLifecycle.COMPLETED:
        raise ValueError(
            f"variant {dynamic.current_variant_id!r} completes before current time for flight {flight.flight_id!r}"
        )
    return events, sequence, tuple(sorted(predicted))


def _mapped_station_s_m(
    station_s_m: float,
    station_mapping_m: tuple[tuple[float, float], ...] | None,
    *,
    current_trajectory: MonotoneTrajectory,
    replacement_trajectory: MonotoneTrajectory,
) -> float:
    station = float(station_s_m)
    if not math.isfinite(station):
        raise ValueError("station mapping input must be finite")
    if station_mapping_m is None:
        mapped = station
    else:
        mapping = np.asarray(station_mapping_m, dtype=np.float64)
        if mapping.ndim != 2 or mapping.shape[1:] != (2,) or len(mapping) < 2:
            raise ValueError(
                "station_mapping_m must contain at least two (parent, child) pairs"
            )
        if not np.all(np.isfinite(mapping)):
            raise ValueError("station_mapping_m must be finite")
        parent_s = mapping[:, 0]
        child_s = mapping[:, 1]
        if np.any(np.diff(parent_s) <= 0.0) or np.any(np.diff(child_s) <= 0.0):
            raise ValueError(
                "station_mapping_m must be strictly monotone in both coordinates"
            )
        if station < parent_s[0] - 1e-7 or station > parent_s[-1] + 1e-7:
            raise ValueError(
                "station mapping does not cover a pending physical station"
            )
        mapped = float(np.interp(station, parent_s, child_s))
    if (
        mapped < replacement_trajectory.downstream_s_m - 1e-7
        or mapped > replacement_trajectory.upstream_s_m + 1e-7
    ):
        raise ValueError("mapped station lies outside the replacement trajectory")
    if (
        station < current_trajectory.downstream_s_m - 1e-7
        or station > current_trajectory.upstream_s_m + 1e-7
    ):
        raise ValueError("source station lies outside the current trajectory")
    return mapped


def _assert_splice_position_continuity(
    current: TrajectorySample,
    replacement: TrajectorySample,
) -> None:
    tolerances = {
        "east_m": 1e-4,
        "north_m": 1e-4,
        "altitude_m": 1e-4,
        "cas_mps": 1e-7,
    }
    discontinuities: list[str] = []
    for name, tolerance in tolerances.items():
        before_value = getattr(current, name)
        after_value = getattr(replacement, name)
        if before_value is None or after_value is None:
            continue
        if abs(float(before_value) - float(after_value)) > tolerance:
            discontinuities.append(name)
    if discontinuities:
        raise ValueError(
            "variant replacement is discontinuous at the live splice for "
            + ", ".join(discontinuities)
        )


def _require_active(dynamic: FlightDynamic, event: ScheduledEvent) -> None:
    if dynamic.lifecycle is not FlightLifecycle.ACTIVE:
        raise RuntimeError(
            f"event {event.event_id!r} requires active flight {dynamic.flight_id!r}; "
            f"lifecycle is {dynamic.lifecycle.value!r}"
        )


def _resource_crossings(flight: FlightDefinition, variant: object) -> tuple[Any, ...]:
    if flight.resource_crossings:
        return flight.resource_crossings
    raw_crossings = getattr(variant, "resource_crossings", ())
    normalized: list[Any] = []
    for station_index, crossing in enumerate(raw_crossings):
        resource_id = str(getattr(crossing, "resource_id", ""))
        s_m = float(getattr(crossing, "s_m"))
        if not resource_id:
            raise ValueError(
                "variant resource crossing must have a non-empty resource_id"
            )
        normalized.append(
            _VariantResourceCrossing(
                resource_id=resource_id,
                s_m=s_m,
                station_index=station_index,
            )
        )
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class _VariantResourceCrossing:
    resource_id: str
    s_m: float
    station_index: int


__all__ = ["PolicyDecisionContext", "Simulator", "initial_state"]
