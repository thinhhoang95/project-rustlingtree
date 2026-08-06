"""Typed role bindings and deterministic resource ordering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from hailmary.ids import stable_id


def _anchor_id(kind: str, payload: Mapping[str, Any]) -> str:
    return stable_id(f"anchor_{kind}", payload, length=28)


@dataclass(frozen=True)
class LeaderFollowerAnchor:
    resource_id: str
    leader_id: str
    follower_id: str
    state_version: str
    epoch: int
    segment_id: str = ""
    entry_resource_id: str = ""
    ordering_basis: str = "eta"
    predicted_exit_interval_s: float | None = None
    catch_up: bool = False
    anchor_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.resource_id or not self.leader_id or not self.follower_id:
            raise ValueError("anchor resource and flight IDs cannot be empty")
        if self.leader_id == self.follower_id:
            raise ValueError("leader and follower must be distinct")
        if self.epoch < 0:
            raise ValueError("epoch cannot be negative")
        if self.segment_id and not self.entry_resource_id:
            raise ValueError("segment anchors require an entry resource")
        if self.ordering_basis not in {"eta", "physical_progress", "entry_eta", "mixed"}:
            raise ValueError("unknown leader/follower ordering basis")
        if self.predicted_exit_interval_s is not None and not np.isfinite(
            self.predicted_exit_interval_s
        ):
            raise ValueError("predicted exit interval must be finite when supplied")
        object.__setattr__(
            self,
            "anchor_id",
            _anchor_id(
                "leader_follower",
                {
                    "resource_id": self.resource_id,
                    "leader_id": self.leader_id,
                    "follower_id": self.follower_id,
                    "state_version": self.state_version,
                    "epoch": self.epoch,
                    "segment_id": self.segment_id,
                    "entry_resource_id": self.entry_resource_id,
                    "ordering_basis": self.ordering_basis,
                },
            ),
        )


@dataclass(frozen=True)
class AircraftResourceAnchor:
    resource_id: str
    aircraft_id: str
    state_version: str
    epoch: int
    anchor_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.resource_id or not self.aircraft_id:
            raise ValueError("anchor resource and aircraft IDs cannot be empty")
        if self.epoch < 0:
            raise ValueError("epoch cannot be negative")
        object.__setattr__(
            self,
            "anchor_id",
            _anchor_id(
                "aircraft_resource",
                {
                    "resource_id": self.resource_id,
                    "aircraft_id": self.aircraft_id,
                    "state_version": self.state_version,
                    "epoch": self.epoch,
                },
            ),
        )


@dataclass(frozen=True)
class FlowAnchor:
    resource_id: str
    ordered_flight_ids: tuple[str, ...]
    state_version: str
    epoch: int
    segment_id: str = ""
    anchor_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id cannot be empty")
        if len(set(self.ordered_flight_ids)) != len(self.ordered_flight_ids):
            raise ValueError("a flow anchor cannot contain duplicate flights")
        if self.epoch < 0:
            raise ValueError("epoch cannot be negative")
        object.__setattr__(
            self,
            "anchor_id",
            _anchor_id(
                "flow",
                {
                    "resource_id": self.resource_id,
                    "ordered_flight_ids": self.ordered_flight_ids,
                    "state_version": self.state_version,
                    "epoch": self.epoch,
                    "segment_id": self.segment_id,
                },
            ),
        )


@dataclass(frozen=True)
class ResourceAnchors:
    flow: FlowAnchor
    leader_follower: tuple[LeaderFollowerAnchor, ...]
    aircraft_resource: tuple[AircraftResourceAnchor, ...]


@dataclass(frozen=True)
class SegmentAnchorSet:
    """All canonical segment queues for one real simulator epoch."""

    flows: tuple[FlowAnchor, ...]
    leader_follower: tuple[LeaderFollowerAnchor, ...]
    aircraft_resource: tuple[AircraftResourceAnchor, ...]

    def flow_for_segment(self, segment_id: str) -> FlowAnchor:
        for flow in self.flows:
            if flow.segment_id == segment_id:
                return flow
        raise KeyError(segment_id)


@dataclass(frozen=True)
class SimulatorFlightPrediction:
    """Canonical current sample and resource ETA for one active flight."""

    flight_id: str
    resource_id: str
    variant_id: str
    eta_s: float
    resource_s_m: float
    sample: Any

    def __post_init__(self) -> None:
        if not self.flight_id or not self.resource_id or not self.variant_id:
            raise ValueError("prediction identities cannot be empty")
        if not np.isfinite(self.eta_s) or not np.isfinite(self.resource_s_m):
            raise ValueError("prediction ETA and resource station must be finite")


def order_flights_by_eta(eta_by_flight: Mapping[str, float]) -> tuple[str, ...]:
    """Sort by current no-further-action ETA, then stable textual flight ID."""

    normalized: list[tuple[float, str]] = []
    for flight_id, eta_s in eta_by_flight.items():
        eta = float(eta_s)
        if not np.isfinite(eta):
            raise ValueError(f"non-finite ETA for {flight_id!r}")
        normalized.append((eta, str(flight_id)))
    return tuple(flight_id for _eta, flight_id in sorted(normalized, key=lambda item: (item[0], item[1])))


def build_resource_anchors(
    *,
    resource_id: str,
    eta_by_flight: Mapping[str, float],
    state_version: str,
    epoch: int,
) -> ResourceAnchors:
    """Build the complete adjacent-edge resource graph for one state."""

    ordered = order_flights_by_eta(eta_by_flight)
    flow = FlowAnchor(resource_id, ordered, state_version, epoch)
    edges = tuple(
        LeaderFollowerAnchor(resource_id, leader, follower, state_version, epoch)
        for leader, follower in zip(ordered, ordered[1:], strict=False)
    )
    aircraft = tuple(
        AircraftResourceAnchor(resource_id, flight_id, state_version, epoch)
        for flight_id in ordered
    )
    return ResourceAnchors(flow=flow, leader_follower=edges, aircraft_resource=aircraft)


@runtime_checkable
class ResourceETAQuery(Protocol):
    """Minimal engine/query boundary needed to construct anchors."""

    def live_flight_ids(self, state: Any, resource_id: str) -> Sequence[str]: ...

    def nominal_eta_s(self, state: Any, flight_id: str, resource_id: str) -> float: ...

    def state_version(self, state: Any) -> str: ...

    def decision_epoch(self, state: Any) -> int: ...


def build_resource_anchors_from_query(
    state: Any,
    *,
    resource_id: str,
    query: ResourceETAQuery,
) -> ResourceAnchors:
    flight_ids = tuple(str(item) for item in query.live_flight_ids(state, resource_id))
    eta_by_flight = {
        flight_id: float(query.nominal_eta_s(state, flight_id, resource_id))
        for flight_id in flight_ids
    }
    return build_resource_anchors(
        resource_id=resource_id,
        eta_by_flight=eta_by_flight,
        state_version=str(query.state_version(state)),
        epoch=int(query.decision_epoch(state)),
    )


def resource_station_m(simulator: Any, flight_id: str, resource_id: str) -> float:
    """Use the same flight-first resource-station precedence as the engine."""

    definition = simulator.state.definition
    flight = definition.flight(str(flight_id))
    # A live replacement event carries the composed physical mapping after a
    # dogleg.  Baseline definitions deliberately remain immutable, so pending
    # event payloads are the authoritative station for an active flight.
    pending = sorted(simulator.state.event_heap, key=lambda event: event.sort_key)
    for event in pending:
        if (
            event.flight_id == str(flight_id)
            and str(getattr(event.kind, "value", event.kind)) == "RESOURCE_CROSSED"
            and event.resource_id == resource_id
        ):
            return float(event.payload_dict["s_m"])
    dynamic = simulator.state.flight(str(flight_id))
    variant = definition.variant(dynamic.current_variant_id)
    for crossing in getattr(variant, "resource_crossings", ()):
        if str(getattr(crossing, "resource_id", "")) == resource_id:
            return float(getattr(crossing, "s_m"))
    for crossing in flight.resource_crossings:
        if crossing.resource_id == resource_id:
            return float(crossing.s_m)
    raise KeyError(f"flight {flight_id!r} has no crossing for resource {resource_id!r}")


def resource_eta_s(simulator: Any, flight_id: str, resource_id: str) -> float:
    """Derive an absolute ETA from the active canonical variant, not cached JSON."""

    from hailmary.simulator.interpolation import MonotoneTrajectory

    state = simulator.state
    dynamic = state.flight(str(flight_id))
    variant = state.definition.variant(dynamic.current_variant_id)
    trajectory = MonotoneTrajectory.from_variant(variant)
    station = resource_station_m(simulator, str(flight_id), str(resource_id))
    return float(dynamic.trajectory_clock_origin_s + trajectory.elapsed_at_station(station))


def active_resource_predictions(
    simulator: Any,
    *,
    resource_id: str,
) -> tuple[SimulatorFlightPrediction, ...]:
    """Return live, not-yet-crossed aircraft samples and canonical ETAs."""

    from hailmary.simulator.interpolation import MonotoneTrajectory

    state = simulator.state
    predictions: list[SimulatorFlightPrediction] = []
    for dynamic in state.flights:
        lifecycle = str(getattr(dynamic.lifecycle, "value", dynamic.lifecycle))
        if lifecycle != "active" or resource_id in dynamic.crossed_resource_ids:
            continue
        variant = state.definition.variant(dynamic.current_variant_id)
        trajectory = MonotoneTrajectory.from_variant(variant)
        elapsed = float(state.sim_time_s - dynamic.trajectory_clock_origin_s)
        try:
            station = resource_station_m(simulator, dynamic.flight_id, resource_id)
        except KeyError:
            # Multi-runway snapshots contain many active flights that do not
            # traverse this explicitly scoped resource.
            continue
        predictions.append(
            SimulatorFlightPrediction(
                flight_id=dynamic.flight_id,
                resource_id=resource_id,
                variant_id=dynamic.current_variant_id,
                eta_s=float(
                    dynamic.trajectory_clock_origin_s
                    + trajectory.elapsed_at_station(station)
                ),
                resource_s_m=station,
                sample=trajectory.sample(elapsed),
            )
        )
    return tuple(sorted(predictions, key=lambda item: (item.eta_s, item.flight_id)))


def build_current_segment_anchors(simulator: Any) -> SegmentAnchorSet:
    """Build queues from physical occupancy and committed entry-gate ETAs.

    Occupants always precede future entrants.  Their established order comes
    from current trajectory progress, while future entrants use entry ETA with
    a stable flight-ID tie break.  Exit ETA is used only to evaluate spacing;
    it never reverses an established physical order.
    """

    from hailmary.simulator.interpolation import MonotoneTrajectory

    state = simulator.state
    definition = state.definition
    traversals_by_segment: dict[str, list[tuple[Any, Any]]] = {}
    for flight in definition.flights:
        dynamic = state.flight(flight.flight_id)
        if str(getattr(dynamic.lifecycle, "value", dynamic.lifecycle)) != "active":
            continue
        for traversal in flight.segment_traversals:
            if traversal.exit_resource_id in dynamic.crossed_resource_ids:
                continue
            traversals_by_segment.setdefault(traversal.segment_id, []).append(
                (flight, traversal)
            )

    flows: list[FlowAnchor] = []
    aircraft: list[AircraftResourceAnchor] = []
    candidates: list[tuple[tuple[str, str], int, LeaderFollowerAnchor]] = []
    for segment_id, records in sorted(traversals_by_segment.items()):
        occupants: list[tuple[float, str, Any]] = []
        future: list[tuple[float, str, Any]] = []
        exit_eta: dict[str, float] = {}
        basis_by_flight: dict[str, str] = {}
        for flight, traversal in records:
            dynamic = state.flight(flight.flight_id)
            variant = definition.variant(dynamic.current_variant_id)
            trajectory = MonotoneTrajectory.from_variant(variant)
            sample = trajectory.sample(
                float(state.sim_time_s - dynamic.trajectory_clock_origin_s)
            )
            entry_station = resource_station_m(
                simulator, flight.flight_id, traversal.entry_resource_id
            )
            exit_station = resource_station_m(
                simulator, flight.flight_id, traversal.exit_resource_id
            )
            entry_eta = float(
                dynamic.trajectory_clock_origin_s
                + trajectory.elapsed_at_station(entry_station)
            )
            exit_eta[flight.flight_id] = float(
                dynamic.trajectory_clock_origin_s
                + trajectory.elapsed_at_station(exit_station)
            )
            is_occupant = (
                traversal.entry_resource_id in dynamic.crossed_resource_ids
                or float(sample.s_m) <= entry_station + 1.0e-6
            )
            if is_occupant:
                denominator = max(entry_station - exit_station, 1.0e-9)
                progress = (entry_station - float(sample.s_m)) / denominator
                occupants.append((-progress, flight.flight_id, traversal))
                basis_by_flight[flight.flight_id] = "physical_progress"
            else:
                future.append((entry_eta, flight.flight_id, traversal))
                basis_by_flight[flight.flight_id] = "entry_eta"
        occupants.sort(key=lambda item: (item[0], item[1]))
        future.sort(key=lambda item: (item[0], item[1]))
        ordered_records = (*occupants, *future)
        ordered = tuple(item[1] for item in ordered_records)
        if not ordered:
            continue
        exit_resource_id = records[0][1].exit_resource_id
        entry_resource_id = records[0][1].entry_resource_id
        flow = FlowAnchor(
            exit_resource_id,
            ordered,
            str(state.dynamic_content_hash),
            int(state.decision_epoch_index),
            segment_id=segment_id,
        )
        flows.append(flow)
        aircraft.extend(
            AircraftResourceAnchor(
                exit_resource_id,
                flight_id,
                str(state.dynamic_content_hash),
                int(state.decision_epoch_index),
            )
            for flight_id in ordered
        )
        traversal_by_flight = {item[1]: item[2] for item in ordered_records}
        for leader_id, follower_id in zip(ordered, ordered[1:], strict=False):
            leader_basis = basis_by_flight[leader_id]
            follower_basis = basis_by_flight[follower_id]
            basis = leader_basis if leader_basis == follower_basis else "mixed"
            interval = float(exit_eta[follower_id] - exit_eta[leader_id])
            anchor = LeaderFollowerAnchor(
                resource_id=exit_resource_id,
                leader_id=leader_id,
                follower_id=follower_id,
                state_version=str(state.dynamic_content_hash),
                epoch=int(state.decision_epoch_index),
                segment_id=segment_id,
                entry_resource_id=entry_resource_id,
                ordering_basis=basis,
                predicted_exit_interval_s=interval,
                catch_up=bool(interval < 0.0),
            )
            first_ordinal = max(
                traversal_by_flight[leader_id].ordinal,
                traversal_by_flight[follower_id].ordinal,
            )
            candidates.append((tuple(sorted((leader_id, follower_id))), first_ordinal, anchor))

    # One pair is bound to the earliest unpassed segment on its common suffix.
    selected: dict[tuple[str, str], tuple[int, LeaderFollowerAnchor]] = {}
    for pair, ordinal, anchor in candidates:
        current = selected.get(pair)
        if current is None or (ordinal, anchor.segment_id) < (
            current[0],
            current[1].segment_id,
        ):
            selected[pair] = (ordinal, anchor)
    return SegmentAnchorSet(
        flows=tuple(sorted(flows, key=lambda item: item.resource_id)),
        leader_follower=tuple(
            sorted((item[1] for item in selected.values()), key=lambda item: item.anchor_id)
        ),
        aircraft_resource=tuple(sorted(aircraft, key=lambda item: item.anchor_id)),
    )


__all__ = [
    "AircraftResourceAnchor",
    "FlowAnchor",
    "LeaderFollowerAnchor",
    "ResourceAnchors",
    "ResourceETAQuery",
    "SimulatorFlightPrediction",
    "SegmentAnchorSet",
    "active_resource_predictions",
    "build_resource_anchors",
    "build_resource_anchors_from_query",
    "build_current_segment_anchors",
    "order_flights_by_eta",
    "resource_eta_s",
    "resource_station_m",
]
