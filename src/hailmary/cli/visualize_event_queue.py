"""Browser-based verifier for the deterministic Hailmary event queue."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from importlib import import_module
import json
from pathlib import Path
import sys
import threading
from threading import RLock
from typing import Any
import webbrowser
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from hailmary.actions import ActionCandidate, ActionCatalog, ActionLever
from hailmary.config import (
    FeatureConfig,
    M_PER_NM,
    OutcomeConfig,
    ScenarioConfig,
    TemplateConfig,
)
from hailmary.evaluation import simulator_outcome_plan
from hailmary.errors import HailmaryError
from hailmary.features import (
    build_current_segment_anchors,
    resource_station_m,
    simulator_state_vector,
)
from hailmary.features.anchors import LeaderFollowerAnchor
from hailmary.ids import content_hash
from hailmary.rollout import NoOpPolicy, paired_simulator_rollout
from hailmary.rollout.paired import PairedArmTrace
from hailmary.rollout.policy import policy_fingerprint
from hailmary.runtime import ActionRuntime, build_action_runtime
from hailmary.scenario import (
    DemandWindowConfig,
    ScenarioDefinition,
    TerminalEntryCorpus,
    TrafficScaleConfig,
    TrafficScenarioBuilder,
    iter_demand_windows,
)
from hailmary.simulator import EventBatchResult, ScheduledEvent, Simulator
from hailmary.simulator.state import SimulationState
from hailmary.templates import TemplateStore
from hailmary.topology import (
    MedoidRoute,
    RouteGraphArtifact,
    RouteGraphConfig,
    build_route_graph,
    partition_medoid_routes,
)

from .event_queue_web import EVENT_QUEUE_HTML
from .visualize_demand_scaling import (
    DemandScalingInspector,
    ScalingSample,
    _default_bounds,
)


def _timezone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise argparse.ArgumentTypeError(f"unknown IANA timezone: {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hailmary-visualize-event-queue",
        description=(
            "Open a local verifier GUI for a scaled demand window and replay the "
            "real Hailmary event queue one event batch at a time."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/artifacts/hailmary/corpus/traffic_corpus.json"),
        help="terminal-entry corpus JSON",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("data/artifacts/hailmary/corpus/hailmary_templates.json"),
        help="Hailmary template store JSON",
    )
    parser.add_argument(
        "--route-graph",
        type=Path,
        help=(
            "compiled route-graph JSON; when omitted, a sibling route_graph.json "
            "or route_graph_input.json is discovered automatically"
        ),
    )
    parser.add_argument(
        "--start",
        type=float,
        help="first candidate window start as a Unix timestamp (defaults to corpus range)",
    )
    parser.add_argument(
        "--stop",
        type=float,
        help="exclusive stop for candidate window starts (defaults to corpus range)",
    )
    parser.add_argument(
        "--window-start",
        help=(
            "window to inspect; accepts a Unix timestamp or an ISO date/time such "
            "as '2026-04-01 09:00' (a seeded non-empty window is chosen by default)"
        ),
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--timezone",
        type=_timezone,
        default=ZoneInfo("UTC"),
        help="IANA timezone used for display and typed window starts (default: UTC)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--rulebook",
        type=Path,
        help=(
            "optional exported frozen rulebook used as the common continuation "
            "policy in action/no-op previews; omitted means no later actions"
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="serve the GUI without opening the system browser",
    )
    return parser


@dataclass(frozen=True, slots=True)
class EventQueueTrace:
    """Compact, browser-ready audit produced from immutable simulator states.

    The first frame is the untouched initial state. Every later frame is the
    state returned by exactly one :meth:`Simulator.advance_next` call. Queue
    deltas avoid serializing the shrinking heap repeatedly while retaining its
    exact contents and production sort order at every frame.
    """

    payload: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)

    def queue_refs_at(self, frame_index: int) -> tuple[str, ...]:
        frames = self.payload["frames"]
        events = self.payload["event_catalog"]
        assert isinstance(frames, list)
        assert isinstance(events, dict)
        if not 0 <= frame_index < len(frames):
            raise IndexError(frame_index)
        queue = set(self.payload["initial_queue_refs"])
        for frame in frames[1 : frame_index + 1]:
            queue.difference_update(frame["queue_removed_refs"])
            queue.update(frame["queue_added_refs"])
        return tuple(sorted(queue, key=lambda ref: tuple(events[ref]["sort_key"])))


@dataclass(frozen=True, slots=True)
class ActionOpportunityRecord:
    """Typed server-side binding hidden behind one browser-safe action ID."""

    anchor: LeaderFollowerAnchor
    candidate: ActionCandidate


@dataclass(frozen=True, slots=True)
class EventQueueVerifierFrame:
    snapshot: Mapping[str, object]
    batch: EventBatchResult | None
    opportunities: tuple[ActionOpportunityRecord, ...]

    @property
    def by_action_id(self) -> dict[str, ActionOpportunityRecord]:
        return {record.candidate.action_id: record for record in self.opportunities}


@dataclass(slots=True)
class EventQueueVerifierSession:
    """Replay trace plus immutable frame roots used for lazy action previews."""

    trace: EventQueueTrace
    definition: ScenarioDefinition
    runtime: ActionRuntime
    frames: tuple[EventQueueVerifierFrame, ...]
    coordinate_mode: str
    continuation_policy: Any = field(default_factory=NoOpPolicy)
    policy_label: str = "No later actions"
    outcome_config: OutcomeConfig = field(default_factory=OutcomeConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    scenario_config: ScenarioConfig = field(default_factory=ScenarioConfig)
    preview_cache_size: int = 128
    _preview_cache: OrderedDict[tuple[str, ...], dict[str, object]] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _preview_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def preview(self, frame_index: int, action_id: str) -> dict[str, object]:
        if not 0 <= int(frame_index) < len(self.frames):
            raise IndexError(frame_index)
        record = self.frames[int(frame_index)].by_action_id.get(str(action_id))
        if record is None:
            raise KeyError(
                f"action {action_id!r} is not available at frame {frame_index}"
            )
        frame_payload = self.trace.payload["frames"][int(frame_index)]
        assert isinstance(frame_payload, Mapping)
        root_hash = str(frame_payload["dynamic_content_hash"])
        key = (
            root_hash,
            record.candidate.action_id,
            policy_fingerprint(self.continuation_policy),
            self.runtime.runtime_configuration_hash,
            content_hash(self.outcome_config, namespace="hailmary.outcome_config"),
        )
        with self._preview_lock:
            cached = self._preview_cache.get(key)
            if cached is not None:
                self._preview_cache.move_to_end(key)
                return cached
            result = _build_preview_payload(
                self,
                frame_index=int(frame_index),
                record=record,
            )
            self._preview_cache[key] = result
            self._preview_cache.move_to_end(key)
            while len(self._preview_cache) > max(1, int(self.preview_cache_size)):
                self._preview_cache.popitem(last=False)
            return result


def _jsonable(value: Any) -> Any:
    """Convert immutable Hailmary diagnostics to strict JSON data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _jsonable(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _event_ref(event: ScheduledEvent) -> str:
    return f"{event.insertion_sequence}:{event.event_id}"


def _event_payload(event: ScheduledEvent, timezone: ZoneInfo) -> dict[str, object]:
    return {
        "ref": _event_ref(event),
        "time_s": float(event.time_s),
        "time_label": _clock_label(event.time_s, timezone),
        "kind": event.kind.value,
        "priority": int(event.priority),
        "event_id": event.event_id,
        "insertion_sequence": int(event.insertion_sequence),
        "flight_id": event.flight_id,
        "station_index": int(event.station_index),
        "resource_id": event.resource_id,
        "payload": event.payload_dict,
        "sort_key": list(event.sort_key),
    }


def _clock_label(time_s: float, timezone: ZoneInfo) -> str:
    return (
        datetime.fromtimestamp(time_s, tz=UTC)
        .astimezone(timezone)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def _window_label(sample: ScalingSample, timezone: ZoneInfo) -> str:
    start = datetime.fromtimestamp(sample.scenario.window.start_s, tz=UTC).astimezone(
        timezone
    )
    end = datetime.fromtimestamp(sample.scenario.window.end_s, tz=UTC).astimezone(
        timezone
    )
    zone = getattr(timezone, "key", str(timezone))
    if start.date() == end.date():
        return f"{start:%Y-%m-%d %H:%M}\u2013{end:%H:%M} {zone}"
    return f"{start:%Y-%m-%d %H:%M}\u2013{end:%Y-%m-%d %H:%M} {zone}"


def _variant_array(variant: object, name: str) -> np.ndarray | None:
    raw: object | None = None
    if isinstance(variant, Mapping):
        raw = variant.get(name)
    if raw is None:
        raw = getattr(variant, name, None)
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        return None
    return values


def _coordinate_mode(sample: ScalingSample) -> str:
    latitudes: list[np.ndarray] = []
    longitudes: list[np.ndarray] = []
    for variant in sample.scenario.definition.variants:
        lat = _variant_array(variant, "lat_deg")
        lon = _variant_array(variant, "lon_deg")
        if lat is None or lon is None or len(lat) != len(lon):
            return "local"
        latitudes.append(lat)
        longitudes.append(lon)
    if not latitudes:
        return "local"
    lat = np.concatenate(latitudes)
    lon = np.concatenate(longitudes)
    geographic_span = float(np.ptp(lat) + np.ptp(lon))
    return "geographic" if geographic_span > 1.0e-5 else "local"


def _downsample_indices(length: int, *, maximum: int = 240) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=np.int64))


def _map_payload(sample: ScalingSample, coordinate_mode: str) -> dict[str, object]:
    routes: list[dict[str, object]] = []
    all_x: list[float] = []
    all_y: list[float] = []
    definition = sample.scenario.definition
    for flight in definition.flights:
        variant = definition.variant(flight.baseline_variant_id)
        x_name, y_name = (
            ("lon_deg", "lat_deg")
            if coordinate_mode == "geographic"
            else ("east_m", "north_m")
        )
        x = _variant_array(variant, x_name)
        y = _variant_array(variant, y_name)
        if x is None or y is None or len(x) != len(y):
            continue
        indices = _downsample_indices(len(x))
        points = [
            [round(float(x[index]), 7), round(float(y[index]), 7)] for index in indices
        ]
        all_x.extend(point[0] for point in points)
        all_y.extend(point[1] for point in points)
        routes.append(
            {
                "flight_id": flight.flight_id,
                "runway": flight.runway,
                "cluster_id": flight.cluster_id,
                "points": points,
            }
        )
    if not all_x or not all_y:
        bounds = {"min_x": -1.0, "max_x": 1.0, "min_y": -1.0, "max_y": 1.0}
    else:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        x_pad = max(
            (max_x - min_x) * 0.06, 1.0e-4 if coordinate_mode == "geographic" else 100.0
        )
        y_pad = max(
            (max_y - min_y) * 0.06, 1.0e-4 if coordinate_mode == "geographic" else 100.0
        )
        bounds = {
            "min_x": min_x - x_pad,
            "max_x": max_x + x_pad,
            "min_y": min_y - y_pad,
            "max_y": max_y + y_pad,
        }
    return {"coordinate_mode": coordinate_mode, "bounds": bounds, "routes": routes}


def _flight_positions(
    simulator: Simulator,
    *,
    coordinate_mode: str,
    at_time_s: float | None = None,
) -> list[dict[str, object]]:
    positions: list[dict[str, object]] = []
    for dynamic in simulator.state.flights:
        sample = simulator.sample_flight(
            dynamic.flight_id,
            at_time_s=at_time_s,
            clip=True,
        )
        x = sample.lon_deg if coordinate_mode == "geographic" else sample.east_m
        y = sample.lat_deg if coordinate_mode == "geographic" else sample.north_m
        if x is None or y is None:
            continue
        definition = simulator.definition.flight(dynamic.flight_id)
        positions.append(
            {
                "flight_id": dynamic.flight_id,
                "callsign": definition.callsign or dynamic.flight_id,
                "runway": definition.runway,
                "lifecycle": dynamic.lifecycle.value,
                "synthetic": bool(definition.metadata_dict.get("synthetic")),
                "x": round(float(x), 7),
                "y": round(float(y), 7),
                "altitude_m": (
                    None
                    if sample.altitude_m is None
                    else round(float(sample.altitude_m), 1)
                ),
                "ground_speed_mps": (
                    None
                    if sample.ground_speed_mps is None
                    else round(float(sample.ground_speed_mps), 2)
                ),
                "remaining_distance_m": round(float(sample.s_m), 1),
                "variant_id": dynamic.current_variant_id,
            }
        )
    return positions


def _available_actions(
    simulator: Simulator,
    batch: EventBatchResult | None,
    catalog: ActionCatalog,
) -> tuple[list[dict[str, object]], tuple[ActionOpportunityRecord, ...]]:
    if batch is None or batch.decision_epoch is None:
        return [], ()
    actions: list[dict[str, object]] = []
    records: list[ActionOpportunityRecord] = []
    anchors = build_current_segment_anchors(simulator)
    for anchor in anchors.leader_follower:
        candidates = catalog.enumerate_for_batch(
            simulator,
            batch,
            anchor_id=anchor.anchor_id,
            bound_flight_id=anchor.follower_id,
            resource_id=anchor.resource_id,
            segment_id=anchor.segment_id,
        )
        for candidate in candidates:
            records.append(ActionOpportunityRecord(anchor, candidate))
            actions.append(
                {
                    "action_id": candidate.action_id,
                    "anchor_id": anchor.anchor_id,
                    "leader_id": anchor.leader_id,
                    "follower_id": anchor.follower_id,
                    "segment_id": anchor.segment_id,
                    "resource_id": anchor.resource_id,
                    "lever": candidate.lever.value,
                    "band": candidate.band,
                    "station_index": candidate.station_index,
                    "station_m": round(float(candidate.s_m), 1),
                    "feasible": bool(candidate.feasible),
                    "reason": candidate.reason,
                    "opportunity_aircraft_id": anchor.follower_id,
                    "entry_resource_id": anchor.entry_resource_id,
                    "ordering_basis": anchor.ordering_basis,
                    "predicted_exit_interval_s": anchor.predicted_exit_interval_s,
                    "catch_up": bool(anchor.catch_up),
                    "candidate_metadata": _jsonable(
                        dict(candidate.realization_metadata)
                    ),
                }
            )
    return actions, tuple(records)


def _frame_payload(
    simulator: Simulator,
    *,
    index: int,
    batch: EventBatchResult | None,
    previous_queue: set[str],
    event_catalog: dict[str, dict[str, object]],
    timezone: ZoneInfo,
    coordinate_mode: str,
    catalog: ActionCatalog,
) -> tuple[
    dict[str, object],
    set[str],
    tuple[ActionOpportunityRecord, ...],
]:
    pending = sorted(simulator.state.event_heap, key=lambda event: event.sort_key)
    for event in pending:
        event_catalog[_event_ref(event)] = _event_payload(event, timezone)
    if batch is not None:
        for event in batch.events:
            event_catalog[_event_ref(event)] = _event_payload(event, timezone)
    queue = {_event_ref(event) for event in pending}
    decision = None if batch is None else batch.decision_epoch
    available_actions, opportunities = _available_actions(
        simulator,
        batch,
        catalog,
    )
    frame = {
        "index": index,
        "time_s": float(simulator.state.sim_time_s),
        "time_label": _clock_label(simulator.state.sim_time_s, timezone),
        "state_id": simulator.state.state_id,
        "state_version": int(simulator.state.version),
        "dynamic_content_hash": simulator.dynamic_content_hash,
        "decision_epoch_index": int(simulator.state.decision_epoch_index),
        "decision_trigger_event_ids": (
            [] if decision is None else list(decision.trigger_event_ids)
        ),
        "processed_event_refs": (
            [] if batch is None else [_event_ref(event) for event in batch.events]
        ),
        "queue_removed_refs": sorted(previous_queue - queue),
        "queue_added_refs": sorted(queue - previous_queue),
        "queue_count": len(queue),
        "positions": _flight_positions(
            simulator,
            coordinate_mode=coordinate_mode,
        ),
        "available_actions": available_actions,
    }
    return frame, queue, opportunities


def _runtime_for_definition(
    definition: ScenarioDefinition,
    *,
    catalog: ActionCatalog | None = None,
) -> ActionRuntime:
    metadata = definition.metadata_dict
    raw_template = metadata.get("template_config")
    template_config = (
        TemplateConfig(**dict(raw_template))
        if isinstance(raw_template, Mapping)
        else catalog.config if catalog is not None else TemplateConfig()
    )
    return build_action_runtime(
        template_config=template_config,
        catalog=catalog,
    )


def _feature_config_for_definition(definition: ScenarioDefinition) -> FeatureConfig:
    payload = definition.metadata_dict.get("feature_config")
    return (
        FeatureConfig(**dict(payload))
        if isinstance(payload, Mapping)
        else FeatureConfig()
    )


def _scenario_config_for_definition(definition: ScenarioDefinition) -> ScenarioConfig:
    payload = definition.metadata_dict.get("scenario_config")
    return (
        ScenarioConfig(**dict(payload))
        if isinstance(payload, Mapping)
        else ScenarioConfig()
    )


def build_event_queue_session(
    sample: ScalingSample,
    *,
    timezone: ZoneInfo = ZoneInfo("UTC"),
    route_graph_source: str | None = None,
    catalog: ActionCatalog | None = None,
    runtime: ActionRuntime | None = None,
    continuation_policy: Any | None = None,
    policy_label: str | None = None,
    outcome_config: OutcomeConfig | None = None,
) -> EventQueueVerifierSession:
    """Replay the production queue and retain exact roots for lazy previews."""

    if runtime is not None and catalog is not None and runtime.catalog is not catalog:
        raise ValueError("provide either runtime or catalog, not two catalog instances")
    action_runtime = (
        _runtime_for_definition(sample.scenario.definition, catalog=catalog)
        if runtime is None
        else runtime
    )
    action_catalog = action_runtime.catalog
    simulator = action_runtime.create_simulator(sample.scenario.definition)
    coordinate_mode = _coordinate_mode(sample)
    event_catalog: dict[str, dict[str, object]] = {}
    initial_queue_events = sorted(
        simulator.state.event_heap, key=lambda event: event.sort_key
    )
    for event in initial_queue_events:
        event_catalog[_event_ref(event)] = _event_payload(event, timezone)
    initial_queue = {_event_ref(event) for event in initial_queue_events}

    first, queue, first_opportunities = _frame_payload(
        simulator,
        index=0,
        batch=None,
        previous_queue=initial_queue,
        event_catalog=event_catalog,
        timezone=timezone,
        coordinate_mode=coordinate_mode,
        catalog=action_catalog,
    )
    frames = [first]
    verifier_frames = [
        EventQueueVerifierFrame(
            snapshot=simulator.snapshot(),
            batch=None,
            opportunities=first_opportunities,
        )
    ]
    while True:
        batch = simulator.advance_next()
        if batch is None:
            break
        frame, queue, opportunities = _frame_payload(
            simulator,
            index=len(frames),
            batch=batch,
            previous_queue=queue,
            event_catalog=event_catalog,
            timezone=timezone,
            coordinate_mode=coordinate_mode,
            catalog=action_catalog,
        )
        frames.append(frame)
        verifier_frames.append(
            EventQueueVerifierFrame(
                snapshot=simulator.snapshot(),
                batch=batch,
                opportunities=opportunities,
            )
        )

    flight_manifest: list[dict[str, object]] = []
    for flight in sorted(
        sample.scenario.definition.flights,
        key=lambda item: (item.release_time_s, item.flight_id),
    ):
        metadata = flight.metadata_dict
        flight_manifest.append(
            {
                "flight_id": flight.flight_id,
                "callsign": flight.callsign or flight.flight_id,
                "runway": flight.runway,
                "cluster_id": flight.cluster_id,
                "release_time_s": float(flight.release_time_s),
                "release_label": _clock_label(flight.release_time_s, timezone),
                "synthetic": bool(metadata.get("synthetic")),
                "donor_flight_id": metadata.get("donor_flight_id") or None,
            }
        )
    removed = [
        {
            "flight_id": item.flight_id,
            "callsign": item.callsign or item.flight_id,
            "runway": item.key.runway,
            "cluster_id": item.key.cluster,
            "release_time_s": float(item.terminal_entry_time_s),
            "release_label": _clock_label(item.terminal_entry_time_s, timezone),
        }
        for item in sample.removed_arrivals
    ]
    payload: dict[str, object] = {
        "schema_version": "hailmary.event-queue-verifier.v2",
        "scenario_id": sample.scenario.definition.scenario_id,
        "definition_hash": sample.scenario.definition.definition_hash,
        "fidelity": {
            "queue_driver": "Simulator.advance_next",
            "batching": "all equal-time events in production sort order",
            "state_boundary": "post-event immutable SimulationState",
            "actions": "ActionCatalog.enumerate_for_batch",
            "position_sampling": "Simulator.sample_flight(clip=True)",
            "action_realization": "ActionRuntime and Simulator.apply",
            "features": "simulator_state_vector",
            "objective": "paired_simulator_rollout and score_simulator_outcome",
        },
        "window": {
            "start_s": float(sample.scenario.window.start_s),
            "end_s": float(sample.scenario.window.end_s),
            "label": _window_label(sample, timezone),
            "timezone": getattr(timezone, "key", str(timezone)),
        },
        "scale": float(sample.scenario.scale),
        "replicate": int(sample.scenario.replicate),
        "original_count": sample.original_count,
        "new_flight_count": sample.new_count,
        "synthetic_count": sample.synthetic_count,
        "removed_count": len(sample.removed_arrivals),
        "route_graph_source": route_graph_source,
        "flights": flight_manifest,
        "removed_flights": removed,
        "map": _map_payload(sample, coordinate_mode),
        "action_vocabulary": [
            identity.to_dict() for identity in action_catalog.vocabulary.identities
        ],
        "runtime_configuration_hash": action_runtime.runtime_configuration_hash,
        "policy": {
            "label": policy_label or "No later actions",
            "fingerprint": policy_fingerprint(
                NoOpPolicy() if continuation_policy is None else continuation_policy
            ),
            "mode": (
                "no_later_actions"
                if continuation_policy is None
                or isinstance(continuation_policy, NoOpPolicy)
                else "frozen_rulebook"
            ),
        },
        "event_catalog": event_catalog,
        "initial_queue_refs": [_event_ref(event) for event in initial_queue_events],
        "frames": frames,
    }
    policy = NoOpPolicy() if continuation_policy is None else continuation_policy
    return EventQueueVerifierSession(
        trace=EventQueueTrace(payload),
        definition=sample.scenario.definition,
        runtime=action_runtime,
        frames=tuple(verifier_frames),
        coordinate_mode=coordinate_mode,
        continuation_policy=policy,
        policy_label=policy_label or "No later actions",
        outcome_config=OutcomeConfig() if outcome_config is None else outcome_config,
        feature_config=_feature_config_for_definition(sample.scenario.definition),
        scenario_config=_scenario_config_for_definition(sample.scenario.definition),
    )


def build_event_queue_trace(
    sample: ScalingSample,
    *,
    timezone: ZoneInfo = ZoneInfo("UTC"),
    route_graph_source: str | None = None,
    catalog: ActionCatalog | None = None,
) -> EventQueueTrace:
    """Backward-compatible browser trace without exposing the preview session."""

    return build_event_queue_session(
        sample,
        timezone=timezone,
        route_graph_source=route_graph_source,
        catalog=catalog,
    ).trace


def _vector_payload(vector: Any) -> dict[str, object]:
    provenance = vector.diagnostics.get("provenance", {})
    values = []
    for field_definition in vector.schema.fields:
        name = field_definition.name
        item = dict(provenance.get(name, {}))
        item.update(
            name=name,
            value=float(vector.named[name]),
            unit=field_definition.unit,
            normalization=field_definition.normalization,
            lower_bound=field_definition.lower_bound,
            upper_bound=field_definition.upper_bound,
            missingness_mask=field_definition.missingness_mask,
        )
        values.append(_jsonable(item))
    diagnostics = {
        key: item for key, item in vector.diagnostics.items() if key != "provenance"
    }
    vector_hash = content_hash(
        {
            "schema_hash": vector.schema_hash,
            "values": tuple(float(value) for value in vector.values),
            "categories": dict(vector.categories),
        },
        namespace="hailmary.verifier.feature_vector.v1",
    )
    return {
        "schema_version": vector.schema.schema_version,
        "schema_hash": vector.schema_hash,
        "vector_hash": vector_hash,
        "categories": dict(vector.categories),
        "bindings": _jsonable(vector.diagnostics.get("bindings", {})),
        "values": values,
        "diagnostics": _jsonable(diagnostics),
    }


def _variant_route_payload(
    variant: Any,
    *,
    coordinate_mode: str,
) -> dict[str, object]:
    x_values = (
        getattr(variant, "lon_deg")
        if coordinate_mode == "geographic"
        else getattr(variant, "east_m")
    )
    y_values = (
        getattr(variant, "lat_deg")
        if coordinate_mode == "geographic"
        else getattr(variant, "north_m")
    )
    points = [
        [round(float(x), 7), round(float(y), 7)]
        for x, y in zip(x_values, y_values, strict=True)
    ]
    return {
        "variant_id": str(variant.variant_id),
        "cluster_id": str(getattr(variant, "cluster_id", "")),
        "points": points,
    }


def _variant_station_point(
    variant: Any,
    station_m: float,
    *,
    coordinate_mode: str,
) -> dict[str, float]:
    x_values = (
        getattr(variant, "lon_deg")
        if coordinate_mode == "geographic"
        else getattr(variant, "east_m")
    )
    y_values = (
        getattr(variant, "lat_deg")
        if coordinate_mode == "geographic"
        else getattr(variant, "north_m")
    )
    stations = getattr(variant, "s_m")
    return {
        "station_m": float(station_m),
        "x": float(np.interp(float(station_m), stations, x_values)),
        "y": float(np.interp(float(station_m), stations, y_values)),
    }


def _append_recorded_state(
    states: list[SimulationState],
    state: SimulationState,
) -> None:
    if states and (
        states[-1].dynamic_content_hash == state.dynamic_content_hash
        and abs(states[-1].sim_time_s - state.sim_time_s) <= 1.0e-9
    ):
        return
    states.append(state)


def _timeline_payload(
    states: Sequence[SimulationState],
    *,
    sample_times_s: Sequence[float],
    runtime: ActionRuntime,
    coordinate_mode: str,
    focus_flight_id: str,
) -> dict[str, object]:
    if not states:
        raise RuntimeError("rollout arm did not retain any simulator states")
    ordered_states = sorted(states, key=lambda item: item.sim_time_s)
    checkpoints = [
        {
            "time_s": float(state.sim_time_s),
            "dynamic_content_hash": state.dynamic_content_hash,
            "state_version": int(state.version),
        }
        for state in ordered_states
    ]
    samples: list[dict[str, object]] = []
    state_index = 0
    for raw_time in sample_times_s:
        time_s = float(raw_time)
        while (
            state_index + 1 < len(ordered_states)
            and ordered_states[state_index + 1].sim_time_s <= time_s + 1.0e-9
        ):
            state_index += 1
        simulator = Simulator.from_state(
            ordered_states[state_index],
            action_applier=runtime.action_applier,
            runtime_configuration_hash=runtime.runtime_configuration_hash,
        )
        samples.append(
            {
                "time_s": time_s,
                "positions": _flight_positions(
                    simulator,
                    coordinate_mode=coordinate_mode,
                    at_time_s=time_s,
                ),
            }
        )

    route_history: list[dict[str, object]] = []
    seen_variants: set[str] = set()
    for state in ordered_states:
        dynamic = state.flight(focus_flight_id)
        variant_id = str(dynamic.current_variant_id)
        if variant_id in seen_variants:
            continue
        seen_variants.add(variant_id)
        route = _variant_route_payload(
            state.definition.variant(variant_id),
            coordinate_mode=coordinate_mode,
        )
        route["valid_from_s"] = float(state.sim_time_s)
        route_history.append(route)
    return {
        "checkpoints": checkpoints,
        "samples": samples,
        "route_history": route_history,
    }


def _arm_realization_payload(
    arm: PairedArmTrace,
    states: Sequence[SimulationState],
) -> dict[str, object]:
    result: dict[str, object] = {
        "action": _jsonable(arm.initial_action),
        "applied_action": _jsonable(arm.applied_action),
        "audit": _jsonable(arm.action_audit),
    }
    variant_id = getattr(arm.action_audit, "variant_id", None)
    if variant_id and states:
        variant = states[0].definition.variant(str(variant_id))
        result["variant"] = {
            "variant_id": str(variant.variant_id),
            "duration_s": float(variant.duration_s),
            "action_provenance": _jsonable(variant.action_provenance),
            "diagnostics": _jsonable(variant.diagnostics),
        }
    else:
        result["variant"] = None
    return result


def _arm_objective_payload(
    arm: PairedArmTrace,
    states: Sequence[SimulationState],
) -> dict[str, object]:
    return {
        "score": float(arm.score),
        "initial_dynamic_content_hash": arm.initial_dynamic_content_hash,
        "final_dynamic_content_hash": arm.final_dynamic_content_hash,
        "outcome": _jsonable(arm.outcome),
        "realization": _arm_realization_payload(arm, states),
    }


def _build_preview_payload(
    session: EventQueueVerifierSession,
    *,
    frame_index: int,
    record: ActionOpportunityRecord,
) -> dict[str, object]:
    verifier_frame = session.frames[frame_index]
    parent = session.runtime.resume_simulator(
        session.definition,
        verifier_frame.snapshot,
    )
    parent_hash = parent.dynamic_content_hash
    frame_payload = session.trace.payload["frames"][frame_index]
    assert isinstance(frame_payload, Mapping)
    if parent_hash != str(frame_payload["dynamic_content_hash"]):
        raise RuntimeError("retained verifier frame does not match its browser hash")

    same_anchor = tuple(
        item.candidate
        for item in verifier_frame.opportunities
        if item.anchor.anchor_id == record.anchor.anchor_id
    )
    try:
        no_op = next(
            candidate
            for candidate in same_anchor
            if candidate.lever is ActionLever.NO_OP
        )
    except StopIteration as exc:
        raise RuntimeError("action opportunity has no canonical no-op arm") from exc

    vector = simulator_state_vector(
        parent,
        record.anchor,
        action_candidates=same_anchor,
        action_applier=session.runtime.action_applier,
        feature_config=session.feature_config,
        scenario_config=session.scenario_config,
        template_config=session.runtime.template_config,
    )
    outcome_plan = simulator_outcome_plan(
        parent,
        record.anchor,
        config=session.outcome_config,
    )
    focus_dynamic = parent.state.flight(record.anchor.follower_id)
    focus_variant = parent.definition.variant(focus_dynamic.current_variant_id)
    action_station_point = _variant_station_point(
        focus_variant,
        record.candidate.s_m,
        coordinate_mode=session.coordinate_mode,
    )
    target_resource_point = _variant_station_point(
        focus_variant,
        resource_station_m(
            parent,
            record.anchor.follower_id,
            record.anchor.resource_id,
        ),
        coordinate_mode=session.coordinate_mode,
    )
    recorded: dict[str, list[SimulationState]] = {}
    labels = iter(("after", "before"))

    def rollout_runner(branch: Any, horizon_s: float, policy: Any) -> Any:
        label = next(labels)
        states = [branch.state]

        def observer(current: Simulator, _batch: EventBatchResult | None) -> None:
            _append_recorded_state(states, current.state)

        branch.run_until(
            float(horizon_s),
            policy=policy,
            observer=observer,
        )
        recorded[label] = states
        return branch

    rollout = paired_simulator_rollout(
        parent,
        selected_action=record.candidate,
        contender_action=no_op,
        frozen_policy=session.continuation_policy,
        outcome_plan=outcome_plan,
        outcome_config=session.outcome_config,
        rollout_runner=rollout_runner,
    )
    if parent.dynamic_content_hash != parent_hash:
        raise RuntimeError("action preview mutated its retained parent root")

    root_time_s = float(outcome_plan.root_time_s)
    horizon_s = float(outcome_plan.horizon_s)
    regular_times = np.linspace(root_time_s, horizon_s, 31)
    checkpoint_times = {
        float(state.sim_time_s)
        for arm_states in recorded.values()
        for state in arm_states
    }
    sample_times_s = tuple(
        sorted(
            {
                round(float(value), 9)
                for value in (*regular_times, *checkpoint_times)
                if root_time_s - 1.0e-9 <= float(value) <= horizon_s + 1.0e-9
            }
        )
    )
    after_states = tuple(recorded["after"])
    before_states = tuple(recorded["before"])
    binding = vector.diagnostics["bindings"]
    vector_payload = _vector_payload(vector)
    outcome_plan_hash = content_hash(
        outcome_plan,
        namespace="hailmary.simulator_outcome_plan.v1",
    )
    return {
        "schema_version": "hailmary.event-queue-action-preview.v1",
        "frame_index": frame_index,
        "action_id": record.candidate.action_id,
        "comparison": {
            "before_label": "No-op arm",
            "after_label": "Selected action arm",
            "delta_after_minus_before": float(rollout.delta),
            "common_root_time_s": root_time_s,
            "common_horizon_s": horizon_s,
            "continuation_policy": session.policy_label,
        },
        "binding": _jsonable(binding),
        "feature_vector": vector_payload,
        "objective": {
            "before": _arm_objective_payload(
                rollout.contender,
                before_states,
            ),
            "after": _arm_objective_payload(
                rollout.selected,
                after_states,
            ),
            "delta": float(rollout.delta),
            "outcome_plan": _jsonable(outcome_plan),
        },
        "map": {
            "coordinate_mode": session.coordinate_mode,
            "root_time_s": root_time_s,
            "horizon_s": horizon_s,
            "sample_times_s": list(sample_times_s),
            "action_station": action_station_point,
            "target_resource": target_resource_point,
            "before": _timeline_payload(
                before_states,
                sample_times_s=sample_times_s,
                runtime=session.runtime,
                coordinate_mode=session.coordinate_mode,
                focus_flight_id=record.anchor.follower_id,
            ),
            "after": _timeline_payload(
                after_states,
                sample_times_s=sample_times_s,
                runtime=session.runtime,
                coordinate_mode=session.coordinate_mode,
                focus_flight_id=record.anchor.follower_id,
            ),
        },
        "provenance": {
            "scenario_definition_hash": session.definition.definition_hash,
            "root_dynamic_content_hash": parent_hash,
            "outcome_plan_hash": outcome_plan_hash,
            "runtime_configuration_hash": session.runtime.runtime_configuration_hash,
            "runtime_configuration": _jsonable(
                session.runtime.realization_configuration
            ),
            "action_vocabulary_hash": session.runtime.action_vocabulary_hash,
            "feature_schema_hash": vector.schema_hash,
            "feature_vector_hash": vector_payload["vector_hash"],
            "continuation_policy_fingerprint": rollout.policy_fingerprint,
            "continuation_policy_label": session.policy_label,
            "outcome_config_hash": content_hash(
                session.outcome_config,
                namespace="hailmary.outcome_config",
            ),
            "outcome_config": _jsonable(session.outcome_config),
            "feature_config": _jsonable(session.feature_config),
            "scenario_config": _jsonable(session.scenario_config),
            "parent_immutable": parent.dynamic_content_hash == parent_hash,
        },
    }


def _route_input(path: Path) -> tuple[MedoidRoute, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("routes") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("route graph input must contain a routes list")
    dataset_id = str(payload.get("dataset_id", ""))
    routes: list[MedoidRoute] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each route graph input record must be an object")
        routes.append(
            MedoidRoute(
                dataset_id=str(record.get("dataset_id", dataset_id)),
                airport=str(record["airport"]),
                runway=str(record["runway"]),
                cluster_id=str(record["cluster_id"]),
                lat_deg=tuple(record["lat_deg"]),
                lon_deg=tuple(record["lon_deg"]),
                dispersion_m=float(record.get("dispersion_m", 0.0)),
                medoid_flight_id=str(record.get("medoid_flight_id", "")),
                source_hash=str(record.get("source_hash", "")),
            )
        )
    return tuple(routes)


def _resolve_route_graph(
    corpus_path: Path,
    explicit_path: Path | None,
) -> tuple[RouteGraphArtifact | None, str | None]:
    if explicit_path is not None:
        resolved = explicit_path.resolve()
        return RouteGraphArtifact.read(resolved), resolved.as_posix()
    compiled = corpus_path.parent / "route_graph.json"
    if compiled.is_file():
        return RouteGraphArtifact.read(compiled), compiled.resolve().as_posix()
    source = corpus_path.parent / "route_graph_input.json"
    if source.is_file():
        config = RouteGraphConfig()
        routes, uncertain = partition_medoid_routes(_route_input(source), config=config)
        for route in uncertain:
            print(
                "uncertain medoid excluded: "
                f"{route.qualified_cluster_id} "
                f"dispersion_nm={route.dispersion_m / M_PER_NM:.3f} "
                f"limit_nm={config.maximum_medoid_dispersion_nm:.3f}",
                file=sys.stderr,
            )
        graph = build_route_graph(
            routes,
            config=config,
            provenance={"input": source.resolve().as_posix(), "in_memory": True},
        )
        return graph, f"{source.resolve().as_posix()} (compiled in memory)"
    return None, None


def _load_sample(
    args: argparse.Namespace,
) -> tuple[ScalingSample, str | None]:
    corpus = TerminalEntryCorpus.read(args.corpus)
    store = TemplateStore.read(args.templates)
    templates = {
        f"{template.airport_id}:{template.runway_id}:{template.cluster_id}": template
        for template in store.templates
    }
    route_graph, route_graph_source = _resolve_route_graph(
        args.corpus, args.route_graph
    )
    builder = TrafficScenarioBuilder(
        corpus.arrivals,
        templates_by_cluster=templates,
        route_graph=route_graph,
        rejection_counts=dict(corpus.rejection_counts),
    )
    start, stop = _default_bounds(
        corpus.arrivals,
        start_s=args.start,
        stop_s=args.stop,
    )
    inspector = DemandScalingInspector(
        corpus,
        builder,
        windows=iter_demand_windows(start, stop, config=DemandWindowConfig()),
        scale_config=TrafficScaleConfig(
            global_scale=args.scale,
            replicate=args.replicate,
            master_seed=args.seed,
        ),
        timezone=args.timezone,
    )
    sample = (
        inspector.draw()
        if args.window_start is None
        else inspector.sample(inspector.parse_window_start(args.window_start))
    )
    return sample, route_graph_source


def create_app(
    verifier: EventQueueTrace | EventQueueVerifierSession,
):
    """Create the local FastAPI app without starting a server (useful in tests)."""

    # Keep the optional web stack outside the static core dependency graph.
    fastapi = import_module("fastapi")
    responses = import_module("fastapi.responses")

    session = verifier if isinstance(verifier, EventQueueVerifierSession) else None
    trace = verifier.trace if session is not None else verifier
    app = fastapi.FastAPI(
        title="Hailmary Event Queue Verifier",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/", response_class=responses.HTMLResponse)
    def index() -> str:
        return EVENT_QUEUE_HTML

    @app.get("/api/trace", response_class=responses.JSONResponse)
    def trace_payload() -> dict[str, object]:
        return trace.to_dict()

    @app.post("/api/preview", response_class=responses.JSONResponse)
    def action_preview(request: dict[str, object]) -> dict[str, object]:
        if session is None:
            raise fastapi.HTTPException(
                status_code=409,
                detail="this app was created from a trace without retained preview roots",
            )
        try:
            frame_index = int(request["frame_index"])
            action_id = str(request["action_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise fastapi.HTTPException(
                status_code=400,
                detail="preview requires integer frame_index and string action_id",
            ) from exc
        try:
            return session.preview(frame_index, action_id)
        except (IndexError, KeyError) as exc:
            raise fastapi.HTTPException(status_code=404, detail=str(exc)) from exc
        except HailmaryError as exc:
            raise fastapi.HTTPException(
                status_code=422,
                detail={"error_type": type(exc).__name__, "message": str(exc)},
            ) from exc
        except (ValueError, TypeError) as exc:
            raise fastapi.HTTPException(
                status_code=422,
                detail={"error_type": type(exc).__name__, "message": str(exc)},
            ) from exc

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def serve(
    verifier: EventQueueTrace | EventQueueVerifierSession,
    *,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    import uvicorn

    if not 1 <= int(port) <= 65_535:
        raise ValueError("port must lie in [1, 65535]")
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in url_host and not url_host.startswith("["):
        url_host = f"[{url_host}]"
    url = f"http://{url_host}:{port}"
    print(
        f"Hailmary event-queue verifier: {url} "
        f"({len((verifier.trace if isinstance(verifier, EventQueueVerifierSession) else verifier).payload['frames']) - 1} event batches)"
    )
    if open_browser:
        timer = threading.Timer(0.7, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    uvicorn.run(
        create_app(verifier),
        host=host,
        port=int(port),
        log_level="warning",
    )


def _configured_continuation_policy(
    path: Path | None,
    *,
    definition: ScenarioDefinition,
    runtime: ActionRuntime,
) -> tuple[Any, str]:
    if path is None:
        return NoOpPolicy(), "No later actions"
    from hailmary.learning import SimulatorRulebookPolicy, load_exported_rulebook

    artifact = load_exported_rulebook(path)
    feature_config = _feature_config_for_definition(definition)
    scenario_config = _scenario_config_for_definition(definition)
    policy = SimulatorRulebookPolicy(
        artifact.to_rulebook(),
        template_config=runtime.template_config,
        feature_config=feature_config,
        scenario_config=scenario_config,
        runtime_configuration_hash=runtime.runtime_configuration_hash,
    )
    return policy, f"Frozen rulebook {artifact.content_hash[:12]}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sample, route_graph_source = _load_sample(args)
        runtime = _runtime_for_definition(sample.scenario.definition)
        continuation_policy, policy_label = _configured_continuation_policy(
            args.rulebook,
            definition=sample.scenario.definition,
            runtime=runtime,
        )
        session = build_event_queue_session(
            sample,
            timezone=args.timezone,
            route_graph_source=route_graph_source,
            runtime=runtime,
            continuation_policy=continuation_policy,
            policy_label=policy_label,
        )
        serve(
            session,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EventQueueTrace",
    "EventQueueVerifierSession",
    "build_event_queue_session",
    "build_event_queue_trace",
    "build_parser",
    "create_app",
    "main",
    "serve",
]
