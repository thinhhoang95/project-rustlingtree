"""Browser-based verifier for the deterministic Hailmary event queue."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import threading
import webbrowser
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from hailmary.actions import ActionCatalog
from hailmary.features import build_current_segment_anchors
from hailmary.scenario import (
    DemandWindowConfig,
    TerminalEntryCorpus,
    TrafficScaleConfig,
    TrafficScenarioBuilder,
    iter_demand_windows,
)
from hailmary.simulator import EventBatchResult, ScheduledEvent, Simulator
from hailmary.simulator.interpolation import MonotoneTrajectory
from hailmary.templates import TemplateStore
from hailmary.topology import MedoidRoute, RouteGraphArtifact, build_route_graph

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
) -> list[dict[str, object]]:
    positions: list[dict[str, object]] = []
    for dynamic in simulator.state.flights:
        sample = simulator.sample_flight(dynamic.flight_id, clip=True)
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
) -> list[dict[str, object]]:
    if batch is None or batch.decision_epoch is None:
        return []
    actions: list[dict[str, object]] = []
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
                }
            )
    return actions


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
) -> tuple[dict[str, object], set[str]]:
    pending = sorted(simulator.state.event_heap, key=lambda event: event.sort_key)
    for event in pending:
        event_catalog[_event_ref(event)] = _event_payload(event, timezone)
    if batch is not None:
        for event in batch.events:
            event_catalog[_event_ref(event)] = _event_payload(event, timezone)
    queue = {_event_ref(event) for event in pending}
    decision = None if batch is None else batch.decision_epoch
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
        "available_actions": _available_actions(simulator, batch, catalog),
    }
    return frame, queue


def build_event_queue_trace(
    sample: ScalingSample,
    *,
    timezone: ZoneInfo = ZoneInfo("UTC"),
    route_graph_source: str | None = None,
    catalog: ActionCatalog | None = None,
) -> EventQueueTrace:
    """Replay one scaled scenario through the production queue and capture it."""

    action_catalog = ActionCatalog() if catalog is None else catalog
    simulator = Simulator(sample.scenario.definition)
    coordinate_mode = _coordinate_mode(sample)
    event_catalog: dict[str, dict[str, object]] = {}
    initial_queue_events = sorted(
        simulator.state.event_heap, key=lambda event: event.sort_key
    )
    for event in initial_queue_events:
        event_catalog[_event_ref(event)] = _event_payload(event, timezone)
    initial_queue = {_event_ref(event) for event in initial_queue_events}

    first, queue = _frame_payload(
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
    while True:
        batch = simulator.advance_next()
        if batch is None:
            break
        frame, queue = _frame_payload(
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
        "schema_version": "hailmary.event-queue-verifier.v1",
        "scenario_id": sample.scenario.definition.scenario_id,
        "definition_hash": sample.scenario.definition.definition_hash,
        "fidelity": {
            "queue_driver": "Simulator.advance_next",
            "batching": "all equal-time events in production sort order",
            "state_boundary": "post-event immutable SimulationState",
            "actions": "ActionCatalog.enumerate_for_batch",
            "position_sampling": "Simulator.sample_flight(clip=True)",
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
        "event_catalog": event_catalog,
        "initial_queue_refs": [_event_ref(event) for event in initial_queue_events],
        "frames": frames,
    }
    return EventQueueTrace(payload)


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
        graph = build_route_graph(
            _route_input(source),
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


def create_app(trace: EventQueueTrace):
    """Create the local FastAPI app without starting a server (useful in tests)."""

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(
        title="Hailmary Event Queue Verifier",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return EVENT_QUEUE_HTML

    @app.get("/api/trace", response_class=JSONResponse)
    def trace_payload() -> dict[str, object]:
        return trace.to_dict()

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def serve(
    trace: EventQueueTrace,
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
        f"({len(trace.payload['frames']) - 1} event batches)"
    )
    if open_browser:
        timer = threading.Timer(0.7, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    uvicorn.run(
        create_app(trace),
        host=host,
        port=int(port),
        log_level="warning",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sample, route_graph_source = _load_sample(args)
        trace = build_event_queue_trace(
            sample,
            timezone=args.timezone,
            route_graph_source=route_graph_source,
        )
        serve(
            trace,
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
    "build_event_queue_trace",
    "build_parser",
    "create_app",
    "main",
    "serve",
]
