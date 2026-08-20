from __future__ import annotations

from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from hailmary.cli.visualize_event_queue import (
    build_event_queue_trace,
    build_parser,
    create_app,
)
from hailmary.scenario import (
    ActionStationDefinition,
    ArrivalClusterKey,
    ClusterCount,
    DemandWindow,
    FlightDefinition,
    ObservedArrival,
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
    SegmentTraversalDefinition,
    TrafficScenario,
)
from hailmary.cli.visualize_demand_scaling import ScalingSample

from .test_templates import _straight_variant


def _sample() -> ScalingSample:
    variant = _straight_variant(cluster_id="C1")
    key = ArrivalClusterKey("TEST", "RWY", "C1")
    cluster_id = key.qualified_id
    resources = (
        ResourceDefinition("ENTRY", kind="segment_entry"),
        ResourceDefinition("RWY", kind="runway_threshold"),
    )
    flights = tuple(
        FlightDefinition(
            flight_id=f"F{index}",
            release_time_s=release,
            observed_release_time_s=release,
            baseline_variant_id=variant.variant_id,
            cluster_id=cluster_id,
            callsign=f"TEST{index}",
            runway="RWY",
            action_stations=(ActionStationDefinition(0, 80_000.0, "speed"),),
            resource_crossings=(
                ResourceCrossingDefinition("ENTRY", 90_000.0),
                ResourceCrossingDefinition("RWY", 0.0),
            ),
            segment_traversals=(
                SegmentTraversalDefinition(
                    0,
                    "FINAL",
                    "ENTRY",
                    "RWY",
                    90_000.0,
                    0.0,
                ),
            ),
            metadata={"synthetic": index == 2},
        )
        for index, release in ((1, 0.0), (2, 10.0))
    )
    definition = ScenarioDefinition(
        scenario_id="QUEUE-GUI",
        seed=23,
        flights=flights,
        resources=resources,
        variants=(variant,),
        schema_version="hailmary.scenario.v2",
    )
    scenario = TrafficScenario(
        window=DemandWindow(0.0, 3_600.0),
        scale=1.0,
        replicate=0,
        definition=definition,
        observed_cluster_counts=(ClusterCount(key, 2),),
        target_cluster_counts=(ClusterCount(key, 2),),
    )
    arrivals = tuple(
        ObservedArrival(
            dataset_id="TEST",
            key=key,
            flight_id=flight.flight_id,
            terminal_entry_time_s=flight.release_time_s,
            terminal_entry_ground_speed_mps=100.0,
            terminal_entry_altitude_m=1_000.0,
            baseline_variant_id=variant.variant_id,
        )
        for flight in flights
    )
    return ScalingSample(
        scenario=scenario,
        original_arrivals=arrivals,
        removed_arrivals=(),
    )


def test_cli_help_and_console_script_registration() -> None:
    help_text = build_parser().format_help()
    scripts = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["scripts"]

    assert "hailmary-visualize-event-queue" in help_text
    assert "--window-start" in help_text
    assert (
        scripts["hailmary-visualize-event-queue"]
        == "hailmary.cli.visualize_event_queue:main"
    )


def test_trace_uses_exact_event_batches_queue_deltas_and_all_flight_positions() -> None:
    trace = build_event_queue_trace(_sample(), timezone=ZoneInfo("UTC"))
    payload = trace.to_dict()
    frames = payload["frames"]

    assert payload["fidelity"]["queue_driver"] == "Simulator.advance_next"
    assert payload["original_count"] == 2
    assert payload["new_flight_count"] == 2
    assert len(frames) > 1
    assert len(trace.queue_refs_at(0)) == frames[0]["queue_count"]
    for index, frame in enumerate(frames):
        assert len(trace.queue_refs_at(index)) == frame["queue_count"]
        assert len(frame["positions"]) == 2
    assert trace.queue_refs_at(len(frames) - 1) == ()
    assert {item["lifecycle"] for item in frames[-1]["positions"]} == {"completed"}

    for frame in frames[1:]:
        processed = [
            payload["event_catalog"][ref] for ref in frame["processed_event_refs"]
        ]
        assert [item["sort_key"] for item in processed] == sorted(
            item["sort_key"] for item in processed
        )


def test_trace_shows_only_catalog_actions_eligible_at_the_exact_station_batch() -> None:
    trace = build_event_queue_trace(_sample())
    action_frames = [
        frame for frame in trace.payload["frames"] if frame["available_actions"]
    ]

    assert action_frames
    actions = action_frames[-1]["available_actions"]
    assert {item["follower_id"] for item in actions} == {"F2"}
    assert [(item["lever"], item["band"]) for item in actions] == [
        ("no_op", "no_op"),
        ("speed", "light"),
        ("speed", "medium"),
        ("speed", "heavy"),
    ]


def test_local_web_app_serves_gui_and_trace_api() -> None:
    trace = build_event_queue_trace(_sample())
    client = TestClient(create_app(trace))

    page = client.get("/")
    response = client.get("/api/trace")

    assert page.status_code == 200
    assert "Previous Event" in page.text
    assert "Next Event" in page.text
    assert "Pending event queue" in page.text
    assert response.status_code == 200
    assert response.json()["scenario_id"] == "QUEUE-GUI"
