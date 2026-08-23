from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from hailmary.config import StretchConfig
from hailmary.cli.visualize_event_queue import (
    EventQueueVerifierSession,
    build_event_queue_session,
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
from hailmary.runtime import build_action_runtime

from .test_stretch_oracle import _oracle_simulator
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
    assert "minmax(430px,.95fr)" in page.text
    assert ".inspector::-webkit-scrollbar{display:none}" in page.text
    assert "overscroll-behavior-x:contain" in page.text
    assert response.status_code == 200
    assert response.json()["scenario_id"] == "QUEUE-GUI"


def _speed_preview_target(
    session: EventQueueVerifierSession,
) -> tuple[int, str]:
    for frame_index, frame in enumerate(session.frames):
        for record in frame.opportunities:
            if record.candidate.lever.value == "speed":
                return frame_index, record.candidate.action_id
    raise AssertionError("fixture did not produce a speed opportunity")


def test_preview_uses_canonical_vector_rollout_and_immutable_frame_root() -> None:
    session = build_event_queue_session(_sample())
    frame_index, action_id = _speed_preview_target(session)
    root_hash = session.trace.payload["frames"][frame_index]["dynamic_content_hash"]

    preview = session.preview(frame_index, action_id)

    assert preview["binding"]["leader_id"] == "F1"
    assert preview["binding"]["follower_id"] == "F2"
    assert preview["binding"]["opportunity_aircraft_id"] == "F2"
    assert len(preview["feature_vector"]["values"]) == 25
    vector_names = [item["name"] for item in preview["feature_vector"]["values"]]
    assert len(set(vector_names)) == 25
    assert vector_names[0] == "spacing_deviation_s"
    assert vector_names[-1] == "trailing_spacing_undefined_mask"
    for item in preview["feature_vector"]["values"]:
        assert item["operation"]
        assert "operands" in item
    assert preview["objective"]["delta"] == (
        preview["objective"]["after"]["score"] - preview["objective"]["before"]["score"]
    )
    assert preview["objective"]["after"]["outcome"]["edge_outcomes"]
    assert preview["objective"]["after"]["outcome"]["diagnostics"]["crossing_times_s"]
    assert preview["provenance"]["root_dynamic_content_hash"] == root_hash
    assert preview["provenance"]["parent_immutable"] is True
    assert preview["map"]["action_station"]["station_m"] > 0.0
    assert preview["map"]["after"]["samples"]
    assert preview["map"]["before"]["samples"]


def test_preview_api_resolves_only_actions_from_the_requested_frame() -> None:
    session = build_event_queue_session(_sample())
    frame_index, action_id = _speed_preview_target(session)
    client = TestClient(create_app(session))

    response = client.post(
        "/api/preview",
        json={"frame_index": frame_index, "action_id": action_id},
    )
    stale = client.post(
        "/api/preview",
        json={"frame_index": 0, "action_id": action_id},
    )

    assert response.status_code == 200
    assert response.json()["action_id"] == action_id
    assert stale.status_code == 404
    page = client.get("/").text
    assert "Objective" in page
    assert 'data-tab="vector"' in page
    assert "preview-scrubber" in page


def test_path_stretch_preview_exposes_selected_variant_and_route_provenance() -> None:
    runtime = build_action_runtime(
        stretch_config=StretchConfig(max_turn_deg=120.0),
        stretch_outcome_evaluator=lambda _variant: 1.0,
        runtime_fingerprint="event-queue-dogleg-preview-fixture-v1",
    )
    source_definition = _oracle_simulator().definition
    dogleg_key = ArrivalClusterKey("TEST", "RWY", "C1")
    definition = replace(
        source_definition,
        schema_version="hailmary.scenario.v2",
        flights=tuple(
            replace(
                flight,
                cluster_id=dogleg_key.qualified_id,
                runway="RWY",
            )
            for flight in source_definition.flights
        ),
    )
    sample = ScalingSample(
        scenario=TrafficScenario(
            window=DemandWindow(0.0, 3_600.0),
            scale=1.0,
            replicate=0,
            definition=definition,
            observed_cluster_counts=(ClusterCount(dogleg_key, 3),),
            target_cluster_counts=(ClusterCount(dogleg_key, 3),),
        ),
        original_arrivals=(),
        removed_arrivals=(),
    )
    session = build_event_queue_session(sample, runtime=runtime)
    target = next(
        (frame_index, record.candidate.action_id)
        for frame_index, frame in enumerate(session.frames)
        for record in frame.opportunities
        if record.candidate.lever.value == "path_stretch"
    )

    preview = session.preview(*target)
    audit = preview["objective"]["after"]["realization"]["audit"]
    audit_items = dict(audit["audit"])
    variant = preview["objective"]["after"]["realization"]["variant"]

    assert audit_items["chosen_variant"] in {"short", "medium", "long"}
    assert audit_items["candidate_scores"]
    assert variant["action_provenance"]["lever"] == "path_stretch"
    assert variant["action_provenance"]["added_distance_m"] > 0.0
    assert (
        preview["map"]["after"]["route_history"][0]["variant_id"]
        != preview["map"]["before"]["route_history"][0]["variant_id"]
    )
