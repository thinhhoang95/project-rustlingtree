from __future__ import annotations

import numpy as np
import pytest

from hailmary.config import M_PER_NM
from hailmary.features import build_current_segment_anchors
from hailmary.geometry import LocalFrame
from hailmary.scenario import (
    FlightDefinition,
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
    SegmentTraversalDefinition,
)
from hailmary.simulator import Simulator
from hailmary.templates import TrajectoryVariant
from hailmary.topology import (
    MedoidRoute,
    RouteGraphArtifact,
    RouteGraphConfig,
    build_route_graph,
)


def _route(
    cluster: str,
    points: tuple[tuple[float, float], ...],
    *,
    runway: str = "RW18R",
) -> MedoidRoute:
    frame = LocalFrame(0.0, 0.0)
    lat, lon = frame.unproject(
        np.asarray([item[0] for item in points]),
        np.asarray([item[1] for item in points]),
    )
    return MedoidRoute(
        dataset_id="phase01-topology-test",
        airport="KATL",
        runway=runway,
        cluster_id=cluster,
        lat_deg=tuple(lat),
        lon_deg=tuple(lon),
    )


def test_continuous_geometry_merges_converged_routes_but_not_parallel_final() -> None:
    graph = build_route_graph(
        (
            _route("C1", ((-30_000.0, 10_000.0), (-16_000.0, 0.0), (0.0, 0.0))),
            _route(
                "C2",
                (
                    (30_000.0, 10_000.0),
                    (-16_000.0, 0.0),
                    (-8_000.0, 0.0),
                    (0.0, 0.0),
                ),
            ),
            _route(
                "C3",
                (
                    (0.0, 30_000.0),
                    (-16_000.0, 5_000.0),
                    (-16_000.0, 0.0),
                    (0.0, 0.0),
                ),
            ),
            _route(
                "P",
                (
                    (30_000.0, 4_000.0),
                    (9_000.0, 4_000.0),
                    (0.0, 4_000.0),
                    (0.0, 0.0),
                ),
            ),
        )
    )
    expected = ("KATL:RW18R:C1", "KATL:RW18R:C2", "KATL:RW18R:C3")
    shared = [item for item in graph.segments if item.cluster_ids == expected]

    assert len(shared) == 1
    assert shared[0].length_m >= 5.0 * 1_852.0
    assert all("KATL:RW18R:P" not in item.cluster_ids for item in shared)
    assert any(item.cluster_ids == ("KATL:RW18R:P",) for item in graph.segments)
    assert RouteGraphArtifact.from_dict(graph.to_dict()).to_dict() == graph.to_dict()

    invalid = graph.to_dict()
    invalid["schema_version"] = "hailmary.route_graph.v1"
    invalid["artifact_content_hash"] = ""
    with pytest.raises(ValueError, match="unsupported route-graph artifact schema"):
        RouteGraphArtifact.from_dict(invalid)

    legacy_config = graph.to_dict()
    legacy_config["config"]["schema_version"] = "hailmary.route_graph.config.v1"
    legacy_config["artifact_content_hash"] = ""
    with pytest.raises(
        ValueError, match="unsupported route-graph configuration schema"
    ):
        RouteGraphArtifact.from_dict(legacy_config)


def test_airport_graph_can_join_split_rejoin_and_split_across_runways() -> None:
    routes = (
        _route(
            "A",
            (
                (-70_000.0, -10_000.0),
                (-60_000.0, 0.0),
                (-45_000.0, 0.0),
                (-40_000.0, -5_000.0),
                (-35_000.0, -5_000.0),
                (-30_000.0, 0.0),
                (-15_000.0, 0.0),
                (0.0, -5_000.0),
            ),
            runway="RW18R",
        ),
        _route(
            "B",
            (
                (-70_000.0, 10_000.0),
                (-60_000.0, 0.0),
                (-45_000.0, 0.0),
                (-40_000.0, 12_000.0),
                (-35_000.0, 12_000.0),
                (-30_000.0, 0.0),
                (-15_000.0, 0.0),
                (0.0, 5_000.0),
            ),
            runway="RW19L",
        ),
    )
    graph = build_route_graph(routes)

    assert build_route_graph(reversed(routes)).to_dict() == graph.to_dict()

    shared = [item for item in graph.segments if len(item.cluster_ids) == 2]
    assert len(shared) == 2
    assert all(item.runway_ids == ("RW18R", "RW19L") for item in shared)
    assert sum(item.kind == "merge" for item in graph.nodes) == 2
    assert sum(item.kind == "split" for item in graph.nodes) == 2

    shared_ranges = []
    for segment in shared:
        records = [
            item for item in graph.traversals if item.segment_id == segment.segment_id
        ]
        assert {item.qualified_cluster_id.split(":")[1] for item in records} == {
            "RW18R",
            "RW19L",
        }
        shared_ranges.append(
            {(round(item.entry_s_m), round(item.exit_s_m)) for item in records}
        )
    # The detour makes the same physical upstream trunk occur at different
    # route-local stations. It must still be one shared resource.
    assert any(len(ranges) == 2 for ranges in shared_ranges)

    for cluster_id in ("KATL:RW18R:A", "KATL:RW19L:B"):
        traversals = graph.traversals_for(cluster_id)
        assert [item.ordinal for item in traversals] == list(range(len(traversals)))
        assert all(
            left.exit_s_m == pytest.approx(right.entry_s_m)
            for left, right in zip(traversals, traversals[1:], strict=False)
        )


def test_complete_link_components_prevent_transitive_proximity_chaining() -> None:
    offset = 0.4 * M_PER_NM
    graph = build_route_graph(
        (
            _route("A", ((-30_000.0, 0.0), (0.0, 0.0)), runway="RW18R"),
            _route(
                "B",
                ((-30_000.0, offset), (0.0, offset)),
                runway="RW19L",
            ),
            _route(
                "C",
                ((-30_000.0, 2.0 * offset), (0.0, 2.0 * offset)),
                runway="RW20R",
            ),
        )
    )

    assert not any(len(item.cluster_ids) == 3 for item in graph.segments)
    assert any(len(item.cluster_ids) == 2 for item in graph.segments)


def test_uncertain_medoid_is_absent_from_route_graph_artifact() -> None:
    certain = _route("CERTAIN", ((-20_000.0, 0.0), (0.0, 0.0)))
    uncertain = MedoidRoute(
        dataset_id=certain.dataset_id,
        airport=certain.airport,
        runway=certain.runway,
        cluster_id="UNCERTAIN",
        lat_deg=certain.lat_deg,
        lon_deg=certain.lon_deg,
        dispersion_m=5.1 * M_PER_NM,
    )

    graph = build_route_graph((certain, uncertain), config=RouteGraphConfig())

    assert {item.qualified_cluster_id for item in graph.traversals} == {
        certain.qualified_cluster_id
    }


def _variant(cluster: str, speed_mps: float) -> TrajectoryVariant:
    stations = np.asarray([0.0, 500.0, 1_500.0, 2_000.0])
    speed = np.full(len(stations), speed_mps)
    return TrajectoryVariant.from_kinematic_profile(
        template_id=f"template:{cluster}:{speed_mps:g}",
        cluster_id=cluster,
        s_m=stations,
        east_m=stations,
        north_m=np.zeros(len(stations)),
        altitude_m=stations,
        cas_mps=speed,
        lower_cas_mps=np.full(len(stations), 40.0),
        upper_cas_mps=np.full(len(stations), 130.0),
        threshold_resource_id="RWY",
        resource_stations_m=(("S:entry", 1_500.0), ("S:exit", 500.0)),
    )


def test_segment_queue_uses_occupancy_then_entry_eta_and_reports_catch_up() -> None:
    fast = _variant("FAST", 100.0)
    slow = _variant("SLOW", 50.0)
    resources = (
        ResourceDefinition("S:entry", kind="segment_entry"),
        ResourceDefinition("S:exit", kind="segment_exit"),
    )
    crossings = (
        ResourceCrossingDefinition("S:entry", 1_500.0),
        ResourceCrossingDefinition("S:exit", 500.0),
    )
    traversal = (
        SegmentTraversalDefinition(0, "S", "S:entry", "S:exit", 1_500.0, 500.0),
    )
    definition = ScenarioDefinition(
        scenario_id="segment-queue",
        seed=5,
        flights=(
            # At t=0, A is physically downstream of Z, but its slower profile
            # predicts a later exit. The queue must not reverse them.
            FlightDefinition(
                "A",
                -24.0,
                slow.variant_id,
                "SLOW",
                resource_crossings=crossings,
                segment_traversals=traversal,
            ),
            FlightDefinition(
                "Z",
                -10.0,
                fast.variant_id,
                "FAST",
                resource_crossings=crossings,
                segment_traversals=traversal,
            ),
            FlightDefinition(
                "B",
                0.0,
                fast.variant_id,
                "FAST",
                resource_crossings=crossings,
                segment_traversals=traversal,
            ),
            FlightDefinition(
                "C",
                0.0,
                fast.variant_id,
                "FAST",
                resource_crossings=crossings,
                segment_traversals=traversal,
            ),
        ),
        resources=resources,
        variants=(fast, slow),
        schema_version="hailmary.scenario.v2",
    )
    simulator = Simulator(definition).run_until(0.0)
    anchors = build_current_segment_anchors(simulator)
    flow = anchors.flow_for_segment("S")
    by_pair = {
        (item.leader_id, item.follower_id): item for item in anchors.leader_follower
    }

    assert flow.ordered_flight_ids == ("A", "Z", "B", "C")
    assert set(by_pair) == {("A", "Z"), ("Z", "B"), ("B", "C")}
    assert by_pair[("A", "Z")].ordering_basis == "physical_progress"
    assert by_pair[("A", "Z")].predicted_exit_interval_s == -1.0
    assert by_pair[("A", "Z")].catch_up is True


def test_segment_awareness_keeps_the_same_pair_on_each_future_corridor() -> None:
    variant = TrajectoryVariant.from_kinematic_profile(
        template_id="template:multi-segment",
        cluster_id="KATL:RW18R:C1",
        s_m=np.asarray([0.0, 500.0, 1_000.0, 1_500.0, 2_000.0]),
        east_m=np.asarray([0.0, 500.0, 1_000.0, 1_500.0, 2_000.0]),
        north_m=np.zeros(5),
        altitude_m=np.asarray([0.0, 500.0, 1_000.0, 1_500.0, 2_000.0]),
        cas_mps=np.full(5, 100.0),
        lower_cas_mps=np.full(5, 40.0),
        upper_cas_mps=np.full(5, 130.0),
        threshold_resource_id="RWY",
        resource_stations_m=(
            ("S1:entry", 1_800.0),
            ("S1:exit", 1_200.0),
            ("S2:entry", 800.0),
            ("S2:exit", 200.0),
        ),
    )
    crossings = tuple(
        ResourceCrossingDefinition(resource_id, station)
        for resource_id, station in (
            ("S1:entry", 1_800.0),
            ("S1:exit", 1_200.0),
            ("S2:entry", 800.0),
            ("S2:exit", 200.0),
        )
    )
    traversals = (
        SegmentTraversalDefinition(0, "S1", "S1:entry", "S1:exit", 1_800.0, 1_200.0),
        SegmentTraversalDefinition(1, "S2", "S2:entry", "S2:exit", 800.0, 200.0),
    )
    scenario = ScenarioDefinition(
        scenario_id="segment-awareness",
        seed=5,
        flights=tuple(
            FlightDefinition(
                flight_id,
                0.0,
                variant.variant_id,
                "KATL:RW18R:C1",
                runway="RW18R",
                resource_crossings=crossings,
                segment_traversals=traversals,
            )
            for flight_id in ("A", "B")
        ),
        resources=tuple(
            ResourceDefinition(resource_id, kind=kind)
            for resource_id, kind in (
                ("S1:entry", "segment_entry"),
                ("S1:exit", "segment_exit"),
                ("S2:entry", "segment_entry"),
                ("S2:exit", "segment_exit"),
            )
        ),
        variants=(variant,),
        schema_version="hailmary.scenario.v2",
    )

    awareness = build_current_segment_anchors(Simulator(scenario).run_until(0.0))
    assert {
        (item.segment_id, item.leader_id, item.follower_id)
        for item in awareness.leader_follower
    } == {("S1", "A", "B"), ("S2", "A", "B")}
