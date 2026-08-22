"""Build a tiny route graph, then inspect live segment queues before/after delay."""

from __future__ import annotations

import json

import numpy as np

from hailmary.actions import ActionCatalog, ActionLever, realize_speed_variant
from hailmary.actions.splice import preserve_compiled_live_prefix
from hailmary.features import build_current_segment_anchors
from hailmary.geometry import LocalFrame
from hailmary.scenario import (
    ArrivalClusterKey,
    DemandWindow,
    ObservedArrival,
    TrafficScaleConfig,
    TrafficScenarioBuilder,
)
from hailmary.simulator import Simulator
from hailmary.templates import ActionStation, ClusterTemplate, TrajectoryVariant
from hailmary.topology import MedoidRoute, build_route_graph


LOCAL_PATHS = {
    "C1": ((-30_000.0, 10_000.0), (-16_000.0, 0.0), (0.0, 0.0)),
    "C2": ((30_000.0, 10_000.0), (-16_000.0, 0.0), (0.0, 0.0)),
    "C3": (
        (0.0, 30_000.0),
        (-16_000.0, 5_000.0),
        (-16_000.0, 0.0),
        (0.0, 0.0),
    ),
    # P remains about 2.16 NM away until its short runway connection. It must
    # not satisfy the registered five-NM sustained-corridor gate.
    "P": ((30_000.0, 4_000.0), (9_000.0, 4_000.0), (0.0, 4_000.0), (0.0, 0.0)),
}


def _resample_path(points: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, ...]:
    path = np.asarray(points, dtype=float)
    progress = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    remaining = progress[-1] - progress
    stations = np.linspace(0.0, float(progress[-1]), 161)
    east = np.interp(stations, remaining[::-1], path[::-1, 0])
    north = np.interp(stations, remaining[::-1], path[::-1, 1])
    return stations, east, north


def _variant(
    cluster_id: str, points: tuple[tuple[float, float], ...], speed: float
) -> TrajectoryVariant:
    stations, east, north = _resample_path(points)
    frame = LocalFrame(0.0, 0.0)
    lat, lon = frame.unproject(east, north)
    speeds = np.full(len(stations), speed)
    return TrajectoryVariant.from_kinematic_profile(
        template_id=f"template:{cluster_id}:{speed:g}",
        cluster_id=cluster_id,
        s_m=stations,
        east_m=east,
        north_m=north,
        lat_deg=lat,
        lon_deg=lon,
        altitude_m=np.linspace(0.0, 3_000.0, len(stations)),
        cas_mps=speeds,
        lower_cas_mps=np.full(len(stations), 60.0),
        upper_cas_mps=np.full(len(stations), 130.0),
        threshold_resource_id="KATL:RW18R:threshold",
    )


def _template(
    cluster_id: str,
    points: tuple[tuple[float, float], ...],
    speed: float,
) -> ClusterTemplate:
    variant = _variant(cluster_id, points, speed)
    station_index = len(variant.s_m) - 1
    station = dict(
        entry_order=0,
        grid_index=station_index,
        s_m=float(variant.s_m[station_index]),
        east_m=float(variant.east_m[station_index]),
        north_m=float(variant.north_m[station_index]),
    )
    return ClusterTemplate(
        cluster_id=cluster_id,
        medoid_flight_id=f"MEDOID:{cluster_id}",
        member_count=1,
        baseline_variant=variant,
        speed_action_stations=(ActionStation(kind="speed", **station),),
        path_stretch_stations=(ActionStation(kind="path_stretch", **station),),
        dataset_id="phase01-topology-example",
        airport_id="KATL",
        runway_id="RW18R",
        expected_speed_station_count=1,
        expected_path_station_count=1,
    )


def _anchor_summary(simulator: Simulator, shared_segment_id: str) -> dict[str, object]:
    anchors = build_current_segment_anchors(simulator)
    flow = anchors.flow_for_segment(shared_segment_id)
    order = {
        flight_id: index for index, flight_id in enumerate(flow.ordered_flight_ids)
    }
    pairs = [
        {
            "leader": item.leader_id,
            "follower": item.follower_id,
            "ordering_basis": item.ordering_basis,
            "exit_interval_s": round(float(item.predicted_exit_interval_s), 3),
            "catch_up": item.catch_up,
        }
        for item in sorted(
            (
                candidate
                for candidate in anchors.leader_follower
                if candidate.segment_id == shared_segment_id
            ),
            key=lambda candidate: order[candidate.leader_id],
        )
    ]
    return {"queue": flow.ordered_flight_ids, "pairs": pairs}


def main() -> None:
    frame = LocalFrame(0.0, 0.0)
    routes = []
    for cluster, points in LOCAL_PATHS.items():
        east = np.asarray([point[0] for point in points])
        north = np.asarray([point[1] for point in points])
        lat, lon = frame.unproject(east, north)
        routes.append(
            MedoidRoute(
                dataset_id="phase01-topology-example",
                airport="KATL",
                runway="RW18R",
                cluster_id=cluster,
                lat_deg=tuple(lat),
                lon_deg=tuple(lon),
            )
        )
    graph = build_route_graph(routes)
    shared = next(
        segment
        for segment in graph.segments
        if segment.cluster_ids == ("KATL:RW18R:C1", "KATL:RW18R:C2", "KATL:RW18R:C3")
    )

    templates = {
        f"KATL:RW18R:{cluster}": _template(cluster, points, 100.0)
        for cluster, points in LOCAL_PATHS.items()
    }
    releases = {
        "A_C3_OCCUPANT": ("C3", -200.0),
        "B_C1_FUTURE": ("C1", 80.0),
        "C_C3_FUTURE": ("C3", -68.6),
        "D_C2_FUTURE": ("C2", -183.3),
        "E_PARALLEL": ("P", 0.0),
    }
    arrivals = []
    for flight_id, (cluster, release) in releases.items():
        key = ArrivalClusterKey("KATL", "RW18R", cluster)
        arrivals.append(
            ObservedArrival(
                dataset_id="phase01-topology-example",
                key=key,
                flight_id=flight_id,
                terminal_entry_time_s=release,
                terminal_entry_ground_speed_mps=100.0,
                terminal_entry_altitude_m=3_000.0,
                baseline_variant_id=templates[
                    key.qualified_id
                ].baseline_variant.variant_id,
            )
        )
    scenario = TrafficScenarioBuilder(
        arrivals,
        templates_by_cluster=templates,
        route_graph=graph,
    ).build_scenario(
        DemandWindow(-300.0, 3_300.0),
        scale_config=TrafficScaleConfig(global_scale=1.0),
    )
    probe = Simulator(scenario.definition)
    first_trainable_epoch: dict[str, object] | None = None
    while first_trainable_epoch is None:
        batch = probe.advance_next()
        if batch is None:
            raise RuntimeError("example scenario produced no trainable segment epoch")
        if batch.decision_epoch is None:
            continue
        for anchor in build_current_segment_anchors(probe).leader_follower:
            candidates = ActionCatalog().enumerate_for_batch(
                probe,
                batch,
                anchor_id=anchor.anchor_id,
                bound_flight_id=anchor.follower_id,
                resource_id=anchor.resource_id,
                segment_id=anchor.segment_id,
            )
            physical = [
                item for item in candidates if item.lever is not ActionLever.NO_OP
            ]
            if physical:
                first_trainable_epoch = {
                    "time_s": probe.state.sim_time_s,
                    "leader": anchor.leader_id,
                    "follower": anchor.follower_id,
                    "segment_id": anchor.segment_id,
                    "candidate_actions": [
                        f"{item.lever.value}:{item.band}" for item in candidates
                    ],
                }
                break
    simulator = Simulator(scenario.definition).run_until(200.0)
    before_membership = tuple(
        item.segment_id
        for item in simulator.definition.flight("C_C3_FUTURE").segment_traversals
    )
    before = _anchor_summary(simulator, shared.segment_id)

    current_variant = simulator.definition.variant(
        simulator.state.flight("C_C3_FUTURE").current_variant_id
    )
    current_station = simulator.sample_flight("C_C3_FUTURE").s_m
    realized_variant = realize_speed_variant(
        current_variant,
        anchor_s_m=current_station,
        band="example_delay",
        reduction_kts=55.0,
    )
    slow_variant = preserve_compiled_live_prefix(
        current_variant,
        realized_variant,
        parent_anchor_s_m=current_station,
        child_anchor_s_m=current_station,
    )
    simulator.install_variant(slow_variant)
    simulator.replace_flight_variant(
        "C_C3_FUTURE",
        slow_variant.variant_id,
        action_id="example-speed-delay",
        action_lever="speed",
        expected_version=simulator.state.version,
    )
    after = _anchor_summary(simulator, shared.segment_id)
    after_membership = tuple(
        item.segment_id
        for item in simulator.definition.flight("C_C3_FUTURE").segment_traversals
    )
    variant_crossing_ids = {
        item.resource_id
        for item in scenario.definition.variant(
            scenario.definition.flight("C_C3_FUTURE").baseline_variant_id
        ).resource_crossings
    }
    required_crossing_ids = {
        resource_id
        for traversal in scenario.definition.flight("C_C3_FUTURE").segment_traversals
        for resource_id in (
            traversal.entry_resource_id,
            traversal.exit_resource_id,
        )
    }

    parallel_segments = [
        item.segment_id
        for item in graph.segments
        if item.cluster_ids == ("KATL:RW18R:P",)
    ]
    result = {
        "route_graph_hash": graph.artifact_content_hash,
        "registered_thresholds": {
            "lateral_floor_nm": graph.config.lateral_floor_nm,
            "tangent_tolerance_deg": graph.config.tangent_tolerance_deg,
            "minimum_common_length_nm": graph.config.minimum_common_length_nm,
        },
        "shared_segment": {
            "segment_id": shared.segment_id,
            "clusters": shared.cluster_ids,
            "length_nm": round(shared.length_m / 1_852.0, 3),
        },
        "parallel_route": {
            "segment_ids": parallel_segments,
            "appears_in_shared_segment": "KATL:RW18R:P" in shared.cluster_ids,
        },
        "at_time_s": 200.0,
        "first_trainable_epoch": first_trainable_epoch,
        "before_speed_delay": before,
        "after_speed_delay": after,
        "segment_membership_unchanged": before_membership == after_membership,
        "segment_membership": before_membership,
        "variant_segment_crossings_complete": (
            required_crossing_ids <= variant_crossing_ids
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
