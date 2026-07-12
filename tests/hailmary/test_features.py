from __future__ import annotations

import numpy as np
import pytest

from hailmary.config import FeatureConfig, ScenarioConfig
from hailmary.features.anchors import build_resource_anchors, order_flights_by_eta
from hailmary.features.minimum_time import ReachabilityMap, compute_reachability_map
from hailmary.features.schema import leader_follower_feature_schema
from hailmary.features.state_vector import (
    LeaderFollowerFeatureInputs,
    commitment_components,
    derive_leader_follower_state_vector,
    pressure_in_half_open_window,
)


def test_threshold_ordering_uses_eta_then_stable_flight_id() -> None:
    eta_by_flight = {"B": 100.0, "A": 100.0, "C": 190.0}

    assert order_flights_by_eta(eta_by_flight) == ("A", "B", "C")
    anchors = build_resource_anchors(
        resource_id="RW17L",
        eta_by_flight=eta_by_flight,
        state_version="state-v1",
        epoch=4,
    )
    assert anchors.flow.ordered_flight_ids == ("A", "B", "C")
    assert [(edge.leader_id, edge.follower_id) for edge in anchors.leader_follower] == [
        ("A", "B"),
        ("B", "C"),
    ]
    assert anchors.leader_follower[0].anchor_id != anchors.leader_follower[1].anchor_id


def test_reachability_capacity_uses_only_supplied_current_station_actions() -> None:
    reachability = compute_reachability_map(
        eta_nominal_s=500.0,
        eta_earliest_s=470.0,
        speed_action_etas_s=(507.0, 515.0),
        path_action_etas_s=(),
    )

    assert reachability.eta_latest_speed_s == 515.0
    assert reachability.speed_capacity_s == 15.0
    assert reachability.eta_latest_path_s == 500.0
    assert reachability.path_capacity_s == 0.0


def test_pressure_window_is_half_open() -> None:
    count, capacity, ratio = pressure_in_half_open_window(
        (99.999, 100.0, 699.999, 700.0),
        sim_time_s=100.0,
        window_s=600.0,
        required_interval_s=90.0,
    )

    assert count == 2
    assert capacity == pytest.approx(600.0 / 90.0)
    assert ratio == pytest.approx(2.0 / (600.0 / 90.0))


def test_commitment_uses_declared_transparent_formula() -> None:
    components = commitment_components(
        time_to_threshold_s=600.0,
        remaining_station_fraction=0.5,
        remaining_intervention_budget_fraction=1.0 / 3.0,
        intercept_or_final_gate_flag=1.0,
    )

    assert components.time_component == pytest.approx(0.5)
    assert components.freedom_remaining == pytest.approx(0.5 * 0.5 + 0.5 * (1.0 / 3.0))
    assert components.freedom_component == pytest.approx(1.0 - components.freedom_remaining)
    assert components.commitment_fraction == pytest.approx(
        (components.time_component + components.freedom_component + 1.0) / 3.0
    )


def test_state_vector_is_finite_versioned_and_masks_zero_capacity() -> None:
    reachability = ReachabilityMap(
        eta_nominal_s=260.0,
        eta_earliest_s=240.0,
        eta_latest_speed_s=260.0,
        eta_latest_path_s=290.0,
    )
    inputs = LeaderFollowerFeatureInputs(
        sim_time_s=100.0,
        leader_eta_s=200.0,
        follower_eta_s=260.0,
        required_interval_s=90.0,
        follower_distance_to_resource_m=15_000.0,
        leader_distance_to_resource_m=5_000.0,
        follower_cas_kts=180.0,
        follower_cas_lower_kts=145.0,
        reachability=reachability,
        intercept_or_final_gate_flag=1.0,
        remaining_action_station_count=12,
        total_action_station_count=24,
        remaining_intervention_budget=1,
        total_intervention_budget=3,
        live_nominal_etas_s=(99.0, 100.0, 699.999, 700.0),
        trailing_spacing_margins_s=(),
        cluster_index=7,
    )

    vector = derive_leader_follower_state_vector(
        inputs,
        feature_config=FeatureConfig(),
        scenario_config=ScenarioConfig(),
    )

    assert vector.schema.schema_version == "hailmary.features.leader_follower.v1"
    assert vector.schema_hash == leader_follower_feature_schema().schema_hash
    assert vector.values.dtype == np.float64
    assert np.isfinite(vector.values).all()
    assert not vector.values.flags.writeable
    assert vector.value("spacing_deviation_s") == -30.0
    assert vector.value("required_delay_s") == 30.0
    assert vector.value("speed_capacity_undefined_mask") == 1.0
    assert vector.value("path_capacity_undefined_mask") == 0.0
    assert vector.value("required_delay_over_speed_capacity") == 10.0
    assert vector.value("required_delay_over_path_capacity") == 1.0
    assert vector.value("trailing_spacing_undefined_mask") == 1.0
    assert vector.diagnostics["pressure"]["count"] == 2


def test_reachability_rejects_speedup_as_delay_capacity() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        compute_reachability_map(
            eta_nominal_s=500.0,
            eta_earliest_s=480.0,
            speed_action_etas_s=(499.0,),
        )
