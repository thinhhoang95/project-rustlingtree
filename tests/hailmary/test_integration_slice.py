from __future__ import annotations

import numpy as np

from hailmary.actions import ActionCatalog, apply_action
from hailmary.actions.models import ActionLever
from hailmary.evaluation import simulator_outcome_plan
from hailmary.features import (
    active_threshold_predictions,
    build_current_leader_follower_anchors,
    simulator_state_vector,
)
from hailmary.rollout import NoOpPolicy, paired_simulator_rollout
from hailmary.scenario import (
    ActionStationDefinition,
    FlightDefinition,
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
)
from hailmary.simulator import Simulator
from hailmary.templates import TrajectoryVariant


def _baseline_variant() -> TrajectoryVariant:
    s_m = np.asarray([0.0, 5_000.0, 10_000.0, 15_000.0, 20_000.0])
    count = len(s_m)
    command = np.full(count, 100.0)
    return TrajectoryVariant.from_kinematic_profile(
        template_id="SYNTHETIC_TEMPLATE",
        cluster_id="SYNTHETIC_CLUSTER",
        s_m=s_m,
        east_m=s_m,
        north_m=np.zeros(count),
        altitude_m=np.zeros(count),
        cas_mps=command,
        tas_mps=command,
        ground_speed_mps=command,
        command_cas_mps=command,
        reference_command_cas_mps=command,
        lower_cas_mps=np.full(count, 70.0),
        upper_cas_mps=np.full(count, 120.0),
        threshold_resource_id="RWY",
    )


def _scenario() -> ScenarioDefinition:
    variant = _baseline_variant()
    threshold = (ResourceCrossingDefinition(resource_id="RWY", s_m=0.0),)
    return ScenarioDefinition(
        scenario_id="INTEGRATION_SLICE",
        seed=17,
        flights=(
            FlightDefinition(
                flight_id="LEADER",
                release_time_s=0.0,
                baseline_variant_id=variant.variant_id,
                cluster_id="SYNTHETIC_CLUSTER",
                resource_crossings=threshold,
            ),
            FlightDefinition(
                flight_id="FOLLOWER",
                release_time_s=60.0,
                baseline_variant_id=variant.variant_id,
                cluster_id="SYNTHETIC_CLUSTER",
                action_stations=(
                    ActionStationDefinition(
                        station_index=0,
                        s_m=10_000.0,
                        station_type="speed",
                    ),
                ),
                resource_crossings=threshold,
            ),
        ),
        resources=(ResourceDefinition(resource_id="RWY", required_interval_s=90.0),),
        variants=(variant,),
    )


def test_real_leader_follower_vector_and_paired_action_slice() -> None:
    parent = Simulator(_scenario(), action_applier=apply_action)
    while True:
        batch = parent.advance_next()
        assert batch is not None
        if batch.time_s == 160.0:
            break

    parent_hash = parent.dynamic_content_hash
    predictions = active_threshold_predictions(parent)
    assert [(item.flight_id, item.eta_s) for item in predictions] == [
        ("LEADER", 200.0),
        ("FOLLOWER", 260.0),
    ]
    assert predictions[1].sample.s_m == 10_000.0

    resource_anchors = build_current_leader_follower_anchors(parent)
    anchor = resource_anchors.leader_follower[0]
    assert (anchor.leader_id, anchor.follower_id) == ("LEADER", "FOLLOWER")
    candidates = ActionCatalog().enumerate_for_batch(
        parent,
        batch,
        anchor_id=anchor.anchor_id,
        bound_flight_id=anchor.follower_id,
    )
    no_op = next(item for item in candidates if item.lever is ActionLever.NO_OP)
    heavy = next(
        item
        for item in candidates
        if item.lever is ActionLever.SPEED and item.band == "heavy"
    )

    vector = simulator_state_vector(parent, anchor, action_candidates=candidates)
    assert np.isfinite(vector.values).all()
    assert vector.named["spacing_deviation_s"] == -30.0
    assert vector.named["required_delay_s"] == 30.0
    assert vector.named["speed_capacity_s"] > 0.0
    assert vector.named["speed_capacity_undefined_mask"] == 0.0
    assert parent.dynamic_content_hash == parent_hash

    plan = simulator_outcome_plan(parent, anchor)
    assert plan.cohort.effective_trailer_count == 0
    assert plan.horizon_s == 350.0
    result = paired_simulator_rollout(
        parent,
        selected_action=heavy,
        contender_action=no_op,
        frozen_policy=NoOpPolicy(),
        outcome_plan=plan,
    )

    assert result.delta > 0.0
    assert result.selected.score > result.contender.score
    assert result.selected.initial_dynamic_content_hash == parent_hash
    assert result.contender.initial_dynamic_content_hash == parent_hash
    assert result.selected.applied_action.state_id != result.contender.applied_action.state_id
    assert result.selected.action_audit.action.action_id == result.selected.applied_action.action_id
    assert result.selected.action_audit.realized_delay_s > 0.0
    assert result.contender.action_audit.realized_delay_s == 0.0
    assert result.selected.outcome.intervention_penalty > 0.0
    assert result.contender.outcome.intervention_penalty == 0.0
    assert "intervention_components" in result.selected.outcome.diagnostics
    assert result.selected.final_dynamic_content_hash != result.contender.final_dynamic_content_hash
    assert parent.dynamic_content_hash == parent_hash
    assert parent.state.sim_time_s == 160.0
    assert parent.state.flight("FOLLOWER").current_variant_id == parent.definition.variants[0].variant_id


def test_identical_real_speed_arms_have_equal_content_and_zero_delta() -> None:
    parent = Simulator(_scenario(), action_applier=apply_action)
    while True:
        batch = parent.advance_next()
        assert batch is not None
        if batch.time_s == 160.0:
            break
    anchor = build_current_leader_follower_anchors(parent).leader_follower[0]
    heavy = next(
        item
        for item in ActionCatalog().enumerate_for_batch(
            parent,
            batch,
            anchor_id=anchor.anchor_id,
            bound_flight_id=anchor.follower_id,
        )
        if item.lever is ActionLever.SPEED and item.band == "heavy"
    )
    plan = simulator_outcome_plan(parent, anchor)

    result = paired_simulator_rollout(
        parent,
        selected_action=heavy,
        contender_action=heavy,
        frozen_policy=NoOpPolicy(),
        outcome_plan=plan,
    )

    assert result.delta == 0.0
    assert result.selected.applied_action.action_id == result.contender.applied_action.action_id
    assert result.selected.final_dynamic_content_hash == result.contender.final_dynamic_content_hash
