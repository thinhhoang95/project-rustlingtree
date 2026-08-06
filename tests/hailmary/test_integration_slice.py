from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hailmary.actions import ActionCatalog, apply_action
from hailmary.actions.models import ActionLever
from hailmary.config import StretchConfig
from hailmary.evaluation import simulator_intervention_summary, simulator_outcome_plan
from hailmary.features import (
    active_resource_predictions,
    build_current_segment_anchors,
    simulator_state_vector,
)
from hailmary.rollout import (
    NoOpPolicy,
    paired_simulator_rollout,
    policy_fingerprint,
    three_arm_simulator_rollout,
)
from hailmary.runtime import build_action_runtime
from hailmary.scenario import (
    ActionStationDefinition,
    FlightDefinition,
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
    SegmentTraversalDefinition,
)
from hailmary.simulator import EventKind, Simulator
from hailmary.templates import TrajectoryVariant

from .test_stretch_oracle import _oracle_simulator


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
    crossings = (
        ResourceCrossingDefinition(resource_id="FINAL:entry", s_m=15_000.0),
        ResourceCrossingDefinition(resource_id="RWY", s_m=0.0),
    )
    traversals = (
        SegmentTraversalDefinition(
            0, "FINAL", "FINAL:entry", "RWY", 15_000.0, 0.0
        ),
    )
    return ScenarioDefinition(
        scenario_id="INTEGRATION_SLICE",
        seed=17,
        flights=(
            FlightDefinition(
                flight_id="LEADER",
                release_time_s=0.0,
                baseline_variant_id=variant.variant_id,
                cluster_id="SYNTHETIC_CLUSTER",
                resource_crossings=crossings,
                segment_traversals=traversals,
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
                resource_crossings=crossings,
                segment_traversals=traversals,
            ),
        ),
        resources=(
            ResourceDefinition(resource_id="RWY", required_interval_s=90.0),
            ResourceDefinition(resource_id="FINAL:entry", kind="segment_entry"),
        ),
        variants=(variant,),
    )


def _advance_to_action_station(
    simulator: Simulator,
    *,
    flight_id: str = "FOLLOWER",
):
    while True:
        batch = simulator.advance_next()
        assert batch is not None
        if any(
            event.kind is EventKind.ACTION_STATION_CROSSED
            and event.flight_id == flight_id
            for event in batch.events
        ):
            return batch


def test_real_leader_follower_vector_and_paired_action_slice() -> None:
    parent = Simulator(
        _scenario(),
        action_applier=apply_action,
        runtime_configuration_hash="test-native-apply-action-v1",
    )
    while True:
        batch = parent.advance_next()
        assert batch is not None
        if batch.time_s == 160.0:
            break

    parent_hash = parent.dynamic_content_hash
    predictions = active_resource_predictions(parent, resource_id="RWY")
    assert [(item.flight_id, item.eta_s) for item in predictions] == [
        ("LEADER", 200.0),
        ("FOLLOWER", 260.0),
    ]
    assert predictions[1].sample.s_m == 10_000.0

    resource_anchors = build_current_segment_anchors(parent)
    anchor = resource_anchors.leader_follower[0]
    assert (anchor.leader_id, anchor.follower_id) == ("LEADER", "FOLLOWER")
    candidates = ActionCatalog().enumerate_for_batch(
        parent,
        batch,
        anchor_id=anchor.anchor_id,
        bound_flight_id=anchor.follower_id,
        resource_id=anchor.resource_id,
        segment_id=anchor.segment_id,
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
    assert (
        result.selected.applied_action.state_id
        != result.contender.applied_action.state_id
    )
    assert (
        result.selected.action_audit.action.action_id
        == result.selected.applied_action.action_id
    )
    assert result.selected.action_audit.realized_delay_s > 0.0
    assert result.contender.action_audit.realized_delay_s == 0.0
    assert result.selected.outcome.intervention_penalty > 0.0
    assert result.contender.outcome.intervention_penalty == 0.0
    assert "intervention_components" in result.selected.outcome.diagnostics
    assert (
        result.selected.final_dynamic_content_hash
        != result.contender.final_dynamic_content_hash
    )
    assert parent.dynamic_content_hash == parent_hash
    assert parent.state.sim_time_s == 160.0
    assert (
        parent.state.flight("FOLLOWER").current_variant_id
        == parent.definition.variants[0].variant_id
    )


def test_identical_real_speed_arms_have_equal_content_and_zero_delta() -> None:
    parent = Simulator(
        _scenario(),
        action_applier=apply_action,
        runtime_configuration_hash="test-native-apply-action-v1",
    )
    while True:
        batch = parent.advance_next()
        assert batch is not None
        if batch.time_s == 160.0:
            break
    anchor = build_current_segment_anchors(parent).leader_follower[0]
    heavy = next(
        item
        for item in ActionCatalog().enumerate_for_batch(
            parent,
            batch,
            anchor_id=anchor.anchor_id,
            bound_flight_id=anchor.follower_id,
            resource_id=anchor.resource_id,
            segment_id=anchor.segment_id,
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
    assert (
        result.selected.applied_action.action_id
        == result.contender.applied_action.action_id
    )
    assert (
        result.selected.final_dynamic_content_hash
        == result.contender.final_dynamic_content_hash
    )


def test_native_three_arm_speed_rollout_uses_one_runtime_policy_and_parent() -> None:
    runtime = build_action_runtime()
    parent = runtime.create_simulator(_scenario())
    batch = _advance_to_action_station(parent)
    anchor = build_current_segment_anchors(parent).leader_follower[0]
    candidates = runtime.catalog.enumerate_for_batch(
        parent,
        batch,
        anchor_id=anchor.anchor_id,
        bound_flight_id=anchor.follower_id,
        resource_id=anchor.resource_id,
        segment_id=anchor.segment_id,
    )
    no_op = next(item for item in candidates if item.lever is ActionLever.NO_OP)
    heavy = next(
        item
        for item in candidates
        if item.lever is ActionLever.SPEED and item.band == "heavy"
    )
    plan = simulator_outcome_plan(parent, anchor)
    policy = NoOpPolicy()
    frozen_policy_hash = policy_fingerprint(policy)
    parent_hash = parent.dynamic_content_hash

    result = three_arm_simulator_rollout(
        parent,
        selected_action=heavy,
        contender_action=no_op,
        no_op_action=no_op,
        frozen_policy=policy,
        outcome_plan=plan,
    )

    assert result.parent_dynamic_content_hash == parent_hash
    assert result.policy_fingerprint == frozen_policy_hash
    assert policy_fingerprint(policy) == frozen_policy_hash
    assert {
        result.selected.initial_dynamic_content_hash,
        result.contender.initial_dynamic_content_hash,
        result.no_op.initial_dynamic_content_hash,
    } == {parent_hash}
    assert result.delta_rival == result.selected.score - result.contender.score
    assert result.delta_selected_noop == result.selected.score - result.no_op.score
    assert result.delta_contender_noop == result.contender.score - result.no_op.score
    assert result.delta_veto == result.no_op.score - max(
        result.selected.score,
        result.contender.score,
    )
    assert (
        result.selected.final_dynamic_content_hash
        != result.no_op.final_dynamic_content_hash
    )
    assert (
        result.contender.final_dynamic_content_hash
        == result.no_op.final_dynamic_content_hash
    )
    assert result.selected.action_audit.action.lever is ActionLever.SPEED
    assert result.contender.action_audit.action.lever is ActionLever.NO_OP
    assert result.no_op.action_audit.action.lever is ActionLever.NO_OP
    assert parent.dynamic_content_hash == parent_hash
    assert parent.runtime_configuration_hash == runtime.runtime_configuration_hash


def test_native_three_arm_stretch_rollout_uses_real_runtime_realizer() -> None:
    runtime = build_action_runtime(
        stretch_config=StretchConfig(max_turn_deg=120.0),
        stretch_outcome_evaluator=lambda _variant: 1.0,
        runtime_fingerprint="native-three-arm-stretch-fixture-v1",
    )
    parent = runtime.create_simulator(_oracle_simulator().definition)
    batch = _advance_to_action_station(parent)
    anchor = next(
        item
        for item in build_current_segment_anchors(parent).leader_follower
        if item.follower_id == "FOLLOWER"
    )
    candidates = runtime.catalog.enumerate_for_batch(
        parent,
        batch,
        anchor_id=anchor.anchor_id,
        bound_flight_id=anchor.follower_id,
        resource_id=anchor.resource_id,
        segment_id=anchor.segment_id,
    )
    no_op = next(item for item in candidates if item.lever is ActionLever.NO_OP)
    stretch = next(
        item for item in candidates if item.lever is ActionLever.PATH_STRETCH
    )
    plan = simulator_outcome_plan(parent, anchor)
    policy = NoOpPolicy()
    frozen_policy_hash = policy_fingerprint(policy)
    parent_hash = parent.dynamic_content_hash

    result = three_arm_simulator_rollout(
        parent,
        selected_action=stretch,
        contender_action=no_op,
        no_op_action=no_op,
        frozen_policy=policy,
        outcome_plan=plan,
    )

    assert result.parent_dynamic_content_hash == parent_hash
    assert result.policy_fingerprint == frozen_policy_hash
    assert policy_fingerprint(policy) == frozen_policy_hash
    assert {
        result.selected.initial_dynamic_content_hash,
        result.contender.initial_dynamic_content_hash,
        result.no_op.initial_dynamic_content_hash,
    } == {parent_hash}
    assert result.selected.action_audit.action.band == "oracle_short_medium_long"
    assert dict(result.selected.action_audit.audit)["chosen_variant"] == "short"
    assert (
        result.selected.final_dynamic_content_hash
        != result.no_op.final_dynamic_content_hash
    )
    assert (
        result.contender.final_dynamic_content_hash
        == result.no_op.final_dynamic_content_hash
    )
    assert parent.dynamic_content_hash == parent_hash
    assert parent.runtime_configuration_hash == runtime.runtime_configuration_hash


def test_real_action_lineage_charges_only_post_root_interventions() -> None:
    definition = _scenario()
    follower = replace(
        definition.flights[1],
        action_stations=(
            ActionStationDefinition(0, 15_000.0, "speed"),
            ActionStationDefinition(1, 10_000.0, "speed"),
        ),
    )
    definition = replace(
        definition,
        scenario_id="INTEGRATION_LINEAGE_BASELINE",
        flights=(definition.flights[0], follower),
    )
    runtime = build_action_runtime()
    parent = runtime.create_simulator(definition)

    first_batch = _advance_to_action_station(parent)
    first_anchor = next(
        item
        for item in build_current_segment_anchors(parent).leader_follower
        if item.follower_id == "FOLLOWER"
    )
    first_candidates = runtime.catalog.enumerate_for_batch(
        parent,
        first_batch,
        anchor_id=first_anchor.anchor_id,
        bound_flight_id=first_anchor.follower_id,
        resource_id=first_anchor.resource_id,
        segment_id=first_anchor.segment_id,
    )
    historical_action = next(
        item
        for item in first_candidates
        if item.lever is ActionLever.SPEED and item.band == "light"
    )
    parent.apply(historical_action)
    historical = simulator_intervention_summary(parent)
    assert historical.action_count == 1
    assert historical.speed_action_count == 1
    assert historical.total_speed_reduction_kts == pytest.approx(10.0)

    second_batch = _advance_to_action_station(parent)
    second_anchor = next(
        item
        for item in build_current_segment_anchors(parent).leader_follower
        if item.follower_id == "FOLLOWER"
    )
    candidates = runtime.catalog.enumerate_for_batch(
        parent,
        second_batch,
        anchor_id=second_anchor.anchor_id,
        bound_flight_id=second_anchor.follower_id,
        resource_id=second_anchor.resource_id,
        segment_id=second_anchor.segment_id,
    )
    no_op = next(item for item in candidates if item.lever is ActionLever.NO_OP)
    post_root_action = next(
        item
        for item in candidates
        if item.lever is ActionLever.SPEED and item.band == "medium"
    )
    plan = simulator_outcome_plan(parent, second_anchor)
    assert plan.baseline_intervention_summary == historical

    result = three_arm_simulator_rollout(
        parent,
        selected_action=post_root_action,
        contender_action=no_op,
        no_op_action=no_op,
        frozen_policy=NoOpPolicy(),
        outcome_plan=plan,
    )

    assert result.contender.outcome.intervention_penalty == 0.0
    assert result.no_op.outcome.intervention_penalty == 0.0
    assert result.selected.outcome.intervention_penalty == pytest.approx(13.0 / 36.0)
    assert result.selected.action_audit.intervention_magnitude == pytest.approx(15.0)
    assert result.selected.outcome.diagnostics[
        "intervention_components"
    ] == pytest.approx(
        {
            "count_component": 1.0 / 3.0,
            "speed_magnitude_component": 0.75,
            "stretch_magnitude_component": 0.0,
        }
    )
    assert simulator_intervention_summary(parent) == historical


def test_continuation_policy_interventions_contribute_to_rollout_cost() -> None:
    definition = _scenario()
    follower = replace(
        definition.flights[1],
        action_stations=(
            ActionStationDefinition(0, 15_000.0, "speed"),
            ActionStationDefinition(1, 10_000.0, "speed"),
        ),
    )
    definition = replace(
        definition,
        scenario_id="INTEGRATION_CONTINUATION_COST",
        flights=(definition.flights[0], follower),
    )
    runtime = build_action_runtime()
    parent = runtime.create_simulator(definition)
    first_batch = _advance_to_action_station(parent)
    first_anchor = next(
        item
        for item in build_current_segment_anchors(parent).leader_follower
        if item.follower_id == "FOLLOWER"
    )
    first_candidates = runtime.catalog.enumerate_for_batch(
        parent,
        first_batch,
        anchor_id=first_anchor.anchor_id,
        bound_flight_id=first_anchor.follower_id,
        resource_id=first_anchor.resource_id,
        segment_id=first_anchor.segment_id,
    )
    no_op = next(item for item in first_candidates if item.lever is ActionLever.NO_OP)
    plan = simulator_outcome_plan(parent, first_anchor)

    class LaterLightPolicy:
        def policy_fingerprint(self) -> str:
            return f"later-light:{runtime.runtime_configuration_hash}"

        def select_action(self, context):
            later_anchor = next(
                (
                    item
                    for item in build_current_segment_anchors(
                        context.simulator
                    ).leader_follower
                    if item.follower_id == "FOLLOWER"
                ),
                None,
            )
            if later_anchor is None:
                return None
            later_candidates = runtime.catalog.enumerate_for_batch(
                context.simulator,
                context.event_batch,
                anchor_id=later_anchor.anchor_id,
                bound_flight_id=later_anchor.follower_id,
                resource_id=later_anchor.resource_id,
                segment_id=later_anchor.segment_id,
            )
            return next(
                item
                for item in later_candidates
                if item.lever is ActionLever.SPEED and item.band == "light"
            )

    continued = three_arm_simulator_rollout(
        parent,
        selected_action=no_op,
        contender_action=no_op,
        no_op_action=no_op,
        frozen_policy=LaterLightPolicy(),
        outcome_plan=plan,
    )
    no_later_action = three_arm_simulator_rollout(
        parent,
        selected_action=no_op,
        contender_action=no_op,
        no_op_action=no_op,
        frozen_policy=NoOpPolicy(),
        outcome_plan=plan,
    )

    expected_penalty = (1.0 / 3.0 + 10.0 / 20.0) / 3.0
    assert (
        continued.selected.outcome.intervention_penalty,
        continued.contender.outcome.intervention_penalty,
        continued.no_op.outcome.intervention_penalty,
    ) == pytest.approx((expected_penalty,) * 3)
    assert no_later_action.selected.outcome.intervention_penalty == 0.0
    assert continued.selected.outcome.intervention_term < 0.0
    assert parent.dynamic_content_hash == plan.root_dynamic_content_hash
