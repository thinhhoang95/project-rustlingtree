from __future__ import annotations

from dataclasses import replace

import pytest

from hailmary.actions import ActionCatalog, ActionLever, PathStretchRealizer, apply_action
from hailmary.actions.catalog import _future_conflicts
from hailmary.config import StretchConfig
from hailmary.features import build_current_leader_follower_anchors
from hailmary.scenario import (
    ActionStationDefinition,
    FlightDefinition,
    MaterializedExogenousEvent,
    ResourceDefinition,
    ScenarioDefinition,
)
from hailmary.simulator import EventKind, Simulator

from .test_templates import _straight_variant


def _oracle_simulator() -> Simulator:
    variant = _straight_variant()
    definition = ScenarioDefinition(
        scenario_id="STRETCH_ORACLE",
        seed=29,
        flights=(
            FlightDefinition(
                flight_id="LEADER",
                release_time_s=0.0,
                baseline_variant_id=variant.variant_id,
            ),
            FlightDefinition(
                flight_id="FOLLOWER",
                release_time_s=60.0,
                baseline_variant_id=variant.variant_id,
                action_stations=(
                    ActionStationDefinition(0, 90_000.0, "path_stretch"),
                ),
            ),
            FlightDefinition(
                flight_id="TRAILER",
                release_time_s=150.0,
                baseline_variant_id=variant.variant_id,
            ),
        ),
        resources=(
            ResourceDefinition(resource_id="RWY", required_interval_s=90.0),
            ResourceDefinition(
                resource_id="MERGE",
                kind="merge_fix",
                required_interval_s=90.0,
            ),
        ),
        variants=(variant,),
    )
    return Simulator(definition)


def _oracle_simulator_with_future_shift() -> Simulator:
    baseline = _oracle_simulator()
    definition = replace(
        baseline.definition,
        exogenous_events=(
            MaterializedExogenousEvent(
                event_id="FUTURE_FOLLOWER_DELAY",
                time_s=500.0,
                payload={
                    "target_flight_id": "FOLLOWER",
                    "flight_time_shift_s": 120.0,
                },
            ),
        ),
    )
    return Simulator(definition)


def test_default_stretch_selector_runs_identical_no_later_action_rollouts() -> None:
    simulator = _oracle_simulator()
    parent_hash_before = simulator.dynamic_content_hash
    while True:
        batch = simulator.advance_next()
        assert batch is not None
        if any(
            event.kind is EventKind.ACTION_STATION_CROSSED
            and event.flight_id == "FOLLOWER"
            for event in batch.events
        ):
            break

    parent_hash_at_epoch = simulator.dynamic_content_hash
    anchor = next(
        item
        for item in build_current_leader_follower_anchors(simulator).leader_follower
        if item.follower_id == "FOLLOWER"
    )
    action = next(
        item
        for item in ActionCatalog().enumerate_for_batch(
            simulator,
            batch,
            anchor_id=anchor.anchor_id,
            bound_flight_id="FOLLOWER",
        )
        if item.lever is ActionLever.PATH_STRETCH
    )

    realization = apply_action(
        simulator,
        action,
        stretch_realizer=PathStretchRealizer(
            config=StretchConfig(max_turn_deg=120.0),
        ),
    )

    audit = dict(realization.audit)
    scores = dict(audit["candidate_scores"])
    diagnostics = dict(audit["candidate_diagnostics"])
    assert set(scores) == {"short", "medium", "long"}
    assert all(score is not None for score in scores.values())
    assert audit["chosen_variant"] in scores
    assert scores[audit["chosen_variant"]] == max(scores.values())
    assert all(
        dict(values)["later_interventions_suppressed"] is True
        for values in diagnostics.values()
    )
    assert simulator.state.flight("FOLLOWER").path_stretch_count == 1
    assert len(simulator.state.action_log) == 1
    assert simulator.dynamic_content_hash != parent_hash_at_epoch
    assert parent_hash_before != parent_hash_at_epoch


def test_conflict_window_remains_frozen_after_inner_branch_reaches_horizon() -> None:
    simulator = _oracle_simulator()
    while True:
        batch = simulator.advance_next()
        assert batch is not None
        if any(
            event.kind is EventKind.ACTION_STATION_CROSSED
            and event.flight_id == "FOLLOWER"
            for event in batch.events
        ):
            break
    decision_time_s = simulator.state.sim_time_s
    horizon_s = 1_300.0

    simulator.run_until(horizon_s)
    records = _future_conflicts(
        simulator,
        start_time_s=decision_time_s,
        horizon_s=horizon_s,
    )

    assert records
    assert all(record.end_time_s < horizon_s for record in records)
    assert any(record.end_time_s > decision_time_s for record in records)


def test_default_selector_accumulates_conflicts_before_a_future_time_shift() -> None:
    simulator = _oracle_simulator_with_future_shift()
    while True:
        batch = simulator.advance_next()
        assert batch is not None
        if any(
            event.kind is EventKind.ACTION_STATION_CROSSED
            and event.flight_id == "FOLLOWER"
            for event in batch.events
        ):
            break
    decision_time_s = simulator.state.sim_time_s
    anchor = next(
        item
        for item in build_current_leader_follower_anchors(simulator).leader_follower
        if item.follower_id == "FOLLOWER"
    )
    action = next(
        item
        for item in ActionCatalog().enumerate_for_batch(
            simulator,
            batch,
            anchor_id=anchor.anchor_id,
            bound_flight_id="FOLLOWER",
        )
        if item.lever is ActionLever.PATH_STRETCH
    )

    realization = apply_action(
        simulator,
        action,
        stretch_realizer=PathStretchRealizer(
            config=StretchConfig(max_turn_deg=120.0),
        ),
    )

    diagnostics = dict(dict(realization.audit)["candidate_diagnostics"])
    for candidate_diagnostics in diagnostics.values():
        values = dict(candidate_diagnostics)
        baseline_durations = dict(values["baseline_conflict_pair_durations_s"])
        # Leader/follower are 60 seconds (6 km) apart until the disturbance at
        # t=500.  The frozen comparison starts at t=160, yielding 340 seconds
        # of conflict that a post-horizon reconstruction would erase.
        assert baseline_durations[("FOLLOWER", "LEADER")] == pytest.approx(340.0)
        assert values["later_interventions_suppressed"] is True
    assert simulator.state.sim_time_s == decision_time_s
    assert simulator.state.exogenous_event_log == ()
