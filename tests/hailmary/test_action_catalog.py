from __future__ import annotations

import json

import pytest

from hailmary.actions import ActionCatalog, ActionLever, PathStretchRealizer, apply_action
from hailmary.config import StretchConfig
from hailmary.errors import StaleActionError
from hailmary.scenario import (
    ActionStationDefinition,
    FlightDefinition,
    ResourceDefinition,
    ScenarioDefinition,
)
from hailmary.simulator import EventKind, Simulator

from .test_templates import _straight_variant


def _simulator_with_stations(
    stations: tuple[ActionStationDefinition, ...],
    *,
    action_applier=None,
) -> Simulator:
    variant = _straight_variant()
    definition = ScenarioDefinition(
        scenario_id="ACTIONS",
        seed=23,
        flights=(
            FlightDefinition(
                flight_id="F1",
                release_time_s=0.0,
                baseline_variant_id=variant.variant_id,
                action_stations=stations,
            ),
        ),
        resources=(ResourceDefinition(resource_id="RWY"), ResourceDefinition(resource_id="MERGE")),
        variants=(variant,),
    )
    return Simulator(definition, action_applier=action_applier)


def _next_station_batch(simulator: Simulator):
    while True:
        batch = simulator.advance_next()
        if batch is None:
            raise AssertionError("scenario completed before another action station")
        if any(event.kind is EventKind.ACTION_STATION_CROSSED for event in batch.events):
            return batch


def _candidate(candidates, lever: ActionLever, band: str | None = None):
    return next(
        item
        for item in candidates
        if item.lever is lever and (band is None or item.band == band)
    )


def test_catalog_enumerates_noop_three_speed_bands_and_one_stretch_at_epoch() -> None:
    simulator = _simulator_with_stations(
        (
            ActionStationDefinition(0, 80_000.0, "speed"),
            ActionStationDefinition(0, 80_000.0, "path_stretch"),
        )
    )
    batch = _next_station_batch(simulator)

    candidates = ActionCatalog().enumerate_for_batch(
        simulator,
        batch,
        anchor_id="ANCHOR",
        bound_flight_id="F1",
    )

    assert [(item.lever, item.band) for item in candidates] == [
        (ActionLever.NO_OP, "no_op"),
        (ActionLever.SPEED, "light"),
        (ActionLever.SPEED, "medium"),
        (ActionLever.SPEED, "heavy"),
        (ActionLever.PATH_STRETCH, "oracle_short_medium_long"),
    ]
    assert all(item.state_id == simulator.state.state_id for item in candidates)
    assert all(item.epoch_index == simulator.state.decision_epoch_index for item in candidates)


def test_candidate_becomes_stale_after_any_state_transition() -> None:
    simulator = _simulator_with_stations((ActionStationDefinition(0, 80_000.0, "speed"),))
    batch = _next_station_batch(simulator)
    candidate = _candidate(
        ActionCatalog().enumerate_for_batch(
            simulator,
            batch,
            anchor_id="ANCHOR",
            bound_flight_id="F1",
        ),
        ActionLever.SPEED,
        "light",
    )
    simulator.add_metric("unrelated", 1.0)

    with pytest.raises(StaleActionError, match="stale simulation state"):
        apply_action(simulator, candidate)


def test_two_speed_actions_are_composed_and_third_station_offers_no_speed() -> None:
    simulator = _simulator_with_stations(
        (
            ActionStationDefinition(0, 80_000.0, "speed"),
            ActionStationDefinition(1, 60_000.0, "speed"),
            ActionStationDefinition(2, 40_000.0, "speed"),
        )
    )
    catalog = ActionCatalog()

    first_batch = _next_station_batch(simulator)
    first = _candidate(
        catalog.enumerate_for_batch(
            simulator, first_batch, anchor_id="A1", bound_flight_id="F1"
        ),
        ActionLever.SPEED,
        "light",
    )
    first_realization = apply_action(simulator, first)

    second_batch = _next_station_batch(simulator)
    second = _candidate(
        catalog.enumerate_for_batch(
            simulator, second_batch, anchor_id="A2", bound_flight_id="F1"
        ),
        ActionLever.SPEED,
        "medium",
    )
    second_realization = apply_action(simulator, second)

    assert first_realization.variant_id is not None
    assert second_realization.variant_id is not None
    current = simulator.definition.variant(second_realization.variant_id)
    first_variant = simulator.definition.variant(first_realization.variant_id)
    assert current.action_provenance.parent_variant_id == first_variant.variant_id
    assert simulator.state.flight("F1").speed_action_count == 2

    third_batch = _next_station_batch(simulator)
    third_candidates = catalog.enumerate_for_batch(
        simulator,
        third_batch,
        anchor_id="A3",
        bound_flight_id="F1",
    )
    assert [(item.lever, item.band) for item in third_candidates] == [
        (ActionLever.NO_OP, "no_op")
    ]


def test_applied_speed_action_is_branch_local_and_parent_remains_byte_equivalent() -> None:
    parent = _simulator_with_stations((ActionStationDefinition(0, 80_000.0, "speed"),))
    parent_snapshot = json.dumps(parent.snapshot(), sort_keys=True, separators=(",", ":"))
    parent_variant = parent.definition.variants[0]
    child = parent.fork(label="temporary-action")
    batch = _next_station_batch(child)
    speed = _candidate(
        ActionCatalog().enumerate_for_batch(
            child,
            batch,
            anchor_id="CHILD",
            bound_flight_id="F1",
        ),
        ActionLever.SPEED,
        "light",
    )

    apply_action(child, speed)

    assert json.dumps(parent.snapshot(), sort_keys=True, separators=(",", ":")) == parent_snapshot
    assert len(parent.definition.variants) == 1
    assert parent.definition.variants[0] is parent_variant
    assert parent.state.flight("F1").current_variant_id == parent_variant.variant_id
    assert len(child.definition.variants) == 2
    assert child.state.flight("F1").current_variant_id != parent_variant.variant_id


def test_one_path_stretch_exhausts_later_path_stretch_candidates() -> None:
    simulator = _simulator_with_stations(
        (
            ActionStationDefinition(0, 90_000.0, "path_stretch"),
            ActionStationDefinition(1, 80_000.0, "path_stretch"),
        )
    )
    catalog = ActionCatalog()
    first_batch = _next_station_batch(simulator)
    stretch = _candidate(
        catalog.enumerate_for_batch(
            simulator,
            first_batch,
            anchor_id="P1",
            bound_flight_id="F1",
        ),
        ActionLever.PATH_STRETCH,
    )
    realization = apply_action(
        simulator,
        stretch,
        stretch_realizer=PathStretchRealizer(config=StretchConfig(max_turn_deg=120.0)),
        stretch_outcome_evaluator=lambda _variant: 1.0,
    )

    assert realization.variant_id is not None
    assert simulator.state.flight("F1").path_stretch_count == 1
    second_batch = _next_station_batch(simulator)
    later = catalog.enumerate_for_batch(
        simulator,
        second_batch,
        anchor_id="P2",
        bound_flight_id="F1",
    )
    assert [(item.lever, item.band) for item in later] == [(ActionLever.NO_OP, "no_op")]
