from __future__ import annotations

from dataclasses import replace
import json

import pytest

from hailmary.actions import (
    ActionCatalog,
    ActionLever,
    PathStretchRealizer,
    apply_action,
)
from hailmary.config import StretchConfig
from hailmary.errors import InfeasibleActionError, StaleActionError
from hailmary.scenario import (
    ActionStationDefinition,
    FlightDefinition,
    ResourceDefinition,
    ScenarioDefinition,
)
from hailmary.simulator import EventKind, Simulator
from hailmary.runtime import build_action_runtime

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
        resources=(
            ResourceDefinition(resource_id="RWY"),
            ResourceDefinition(resource_id="MERGE"),
        ),
        variants=(variant,),
    )
    return Simulator(
        definition,
        action_applier=action_applier,
        runtime_configuration_hash=(
            None if action_applier is None else "test-applier-v1"
        ),
    )


def _next_station_batch(simulator: Simulator):
    while True:
        batch = simulator.advance_next()
        if batch is None:
            raise AssertionError("scenario completed before another action station")
        if any(
            event.kind is EventKind.ACTION_STATION_CROSSED for event in batch.events
        ):
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
    assert all(
        item.epoch_index == simulator.state.decision_epoch_index for item in candidates
    )


def test_runtime_factory_executes_real_speed_and_stretch_on_sibling_forks() -> None:
    definition = _simulator_with_stations(
        (
            ActionStationDefinition(0, 80_000.0, "speed"),
            ActionStationDefinition(0, 80_000.0, "path_stretch"),
        )
    ).definition
    runtime = build_action_runtime(
        stretch_config=StretchConfig(max_turn_deg=120.0),
        stretch_outcome_evaluator=lambda _variant: 1.0,
        runtime_fingerprint="constant-stretch-evaluator-v1",
    )
    parent = runtime.create_simulator(definition)
    batch = _next_station_batch(parent)
    candidates = runtime.catalog.enumerate_for_batch(
        parent,
        batch,
        anchor_id="ANCHOR",
        bound_flight_id="F1",
    )
    speed = _candidate(candidates, ActionLever.SPEED, "light")
    stretch = _candidate(candidates, ActionLever.PATH_STRETCH)
    parent_hash = parent.dynamic_content_hash

    speed_child = parent.fork(label="runtime-speed")
    speed_child.apply(speed)
    speed_result = speed_child.last_action_result
    assert speed_result is not None
    assert speed_result.variant_id is not None
    speed_variant = speed_child.definition.variant(speed_result.variant_id)
    assert speed_variant.action_provenance.lever == ActionLever.SPEED.value
    assert speed_variant.action_provenance.band == "light"

    stretch_child = parent.fork(label="runtime-stretch")
    stretch_child.apply(stretch)
    stretch_result = stretch_child.last_action_result
    assert stretch_result is not None
    assert stretch_result.variant_id is not None
    stretch_variant = stretch_child.definition.variant(stretch_result.variant_id)
    assert stretch_variant.action_provenance.lever == ActionLever.PATH_STRETCH.value
    assert stretch_variant.action_provenance.band == "short"
    assert dict(stretch_result.audit)["chosen_variant"] == "short"

    assert parent.dynamic_content_hash == parent_hash
    assert (
        parent.state.flight("F1").current_variant_id
        == definition.variants[0].variant_id
    )


@pytest.mark.parametrize(
    ("lever", "band"),
    (
        (ActionLever.NO_OP, "bad"),
        (ActionLever.PATH_STRETCH, "short"),
    ),
)
def test_apply_action_rejects_forged_identity_outside_catalog_vocabulary(
    lever: ActionLever,
    band: str,
) -> None:
    simulator = _simulator_with_stations(
        (ActionStationDefinition(0, 80_000.0, "path_stretch"),)
    )
    batch = _next_station_batch(simulator)
    candidate = _candidate(
        ActionCatalog().enumerate_for_batch(
            simulator,
            batch,
            anchor_id="ANCHOR",
            bound_flight_id="F1",
        ),
        lever,
    )
    forged = replace(candidate, band=band, action_id="")
    parent_hash = simulator.dynamic_content_hash

    with pytest.raises(
        InfeasibleActionError,
        match=rf"unsupported action identity {lever.value}/{band}",
    ):
        apply_action(simulator, forged)

    assert simulator.dynamic_content_hash == parent_hash


def test_candidate_becomes_stale_after_any_state_transition() -> None:
    simulator = _simulator_with_stations(
        (ActionStationDefinition(0, 80_000.0, "speed"),)
    )
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


def test_applied_speed_action_is_branch_local_and_parent_remains_byte_equivalent() -> (
    None
):
    parent = _simulator_with_stations((ActionStationDefinition(0, 80_000.0, "speed"),))
    parent_snapshot = json.dumps(
        parent.snapshot(), sort_keys=True, separators=(",", ":")
    )
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

    assert (
        json.dumps(parent.snapshot(), sort_keys=True, separators=(",", ":"))
        == parent_snapshot
    )
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
