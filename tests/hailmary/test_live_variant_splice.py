from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hailmary.actions import (
    ActionCatalog,
    ActionLever,
    PathStretchRealizer,
    apply_action,
)
from hailmary.adapters import SIMAPAdapter
from hailmary.config import StretchConfig
from hailmary.features import resource_station_m
from hailmary.scenario import (
    ActionStationDefinition,
    FlightDefinition,
    MaterializedExogenousEvent,
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
)
from hailmary.simulator import EventKind, MonotoneTrajectory, Simulator

from .test_templates import _straight_variant
from .test_adapters import _ConstantPerformanceBackend, _aircraft_config


def _live_action_simulator() -> Simulator:
    baseline = _straight_variant()
    definition = ScenarioDefinition(
        scenario_id="LIVE_SPLICE",
        seed=41,
        flights=(
            FlightDefinition(
                flight_id="F1",
                release_time_s=0.0,
                baseline_variant_id=baseline.variant_id,
                action_stations=(
                    ActionStationDefinition(0, 90_000.0, "path_stretch"),
                    ActionStationDefinition(1, 80_000.0, "speed"),
                    ActionStationDefinition(2, 60_000.0, "speed"),
                ),
                resource_crossings=(
                    ResourceCrossingDefinition("RWY", 0.0),
                    ResourceCrossingDefinition("MERGE", 75_000.0, station_index=1),
                ),
            ),
        ),
        resources=(
            ResourceDefinition("RWY"),
            ResourceDefinition("MERGE", kind="merge"),
        ),
        variants=(baseline,),
    )
    return Simulator(
        definition,
        action_applier=apply_action,
        runtime_configuration_hash="test-native-apply-action-v1",
    )


def _next_station_batch(simulator: Simulator, station_type: str):
    while True:
        batch = simulator.advance_next()
        if batch is None:
            raise AssertionError("flight completed before the expected action station")
        if any(
            event.kind is EventKind.ACTION_STATION_CROSSED
            and event.payload_dict["station_type"] == station_type
            for event in batch.events
        ):
            return batch


def _assert_same_live_sample(
    before,
    after,
    *,
    same_station: bool = True,
    same_elapsed: bool = True,
) -> None:
    if same_elapsed:
        assert after.elapsed_time_s == pytest.approx(before.elapsed_time_s)
    if same_station:
        assert after.s_m == pytest.approx(before.s_m, abs=1e-7)
    assert after.east_m == pytest.approx(before.east_m, abs=1e-7)
    assert after.north_m == pytest.approx(before.north_m, abs=1e-7)
    assert after.altitude_m == pytest.approx(before.altitude_m, abs=1e-7)
    assert after.cas_mps == pytest.approx(before.cas_mps, abs=1e-9)


def test_live_stretch_then_speed_preserves_splices_and_maps_future_events() -> None:
    simulator = _live_action_simulator()
    catalog = ActionCatalog()

    stretch_batch = _next_station_batch(simulator, "path_stretch")
    before_stretch = simulator.sample_flight("F1")
    stretch = next(
        candidate
        for candidate in catalog.enumerate_for_batch(
            simulator,
            stretch_batch,
            anchor_id="STRETCH_ANCHOR",
            bound_flight_id="F1",
        )
        if candidate.lever is ActionLever.PATH_STRETCH
    )
    stretch_result = apply_action(
        simulator,
        stretch,
        stretch_realizer=PathStretchRealizer(
            config=StretchConfig(max_turn_deg=120.0),
        ),
        stretch_outcome_evaluator=lambda _variant: 1.0,
    )
    after_stretch = simulator.sample_flight("F1")

    _assert_same_live_sample(before_stretch, after_stretch, same_station=False)
    assert simulator.state.sim_time_s == stretch_batch.time_s
    assert simulator.state.flight("F1").release_time_s == 0.0

    stretched = simulator.definition.variant(stretch_result.variant_id)
    metadata = dict(stretched.action_provenance.realization_metadata)
    mapping = np.asarray(metadata["parent_to_variant_station_mapping_m"], dtype=float)
    mapped_speed_station = float(np.interp(80_000.0, mapping[:, 0], mapping[:, 1]))
    mapped_merge_station = float(np.interp(75_000.0, mapping[:, 0], mapping[:, 1]))
    assert mapped_speed_station > 80_000.0
    assert mapped_merge_station > 75_000.0

    pending = sorted(simulator.state.event_heap, key=lambda event: event.sort_key)
    speed_event = next(
        event
        for event in pending
        if event.kind is EventKind.ACTION_STATION_CROSSED and event.station_index == 1
    )
    merge_event = next(
        event
        for event in pending
        if event.kind is EventKind.RESOURCE_CROSSED and event.resource_id == "MERGE"
    )
    assert speed_event.payload_dict["s_m"] == pytest.approx(mapped_speed_station)
    assert merge_event.payload_dict["s_m"] == pytest.approx(mapped_merge_station)
    assert resource_station_m(simulator, "F1", "MERGE") == pytest.approx(
        mapped_merge_station
    )
    trajectory = MonotoneTrajectory.from_variant(stretched)
    origin = simulator.state.flight("F1").trajectory_clock_origin_s
    assert speed_event.time_s == pytest.approx(
        origin + trajectory.elapsed_at_station(mapped_speed_station)
    )
    assert merge_event.time_s == pytest.approx(
        origin + trajectory.elapsed_at_station(mapped_merge_station)
    )
    assert simulator.state.flight("F1").predicted_resource_time(
        "MERGE"
    ) == pytest.approx(merge_event.time_s)

    speed_batch = _next_station_batch(simulator, "speed")
    before_speed = simulator.sample_flight("F1")
    assert before_speed.s_m == pytest.approx(mapped_speed_station)
    light = next(
        candidate
        for candidate in catalog.enumerate_for_batch(
            simulator,
            speed_batch,
            anchor_id="SPEED_ANCHOR",
            bound_flight_id="F1",
        )
        if candidate.lever is ActionLever.SPEED and candidate.band == "light"
    )
    assert light.s_m == pytest.approx(mapped_speed_station)
    speed_result = apply_action(simulator, light)
    after_speed = simulator.sample_flight("F1")

    _assert_same_live_sample(before_speed, after_speed)
    assert speed_result.realized_delay_s > 0.0
    remapped_merge_event = next(
        event
        for event in simulator.state.event_heap
        if event.kind is EventKind.RESOURCE_CROSSED and event.resource_id == "MERGE"
    )
    assert remapped_merge_event.payload_dict["s_m"] == pytest.approx(
        mapped_merge_station
    )
    assert remapped_merge_event.time_s > merge_event.time_s


def test_replacement_rejects_a_splice_mapping_that_moves_the_live_aircraft() -> None:
    simulator = _live_action_simulator()
    stretch_batch = _next_station_batch(simulator, "path_stretch")
    stretch = next(
        candidate
        for candidate in ActionCatalog().enumerate_for_batch(
            simulator,
            stretch_batch,
            anchor_id="STRETCH_ANCHOR",
            bound_flight_id="F1",
        )
        if candidate.lever is ActionLever.PATH_STRETCH
    )
    realized = PathStretchRealizer(config=StretchConfig(max_turn_deg=120.0)).realize(
        simulator.definition.variant(simulator.state.flight("F1").current_variant_id),
        anchor_s_m=stretch.s_m,
        outcome_evaluator=lambda _variant: 1.0,
    )
    simulator.install_variant(realized.variant)

    with pytest.raises(ValueError, match="discontinuous at the live splice"):
        simulator.replace_flight_variant(
            "F1",
            realized.variant.variant_id,
            splice_s_m=stretch.s_m,
            # Identity ignores the added dogleg distance and therefore points
            # at a different physical location in the replacement geometry.
            station_mapping_m=((0.0, 0.0), (100_000.0, 100_000.0)),
            expected_version=simulator.state.version,
        )


def test_active_time_shift_preserves_stretch_mapped_pending_stations() -> None:
    initial = _live_action_simulator()
    definition = replace(
        initial.definition,
        exogenous_events=(
            MaterializedExogenousEvent(
                event_id="DELAY_F1",
                time_s=120.0,
                payload={
                    "target_flight_id": "F1",
                    "flight_time_shift_s": 30.0,
                },
            ),
        ),
    )
    simulator = Simulator(
        definition,
        action_applier=apply_action,
        runtime_configuration_hash="test-native-apply-action-v1",
    )
    stretch_batch = _next_station_batch(simulator, "path_stretch")
    stretch = next(
        candidate
        for candidate in ActionCatalog().enumerate_for_batch(
            simulator,
            stretch_batch,
            anchor_id="STRETCH_ANCHOR",
            bound_flight_id="F1",
        )
        if candidate.lever is ActionLever.PATH_STRETCH
    )
    apply_action(
        simulator,
        stretch,
        stretch_realizer=PathStretchRealizer(config=StretchConfig(max_turn_deg=120.0)),
        stretch_outcome_evaluator=lambda _variant: 1.0,
    )
    before = {
        event.event_id: event
        for event in simulator.state.event_heap
        if event.flight_id == "F1"
    }
    origin_before = simulator.state.flight("F1").trajectory_clock_origin_s

    disturbance_batch = simulator.advance_next()

    assert disturbance_batch is not None
    assert [event.event_id for event in disturbance_batch.events] == ["DELAY_F1"]
    after = {
        event.event_id: event
        for event in simulator.state.event_heap
        if event.flight_id == "F1"
    }
    assert after.keys() == before.keys()
    for event_id, prior in before.items():
        shifted = after[event_id]
        assert shifted.time_s == pytest.approx(prior.time_s + 30.0)
        assert shifted.payload_dict == prior.payload_dict
    dynamic = simulator.state.flight("F1")
    assert dynamic.release_time_s == 0.0
    assert dynamic.trajectory_clock_origin_s == pytest.approx(origin_before + 30.0)
    merge = next(
        event
        for event in after.values()
        if event.kind is EventKind.RESOURCE_CROSSED and event.resource_id == "MERGE"
    )
    assert dynamic.predicted_resource_time("MERGE") == pytest.approx(merge.time_s)


def test_replay_compiled_stretch_preserves_the_live_simap_splice() -> None:
    adapter = SIMAPAdapter(
        aircraft_config=_aircraft_config(),
        performance_backend=_ConstantPerformanceBackend(),
    )
    baseline = adapter.compile(_straight_variant())
    assert baseline.diagnostics.feasible
    simulator = Simulator(
        ScenarioDefinition(
            scenario_id="LIVE_SIMAP_SPLICE",
            seed=43,
            flights=(
                FlightDefinition(
                    flight_id="F1",
                    release_time_s=0.0,
                    baseline_variant_id=baseline.variant_id,
                    action_stations=(
                        ActionStationDefinition(0, 90_000.0, "path_stretch"),
                    ),
                ),
            ),
            resources=(
                ResourceDefinition("RWY"),
                ResourceDefinition("MERGE", kind="merge"),
            ),
            variants=(baseline,),
        )
    )
    batch = _next_station_batch(simulator, "path_stretch")
    before = simulator.sample_flight("F1")
    action = next(
        candidate
        for candidate in ActionCatalog().enumerate_for_batch(
            simulator,
            batch,
            anchor_id="SIMAP_STRETCH",
            bound_flight_id="F1",
        )
        if candidate.lever is ActionLever.PATH_STRETCH
    )

    realization = apply_action(
        simulator,
        action,
        stretch_realizer=PathStretchRealizer(
            config=StretchConfig(max_turn_deg=120.0),
            validator=adapter,
        ),
        stretch_outcome_evaluator=lambda variant: -variant.duration_s,
        variant_validator=adapter,
    )

    after = simulator.sample_flight("F1")
    _assert_same_live_sample(
        before,
        after,
        same_station=False,
        same_elapsed=False,
    )
    compiled = simulator.definition.variant(realization.variant_id)
    assert (
        compiled.diagnostics.message == "public SIMAP coupled replay validation passed"
    )
    envelope = adapter.envelope(compiled.s_m, compiled.altitude_m)
    np.testing.assert_allclose(compiled.lower_cas_mps, envelope.lower_cas_mps)
    np.testing.assert_allclose(compiled.upper_cas_mps, envelope.upper_cas_mps)


def test_replay_compiled_speed_action_preserves_the_live_physical_prefix() -> None:
    adapter = SIMAPAdapter(
        aircraft_config=_aircraft_config(),
        performance_backend=_ConstantPerformanceBackend(),
    )
    baseline = adapter.compile(_straight_variant())
    simulator = Simulator(
        ScenarioDefinition(
            scenario_id="LIVE_SIMAP_SPEED",
            seed=47,
            flights=(
                FlightDefinition(
                    flight_id="F1",
                    release_time_s=0.0,
                    baseline_variant_id=baseline.variant_id,
                    action_stations=(ActionStationDefinition(0, 80_000.0, "speed"),),
                ),
            ),
            resources=(
                ResourceDefinition("RWY"),
                ResourceDefinition("MERGE", kind="merge"),
            ),
            variants=(baseline,),
        )
    )
    batch = _next_station_batch(simulator, "speed")
    before = simulator.sample_flight("F1")
    action = next(
        candidate
        for candidate in ActionCatalog().enumerate_for_batch(
            simulator,
            batch,
            anchor_id="SIMAP_SPEED",
            bound_flight_id="F1",
        )
        if candidate.lever is ActionLever.SPEED and candidate.band == "light"
    )

    realization = apply_action(simulator, action, variant_validator=adapter)

    _assert_same_live_sample(before, simulator.sample_flight("F1"))
    compiled = simulator.definition.variant(realization.variant_id)
    details = dict(compiled.diagnostics.details)
    assert details["live_splice_prefix_source"] == "parent_physical_profile"
    assert details["live_splice_suffix_source"] == "simap_public_coupled_replay"
    assert compiled.duration_s == pytest.approx(
        compiled.diagnostics.compiled_duration_s
    )
