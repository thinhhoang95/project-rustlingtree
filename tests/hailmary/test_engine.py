from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hailmary.errors import SimulationError
from hailmary.scenario import (
    ActionStationDefinition,
    FlightDefinition,
    FlightGenerationSpec,
    MaterializedExogenousEvent,
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
    ScenarioGenerator,
)
from hailmary.simulator import (
    EventKind,
    FlightLifecycle,
    MonotoneTrajectory,
    Simulator,
    StaleStateError,
)


@dataclass(frozen=True)
class SyntheticVariant:
    variant_id: str
    # Canonical artifact order: threshold to upstream.  Elapsed time therefore
    # decreases in array order and is normalized by MonotoneTrajectory.
    s_m: np.ndarray
    relative_elapsed_time_s: np.ndarray
    east_m: np.ndarray
    altitude_m: np.ndarray
    cas_mps: np.ndarray


def _variant(variant_id: str = "BASE", *, duration_s: float = 20.0) -> SyntheticVariant:
    fractions = np.asarray([1.0, 0.75, 0.25, 0.0], dtype=np.float64)
    arrays = {
        "s_m": np.asarray([0.0, 500.0, 1_500.0, 2_000.0], dtype=np.float64),
        "relative_elapsed_time_s": duration_s * fractions,
        "east_m": np.asarray([0.0, 500.0, 1_500.0, 2_000.0], dtype=np.float64),
        "altitude_m": np.asarray([100.0, 300.0, 700.0, 900.0], dtype=np.float64),
        "cas_mps": np.asarray([65.0, 70.0, 80.0, 90.0], dtype=np.float64),
    }
    for array in arrays.values():
        array.setflags(write=False)
    return SyntheticVariant(variant_id=variant_id, **arrays)


def _definition(*, include_slow_variant: bool = False) -> ScenarioDefinition:
    variants = (_variant(),)
    if include_slow_variant:
        variants = (*variants, _variant("SLOW", duration_s=30.0))
    return ScenarioDefinition(
        scenario_id="SYNTHETIC",
        seed=7,
        flights=(
            FlightDefinition(
                flight_id="F1",
                release_time_s=100.0,
                baseline_variant_id="BASE",
                observed_release_time_s=100.0,
                action_stations=(
                    ActionStationDefinition(
                        station_index=0, s_m=1_500.0, station_type="speed"
                    ),
                    ActionStationDefinition(
                        station_index=1, s_m=500.0, station_type="speed"
                    ),
                ),
                resource_crossings=(
                    ResourceCrossingDefinition(resource_id="RWY", s_m=0.0),
                ),
            ),
        ),
        resources=(ResourceDefinition(resource_id="RWY"),),
        variants=variants,
        exogenous_events=(
            MaterializedExogenousEvent(
                event_id="WX:100",
                time_s=100.0,
                stream_name="weather",
                payload={"wind_east_mps": 2.0},
            ),
        ),
    )


def test_simulator_validates_optional_runtime_configuration_hash() -> None:
    bare = Simulator(_definition())
    configured = Simulator(
        _definition(), runtime_configuration_hash="runtime-config-v1"
    )

    assert bare.runtime_configuration_hash is None
    assert configured.runtime_configuration_hash == "runtime-config-v1"
    with pytest.raises(ValueError, match="exact string"):
        Simulator(_definition(), runtime_configuration_hash=" runtime-config-v1 ")
    with pytest.raises(ValueError, match="cannot be blank"):
        Simulator(_definition(), runtime_configuration_hash="   ")
    with pytest.raises(TypeError, match="must be a string or None"):
        Simulator(
            _definition(),
            runtime_configuration_hash=1,  # type: ignore[arg-type]
        )
    with pytest.raises(
        ValueError,
        match="action_applier requires a non-empty runtime_configuration_hash",
    ):
        Simulator(_definition(), action_applier=lambda _simulator, _action: None)


def test_bare_simulator_accepts_exact_canonical_no_op() -> None:
    simulator = Simulator(_definition())
    before = simulator.dynamic_content_hash

    assert simulator.apply({"lever": "no_op", "band": "no_op"}) is simulator
    assert simulator.dynamic_content_hash == before


@pytest.mark.parametrize(
    "action",
    (
        {"lever": "no_op"},
        {"lever": "noop", "band": "no_op"},
        {"lever": "none", "band": "no_op"},
        {"lever": "NO_OP", "band": "no_op"},
        {"lever": "no_op", "band": "noop"},
    ),
)
def test_bare_simulator_rejects_noncanonical_no_op_aliases(
    action: dict[str, str],
) -> None:
    with pytest.raises(SimulationError, match="no action realization layer"):
        Simulator(_definition()).apply(action)


def test_monotone_interpolation_accepts_canonical_station_order() -> None:
    trajectory = MonotoneTrajectory.from_variant(_variant())

    assert trajectory.duration_s == pytest.approx(20.0)
    assert trajectory.elapsed_at_station(1_500.0) == pytest.approx(5.0)
    assert trajectory.elapsed_at_station(500.0) == pytest.approx(15.0)
    assert trajectory.station_at_elapsed(10.0) == pytest.approx(1_000.0)
    sample = trajectory.sample(10.0)
    assert sample.s_m == pytest.approx(1_000.0)
    assert sample.east_m == pytest.approx(1_000.0)
    assert sample.altitude_m == pytest.approx(500.0)
    assert sample.cas_mps == pytest.approx(75.0)


def test_equal_time_physical_events_are_batched_before_one_decision_epoch() -> None:
    simulator = Simulator(_definition())

    first = simulator.advance_next()

    assert first is not None
    assert first.time_s == 100.0
    assert [event.kind for event in first.events] == [
        EventKind.EXOGENOUS_DISTURBANCE,
        EventKind.FLIGHT_RELEASED,
    ]
    assert first.decision_epoch is not None
    assert first.decision_epoch.epoch_index == 1
    assert first.decision_epoch.trigger_event_ids == ("WX:100", "SYNTHETIC:F1:release")
    assert simulator.state.flight("F1").lifecycle is FlightLifecycle.ACTIVE

    second = simulator.advance_next()
    third = simulator.advance_next()
    final = simulator.advance_next()
    assert second is not None and second.time_s == 105.0
    assert third is not None and third.time_s == 115.0
    assert final is not None and final.time_s == 120.0
    assert [event.kind for event in final.events] == [
        EventKind.RESOURCE_CROSSED,
        EventKind.FLIGHT_COMPLETED,
    ]
    assert final.decision_epoch is not None
    assert simulator.state.flight("F1").crossed_resource_ids == ("RWY",)
    assert simulator.state.flight("F1").lifecycle is FlightLifecycle.COMPLETED
    assert simulator.advance_next() is None


def test_completion_alone_does_not_create_a_decision_epoch() -> None:
    definition = ScenarioDefinition(
        scenario_id="COMPLETION_ONLY",
        seed=1,
        flights=(FlightDefinition("F1", 0.0, "BASE"),),
        resources=(),
        variants=(_variant(),),
    )
    simulator = Simulator(definition)

    release, completion = simulator.run()

    assert release.decision_epoch is not None
    assert [event.kind for event in completion.events] == [EventKind.FLIGHT_COMPLETED]
    assert completion.decision_epoch is None


def test_stale_state_version_is_rejected() -> None:
    simulator = Simulator(_definition())
    stale_version = simulator.state.version
    simulator.advance_next()

    with pytest.raises(StaleStateError, match="stale simulation state version"):
        simulator.schedule_event(
            time_s=110.0,
            kind=EventKind.EXOGENOUS_DISTURBANCE,
            event_id="LATE",
            expected_version=stale_version,
        )


def test_replacing_variant_reschedules_only_bound_flight_future_events() -> None:
    simulator = Simulator(_definition(include_slow_variant=True))
    simulator.advance_next()  # disturbance and release at t=100
    version = simulator.state.version

    simulator.replace_flight_variant(
        "F1",
        "SLOW",
        action_id="slow-at-entry",
        expected_version=version,
    )

    pending = sorted(simulator.state.event_heap, key=lambda event: event.sort_key)
    pending_times = [(event.kind, event.time_s) for event in pending]
    assert pending_times == [
        (EventKind.ACTION_STATION_CROSSED, 107.5),
        (EventKind.ACTION_STATION_CROSSED, 122.5),
        (EventKind.RESOURCE_CROSSED, 130.0),
        (EventKind.FLIGHT_COMPLETED, 130.0),
    ]
    assert simulator.state.flight("F1").current_variant_id == "SLOW"
    assert simulator.state.flight("F1").action_history == ("slow-at-entry",)
    assert simulator.state.action_log_records[0]["to_variant_id"] == "SLOW"


def test_generator_release_jitter_is_order_independent_and_recorded() -> None:
    specs = [
        FlightGenerationSpec("B", "BASE", 200.0),
        FlightGenerationSpec("A", "BASE", 100.0),
    ]
    generator = ScenarioGenerator(master_seed=19)

    first = generator.generate(
        scenario_id="GEN",
        flight_specs=specs,
        variants=(_variant(),),
        resources=(),
        release_jitter_s=10.0,
    )
    second = generator.generate(
        scenario_id="GEN",
        flight_specs=reversed(specs),
        variants=(_variant(),),
        resources=(),
        release_jitter_s=10.0,
    )

    assert first.definition_hash == second.definition_hash
    assert [flight.flight_id for flight in first.flights] == ["A", "B"]
    for flight in first.flights:
        assert flight.observed_release_time_s is not None
        assert flight.release_time_s == pytest.approx(
            flight.observed_release_time_s + flight.release_offset_s
        )


def test_scenario_definition_snapshots_variant_arrays_and_weather_deeply() -> None:
    variant = _variant()
    original_s = variant.s_m.copy()
    weather = {"wind": {"east_mps": [2.0, 3.0]}, "seed": 19}
    definition = ScenarioDefinition(
        scenario_id="IMMUTABLE",
        seed=19,
        flights=(FlightDefinition("F1", 0.0, "BASE"),),
        resources=(),
        variants=(variant,),
        weather=weather,
    )
    definition_hash = definition.definition_hash
    weather_hash = definition.weather_hash
    stored = definition.variant("BASE")

    variant.s_m.setflags(write=True)
    variant.s_m[:] = -1.0
    weather["wind"]["east_mps"][0] = 999.0
    weather["seed"] = 999

    assert np.array_equal(stored.s_m, original_s)
    with pytest.raises(ValueError):
        stored.s_m.setflags(write=True)
    assert definition.weather_payload == {
        "seed": 19,
        "wind": {"east_mps": [2.0, 3.0]},
    }
    assert definition.definition_hash == definition_hash
    assert definition.weather_hash == weather_hash


def test_materialized_exogenous_event_updates_branch_state_and_schedule() -> None:
    definition = ScenarioDefinition(
        scenario_id="DISTURBANCE",
        seed=5,
        flights=(FlightDefinition("F1", 100.0, "BASE"),),
        resources=(),
        variants=(_variant(),),
        exogenous_events=(
            MaterializedExogenousEvent(
                event_id="ERROR",
                time_s=90.0,
                stream_name="factorial_disturbance",
                payload={
                    "target_flight_id": "F1",
                    "flight_time_shift_s": 12.0,
                    "state_updates": {"factorial_values": {"error_magnitude": 12.0}},
                    "metric_deltas": {"injected_error_s": 12.0},
                },
            ),
        ),
    )
    simulator = Simulator(definition)
    initial_hash = simulator.dynamic_content_hash

    batch = simulator.advance_next()

    assert batch is not None
    assert [event.event_id for event in batch.events] == ["ERROR"]
    assert simulator.state.exogenous_state_dict == {
        "factorial_values": {"error_magnitude": 12.0}
    }
    assert simulator.state.metrics_dict == {"injected_error_s": 12.0}
    assert simulator.state.exogenous_event_records[0]["event_id"] == "ERROR"
    assert simulator.state.flight("F1").release_time_s == pytest.approx(112.0)
    assert simulator.state.flight("F1").trajectory_clock_origin_s == pytest.approx(
        112.0
    )
    assert simulator.next_event_time_s == pytest.approx(112.0)
    assert simulator.dynamic_content_hash != initial_hash

    resumed = Simulator.resume(definition, simulator.snapshot())
    assert resumed.dynamic_content_hash == simulator.dynamic_content_hash
    assert resumed.state.exogenous_state == simulator.state.exogenous_state
