from __future__ import annotations

import pytest

from hailmary.config import ScenarioConfig
from hailmary.errors import CorrelationGateError
from hailmary.scenario import (
    ActionStationDefinition,
    FlightGenerationSpec,
    ResourceDefinition,
    ScenarioGenerator,
    audit_factor_correlations,
    build_factorial_conditions,
)
from hailmary.features import (
    build_current_leader_follower_anchors,
    simulator_flight_commitment_components,
    simulator_state_vector,
)
from hailmary.simulator import Simulator

from .test_templates import _straight_variant


def test_balanced_factorial_populates_all_quadrants_and_passes_correlation_gate() -> None:
    levels = {
        "commitment": (0.2, 0.8),
        "error_magnitude": (10.0, 40.0),
        "pressure": (0.5, 1.5),
        "time_to_final": (300.0, 900.0),
    }
    conditions = build_factorial_conditions(levels, replicates=2, master_seed=17)

    assert len(conditions) == 32
    for first, second in (
        ("error_magnitude", "time_to_final"),
        ("error_magnitude", "pressure"),
        ("commitment", "pressure"),
        ("commitment", "error_magnitude"),
    ):
        quadrants = {(item.value(first), item.value(second)) for item in conditions}
        assert quadrants == {
            (levels[first][0], levels[second][0]),
            (levels[first][0], levels[second][1]),
            (levels[first][1], levels[second][0]),
            (levels[first][1], levels[second][1]),
        }
    audit = audit_factor_correlations(
        conditions,
        registered_pairs=(("error_magnitude", "pressure"), ("commitment", "time_to_final")),
    )
    assert audit.passed
    assert all(record.correlation == pytest.approx(0.0, abs=1e-12) for record in audit.records)


def test_factorial_scenarios_preserve_observed_release_and_record_offsets() -> None:
    variant = _straight_variant()
    batch = ScenarioGenerator(23).generate_factorial(
        scenario_id_prefix="TRAIN",
        flight_specs=(
            FlightGenerationSpec(
                flight_id="F1",
                baseline_variant_id=variant.variant_id,
                observed_release_time_s=1_000.0,
            ),
        ),
        variants=(variant,),
        resources=(ResourceDefinition("RWY"), ResourceDefinition("MERGE")),
        factor_levels={"release_offset_s": (-30.0, 30.0), "pressure": (0.5, 1.5)},
        replicates=1,
        config=ScenarioConfig(
            registered_factor_pairs=(("release_offset_s", "pressure"),)
        ),
    )

    assert len(batch.scenarios) == 4
    for item in batch.scenarios:
        flight = item.definition.flight("F1")
        assert flight.observed_release_time_s == 1_000.0
        assert flight.release_offset_s in {-30.0, 30.0}
        assert flight.release_time_s == 1_000.0 + flight.release_offset_s


def test_default_factorial_realizer_materializes_every_design_factor() -> None:
    variant = _straight_variant()
    specs = tuple(
        FlightGenerationSpec(
            flight_id=f"F{index}",
            baseline_variant_id=variant.variant_id,
            observed_release_time_s=1_000.0 + 180.0 * index,
            action_stations=(ActionStationDefinition(0, 0.0, "speed"),),
        )
        for index in range(3)
    )
    levels = {
        "commitment": (0.3, 0.75),
        "error_magnitude": (10.0, 40.0),
        "pressure": (0.5, 1.5),
        "time_to_final": (300.0, 600.0),
    }

    batch = ScenarioGenerator(47).generate_factorial(
        scenario_id_prefix="REALIZED",
        flight_specs=specs,
        variants=(variant,),
        resources=(ResourceDefinition("RWY"), ResourceDefinition("MERGE")),
        factor_levels=levels,
    )

    assert len(batch.scenarios) == 16
    assert batch.correlation_audit.passed
    for scenario in batch.scenarios:
        requested = scenario.condition.values_dict
        realized = scenario.realized_values_dict
        assert realized == pytest.approx(requested)
        assert scenario.definition.metadata_dict["factorial_realized_values"] == pytest.approx(
            realized
        )
        assert len(scenario.definition.exogenous_events) == 1
        disturbance = scenario.definition.exogenous_events[0]
        payload = disturbance.payload_dict
        assert payload["effect_kind"] == "factorial_spacing_disturbance"
        assert payload["target_flight_id"] == "F2"
        assert payload["flight_time_shift_s"] == pytest.approx(realized["error_magnitude"])
        assert payload["post_disturbance_threshold_eta_s"] - disturbance.time_s == pytest.approx(
            realized["time_to_final"]
        )
        assert payload["state_updates"]["factorial_values"] == pytest.approx(realized)

        simulator = Simulator(scenario.definition)
        while not simulator.state.exogenous_event_log:
            assert simulator.advance_next() is not None
        assert simulator.state.exogenous_state_dict["factorial_values"] == pytest.approx(realized)
        assert {
            name: simulator.state.exogenous_state_dict[name] for name in realized
        } == pytest.approx(realized)
        target = simulator.state.flight("F2")
        assert target.trajectory_clock_origin_s == pytest.approx(
            scenario.definition.flight("F2").release_time_s + realized["error_magnitude"]
        )
        assert target.predicted_resource_time("RWY") == pytest.approx(
            payload["post_disturbance_threshold_eta_s"]
        )
        computed = simulator_flight_commitment_components(
            simulator,
            "F2",
            resource_id="RWY",
        )
        assert computed.commitment_fraction == pytest.approx(requested["commitment"])
        assert realized["commitment"] == pytest.approx(computed.commitment_fraction)
        assert payload["canonical_commitment"]["commitment_fraction"] == pytest.approx(
            computed.commitment_fraction
        )
        if realized["pressure"] == 1.5 and realized["time_to_final"] == 600.0:
            anchors = build_current_leader_follower_anchors(simulator, resource_id="RWY")
            anchor = next(item for item in anchors.leader_follower if item.follower_id == "F2")
            vector = simulator_state_vector(simulator, anchor)
            assert vector.named["commitment_fraction"] == pytest.approx(
                requested["commitment"]
            )

    low_pressure = next(
        item
        for item in batch.scenarios
        if item.condition.value("pressure") == 0.5
        and item.condition.value("commitment") == 0.3
        and item.condition.value("error_magnitude") == 10.0
        and item.condition.value("time_to_final") == 300.0
    )
    high_pressure = next(
        item
        for item in batch.scenarios
        if item.condition.value("pressure") == 1.5
        and item.condition.value("commitment") == 0.3
        and item.condition.value("error_magnitude") == 10.0
        and item.condition.value("time_to_final") == 300.0
    )
    low_span = low_pressure.definition.flight("F2").release_time_s - low_pressure.definition.flight(
        "F0"
    ).release_time_s
    high_span = high_pressure.definition.flight("F2").release_time_s - high_pressure.definition.flight(
        "F0"
    ).release_time_s
    assert high_span < low_span


def test_factorial_design_factor_domains_are_validated() -> None:
    variant = _straight_variant()
    kwargs = {
        "scenario_id_prefix": "INVALID",
        "flight_specs": (FlightGenerationSpec("F1", variant.variant_id, 0.0),),
        "variants": (variant,),
        "resources": (ResourceDefinition("RWY"), ResourceDefinition("MERGE")),
        "config": ScenarioConfig(registered_factor_pairs=()),
    }

    with pytest.raises(ValueError, match="pressure must be positive"):
        ScenarioGenerator(1).generate_factorial(
            **kwargs,
            factor_levels={"pressure": (-1.0, 1.0)},
        )
    with pytest.raises(ValueError, match="commitment must lie"):
        ScenarioGenerator(1).generate_factorial(
            **kwargs,
            factor_levels={"commitment": (0.2, 1.2)},
        )


def test_factorial_gate_uses_realized_offsets_not_requested_labels() -> None:
    variant = _straight_variant()

    with pytest.raises(CorrelationGateError, match="release_offset_s/pressure"):
        ScenarioGenerator(13).generate_factorial(
            scenario_id_prefix="CORRELATED",
            flight_specs=(FlightGenerationSpec("F1", variant.variant_id, 0.0),),
            variants=(variant,),
            resources=(ResourceDefinition("RWY"), ResourceDefinition("MERGE")),
            factor_levels={
                "release_offset_s": (-30.0, 30.0),
                "pressure": (0.5, 1.5),
            },
            offset_model=lambda _spec, condition, _rng: 100.0 * condition.value("pressure"),
            config=ScenarioConfig(
                registered_factor_pairs=(("release_offset_s", "pressure"),),
            ),
        )


def test_infeasible_canonical_commitment_target_is_rejected() -> None:
    variant = _straight_variant()
    station = (ActionStationDefinition(0, 0.0, "speed"),)

    with pytest.raises(ValueError, match="infeasible at time_to_final"):
        ScenarioGenerator(29).generate_factorial(
            scenario_id_prefix="INFEASIBLE_COMMITMENT",
            flight_specs=(
                FlightGenerationSpec(
                    "F1",
                    variant.variant_id,
                    0.0,
                    action_stations=station,
                ),
            ),
            variants=(variant,),
            resources=(ResourceDefinition("RWY"), ResourceDefinition("MERGE")),
            factor_levels={
                "commitment": (0.05, 0.10),
                "time_to_final": (100.0, 200.0),
            },
            config=ScenarioConfig(registered_factor_pairs=()),
        )
