from __future__ import annotations

from dataclasses import replace
import json

import pytest

from hailmary.config import LearningConfig, StretchConfig
from hailmary.learning import Phase0ExperimentConfig, Phase0ExperimentRunner
from hailmary.learning.phase0 import _material_scenario_fingerprint
from hailmary.runtime import ActionRuntime, build_action_runtime
from hailmary.scenario import (
    ActionStationDefinition,
    FactorialScenarioBatch,
    FlightGenerationSpec,
    ResourceDefinition,
    ScenarioGenerator,
)

from .test_templates import _straight_variant


_FACTOR_LEVELS = {
    "commitment": (0.30, 0.75),
    "error_magnitude": (10.0, 40.0),
    "pressure": (0.5, 1.5),
    "time_to_final": (300.0, 600.0),
}


def _factorial_batch(
    seed: int,
    prefix: str,
    *,
    extra_flight: bool = False,
    include_path_stretch: bool = True,
    realization_shift: float | None = None,
) -> FactorialScenarioBatch:
    variant = _straight_variant()
    action_stations = (
        ActionStationDefinition(0, 20_000.0, "speed"),
        *(
            (ActionStationDefinition(0, 20_000.0, "path_stretch"),)
            if include_path_stretch
            else ()
        ),
    )
    specs = [
        FlightGenerationSpec(
            flight_id="F0",
            baseline_variant_id=variant.variant_id,
            observed_release_time_s=0.0,
        ),
        FlightGenerationSpec(
            flight_id="F1",
            baseline_variant_id=variant.variant_id,
            observed_release_time_s=60.0,
            action_stations=action_stations,
        ),
    ]
    if extra_flight:
        specs.append(
            FlightGenerationSpec(
                flight_id="F2",
                baseline_variant_id=variant.variant_id,
                observed_release_time_s=180.0,
            )
        )

    shift = (
        float(seed % 7 + 1) if realization_shift is None else float(realization_shift)
    )

    def offset_model(spec, _condition, _rng):
        return 0.0 if spec.flight_id == "F0" else shift

    return ScenarioGenerator(seed).generate_factorial(
        scenario_id_prefix=prefix,
        flight_specs=tuple(specs),
        variants=(variant,),
        resources=(
            ResourceDefinition("RWY", required_interval_s=90.0),
            ResourceDefinition(
                "MERGE",
                kind="merge_fix",
                required_interval_s=90.0,
            ),
        ),
        factor_levels=_FACTOR_LEVELS,
        offset_model=offset_model,
    )


def _learning_config() -> LearningConfig:
    return LearningConfig(
        population_limit=200,
        ga_interval=10_000,
        ga_min_experience=2,
        certification_interval=1,
        action_min_noop_samples=1,
        veto_min_rival_samples=1,
        contender_min_samples=2,
        exploration_coverage_floor=1,
        base_exploration_rate=0.0,
        random_seed=7,
    )


def test_material_fingerprint_ignores_order_and_common_cosmetic_labels() -> None:
    scenario = _factorial_batch(101, "BASE", realization_shift=5.0).scenarios[0]
    definition = scenario.definition
    flight_labels = {
        flight.flight_id: f"COSMETIC_FLIGHT_{index}"
        for index, flight in enumerate(definition.flights)
    }

    def relabel_references(value):
        if isinstance(value, str):
            return flight_labels.get(value, value)
        if isinstance(value, list):
            return [relabel_references(item) for item in value]
        if isinstance(value, dict):
            return {
                flight_labels.get(str(key), str(key)): relabel_references(item)
                for key, item in value.items()
            }
        return value

    relabeled_flights = tuple(
        replace(
            flight,
            flight_id=flight_labels[flight.flight_id],
            cluster_id="COSMETIC_CLUSTER",
            runway="COSMETIC_RUNWAY",
        )
        for flight in definition.flights
    )
    relabeled_events = tuple(
        replace(event, payload=relabel_references(event.payload_dict))
        for event in definition.exogenous_events
    )
    cosmetic = replace(
        scenario,
        definition=replace(
            definition,
            scenario_id="COSMETIC-SCENARIO",
            seed=definition.seed + 10_000,
            flights=tuple(reversed(relabeled_flights)),
            resources=tuple(reversed(definition.resources)),
            variants=tuple(reversed(definition.variants)),
            exogenous_events=tuple(reversed(relabeled_events)),
        ),
    )

    assert _material_scenario_fingerprint(cosmetic) == (
        _material_scenario_fingerprint(scenario)
    )


def test_material_fingerprint_preserves_true_physical_differences() -> None:
    scenario = _factorial_batch(101, "BASE", realization_shift=5.0).scenarios[0]
    flights = scenario.definition.flights
    physically_different = replace(
        scenario,
        definition=replace(
            scenario.definition,
            scenario_id="PHYSICALLY-DIFFERENT",
            flights=(
                flights[0],
                replace(flights[1], release_time_s=flights[1].release_time_s + 1.0),
            ),
        ),
    )

    assert _material_scenario_fingerprint(physically_different) != (
        _material_scenario_fingerprint(scenario)
    )


def test_phase0_runner_rejects_unbalanced_or_overlapping_batches() -> None:
    runtime = build_action_runtime(
        stretch_config=StretchConfig(
            rejoin_span_nm=(4.0, 5.0, 6.0),
            added_distance_nm=(0.5, 1.0, 1.5),
            max_turn_deg=120.0,
        ),
    )
    runner = Phase0ExperimentRunner(
        runtime,
        learning_config=_learning_config(),
        config=Phase0ExperimentConfig(
            seeds=(7,),
            refresh_intervals=(1,),
            run_vanilla=False,
        ),
    )
    batch = _factorial_batch(101, "TRAIN")

    with pytest.raises(ValueError, match="balanced full-factorial"):
        runner._validate_batch(
            FactorialScenarioBatch(
                scenarios=batch.scenarios[:-1],
                correlation_audit=batch.correlation_audit,
            ),
            name="training_batch",
        )

    three_flight_batch = FactorialScenarioBatch(
        scenarios=tuple(
            replace(
                scenario,
                definition=replace(
                    scenario.definition,
                    flights=(
                        *scenario.definition.flights,
                        replace(
                            scenario.definition.flights[-1],
                            flight_id="F2",
                            release_time_s=max(
                                flight.release_time_s
                                for flight in scenario.definition.flights
                            )
                            + 60.0,
                            observed_release_time_s=180.0,
                            action_stations=(),
                        ),
                    ),
                ),
            )
            for scenario in batch.scenarios
        ),
        correlation_audit=batch.correlation_audit,
    )
    with pytest.raises(ValueError, match="exactly one leader-follower pair"):
        runner._validate_batch(three_flight_batch, name="training_batch")

    speed_only_batch = FactorialScenarioBatch(
        scenarios=tuple(
            replace(
                scenario,
                definition=replace(
                    scenario.definition,
                    flights=(
                        scenario.definition.flights[0],
                        replace(
                            scenario.definition.flights[1],
                            action_stations=(
                                scenario.definition.flights[1].action_stations[0],
                            ),
                        ),
                    ),
                ),
            )
            for scenario in batch.scenarios
        ),
        correlation_audit=batch.correlation_audit,
    )
    with pytest.raises(ValueError, match="speed and path-stretch actions"):
        runner._validate_batch(speed_only_batch, name="training_batch")

    cosmetic_batch = FactorialScenarioBatch(
        scenarios=tuple(
            replace(
                scenario,
                definition=replace(
                    scenario.definition,
                    scenario_id=f"COSMETIC-{index}",
                    seed=scenario.definition.seed + 1_000,
                ),
            )
            for index, scenario in enumerate(batch.scenarios)
        ),
        correlation_audit=batch.correlation_audit,
    )
    assert {
        scenario.definition.definition_hash for scenario in batch.scenarios
    }.isdisjoint(
        scenario.definition.definition_hash for scenario in cosmetic_batch.scenarios
    )
    with pytest.raises(ValueError, match="material scenario realizations"):
        runner.run(batch, cosmetic_batch)

    with pytest.raises(ValueError, match="must be disjoint"):
        runner.run(batch, batch)


def test_phase0_runner_executes_real_balanced_causal_and_vanilla_runs() -> None:
    stretch_config = StretchConfig(
        rejoin_span_nm=(4.0, 5.0, 6.0),
        added_distance_nm=(0.5, 1.0, 1.5),
        max_turn_deg=120.0,
    )
    runtime = build_action_runtime(stretch_config=stretch_config)
    geometry_only = build_action_runtime(
        stretch_config=stretch_config,
        stretch_selector="geometry_clearance",
    )
    runner = Phase0ExperimentRunner(
        runtime,
        geometry_only_runtime=geometry_only,
        learning_config=_learning_config(),
        config=Phase0ExperimentConfig(
            seeds=(7,),
            refresh_intervals=(1,),
            training_passes=1,
            run_vanilla=True,
        ),
    )

    result = runner.run(
        _factorial_batch(101, "TRAIN"),
        _factorial_batch(202, "HELD"),
    )

    assert len(result.causal_runs) == 1
    assert len(result.vanilla_runs) == 1
    assert result.causal_runs[0].credit_mode == "causal"
    assert result.vanilla_runs[0].credit_mode == "vanilla_accuracy"
    assert result.causal_runs[0].committed_experiments > 0
    assert result.vanilla_runs[0].committed_experiments > 0
    assert (
        len(result.causal_runs[0].trace_hashes) == result.causal_runs[0].trainer_epoch
    )
    assert (
        len(result.vanilla_runs[0].trace_hashes) == result.vanilla_runs[0].trainer_epoch
    )
    for run in (*result.causal_runs, *result.vanilla_runs):
        assert run.publication_generations
        assert run.publication_generations[-1] == (
            run.rulebook.certification_generation
        )
        assert all(
            generation % run.refresh_interval_epochs == 0
            for generation in run.publication_generations
        )
        assert run.evaluation_snapshot.rulebook == run.rulebook
        assert run.evaluation_snapshot.publication_epoch == (
            run.rulebook.certification_generation
        )
        assert run.evaluation_snapshot.content_hash
    assert len(result.validation_inputs.train_deploy_decisions) == 16
    assert len(result.validation_inputs.held_out_comparisons) == 16
    causal_control = result.validation_inputs.held_out_comparisons
    vanilla_control = result.validation_inputs.vanilla_held_out_comparisons
    assert len(vanilla_control) == 16
    assert tuple(item.scenario_id for item in vanilla_control) == tuple(
        item.scenario_id for item in causal_control
    )
    assert all(item.provenance is not None for item in causal_control)
    assert all(item.provenance is not None for item in vanilla_control)
    assert all(
        causal.provenance is not None
        and vanilla.provenance is not None
        and causal.provenance.permanent_control_identity
        == vanilla.provenance.permanent_control_identity
        for causal, vanilla in zip(causal_control, vanilla_control, strict=True)
    )
    paired_control = result.acceptance_report.check("paired_vanilla_accuracy_control")
    assert paired_control.passed
    assert paired_control.metrics["sample_count"] == 16

    assert result.training_correlation_audit["passed"] is True
    assert result.held_out_correlation_audit["passed"] is True
    # A smoke-sized scientific run is allowed to fail a preregistered gate;
    # the runner must report that honestly instead of manufacturing evidence.
    assert isinstance(result.acceptance_report.passed, bool)

    payload = result.to_dict()
    validation_payload = payload["validation_inputs"]
    assert set(validation_payload) == {
        "train_deploy_decisions",
        "held_out_comparisons",
        "vanilla_held_out_comparisons",
        "path_stretch_ablation",
        "refresh_interval_results",
        "seeded_rule_regions",
        "action_certification_evidence",
    }
    assert payload["causal_runs"][0]["publication_generations"] == list(
        result.causal_runs[0].publication_generations
    )
    assert payload["causal_runs"][0]["publication_count"] == len(
        result.causal_runs[0].publication_generations
    )
    assert payload["causal_runs"][0]["evaluation_snapshot"] == (
        result.causal_runs[0].evaluation_snapshot.to_dict()
    )
    assert payload["causal_runs"][0]["evaluation_snapshot"]["content_hash"] == (
        result.causal_runs[0].evaluation_snapshot.content_hash
    )
    assert payload["causal_runs"][0]["rule_hyperrectangles"] == [
        rectangle.to_dict() for rectangle in result.causal_runs[0].rule_hyperrectangles
    ]
    assert validation_payload["held_out_comparisons"] == [
        item.to_dict() for item in result.validation_inputs.held_out_comparisons
    ]
    assert validation_payload["vanilla_held_out_comparisons"] == [
        item.to_dict() for item in result.validation_inputs.vanilla_held_out_comparisons
    ]
    for comparison in result.validation_inputs.held_out_comparisons:
        provenance = comparison.provenance
        assert provenance is not None
        assert provenance.outcome_plan["root_dynamic_content_hash"] == (
            provenance.parent_dynamic_content_hash
        )
        assert provenance.runtime.configuration_hash == (
            runtime.runtime_configuration_hash
        )
        assert provenance.policy_arm_initial_dynamic_content_hash == (
            provenance.parent_dynamic_content_hash
        )
        assert provenance.permanent_no_op_initial_dynamic_content_hash == (
            provenance.parent_dynamic_content_hash
        )
    assert validation_payload["refresh_interval_results"] == [
        {
            "refresh_interval_epochs": item.refresh_interval_epochs,
            "regime_actions": {
                regime: action.to_dict()
                for regime, action in item.regime_actions.items()
            },
            "publication_count": item.publication_count,
            "publication_generations": list(item.publication_generations),
        }
        for item in result.validation_inputs.refresh_interval_results
    ]
    assert validation_payload["seeded_rule_regions"] == [
        {
            "seed": snapshot.seed,
            "hyperrectangles": [
                rectangle.to_dict() for rectangle in snapshot.hyperrectangles
            ],
        }
        for snapshot in result.validation_inputs.seeded_rule_regions
    ]
    assert validation_payload["action_certification_evidence"] == {
        source_rule_id: {
            "source_rule_id": evidence.source_rule_id,
            "noop_samples": evidence.noop_samples,
            "noop_lcb": evidence.noop_lcb,
        }
        for source_rule_id, evidence in (
            result.validation_inputs.action_certification_evidence.items()
        )
    }
    path_ablation = result.validation_inputs.path_stretch_ablation
    assert validation_payload["path_stretch_ablation"] == (
        None if path_ablation is None else path_ablation.to_dict()
    )
    if path_ablation is not None:
        assert path_ablation.has_authenticated_runtime_provenance
        assert path_ablation.oracle_runtime is not None
        assert path_ablation.geometry_only_runtime is not None
        assert path_ablation.oracle_runtime.configuration_hash == (
            runtime.runtime_configuration_hash
        )
        assert path_ablation.geometry_only_runtime.configuration_hash == (
            geometry_only.runtime_configuration_hash
        )
        assert path_ablation.oracle_runtime.stretch_selector == "semi_local_outcome"
        assert path_ablation.geometry_only_runtime.stretch_selector == (
            "geometry_clearance"
        )
    json.dumps(payload)


def test_phase0_runner_completes_without_vanilla_control() -> None:
    runtime = build_action_runtime(
        stretch_config=StretchConfig(
            rejoin_span_nm=(4.0, 5.0, 6.0),
            added_distance_nm=(0.5, 1.0, 1.5),
            max_turn_deg=120.0,
        ),
    )
    runner = Phase0ExperimentRunner(
        runtime,
        learning_config=_learning_config(),
        config=Phase0ExperimentConfig(
            seeds=(7,),
            refresh_intervals=(1,),
            training_passes=1,
            run_vanilla=False,
        ),
    )

    result = runner.run(
        _factorial_batch(303, "NO-VANILLA-TRAIN"),
        _factorial_batch(404, "NO-VANILLA-HELD"),
    )

    assert result.vanilla_runs == ()
    assert result.validation_inputs.vanilla_held_out_comparisons == ()
    with pytest.raises(KeyError, match="paired_vanilla_accuracy_control"):
        result.acceptance_report.check("paired_vanilla_accuracy_control")
    payload = result.to_dict()
    assert payload["vanilla_runs"] == []
    assert payload["validation_inputs"]["vanilla_held_out_comparisons"] == []
    assert isinstance(payload["acceptance_report"]["passed"], bool)
    json.dumps(payload)


def test_phase0_accepts_geometry_runtime_differing_only_by_selector() -> None:
    stretch_config = StretchConfig(
        rejoin_span_nm=(4.0, 5.0, 6.0),
        added_distance_nm=(0.5, 1.0, 1.5),
        max_turn_deg=120.0,
    )
    oracle = build_action_runtime(stretch_config=stretch_config)
    geometry_only = build_action_runtime(
        stretch_config=stretch_config,
        stretch_selector="geometry_clearance",
    )

    runner = Phase0ExperimentRunner(
        oracle,
        geometry_only_runtime=geometry_only,
    )

    assert runner.geometry_only_runtime is geometry_only
    assert oracle.runtime_configuration_hash != (
        geometry_only.runtime_configuration_hash
    )


def test_phase0_rejects_identical_geometry_runtime_or_manifest() -> None:
    stretch_config = StretchConfig(
        rejoin_span_nm=(4.0, 5.0, 6.0),
        added_distance_nm=(0.5, 1.0, 1.5),
        max_turn_deg=120.0,
    )
    oracle = build_action_runtime(stretch_config=stretch_config)

    with pytest.raises(ValueError, match="must be distinct"):
        Phase0ExperimentRunner(oracle, geometry_only_runtime=oracle)

    separately_built_oracle = build_action_runtime(stretch_config=stretch_config)
    with pytest.raises(ValueError, match="must be distinct"):
        Phase0ExperimentRunner(
            oracle,
            geometry_only_runtime=separately_built_oracle,
        )


@pytest.mark.parametrize("extra_fingerprint", (False, True))
def test_phase0_rejects_non_selector_geometry_runtime_changes(
    extra_fingerprint: bool,
) -> None:
    oracle_stretch = StretchConfig(
        rejoin_span_nm=(4.0, 5.0, 6.0),
        added_distance_nm=(0.5, 1.0, 1.5),
        max_turn_deg=120.0,
    )
    geometry_stretch = (
        oracle_stretch
        if extra_fingerprint
        else StretchConfig(
            rejoin_span_nm=(4.0, 5.0, 6.0),
            added_distance_nm=(0.5, 1.0, 1.5),
            max_turn_deg=119.0,
        )
    )
    oracle = build_action_runtime(stretch_config=oracle_stretch)
    geometry_only = build_action_runtime(
        stretch_config=geometry_stretch,
        stretch_selector="geometry_clearance",
        runtime_fingerprint="unexpected-extra-change" if extra_fingerprint else None,
    )

    with pytest.raises(
        ValueError,
        match="except.*selector='geometry_clearance'",
    ):
        Phase0ExperimentRunner(
            oracle,
            geometry_only_runtime=geometry_only,
        )


def test_phase0_rejects_unauthenticated_geometry_runtime_manifest() -> None:
    class ForgedHashRuntime(ActionRuntime):
        @property
        def runtime_configuration_hash(self) -> str:
            return "0" * 64

    stretch_config = StretchConfig(
        rejoin_span_nm=(4.0, 5.0, 6.0),
        added_distance_nm=(0.5, 1.0, 1.5),
        max_turn_deg=120.0,
    )
    oracle = build_action_runtime(stretch_config=stretch_config)
    geometry_only = build_action_runtime(
        stretch_config=stretch_config,
        stretch_selector="geometry_clearance",
    )
    forged_geometry = ForgedHashRuntime(
        catalog=geometry_only.catalog,
        stretch_realizer=geometry_only.stretch_realizer,
        action_applier=geometry_only.action_applier,
        custom_runtime_fingerprint=geometry_only.custom_runtime_fingerprint,
    )

    with pytest.raises(ValueError, match="does not authenticate"):
        Phase0ExperimentRunner(
            oracle,
            geometry_only_runtime=forged_geometry,
        )


def test_phase0_rejects_geometry_selector_as_primary_without_ablation() -> None:
    stretch_config = StretchConfig(
        rejoin_span_nm=(4.0, 5.0, 6.0),
        added_distance_nm=(0.5, 1.0, 1.5),
        max_turn_deg=120.0,
    )
    geometry_primary = build_action_runtime(
        stretch_config=stretch_config,
        stretch_selector="geometry_clearance",
    )

    with pytest.raises(
        ValueError,
        match="oracle runtime must use selector='semi_local_outcome'",
    ):
        Phase0ExperimentRunner(geometry_primary)
