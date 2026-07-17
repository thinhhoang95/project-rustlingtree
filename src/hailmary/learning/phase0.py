"""Executable, evidence-producing Phase-0 experiment orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from itertools import product
import json
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from hailmary.config import (
    FeatureConfig,
    LearningConfig,
    OutcomeConfig,
    ScenarioConfig,
)
from hailmary.evaluation.outcome import simulator_outcome_plan
from hailmary.features import build_current_leader_follower_anchors
from hailmary.ids import canonical_data, canonical_json, content_hash
from hailmary.learning.artifacts import EvaluationSnapshot
from hailmary.learning.credit import (
    CausalCreditAssigner,
    VanillaAccuracyCreditAssigner,
)
from hailmary.learning.modes import (
    CAUSAL_CREDIT_MODE,
    VANILLA_ACCURACY_CREDIT_MODE,
    validate_credit_mode,
)
from hailmary.learning.rulebook import (
    FrozenRulebookPolicy,
    RulebookDecisionRecord,
    SimulatorRulebookPolicy,
)
from hailmary.learning.rules import RuleAction
from hailmary.learning.trainer import CausalTrainer
from hailmary.learning.validation import (
    ActionCertificationEvidence,
    AuthenticatedRuntimeManifest,
    CAPACITY_RATIO_FEATURES,
    HeldOutComparison,
    HeldOutOutcomeDetails,
    HeldOutRolloutProvenance,
    PATH_STRETCH_ACTION_KEY,
    PathStretchAblationResult,
    Phase0AcceptanceReport,
    Phase0ValidationInputs,
    RefreshIntervalResult,
    RuleHyperrectangle,
    RuleRegion,
    SeededRuleRegions,
    TrainDeployDecision,
    evaluate_phase0_acceptance,
)
from hailmary.rollout import policy_vs_permanent_no_op_simulator_rollout
from hailmary.runtime import ActionRuntime
from hailmary.scenario import FactorialScenario, FactorialScenarioBatch


_REQUIRED_FACTORS = (
    "commitment",
    "error_magnitude",
    "pressure",
    "time_to_final",
)
_ACTION_RUNTIME_HASH_NAMESPACE = "hailmary.action_runtime.v1"
_OUTCOME_PLAN_HASH_NAMESPACE = "hailmary.phase0.outcome_plan.v1"
_ORACLE_STRETCH_SELECTOR = "semi_local_outcome"
_GEOMETRY_ONLY_STRETCH_SELECTOR = "geometry_clearance"


def _material_event_payload(event: Any) -> Any:
    """Drop design-only identifiers from an otherwise physical event payload."""

    payload = canonical_data(event.payload_dict)
    if not isinstance(payload, dict):
        return payload
    updates = payload.get("state_updates")
    if isinstance(updates, dict) and "factorial_condition_id" in updates:
        payload = dict(payload)
        physical_updates = dict(updates)
        physical_updates.pop("factorial_condition_id", None)
        payload["state_updates"] = physical_updates
    return payload


def _replace_identifier_references(value: Any, labels: Mapping[str, str]) -> Any:
    """Recursively normalize exact identifier references in canonical payloads."""

    if isinstance(value, str):
        return labels.get(value, value)
    if isinstance(value, list):
        return [_replace_identifier_references(item, labels) for item in value]
    if isinstance(value, dict):
        return {
            labels.get(str(key), str(key)): _replace_identifier_references(item, labels)
            for key, item in value.items()
        }
    return value


def _labels_from_physical_signatures(
    signatures: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, str]:
    """Assign labels from material signatures, collapsing symmetric identities."""

    serialized = {
        identifier: canonical_json(value) for identifier, value in signatures.items()
    }
    ranks = {
        signature: index
        for index, signature in enumerate(sorted(set(serialized.values())))
    }
    return {
        identifier: f"{prefix}_{ranks[signature]}"
        for identifier, signature in serialized.items()
    }


def _variant_identifier(variant: Any) -> str:
    payload = canonical_data(variant)
    if isinstance(payload, dict):
        for name in ("variant_id", "content_hash", "id"):
            if payload.get(name):
                return str(payload[name])
    for name in ("variant_id", "content_hash", "id"):
        value = getattr(variant, name, None)
        if value:
            return str(value)
    raise ValueError("scenario variant has no identifier")


def _material_scenario_fingerprint(scenario: FactorialScenario) -> str:
    """Hash a canonically relabeled, order-independent physical realization."""

    definition = scenario.definition

    resource_signatures = {
        resource.resource_id: {
            "kind": resource.kind,
            "required_interval_s": resource.required_interval_s,
            "metadata": canonical_data(resource.metadata_dict),
        }
        for resource in definition.resources
    }
    resource_labels = _labels_from_physical_signatures(
        resource_signatures,
        prefix="resource",
    )

    preliminary_variants: dict[str, dict[str, Any]] = {}
    variant_clusters: dict[str, str] = {}
    for variant in definition.variants:
        variant_id = _variant_identifier(variant)
        payload = canonical_data(variant)
        if not isinstance(payload, dict):
            raise ValueError("scenario variants must have canonical object payloads")
        payload = dict(payload)
        payload.pop("variant_id", None)
        payload.pop("content_hash", None)
        payload.pop("id", None)
        variant_clusters[variant_id] = str(payload.pop("cluster_id", ""))
        provenance = payload.get("action_provenance")
        if isinstance(provenance, dict):
            provenance = dict(provenance)
            provenance.pop("parent_variant_id", None)
            payload["action_provenance"] = provenance
        preliminary_variants[variant_id] = _replace_identifier_references(
            payload,
            resource_labels,
        )

    variant_cluster_members: dict[str, list[Any]] = {}
    for variant_id, cluster_id in variant_clusters.items():
        if cluster_id:
            variant_cluster_members.setdefault(cluster_id, []).append(
                preliminary_variants[variant_id]
            )
    variant_cluster_labels = _labels_from_physical_signatures(
        {
            cluster_id: sorted(members, key=canonical_json)
            for cluster_id, members in variant_cluster_members.items()
        },
        prefix="variant_cluster",
    )

    variant_signatures: dict[str, Any] = {}
    for variant_id, payload in preliminary_variants.items():
        material = dict(payload)
        cluster_id = variant_clusters[variant_id]
        if cluster_id:
            material["cluster_id"] = variant_cluster_labels[cluster_id]
        variant_signatures[variant_id] = material
    variant_labels = _labels_from_physical_signatures(
        variant_signatures,
        prefix="variant",
    )

    effective_cluster_by_flight = {
        flight.flight_id: (
            flight.cluster_id
            or variant_clusters.get(flight.baseline_variant_id, "")
            or "__unassigned_cluster__"
        )
        for flight in definition.flights
    }
    flight_cluster_members: dict[str, list[Any]] = {}
    for flight in definition.flights:
        flight_cluster_members.setdefault(
            effective_cluster_by_flight[flight.flight_id],
            [],
        ).append(
            {
                "release_time_s": flight.release_time_s,
                "observed_release_time_s": flight.observed_release_time_s,
                "release_offset_s": flight.release_offset_s,
                "baseline_variant_id": variant_labels[flight.baseline_variant_id],
                "action_stations": canonical_data(flight.action_stations),
                "resource_crossings": _replace_identifier_references(
                    canonical_data(flight.resource_crossings),
                    resource_labels,
                ),
            }
        )
    flight_cluster_labels = _labels_from_physical_signatures(
        {
            cluster_id: sorted(members, key=canonical_json)
            for cluster_id, members in flight_cluster_members.items()
        },
        prefix="flight_cluster",
    )

    flight_without_runway: dict[str, dict[str, Any]] = {}
    for flight in definition.flights:
        effective_cluster = effective_cluster_by_flight[flight.flight_id]
        flight_without_runway[flight.flight_id] = {
            "release_time_s": flight.release_time_s,
            "observed_release_time_s": flight.observed_release_time_s,
            "release_offset_s": flight.release_offset_s,
            "baseline_variant_id": variant_labels[flight.baseline_variant_id],
            "cluster_id": flight_cluster_labels[effective_cluster],
            "action_stations": canonical_data(flight.action_stations),
            "resource_crossings": _replace_identifier_references(
                canonical_data(flight.resource_crossings),
                resource_labels,
            ),
        }
    effective_runway_by_flight = {
        flight.flight_id: (
            flight.runway
            if flight.runway in resource_labels
            else flight.runway or "__unassigned_runway__"
        )
        for flight in definition.flights
    }
    standalone_runways = {
        effective_runway
        for effective_runway in effective_runway_by_flight.values()
        if effective_runway not in resource_labels
    }
    runway_signatures = {
        runway: sorted(
            (
                flight_without_runway[flight.flight_id]
                for flight in definition.flights
                if effective_runway_by_flight[flight.flight_id] == runway
            ),
            key=canonical_json,
        )
        for runway in standalone_runways
    }
    standalone_runway_labels = _labels_from_physical_signatures(
        runway_signatures,
        prefix="runway_partition",
    )
    runway_labels = {
        **resource_labels,
        **standalone_runway_labels,
    }

    flight_signatures = {
        flight.flight_id: {
            **flight_without_runway[flight.flight_id],
            "runway": runway_labels[effective_runway_by_flight[flight.flight_id]],
        }
        for flight in definition.flights
    }
    flight_labels = _labels_from_physical_signatures(
        flight_signatures,
        prefix="flight",
    )
    all_labels = {
        **resource_labels,
        **variant_cluster_labels,
        **flight_cluster_labels,
        **variant_labels,
        **runway_labels,
        **flight_labels,
    }

    resources = [
        {
            "resource_id": resource_labels[resource.resource_id],
            **_replace_identifier_references(
                resource_signatures[resource.resource_id],
                all_labels,
            ),
        }
        for resource in definition.resources
    ]
    variants = [
        {
            "variant_id": variant_labels[variant_id],
            **_replace_identifier_references(
                {
                    **variant_signatures[variant_id],
                    "action_provenance": canonical_data(
                        getattr(variant, "action_provenance", None)
                    ),
                },
                all_labels,
            ),
        }
        for variant, variant_id in (
            (variant, _variant_identifier(variant)) for variant in definition.variants
        )
    ]
    flights = [
        {
            "flight_id": flight_labels[flight.flight_id],
            **_replace_identifier_references(
                flight_signatures[flight.flight_id],
                all_labels,
            ),
        }
        for flight in definition.flights
    ]
    exogenous_events = [
        {
            "time_s": event.time_s,
            "stream_name": event.stream_name,
            "payload": _replace_identifier_references(
                _material_event_payload(event),
                all_labels,
            ),
        }
        for event in definition.exogenous_events
    ]
    payload = {
        "flights": sorted(flights, key=canonical_json),
        "resources": sorted(resources, key=canonical_json),
        "variants": sorted(variants, key=canonical_json),
        "exogenous_events": sorted(exogenous_events, key=canonical_json),
        "decision_trigger_kinds": sorted(definition.decision_trigger_kinds),
        "weather": _replace_identifier_references(
            canonical_data(definition.weather),
            all_labels,
        ),
        "schema_version": definition.schema_version,
    }
    return content_hash(payload, namespace="hailmary.phase0.realization.v2")


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer_tuple(
    values: Sequence[int],
    *,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(values)
    if name == "seeds":
        if any(type(value) is not int or value < 0 for value in normalized):
            raise ValueError("seeds must contain non-negative integers")
    else:
        normalized = tuple(_positive_int(value, name=name) for value in normalized)
    if not normalized:
        raise ValueError(f"{name} requires at least one value")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(normalized))


def _authenticated_runtime_configuration(
    runtime: ActionRuntime,
    *,
    name: str,
) -> tuple[dict[str, Any], str]:
    """Return a canonical manifest whose advertised hash authenticates it."""

    configuration = canonical_data(runtime.realization_configuration)
    if not isinstance(configuration, dict):
        raise ValueError(f"{name} runtime realization configuration must be a mapping")
    expected_hash = content_hash(
        configuration,
        namespace=_ACTION_RUNTIME_HASH_NAMESPACE,
    )
    advertised_hash = runtime.runtime_configuration_hash
    if advertised_hash != expected_hash:
        raise ValueError(
            f"{name} runtime configuration hash does not authenticate its manifest"
        )
    return configuration, advertised_hash


def _validate_oracle_runtime(
    runtime: ActionRuntime,
) -> tuple[dict[str, Any], str]:
    """Authenticate a primary runtime and require oracle selection."""

    configuration, configuration_hash = _authenticated_runtime_configuration(
        runtime,
        name="oracle",
    )
    stretch = configuration.get("stretch")
    if not isinstance(stretch, dict):
        raise ValueError("oracle runtime manifest must contain a stretch mapping")
    if stretch.get("selector") != _ORACLE_STRETCH_SELECTOR:
        raise ValueError("oracle runtime must use selector='semi_local_outcome'")
    return configuration, configuration_hash


def _validate_geometry_only_runtime(
    runtime: ActionRuntime,
    geometry_only_runtime: ActionRuntime,
) -> None:
    """Prove that an ablation runtime changes only the stretch selector."""

    oracle_configuration, oracle_hash = _validate_oracle_runtime(runtime)
    geometry_configuration, geometry_hash = _authenticated_runtime_configuration(
        geometry_only_runtime,
        name="geometry-only ablation",
    )
    if geometry_only_runtime is runtime or geometry_hash == oracle_hash:
        raise ValueError(
            "geometry-only ablation runtime must be distinct from the oracle runtime"
        )

    geometry_stretch = geometry_configuration.get("stretch")
    if not isinstance(geometry_stretch, dict):
        raise ValueError(
            "geometry-only runtime manifest must contain a stretch mapping"
        )
    if geometry_stretch.get("selector") != _GEOMETRY_ONLY_STRETCH_SELECTOR:
        raise ValueError(
            "geometry-only ablation runtime must use selector='geometry_clearance'"
        )

    expected_geometry_configuration = canonical_data(oracle_configuration)
    expected_geometry_stretch = expected_geometry_configuration["stretch"]
    expected_geometry_stretch["selector"] = _GEOMETRY_ONLY_STRETCH_SELECTOR
    expected_geometry_hash = content_hash(
        expected_geometry_configuration,
        namespace=_ACTION_RUNTIME_HASH_NAMESPACE,
    )
    if geometry_configuration != expected_geometry_configuration or (
        geometry_hash != expected_geometry_hash
    ):
        raise ValueError(
            "geometry-only ablation runtime must match the oracle runtime except "
            "for stretch selector='geometry_clearance'"
        )


def _runtime_provenance(
    runtime: ActionRuntime,
    *,
    name: str,
) -> AuthenticatedRuntimeManifest:
    configuration, configuration_hash = _authenticated_runtime_configuration(
        runtime,
        name=name,
    )
    return AuthenticatedRuntimeManifest.from_configuration(
        configuration,
        configuration_hash,
    )


def _outcome_plan_manifest(outcome_plan: Any) -> dict[str, Any]:
    return {
        "schema_version": "hailmary.phase0.outcome_plan.v1",
        "root_dynamic_content_hash": outcome_plan.root_dynamic_content_hash,
        "root_time_s": outcome_plan.root_time_s,
        "horizon_s": outcome_plan.horizon_s,
        "cohort": canonical_data(outcome_plan.cohort),
        "baseline_crossing_times_s": canonical_data(
            outcome_plan.baseline_crossing_times_s
        ),
        "baseline_intervention_summary": canonical_data(
            outcome_plan.baseline_intervention_summary
        ),
    }


def _held_outcome_details(outcome: Any) -> HeldOutOutcomeDetails:
    return HeldOutOutcomeDetails(
        score=outcome.score,
        pair_score=outcome.pair_score,
        propagation_score=outcome.propagation_score,
        intervention_penalty=outcome.intervention_penalty,
        throughput_score=outcome.throughput_score,
        effective_trailer_count=outcome.effective_trailer_count,
        horizon_s=outcome.horizon_s,
    )


def _held_out_provenance(
    rollout: Any,
    *,
    outcome_plan: Any,
    runtime: ActionRuntime,
) -> HeldOutRolloutProvenance:
    plan_manifest = _outcome_plan_manifest(outcome_plan)
    return HeldOutRolloutProvenance(
        parent_dynamic_content_hash=rollout.parent_dynamic_content_hash,
        outcome_plan_json=canonical_json(plan_manifest),
        outcome_plan_hash=content_hash(
            plan_manifest,
            namespace=_OUTCOME_PLAN_HASH_NAMESPACE,
        ),
        runtime=_runtime_provenance(runtime, name="held-out"),
        learned_policy_fingerprint=rollout.learned_policy_fingerprint,
        permanent_no_op_policy_fingerprint=(rollout.permanent_no_op_policy_fingerprint),
        policy_arm_initial_dynamic_content_hash=(
            rollout.policy_arm.initial_dynamic_content_hash
        ),
        policy_arm_final_dynamic_content_hash=(
            rollout.policy_arm.final_dynamic_content_hash
        ),
        permanent_no_op_initial_dynamic_content_hash=(
            rollout.permanent_no_op.initial_dynamic_content_hash
        ),
        permanent_no_op_final_dynamic_content_hash=(
            rollout.permanent_no_op.final_dynamic_content_hash
        ),
        policy_outcome=_held_outcome_details(rollout.policy_arm.outcome),
        permanent_no_op_outcome=_held_outcome_details(rollout.permanent_no_op.outcome),
    )


@dataclass(frozen=True, slots=True)
class Phase0ExperimentConfig:
    """Finite orchestration settings for causal, refresh, seed, and vanilla runs."""

    seeds: tuple[int, ...] | Sequence[int] = (17, 29)
    refresh_intervals: tuple[int, ...] | Sequence[int] = (100, 500)
    training_passes: int = 1
    run_vanilla: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seeds",
            _integer_tuple(self.seeds, name="seeds"),
        )
        object.__setattr__(
            self,
            "refresh_intervals",
            _integer_tuple(self.refresh_intervals, name="refresh_intervals"),
        )
        object.__setattr__(
            self,
            "training_passes",
            _positive_int(self.training_passes, name="training_passes"),
        )
        if type(self.run_vanilla) is not bool:
            raise TypeError("run_vanilla must be bool")


@dataclass(frozen=True, slots=True)
class Phase0TrainingRun:
    """Detached summary of one real trainer run over the balanced batch."""

    seed: int
    refresh_interval_epochs: int
    credit_mode: str
    trainer_epoch: int
    committed_experiments: int
    publication_generations: tuple[int, ...] | Sequence[int]
    trace_hashes: tuple[str, ...] | Sequence[str]
    rulebook: FrozenRulebookPolicy
    evaluation_snapshot: EvaluationSnapshot
    population_json: str
    rule_hyperrectangles: tuple[RuleHyperrectangle, ...] | Sequence[RuleHyperrectangle]
    certification_evidence: Mapping[str, ActionCertificationEvidence]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("training run seed must be a non-negative integer")
        _positive_int(
            self.refresh_interval_epochs,
            name="refresh_interval_epochs",
        )
        mode = validate_credit_mode(self.credit_mode, name="training run credit_mode")
        if type(self.trainer_epoch) is not int or self.trainer_epoch < 0:
            raise ValueError("trainer_epoch must be a non-negative integer")
        if (
            type(self.committed_experiments) is not int
            or self.committed_experiments < 0
        ):
            raise ValueError("committed_experiments must be a non-negative integer")
        publications = tuple(self.publication_generations)
        if any(type(value) is not int or value < 1 for value in publications):
            raise ValueError("publication generations must be positive integers")
        if publications != tuple(sorted(set(publications))):
            raise ValueError(
                "publication generations must be unique and strictly increasing"
            )
        if any(
            value > self.trainer_epoch or value % self.refresh_interval_epochs != 0
            for value in publications
        ):
            raise ValueError(
                "publication generations must be due within the trainer epoch"
            )
        traces = tuple(str(value).strip() for value in self.trace_hashes)
        if any(not value for value in traces):
            raise ValueError("trace hashes cannot be empty")
        if len(traces) != self.trainer_epoch:
            raise ValueError("one trace hash is required for every trainer epoch")
        if not isinstance(self.rulebook, FrozenRulebookPolicy):
            raise TypeError("training run rulebook must be FrozenRulebookPolicy")
        detached = FrozenRulebookPolicy.from_dict(self.rulebook.to_dict())
        if detached.credit_mode != mode:
            raise ValueError("training run mode does not match its rulebook")
        latest_publication = publications[-1] if publications else 0
        if detached.certification_generation != latest_publication:
            raise ValueError(
                "training run publications do not match the detached rulebook"
            )
        if not isinstance(self.evaluation_snapshot, EvaluationSnapshot):
            raise TypeError("training run requires an EvaluationSnapshot")
        evaluation_snapshot = EvaluationSnapshot.from_dict(
            self.evaluation_snapshot.to_dict()
        )
        if evaluation_snapshot.rulebook != detached:
            raise ValueError(
                "training run evaluation snapshot has a different rulebook"
            )
        if evaluation_snapshot.publication_epoch != latest_publication:
            raise ValueError(
                "training run evaluation snapshot has a different publication epoch"
            )
        if not isinstance(self.population_json, str):
            raise TypeError("population_json must be text")
        population_payload = json.loads(self.population_json)
        if not isinstance(population_payload, dict):
            raise ValueError("population_json must contain an object")
        if population_payload.get("credit_mode") != mode:
            raise ValueError("training run mode does not match its population")
        hyperrectangles = tuple(self.rule_hyperrectangles)
        if any(
            not isinstance(rectangle, RuleHyperrectangle)
            for rectangle in hyperrectangles
        ):
            raise TypeError(
                "training run rule_hyperrectangles must contain RuleHyperrectangle"
            )
        evidence = dict(self.certification_evidence)
        if any(
            not isinstance(value, ActionCertificationEvidence)
            for value in evidence.values()
        ):
            raise TypeError("certification_evidence contains an invalid record")
        published = evaluation_snapshot.certification_evidence
        expected_evidence = (
            {
                source_rule_id: ActionCertificationEvidence(
                    source_rule_id=source_rule_id,
                    noop_samples=record.noop_samples,
                    noop_lcb=record.noop_lcb,
                )
                for source_rule_id, record in published.items()
                if record.rule_kind == "action"
            }
            if mode == CAUSAL_CREDIT_MODE
            else {}
        )
        if evidence != expected_evidence:
            raise ValueError(
                "training run acceptance evidence must derive from its evaluation snapshot"
            )
        object.__setattr__(self, "credit_mode", mode)
        object.__setattr__(self, "publication_generations", publications)
        object.__setattr__(self, "trace_hashes", traces)
        object.__setattr__(self, "rulebook", detached)
        object.__setattr__(self, "evaluation_snapshot", evaluation_snapshot)
        object.__setattr__(self, "rule_hyperrectangles", hyperrectangles)
        object.__setattr__(
            self,
            "certification_evidence",
            MappingProxyType(dict(sorted(evidence.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "refresh_interval_epochs": self.refresh_interval_epochs,
            "credit_mode": self.credit_mode,
            "trainer_epoch": self.trainer_epoch,
            "committed_experiments": self.committed_experiments,
            "publication_count": len(self.publication_generations),
            "publication_generations": list(self.publication_generations),
            "trace_hashes": list(self.trace_hashes),
            "rulebook": self.rulebook.to_dict(),
            "evaluation_snapshot": self.evaluation_snapshot.to_dict(),
            "population": json.loads(self.population_json),
            "rule_hyperrectangles": [
                rectangle.to_dict() for rectangle in self.rule_hyperrectangles
            ],
            "certification_evidence": {
                key: {
                    "source_rule_id": value.source_rule_id,
                    "noop_samples": value.noop_samples,
                    "noop_lcb": value.noop_lcb,
                }
                for key, value in sorted(self.certification_evidence.items())
            },
        }


@dataclass(frozen=True, slots=True)
class Phase0ExperimentResult:
    """Complete real-run bundle plus the acceptance report it produced."""

    causal_runs: tuple[Phase0TrainingRun, ...] | Sequence[Phase0TrainingRun]
    vanilla_runs: tuple[Phase0TrainingRun, ...] | Sequence[Phase0TrainingRun]
    selected_causal_run: Phase0TrainingRun
    validation_inputs: Phase0ValidationInputs
    acceptance_report: Phase0AcceptanceReport
    training_correlation_audit: Mapping[str, Any]
    held_out_correlation_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        causal = tuple(self.causal_runs)
        vanilla = tuple(self.vanilla_runs)
        if not causal or any(run.credit_mode != CAUSAL_CREDIT_MODE for run in causal):
            raise ValueError("causal_runs must contain causal training runs")
        if any(run.credit_mode != VANILLA_ACCURACY_CREDIT_MODE for run in vanilla):
            raise ValueError("vanilla_runs must contain vanilla training runs")
        if self.selected_causal_run not in causal:
            raise ValueError("selected_causal_run must belong to causal_runs")
        if not isinstance(self.validation_inputs, Phase0ValidationInputs):
            raise TypeError("validation_inputs must be Phase0ValidationInputs")
        if not isinstance(self.acceptance_report, Phase0AcceptanceReport):
            raise TypeError("acceptance_report must be Phase0AcceptanceReport")
        vanilla_evidence = self.validation_inputs.vanilla_held_out_comparisons
        if bool(vanilla) != bool(vanilla_evidence):
            raise ValueError(
                "vanilla runs and held-out control evidence must be present together"
            )
        if (
            self.acceptance_report.rulebook_fingerprint
            != self.selected_causal_run.rulebook.policy_fingerprint()
        ):
            raise ValueError("acceptance report belongs to a different rulebook")
        object.__setattr__(self, "causal_runs", causal)
        object.__setattr__(self, "vanilla_runs", vanilla)
        object.__setattr__(
            self,
            "training_correlation_audit",
            dict(self.training_correlation_audit),
        )
        object.__setattr__(
            self,
            "held_out_correlation_audit",
            dict(self.held_out_correlation_audit),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "causal_runs": [run.to_dict() for run in self.causal_runs],
            "vanilla_runs": [run.to_dict() for run in self.vanilla_runs],
            "selected_causal_run": {
                "seed": self.selected_causal_run.seed,
                "refresh_interval_epochs": (
                    self.selected_causal_run.refresh_interval_epochs
                ),
                "rulebook_hash": self.selected_causal_run.rulebook.content_hash,
            },
            "validation_inputs": {
                "train_deploy_decisions": [
                    {
                        "scenario_id": item.scenario_id,
                        "training_action": item.training_action.to_dict(),
                        "deployment_action": item.deployment_action.to_dict(),
                        "exploration_rate": item.exploration_rate,
                    }
                    for item in self.validation_inputs.train_deploy_decisions
                ],
                "held_out_comparisons": [
                    item.to_dict()
                    for item in self.validation_inputs.held_out_comparisons
                ],
                "vanilla_held_out_comparisons": [
                    item.to_dict()
                    for item in self.validation_inputs.vanilla_held_out_comparisons
                ],
                "path_stretch_ablation": (
                    None
                    if self.validation_inputs.path_stretch_ablation is None
                    else self.validation_inputs.path_stretch_ablation.to_dict()
                ),
                "refresh_interval_results": [
                    {
                        "refresh_interval_epochs": item.refresh_interval_epochs,
                        "regime_actions": {
                            regime: action.to_dict()
                            for regime, action in item.regime_actions.items()
                        },
                        "publication_count": item.publication_count,
                        "publication_generations": list(item.publication_generations),
                    }
                    for item in self.validation_inputs.refresh_interval_results
                ],
                "seeded_rule_regions": [
                    {
                        "seed": snapshot.seed,
                        "hyperrectangles": [
                            rectangle.to_dict()
                            for rectangle in snapshot.hyperrectangles
                        ],
                    }
                    for snapshot in self.validation_inputs.seeded_rule_regions
                ],
                "action_certification_evidence": {
                    source_rule_id: {
                        "source_rule_id": evidence.source_rule_id,
                        "noop_samples": evidence.noop_samples,
                        "noop_lcb": evidence.noop_lcb,
                    }
                    for source_rule_id, evidence in (
                        self.validation_inputs.action_certification_evidence.items()
                    )
                },
            },
            "acceptance_report": self.acceptance_report.to_dict(),
            "training_correlation_audit": dict(self.training_correlation_audit),
            "held_out_correlation_audit": dict(self.held_out_correlation_audit),
        }


@dataclass(slots=True)
class _TrainingState:
    trainer: CausalTrainer
    summary: Phase0TrainingRun


@dataclass(frozen=True, slots=True)
class _DecisionFixture:
    simulator: Any
    event_batch: Any
    policy: SimulatorRulebookPolicy
    record: RulebookDecisionRecord
    selected_candidate: Any
    no_op_candidate: Any


class Phase0ExperimentRunner:
    """Run the plan's balanced causal/vanilla experiment on real simulators.

    This class intentionally does not synthesize successful evidence. Every
    trace comes from CausalTrainer.process_epoch, held-out scores come from
    common-root learned-policy versus permanent-no-op rollouts, and failed gates stay
    failed in the returned report.
    """

    def __init__(
        self,
        runtime: ActionRuntime,
        *,
        geometry_only_runtime: ActionRuntime | None = None,
        learning_config: LearningConfig | None = None,
        outcome_config: OutcomeConfig | None = None,
        feature_config: FeatureConfig | None = None,
        scenario_config: ScenarioConfig | None = None,
        config: Phase0ExperimentConfig | None = None,
    ) -> None:
        if not isinstance(runtime, ActionRuntime):
            raise TypeError("runtime must be ActionRuntime")
        if geometry_only_runtime is not None and not isinstance(
            geometry_only_runtime,
            ActionRuntime,
        ):
            raise TypeError("geometry_only_runtime must be ActionRuntime or None")
        if geometry_only_runtime is None:
            _validate_oracle_runtime(runtime)
        else:
            _validate_geometry_only_runtime(runtime, geometry_only_runtime)
        self.runtime = runtime
        self.geometry_only_runtime = geometry_only_runtime
        self.learning_config = (
            LearningConfig() if learning_config is None else learning_config
        )
        self.outcome_config = (
            OutcomeConfig() if outcome_config is None else outcome_config
        )
        self.feature_config = (
            FeatureConfig() if feature_config is None else feature_config
        )
        self.scenario_config = (
            ScenarioConfig() if scenario_config is None else scenario_config
        )
        self.config = Phase0ExperimentConfig() if config is None else config

    def _validate_batch(
        self,
        batch: FactorialScenarioBatch,
        *,
        name: str,
    ) -> tuple[FactorialScenario, ...]:
        if not isinstance(batch, FactorialScenarioBatch):
            raise TypeError(f"{name} must be FactorialScenarioBatch")
        scenarios = tuple(
            sorted(batch.scenarios, key=lambda item: item.condition.condition_id)
        )
        if not scenarios:
            raise ValueError(f"{name} cannot be empty")
        for scenario in scenarios:
            flights = tuple(
                sorted(
                    scenario.definition.flights,
                    key=lambda flight: (flight.release_time_s, flight.flight_id),
                )
            )
            if len(flights) != 2:
                raise ValueError(
                    f"{name} must contain exactly one leader-follower pair"
                )
            if flights[0].release_time_s >= flights[1].release_time_s:
                raise ValueError(f"{name} leader must release before its follower")
            follower_station_types = {
                station.station_type for station in flights[1].action_stations
            }
            if not {"speed", "path_stretch"} <= follower_station_types:
                raise ValueError(
                    f"{name} follower must expose speed and path-stretch actions "
                    "in every factorial cell"
                )
        batch.correlation_audit.assert_passed()
        audited_pairs = {
            frozenset((record.first_feature, record.second_feature))
            for record in batch.correlation_audit.records
        }
        required_pairs = {
            frozenset(pair) for pair in self.scenario_config.registered_factor_pairs
        }
        if not required_pairs <= audited_pairs:
            raise ValueError(f"{name} omits a registered factor-correlation audit")

        rows = [scenario.realized_values_dict for scenario in scenarios]
        if any(not set(_REQUIRED_FACTORS) <= set(row) for row in rows):
            raise ValueError(f"{name} must realize all four Phase-0 factors")
        levels = {
            factor: tuple(sorted({row[factor] for row in rows}))
            for factor in _REQUIRED_FACTORS
        }
        if any(len(values) != 2 for values in levels.values()):
            raise ValueError(f"{name} factors must each have exactly low/high levels")
        combinations = Counter(
            tuple(row[factor] for factor in _REQUIRED_FACTORS) for row in rows
        )
        expected = set(product(*(levels[factor] for factor in _REQUIRED_FACTORS)))
        if set(combinations) != expected or len(set(combinations.values())) != 1:
            raise ValueError(f"{name} must be a balanced full-factorial design")
        return scenarios

    def _train_one(
        self,
        scenarios: Sequence[FactorialScenario],
        *,
        seed: int,
        refresh_interval: int,
        credit_mode: str,
    ) -> _TrainingState:
        mode = validate_credit_mode(credit_mode)
        learning = replace(
            self.learning_config,
            random_seed=seed,
            certification_interval=refresh_interval,
        )
        assigner = (
            CausalCreditAssigner()
            if mode == CAUSAL_CREDIT_MODE
            else VanillaAccuracyCreditAssigner()
        )
        trainer = CausalTrainer(
            self.runtime,
            config=learning,
            outcome_config=self.outcome_config,
            feature_config=self.feature_config,
            scenario_config=self.scenario_config,
            credit_assigner=assigner,
        )
        trace_hashes: list[str] = []
        publication_generations: list[int] = []
        committed = 0
        for _ in range(self.config.training_passes):
            for scenario in scenarios:
                simulator = self.runtime.create_simulator(scenario.definition)
                while True:
                    event_batch = simulator.advance_next()
                    if event_batch is None:
                        break
                    if event_batch.decision_epoch is None:
                        continue
                    result = trainer.process_epoch(simulator, event_batch)
                    trace_hashes.append(result.trace.content_hash)
                    publication_generations.extend(
                        event["publication_epoch"]
                        for event in result.trace.certification_events
                    )
                    committed += int(result.committed)
        if not trace_hashes:
            raise RuntimeError("Phase-0 training produced no decision epochs")
        if not committed:
            raise RuntimeError("Phase-0 training produced no physical experiments")

        evaluation_snapshot = trainer.evaluation_snapshot
        rulebook = evaluation_snapshot.rulebook
        summary = Phase0TrainingRun(
            seed=seed,
            refresh_interval_epochs=refresh_interval,
            credit_mode=mode,
            trainer_epoch=trainer.epoch,
            committed_experiments=committed,
            publication_generations=tuple(publication_generations),
            trace_hashes=tuple(trace_hashes),
            rulebook=rulebook,
            evaluation_snapshot=evaluation_snapshot,
            population_json=canonical_json(trainer.population.to_dict()),
            rule_hyperrectangles=self._rule_hyperrectangles(rulebook),
            certification_evidence=self._certification_evidence(trainer),
        )
        return _TrainingState(trainer=trainer, summary=summary)

    @staticmethod
    def _certification_evidence(
        trainer: CausalTrainer,
    ) -> Mapping[str, ActionCertificationEvidence]:
        if trainer.credit_mode != CAUSAL_CREDIT_MODE:
            return {}
        return {
            source_rule_id: ActionCertificationEvidence(
                source_rule_id=source_rule_id,
                noop_samples=record.noop_samples,
                noop_lcb=record.noop_lcb,
            )
            for source_rule_id, record in (
                trainer.evaluation_snapshot.certification_evidence.items()
            )
            if record.rule_kind == "action"
        }

    @staticmethod
    def _rule_hyperrectangles(
        rulebook: FrozenRulebookPolicy,
    ) -> tuple[RuleHyperrectangle, ...]:
        hyperrectangles: list[RuleHyperrectangle] = []
        for rule in rulebook.action_rules:
            if rule.action.is_no_op:
                continue
            dimensions: list[RuleRegion] = []
            for feature_name in sorted(
                CAPACITY_RATIO_FEATURES & set(rule.condition.intervals)
            ):
                interval = rule.condition.intervals[feature_name]
                dimensions.append(
                    RuleRegion(
                        action=rule.action,
                        feature_name=feature_name,
                        lower=0.0 if interval.lower is None else interval.lower,
                        upper=10.0 if interval.upper is None else interval.upper,
                    )
                )
            if dimensions:
                hyperrectangles.append(
                    RuleHyperrectangle(
                        action=rule.action,
                        dimensions=tuple(dimensions),
                    )
                )
        return tuple(
            sorted(
                hyperrectangles,
                key=lambda rectangle: (rectangle.key, rectangle.bounds_key),
            )
        )

    def _decision_fixture(
        self,
        runtime: ActionRuntime,
        rulebook: FrozenRulebookPolicy,
        scenario: FactorialScenario,
        decision_provider: Callable[[Any, Any], Any] | None = None,
    ) -> _DecisionFixture:
        simulator = runtime.create_simulator(scenario.definition)
        policy = SimulatorRulebookPolicy(
            rulebook,
            template_config=runtime.template_config,
            feature_config=self.feature_config,
            scenario_config=self.scenario_config,
            runtime_configuration_hash=runtime.runtime_configuration_hash,
        )
        while True:
            event_batch = simulator.advance_next()
            if event_batch is None:
                raise RuntimeError(
                    f"held-out scenario {scenario.condition.condition_id!r} "
                    "has no physical action epoch"
                )
            if event_batch.decision_epoch is None:
                continue
            context = SimpleNamespace(
                simulator=simulator,
                event_batch=event_batch,
            )
            records = policy.rulebook_records(context)
            physical = tuple(
                record
                for record in records
                if any(
                    not RuleAction.from_candidate(candidate).is_no_op
                    for candidate in record.candidates
                )
            )
            if not physical:
                continue
            decision = (
                rulebook.evaluate(records)
                if decision_provider is None
                else decision_provider(simulator, event_batch)
            )
            by_anchor = {record.anchor_id: record for record in records}
            record = (
                by_anchor[str(decision.anchor_id)]
                if decision.anchor_id in by_anchor
                else min(physical, key=lambda item: item.anchor_id)
            )
            no_op = next(
                candidate
                for candidate in record.candidates
                if RuleAction.from_candidate(candidate).is_no_op
            )
            selected = no_op if decision.candidate is None else decision.candidate
            return _DecisionFixture(
                simulator=simulator,
                event_batch=event_batch,
                policy=policy,
                record=record,
                selected_candidate=selected,
                no_op_candidate=no_op,
            )

    @staticmethod
    def _anchor_for_fixture(fixture: _DecisionFixture) -> Any:
        return next(
            anchor
            for anchor in build_current_leader_follower_anchors(
                fixture.simulator
            ).leader_follower
            if anchor.anchor_id == fixture.record.anchor_id
        )

    def _held_out_evidence(
        self,
        state: _TrainingState,
        scenarios: Sequence[FactorialScenario],
    ) -> tuple[
        tuple[TrainDeployDecision, ...],
        tuple[HeldOutComparison, ...],
        PathStretchAblationResult | None,
    ]:
        decisions: list[TrainDeployDecision] = []
        comparisons: list[HeldOutComparison] = []
        oracle_path_scores: list[float] = []
        geometry_path_scores: list[float] = []
        for scenario in scenarios:
            fixture = self._decision_fixture(
                self.runtime,
                state.summary.rulebook,
                scenario,
            )
            deployment_action = RuleAction.from_candidate(fixture.selected_candidate)
            training_fixture = self._decision_fixture(
                self.runtime,
                state.trainer.current_rulebook,
                scenario,
                decision_provider=state.trainer.exploration_disabled_decision,
            )
            training_action = RuleAction.from_candidate(
                training_fixture.selected_candidate
            )
            plan = simulator_outcome_plan(
                fixture.simulator,
                self._anchor_for_fixture(fixture),
                config=self.outcome_config,
            )
            rollout = policy_vs_permanent_no_op_simulator_rollout(
                fixture.simulator,
                policy_action=fixture.selected_candidate,
                no_op_action=fixture.no_op_candidate,
                frozen_policy=fixture.policy,
                outcome_plan=plan,
                outcome_config=self.outcome_config,
            )
            scenario_id = scenario.condition.condition_id
            decisions.append(
                TrainDeployDecision(
                    scenario_id=scenario_id,
                    training_action=training_action,
                    deployment_action=deployment_action,
                    exploration_rate=0.0,
                )
            )
            comparisons.append(
                HeldOutComparison(
                    scenario_id=scenario_id,
                    policy_score=rollout.policy_arm.score,
                    permanent_no_op_score=rollout.permanent_no_op.score,
                    provenance=_held_out_provenance(
                        rollout,
                        outcome_plan=plan,
                        runtime=self.runtime,
                    ),
                )
            )
            if deployment_action.key != PATH_STRETCH_ACTION_KEY:
                continue
            oracle_path_scores.append(rollout.policy_arm.score)
            if self.geometry_only_runtime is None:
                continue
            geometry = self._decision_fixture(
                self.geometry_only_runtime,
                state.summary.rulebook,
                scenario,
            )
            geometry_action = RuleAction.from_candidate(geometry.selected_candidate)
            if geometry_action.key != PATH_STRETCH_ACTION_KEY:
                raise RuntimeError(
                    "geometry ablation changed the learned action identity"
                )
            geometry_plan = simulator_outcome_plan(
                geometry.simulator,
                self._anchor_for_fixture(geometry),
                config=self.outcome_config,
            )
            geometry_rollout = policy_vs_permanent_no_op_simulator_rollout(
                geometry.simulator,
                policy_action=geometry.selected_candidate,
                no_op_action=geometry.no_op_candidate,
                frozen_policy=geometry.policy,
                outcome_plan=geometry_plan,
                outcome_config=self.outcome_config,
            )
            geometry_path_scores.append(geometry_rollout.policy_arm.score)

        path_result = None
        if oracle_path_scores:
            complete_geometry = len(geometry_path_scores) == len(oracle_path_scores)
            path_result = PathStretchAblationResult(
                action_key=PATH_STRETCH_ACTION_KEY,
                oracle_score=sum(oracle_path_scores) / len(oracle_path_scores),
                geometry_only_score=(
                    sum(geometry_path_scores) / len(geometry_path_scores)
                    if complete_geometry
                    else None
                ),
                oracle_runtime=(
                    _runtime_provenance(self.runtime, name="oracle ablation")
                    if complete_geometry
                    else None
                ),
                geometry_only_runtime=(
                    _runtime_provenance(
                        self.geometry_only_runtime,
                        name="geometry-only ablation",
                    )
                    if complete_geometry and self.geometry_only_runtime is not None
                    else None
                ),
            )
        return tuple(decisions), tuple(comparisons), path_result

    def _held_out_comparisons(
        self,
        state: _TrainingState,
        scenarios: Sequence[FactorialScenario],
    ) -> tuple[HeldOutComparison, ...]:
        comparisons: list[HeldOutComparison] = []
        for scenario in scenarios:
            fixture = self._decision_fixture(
                self.runtime,
                state.summary.rulebook,
                scenario,
            )
            plan = simulator_outcome_plan(
                fixture.simulator,
                self._anchor_for_fixture(fixture),
                config=self.outcome_config,
            )
            rollout = policy_vs_permanent_no_op_simulator_rollout(
                fixture.simulator,
                policy_action=fixture.selected_candidate,
                no_op_action=fixture.no_op_candidate,
                frozen_policy=fixture.policy,
                outcome_plan=plan,
                outcome_config=self.outcome_config,
            )
            comparisons.append(
                HeldOutComparison(
                    scenario_id=scenario.condition.condition_id,
                    policy_score=rollout.policy_arm.score,
                    permanent_no_op_score=rollout.permanent_no_op.score,
                    provenance=_held_out_provenance(
                        rollout,
                        outcome_plan=plan,
                        runtime=self.runtime,
                    ),
                )
            )
        return tuple(comparisons)

    def _qualitative_actions(
        self,
        state: _TrainingState,
        scenarios: Sequence[FactorialScenario],
    ) -> Mapping[str, RuleAction]:
        actions: dict[str, RuleAction] = {}
        for scenario in scenarios:
            fixture = self._decision_fixture(
                self.runtime,
                state.summary.rulebook,
                scenario,
            )
            actions[scenario.condition.condition_id] = RuleAction.from_candidate(
                fixture.selected_candidate
            )
        return actions

    def run(
        self,
        training_batch: FactorialScenarioBatch,
        held_out_batch: FactorialScenarioBatch,
    ) -> Phase0ExperimentResult:
        """Execute balanced training and return evidence without forcing a pass."""

        training = self._validate_batch(training_batch, name="training_batch")
        held_out = self._validate_batch(held_out_batch, name="held_out_batch")
        training_realizations = {
            _material_scenario_fingerprint(item) for item in training
        }
        held_out_realizations = {
            _material_scenario_fingerprint(item) for item in held_out
        }
        if training_realizations & held_out_realizations:
            raise ValueError(
                "training and held-out material scenario realizations must be disjoint"
            )
        causal_states = tuple(
            self._train_one(
                training,
                seed=seed,
                refresh_interval=refresh,
                credit_mode=CAUSAL_CREDIT_MODE,
            )
            for refresh in self.config.refresh_intervals
            for seed in self.config.seeds
        )
        primary_seed = self.config.seeds[0]
        primary_refresh = self.config.refresh_intervals[-1]
        selected = next(
            state
            for state in causal_states
            if state.summary.seed == primary_seed
            and state.summary.refresh_interval_epochs == primary_refresh
        )
        vanilla_states = (
            tuple(
                self._train_one(
                    training,
                    seed=seed,
                    refresh_interval=primary_refresh,
                    credit_mode=VANILLA_ACCURACY_CREDIT_MODE,
                )
                for seed in self.config.seeds
            )
            if self.config.run_vanilla
            else ()
        )

        selected_vanilla = (
            next(
                state for state in vanilla_states if state.summary.seed == primary_seed
            )
            if vanilla_states
            else None
        )

        decisions, comparisons, path_result = self._held_out_evidence(
            selected,
            held_out,
        )
        vanilla_comparisons = (
            self._held_out_comparisons(selected_vanilla, held_out)
            if selected_vanilla is not None
            else ()
        )
        refresh_results = tuple(
            RefreshIntervalResult(
                state.summary.refresh_interval_epochs,
                self._qualitative_actions(state, held_out),
                state.summary.publication_generations,
            )
            for state in causal_states
            if state.summary.seed == primary_seed
        )
        seeded_regions = tuple(
            SeededRuleRegions(
                state.summary.seed,
                state.summary.rule_hyperrectangles,
            )
            for state in causal_states
            if state.summary.refresh_interval_epochs == primary_refresh
        )
        inputs = Phase0ValidationInputs(
            train_deploy_decisions=decisions,
            held_out_comparisons=comparisons,
            vanilla_held_out_comparisons=vanilla_comparisons,
            path_stretch_ablation=path_result,
            refresh_interval_results=refresh_results,
            seeded_rule_regions=seeded_regions,
            action_certification_evidence=selected.summary.certification_evidence,
        )
        report = evaluate_phase0_acceptance(
            selected.summary.rulebook,
            inputs,
            action_min_noop_samples=self.learning_config.action_min_noop_samples,
        )
        return Phase0ExperimentResult(
            causal_runs=tuple(state.summary for state in causal_states),
            vanilla_runs=tuple(state.summary for state in vanilla_states),
            selected_causal_run=selected.summary,
            validation_inputs=inputs,
            acceptance_report=report,
            training_correlation_audit=training_batch.correlation_audit.to_dict(),
            held_out_correlation_audit=held_out_batch.correlation_audit.to_dict(),
        )


__all__ = [
    "Phase0ExperimentConfig",
    "Phase0ExperimentResult",
    "Phase0ExperimentRunner",
    "Phase0TrainingRun",
]
