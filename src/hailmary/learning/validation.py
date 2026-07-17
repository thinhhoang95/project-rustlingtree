"""Lightweight Phase-0 scientific acceptance evidence and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from hailmary.actions.models import ActionLever
from hailmary.actions.vocabulary import (
    PATH_STRETCH_MACRO_BAND,
    action_vocabulary,
)
from hailmary.ids import canonical_data, canonical_json, content_hash
from hailmary.learning.rulebook import FrozenRulebookPolicy
from hailmary.learning.rules import RuleAction
from hailmary.rollout.policy import NoOpPolicy, policy_fingerprint


CAPACITY_RATIO_FEATURES = frozenset(
    {
        "required_delay_over_speed_capacity",
        "required_delay_over_path_capacity",
    }
)
PHASE0_ACTION_KEYS = tuple(
    RuleAction(identity.lever, identity.band).key
    for identity in action_vocabulary().identities
)
PATH_STRETCH_ACTION_KEY = RuleAction(
    ActionLever.PATH_STRETCH,
    PATH_STRETCH_MACRO_BAND,
).key
_ACTION_RUNTIME_HASH_NAMESPACE = "hailmary.action_runtime.v1"
_OUTCOME_PLAN_HASH_NAMESPACE = "hailmary.phase0.outcome_plan.v1"
_ORACLE_STRETCH_SELECTOR = "semi_local_outcome"
_GEOMETRY_ONLY_STRETCH_SELECTOR = "geometry_clearance"


def _finite(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _coerce_action(value: RuleAction | Mapping[str, Any] | Any | None) -> RuleAction:
    if value is None:
        return RuleAction.no_op()
    if isinstance(value, str):
        lever, separator, band = value.partition("/")
        if not separator:
            raise ValueError("action strings must use lever/band")
        return RuleAction(lever, band)
    return RuleAction.from_candidate(value)


@dataclass(frozen=True, slots=True)
class TrainDeployDecision:
    """One exploration-disabled training exploit and deployment decision pair."""

    scenario_id: str
    training_action: RuleAction | Any | None
    deployment_action: RuleAction | Any | None
    exploration_rate: float = 0.0

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        if not scenario_id:
            raise ValueError("train/deploy decisions require a scenario_id")
        rate = _finite(self.exploration_rate, name="exploration_rate")
        if not 0.0 <= rate <= 1.0:
            raise ValueError("exploration_rate must be in [0, 1]")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(
            self,
            "training_action",
            _coerce_action(self.training_action),
        )
        object.__setattr__(
            self,
            "deployment_action",
            _coerce_action(self.deployment_action),
        )
        object.__setattr__(self, "exploration_rate", rate)


@dataclass(frozen=True, slots=True)
class AuthenticatedRuntimeManifest:
    """Canonical runtime manifest paired with its content-addressed hash."""

    manifest_json: str
    configuration_hash: str

    def __post_init__(self) -> None:
        if type(self.manifest_json) is not str or not self.manifest_json:
            raise ValueError("runtime manifest_json must be non-empty text")
        try:
            manifest = json.loads(self.manifest_json)
        except json.JSONDecodeError as exc:
            raise ValueError("runtime manifest_json must contain valid JSON") from exc
        if not isinstance(manifest, dict):
            raise ValueError("runtime manifest must contain an object")
        normalized_json = canonical_json(manifest)
        configuration_hash = str(self.configuration_hash).strip()
        expected_hash = content_hash(
            manifest,
            namespace=_ACTION_RUNTIME_HASH_NAMESPACE,
        )
        if configuration_hash != expected_hash:
            raise ValueError(
                "runtime configuration hash does not authenticate its manifest"
            )
        object.__setattr__(self, "manifest_json", normalized_json)
        object.__setattr__(self, "configuration_hash", configuration_hash)

    @classmethod
    def from_configuration(
        cls,
        configuration: Mapping[str, Any],
        configuration_hash: str,
    ) -> "AuthenticatedRuntimeManifest":
        return cls(canonical_json(configuration), configuration_hash)

    @property
    def manifest(self) -> dict[str, Any]:
        payload = json.loads(self.manifest_json)
        assert isinstance(payload, dict)
        return payload

    @property
    def stretch_selector(self) -> str:
        stretch = self.manifest.get("stretch")
        return str(stretch.get("selector", "")) if isinstance(stretch, dict) else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "configuration_hash": self.configuration_hash,
        }


@dataclass(frozen=True, slots=True)
class HeldOutOutcomeDetails:
    """Compact typed outcome decomposition for one held-out rollout arm."""

    score: float
    pair_score: float
    propagation_score: float
    intervention_penalty: float
    throughput_score: float
    effective_trailer_count: int
    horizon_s: float

    def __post_init__(self) -> None:
        for name in (
            "score",
            "pair_score",
            "propagation_score",
            "intervention_penalty",
            "throughput_score",
            "horizon_s",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if (
            type(self.effective_trailer_count) is not int
            or self.effective_trailer_count < 0
        ):
            raise ValueError("effective_trailer_count must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "pair_score": self.pair_score,
            "propagation_score": self.propagation_score,
            "intervention_penalty": self.intervention_penalty,
            "throughput_score": self.throughput_score,
            "effective_trailer_count": self.effective_trailer_count,
            "horizon_s": self.horizon_s,
        }


@dataclass(frozen=True, slots=True)
class HeldOutRolloutProvenance:
    """Authenticated common-root provenance for learned versus permanent no-op."""

    parent_dynamic_content_hash: str
    outcome_plan_json: str
    outcome_plan_hash: str
    runtime: AuthenticatedRuntimeManifest
    learned_policy_fingerprint: str
    permanent_no_op_policy_fingerprint: str
    policy_arm_initial_dynamic_content_hash: str
    policy_arm_final_dynamic_content_hash: str
    permanent_no_op_initial_dynamic_content_hash: str
    permanent_no_op_final_dynamic_content_hash: str
    policy_outcome: HeldOutOutcomeDetails
    permanent_no_op_outcome: HeldOutOutcomeDetails

    def __post_init__(self) -> None:
        string_fields = (
            "parent_dynamic_content_hash",
            "outcome_plan_json",
            "outcome_plan_hash",
            "learned_policy_fingerprint",
            "permanent_no_op_policy_fingerprint",
            "policy_arm_initial_dynamic_content_hash",
            "policy_arm_final_dynamic_content_hash",
            "permanent_no_op_initial_dynamic_content_hash",
            "permanent_no_op_final_dynamic_content_hash",
        )
        for name in string_fields:
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty exact text")
        if not isinstance(self.runtime, AuthenticatedRuntimeManifest):
            raise TypeError("held-out provenance requires an authenticated runtime")
        if not isinstance(self.policy_outcome, HeldOutOutcomeDetails) or not isinstance(
            self.permanent_no_op_outcome,
            HeldOutOutcomeDetails,
        ):
            raise TypeError("held-out provenance requires typed arm outcomes")
        try:
            outcome_plan = json.loads(self.outcome_plan_json)
        except json.JSONDecodeError as exc:
            raise ValueError("outcome_plan_json must contain valid JSON") from exc
        if not isinstance(outcome_plan, dict):
            raise ValueError("outcome plan manifest must contain an object")
        normalized_plan_json = canonical_json(outcome_plan)
        expected_plan_hash = content_hash(
            outcome_plan,
            namespace=_OUTCOME_PLAN_HASH_NAMESPACE,
        )
        if self.outcome_plan_hash != expected_plan_hash:
            raise ValueError("outcome-plan hash does not authenticate its manifest")
        if (
            outcome_plan.get("root_dynamic_content_hash")
            != self.parent_dynamic_content_hash
        ):
            raise ValueError(
                "outcome plan and held-out arms must share one parent root"
            )
        horizon_s = _finite(outcome_plan.get("horizon_s"), name="outcome plan horizon")
        if not math.isclose(
            self.policy_outcome.horizon_s, horizon_s
        ) or not math.isclose(
            self.permanent_no_op_outcome.horizon_s,
            horizon_s,
        ):
            raise ValueError("held-out arm outcomes must use the outcome-plan horizon")
        if (
            self.policy_arm_initial_dynamic_content_hash
            != self.parent_dynamic_content_hash
        ):
            raise ValueError("learned arm initial hash must equal the common parent")
        if (
            self.permanent_no_op_initial_dynamic_content_hash
            != self.parent_dynamic_content_hash
        ):
            raise ValueError(
                "permanent no-op arm initial hash must equal the common parent"
            )
        canonical_no_op_fingerprint = policy_fingerprint(NoOpPolicy())
        if self.permanent_no_op_policy_fingerprint != canonical_no_op_fingerprint:
            raise ValueError(
                "permanent no-op policy fingerprint is not canonical NoOpPolicy"
            )
        object.__setattr__(self, "outcome_plan_json", normalized_plan_json)

    @property
    def outcome_plan(self) -> dict[str, Any]:
        payload = json.loads(self.outcome_plan_json)
        assert isinstance(payload, dict)
        return payload

    @property
    def runtime_configuration_hash(self) -> str:
        return self.runtime.configuration_hash

    @property
    def permanent_control_identity(self) -> tuple[str, ...]:
        return (
            self.parent_dynamic_content_hash,
            self.outcome_plan_hash,
            self.runtime_configuration_hash,
            self.permanent_no_op_policy_fingerprint,
            self.permanent_no_op_initial_dynamic_content_hash,
            self.permanent_no_op_final_dynamic_content_hash,
            canonical_json(self.permanent_no_op_outcome.to_dict()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_dynamic_content_hash": self.parent_dynamic_content_hash,
            "outcome_plan": self.outcome_plan,
            "outcome_plan_hash": self.outcome_plan_hash,
            "runtime": self.runtime.to_dict(),
            "learned_policy_fingerprint": self.learned_policy_fingerprint,
            "permanent_no_op_policy_fingerprint": self.permanent_no_op_policy_fingerprint,
            "policy_arm": {
                "initial_dynamic_content_hash": self.policy_arm_initial_dynamic_content_hash,
                "final_dynamic_content_hash": self.policy_arm_final_dynamic_content_hash,
                "outcome": self.policy_outcome.to_dict(),
            },
            "permanent_no_op_arm": {
                "initial_dynamic_content_hash": self.permanent_no_op_initial_dynamic_content_hash,
                "final_dynamic_content_hash": self.permanent_no_op_final_dynamic_content_hash,
                "outcome": self.permanent_no_op_outcome.to_dict(),
            },
        }


@dataclass(frozen=True, slots=True)
class HeldOutComparison:
    """Paired held-out score for the learned policy and permanent no-op."""

    scenario_id: str
    policy_score: float
    permanent_no_op_score: float
    provenance: HeldOutRolloutProvenance | None = None

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        if not scenario_id:
            raise ValueError("held-out comparisons require a scenario_id")
        policy_score = _finite(self.policy_score, name="policy_score")
        permanent_score = _finite(
            self.permanent_no_op_score,
            name="permanent_no_op_score",
        )
        if self.provenance is not None:
            if not isinstance(self.provenance, HeldOutRolloutProvenance):
                raise TypeError("held-out provenance has an invalid type")
            if not math.isclose(policy_score, self.provenance.policy_outcome.score):
                raise ValueError("policy score must match its rollout provenance")
            if not math.isclose(
                permanent_score,
                self.provenance.permanent_no_op_outcome.score,
            ):
                raise ValueError(
                    "permanent no-op score must match its rollout provenance"
                )
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "policy_score", policy_score)
        object.__setattr__(self, "permanent_no_op_score", permanent_score)

    @property
    def improvement(self) -> float:
        return self.policy_score - self.permanent_no_op_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "policy_score": self.policy_score,
            "permanent_no_op_score": self.permanent_no_op_score,
            "provenance": None
            if self.provenance is None
            else self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PathStretchAblationResult:
    """Reported oracle and geometry-only scores with authenticated runtimes."""

    action_key: str
    oracle_score: float | None
    geometry_only_score: float | None
    oracle_runtime: AuthenticatedRuntimeManifest | None = None
    geometry_only_runtime: AuthenticatedRuntimeManifest | None = None

    def __post_init__(self) -> None:
        action_key = str(self.action_key).strip()
        if not action_key:
            raise ValueError("path-stretch result requires an action_key")
        object.__setattr__(self, "action_key", action_key)
        if self.oracle_score is not None:
            object.__setattr__(
                self,
                "oracle_score",
                _finite(self.oracle_score, name="oracle_score"),
            )
        if self.geometry_only_score is not None:
            object.__setattr__(
                self,
                "geometry_only_score",
                _finite(
                    self.geometry_only_score,
                    name="geometry_only_score",
                ),
            )
        manifests = (self.oracle_runtime, self.geometry_only_runtime)
        if any(manifest is not None for manifest in manifests):
            if not all(
                isinstance(manifest, AuthenticatedRuntimeManifest)
                for manifest in manifests
            ):
                raise TypeError(
                    "path-stretch runtime provenance must provide both manifests"
                )
            assert self.oracle_runtime is not None
            assert self.geometry_only_runtime is not None
            if self.oracle_runtime.stretch_selector != _ORACLE_STRETCH_SELECTOR:
                raise ValueError(
                    "oracle runtime manifest has the wrong stretch selector"
                )
            if (
                self.geometry_only_runtime.stretch_selector
                != _GEOMETRY_ONLY_STRETCH_SELECTOR
            ):
                raise ValueError(
                    "geometry-only runtime manifest has the wrong stretch selector"
                )
            expected_geometry = self.oracle_runtime.manifest
            stretch = expected_geometry.get("stretch")
            if not isinstance(stretch, dict):
                raise ValueError(
                    "oracle runtime manifest must contain stretch settings"
                )
            stretch["selector"] = _GEOMETRY_ONLY_STRETCH_SELECTOR
            if expected_geometry != self.geometry_only_runtime.manifest:
                raise ValueError(
                    "geometry-only runtime must differ from oracle only by selector"
                )

    @property
    def has_authenticated_runtime_provenance(self) -> bool:
        return (
            self.oracle_runtime is not None and self.geometry_only_runtime is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_key": self.action_key,
            "oracle_score": self.oracle_score,
            "geometry_only_score": self.geometry_only_score,
            "oracle_runtime": (
                None if self.oracle_runtime is None else self.oracle_runtime.to_dict()
            ),
            "geometry_only_runtime": (
                None
                if self.geometry_only_runtime is None
                else self.geometry_only_runtime.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class ActionCertificationEvidence:
    """Auditable no-op-grounded gate evidence for one frozen action rule."""

    source_rule_id: str
    noop_samples: int
    noop_lcb: float

    def __post_init__(self) -> None:
        source_rule_id = str(self.source_rule_id).strip()
        if not source_rule_id:
            raise ValueError("certification evidence requires a source_rule_id")
        if (
            isinstance(self.noop_samples, bool)
            or not isinstance(self.noop_samples, int)
            or self.noop_samples < 0
        ):
            raise ValueError("noop_samples must be a non-negative integer")
        object.__setattr__(self, "source_rule_id", source_rule_id)
        object.__setattr__(
            self,
            "noop_lcb",
            _finite(self.noop_lcb, name="noop_lcb"),
        )


@dataclass(frozen=True)
class RefreshIntervalResult:
    """Qualitative action choice and real publications for one refresh interval."""

    refresh_interval_epochs: int
    regime_actions: Mapping[str, RuleAction | Any]
    publication_generations: tuple[int, ...] | Sequence[int]

    def __post_init__(self) -> None:
        if (
            type(self.refresh_interval_epochs) is not int
            or self.refresh_interval_epochs < 1
        ):
            raise ValueError("refresh_interval_epochs must be a positive integer")
        interval = self.refresh_interval_epochs
        if not isinstance(self.regime_actions, Mapping) or not self.regime_actions:
            raise ValueError("refresh results require regime actions")
        normalized: dict[str, RuleAction] = {}
        for raw_regime, raw_action in self.regime_actions.items():
            regime = str(raw_regime).strip()
            if not regime:
                raise ValueError("refresh regime names cannot be empty")
            normalized[regime] = _coerce_action(raw_action)
        if isinstance(self.publication_generations, (str, bytes)) or not isinstance(
            self.publication_generations,
            Sequence,
        ):
            raise TypeError("publication_generations must be a sequence")
        publications = tuple(self.publication_generations)
        if any(type(value) is not int or value < 1 for value in publications):
            raise ValueError("publication generations must be positive integers")
        if publications != tuple(sorted(set(publications))):
            raise ValueError(
                "publication generations must be unique and strictly increasing"
            )
        if any(value % interval != 0 for value in publications):
            raise ValueError(
                "publication generations must align with the refresh interval"
            )
        object.__setattr__(self, "refresh_interval_epochs", interval)
        object.__setattr__(
            self,
            "regime_actions",
            MappingProxyType(dict(sorted(normalized.items()))),
        )
        object.__setattr__(self, "publication_generations", publications)

    @property
    def signature(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (regime, action.key) for regime, action in self.regime_actions.items()
        )

    @property
    def publication_count(self) -> int:
        return len(self.publication_generations)

    @property
    def latest_publication_generation(self) -> int:
        return self.publication_generations[-1] if self.publication_generations else 0


@dataclass(frozen=True, slots=True)
class RuleRegion:
    """One finite one-dimensional learned interval used for seed comparison."""

    action: RuleAction | Any
    feature_name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        action = _coerce_action(self.action)
        feature_name = str(self.feature_name).strip()
        if not feature_name:
            raise ValueError("rule regions require a feature_name")
        lower = _finite(self.lower, name="region lower")
        upper = _finite(self.upper, name="region upper")
        if lower > upper:
            raise ValueError("rule region lower cannot exceed upper")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "feature_name", feature_name)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def key(self) -> tuple[str, str]:
        return self.action.key, self.feature_name


@dataclass(frozen=True, slots=True)
class RuleHyperrectangle:
    """One learned rule's joint capacity-ratio geometry."""

    action: RuleAction | Any
    dimensions: tuple[RuleRegion, ...] | Sequence[RuleRegion]

    def __post_init__(self) -> None:
        action = _coerce_action(self.action)
        dimensions = tuple(self.dimensions)
        if not dimensions or any(
            not isinstance(dimension, RuleRegion) for dimension in dimensions
        ):
            raise ValueError("rule hyperrectangles require typed dimensions")
        if any(dimension.action != action for dimension in dimensions):
            raise ValueError("hyperrectangle dimensions must share one action")
        names = tuple(dimension.feature_name for dimension in dimensions)
        if len(names) != len(set(names)):
            raise ValueError("hyperrectangle feature dimensions must be unique")
        dimensions = tuple(sorted(dimensions, key=lambda item: item.feature_name))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "dimensions", dimensions)

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return self.action.key, tuple(
            dimension.feature_name for dimension in self.dimensions
        )

    @property
    def bounds_key(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (dimension.lower, dimension.upper) for dimension in self.dimensions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "dimensions": [
                {
                    "feature_name": dimension.feature_name,
                    "lower": dimension.lower,
                    "upper": dimension.upper,
                }
                for dimension in self.dimensions
            ],
        }


@dataclass(frozen=True)
class SeededRuleRegions:
    """Comparable joint rule hyperrectangles published by one seeded run."""

    seed: int
    hyperrectangles: tuple[RuleHyperrectangle, ...] | Sequence[RuleHyperrectangle]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        hyperrectangles = tuple(self.hyperrectangles)
        if any(
            not isinstance(rectangle, RuleHyperrectangle)
            for rectangle in hyperrectangles
        ):
            raise TypeError("seeded regions must contain RuleHyperrectangle records")
        hyperrectangles = tuple(
            sorted(
                hyperrectangles,
                key=lambda rectangle: (rectangle.key, rectangle.bounds_key),
            )
        )
        object.__setattr__(self, "hyperrectangles", hyperrectangles)


@dataclass(frozen=True)
class Phase0ValidationInputs:
    """Small precomputed evidence bundle; no simulation is run here."""

    train_deploy_decisions: (
        tuple[TrainDeployDecision, ...] | Sequence[TrainDeployDecision]
    )
    held_out_comparisons: tuple[HeldOutComparison, ...] | Sequence[HeldOutComparison]
    path_stretch_ablation: PathStretchAblationResult | None
    refresh_interval_results: (
        tuple[RefreshIntervalResult, ...] | Sequence[RefreshIntervalResult]
    )
    seeded_rule_regions: tuple[SeededRuleRegions, ...] | Sequence[SeededRuleRegions]
    vanilla_held_out_comparisons: (
        tuple[HeldOutComparison, ...] | Sequence[HeldOutComparison]
    ) = ()
    action_certification_evidence: Mapping[
        str,
        ActionCertificationEvidence,
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        decisions = tuple(self.train_deploy_decisions)
        held_out = tuple(self.held_out_comparisons)
        vanilla_held_out = tuple(self.vanilla_held_out_comparisons)
        refresh = tuple(self.refresh_interval_results)
        seeded = tuple(self.seeded_rule_regions)
        if any(not isinstance(item, TrainDeployDecision) for item in decisions):
            raise TypeError("train_deploy_decisions contains an invalid record")
        if any(not isinstance(item, HeldOutComparison) for item in held_out):
            raise TypeError("held_out_comparisons contains an invalid record")
        if any(not isinstance(item, HeldOutComparison) for item in vanilla_held_out):
            raise TypeError("vanilla_held_out_comparisons contains an invalid record")
        held_out_ids = tuple(item.scenario_id for item in held_out)
        vanilla_ids = tuple(item.scenario_id for item in vanilla_held_out)
        if len(held_out_ids) != len(set(held_out_ids)):
            raise ValueError("held_out_comparisons cannot duplicate scenario IDs")
        if len(vanilla_ids) != len(set(vanilla_ids)):
            raise ValueError(
                "vanilla_held_out_comparisons cannot duplicate scenario IDs"
            )
        if vanilla_ids and vanilla_ids != held_out_ids:
            raise ValueError(
                "vanilla held-out control must use the exact causal scenario order"
            )
        if vanilla_held_out and any(
            causal.provenance is None
            or vanilla.provenance is None
            or causal.provenance.permanent_control_identity
            != vanilla.provenance.permanent_control_identity
            for causal, vanilla in zip(held_out, vanilla_held_out, strict=True)
        ):
            raise ValueError(
                "causal and vanilla held-out controls must have identical "
                "auditable permanent no-op provenance"
            )
        if self.path_stretch_ablation is not None and not isinstance(
            self.path_stretch_ablation,
            PathStretchAblationResult,
        ):
            raise TypeError("path_stretch_ablation has an invalid type")
        if any(not isinstance(item, RefreshIntervalResult) for item in refresh):
            raise TypeError("refresh_interval_results contains an invalid record")
        if any(not isinstance(item, SeededRuleRegions) for item in seeded):
            raise TypeError("seeded_rule_regions contains an invalid record")
        if not isinstance(self.action_certification_evidence, Mapping):
            raise TypeError("action_certification_evidence must be a mapping")
        certification_evidence: dict[str, ActionCertificationEvidence] = {}
        for raw_source_rule_id, evidence in self.action_certification_evidence.items():
            source_rule_id = str(raw_source_rule_id).strip()
            if not source_rule_id:
                raise ValueError("certification evidence keys cannot be empty")
            if not isinstance(evidence, ActionCertificationEvidence):
                raise TypeError(
                    "action_certification_evidence contains an invalid record"
                )
            if evidence.source_rule_id != source_rule_id:
                raise ValueError(
                    "certification evidence key must equal its source_rule_id"
                )
            certification_evidence[source_rule_id] = evidence
        object.__setattr__(self, "train_deploy_decisions", decisions)
        object.__setattr__(self, "held_out_comparisons", held_out)
        object.__setattr__(
            self,
            "vanilla_held_out_comparisons",
            vanilla_held_out,
        )
        object.__setattr__(self, "refresh_interval_results", refresh)
        object.__setattr__(self, "seeded_rule_regions", seeded)
        object.__setattr__(
            self,
            "action_certification_evidence",
            MappingProxyType(dict(sorted(certification_evidence.items()))),
        )


AcceptanceMetric = str | int | float | bool | None


@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    passed: bool
    detail: str
    metrics: Mapping[str, AcceptanceMetric] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        detail = str(self.detail).strip()
        if not name or not detail:
            raise ValueError("acceptance checks require name and detail")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("acceptance check metrics must be a mapping")
        metrics: dict[str, AcceptanceMetric] = {}
        for raw_key, raw_value in self.metrics.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError("acceptance metric names cannot be empty")
            value = canonical_data(raw_value)
            if value is not None and not isinstance(
                value,
                (str, int, float, bool),
            ):
                raise TypeError("acceptance metrics must be scalar")
            metrics[key] = value
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "detail", detail)
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(dict(sorted(metrics.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class Phase0AcceptanceReport:
    rulebook_fingerprint: str
    checks: tuple[AcceptanceCheck, ...] | Sequence[AcceptanceCheck]

    def __post_init__(self) -> None:
        fingerprint = str(self.rulebook_fingerprint).strip()
        checks = tuple(self.checks)
        if not fingerprint:
            raise ValueError("acceptance report requires a rulebook fingerprint")
        if not checks or any(
            not isinstance(check, AcceptanceCheck) for check in checks
        ):
            raise ValueError("acceptance report requires typed checks")
        names = tuple(check.name for check in checks)
        if len(names) != len(set(names)):
            raise ValueError("acceptance check names must be unique")
        object.__setattr__(self, "rulebook_fingerprint", fingerprint)
        object.__setattr__(self, "checks", checks)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[AcceptanceCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def check(self, name: str) -> AcceptanceCheck:
        for check in self.checks:
            if check.name == str(name):
                return check
        raise KeyError(f"unknown acceptance check {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rulebook_fingerprint": self.rulebook_fingerprint,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


def _configured_action_keys(
    rulebook: FrozenRulebookPolicy,
) -> tuple[tuple[str, ...], bool]:
    vocabulary = action_vocabulary()
    configuration = canonical_data(rulebook.action_configuration)
    if isinstance(configuration, str):
        return (), configuration == vocabulary.content_hash
    if not isinstance(configuration, dict):
        return (), False
    actions = configuration.get("actions")
    if not isinstance(actions, list):
        return (), False
    try:
        keys = tuple(RuleAction.from_candidate(action).key for action in actions)
    except (KeyError, TypeError, ValueError):
        return (), False
    return keys, keys == PHASE0_ACTION_KEYS


def _is_distance_confound(feature_name: str) -> bool:
    return (
        "distance_to_resource" in feature_name
        or "distance_to_final" in feature_name
        or "time_to_final" in feature_name
    )


def _hyperrectangle_iou(
    left: RuleHyperrectangle,
    right: RuleHyperrectangle,
) -> float:
    """Return full-dimensional IoU; flattened marginal overlap is insufficient."""

    if left.key != right.key:
        return 0.0
    left_volume = math.prod(
        dimension.upper - dimension.lower for dimension in left.dimensions
    )
    right_volume = math.prod(
        dimension.upper - dimension.lower for dimension in right.dimensions
    )
    intersection_volume = math.prod(
        max(
            0.0,
            min(left_dimension.upper, right_dimension.upper)
            - max(left_dimension.lower, right_dimension.lower),
        )
        for left_dimension, right_dimension in zip(
            left.dimensions,
            right.dimensions,
            strict=True,
        )
    )
    union_volume = left_volume + right_volume - intersection_volume
    if union_volume == 0.0:
        return float(left.bounds_key == right.bounds_key)
    return intersection_volume / union_volume


def _joint_geometry_iou(
    left: SeededRuleRegions,
    right: SeededRuleRegions,
) -> float | None:
    """Optimally pair rectangles only within identical action/dimension groups."""

    left_groups: dict[tuple[str, tuple[str, ...]], list[RuleHyperrectangle]] = {}
    right_groups: dict[tuple[str, tuple[str, ...]], list[RuleHyperrectangle]] = {}
    for rectangle in left.hyperrectangles:
        left_groups.setdefault(rectangle.key, []).append(rectangle)
    for rectangle in right.hyperrectangles:
        right_groups.setdefault(rectangle.key, []).append(rectangle)
    if set(left_groups) != set(right_groups) or any(
        len(left_groups[key]) != len(right_groups[key]) for key in left_groups
    ):
        return None
    matched_scores: list[float] = []
    for key in sorted(left_groups):
        left_group = left_groups[key]
        right_group = right_groups[key]
        matched_scores.extend(
            _hyperrectangle_iou(left_rectangle, right_rectangle)
            for left_rectangle, right_rectangle in zip(
                sorted(left_group, key=lambda item: item.bounds_key),
                sorted(right_group, key=lambda item: item.bounds_key),
                strict=True,
            )
        )
    return sum(matched_scores) / len(matched_scores) if matched_scores else None


class Phase0AcceptanceEvaluator:
    """Evaluate the revision plan's Phase-0 gates from precomputed evidence."""

    def __init__(
        self,
        *,
        minimum_held_out_improvement: float = 0.0,
        minimum_region_iou: float = 0.5,
        action_min_noop_samples: int = 30,
    ) -> None:
        self.minimum_held_out_improvement = _finite(
            minimum_held_out_improvement,
            name="minimum_held_out_improvement",
        )
        self.minimum_region_iou = _finite(
            minimum_region_iou,
            name="minimum_region_iou",
        )
        if not 0.0 <= self.minimum_region_iou <= 1.0:
            raise ValueError("minimum_region_iou must be in [0, 1]")
        if (
            isinstance(action_min_noop_samples, bool)
            or not isinstance(action_min_noop_samples, int)
            or action_min_noop_samples < 1
        ):
            raise ValueError("action_min_noop_samples must be a positive integer")
        self.action_min_noop_samples = action_min_noop_samples

    def evaluate(
        self,
        rulebook: FrozenRulebookPolicy,
        inputs: Phase0ValidationInputs,
    ) -> Phase0AcceptanceReport:
        if not isinstance(rulebook, FrozenRulebookPolicy):
            raise TypeError("rulebook must be FrozenRulebookPolicy")
        if not isinstance(inputs, Phase0ValidationInputs):
            raise TypeError("inputs must be Phase0ValidationInputs")

        configured_keys, vocabulary_passed = _configured_action_keys(rulebook)
        vocabulary_check = AcceptanceCheck(
            name="exact_five_action_vocabulary",
            passed=vocabulary_passed,
            detail=(
                "Frozen action configuration matches the shared five identities."
                if vocabulary_passed
                else "Frozen action configuration differs from the shared five identities."
            ),
            metrics={
                "expected_count": len(PHASE0_ACTION_KEYS),
                "configured_count": (
                    len(configured_keys)
                    if configured_keys
                    else len(PHASE0_ACTION_KEYS)
                    if vocabulary_passed
                    else 0
                ),
            },
        )

        action_rules = tuple(rulebook.action_rules)
        ratio_rule_count = sum(
            bool(CAPACITY_RATIO_FEATURES & set(rule.condition.intervals))
            for rule in action_rules
        )
        distance_confound_rule_count = sum(
            any(_is_distance_confound(name) for name in rule.condition.intervals)
            for rule in action_rules
        )
        distance_only_count = sum(
            any(_is_distance_confound(name) for name in rule.condition.intervals)
            and not (CAPACITY_RATIO_FEATURES & set(rule.condition.intervals))
            for rule in action_rules
        )
        capacity_passed = (
            bool(action_rules)
            and ratio_rule_count == len(action_rules)
            and distance_confound_rule_count == 0
        )
        capacity_check = AcceptanceCheck(
            name="capacity_ratio_predicates",
            passed=capacity_passed,
            detail=(
                "Every published action rule uses a capacity-ratio predicate without "
                "a direct distance-to-resource/final confound."
                if capacity_passed
                else "At least one action rule is missing a capacity-ratio predicate "
                "or contains a direct distance-to-resource/final confound."
            ),
            metrics={
                "action_rule_count": len(action_rules),
                "capacity_ratio_rule_count": ratio_rule_count,
                "distance_confound_rule_count": distance_confound_rule_count,
                "distance_only_rule_count": distance_only_count,
            },
        )

        expected_source_ids = {rule.source_rule_id for rule in action_rules}
        evidence_by_source = inputs.action_certification_evidence
        evidence_source_ids = set(evidence_by_source)
        missing_source_ids = expected_source_ids - evidence_source_ids
        unexpected_source_ids = evidence_source_ids - expected_source_ids
        matching_evidence = tuple(
            evidence_by_source[source_rule_id]
            for source_rule_id in sorted(expected_source_ids & evidence_source_ids)
        )
        insufficient_samples = tuple(
            evidence
            for evidence in matching_evidence
            if evidence.noop_samples < self.action_min_noop_samples
        )
        nonpositive_lcbs = tuple(
            evidence for evidence in matching_evidence if evidence.noop_lcb <= 0.0
        )
        certified_passed = (
            bool(action_rules)
            and not (
                missing_source_ids
                or unexpected_source_ids
                or insufficient_samples
                or nonpositive_lcbs
            )
            and all(rule.w > 0.0 and rule.precision > 0.0 for rule in action_rules)
        )
        certification_check = AcceptanceCheck(
            name="positive_noop_grounded_certification",
            passed=certified_passed,
            detail=(
                "Every frozen action rule has threshold-satisfying, positive-LCB no-op evidence."
                if certified_passed
                else "Frozen action certification evidence is missing, stale, or below its gate."
            ),
            metrics={
                "certified_action_rule_count": len(action_rules),
                "certification_evidence_count": len(evidence_by_source),
                "required_noop_samples": self.action_min_noop_samples,
                "missing_evidence_count": len(missing_source_ids),
                "unexpected_evidence_count": len(unexpected_source_ids),
                "insufficient_sample_count": len(insufficient_samples),
                "nonpositive_lcb_count": len(nonpositive_lcbs),
                "minimum_noop_samples": min(
                    (evidence.noop_samples for evidence in matching_evidence),
                    default=0,
                ),
                "minimum_noop_lcb": min(
                    (evidence.noop_lcb for evidence in matching_evidence),
                    default=0.0,
                ),
                "minimum_effect": (min((rule.w for rule in action_rules), default=0.0)),
            },
        )

        decisions = tuple(inputs.train_deploy_decisions)
        agreement_count = sum(
            decision.training_action == decision.deployment_action
            for decision in decisions
        )
        exploration_disabled = all(
            decision.exploration_rate == 0.0 for decision in decisions
        )
        agreement_passed = (
            bool(decisions)
            and exploration_disabled
            and agreement_count == len(decisions)
        )
        agreement_check = AcceptanceCheck(
            name="exploit_deployment_agreement",
            passed=agreement_passed,
            detail=(
                "Exploration-disabled training exploit matches deployment."
                if agreement_passed
                else "Training/deployment actions disagree or exploration was enabled."
            ),
            metrics={
                "sample_count": len(decisions),
                "agreement_count": agreement_count,
                "exploration_disabled": exploration_disabled,
            },
        )

        held_out = tuple(inputs.held_out_comparisons)
        mean_improvement = (
            sum(item.improvement for item in held_out) / len(held_out)
            if held_out
            else 0.0
        )
        held_out_provenance_complete = bool(held_out) and all(
            item.provenance is not None for item in held_out
        )
        held_out_passed = bool(
            held_out_provenance_complete
            and mean_improvement > self.minimum_held_out_improvement
        )
        held_out_check = AcceptanceCheck(
            name="held_out_beats_permanent_no_op",
            passed=held_out_passed,
            detail=(
                "Mean paired held-out improvement exceeds permanent no-op."
                if held_out_passed
                else "Held-out improvement does not exceed permanent no-op."
            ),
            metrics={
                "sample_count": len(held_out),
                "provenance_complete": held_out_provenance_complete,
                "mean_improvement": mean_improvement,
                "required_improvement": self.minimum_held_out_improvement,
            },
        )

        vanilla_held_out = tuple(inputs.vanilla_held_out_comparisons)
        vanilla_check = None
        if vanilla_held_out:
            paired = tuple(zip(held_out, vanilla_held_out, strict=True))
            causal_mean = sum(causal.policy_score for causal, _ in paired) / len(paired)
            vanilla_mean = sum(vanilla.policy_score for _, vanilla in paired) / len(
                paired
            )
            causal_improvement = sum(causal.improvement for causal, _ in paired) / len(
                paired
            )
            vanilla_improvement = sum(
                vanilla.improvement for _, vanilla in paired
            ) / len(paired)
            deltas = tuple(
                causal.policy_score - vanilla.policy_score for causal, vanilla in paired
            )
            vanilla_check = AcceptanceCheck(
                name="paired_vanilla_accuracy_control",
                passed=True,
                detail=(
                    "Vanilla accuracy control is paired on the exact held-out scenarios."
                ),
                metrics={
                    "sample_count": len(paired),
                    "mean_causal_policy_score": causal_mean,
                    "mean_vanilla_policy_score": vanilla_mean,
                    "mean_causal_improvement": causal_improvement,
                    "mean_vanilla_improvement": vanilla_improvement,
                    "mean_causal_minus_vanilla": sum(deltas) / len(deltas),
                    "causal_win_count": sum(delta > 0.0 for delta in deltas),
                    "vanilla_win_count": sum(delta < 0.0 for delta in deltas),
                    "tie_count": sum(delta == 0.0 for delta in deltas),
                },
            )

        path_result = inputs.path_stretch_ablation
        path_published = any(
            rule.action.key == PATH_STRETCH_ACTION_KEY for rule in action_rules
        )
        path_passed = bool(
            path_published
            and path_result is not None
            and path_result.action_key == PATH_STRETCH_ACTION_KEY
            and path_result.oracle_score is not None
            and path_result.geometry_only_score is not None
            and path_result.has_authenticated_runtime_provenance
        )
        path_check = AcceptanceCheck(
            name="path_stretch_oracle_and_geometry_ablation",
            passed=path_passed,
            detail=(
                "Oracle macro-action and geometry-only ablation are both reported."
                if path_passed
                else "Path-stretch naming, publication, or ablation evidence is missing."
            ),
            metrics={
                "path_rule_published": path_published,
                "reported_action_key": (
                    path_result.action_key if path_result is not None else ""
                ),
                "oracle_result_present": bool(
                    path_result is not None and path_result.oracle_score is not None
                ),
                "geometry_result_present": bool(
                    path_result is not None
                    and path_result.geometry_only_score is not None
                ),
                "runtime_provenance_authenticated": bool(
                    path_result is not None
                    and path_result.has_authenticated_runtime_provenance
                ),
            },
        )

        refresh = tuple(inputs.refresh_interval_results)
        signatures = tuple(result.signature for result in refresh)
        unique_intervals = len({result.refresh_interval_epochs for result in refresh})
        publication_backed = bool(refresh) and all(
            result.publication_count > 0 for result in refresh
        )
        refresh_passed = bool(
            len(refresh) >= 2
            and unique_intervals == len(refresh)
            and len(set(signatures)) == 1
            and publication_backed
        )
        refresh_check = AcceptanceCheck(
            name="refresh_interval_robustness",
            passed=refresh_passed,
            detail=(
                "All tested refresh intervals publish and preserve the qualitative concept."
                if refresh_passed
                else (
                    "Refresh intervals lack real publications, reverse, or "
                    "incompletely sample the concept."
                )
            ),
            metrics={
                "interval_count": len(refresh),
                "unique_signature_count": len(set(signatures)),
                "publication_backed": publication_backed,
                "minimum_publication_count": min(
                    (result.publication_count for result in refresh),
                    default=0,
                ),
                "minimum_latest_generation": min(
                    (result.latest_publication_generation for result in refresh),
                    default=0,
                ),
            },
        )

        snapshots = tuple(inputs.seeded_rule_regions)
        seeds_unique = len({snapshot.seed for snapshot in snapshots}) == len(snapshots)
        pairwise_means: list[float] = []
        comparable_geometry = bool(snapshots) and all(
            snapshot.hyperrectangles for snapshot in snapshots
        )
        if len(snapshots) >= 2 and comparable_geometry:
            for left, right in combinations(snapshots, 2):
                pairwise_iou = _joint_geometry_iou(left, right)
                if pairwise_iou is None:
                    comparable_geometry = False
                    pairwise_means.clear()
                    break
                pairwise_means.append(pairwise_iou)
        minimum_pairwise_mean = min(pairwise_means, default=0.0)
        seeded_passed = bool(
            len(snapshots) >= 2
            and seeds_unique
            and comparable_geometry
            and minimum_pairwise_mean >= self.minimum_region_iou
        )
        seeded_check = AcceptanceCheck(
            name="seeded_rule_region_comparability",
            passed=seeded_passed,
            detail=(
                "Seeded runs publish sufficiently overlapping joint rule hyperrectangles."
                if seeded_passed
                else (
                    "Seeded joint rule geometry is missing, mismatched, or "
                    "insufficiently overlapping."
                )
            ),
            metrics={
                "seed_count": len(snapshots),
                "joint_geometry_comparable": comparable_geometry,
                "minimum_pairwise_mean_iou": minimum_pairwise_mean,
                "required_iou": self.minimum_region_iou,
            },
        )

        return Phase0AcceptanceReport(
            rulebook_fingerprint=rulebook.policy_fingerprint(),
            checks=(
                vocabulary_check,
                capacity_check,
                certification_check,
                agreement_check,
                held_out_check,
                *((vanilla_check,) if vanilla_check is not None else ()),
                path_check,
                refresh_check,
                seeded_check,
            ),
        )


def evaluate_phase0_acceptance(
    rulebook: FrozenRulebookPolicy,
    inputs: Phase0ValidationInputs,
    *,
    minimum_held_out_improvement: float = 0.0,
    minimum_region_iou: float = 0.5,
    action_min_noop_samples: int = 30,
) -> Phase0AcceptanceReport:
    return Phase0AcceptanceEvaluator(
        minimum_held_out_improvement=minimum_held_out_improvement,
        minimum_region_iou=minimum_region_iou,
        action_min_noop_samples=action_min_noop_samples,
    ).evaluate(rulebook, inputs)


__all__ = [
    "AcceptanceCheck",
    "ActionCertificationEvidence",
    "AuthenticatedRuntimeManifest",
    "CAPACITY_RATIO_FEATURES",
    "HeldOutComparison",
    "HeldOutOutcomeDetails",
    "HeldOutRolloutProvenance",
    "PATH_STRETCH_ACTION_KEY",
    "PHASE0_ACTION_KEYS",
    "PathStretchAblationResult",
    "Phase0AcceptanceEvaluator",
    "Phase0AcceptanceReport",
    "Phase0ValidationInputs",
    "RefreshIntervalResult",
    "RuleHyperrectangle",
    "RuleRegion",
    "SeededRuleRegions",
    "TrainDeployDecision",
    "evaluate_phase0_acceptance",
]
