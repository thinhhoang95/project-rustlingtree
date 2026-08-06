"""Immutable certified rulebook and deterministic deployment arbitration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from hailmary.actions.vocabulary import action_vocabulary
from hailmary.config import FeatureConfig, ScenarioConfig, TemplateConfig
from hailmary.features.schema import (
    FeatureSchema,
    leader_follower_feature_schema,
)
from hailmary.ids import canonical_data, content_hash as hash_content
from hailmary.learning.certification import (
    CertificationThresholds,
    FrozenActionRule,
    FrozenVetoRule,
    certify_population,
)
from hailmary.learning.modes import (
    CAUSAL_CREDIT_MODE,
    VANILLA_ACCURACY_CREDIT_MODE,
    validate_credit_mode,
)
from hailmary.learning.rules import MutableRule, RuleAction


_TIE_BREAK = ("score_desc", "anchor_id", "lever", "band")
FROZEN_RULEBOOK_VERSION = "hailmary.frozen_rulebook.v2"


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


@runtime_checkable
class RulebookContextAdapter(Protocol):
    """Adapter accepted by :meth:`FrozenRulebookPolicy.select_action`."""

    def rulebook_records(self) -> Iterable[Any]: ...


@dataclass(frozen=True)
class RulebookDecisionRecord:
    """One anchor's immutable vector and currently feasible candidates."""

    anchor_id: str
    features: Mapping[str, float] | Any
    candidates: tuple[Any, ...] | Sequence[Any]
    role_type: str = "leader_follower"
    schema_hash: str = ""

    def __post_init__(self) -> None:
        anchor_id = str(self.anchor_id).strip()
        role_type = str(self.role_type).strip()
        if not anchor_id or not role_type:
            raise ValueError("decision record anchor_id and role_type cannot be empty")
        inferred_schema = self.schema_hash or getattr(self.features, "schema_hash", "")
        schema_hash = str(inferred_schema).strip()
        if not schema_hash:
            raise ValueError("decision record requires a feature schema hash")
        named = getattr(self.features, "named", self.features)
        if not isinstance(named, Mapping):
            raise TypeError(
                "decision record features must be a mapping or FeatureVector"
            )
        normalized_features: dict[str, float] = {}
        for raw_name, raw_value in named.items():
            name = str(raw_name)
            value = float(raw_value)
            if not name or not math.isfinite(value):
                raise ValueError(
                    "decision record feature names and values must be finite"
                )
            normalized_features[name] = value
        if isinstance(self.candidates, (str, bytes)) or not isinstance(
            self.candidates, Sequence
        ):
            raise TypeError("decision record candidates must be a sequence")
        object.__setattr__(self, "anchor_id", anchor_id)
        object.__setattr__(self, "role_type", role_type)
        object.__setattr__(self, "schema_hash", schema_hash)
        object.__setattr__(
            self,
            "features",
            MappingProxyType(dict(sorted(normalized_features.items()))),
        )
        object.__setattr__(self, "candidates", tuple(self.candidates))

    @classmethod
    def coerce(
        cls,
        record: Any,
        *,
        default_role_type: str,
        default_schema_hash: str,
    ) -> "RulebookDecisionRecord":
        if isinstance(record, cls):
            return record
        if isinstance(record, Mapping):
            getter = record.get
        else:

            def getter(name: str, default: Any = None) -> Any:
                return getattr(record, name, default)

        features = getter("features", None)
        if features is None:
            features = getter("feature_vector", getter("vector", None))
        candidates = getter("candidates", None)
        if candidates is None:
            candidates = getter(
                "action_candidates",
                getter("feasible_actions", None),
            )
        if features is None or candidates is None:
            raise TypeError(
                "rulebook records must supply features/vector and candidates"
            )
        return cls(
            anchor_id=str(getter("anchor_id", "")),
            features=features,
            candidates=tuple(candidates),
            role_type=str(getter("role_type", default_role_type)),
            schema_hash=str(getter("schema_hash", "") or default_schema_hash),
        )


@dataclass(frozen=True, slots=True)
class RulebookDecision:
    candidate: Any | None
    anchor_id: str | None
    action: RuleAction | None
    score: float
    vetoed_anchor_ids: tuple[str, ...] = ()

    @property
    def is_no_op(self) -> bool:
        return self.action is None or self.action.is_no_op


@dataclass(frozen=True, slots=True)
class RankedRival:
    candidate: Any
    action: RuleAction
    score: float
    source_rule_ids: tuple[str, ...]


def _candidate_feasible(candidate: Any) -> bool:
    if isinstance(candidate, Mapping):
        return bool(candidate.get("feasible", True))
    return bool(getattr(candidate, "feasible", True))


def _records_from_context(context: Any) -> Iterable[Any]:
    for name in ("rulebook_records", "decision_records"):
        provider = getattr(context, name, None)
        if provider is not None:
            return provider() if callable(provider) else provider
    if isinstance(context, Mapping) and "records" in context:
        return context["records"]
    records = getattr(context, "records", None)
    if records is not None:
        return records() if callable(records) else records
    if isinstance(context, Iterable) and not isinstance(context, (str, bytes, Mapping)):
        return context
    raise TypeError(
        "rulebook context must expose rulebook_records()/decision_records or be an iterable of records"
    )


@dataclass(frozen=True)
class FrozenRulebookPolicy:
    """Content-addressed policy detached from the mutable population."""

    feature_schema_hash: str
    credit_mode: str = CAUSAL_CREDIT_MODE
    certification_generation: int = 0
    contender_min_samples: int = 2
    action_rules: tuple[FrozenActionRule, ...] | Sequence[FrozenActionRule] = ()
    veto_rules: tuple[FrozenVetoRule, ...] | Sequence[FrozenVetoRule] = ()
    role_type: str = "leader_follower"
    action_configuration: Mapping[str, Any] | str | None = None
    tie_break: tuple[str, ...] | Sequence[str] = _TIE_BREAK
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.feature_schema_hash) is not str or type(self.role_type) is not str:
            raise TypeError(
                "rulebook feature_schema_hash and role_type must be strings"
            )
        schema_hash = self.feature_schema_hash.strip()
        role_type = self.role_type.strip()
        mode = validate_credit_mode(
            self.credit_mode,
            name="rulebook credit_mode",
        )
        if not schema_hash or not role_type:
            raise ValueError(
                "rulebook feature_schema_hash and role_type cannot be empty"
            )
        if (
            type(self.certification_generation) is not int
            or self.certification_generation < 0
        ):
            raise ValueError("certification_generation must be a non-negative integer")
        if (
            type(self.contender_min_samples) is not int
            or self.contender_min_samples < 2
        ):
            raise ValueError("contender_min_samples must be an integer of at least two")
        tie_break = tuple(str(item) for item in self.tie_break)
        if tie_break != _TIE_BREAK:
            raise ValueError(f"tie_break must be {_TIE_BREAK!r}")

        actions = tuple(
            FrozenActionRule.from_dict(rule.to_dict()) for rule in self.action_rules
        )
        vetoes = tuple(
            FrozenVetoRule.from_dict(rule.to_dict()) for rule in self.veto_rules
        )
        if mode == CAUSAL_CREDIT_MODE and any(rule.action.is_no_op for rule in actions):
            raise ValueError(
                "causal rulebooks represent no-op classifiers only as scoped vetoes"
            )
        if mode == VANILLA_ACCURACY_CREDIT_MODE and vetoes:
            raise ValueError("vanilla_accuracy rulebooks cannot contain causal vetoes")
        for rule in (*actions, *vetoes):
            if rule.condition.role_type != role_type:
                raise ValueError("all frozen rules must use the rulebook role_type")
            if rule.condition.schema_hash != schema_hash:
                raise ValueError(
                    "all frozen rules must use the rulebook feature schema"
                )
        source_ids = [rule.source_rule_id for rule in (*actions, *vetoes)]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("a source rule can appear only once in a frozen rulebook")
        actions = tuple(
            sorted(
                actions,
                key=lambda rule: (
                    rule.action.key,
                    rule.source_rule_id,
                    hash_content(
                        rule.condition.to_dict(), namespace="hailmary.rule_condition"
                    ),
                ),
            )
        )
        vetoes = tuple(sorted(vetoes, key=lambda rule: rule.source_rule_id))

        raw_configuration: Any
        if self.action_configuration is None:
            raw_configuration = action_vocabulary().payload
        else:
            raw_configuration = self.action_configuration
        normalized_configuration = canonical_data(raw_configuration)
        if not (
            isinstance(normalized_configuration, dict)
            or (isinstance(normalized_configuration, str) and normalized_configuration)
        ):
            raise ValueError(
                "action_configuration must be a non-empty mapping or hash string"
            )
        stored_configuration = _deep_freeze(normalized_configuration)

        object.__setattr__(self, "feature_schema_hash", schema_hash)
        object.__setattr__(self, "role_type", role_type)
        object.__setattr__(self, "credit_mode", mode)
        object.__setattr__(
            self, "certification_generation", self.certification_generation
        )
        object.__setattr__(self, "contender_min_samples", self.contender_min_samples)
        object.__setattr__(self, "action_rules", actions)
        object.__setattr__(self, "veto_rules", vetoes)
        object.__setattr__(self, "tie_break", tie_break)
        object.__setattr__(self, "action_configuration", stored_configuration)
        object.__setattr__(
            self,
            "content_hash",
            hash_content(self._hash_payload(), namespace=FROZEN_RULEBOOK_VERSION),
        )

    @property
    def action_config_hash(self) -> str:
        configuration = canonical_data(self.action_configuration)
        if isinstance(configuration, str):
            return configuration
        raw_namespace = configuration.get("schema_version")
        namespace = (
            str(raw_namespace).strip()
            if raw_namespace is not None and str(raw_namespace).strip()
            else "hailmary.action_configuration"
        )
        return hash_content(configuration, namespace=namespace)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "artifact_version": FROZEN_RULEBOOK_VERSION,
            "feature_schema_hash": self.feature_schema_hash,
            "credit_mode": self.credit_mode,
            "role_type": self.role_type,
            "certification_generation": self.certification_generation,
            "contender_min_samples": self.contender_min_samples,
            "action_rules": [rule.to_dict() for rule in self.action_rules],
            "veto_rules": [rule.to_dict() for rule in self.veto_rules],
            "tie_break": list(self.tie_break),
            "action_configuration": canonical_data(self.action_configuration),
        }

    def policy_fingerprint(self) -> str:
        return self.content_hash

    def _normalize_records(
        self, records: Iterable[Any]
    ) -> tuple[RulebookDecisionRecord, ...]:
        normalized = tuple(
            RulebookDecisionRecord.coerce(
                record,
                default_role_type=self.role_type,
                default_schema_hash=self.feature_schema_hash,
            )
            for record in records
        )
        anchor_ids = [record.anchor_id for record in normalized]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("rulebook decision records must have unique anchor IDs")
        return tuple(sorted(normalized, key=lambda record: record.anchor_id))

    @staticmethod
    def _available_candidates(
        record: RulebookDecisionRecord,
    ) -> dict[str, tuple[RuleAction, Any]]:
        available: dict[str, tuple[RuleAction, Any]] = {}
        for candidate in record.candidates:
            if not _candidate_feasible(candidate):
                continue
            try:
                action = RuleAction.from_candidate(candidate)
            except (KeyError, TypeError, ValueError):
                continue
            if action.key in available:
                raise ValueError(
                    f"anchor {record.anchor_id!r} has duplicate candidate {action.key!r}"
                )
            available[action.key] = (action, candidate)
        return available

    def rank_rivals(
        self,
        record: RulebookDecisionRecord | Any,
        *,
        selected_action: RuleAction | Any,
    ) -> tuple[RankedRival, ...]:
        """Rank feasible certified rivals using only snapshot evolution fields."""

        normalized = RulebookDecisionRecord.coerce(
            record,
            default_role_type=self.role_type,
            default_schema_hash=self.feature_schema_hash,
        )
        if (
            normalized.role_type != self.role_type
            or normalized.schema_hash != self.feature_schema_hash
        ):
            return ()
        selected = RuleAction.from_candidate(selected_action)
        available = self._available_candidates(normalized)

        ranked: list[RankedRival] = []
        for action_key, (action, candidate) in available.items():
            if action == selected or (
                self.credit_mode == CAUSAL_CREDIT_MODE and action.is_no_op
            ):
                continue
            matching = tuple(
                rule
                for rule in self.action_rules
                if rule.action.key == action_key
                and rule.rival_samples >= self.contender_min_samples
                and rule.condition.matches(
                    normalized.features,
                    role_type=normalized.role_type,
                    schema_hash=normalized.schema_hash,
                )
            )
            if not matching:
                continue
            if self.credit_mode == VANILLA_ACCURACY_CREDIT_MODE:
                precision_sum = sum(rule.precision for rule in matching)
                score = (
                    sum(rule.w * rule.precision for rule in matching) / precision_sum
                )
            else:
                sample_weight = sum(rule.rival_samples for rule in matching)
                score = (
                    sum(rule.rival_lcb * rule.rival_samples for rule in matching)
                    / sample_weight
                )
            if not math.isfinite(score):
                raise ArithmeticError("snapshot rival score is not finite")
            ranked.append(
                RankedRival(
                    candidate=candidate,
                    action=action,
                    score=float(score),
                    source_rule_ids=tuple(
                        sorted(rule.source_rule_id for rule in matching)
                    ),
                )
            )
        return tuple(
            sorted(
                ranked,
                key=lambda item: (
                    -item.score,
                    item.action.lever.value,
                    item.action.band,
                ),
            )
        )

    def strongest_rival(
        self,
        record: RulebookDecisionRecord | Any,
        *,
        selected_action: RuleAction | Any,
    ) -> Any | None:
        ranked = self.rank_rivals(record, selected_action=selected_action)
        return None if not ranked else ranked[0].candidate

    def evaluate(self, records: Iterable[Any]) -> RulebookDecision:
        normalized = self._normalize_records(records)
        scored: list[tuple[float, str, RuleAction, Any]] = []
        no_ops: list[tuple[str, RuleAction, Any]] = []
        vetoed: list[str] = []

        for record in normalized:
            matched_no_op = False
            if (
                record.role_type != self.role_type
                or record.schema_hash != self.feature_schema_hash
            ):
                continue
            available = self._available_candidates(record)
            no_op = available.get(RuleAction.no_op().key)
            if no_op is not None:
                no_ops.append((record.anchor_id, no_op[0], no_op[1]))

            is_vetoed = self.credit_mode == CAUSAL_CREDIT_MODE and any(
                rule.condition.matches(
                    record.features,
                    role_type=record.role_type,
                    schema_hash=record.schema_hash,
                )
                for rule in self.veto_rules
            )
            if is_vetoed:
                vetoed.append(record.anchor_id)
                continue

            for action_key, (action, candidate) in available.items():
                if self.credit_mode == CAUSAL_CREDIT_MODE and action.is_no_op:
                    continue
                matching = tuple(
                    rule
                    for rule in self.action_rules
                    if rule.action.key == action_key
                    and rule.condition.matches(
                        record.features,
                        role_type=record.role_type,
                        schema_hash=record.schema_hash,
                    )
                )
                if not matching:
                    continue
                if action.is_no_op:
                    matched_no_op = True
                precision_sum = sum(rule.precision for rule in matching)
                pooled = (
                    sum(rule.w * rule.precision for rule in matching) / precision_sum
                )
                if not math.isfinite(pooled):
                    raise ArithmeticError(
                        "precision-pooled rulebook score is not finite"
                    )
                if self.credit_mode == VANILLA_ACCURACY_CREDIT_MODE or pooled > 0.0:
                    scored.append((float(pooled), record.anchor_id, action, candidate))

            if (
                self.credit_mode == VANILLA_ACCURACY_CREDIT_MODE
                and no_op is not None
                and not matched_no_op
            ):
                scored.append((0.0, record.anchor_id, no_op[0], no_op[1]))

        if scored:
            score, anchor_id, action, candidate = min(
                scored,
                key=lambda item: (
                    -item[0],
                    item[1],
                    item[2].lever.value,
                    item[2].band,
                ),
            )
            return RulebookDecision(
                candidate=candidate,
                anchor_id=anchor_id,
                action=action,
                score=score,
                vetoed_anchor_ids=tuple(vetoed),
            )

        if no_ops:
            anchor_id, action, candidate = min(
                no_ops,
                key=lambda item: (item[0], item[1].lever.value, item[1].band),
            )
            return RulebookDecision(
                candidate=candidate,
                anchor_id=anchor_id,
                action=action,
                score=0.0,
                vetoed_anchor_ids=tuple(vetoed),
            )
        return RulebookDecision(
            candidate=None,
            anchor_id=None,
            action=None,
            score=0.0,
            vetoed_anchor_ids=tuple(vetoed),
        )

    def decide(self, records: Iterable[Any]) -> Any | None:
        return self.evaluate(records).candidate

    def select_action(self, context: RulebookContextAdapter | Any) -> Any | None:
        return self.decide(_records_from_context(context))

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "content_hash": self.content_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenRulebookPolicy":
        if not isinstance(payload, Mapping):
            raise TypeError("rulebook payload must be a mapping")
        expected_fields = {
            "artifact_version",
            "feature_schema_hash",
            "credit_mode",
            "role_type",
            "certification_generation",
            "contender_min_samples",
            "action_rules",
            "veto_rules",
            "tie_break",
            "action_configuration",
            "content_hash",
        }
        actual_fields = set(payload)
        if actual_fields != expected_fields:
            if actual_fields == expected_fields - {"content_hash"}:
                raise ValueError("canonical schema requires content_hash")
            raise ValueError(
                "rulebook payload fields must exactly match the canonical schema"
            )
        if payload["artifact_version"] != FROZEN_RULEBOOK_VERSION:
            raise ValueError("unsupported frozen rulebook artifact version")
        credit_mode = validate_credit_mode(
            payload["credit_mode"],
            name="serialized rulebook credit_mode",
        )

        feature_schema_hash = payload["feature_schema_hash"]
        role_type = payload["role_type"]
        if type(feature_schema_hash) is not str or type(role_type) is not str:
            raise TypeError(
                "serialized rulebook feature_schema_hash and role_type must be strings"
            )
        if not feature_schema_hash.strip() or not role_type.strip():
            raise ValueError(
                "serialized rulebook feature_schema_hash and role_type cannot be empty"
            )

        certification_generation = payload["certification_generation"]
        contender_min_samples = payload["contender_min_samples"]
        if type(certification_generation) is not int or certification_generation < 0:
            raise ValueError("certification_generation must be a non-negative integer")
        if type(contender_min_samples) is not int or contender_min_samples < 2:
            raise ValueError("contender_min_samples must be an integer of at least two")

        raw_action_rules = payload["action_rules"]
        raw_veto_rules = payload["veto_rules"]
        raw_tie_break = payload["tie_break"]
        for name, value in (
            ("action_rules", raw_action_rules),
            ("veto_rules", raw_veto_rules),
            ("tie_break", raw_tie_break),
        ):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"serialized rulebook {name} must be a sequence")
        if any(type(item) is not str for item in raw_tie_break):
            raise TypeError("serialized rulebook tie_break entries must be strings")

        expected_hash = payload["content_hash"]
        if type(expected_hash) is not str or not expected_hash.strip():
            raise TypeError(
                "serialized rulebook content_hash must be a non-empty string"
            )

        rulebook = cls(
            feature_schema_hash=feature_schema_hash,
            credit_mode=credit_mode,
            role_type=role_type,
            certification_generation=certification_generation,
            contender_min_samples=contender_min_samples,
            action_rules=tuple(
                FrozenActionRule.from_dict(item) for item in raw_action_rules
            ),
            veto_rules=tuple(FrozenVetoRule.from_dict(item) for item in raw_veto_rules),
            tie_break=tuple(raw_tie_break),
            action_configuration=payload["action_configuration"],
        )
        if expected_hash != rulebook.content_hash:
            raise ValueError(
                "serialized rulebook content hash does not match its contents"
            )
        return rulebook

    @classmethod
    def from_population(
        cls,
        rules: Iterable[MutableRule],
        *,
        feature_schema_hash: str,
        certification_generation: int,
        contender_min_samples: int | None = None,
        thresholds: CertificationThresholds | Any | None = None,
        credit_mode: str = CAUSAL_CREDIT_MODE,
        action_configuration: Mapping[str, Any] | str | None = None,
        role_type: str = "leader_follower",
    ) -> "FrozenRulebookPolicy":
        mode = validate_credit_mode(credit_mode, name="rulebook credit_mode")
        actions, vetoes = certify_population(
            rules,
            thresholds,
            credit_mode=mode,
        )
        resolved_contender_min = (
            int(getattr(thresholds, "contender_min_samples", 2))
            if contender_min_samples is None
            else int(contender_min_samples)
        )
        return cls(
            feature_schema_hash=feature_schema_hash,
            credit_mode=mode,
            certification_generation=certification_generation,
            contender_min_samples=resolved_contender_min,
            action_rules=actions,
            veto_rules=vetoes,
            role_type=role_type,
            action_configuration=action_configuration,
        )


@dataclass(frozen=True)
class SimulatorRulebookPolicy:
    """Pure simulator adapter around a content-addressed frozen rulebook."""

    rulebook: FrozenRulebookPolicy
    template_config: TemplateConfig = field(default_factory=TemplateConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    scenario_config: ScenarioConfig = field(default_factory=ScenarioConfig)
    schema: FeatureSchema | None = None
    runtime_configuration_hash: str = ""
    content_hash: str = field(init=False)
    action_vocabulary_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.rulebook, FrozenRulebookPolicy):
            raise TypeError("simulator rulebook wrapper requires FrozenRulebookPolicy")
        schema = (
            leader_follower_feature_schema(self.feature_config.schema_version)
            if self.schema is None
            else self.schema
        )
        if schema.schema_hash != self.rulebook.feature_schema_hash:
            raise ValueError(
                "simulator feature schema does not match the frozen rulebook"
            )
        vocabulary = action_vocabulary(self.template_config)
        frozen_action_config = canonical_data(self.rulebook.action_configuration)
        if isinstance(frozen_action_config, dict):
            if frozen_action_config != canonical_data(vocabulary.payload):
                raise ValueError(
                    "simulator action vocabulary does not match the frozen rulebook"
                )
        elif self.rulebook.action_config_hash != vocabulary.content_hash:
            raise ValueError(
                "simulator action vocabulary hash does not match the frozen rulebook"
            )

        if not isinstance(self.runtime_configuration_hash, str):
            raise TypeError("runtime_configuration_hash must be a string")
        runtime_hash = self.runtime_configuration_hash.strip()
        if not runtime_hash:
            raise ValueError(
                "simulator rulebook wrapper requires runtime_configuration_hash"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(
            self,
            "runtime_configuration_hash",
            runtime_hash,
        )
        object.__setattr__(self, "action_vocabulary_hash", vocabulary.content_hash)
        object.__setattr__(
            self,
            "content_hash",
            hash_content(
                {
                    "rulebook_hash": self.rulebook.content_hash,
                    "action_vocabulary_hash": vocabulary.content_hash,
                    "runtime_configuration_hash": runtime_hash,
                    "feature_schema_hash": schema.schema_hash,
                    "template_config": self.template_config,
                    "feature_config": self.feature_config,
                    "scenario_config": self.scenario_config,
                },
                namespace="hailmary.simulator_rulebook_policy.v1",
            ),
        )

    def policy_fingerprint(self) -> str:
        return self.content_hash

    def _validated_context_simulator(self, context: Any) -> Any:
        simulator = getattr(context, "simulator", None)
        if simulator is None:
            raise TypeError("simulator policy context must expose simulator")
        runtime_hash = getattr(simulator, "runtime_configuration_hash", None)
        if not isinstance(runtime_hash, str) or not runtime_hash.strip():
            raise ValueError(
                "context simulator must expose a nonblank runtime_configuration_hash"
            )
        if runtime_hash != self.runtime_configuration_hash:
            raise ValueError(
                "context simulator runtime_configuration_hash does not match "
                "the rulebook policy"
            )
        return simulator

    def rulebook_records(self, context: Any) -> tuple[RulebookDecisionRecord, ...]:
        simulator = self._validated_context_simulator(context)
        event_batch = getattr(context, "event_batch", None)
        if event_batch is None:
            raise TypeError("simulator policy context must expose event_batch")

        from hailmary.actions import ActionCatalog
        from hailmary.features import (
            build_current_segment_anchors,
            simulator_state_vector,
        )

        catalog = ActionCatalog(self.template_config)
        anchors = build_current_segment_anchors(simulator)
        records: list[RulebookDecisionRecord] = []
        for anchor in anchors.leader_follower:
            candidates = catalog.enumerate_for_batch(
                simulator,
                event_batch,
                anchor_id=anchor.anchor_id,
                bound_flight_id=anchor.follower_id,
                resource_id=anchor.resource_id,
                segment_id=anchor.segment_id,
            )
            if not candidates:
                continue
            vector = simulator_state_vector(
                simulator,
                anchor,
                action_candidates=candidates,
                feature_config=self.feature_config,
                scenario_config=self.scenario_config,
                template_config=self.template_config,
                schema=self.schema,
            )
            records.append(
                RulebookDecisionRecord(
                    anchor_id=anchor.anchor_id,
                    features=vector,
                    candidates=candidates,
                    role_type="leader_follower",
                    schema_hash=vector.schema_hash,
                )
            )
        return tuple(records)

    def select_action(self, context: Any) -> Any | None:
        return self.rulebook.decide(self.rulebook_records(context))


__all__ = [
    "FROZEN_RULEBOOK_VERSION",
    "FrozenRulebookPolicy",
    "RankedRival",
    "RulebookContextAdapter",
    "RulebookDecision",
    "RulebookDecisionRecord",
    "SimulatorRulebookPolicy",
]
