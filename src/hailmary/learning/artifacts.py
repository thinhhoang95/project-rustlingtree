"""Versioned, canonical learning artifacts with atomic persistence."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias

import numpy as np

from hailmary.ids import canonical_data, canonical_json, content_hash as hash_content
from hailmary.learning.certification import (
    CertificationThresholds,
    FrozenActionRule,
    FrozenVetoRule,
    independent_evidence_moments,
)
from hailmary.learning.conditions import RuleCondition
from hailmary.learning.modes import (
    CAUSAL_CREDIT_MODE,
    VANILLA_ACCURACY_CREDIT_MODE,
    validate_credit_mode,
)
from hailmary.learning.rules import MutableRule, RuleAction
from hailmary.learning.population import Population
from hailmary.learning.rulebook import FrozenRulebookPolicy
from hailmary.learning.statistics import OnlineMoments


PUBLISHED_CERTIFICATION_EVIDENCE_VERSION = (
    "hailmary.published_certification_evidence.v1"
)
EVALUATION_SNAPSHOT_VERSION = "hailmary.evaluation_snapshot.v3"
TRAINING_CHECKPOINT_VERSION = "hailmary.training_checkpoint.v2"
EXPORTED_RULEBOOK_VERSION = "hailmary.exported_rulebook.v2"
EPOCH_TRACE_VERSION = "hailmary.epoch_trace.v2"
_CONTENDER_FALLBACK = "mandatory_no_op_no_certified_rival"
_PENDING_DECISION_NAMESPACE = "hailmary.trainer.pending_decision.v1"


def _token(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _epoch(value: Any, name: str = "epoch") -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _frozen_canonical(value: Any) -> Any:
    return _freeze(canonical_data(value))


def _serialized(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serialized(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialized(item) for item in value]
    return value


def _restored(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get("__ndarray__") is True and {"dtype", "shape", "data"}.issubset(
            value
        ):
            array = np.asarray(
                _restored(value["data"]),
                dtype=str(value["dtype"]),
            )
            return array.reshape(tuple(int(item) for item in value["shape"])).copy()
        if set(value) == {"__bytes__"}:
            return base64.b64decode(str(value["__bytes__"]))
        return {str(key): _restored(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_restored(item) for item in value]
    if isinstance(value, list):
        return [_restored(item) for item in value]
    return value


def _frozen_mapping(value: Any, name: str) -> Mapping[str, Any]:
    normalized = canonical_data(value)
    if not isinstance(normalized, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _freeze(normalized)


def _frozen_events(
    values: Sequence[Mapping[str, Any]] | None,
    name: str,
) -> tuple[Mapping[str, Any], ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of mappings")
    events: list[Mapping[str, Any]] = []
    for value in values:
        events.append(_frozen_mapping(value, name))
    return tuple(events)


def _ids(values: Sequence[str] | None, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(_token(value, name) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} cannot contain duplicates")
    return normalized


def _finite(value: Any, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _pending_decision_reference(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint pending_decision must be a mapping or null")
    required = {"kind", "next_epoch", "payload", "fingerprint"}
    if set(value) != required:
        raise ValueError(
            "checkpoint pending_decision must contain exactly "
            + ", ".join(sorted(required))
        )
    kind = value["kind"]
    if kind not in {"event_batch", "decision_reference"}:
        raise ValueError("checkpoint pending_decision has an unsupported kind")
    next_epoch = value["next_epoch"]
    if type(next_epoch) is not int or next_epoch < 1:
        raise ValueError("checkpoint pending_decision next_epoch must be positive")
    payload = canonical_data(value["payload"])
    if payload in (None, "", [], {}):
        raise ValueError("checkpoint pending_decision payload cannot be empty")
    fingerprint = value["fingerprint"]
    if type(fingerprint) is not str or not fingerprint.strip():
        raise ValueError(
            "checkpoint pending_decision fingerprint must be a non-empty string"
        )
    expected = hash_content(
        {"kind": kind, "payload": payload},
        namespace=_PENDING_DECISION_NAMESPACE,
    )
    if fingerprint != expected:
        raise ValueError(
            "checkpoint pending_decision fingerprint does not match its payload"
        )
    return _restored(
        _frozen_canonical(
            {
                "kind": kind,
                "next_epoch": next_epoch,
                "payload": payload,
                "fingerprint": fingerprint,
            }
        )
    )


def _content_checked(
    payload: Mapping[str, Any],
    *,
    namespace: str,
    expected: Any | None,
) -> str:
    computed = hash_content(payload, namespace=namespace)
    if expected is not None and str(expected) != computed:
        raise ValueError("serialized artifact content hash does not match its contents")
    return computed


def _verify_serialized_hash(
    payload: Mapping[str, Any],
    *,
    namespace: str,
) -> None:
    expected = payload.get("content_hash")
    if expected is None:
        raise ValueError("serialized artifact requires content_hash")
    unhashed = dict(payload)
    unhashed.pop("content_hash", None)
    _content_checked(unhashed, namespace=namespace, expected=expected)


def _json_mapping(payload: str | bytes | bytearray) -> Mapping[str, Any]:
    if isinstance(payload, (bytes, bytearray)):
        payload = bytes(payload).decode("utf-8")
    if not isinstance(payload, str):
        raise TypeError("JSON artifact must be text or UTF-8 bytes")
    decoded = json.loads(payload)
    if not isinstance(decoded, Mapping):
        raise TypeError("artifact JSON must contain an object")
    return decoded


class _CanonicalArtifact:
    ARTIFACT_VERSION: ClassVar[str]
    HASH_NAMESPACE: ClassVar[str]

    __slots__ = ("_sealed",)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(f"{type(self).__name__} is immutable")
        object.__setattr__(self, name, value)

    def _seal(self) -> None:
        object.__setattr__(self, "_sealed", True)

    @property
    def artifact_version(self) -> str:
        return self.ARTIFACT_VERSION

    def _payload(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def content_hash(self) -> str:
        return hash_content(self._payload(), namespace=self.HASH_NAMESPACE)

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_hash": self.content_hash}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.to_dict() == other.to_dict()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self), self.content_hash))


class PublishedCertificationEvidence(_CanonicalArtifact):
    """Detached sufficient statistics that justified one published rule."""

    ARTIFACT_VERSION = PUBLISHED_CERTIFICATION_EVIDENCE_VERSION
    HASH_NAMESPACE = PUBLISHED_CERTIFICATION_EVIDENCE_VERSION

    __slots__ = (
        "source_rule_id",
        "credit_mode",
        "rule_kind",
        "_action_payload",
        "_evolution_payload",
        "_deployment_payload",
        "_threshold_payload",
    )

    def __init__(
        self,
        *,
        source_rule_id: str,
        credit_mode: str,
        rule_kind: str,
        action: RuleAction | Mapping[str, Any],
        independent_evolution: OnlineMoments | Mapping[str, Any],
        independent_deployment: OnlineMoments | Mapping[str, Any],
        thresholds: CertificationThresholds | Mapping[str, Any],
    ) -> None:
        source = _token(source_rule_id, "published evidence source_rule_id")
        mode = validate_credit_mode(
            credit_mode,
            name="published evidence credit_mode",
        )
        if type(rule_kind) is not str or rule_kind not in {"action", "veto"}:
            raise ValueError("published evidence rule_kind must be action or veto")
        detached_action = (
            RuleAction.from_dict(action)
            if isinstance(action, Mapping)
            else RuleAction.from_dict(action.to_dict())
            if isinstance(action, RuleAction)
            else None
        )
        if detached_action is None:
            raise TypeError("published evidence action must be RuleAction or mapping")
        evolution = (
            OnlineMoments.from_dict(independent_evolution)
            if isinstance(independent_evolution, Mapping)
            else OnlineMoments.from_dict(independent_evolution.to_dict())
            if isinstance(independent_evolution, OnlineMoments)
            else None
        )
        deployment = (
            OnlineMoments.from_dict(independent_deployment)
            if isinstance(independent_deployment, Mapping)
            else OnlineMoments.from_dict(independent_deployment.to_dict())
            if isinstance(independent_deployment, OnlineMoments)
            else None
        )
        if evolution is None or deployment is None:
            raise TypeError(
                "published evidence ledgers must be OnlineMoments or mappings"
            )
        settings = (
            CertificationThresholds.from_dict(thresholds)
            if isinstance(thresholds, Mapping)
            else CertificationThresholds.from_dict(thresholds.to_dict())
            if isinstance(thresholds, CertificationThresholds)
            else None
        )
        if settings is None:
            raise TypeError(
                "published evidence thresholds must be CertificationThresholds "
                "or mapping"
            )

        self._validate_gate(
            mode=mode,
            rule_kind=rule_kind,
            action=detached_action,
            evolution=evolution,
            deployment=deployment,
            thresholds=settings,
        )
        self.source_rule_id = source
        self.credit_mode = mode
        self.rule_kind = rule_kind
        self._action_payload = _frozen_mapping(detached_action.to_dict(), "action")
        self._evolution_payload = _frozen_mapping(
            evolution.to_dict(),
            "independent_evolution",
        )
        self._deployment_payload = _frozen_mapping(
            deployment.to_dict(),
            "independent_deployment",
        )
        self._threshold_payload = _frozen_mapping(
            settings.to_dict(),
            "certification thresholds",
        )
        self._seal()

    @staticmethod
    def _validate_gate(
        *,
        mode: str,
        rule_kind: str,
        action: RuleAction,
        evolution: OnlineMoments,
        deployment: OnlineMoments,
        thresholds: CertificationThresholds,
    ) -> None:
        if mode == CAUSAL_CREDIT_MODE:
            if rule_kind == "action":
                if action.is_no_op:
                    raise ValueError(
                        "causal action evidence cannot publish the no-op action"
                    )
                if deployment.n < thresholds.action_min_noop_samples:
                    raise ValueError(
                        "causal action evidence is below the no-op sample gate"
                    )
                if (
                    deployment.lcb(
                        thresholds.action_lcb_z,
                        thresholds.variance_floor,
                    )
                    <= 0.0
                ):
                    raise ValueError(
                        "causal action evidence is below the positive no-op LCB gate"
                    )
                return
            if not action.is_no_op:
                raise ValueError("causal veto evidence must bind the no-op action")
            if evolution.n < thresholds.veto_min_rival_samples:
                raise ValueError("causal veto evidence is below the rival sample gate")
            if (
                evolution.lcb(
                    thresholds.veto_lcb_z,
                    thresholds.variance_floor,
                )
                <= 0.0
            ):
                raise ValueError(
                    "causal veto evidence is below the positive rival LCB gate"
                )
            return

        assert mode == VANILLA_ACCURACY_CREDIT_MODE
        if rule_kind != "action":
            raise ValueError("vanilla accuracy publication cannot contain veto rules")
        if (
            evolution.n < thresholds.action_min_noop_samples
            or deployment.n < thresholds.action_min_noop_samples
        ):
            raise ValueError("vanilla evidence is below the prediction/accuracy gate")
        if not 0.0 < evolution.mean <= 1.0:
            raise ValueError("vanilla accuracy evidence mean must be in (0, 1]")

    @property
    def action(self) -> RuleAction:
        return RuleAction.from_dict(_serialized(self._action_payload))

    @property
    def independent_evolution(self) -> OnlineMoments:
        return OnlineMoments.from_dict(_serialized(self._evolution_payload))

    @property
    def independent_deployment(self) -> OnlineMoments:
        return OnlineMoments.from_dict(_serialized(self._deployment_payload))

    @property
    def thresholds(self) -> CertificationThresholds:
        return CertificationThresholds.from_dict(_serialized(self._threshold_payload))

    @property
    def noop_samples(self) -> int:
        """Independent no-op-comparator samples for a causal action rule."""

        self._require_causal_action()
        return self.independent_deployment.n

    @property
    def noop_lcb(self) -> float:
        """Publication-time causal action LCB against permanent no-op."""

        self._require_causal_action()
        settings = self.thresholds
        return self.independent_deployment.lcb(
            settings.action_lcb_z,
            settings.variance_floor,
        )

    def _require_causal_action(self) -> None:
        if self.credit_mode != CAUSAL_CREDIT_MODE or self.rule_kind != "action":
            raise ValueError(
                "noop_samples/noop_lcb are defined only for causal action evidence"
            )

    @classmethod
    def from_rule(
        cls,
        rule: MutableRule,
        certified_rule: FrozenActionRule | FrozenVetoRule,
        *,
        thresholds: CertificationThresholds,
        credit_mode: str,
    ) -> "PublishedCertificationEvidence":
        if not isinstance(rule, MutableRule):
            raise TypeError("published evidence requires a MutableRule")
        if not isinstance(certified_rule, (FrozenActionRule, FrozenVetoRule)):
            raise TypeError("published evidence requires a frozen certified rule")
        if rule.rule_id != certified_rule.source_rule_id:
            raise ValueError("mutable and frozen publication source IDs do not match")
        record = cls(
            source_rule_id=rule.rule_id,
            credit_mode=credit_mode,
            rule_kind=(
                "action" if isinstance(certified_rule, FrozenActionRule) else "veto"
            ),
            action=rule.action,
            independent_evolution=independent_evidence_moments(rule, "evolution"),
            independent_deployment=independent_evidence_moments(rule, "deployment"),
            thresholds=thresholds,
        )
        record.validate_certified_rule(certified_rule)
        return record

    def validate_certified_rule(
        self,
        rule: FrozenActionRule | FrozenVetoRule,
    ) -> None:
        """Reject evidence that cannot have produced the embedded frozen rule."""

        if not isinstance(rule, (FrozenActionRule, FrozenVetoRule)):
            raise TypeError("published evidence must bind a frozen certified rule")
        expected_kind = "action" if isinstance(rule, FrozenActionRule) else "veto"
        expected_action = (
            rule.action if isinstance(rule, FrozenActionRule) else RuleAction.no_op()
        )
        if self.source_rule_id != rule.source_rule_id:
            raise ValueError("published evidence source does not match its frozen rule")
        if self.rule_kind != expected_kind or self.action != expected_action:
            raise ValueError(
                "published evidence kind/action does not match its frozen rule"
            )
        if isinstance(rule, FrozenVetoRule):
            return

        settings = self.thresholds
        evolution = self.independent_evolution
        deployment = self.independent_deployment
        if self.credit_mode == CAUSAL_CREDIT_MODE:
            expected_values = (
                (rule.w, deployment.mean, "weight"),
                (
                    rule.precision,
                    deployment.precision(settings.variance_floor),
                    "precision",
                ),
                (
                    rule.rival_lcb,
                    evolution.lcb(
                        settings.contender_lcb_z,
                        settings.variance_floor,
                    ),
                    "rival LCB",
                ),
            )
            expected_samples = evolution.n
        else:
            expected_values = (
                (rule.w, deployment.mean, "prediction"),
                (rule.precision, evolution.mean, "accuracy"),
                (
                    rule.rival_lcb,
                    deployment.lcb(
                        settings.contender_lcb_z,
                        settings.variance_floor,
                    ),
                    "prediction LCB",
                ),
            )
            expected_samples = deployment.n
        for actual, expected, name in expected_values:
            if not math.isclose(
                actual,
                expected,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"published evidence does not reproduce frozen rule {name}"
                )
        if rule.rival_samples != expected_samples:
            raise ValueError(
                "published evidence does not reproduce frozen rule sample count"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_version": self.ARTIFACT_VERSION,
            "source_rule_id": self.source_rule_id,
            "credit_mode": self.credit_mode,
            "rule_kind": self.rule_kind,
            "action": _serialized(self._action_payload),
            "independent_evolution": _serialized(self._evolution_payload),
            "independent_deployment": _serialized(self._deployment_payload),
            "thresholds": _serialized(self._threshold_payload),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PublishedCertificationEvidence":
        if not isinstance(payload, Mapping):
            raise TypeError("published certification evidence must be a mapping")
        if payload.get("artifact_version") != cls.ARTIFACT_VERSION:
            raise ValueError("unsupported published certification evidence version")
        _verify_serialized_hash(payload, namespace=cls.HASH_NAMESPACE)
        expected_fields = {
            "artifact_version",
            "source_rule_id",
            "credit_mode",
            "rule_kind",
            "action",
            "independent_evolution",
            "independent_deployment",
            "thresholds",
            "content_hash",
        }
        if set(payload) != expected_fields:
            raise ValueError(
                "published certification evidence fields must exactly match "
                "the canonical schema"
            )
        for name in ("source_rule_id", "credit_mode", "rule_kind"):
            if type(payload[name]) is not str:
                raise TypeError(f"published evidence {name} must be a string")
        record = cls(
            source_rule_id=payload["source_rule_id"],
            credit_mode=payload["credit_mode"],
            rule_kind=payload["rule_kind"],
            action=payload["action"],
            independent_evolution=payload["independent_evolution"],
            independent_deployment=payload["independent_deployment"],
            thresholds=payload["thresholds"],
        )
        _content_checked(
            record._payload(),
            namespace=cls.HASH_NAMESPACE,
            expected=payload["content_hash"],
        )
        return record

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
    ) -> "PublishedCertificationEvidence":
        return cls.from_dict(_json_mapping(payload))


class ExportedRulebook(_CanonicalArtifact):
    """Deployment-only rulebook with no comparator or population metadata."""

    ARTIFACT_VERSION = EXPORTED_RULEBOOK_VERSION
    HASH_NAMESPACE = EXPORTED_RULEBOOK_VERSION

    __slots__ = (
        "feature_schema_hash",
        "role_type",
        "credit_mode",
        "_action_configuration",
        "tie_break",
        "_action_rules",
        "_veto_rules",
    )

    def __init__(self, rulebook: FrozenRulebookPolicy) -> None:
        if not isinstance(rulebook, FrozenRulebookPolicy):
            raise TypeError("export requires a FrozenRulebookPolicy")
        detached = FrozenRulebookPolicy.from_dict(rulebook.to_dict())
        self.feature_schema_hash = _token(
            detached.feature_schema_hash,
            "feature_schema_hash",
        )
        self.role_type = _token(detached.role_type, "role_type")
        self.credit_mode = validate_credit_mode(detached.credit_mode)
        self._action_configuration = _frozen_canonical(detached.action_configuration)
        self.tie_break = tuple(detached.tie_break)
        self._action_rules = tuple(
            _frozen_mapping(
                {
                    "condition": rule.condition.to_dict(),
                    "action": rule.action.to_dict(),
                    "w": rule.w,
                    "precision": rule.precision,
                },
                "exported action rule",
            )
            for rule in detached.action_rules
        )
        self._veto_rules = tuple(
            _frozen_mapping(
                {
                    "condition": rule.condition.to_dict(),
                    "veto": True,
                },
                "exported veto rule",
            )
            for rule in detached.veto_rules
        )
        self._seal()

    @classmethod
    def from_rulebook(
        cls,
        rulebook: FrozenRulebookPolicy,
    ) -> "ExportedRulebook":
        return cls(rulebook)

    @property
    def action_configuration(self) -> Any:
        return _restored(self._action_configuration)

    @property
    def action_rules(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(_restored(rule) for rule in self._action_rules)

    @property
    def veto_rules(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(_restored(rule) for rule in self._veto_rules)

    @property
    def rulebook(self) -> FrozenRulebookPolicy:
        actions = tuple(
            FrozenActionRule(
                source_rule_id=f"exported-action-{index:08d}",
                condition=RuleCondition.from_dict(rule["condition"]),
                action=RuleAction.from_dict(rule["action"]),
                w=float(rule["w"]),
                precision=float(rule["precision"]),
                rival_lcb=0.0,
                rival_samples=0,
            )
            for index, rule in enumerate(self.action_rules)
        )
        vetoes = tuple(
            FrozenVetoRule(
                source_rule_id=f"exported-veto-{index:08d}",
                condition=RuleCondition.from_dict(rule["condition"]),
            )
            for index, rule in enumerate(self.veto_rules)
        )
        return FrozenRulebookPolicy(
            feature_schema_hash=self.feature_schema_hash,
            credit_mode=self.credit_mode,
            role_type=self.role_type,
            action_rules=actions,
            veto_rules=vetoes,
            action_configuration=self.action_configuration,
            tie_break=self.tie_break,
        )

    def to_rulebook(self) -> FrozenRulebookPolicy:
        return self.rulebook

    def policy_fingerprint(self) -> str:
        return self.content_hash

    def evaluate(self, records: Any) -> Any:
        return self.rulebook.evaluate(records)

    def decide(self, records: Any) -> Any | None:
        return self.rulebook.decide(records)

    def select_action(self, context: Any) -> Any | None:
        return self.rulebook.select_action(context)

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_version": self.ARTIFACT_VERSION,
            "feature_schema_hash": self.feature_schema_hash,
            "role_type": self.role_type,
            "credit_mode": self.credit_mode,
            "action_configuration": _serialized(self._action_configuration),
            "tie_break": list(self.tie_break),
            "action_rules": [_serialized(rule) for rule in self._action_rules],
            "veto_rules": [_serialized(rule) for rule in self._veto_rules],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExportedRulebook":
        if not isinstance(payload, Mapping):
            raise TypeError("exported rulebook payload must be a mapping")
        expected_fields = {
            "artifact_version",
            "feature_schema_hash",
            "role_type",
            "credit_mode",
            "action_configuration",
            "tie_break",
            "action_rules",
            "veto_rules",
            "content_hash",
        }
        if set(payload) != expected_fields:
            if set(payload) == expected_fields - {"content_hash"}:
                raise ValueError("serialized artifact requires content_hash")
            raise ValueError(
                "exported rulebook fields must exactly match the canonical schema"
            )
        if payload.get("artifact_version") != cls.ARTIFACT_VERSION:
            raise ValueError("unsupported exported rulebook artifact version")
        credit_mode = validate_credit_mode(
            payload["credit_mode"], name="serialized exported rulebook credit_mode"
        )
        _verify_serialized_hash(payload, namespace=cls.HASH_NAMESPACE)
        raw_actions = payload.get("action_rules")
        raw_vetoes = payload.get("veto_rules")
        if (
            isinstance(raw_actions, (str, bytes))
            or not isinstance(raw_actions, Sequence)
            or isinstance(raw_vetoes, (str, bytes))
            or not isinstance(raw_vetoes, Sequence)
        ):
            raise TypeError("exported rules must be sequences")

        actions: list[FrozenActionRule] = []
        for index, raw in enumerate(raw_actions):
            if not isinstance(raw, Mapping) or set(raw) != {
                "condition",
                "action",
                "w",
                "precision",
            }:
                raise ValueError(
                    "exported action rules may contain only inference fields"
                )
            actions.append(
                FrozenActionRule(
                    source_rule_id=f"exported-action-{index:08d}",
                    condition=RuleCondition.from_dict(raw["condition"]),
                    action=RuleAction.from_dict(raw["action"]),
                    w=float(raw["w"]),
                    precision=float(raw["precision"]),
                    rival_lcb=0.0,
                    rival_samples=0,
                )
            )

        vetoes: list[FrozenVetoRule] = []
        for index, raw in enumerate(raw_vetoes):
            if (
                not isinstance(raw, Mapping)
                or set(raw) != {"condition", "veto"}
                or raw["veto"] is not True
            ):
                raise ValueError(
                    "exported veto rules may contain only condition and veto=true"
                )
            vetoes.append(
                FrozenVetoRule(
                    source_rule_id=f"exported-veto-{index:08d}",
                    condition=RuleCondition.from_dict(raw["condition"]),
                )
            )

        policy = FrozenRulebookPolicy(
            feature_schema_hash=str(payload["feature_schema_hash"]),
            credit_mode=credit_mode,
            role_type=str(payload["role_type"]),
            action_rules=tuple(actions),
            veto_rules=tuple(vetoes),
            action_configuration=payload["action_configuration"],
            tie_break=tuple(payload["tie_break"]),
        )
        exported = cls(policy)
        _content_checked(
            exported._payload(),
            namespace=cls.HASH_NAMESPACE,
            expected=payload["content_hash"],
        )
        return exported

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
    ) -> "ExportedRulebook":
        return cls.from_dict(_json_mapping(payload))


class EvaluationSnapshot(_CanonicalArtifact):
    """Detached certified rulebook and ranking contract for one publication."""

    ARTIFACT_VERSION = EVALUATION_SNAPSHOT_VERSION
    HASH_NAMESPACE = EVALUATION_SNAPSHOT_VERSION

    __slots__ = (
        "_rulebook_payload",
        "publication_epoch",
        "certified_source_rule_ids",
        "_contender_ranking_fields",
        "_certification_evidence_payload",
        "feature_schema_hash",
        "action_vocabulary_hash",
        "config_hash",
    )

    def __init__(
        self,
        rulebook: FrozenRulebookPolicy | Mapping[str, Any],
        *,
        config_hash: str,
        publication_epoch: int | None = None,
        certified_source_rule_ids: Sequence[str] | None = None,
        contender_ranking_fields: Mapping[str, Any] | Sequence[str] | None = None,
        certification_evidence: Mapping[
            str,
            PublishedCertificationEvidence | Mapping[str, Any],
        ]
        | None = None,
        feature_schema_hash: str | None = None,
        action_vocabulary_hash: str | None = None,
    ) -> None:
        detached = (
            FrozenRulebookPolicy.from_dict(rulebook)
            if isinstance(rulebook, Mapping)
            else FrozenRulebookPolicy.from_dict(rulebook.to_dict())
            if isinstance(rulebook, FrozenRulebookPolicy)
            else None
        )
        if detached is None:
            raise TypeError(
                "rulebook must be FrozenRulebookPolicy or serialized mapping"
            )

        schema_hash = _token(
            detached.feature_schema_hash
            if feature_schema_hash is None
            else feature_schema_hash,
            "feature_schema_hash",
        )
        if schema_hash != detached.feature_schema_hash:
            raise ValueError("evaluation feature schema does not match its rulebook")
        action_hash = _token(
            detached.action_config_hash
            if action_vocabulary_hash is None
            else action_vocabulary_hash,
            "action_vocabulary_hash",
        )
        if action_hash != detached.action_config_hash:
            raise ValueError("evaluation action vocabulary does not match its rulebook")
        publication = (
            detached.certification_generation
            if publication_epoch is None
            else publication_epoch
        )

        default_ids = tuple(
            rule.source_rule_id
            for rule in (*detached.action_rules, *detached.veto_rules)
        )
        source_ids = _ids(
            default_ids
            if certified_source_rule_ids is None
            else certified_source_rule_ids,
            "certified_source_rule_ids",
        )
        if source_ids != default_ids:
            raise ValueError(
                "certified_source_rule_ids must exactly match the embedded rulebook"
            )
        if contender_ranking_fields is None:
            ranking: Mapping[str, Any] = {}
        elif isinstance(contender_ranking_fields, Mapping):
            ranking = contender_ranking_fields
        elif isinstance(contender_ranking_fields, Sequence) and not isinstance(
            contender_ranking_fields, (str, bytes)
        ):
            ranking = {"fields": [str(item) for item in contender_ranking_fields]}
        else:
            raise TypeError(
                "contender_ranking_fields must be a mapping or sequence of names"
            )

        if certification_evidence is None:
            raw_evidence: Mapping[
                str,
                PublishedCertificationEvidence | Mapping[str, Any],
            ] = {}
        elif isinstance(certification_evidence, Mapping):
            raw_evidence = certification_evidence
        else:
            raise TypeError("certification_evidence must be a mapping")
        evidence: dict[str, PublishedCertificationEvidence] = {}
        for raw_source_rule_id, raw_record in raw_evidence.items():
            if type(raw_source_rule_id) is not str:
                raise TypeError("certification evidence keys must be strings")
            source_rule_id = _token(
                raw_source_rule_id,
                "certification evidence key",
            )
            record = (
                PublishedCertificationEvidence.from_dict(raw_record)
                if isinstance(raw_record, Mapping)
                else PublishedCertificationEvidence.from_dict(raw_record.to_dict())
                if isinstance(raw_record, PublishedCertificationEvidence)
                else None
            )
            if record is None:
                raise TypeError(
                    "certification evidence values must be published evidence records"
                )
            if record.source_rule_id != source_rule_id:
                raise ValueError(
                    "certification evidence key must equal its source_rule_id"
                )
            evidence[source_rule_id] = record
        if set(evidence) != set(default_ids):
            raise ValueError(
                "certification evidence keys must exactly match the embedded rulebook"
            )
        frozen_by_source = {
            rule.source_rule_id: rule
            for rule in (*detached.action_rules, *detached.veto_rules)
        }
        threshold_payloads: set[str] = set()
        for source_rule_id, record in evidence.items():
            if record.credit_mode != detached.credit_mode:
                raise ValueError(
                    "certification evidence credit_mode does not match its rulebook"
                )
            record.validate_certified_rule(frozen_by_source[source_rule_id])
            threshold_payloads.add(canonical_json(record.thresholds.to_dict()))
        if len(threshold_payloads) > 1:
            raise ValueError("one publication must use one certification threshold set")

        self._rulebook_payload = _frozen_mapping(
            detached.to_dict(),
            "rulebook",
        )
        self.publication_epoch = _epoch(publication, "publication_epoch")
        self.certified_source_rule_ids = source_ids
        self._contender_ranking_fields = _frozen_mapping(
            ranking,
            "contender_ranking_fields",
        )
        self._certification_evidence_payload = _frozen_mapping(
            {
                source_rule_id: record.to_dict()
                for source_rule_id, record in sorted(evidence.items())
            },
            "certification_evidence",
        )
        self.feature_schema_hash = schema_hash
        self.action_vocabulary_hash = action_hash
        self.config_hash = _token(config_hash, "config_hash")
        self._seal()

    @property
    def rulebook(self) -> FrozenRulebookPolicy:
        return FrozenRulebookPolicy.from_dict(_serialized(self._rulebook_payload))

    @property
    def frozen_rulebook(self) -> FrozenRulebookPolicy:
        return self.rulebook

    @property
    def rulebook_hash(self) -> str:
        return str(self._rulebook_payload["content_hash"])

    @property
    def contender_ranking_fields(self) -> Mapping[str, Any]:
        return _restored(self._contender_ranking_fields)

    @property
    def certification_evidence(
        self,
    ) -> Mapping[str, PublishedCertificationEvidence]:
        return MappingProxyType(
            {
                source_rule_id: PublishedCertificationEvidence.from_dict(
                    _serialized(payload)
                )
                for source_rule_id, payload in self._certification_evidence_payload.items()
            }
        )

    @property
    def certification_thresholds(self) -> CertificationThresholds | None:
        evidence = self.certification_evidence
        if not evidence:
            return None
        return next(iter(evidence.values())).thresholds

    @property
    def schema_hash(self) -> str:
        return self.feature_schema_hash

    @property
    def action_configuration_hash(self) -> str:
        return self.action_vocabulary_hash

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_version": self.ARTIFACT_VERSION,
            "publication_epoch": self.publication_epoch,
            "rulebook": _serialized(self._rulebook_payload),
            "certified_source_rule_ids": list(self.certified_source_rule_ids),
            "contender_ranking_fields": _serialized(self._contender_ranking_fields),
            "certification_evidence": _serialized(self._certification_evidence_payload),
            "feature_schema_hash": self.feature_schema_hash,
            "action_vocabulary_hash": self.action_vocabulary_hash,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationSnapshot":
        if not isinstance(payload, Mapping):
            raise TypeError("evaluation snapshot payload must be a mapping")
        if payload.get("artifact_version") != cls.ARTIFACT_VERSION:
            raise ValueError("unsupported evaluation snapshot artifact version")
        _verify_serialized_hash(payload, namespace=cls.HASH_NAMESPACE)
        expected_fields = {
            "artifact_version",
            "publication_epoch",
            "rulebook",
            "certified_source_rule_ids",
            "contender_ranking_fields",
            "certification_evidence",
            "feature_schema_hash",
            "action_vocabulary_hash",
            "config_hash",
            "content_hash",
        }
        if set(payload) != expected_fields:
            raise ValueError(
                "evaluation snapshot fields must exactly match the canonical schema"
            )
        rulebook_payload = payload["rulebook"]
        if not isinstance(rulebook_payload, Mapping):
            raise TypeError("evaluation snapshot requires a serialized rulebook")
        snapshot = cls(
            rulebook_payload,
            publication_epoch=payload["publication_epoch"],
            certified_source_rule_ids=tuple(
                str(item) for item in payload.get("certified_source_rule_ids", ())
            ),
            contender_ranking_fields=payload.get("contender_ranking_fields", {}),
            certification_evidence=payload["certification_evidence"],
            feature_schema_hash=str(payload["feature_schema_hash"]),
            action_vocabulary_hash=str(payload["action_vocabulary_hash"]),
            config_hash=str(payload["config_hash"]),
        )
        _content_checked(
            snapshot._payload(),
            namespace=cls.HASH_NAMESPACE,
            expected=payload.get("content_hash"),
        )
        return snapshot

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "EvaluationSnapshot":
        return cls.from_dict(_json_mapping(payload))


class TrainingCheckpoint(_CanonicalArtifact):
    """Complete restart state with detached mutable training data."""

    ARTIFACT_VERSION = TRAINING_CHECKPOINT_VERSION
    HASH_NAMESPACE = TRAINING_CHECKPOINT_VERSION

    __slots__ = (
        "scenario_definition_hash",
        "_simulator_snapshot",
        "_population_payload",
        "_evaluation_payload",
        "epoch",
        "_rng_state",
        "_exploration_counts",
        "config_hash",
        "action_vocabulary_hash",
        "feature_schema_hash",
    )

    def __init__(
        self,
        scenario_definition_hash: str | None = None,
        simulator_snapshot: Mapping[str, Any] | None = None,
        population: Population | Mapping[str, Any] | None = None,
        evaluation_snapshot: EvaluationSnapshot | Mapping[str, Any] | None = None,
        *,
        epoch: int,
        rng_state: Mapping[str, Any] | np.random.Generator,
        exploration_counts: Mapping[str, int] | None = None,
        config_hash: str | None = None,
        action_vocabulary_hash: str | None = None,
        feature_schema_hash: str | None = None,
        scenario_hash: str | None = None,
    ) -> None:
        if scenario_definition_hash is not None and scenario_hash is not None:
            if str(scenario_definition_hash) != str(scenario_hash):
                raise ValueError("scenario hash aliases disagree")
        scenario = scenario_definition_hash or scenario_hash
        if scenario is None:
            raise ValueError("scenario_definition_hash is required")
        if simulator_snapshot is None:
            raise ValueError("simulator_snapshot is required")
        if population is None:
            raise ValueError("population is required")
        if evaluation_snapshot is None:
            raise ValueError("evaluation_snapshot is required")

        detached_population = (
            Population.from_dict(population)
            if isinstance(population, Mapping)
            else Population.from_dict(population.to_dict())
            if isinstance(population, Population)
            else None
        )
        if detached_population is None:
            raise TypeError("population must be Population or serialized mapping")
        detached_evaluation = (
            EvaluationSnapshot.from_dict(evaluation_snapshot)
            if isinstance(evaluation_snapshot, Mapping)
            else EvaluationSnapshot.from_dict(evaluation_snapshot.to_dict())
            if isinstance(evaluation_snapshot, EvaluationSnapshot)
            else None
        )
        if detached_evaluation is None:
            raise TypeError(
                "evaluation_snapshot must be EvaluationSnapshot or serialized mapping"
            )

        normalized_scenario = _token(scenario, "scenario_definition_hash")
        snapshot = _frozen_mapping(simulator_snapshot, "simulator_snapshot")
        snapshot_definition = snapshot.get("definition_hash")
        if (
            snapshot_definition is not None
            and str(snapshot_definition) != normalized_scenario
        ):
            raise ValueError(
                "simulator snapshot definition hash does not match checkpoint scenario"
            )

        if isinstance(rng_state, np.random.Generator):
            raw_rng_state: Mapping[str, Any] = rng_state.bit_generator.state
        elif isinstance(rng_state, Mapping):
            raw_rng_state = rng_state
        else:
            raise TypeError("rng_state must be a mapping or numpy Generator")
        if "pending_decision" in raw_rng_state:
            _pending_decision_reference(raw_rng_state["pending_decision"])

        config = _token(
            detached_evaluation.config_hash if config_hash is None else config_hash,
            "config_hash",
        )
        action_hash = _token(
            detached_evaluation.action_vocabulary_hash
            if action_vocabulary_hash is None
            else action_vocabulary_hash,
            "action_vocabulary_hash",
        )
        schema_hash = _token(
            detached_evaluation.feature_schema_hash
            if feature_schema_hash is None
            else feature_schema_hash,
            "feature_schema_hash",
        )
        if (
            config != detached_evaluation.config_hash
            or action_hash != detached_evaluation.action_vocabulary_hash
            or schema_hash != detached_evaluation.feature_schema_hash
        ):
            raise ValueError(
                "checkpoint configuration hashes must match its evaluation snapshot"
            )

        counts: dict[str, int] = {}
        for raw_name, raw_count in (exploration_counts or {}).items():
            name = _token(raw_name, "exploration count name")
            if type(raw_count) is not int or raw_count < 0:
                raise ValueError("exploration counts must be non-negative integers")
            counts[name] = raw_count

        self.scenario_definition_hash = normalized_scenario
        self._simulator_snapshot = snapshot
        self._population_payload = _frozen_mapping(
            detached_population.to_dict(),
            "population",
        )
        self._evaluation_payload = _frozen_mapping(
            detached_evaluation.to_dict(),
            "evaluation_snapshot",
        )
        self.epoch = _epoch(epoch)
        self._rng_state = _frozen_mapping(raw_rng_state, "rng_state")
        self._exploration_counts = MappingProxyType(dict(sorted(counts.items())))
        self.config_hash = config
        self.action_vocabulary_hash = action_hash
        self.feature_schema_hash = schema_hash
        self._seal()

    @property
    def scenario_hash(self) -> str:
        return self.scenario_definition_hash

    @property
    def simulator_snapshot(self) -> Mapping[str, Any]:
        return _restored(self._simulator_snapshot)

    @property
    def population(self) -> Population:
        return Population.from_dict(_serialized(self._population_payload))

    @property
    def evaluation_snapshot(self) -> EvaluationSnapshot:
        return EvaluationSnapshot.from_dict(_serialized(self._evaluation_payload))

    @property
    def frozen_rulebook(self) -> FrozenRulebookPolicy:
        return self.evaluation_snapshot.rulebook

    @property
    def rng_state(self) -> Mapping[str, Any]:
        return _restored(self._rng_state)

    @property
    def pending_decision_reference(self) -> Mapping[str, Any] | None:
        return _pending_decision_reference(self.rng_state.get("pending_decision"))

    @property
    def exploration_counts(self) -> Mapping[str, int]:
        return self._exploration_counts

    @property
    def schema_hash(self) -> str:
        return self.feature_schema_hash

    @property
    def action_configuration_hash(self) -> str:
        return self.action_vocabulary_hash

    def restore_rng(self, rng: np.random.Generator) -> np.random.Generator:
        state = self.rng_state
        trainer_state = state.get("trainer_rng")
        if trainer_state is not None:
            if not isinstance(trainer_state, Mapping):
                raise TypeError("checkpoint trainer_rng must be a mapping")
            state = trainer_state
        return restore_rng_state(rng, state)

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_version": self.ARTIFACT_VERSION,
            "scenario_definition_hash": self.scenario_definition_hash,
            "simulator_snapshot": _serialized(self._simulator_snapshot),
            "population": _serialized(self._population_payload),
            "evaluation_snapshot": _serialized(self._evaluation_payload),
            "epoch": self.epoch,
            "rng_state": _serialized(self._rng_state),
            "exploration_counts": dict(self._exploration_counts),
            "config_hash": self.config_hash,
            "action_vocabulary_hash": self.action_vocabulary_hash,
            "feature_schema_hash": self.feature_schema_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingCheckpoint":
        if not isinstance(payload, Mapping):
            raise TypeError("training checkpoint payload must be a mapping")
        if payload.get("artifact_version") != cls.ARTIFACT_VERSION:
            raise ValueError("unsupported training checkpoint artifact version")
        _verify_serialized_hash(payload, namespace=cls.HASH_NAMESPACE)
        checkpoint = cls(
            scenario_definition_hash=str(payload["scenario_definition_hash"]),
            simulator_snapshot=payload["simulator_snapshot"],
            population=payload["population"],
            evaluation_snapshot=payload["evaluation_snapshot"],
            epoch=payload["epoch"],
            rng_state=payload["rng_state"],
            exploration_counts=payload.get("exploration_counts", {}),
            config_hash=str(payload["config_hash"]),
            action_vocabulary_hash=str(payload["action_vocabulary_hash"]),
            feature_schema_hash=str(payload["feature_schema_hash"]),
        )
        _content_checked(
            checkpoint._payload(),
            namespace=cls.HASH_NAMESPACE,
            expected=payload.get("content_hash"),
        )
        return checkpoint

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "TrainingCheckpoint":
        return cls.from_dict(_json_mapping(payload))


class EpochTrace(_CanonicalArtifact):
    """One causal training decision expressed entirely as audit-safe values."""

    ARTIFACT_VERSION = EPOCH_TRACE_VERSION
    HASH_NAMESPACE = "hailmary.epoch_trace.v2"

    __slots__ = (
        "scenario_definition_hash",
        "epoch",
        "root_parent_hash",
        "committed_state_hash",
        "outcome_plan_hash",
        "selected_anchor_id",
        "_selected_action",
        "_contender_action",
        "_no_op_action",
        "_committed_action",
        "matched_rule_ids",
        "coadvocate_rule_ids",
        "exploration_reason",
        "_contender_selection",
        "rulebook_hash",
        "_arm_initial_hashes",
        "_arm_final_hashes",
        "_arm_scores",
        "delta_rival",
        "delta_selected_noop",
        "delta_contender_noop",
        "delta_veto",
        "_ledger_recipients",
        "_ga_events",
        "_certification_events",
        "config_hash",
        "action_vocabulary_hash",
        "feature_schema_hash",
    )

    def __init__(
        self,
        scenario_definition_hash: str | None = None,
        *,
        epoch: int,
        root_parent_hash: str,
        rulebook_hash: str,
        config_hash: str,
        action_vocabulary_hash: str,
        feature_schema_hash: str,
        committed_state_hash: str | None = None,
        outcome_plan_hash: str | None = None,
        selected_anchor_id: str | None = None,
        selected_action: Any | None = None,
        contender_action: Any | None = None,
        no_op_action: Any | None = None,
        committed_action: Any | None = None,
        matched_rule_ids: Sequence[str] | None = None,
        coadvocate_rule_ids: Sequence[str] | None = None,
        exploration_reason: str | None = None,
        contender_selection: Mapping[str, Any] | None = None,
        arm_initial_hashes: Mapping[str, str] | None = None,
        arm_final_hashes: Mapping[str, str] | None = None,
        arm_scores: Mapping[str, float] | None = None,
        delta_rival: float = 0.0,
        delta_selected_noop: float = 0.0,
        delta_contender_noop: float = 0.0,
        delta_veto: float = 0.0,
        ledger_recipients: Mapping[str, Any] | None = None,
        ga_events: Sequence[Mapping[str, Any]] | None = None,
        certification_events: Sequence[Mapping[str, Any]] | None = None,
        scenario_hash: str | None = None,
    ) -> None:
        if scenario_definition_hash is not None and scenario_hash is not None:
            if str(scenario_definition_hash) != str(scenario_hash):
                raise ValueError("scenario hash aliases disagree")
        scenario = scenario_definition_hash or scenario_hash
        if scenario is None:
            raise ValueError("scenario_definition_hash is required")

        self.scenario_definition_hash = _token(
            scenario,
            "scenario_definition_hash",
        )
        self.epoch = _epoch(epoch)
        self.root_parent_hash = _token(root_parent_hash, "root_parent_hash")
        self.committed_state_hash = (
            None
            if committed_state_hash is None
            else _token(committed_state_hash, "committed_state_hash")
        )
        self.outcome_plan_hash = (
            None
            if outcome_plan_hash is None
            else _token(outcome_plan_hash, "outcome_plan_hash")
        )
        self.selected_anchor_id = (
            None
            if selected_anchor_id is None
            else _token(selected_anchor_id, "selected_anchor_id")
        )
        self._selected_action = (
            None if selected_action is None else _frozen_canonical(selected_action)
        )
        self._contender_action = (
            None if contender_action is None else _frozen_canonical(contender_action)
        )
        self._no_op_action = (
            None if no_op_action is None else _frozen_canonical(no_op_action)
        )
        self._committed_action = (
            None if committed_action is None else _frozen_canonical(committed_action)
        )
        self.matched_rule_ids = _ids(matched_rule_ids, "matched_rule_ids")
        self.coadvocate_rule_ids = _ids(
            coadvocate_rule_ids,
            "coadvocate_rule_ids",
        )
        self.exploration_reason = (
            None if exploration_reason is None else str(exploration_reason)
        )
        self._contender_selection = self._normalize_contender_selection(
            contender_selection
        )
        self.rulebook_hash = _token(rulebook_hash, "rulebook_hash")
        self._arm_initial_hashes = self._hash_mapping(
            arm_initial_hashes,
            "arm_initial_hashes",
        )
        self._arm_final_hashes = self._hash_mapping(
            arm_final_hashes,
            "arm_final_hashes",
        )
        self._arm_scores = self._score_mapping(arm_scores)
        self.delta_rival = _finite(delta_rival, "delta_rival")
        self.delta_selected_noop = _finite(
            delta_selected_noop,
            "delta_selected_noop",
        )
        self.delta_contender_noop = _finite(
            delta_contender_noop,
            "delta_contender_noop",
        )
        self.delta_veto = _finite(delta_veto, "delta_veto")
        self._ledger_recipients = _frozen_mapping(
            ledger_recipients or {},
            "ledger_recipients",
        )
        self._validate_arm_evidence()
        self._ga_events = _frozen_events(ga_events, "ga_events")
        self._certification_events = _frozen_events(
            certification_events,
            "certification_events",
        )
        self.config_hash = _token(config_hash, "config_hash")
        self.action_vocabulary_hash = _token(
            action_vocabulary_hash,
            "action_vocabulary_hash",
        )
        self.feature_schema_hash = _token(
            feature_schema_hash,
            "feature_schema_hash",
        )
        self._seal()

    @staticmethod
    def _normalize_contender_selection(
        value: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if value is None:
            return MappingProxyType({})
        if not isinstance(value, Mapping):
            raise TypeError("contender_selection must be a mapping")
        required = {
            "ranking_fields",
            "ranked_candidates",
            "selected_rank",
            "fallback",
        }
        if set(value) != required:
            raise ValueError(
                "contender_selection fields must exactly match the audit schema"
            )
        ranking_fields = value["ranking_fields"]
        if not isinstance(ranking_fields, Mapping) or not ranking_fields:
            raise ValueError(
                "contender_selection ranking_fields must be a non-empty mapping"
            )
        raw_ranked = value["ranked_candidates"]
        if isinstance(raw_ranked, (str, bytes)) or not isinstance(raw_ranked, Sequence):
            raise TypeError("contender_selection ranked_candidates must be a sequence")

        ranked: list[dict[str, Any]] = []
        actions: list[RuleAction] = []
        for expected_rank, record in enumerate(raw_ranked, start=1):
            if not isinstance(record, Mapping):
                raise TypeError("ranked contender records must be mappings")
            if set(record) != {"rank", "action", "score", "source_rule_ids"}:
                raise ValueError(
                    "ranked contender fields must exactly match the audit schema"
                )
            rank = record["rank"]
            if type(rank) is not int or rank != expected_rank:
                raise ValueError(
                    "ranked contender ranks must be consecutive positive integers"
                )
            action = RuleAction.from_candidate(record["action"])
            score = _finite(record["score"], "ranked contender score")
            source_rule_ids = _ids(
                record["source_rule_ids"],
                "ranked contender source_rule_ids",
            )
            if not source_rule_ids:
                raise ValueError("ranked contenders require source_rule_ids")
            actions.append(action)
            ranked.append(
                {
                    "rank": rank,
                    "action": action.to_dict(),
                    "score": score,
                    "source_rule_ids": list(source_rule_ids),
                }
            )
        if len(actions) != len(set(actions)):
            raise ValueError("ranked contender actions cannot contain duplicates")
        expected_order = sorted(
            ranked,
            key=lambda record: (
                -record["score"],
                record["action"]["lever"],
                record["action"]["band"],
            ),
        )
        if ranked != expected_order:
            raise ValueError("ranked contenders are not in canonical rank order")

        selected_rank = value["selected_rank"]
        fallback = value["fallback"]
        if fallback is None:
            if selected_rank != 1 or type(selected_rank) is not int or not ranked:
                raise ValueError("ranked contender selection must select rank one")
        elif fallback == _CONTENDER_FALLBACK:
            if selected_rank is not None or ranked:
                raise ValueError(
                    "fallback contender selection cannot contain a ranked choice"
                )
        else:
            raise ValueError("unsupported contender selection fallback")

        return _frozen_mapping(
            {
                "ranking_fields": ranking_fields,
                "ranked_candidates": ranked,
                "selected_rank": selected_rank,
                "fallback": fallback,
            },
            "contender_selection",
        )

    def _validate_contender_selection(self) -> None:
        if not self._contender_selection:
            return
        if self._contender_action is None:
            raise ValueError("contender selection requires contender_action")
        contender = RuleAction.from_candidate(_restored(self._contender_action))
        fallback = self._contender_selection["fallback"]
        if fallback is not None:
            if not contender.is_no_op:
                raise ValueError(
                    "fallback contender must be the mandatory no-op action"
                )
            return
        ranked = self._contender_selection["ranked_candidates"]
        selected = RuleAction.from_candidate(ranked[0]["action"])
        if selected != contender:
            raise ValueError(
                "rank-one contender metadata does not match contender_action"
            )
        if self._selected_action is not None:
            root = RuleAction.from_candidate(_restored(self._selected_action))
            if any(
                RuleAction.from_candidate(record["action"]) == root for record in ranked
            ):
                raise ValueError(
                    "contender ranking cannot include the selected root action"
                )

    def _validate_arm_evidence(self) -> None:
        self._validate_contender_selection()
        required = {"selected", "contender", "no_op"}
        mappings = (
            self._arm_initial_hashes,
            self._arm_final_hashes,
            self._arm_scores,
        )
        if not any(mappings):
            if any(
                value != 0.0
                for value in (
                    self.delta_rival,
                    self.delta_selected_noop,
                    self.delta_contender_noop,
                    self.delta_veto,
                )
            ):
                raise ValueError("traces without arm evidence must have zero deltas")
            if self._committed_action is not None:
                raise ValueError("traces without arm evidence cannot commit an action")
            return
        required_fields = {
            "committed_state_hash": self.committed_state_hash,
            "outcome_plan_hash": self.outcome_plan_hash,
            "selected_anchor_id": self.selected_anchor_id,
            "selected_action": self._selected_action,
            "contender_action": self._contender_action,
            "no_op_action": self._no_op_action,
            "committed_action": self._committed_action,
        }
        missing = sorted(
            name for name, value in required_fields.items() if value is None
        )
        if missing:
            raise ValueError("completed traces require " + ", ".join(missing))
        if not self.exploration_reason or not self.exploration_reason.strip():
            raise ValueError("completed traces require exploration_reason")
        if not self.matched_rule_ids:
            raise ValueError("completed traces require matched_rule_ids")
        if not self.coadvocate_rule_ids:
            raise ValueError("completed traces require coadvocate_rule_ids")
        if not self._ledger_recipients:
            raise ValueError("completed traces require ledger_recipients")
        if not self._contender_selection:
            raise ValueError("completed traces require contender_selection")

        try:
            selected_action = RuleAction.from_candidate(
                _restored(self._selected_action)
            )
            contender_action = RuleAction.from_candidate(
                _restored(self._contender_action)
            )
            no_op_action = RuleAction.from_candidate(_restored(self._no_op_action))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "completed trace actions must be canonical Hailmary actions"
            ) from exc
        if self._committed_action != self._selected_action:
            raise ValueError(
                "completed committed_action must exactly equal selected_action"
            )
        if not no_op_action.is_no_op:
            raise ValueError("completed Arm C action must be canonical no-op")
        if (
            selected_action.is_no_op
            and contender_action.is_no_op
            and self._ledger_recipients.get("credit_mode")
            != VANILLA_ACCURACY_CREDIT_MODE
        ):
            raise ValueError("an all-no-op trace must remain a skipped trace")

        if any(set(values) != required for values in mappings):
            raise ValueError(
                "completed traces require exact selected/contender/no_op arm keys"
            )
        if any(
            value != self.root_parent_hash
            for value in self._arm_initial_hashes.values()
        ):
            raise ValueError(
                "all completed arm initial hashes must equal root_parent_hash"
            )

        arm_actions = {
            "selected": selected_action,
            "contender": contender_action,
            "no_op": no_op_action,
        }
        arm_names = tuple(arm_actions)
        for index, left in enumerate(arm_names):
            for right in arm_names[index + 1 :]:
                if arm_actions[left] != arm_actions[right]:
                    continue
                if (
                    self._arm_final_hashes[left] != self._arm_final_hashes[right]
                    or self._arm_scores[left] != self._arm_scores[right]
                ):
                    raise ValueError(
                        "identical-action arms must have exact matching evidence"
                    )

        selected = self._arm_scores["selected"]
        contender = self._arm_scores["contender"]
        no_op = self._arm_scores["no_op"]
        expected = {
            "delta_rival": selected - contender,
            "delta_selected_noop": selected - no_op,
            "delta_contender_noop": contender - no_op,
            "delta_veto": no_op - max(selected, contender),
        }
        for name, expected_value in expected.items():
            if not math.isclose(
                getattr(self, name),
                expected_value,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"{name} is inconsistent with completed arm scores")

    @staticmethod
    def _hash_mapping(
        values: Mapping[str, str] | None,
        name: str,
    ) -> Mapping[str, str]:
        normalized: dict[str, str] = {}
        for key, value in (values or {}).items():
            normalized[_token(key, f"{name} arm")] = _token(value, name)
        return MappingProxyType(dict(sorted(normalized.items())))

    @staticmethod
    def _score_mapping(
        values: Mapping[str, float] | None,
    ) -> Mapping[str, float]:
        normalized = {
            _token(key, "arm score name"): _finite(value, "arm score")
            for key, value in (values or {}).items()
        }
        return MappingProxyType(dict(sorted(normalized.items())))

    @property
    def scenario_hash(self) -> str:
        return self.scenario_definition_hash

    @property
    def selected_action(self) -> Any | None:
        return _restored(self._selected_action)

    @property
    def contender_action(self) -> Any | None:
        return _restored(self._contender_action)

    @property
    def no_op_action(self) -> Any | None:
        return _restored(self._no_op_action)

    @property
    def committed_action(self) -> Any | None:
        return _restored(self._committed_action)

    @property
    def contender_selection(self) -> Mapping[str, Any]:
        return _restored(self._contender_selection)

    @property
    def arm_initial_hashes(self) -> Mapping[str, str]:
        return self._arm_initial_hashes

    @property
    def arm_final_hashes(self) -> Mapping[str, str]:
        return self._arm_final_hashes

    @property
    def arm_scores(self) -> Mapping[str, float]:
        return self._arm_scores

    @property
    def ledger_recipients(self) -> Mapping[str, Any]:
        return _restored(self._ledger_recipients)

    @property
    def ga_events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(_restored(event) for event in self._ga_events)

    @property
    def certification_events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(_restored(event) for event in self._certification_events)

    @property
    def schema_hash(self) -> str:
        return self.feature_schema_hash

    @property
    def action_configuration_hash(self) -> str:
        return self.action_vocabulary_hash

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_version": self.ARTIFACT_VERSION,
            "scenario_definition_hash": self.scenario_definition_hash,
            "epoch": self.epoch,
            "root_parent_hash": self.root_parent_hash,
            "committed_state_hash": self.committed_state_hash,
            "outcome_plan_hash": self.outcome_plan_hash,
            "selected_anchor_id": self.selected_anchor_id,
            "selected_action": _serialized(self._selected_action),
            "contender_action": _serialized(self._contender_action),
            "no_op_action": _serialized(self._no_op_action),
            "committed_action": _serialized(self._committed_action),
            "matched_rule_ids": list(self.matched_rule_ids),
            "coadvocate_rule_ids": list(self.coadvocate_rule_ids),
            "exploration_reason": self.exploration_reason,
            "contender_selection": _serialized(self._contender_selection),
            "rulebook_hash": self.rulebook_hash,
            "arm_initial_hashes": dict(self._arm_initial_hashes),
            "arm_final_hashes": dict(self._arm_final_hashes),
            "arm_scores": dict(self._arm_scores),
            "delta_rival": self.delta_rival,
            "delta_selected_noop": self.delta_selected_noop,
            "delta_contender_noop": self.delta_contender_noop,
            "delta_veto": self.delta_veto,
            "ledger_recipients": _serialized(self._ledger_recipients),
            "ga_events": [_serialized(event) for event in self._ga_events],
            "certification_events": [
                _serialized(event) for event in self._certification_events
            ],
            "config_hash": self.config_hash,
            "action_vocabulary_hash": self.action_vocabulary_hash,
            "feature_schema_hash": self.feature_schema_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpochTrace":
        if not isinstance(payload, Mapping):
            raise TypeError("epoch trace payload must be a mapping")
        if payload.get("artifact_version") != cls.ARTIFACT_VERSION:
            raise ValueError("unsupported epoch trace artifact version")
        _verify_serialized_hash(payload, namespace=cls.HASH_NAMESPACE)
        trace = cls(
            scenario_definition_hash=str(payload["scenario_definition_hash"]),
            epoch=payload["epoch"],
            root_parent_hash=str(payload["root_parent_hash"]),
            committed_state_hash=payload.get("committed_state_hash"),
            outcome_plan_hash=payload.get("outcome_plan_hash"),
            selected_anchor_id=payload.get("selected_anchor_id"),
            selected_action=payload.get("selected_action"),
            contender_action=payload.get("contender_action"),
            no_op_action=payload.get("no_op_action"),
            committed_action=payload.get("committed_action"),
            matched_rule_ids=tuple(
                str(item) for item in payload.get("matched_rule_ids", ())
            ),
            coadvocate_rule_ids=tuple(
                str(item) for item in payload.get("coadvocate_rule_ids", ())
            ),
            exploration_reason=payload.get("exploration_reason"),
            contender_selection=payload.get("contender_selection"),
            rulebook_hash=str(payload["rulebook_hash"]),
            arm_initial_hashes=payload.get("arm_initial_hashes", {}),
            arm_final_hashes=payload.get("arm_final_hashes", {}),
            arm_scores=payload.get("arm_scores", {}),
            delta_rival=float(payload.get("delta_rival", 0.0)),
            delta_selected_noop=float(payload.get("delta_selected_noop", 0.0)),
            delta_contender_noop=float(payload.get("delta_contender_noop", 0.0)),
            delta_veto=float(payload.get("delta_veto", 0.0)),
            ledger_recipients=payload.get("ledger_recipients", {}),
            ga_events=payload.get("ga_events", ()),
            certification_events=payload.get("certification_events", ()),
            config_hash=str(payload["config_hash"]),
            action_vocabulary_hash=str(payload["action_vocabulary_hash"]),
            feature_schema_hash=str(payload["feature_schema_hash"]),
        )
        _content_checked(
            trace._payload(),
            namespace=cls.HASH_NAMESPACE,
            expected=payload.get("content_hash"),
        )
        return trace

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "EpochTrace":
        return cls.from_dict(_json_mapping(payload))


Artifact: TypeAlias = (
    PublishedCertificationEvidence
    | ExportedRulebook
    | EvaluationSnapshot
    | TrainingCheckpoint
    | EpochTrace
)


def capture_rng_state(rng: np.random.Generator) -> Mapping[str, Any]:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    return _restored(_frozen_canonical(rng.bit_generator.state))


def restore_rng_state(
    rng: np.random.Generator,
    state: Mapping[str, Any],
) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if not isinstance(state, Mapping):
        raise TypeError("RNG state must be a mapping")
    rng.bit_generator.state = _restored(_frozen_canonical(state))
    return rng


def artifact_from_dict(payload: Mapping[str, Any]) -> Artifact:
    if not isinstance(payload, Mapping):
        raise TypeError("artifact payload must be a mapping")
    version = payload.get("artifact_version")
    artifact_type = {
        PUBLISHED_CERTIFICATION_EVIDENCE_VERSION: PublishedCertificationEvidence,
        EVALUATION_SNAPSHOT_VERSION: EvaluationSnapshot,
        EXPORTED_RULEBOOK_VERSION: ExportedRulebook,
        TRAINING_CHECKPOINT_VERSION: TrainingCheckpoint,
        EPOCH_TRACE_VERSION: EpochTrace,
    }.get(version)
    if artifact_type is None:
        raise ValueError(f"unsupported learning artifact version {version!r}")
    return artifact_type.from_dict(payload)


def atomic_save_json(
    path: str | os.PathLike[str],
    value: _CanonicalArtifact | Mapping[str, Any],
) -> Path:
    """Write canonical JSON through a same-directory fsync and atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = value.to_dict() if isinstance(value, _CanonicalArtifact) else value
    if not isinstance(payload, Mapping):
        raise TypeError("atomic JSON value must be an artifact or mapping")
    encoded = canonical_json(payload).encode("utf-8")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if hasattr(os, "O_DIRECTORY"):
            try:
                directory_descriptor = os.open(
                    destination.parent,
                    os.O_RDONLY | os.O_DIRECTORY,
                )
            except OSError:
                pass
            else:
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_json(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    return _json_mapping(Path(path).read_bytes())


def save_artifact(
    path: str | os.PathLike[str],
    artifact: Artifact,
) -> Path:
    if not isinstance(
        artifact,
        (
            PublishedCertificationEvidence,
            ExportedRulebook,
            EvaluationSnapshot,
            TrainingCheckpoint,
            EpochTrace,
        ),
    ):
        raise TypeError("unsupported learning artifact type")
    return atomic_save_json(path, artifact)


def load_artifact(path: str | os.PathLike[str]) -> Artifact:
    return artifact_from_dict(load_json(path))


atomic_save_artifact = save_artifact
atomic_load_artifact = load_artifact


def save_exported_rulebook(
    path: str | os.PathLike[str],
    rulebook: ExportedRulebook,
) -> Path:
    if not isinstance(rulebook, ExportedRulebook):
        raise TypeError("rulebook must be ExportedRulebook")
    return save_artifact(path, rulebook)


def load_exported_rulebook(
    path: str | os.PathLike[str],
) -> ExportedRulebook:
    artifact = load_artifact(path)
    if not isinstance(artifact, ExportedRulebook):
        raise TypeError("artifact is not an ExportedRulebook")
    return artifact


def save_evaluation_snapshot(
    path: str | os.PathLike[str],
    snapshot: EvaluationSnapshot,
) -> Path:
    if not isinstance(snapshot, EvaluationSnapshot):
        raise TypeError("snapshot must be EvaluationSnapshot")
    return save_artifact(path, snapshot)


def load_evaluation_snapshot(
    path: str | os.PathLike[str],
) -> EvaluationSnapshot:
    artifact = load_artifact(path)
    if not isinstance(artifact, EvaluationSnapshot):
        raise TypeError("artifact is not an EvaluationSnapshot")
    return artifact


def save_training_checkpoint(
    path: str | os.PathLike[str],
    checkpoint: TrainingCheckpoint,
) -> Path:
    if not isinstance(checkpoint, TrainingCheckpoint):
        raise TypeError("checkpoint must be TrainingCheckpoint")
    return save_artifact(path, checkpoint)


def load_training_checkpoint(
    path: str | os.PathLike[str],
) -> TrainingCheckpoint:
    artifact = load_artifact(path)
    if not isinstance(artifact, TrainingCheckpoint):
        raise TypeError("artifact is not a TrainingCheckpoint")
    return artifact


def save_epoch_trace(
    path: str | os.PathLike[str],
    trace: EpochTrace,
) -> Path:
    if not isinstance(trace, EpochTrace):
        raise TypeError("trace must be EpochTrace")
    return save_artifact(path, trace)


def load_epoch_trace(path: str | os.PathLike[str]) -> EpochTrace:
    artifact = load_artifact(path)
    if not isinstance(artifact, EpochTrace):
        raise TypeError("artifact is not an EpochTrace")
    return artifact


__all__ = [
    "Artifact",
    "EPOCH_TRACE_VERSION",
    "EVALUATION_SNAPSHOT_VERSION",
    "EpochTrace",
    "EvaluationSnapshot",
    "PUBLISHED_CERTIFICATION_EVIDENCE_VERSION",
    "PublishedCertificationEvidence",
    "TRAINING_CHECKPOINT_VERSION",
    "EXPORTED_RULEBOOK_VERSION",
    "TrainingCheckpoint",
    "artifact_from_dict",
    "ExportedRulebook",
    "atomic_load_artifact",
    "atomic_save_artifact",
    "atomic_save_json",
    "capture_rng_state",
    "load_artifact",
    "load_epoch_trace",
    "load_evaluation_snapshot",
    "load_json",
    "load_training_checkpoint",
    "restore_rng_state",
    "load_exported_rulebook",
    "save_artifact",
    "save_epoch_trace",
    "save_evaluation_snapshot",
    "save_training_checkpoint",
    "save_exported_rulebook",
]
