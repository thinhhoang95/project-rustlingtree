"""No-op-grounded certification into immutable deployment rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable, Mapping

from hailmary.learning.conditions import RuleCondition
from hailmary.learning.evolution import (
    OFFSPRING_EVIDENCE_BASELINE_VERSION,
    OFFSPRING_EVIDENCE_PROVENANCE_KEY,
)
from hailmary.learning.modes import (
    CAUSAL_CREDIT_MODE,
    VANILLA_ACCURACY_CREDIT_MODE,
    validate_credit_mode,
)
from hailmary.learning.rules import MutableRule, RuleAction
from hailmary.learning.statistics import OnlineMoments


def _finite_numeric(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-bool finite numeric value")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a non-bool finite numeric value")
    return normalized


@dataclass(frozen=True, slots=True)
class CertificationThresholds:
    action_min_noop_samples: int = 30
    veto_min_rival_samples: int = 60
    contender_min_samples: int = 2
    action_lcb_z: float = 1.96
    veto_lcb_z: float = 1.96
    contender_lcb_z: float = 1.96
    variance_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        for name, minimum in (
            ("action_min_noop_samples", 1),
            ("veto_min_rival_samples", 1),
            ("contender_min_samples", 2),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"certification {name} must be an integer at least {minimum}"
                )
        for name, value in (
            ("action_lcb_z", self.action_lcb_z),
            ("veto_lcb_z", self.veto_lcb_z),
            ("contender_lcb_z", self.contender_lcb_z),
        ):
            normalized = _finite_numeric(value, name=f"certification {name}")
            if normalized < 0.0:
                raise ValueError(
                    f"certification {name} must be finite and non-negative"
                )
            object.__setattr__(self, name, normalized)
        variance_floor = _finite_numeric(
            self.variance_floor,
            name="certification variance_floor",
        )
        if variance_floor <= 0.0:
            raise ValueError("certification variance_floor must be finite and positive")
        object.__setattr__(self, "variance_floor", variance_floor)

    def to_dict(self) -> dict[str, int | float]:
        """Return the exact canonical gate settings used for certification."""

        return {
            "action_min_noop_samples": self.action_min_noop_samples,
            "veto_min_rival_samples": self.veto_min_rival_samples,
            "contender_min_samples": self.contender_min_samples,
            "action_lcb_z": self.action_lcb_z,
            "veto_lcb_z": self.veto_lcb_z,
            "contender_lcb_z": self.contender_lcb_z,
            "variance_floor": self.variance_floor,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CertificationThresholds":
        if not isinstance(payload, Mapping):
            raise TypeError("certification threshold payload must be a mapping")
        expected_fields = {
            "action_min_noop_samples",
            "veto_min_rival_samples",
            "contender_min_samples",
            "action_lcb_z",
            "veto_lcb_z",
            "contender_lcb_z",
            "variance_floor",
        }
        if set(payload) != expected_fields:
            raise ValueError(
                "certification threshold payload fields must exactly match "
                "the canonical schema"
            )
        return cls(
            action_min_noop_samples=payload["action_min_noop_samples"],
            veto_min_rival_samples=payload["veto_min_rival_samples"],
            contender_min_samples=payload["contender_min_samples"],
            action_lcb_z=payload["action_lcb_z"],
            veto_lcb_z=payload["veto_lcb_z"],
            contender_lcb_z=payload["contender_lcb_z"],
            variance_floor=payload["variance_floor"],
        )

    @classmethod
    def from_config(cls, config: Any) -> "CertificationThresholds":
        """Read the relevant fields from LearningConfig or a duck-typed equivalent."""

        return cls(
            action_min_noop_samples=getattr(config, "action_min_noop_samples"),
            veto_min_rival_samples=getattr(config, "veto_min_rival_samples"),
            contender_min_samples=getattr(config, "contender_min_samples", 2),
            action_lcb_z=getattr(config, "action_lcb_z", 1.96),
            veto_lcb_z=getattr(config, "veto_lcb_z", 1.96),
            contender_lcb_z=getattr(config, "contender_lcb_z", 1.96),
            variance_floor=getattr(config, "variance_floor", 1.0e-6),
        )


def _certification_thresholds(
    thresholds: CertificationThresholds | Any | None,
) -> CertificationThresholds:
    if thresholds is None:
        return CertificationThresholds()
    if isinstance(thresholds, CertificationThresholds):
        return thresholds
    return CertificationThresholds.from_config(thresholds)


@dataclass(frozen=True)
class FrozenActionRule:
    source_rule_id: str
    condition: RuleCondition
    action: RuleAction
    w: float
    precision: float
    rival_lcb: float = 0.0
    rival_samples: int = 0

    def __post_init__(self) -> None:
        if type(self.source_rule_id) is not str:
            raise TypeError("frozen action rule source_rule_id must be a string")
        source = self.source_rule_id.strip()
        if not source:
            raise ValueError("frozen action rule requires a source_rule_id")
        if not isinstance(self.condition, RuleCondition):
            raise TypeError("frozen action condition must be RuleCondition")
        if not isinstance(self.action, RuleAction):
            object.__setattr__(self, "action", RuleAction.from_candidate(self.action))
        weight = _finite_numeric(self.w, name="frozen expected effect")
        precision = _finite_numeric(self.precision, name="frozen precision")
        rival_lcb = _finite_numeric(self.rival_lcb, name="frozen rival LCB")
        rival_samples = self.rival_samples
        if precision <= 0.0:
            raise ValueError("frozen precision must be finite and positive")
        if type(rival_samples) is not int or rival_samples < 0:
            raise ValueError("frozen rival sample count must be a non-negative integer")
        # Defensively rebuild nested values even if a caller passes objects
        # currently owned by the mutable population.
        object.__setattr__(self, "source_rule_id", source)
        object.__setattr__(
            self, "condition", RuleCondition.from_dict(self.condition.to_dict())
        )
        object.__setattr__(self, "action", RuleAction.from_dict(self.action.to_dict()))
        object.__setattr__(self, "w", weight)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "rival_lcb", rival_lcb)
        object.__setattr__(self, "rival_samples", rival_samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_rule_id": self.source_rule_id,
            "condition": self.condition.to_dict(),
            "action": self.action.to_dict(),
            "w": self.w,
            "precision": self.precision,
            "rival_lcb": self.rival_lcb,
            "rival_samples": self.rival_samples,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenActionRule":
        if not isinstance(payload, Mapping):
            raise TypeError("frozen action rule payload must be a mapping")
        expected_fields = {
            "source_rule_id",
            "condition",
            "action",
            "w",
            "precision",
            "rival_lcb",
            "rival_samples",
        }
        if set(payload) != expected_fields:
            raise ValueError(
                "frozen action rule payload fields must exactly match "
                "the canonical schema"
            )
        return cls(
            source_rule_id=payload["source_rule_id"],
            condition=RuleCondition.from_dict(payload["condition"]),
            action=RuleAction.from_dict(payload["action"]),
            w=payload["w"],
            precision=payload["precision"],
            rival_lcb=payload["rival_lcb"],
            rival_samples=payload["rival_samples"],
        )


@dataclass(frozen=True)
class FrozenVetoRule:
    source_rule_id: str
    condition: RuleCondition

    def __post_init__(self) -> None:
        if type(self.source_rule_id) is not str:
            raise TypeError("frozen veto rule source_rule_id must be a string")
        source = self.source_rule_id.strip()
        if not source:
            raise ValueError("frozen veto rule requires a source_rule_id")
        if not isinstance(self.condition, RuleCondition):
            raise TypeError("frozen veto condition must be RuleCondition")
        object.__setattr__(self, "source_rule_id", source)
        object.__setattr__(
            self, "condition", RuleCondition.from_dict(self.condition.to_dict())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_rule_id": self.source_rule_id,
            "condition": self.condition.to_dict(),
            "veto": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenVetoRule":
        if not isinstance(payload, Mapping):
            raise TypeError("frozen veto rule payload must be a mapping")
        expected_fields = {"source_rule_id", "condition", "veto"}
        if set(payload) != expected_fields:
            raise ValueError(
                "frozen veto rule payload fields must exactly match "
                "the canonical schema"
            )
        if payload["veto"] is not True:
            raise ValueError("serialized veto rule must have veto=true")
        return cls(
            source_rule_id=payload["source_rule_id"],
            condition=RuleCondition.from_dict(payload["condition"]),
        )


CertifiedRule = FrozenActionRule | FrozenVetoRule


def _subtract_moments(
    total: OnlineMoments,
    inherited: OnlineMoments,
) -> OnlineMoments:
    """Remove one exact Welford accumulator from their combined statistics."""

    if inherited.n > total.n:
        raise ValueError(
            "offspring evidence ledger precedes its inherited birth baseline"
        )
    remaining_n = total.n - inherited.n
    if inherited.n == 0:
        return total.copy()
    if remaining_n == 0:
        if not (
            math.isclose(total.mean, inherited.mean, rel_tol=1.0e-12, abs_tol=1.0e-12)
            and math.isclose(total.m2, inherited.m2, rel_tol=1.0e-12, abs_tol=1.0e-12)
        ):
            raise ValueError("offspring evidence ledger does not contain its baseline")
        return OnlineMoments()
    remaining_mean = (total.mean * total.n - inherited.mean * inherited.n) / remaining_n
    delta = remaining_mean - inherited.mean
    cross_term = delta * delta * inherited.n * remaining_n / float(total.n)
    remaining_m2 = total.m2 - inherited.m2 - cross_term
    tolerance = 1.0e-10 * max(1.0, abs(total.m2), abs(inherited.m2), abs(cross_term))
    if remaining_m2 < -tolerance:
        raise ValueError(
            "offspring evidence ledger is inconsistent with its inherited baseline"
        )
    return OnlineMoments(
        n=remaining_n,
        mean=remaining_mean,
        m2=max(0.0, remaining_m2),
    )


def independent_evidence_moments(
    rule: MutableRule,
    ledger_name: str,
) -> OnlineMoments:
    """Return exact moments observed independently after this rule's GA birth."""

    if ledger_name not in {"evolution", "deployment"}:
        raise ValueError("unknown rule evidence ledger")
    ledger: OnlineMoments = getattr(rule, ledger_name)
    if not rule.parent_ids:
        return ledger.copy()
    raw_baseline = rule.provenance.get(OFFSPRING_EVIDENCE_PROVENANCE_KEY)
    if raw_baseline is None:
        return OnlineMoments()
    if not isinstance(raw_baseline, Mapping):
        raise ValueError("offspring evidence baseline must be a mapping")
    expected_fields = {"artifact_version", "evolution", "deployment"}
    if set(raw_baseline) != expected_fields:
        raise ValueError(
            "offspring evidence baseline fields must exactly match the canonical schema"
        )
    if raw_baseline["artifact_version"] != OFFSPRING_EVIDENCE_BASELINE_VERSION:
        raise ValueError("unsupported offspring evidence baseline version")
    inherited = OnlineMoments.from_dict(raw_baseline[ledger_name])
    return _subtract_moments(ledger, inherited)


def independent_evidence_samples(rule: MutableRule, ledger_name: str) -> int:
    """Return the independently observed post-birth sample count."""

    return independent_evidence_moments(rule, ledger_name).n


def _certify_vanilla_accuracy_rule(
    rule: MutableRule,
    settings: CertificationThresholds,
) -> FrozenActionRule | None:
    """Publish predicted reward with bounded mean accuracy as its weight."""

    accuracy = independent_evidence_moments(rule, "evolution")
    prediction = independent_evidence_moments(rule, "deployment")
    if (
        prediction.n < settings.action_min_noop_samples
        or accuracy.n < settings.action_min_noop_samples
    ):
        return None
    accuracy_weight = float(accuracy.mean)
    if not math.isfinite(accuracy_weight) or not 0.0 < accuracy_weight <= 1.0:
        raise ValueError("vanilla accuracy ledger mean must be finite and in (0, 1]")
    return FrozenActionRule(
        source_rule_id=rule.rule_id,
        condition=rule.condition,
        action=rule.action,
        w=prediction.mean,
        precision=accuracy_weight,
        rival_lcb=prediction.lcb(
            settings.contender_lcb_z,
            settings.variance_floor,
        ),
        rival_samples=prediction.n,
    )


def certify_rule(
    rule: MutableRule,
    thresholds: CertificationThresholds | Any | None = None,
    *,
    credit_mode: str = "causal",
) -> CertifiedRule | None:
    """Return a detached certified snapshot, or ``None`` if evidence fails."""

    if not isinstance(rule, MutableRule):
        raise TypeError("certification requires a MutableRule")
    settings = _certification_thresholds(thresholds)
    mode = validate_credit_mode(credit_mode, name="certification credit_mode")
    if mode == VANILLA_ACCURACY_CREDIT_MODE:
        return _certify_vanilla_accuracy_rule(rule, settings)
    assert mode == CAUSAL_CREDIT_MODE
    if rule.action.is_no_op:
        evidence = independent_evidence_moments(rule, "evolution")
        if evidence.n < settings.veto_min_rival_samples:
            return None
        if evidence.lcb(settings.veto_lcb_z, settings.variance_floor) <= 0.0:
            return None
        return FrozenVetoRule(
            source_rule_id=rule.rule_id,
            condition=rule.condition,
        )

    evidence = independent_evidence_moments(rule, "deployment")
    if evidence.n < settings.action_min_noop_samples:
        return None
    if evidence.lcb(settings.action_lcb_z, settings.variance_floor) <= 0.0:
        return None
    rival_evidence = independent_evidence_moments(rule, "evolution")
    return FrozenActionRule(
        source_rule_id=rule.rule_id,
        condition=rule.condition,
        action=rule.action,
        w=evidence.mean,
        precision=evidence.precision(settings.variance_floor),
        rival_lcb=rival_evidence.lcb(
            settings.contender_lcb_z,
            settings.variance_floor,
        ),
        rival_samples=rival_evidence.n,
    )


def certify_population(
    rules: Iterable[MutableRule],
    thresholds: CertificationThresholds | Any | None = None,
    *,
    credit_mode: str = "causal",
) -> tuple[tuple[FrozenActionRule, ...], tuple[FrozenVetoRule, ...]]:
    mode = validate_credit_mode(credit_mode, name="certification credit_mode")
    actions: list[FrozenActionRule] = []
    vetoes: list[FrozenVetoRule] = []
    for rule in rules:
        certified = certify_rule(
            rule,
            thresholds,
            credit_mode=mode,
        )
        if isinstance(certified, FrozenActionRule):
            actions.append(certified)
        elif isinstance(certified, FrozenVetoRule):
            vetoes.append(certified)
    actions.sort(key=lambda item: (item.action.key, item.source_rule_id))
    vetoes.sort(key=lambda item: item.source_rule_id)
    return tuple(actions), tuple(vetoes)


__all__ = [
    "CertificationThresholds",
    "CertifiedRule",
    "FrozenActionRule",
    "FrozenVetoRule",
    "certify_population",
    "certify_rule",
    "independent_evidence_moments",
    "independent_evidence_samples",
]
