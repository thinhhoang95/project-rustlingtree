"""Frozen three-arm credit routing and a labeled vanilla-XCS control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from hailmary.learning.matching import MatchSet
from hailmary.learning.modes import (
    CAUSAL_CREDIT_MODE,
    VANILLA_ACCURACY_CREDIT_MODE,
)
from hailmary.learning.population import Population
from hailmary.learning.rules import MutableRule, RuleAction


def _finite(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _rule_ids(values: tuple[str, ...] | Any, *, name: str) -> tuple[str, ...]:
    try:
        normalized = tuple(sorted({str(value).strip() for value in values}))
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of rule IDs") from exc
    if any(not value for value in normalized):
        raise ValueError(f"{name} cannot contain empty rule IDs")
    return normalized


@dataclass(frozen=True, slots=True)
class CreditSignals:
    """The four signed effects emitted by one coupled three-arm rollout."""

    delta_rival: float
    delta_selected_noop: float
    delta_contender_noop: float
    delta_veto: float

    def __post_init__(self) -> None:
        for name in (
            "delta_rival",
            "delta_selected_noop",
            "delta_contender_noop",
            "delta_veto",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name=name),
            )

    @classmethod
    def from_arm_scores(
        cls,
        *,
        selected: float,
        contender: float,
        no_op: float,
    ) -> "CreditSignals":
        """Derive the audit-plan deltas from arm scores A, B, and C."""

        y_a = _finite(selected, name="selected")
        y_b = _finite(contender, name="contender")
        y_c = _finite(no_op, name="no_op")
        return cls(
            delta_rival=y_a - y_b,
            delta_selected_noop=y_a - y_c,
            delta_contender_noop=y_b - y_c,
            delta_veto=y_c - max(y_a, y_b),
        )

    @classmethod
    def coerce(
        cls,
        value: "CreditSignals | Mapping[str, Any] | Any",
    ) -> "CreditSignals":
        if isinstance(value, cls):
            return value
        getter = (
            value.get
            if isinstance(value, Mapping)
            else lambda name: getattr(value, name)
        )
        return cls(
            delta_rival=getter("delta_rival"),
            delta_selected_noop=getter("delta_selected_noop"),
            delta_contender_noop=getter("delta_contender_noop"),
            delta_veto=getter("delta_veto"),
        )

    from_rollout = coerce


@dataclass(frozen=True, slots=True)
class CreditRecipients:
    """Exact immutable rule IDs that owned the root match before rollout."""

    anchor_id: str
    selected_action: RuleAction
    contender_action: RuleAction
    selected_rule_ids: tuple[str, ...] = ()
    contender_rule_ids: tuple[str, ...] = ()
    no_op_rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        anchor_id = str(self.anchor_id).strip()
        if not anchor_id:
            raise ValueError("credit recipients require an anchor_id")
        selected = RuleAction.from_candidate(self.selected_action)
        contender = RuleAction.from_candidate(self.contender_action)
        selected_ids = _rule_ids(
            self.selected_rule_ids,
            name="selected_rule_ids",
        )
        contender_ids = _rule_ids(
            self.contender_rule_ids,
            name="contender_rule_ids",
        )
        no_op_ids = _rule_ids(self.no_op_rule_ids, name="no_op_rule_ids")
        groups = (set(selected_ids), set(contender_ids), set(no_op_ids))
        if any(
            groups[left] & groups[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("credit recipient groups must be disjoint")
        if selected.is_no_op and selected_ids:
            raise ValueError(
                "no-op selected rules belong only to the veto recipient group"
            )
        if contender.is_no_op and contender_ids:
            raise ValueError(
                "no-op contender rules belong only to the veto recipient group"
            )

        object.__setattr__(self, "anchor_id", anchor_id)
        object.__setattr__(self, "selected_action", selected)
        object.__setattr__(self, "contender_action", contender)
        object.__setattr__(self, "selected_rule_ids", selected_ids)
        object.__setattr__(self, "contender_rule_ids", contender_ids)
        object.__setattr__(self, "no_op_rule_ids", no_op_ids)

    @classmethod
    def from_match_set(
        cls,
        match_set: MatchSet,
        *,
        selected_action: RuleAction | Any,
        contender_action: RuleAction | Any,
    ) -> "CreditRecipients":
        if not isinstance(match_set, MatchSet):
            raise TypeError("match_set must be MatchSet")
        selected = RuleAction.from_candidate(selected_action)
        contender = RuleAction.from_candidate(contender_action)
        if selected == contender and not selected.is_no_op:
            raise ValueError("selected and contender ordinary actions must be distinct")

        selected_advocates = match_set.advocates(selected)
        contender_advocates = match_set.advocates(contender)
        no_op = RuleAction.no_op()
        no_op_advocates = (
            match_set.advocates(no_op) if no_op in match_set.candidate_actions else ()
        )
        return cls(
            anchor_id=match_set.anchor_id,
            selected_action=selected,
            contender_action=contender,
            selected_rule_ids=() if selected.is_no_op else selected_advocates,
            contender_rule_ids=() if contender.is_no_op else contender_advocates,
            no_op_rule_ids=no_op_advocates,
        )

    @property
    def all_rule_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    *self.selected_rule_ids,
                    *self.contender_rule_ids,
                    *self.no_op_rule_ids,
                )
            )
        )


def _resolve_rules(
    population: Population,
    rule_ids: tuple[str, ...],
    *,
    expected_action: RuleAction,
) -> tuple[MutableRule, ...]:
    rules: list[MutableRule] = []
    for rule_id in rule_ids:
        rule = population.get(rule_id)
        if rule.action != expected_action:
            raise RuntimeError(
                f"frozen recipient {rule_id!r} no longer owns action "
                f"{expected_action.key!r}"
            )
        rules.append(rule)
    return tuple(rules)


class CausalCreditAssigner:
    """Apply the exact signed three-arm samples to frozen co-advocates.

    Selected ordinary advocates receive positive rival and selected-versus-
    no-op samples in their evolution/deployment ledgers. Contender ordinary
    advocates receive the negated rival and contender-versus-no-op samples.
    Matching no-op rules receive only veto evidence in the evolution ledger.
    All recipients resolve before mutation, preventing partial credit when a
    frozen rule ID is missing or has changed action.
    """

    __slots__ = ()

    @staticmethod
    def recipients(
        match_set: MatchSet,
        *,
        selected_action: RuleAction | Any,
        contender_action: RuleAction | Any,
    ) -> CreditRecipients:
        return CreditRecipients.from_match_set(
            match_set,
            selected_action=selected_action,
            contender_action=contender_action,
        )

    def assign(
        self,
        population: Population,
        match_set: MatchSet,
        *,
        selected_action: RuleAction | Any,
        contender_action: RuleAction | Any,
        signals: CreditSignals | Mapping[str, Any] | Any,
    ) -> CreditRecipients:
        if not isinstance(population, Population):
            raise TypeError("population must be Population")
        if population.credit_mode != CAUSAL_CREDIT_MODE:
            raise ValueError("causal credit requires a causal-ledger population")
        recipients = self.recipients(
            match_set,
            selected_action=selected_action,
            contender_action=contender_action,
        )
        if recipients.selected_action.is_no_op and recipients.contender_action.is_no_op:
            raise ValueError("an all-no-op epoch is not a credit-bearing experiment")
        normalized = CreditSignals.coerce(signals)

        selected_rules = _resolve_rules(
            population,
            recipients.selected_rule_ids,
            expected_action=recipients.selected_action,
        )
        contender_rules = _resolve_rules(
            population,
            recipients.contender_rule_ids,
            expected_action=recipients.contender_action,
        )
        no_op_rules = _resolve_rules(
            population,
            recipients.no_op_rule_ids,
            expected_action=RuleAction.no_op(),
        )
        if not recipients.selected_action.is_no_op and not selected_rules:
            raise ValueError("selected ordinary action has no frozen co-advocates")
        if not recipients.contender_action.is_no_op and not contender_rules:
            raise ValueError("contender ordinary action has no frozen co-advocates")

        for rule in selected_rules:
            rule.update_evolution(normalized.delta_rival)
            rule.update_deployment(normalized.delta_selected_noop)
        for rule in contender_rules:
            rule.update_evolution(-normalized.delta_rival)
            rule.update_deployment(normalized.delta_contender_noop)
        for rule in no_op_rules:
            rule.update_evolution(normalized.delta_veto)
        return recipients


@dataclass(frozen=True, slots=True)
class VanillaAccuracyUpdate:
    """One auditable vanilla-control update computed before rule mutation."""

    rule_id: str
    pre_update_prediction: float
    selected_outcome: float
    absolute_error: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class VanillaAccuracyCreditAssigner:
    """A clearly labeled accuracy-based control using the same population.

    This mode must use a separate experimental population from causal credit.
    For each selected-action co-advocate, deployment stores raw selected-arm
    outcomes and its mean is therefore the pre-update prediction. Evolution
    stores bounded accuracy: 1 / (1 + abs(error) / accuracy_scale), making the
    existing GA, deletion, and subsumption mechanics accuracy-based.
    Contender and generic no-op recipients are untouched. Direct deployment
    updates intentionally include a selected no-op classifier because, in this
    control only, both ledgers are accuracy/prediction statistics rather than
    causal deployment/veto ledgers.
    """

    initial_prediction: float = 0.0
    accuracy_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_prediction",
            _finite(self.initial_prediction, name="initial_prediction"),
        )
        object.__setattr__(
            self,
            "accuracy_scale",
            _finite(self.accuracy_scale, name="accuracy_scale"),
        )
        if self.accuracy_scale <= 0.0:
            raise ValueError("accuracy_scale must be positive")

    def assign(
        self,
        population: Population,
        match_set: MatchSet,
        *,
        selected_action: RuleAction | Any,
        selected_outcome: float,
    ) -> tuple[VanillaAccuracyUpdate, ...]:
        if not isinstance(population, Population):
            raise TypeError("population must be Population")
        if population.credit_mode != VANILLA_ACCURACY_CREDIT_MODE:
            raise ValueError(
                "vanilla accuracy credit requires a vanilla_accuracy-ledger population"
            )
        if not isinstance(match_set, MatchSet):
            raise TypeError("match_set must be MatchSet")
        action = RuleAction.from_candidate(selected_action)
        rule_ids = match_set.advocates(action)
        if not rule_ids:
            raise ValueError("selected action has no frozen co-advocates")
        rules = _resolve_rules(
            population,
            rule_ids,
            expected_action=action,
        )
        outcome = _finite(selected_outcome, name="selected_outcome")

        updates: list[VanillaAccuracyUpdate] = []
        for rule in rules:
            prediction = (
                rule.deployment.mean if rule.deployment.n else self.initial_prediction
            )
            absolute_error = abs(outcome - prediction)
            accuracy = 1.0 / (1.0 + absolute_error / self.accuracy_scale)
            updates.append(
                VanillaAccuracyUpdate(
                    rule_id=rule.rule_id,
                    pre_update_prediction=prediction,
                    selected_outcome=outcome,
                    absolute_error=absolute_error,
                    accuracy=accuracy,
                )
            )

        for rule, update in zip(rules, updates, strict=True):
            rule.evolution.update(update.accuracy)
            rule.deployment.update(outcome)
        return tuple(updates)


CreditAssigner = CausalCreditAssigner | VanillaAccuracyCreditAssigner


def credit_assigner_configuration(
    assigner: CreditAssigner,
) -> dict[str, Any]:
    """Return the complete canonical identity of one supported credit mode."""

    if type(assigner) is CausalCreditAssigner:
        return {
            "mode": CAUSAL_CREDIT_MODE,
            "settings": {},
        }
    if type(assigner) is VanillaAccuracyCreditAssigner:
        return {
            "mode": VANILLA_ACCURACY_CREDIT_MODE,
            "settings": {
                "initial_prediction": assigner.initial_prediction,
                "accuracy_scale": assigner.accuracy_scale,
            },
        }
    raise TypeError(
        "credit_assigner must be CausalCreditAssigner or VanillaAccuracyCreditAssigner"
    )


def credit_assigner_from_configuration(
    payload: Mapping[str, Any],
) -> CreditAssigner:
    """Rebuild a supported immutable assigner from checkpoint metadata."""

    if not isinstance(payload, Mapping):
        raise TypeError("credit assigner configuration must be a mapping")
    if set(payload) != {"mode", "settings"}:
        raise ValueError(
            "credit assigner configuration requires exact mode/settings fields"
        )
    mode = str(payload["mode"]).strip()
    settings = payload["settings"]
    if not isinstance(settings, Mapping):
        raise TypeError("credit assigner settings must be a mapping")
    if mode == CAUSAL_CREDIT_MODE:
        if settings:
            raise ValueError("causal credit mode does not accept settings")
        return CausalCreditAssigner()
    if mode == VANILLA_ACCURACY_CREDIT_MODE:
        required = {"initial_prediction", "accuracy_scale"}
        if set(settings) != required:
            raise ValueError(
                "vanilla accuracy settings require exact "
                "initial_prediction/accuracy_scale fields"
            )
        return VanillaAccuracyCreditAssigner(
            initial_prediction=float(settings["initial_prediction"]),
            accuracy_scale=float(settings["accuracy_scale"]),
        )
    raise ValueError(f"unsupported credit mode {mode!r}")


__all__ = [
    "CAUSAL_CREDIT_MODE",
    "CausalCreditAssigner",
    "CreditAssigner",
    "CreditRecipients",
    "CreditSignals",
    "VANILLA_ACCURACY_CREDIT_MODE",
    "VanillaAccuracyCreditAssigner",
    "VanillaAccuracyUpdate",
    "credit_assigner_configuration",
    "credit_assigner_from_configuration",
]
