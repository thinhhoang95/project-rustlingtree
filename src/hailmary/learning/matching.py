"""Immutable epoch-local anchor contexts and rule match sets."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from hailmary.features.schema import FeatureVector
from hailmary.learning.population import Population
from hailmary.learning.rules import RuleAction


def _require_immutable_candidate(candidate: object) -> None:
    parameters = getattr(type(candidate), "__dataclass_params__", None)
    if parameters is None or not bool(getattr(parameters, "frozen", False)):
        raise TypeError("anchor candidates must be frozen dataclass records")


@dataclass(frozen=True, slots=True)
class AnchorContext:
    """One fully frozen anchor decision boundary for a training epoch."""

    role_type: str
    schema_hash: str
    anchor_id: str
    vector: FeatureVector
    candidates: tuple[Any, ...]

    def __post_init__(self) -> None:
        role = str(self.role_type).strip()
        schema = str(self.schema_hash).strip()
        anchor = str(self.anchor_id).strip()
        if not role or not schema or not anchor:
            raise ValueError("anchor role, schema, and identity cannot be empty")
        if not isinstance(self.vector, FeatureVector):
            raise TypeError("anchor vector must be a FeatureVector")
        if self.vector.schema_hash != schema:
            raise ValueError("anchor schema_hash does not match its feature vector")

        feasible: list[Any] = []
        for candidate in tuple(self.candidates):
            _require_immutable_candidate(candidate)
            RuleAction.from_candidate(candidate)
            if bool(getattr(candidate, "feasible", True)):
                feasible.append(candidate)
        if not feasible:
            raise ValueError(
                "an anchor context requires at least one feasible action candidate"
            )

        object.__setattr__(self, "role_type", role)
        object.__setattr__(self, "schema_hash", schema)
        object.__setattr__(self, "anchor_id", anchor)
        object.__setattr__(self, "candidates", tuple(feasible))

    @property
    def candidate_actions(self) -> tuple[RuleAction, ...]:
        result: list[RuleAction] = []
        seen: set[RuleAction] = set()
        for candidate in self.candidates:
            action = RuleAction.from_candidate(candidate)
            if action not in seen:
                seen.add(action)
                result.append(action)
        return tuple(result)

    def candidate_for(self, action: RuleAction | Any) -> Any:
        normalized = RuleAction.from_candidate(action)
        for candidate in self.candidates:
            if RuleAction.from_candidate(candidate) == normalized:
                return candidate
        raise KeyError(
            f"action {normalized.key!r} is not feasible for anchor {self.anchor_id!r}"
        )


@dataclass(frozen=True)
class MatchSet:
    """Frozen rule recipients grouped by each exact feasible action."""

    context: AnchorContext
    rule_ids_by_action: Mapping[RuleAction, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.context, AnchorContext):
            raise TypeError("match-set context must be AnchorContext")
        feasible = self.context.candidate_actions
        feasible_set = set(feasible)
        raw = self.rule_ids_by_action
        if not isinstance(raw, Mapping):
            raise TypeError("rule_ids_by_action must be a mapping")
        extra = {
            RuleAction.from_candidate(action)
            for action in raw
            if RuleAction.from_candidate(action) not in feasible_set
        }
        if extra:
            raise ValueError(
                "match sets cannot retain advocates for infeasible actions"
            )

        normalized: dict[RuleAction, tuple[str, ...]] = {}
        for action in feasible:
            identifiers = tuple(
                sorted({str(rule_id).strip() for rule_id in raw.get(action, ())})
            )
            if any(not rule_id for rule_id in identifiers):
                raise ValueError("matched rule IDs cannot be empty")
            normalized[action] = identifiers
        object.__setattr__(
            self,
            "rule_ids_by_action",
            MappingProxyType(normalized),
        )

    @property
    def role_type(self) -> str:
        return self.context.role_type

    @property
    def schema_hash(self) -> str:
        return self.context.schema_hash

    @property
    def anchor_id(self) -> str:
        return self.context.anchor_id

    @property
    def vector(self) -> FeatureVector:
        return self.context.vector

    @property
    def candidates(self) -> tuple[Any, ...]:
        return self.context.candidates

    @property
    def candidate_actions(self) -> tuple[RuleAction, ...]:
        return self.context.candidate_actions

    @property
    def matched_rule_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    rule_id
                    for identifiers in self.rule_ids_by_action.values()
                    for rule_id in identifiers
                }
            )
        )

    @property
    def missing_actions(self) -> tuple[RuleAction, ...]:
        return tuple(
            action
            for action in self.candidate_actions
            if not self.rule_ids_by_action[action]
        )

    def advocates(self, action: RuleAction | Any) -> tuple[str, ...]:
        normalized = RuleAction.from_candidate(action)
        try:
            return self.rule_ids_by_action[normalized]
        except KeyError as exc:
            raise KeyError(
                f"action {normalized.key!r} is not feasible for anchor {self.anchor_id!r}"
            ) from exc


def build_match_set(context: AnchorContext, population: Population) -> MatchSet:
    """Freeze matching co-advocate IDs for the context's feasible actions."""

    if not isinstance(context, AnchorContext):
        raise TypeError("context must be AnchorContext")
    if not isinstance(population, Population):
        raise TypeError("population must be Population")

    grouped: dict[RuleAction, list[str]] = {
        action: [] for action in context.candidate_actions
    }
    for rule in population.match(
        context.vector,
        role_type=context.role_type,
        schema_hash=context.schema_hash,
    ):
        recipients = grouped.get(rule.action)
        if recipients is not None:
            recipients.append(rule.rule_id)
    return MatchSet(
        context=context,
        rule_ids_by_action={
            action: tuple(rule_ids) for action, rule_ids in grouped.items()
        },
    )


__all__ = ["AnchorContext", "MatchSet", "build_match_set"]
