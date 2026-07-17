"""Bounded mutable population and deterministic match queries."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Mapping, Sequence

from hailmary.learning.modes import CAUSAL_CREDIT_MODE, validate_credit_mode
from hailmary.learning.rules import MutableRule, RuleAction


POPULATION_VERSION = "hailmary.population.v2"


class Population:
    """A bounded XCSR population measured by total classifier numerosity.

    Deletion pressure is deliberately not hidden here: until the evolutionary
    layer supplies a deletion choice, exceeding the configured bound raises an
    explicit error.  Identical conditions/actions are merged in standard XCS
    fashion and their sufficient statistics are combined exactly.
    """

    def __init__(
        self,
        rules: Sequence[MutableRule] = (),
        *,
        max_size: int = 1_000,
        credit_mode: str = CAUSAL_CREDIT_MODE,
    ) -> None:
        if type(max_size) is not int or max_size < 1:
            raise ValueError("population max_size must be a positive integer")
        self.max_size = max_size
        self._credit_mode = validate_credit_mode(
            credit_mode,
            name="population credit_mode",
        )
        self._rules: dict[str, MutableRule] = {}
        for rule in rules:
            self.add(rule)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "credit_mode" or (
            name == "_credit_mode" and hasattr(self, "_credit_mode")
        ):
            raise AttributeError("population credit_mode is immutable")
        object.__setattr__(self, name, value)

    @property
    def credit_mode(self) -> str:
        """Semantic identity of both mutable evidence ledgers."""

        return self._credit_mode

    def __len__(self) -> int:
        """Return the number of macro-classifiers."""

        return len(self._rules)

    def __iter__(self) -> Iterator[MutableRule]:
        return iter(self.rules)

    @property
    def rules(self) -> tuple[MutableRule, ...]:
        return tuple(self._rules[key] for key in sorted(self._rules))

    @property
    def macro_size(self) -> int:
        return len(self._rules)

    @property
    def total_numerosity(self) -> int:
        return sum(rule.numerosity for rule in self._rules.values())

    def get(self, rule_id: str) -> MutableRule:
        try:
            return self._rules[str(rule_id)]
        except KeyError as exc:
            raise KeyError(f"unknown rule {rule_id!r}") from exc

    @staticmethod
    def _equivalent(left: MutableRule, right: MutableRule) -> bool:
        return left.condition == right.condition and left.action == right.action

    @staticmethod
    def _merge_into(target: MutableRule, source: MutableRule) -> MutableRule:
        if target.parent_ids or source.parent_ids:
            raise ValueError(
                "equivalent merges involving parented rules would corrupt "
                "post-birth evidence provenance"
            )
        target.numerosity += source.numerosity
        target.evolution.combine(source.evolution)
        target.deployment.combine(source.deployment)
        target.last_ga_epoch = max(target.last_ga_epoch, source.last_ga_epoch)
        return target

    def add(self, rule: MutableRule, *, merge_equivalent: bool = True) -> MutableRule:
        if not isinstance(rule, MutableRule):
            raise TypeError("population entries must be MutableRule")

        existing_by_id = self._rules.get(rule.rule_id)
        if existing_by_id is not None:
            if existing_by_id is rule:
                return existing_by_id
            if merge_equivalent and self._equivalent(existing_by_id, rule):
                if self.total_numerosity + rule.numerosity > self.max_size:
                    raise OverflowError("population capacity would be exceeded")
                return self._merge_into(existing_by_id, rule)
            raise ValueError(f"duplicate rule_id {rule.rule_id!r}")

        if self.total_numerosity + rule.numerosity > self.max_size:
            raise OverflowError("population capacity would be exceeded")

        if merge_equivalent:
            equivalent = next(
                (item for item in self.rules if self._equivalent(item, rule)),
                None,
            )
            if equivalent is not None:
                return self._merge_into(equivalent, rule)

        self._rules[rule.rule_id] = rule
        return rule

    def remove(self, rule_id: str, *, numerosity: int = 1) -> MutableRule | None:
        if isinstance(numerosity, bool) or int(numerosity) < 1:
            raise ValueError("removal numerosity must be positive")
        rule = self.get(rule_id)
        count = int(numerosity)
        if count > rule.numerosity:
            raise ValueError("cannot remove more numerosity than a rule owns")
        if count == rule.numerosity:
            return self._rules.pop(rule.rule_id)
        rule.numerosity -= count
        return None

    def match(
        self,
        values: Mapping[str, float] | Any,
        *,
        role_type: str | None = None,
        schema_hash: str | None = None,
        action: RuleAction | Any | None = None,
    ) -> tuple[MutableRule, ...]:
        inferred_role = role_type or getattr(values, "role_type", None)
        inferred_schema = schema_hash or getattr(values, "schema_hash", None)
        normalized_action = (
            None if action is None else RuleAction.from_candidate(action)
        )
        matches: list[MutableRule] = []
        for rule in self.rules:
            if inferred_role is not None and rule.role_type != str(inferred_role):
                continue
            if inferred_schema is not None and rule.schema_hash != str(inferred_schema):
                continue
            if normalized_action is not None and rule.action != normalized_action:
                continue
            if rule.matches(
                values,
                role_type=None if inferred_role is None else str(inferred_role),
                schema_hash=None if inferred_schema is None else str(inferred_schema),
            ):
                matches.append(rule)
        return tuple(matches)

    def by_action(self, action: RuleAction | Any) -> tuple[MutableRule, ...]:
        normalized = RuleAction.from_candidate(action)
        return tuple(rule for rule in self.rules if rule.action == normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": POPULATION_VERSION,
            "credit_mode": self.credit_mode,
            "max_size": self.max_size,
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Population":
        if not isinstance(payload, Mapping):
            raise TypeError("population payload must be a mapping")
        expected_fields = {"artifact_version", "credit_mode", "max_size", "rules"}
        if set(payload) != expected_fields:
            raise ValueError(
                "population payload fields must exactly match the canonical schema"
            )
        if payload["artifact_version"] != POPULATION_VERSION:
            raise ValueError("unsupported population artifact version")
        credit_mode = validate_credit_mode(
            payload["credit_mode"],
            name="serialized population credit_mode",
        )
        max_size = payload["max_size"]
        if type(max_size) is not int or max_size < 1:
            raise ValueError("population max_size must be a positive integer")
        raw_rules = payload["rules"]
        if isinstance(raw_rules, (str, bytes)) or not isinstance(raw_rules, Sequence):
            raise TypeError("serialized population rules must be a sequence")
        population = cls(max_size=max_size, credit_mode=credit_mode)
        for item in raw_rules:
            population.add(
                MutableRule.from_dict(item),
                merge_equivalent=False,
            )
        return population


RulePopulation = Population

__all__ = ["POPULATION_VERSION", "Population", "RulePopulation"]
