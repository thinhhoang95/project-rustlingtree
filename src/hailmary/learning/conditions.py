"""Immutable interval conditions bound to a role and feature schema."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Interval:
    """Inclusive finite interval with ``None`` representing an open end."""

    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        lower = None if self.lower is None else float(self.lower)
        upper = None if self.upper is None else float(self.upper)
        if lower is not None and not math.isfinite(lower):
            raise ValueError("interval lower bound must be finite or None")
        if upper is not None and not math.isfinite(upper):
            raise ValueError("interval upper bound must be finite or None")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("interval lower bound cannot exceed upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def coerce(cls, value: "Interval | Sequence[float | None]") -> "Interval":
        if isinstance(value, cls):
            return value
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) != 2
        ):
            raise TypeError("an interval must be Interval or a two-item sequence")
        return cls(value[0], value[1])

    def matches(self, value: float) -> bool:
        normalized = float(value)
        if not math.isfinite(normalized):
            return False
        return bool(
            (self.lower is None or normalized >= self.lower)
            and (self.upper is None or normalized <= self.upper)
        )

    def contains(self, other: "Interval") -> bool:
        candidate = Interval.coerce(other)
        lower_contains = self.lower is None or (
            candidate.lower is not None and self.lower <= candidate.lower
        )
        upper_contains = self.upper is None or (
            candidate.upper is not None and self.upper >= candidate.upper
        )
        return lower_contains and upper_contains

    def shifted(self, delta: float) -> "Interval":
        offset = float(delta)
        if not math.isfinite(offset):
            raise ValueError("interval shift must be finite")
        return Interval(
            None if self.lower is None else self.lower + offset,
            None if self.upper is None else self.upper + offset,
        )

    def widened(self, amount: float) -> "Interval":
        width = float(amount)
        if not math.isfinite(width) or width < 0.0:
            raise ValueError("widening amount must be finite and non-negative")
        return Interval(
            None if self.lower is None else self.lower - width,
            None if self.upper is None else self.upper + width,
        )

    def clipped(self, lower: float | None, upper: float | None) -> "Interval":
        bounds = Interval(lower, upper)
        new_lower = self.lower
        if bounds.lower is not None:
            new_lower = (
                bounds.lower if new_lower is None else max(new_lower, bounds.lower)
            )
        new_upper = self.upper
        if bounds.upper is not None:
            new_upper = (
                bounds.upper if new_upper is None else min(new_upper, bounds.upper)
            )
        return Interval(new_lower, new_upper)

    def to_list(self) -> list[float | None]:
        return [self.lower, self.upper]


@dataclass(frozen=True)
class RuleCondition:
    """A conjunction of inclusive predicates for one role/schema contract.

    A missing predicate is unconstrained.  The mapping is defensively copied
    and exposed through ``MappingProxyType``, preventing a population mutation
    from changing a published condition by aliasing the original dictionary.
    """

    role_type: str
    schema_hash: str
    intervals: Mapping[str, Interval | Sequence[float | None]]

    def __post_init__(self) -> None:
        role = str(self.role_type).strip()
        schema = str(self.schema_hash).strip()
        if not role or not schema:
            raise ValueError("condition role_type and schema_hash cannot be empty")
        normalized: dict[str, Interval] = {}
        if not isinstance(self.intervals, Mapping):
            raise TypeError("condition intervals must be a mapping")
        for raw_name, raw_interval in self.intervals.items():
            name = str(raw_name).strip()
            if not name or name != str(raw_name):
                raise ValueError(
                    "condition feature names must be non-empty stripped strings"
                )
            normalized[name] = Interval.coerce(raw_interval)
        object.__setattr__(self, "role_type", role)
        object.__setattr__(self, "schema_hash", schema)
        object.__setattr__(
            self, "intervals", MappingProxyType(dict(sorted(normalized.items())))
        )

    def matches(
        self,
        values: Mapping[str, float] | Any,
        *,
        role_type: str | None = None,
        schema_hash: str | None = None,
    ) -> bool:
        supplied_role = role_type
        if supplied_role is None:
            supplied_role = getattr(values, "role_type", None)
        if supplied_role is not None and str(supplied_role) != self.role_type:
            return False

        supplied_schema = schema_hash
        if supplied_schema is None:
            supplied_schema = getattr(values, "schema_hash", None)
        if supplied_schema is not None and str(supplied_schema) != self.schema_hash:
            return False

        named = getattr(values, "named", values)
        if not isinstance(named, Mapping):
            raise TypeError(
                "condition matching requires a feature mapping or FeatureVector"
            )
        for name, interval in self.intervals.items():
            if name not in named or not interval.matches(float(named[name])):
                return False
        return True

    def contains(self, other: "RuleCondition") -> bool:
        if not isinstance(other, RuleCondition):
            return False
        if self.role_type != other.role_type or self.schema_hash != other.schema_hash:
            return False
        for name, interval in self.intervals.items():
            candidate = other.intervals.get(name)
            if candidate is None or not interval.contains(candidate):
                return False
        return True

    def with_interval(
        self,
        feature_name: str,
        interval: Interval | Sequence[float | None],
    ) -> "RuleCondition":
        name = str(feature_name).strip()
        if not name:
            raise ValueError("feature_name cannot be empty")
        updated = dict(self.intervals)
        updated[name] = Interval.coerce(interval)
        return RuleCondition(self.role_type, self.schema_hash, updated)

    def without_interval(self, feature_name: str) -> "RuleCondition":
        updated = dict(self.intervals)
        updated.pop(str(feature_name), None)
        return RuleCondition(self.role_type, self.schema_hash, updated)

    def shifted(self, feature_name: str, delta: float) -> "RuleCondition":
        try:
            interval = self.intervals[str(feature_name)]
        except KeyError as exc:
            raise KeyError(f"condition has no predicate for {feature_name!r}") from exc
        return self.with_interval(str(feature_name), interval.shifted(delta))

    def widened(self, feature_name: str, amount: float) -> "RuleCondition":
        try:
            interval = self.intervals[str(feature_name)]
        except KeyError as exc:
            raise KeyError(f"condition has no predicate for {feature_name!r}") from exc
        return self.with_interval(str(feature_name), interval.widened(amount))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_type": self.role_type,
            "schema_hash": self.schema_hash,
            "intervals": {
                name: interval.to_list() for name, interval in self.intervals.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuleCondition":
        if not isinstance(payload, Mapping):
            raise TypeError("condition payload must be a mapping")
        expected_fields = {"role_type", "schema_hash", "intervals"}
        if set(payload) != expected_fields:
            raise ValueError(
                "condition payload fields must be exactly role_type, schema_hash, "
                "and intervals"
            )
        role_type = payload["role_type"]
        schema_hash = payload["schema_hash"]
        if type(role_type) is not str or type(schema_hash) is not str:
            raise TypeError(
                "serialized condition role_type and schema_hash must be strings"
            )
        intervals = payload["intervals"]
        if not isinstance(intervals, Mapping):
            raise TypeError("serialized condition intervals must be a mapping")
        return cls(
            role_type=role_type,
            schema_hash=schema_hash,
            intervals=intervals,
        )


__all__ = ["Interval", "RuleCondition"]
