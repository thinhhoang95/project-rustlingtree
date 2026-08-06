"""Deterministic covering for currently feasible, unrepresented actions."""

from __future__ import annotations

import math

import numpy as np

from hailmary.config import LearningConfig
from hailmary.features.schema import FeatureField
from hailmary.learning.conditions import Interval, RuleCondition
from hailmary.learning.matching import MatchSet
from hailmary.learning.rules import MutableRule


def _feature_span(field: FeatureField, value: float) -> float:
    lower = field.lower_bound
    upper = field.upper_bound
    if lower is not None and upper is not None:
        return float(upper - lower)
    if lower is not None:
        return max(float(value - lower), abs(float(value)), 1.0)
    if upper is not None:
        return max(float(upper - value), abs(float(value)), 1.0)
    return max(2.0 * abs(float(value)), 1.0)


def _centered_interval(
    field: FeatureField,
    value: float,
    width_fraction: float,
) -> Interval:
    center = float(value)
    fraction = float(width_fraction)
    if not math.isfinite(center) or not math.isfinite(fraction):
        raise ValueError("covering values and width fractions must be finite")
    span = _feature_span(field, center)
    if span <= 0.0:
        return Interval(center, center)

    width = fraction * span
    lower = center - width / 2.0
    upper = center + width / 2.0
    if field.lower_bound is not None and lower < field.lower_bound:
        upper += float(field.lower_bound - lower)
        lower = float(field.lower_bound)
    if field.upper_bound is not None and upper > field.upper_bound:
        lower -= float(upper - field.upper_bound)
        upper = float(field.upper_bound)
    if field.lower_bound is not None:
        lower = max(lower, float(field.lower_bound))
    if field.upper_bound is not None:
        upper = min(upper, float(field.upper_bound))
    lower = min(lower, center)
    upper = max(upper, center)
    return Interval(lower, upper)


def cover_missing_actions(
    match_set: MatchSet,
    *,
    config: LearningConfig | None = None,
    creation_epoch: int = 0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[MutableRule, ...]:
    """Create empty-ledger rules only for feasible actions lacking advocates.

    The caller owns population insertion. After insertion, it should build the
    final frozen match set used for rollout credit so the new advocates are
    eligible recipients; covering never mutates an already-published record.
    """

    if not isinstance(match_set, MatchSet):
        raise TypeError("match_set must be MatchSet")
    settings = LearningConfig() if config is None else config
    if not isinstance(settings, LearningConfig):
        raise TypeError("config must be LearningConfig")
    if isinstance(creation_epoch, bool) or int(creation_epoch) < 0:
        raise ValueError("creation_epoch must be a non-negative integer")
    epoch = int(creation_epoch)
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")

    resolved_seed: int | None
    if rng is None:
        raw_seed = settings.random_seed if seed is None else seed
        if isinstance(raw_seed, bool) or int(raw_seed) < 0:
            raise ValueError("covering seed must be a non-negative integer")
        resolved_seed = int(raw_seed)
        generator = np.random.default_rng(resolved_seed)
    else:
        resolved_seed = None
        generator = rng

    vector = match_set.vector
    covered: list[MutableRule] = []
    for action in match_set.missing_actions:
        intervals: dict[str, Interval] = {}
        width_fractions: dict[str, float] = {}
        for field in vector.schema.fields:
            fraction = float(
                generator.uniform(
                    settings.covering_width_min_fraction,
                    settings.covering_width_max_fraction,
                )
            )
            width_fractions[field.name] = fraction
            intervals[field.name] = _centered_interval(
                field,
                float(vector.named[field.name]),
                fraction,
            )

        condition = RuleCondition(
            role_type=match_set.role_type,
            schema_hash=match_set.schema_hash,
            intervals=intervals,
            categories=vector.categories,
        )
        rule = MutableRule(
            condition=condition,
            action=action,
            creation_epoch=epoch,
            last_ga_epoch=epoch,
            provenance={
                "kind": "covering",
                "anchor_id": match_set.anchor_id,
                "schema_hash": match_set.schema_hash,
                "action_key": action.key,
                "covering_seed": resolved_seed,
                "width_fractions": width_fractions,
            },
        )
        if not rule.matches(
            vector,
            role_type=match_set.role_type,
            schema_hash=match_set.schema_hash,
        ):
            raise RuntimeError("a covering rule failed to match its center point")
        if rule.evolution.n != 0 or rule.deployment.n != 0:
            raise RuntimeError("covering rules must start with empty evidence")
        covered.append(rule)
    return tuple(covered)


__all__ = ["cover_missing_actions"]
