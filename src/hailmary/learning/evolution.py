"""Deterministic niche evolution for the causal rule population."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any, Literal, TypeAlias

import numpy as np

from hailmary.config import LearningConfig
from hailmary.features.schema import FeatureSchema
from hailmary.learning.conditions import Interval, RuleCondition
from hailmary.learning.population import Population
from hailmary.learning.rules import MutableRule, RuleAction
from hailmary.learning.statistics import OnlineMoments


Niche: TypeAlias = tuple[str, RuleAction]
MutationOperation: TypeAlias = Literal["widen", "narrow", "shift", "add", "remove"]
RngLike: TypeAlias = np.random.Generator | int | None

_MUTATION_OPERATIONS: tuple[MutationOperation, ...] = (
    "widen",
    "narrow",
    "shift",
    "add",
    "remove",
)

# Stored in the already-serialized, immutable MutableRule provenance so a GA
# child's discounted parent moments remain useful for evolution without ever
# masquerading as independent post-birth certification observations.
OFFSPRING_EVIDENCE_PROVENANCE_KEY = "offspring_evidence_baseline"
OFFSPRING_EVIDENCE_BASELINE_VERSION = "hailmary.offspring_evidence_baseline.v2"


def _rng(value: RngLike) -> np.random.Generator:
    if isinstance(value, np.random.Generator):
        return value
    return np.random.default_rng(0 if value is None else int(value))


def rule_niche(rule: MutableRule) -> Niche:
    """Return the exact role by physical-action niche."""

    if not isinstance(rule, MutableRule):
        raise TypeError("niche membership requires a MutableRule")
    return (rule.role_type, rule.action)


def same_niche(left: MutableRule, right: MutableRule) -> bool:
    return rule_niche(left) == rule_niche(right)


def rules_in_niche(
    rules: Iterable[MutableRule],
    niche: Niche,
) -> tuple[MutableRule, ...]:
    role_type, action = niche
    normalized = RuleAction.from_candidate(action)
    return tuple(
        sorted(
            (
                rule
                for rule in rules
                if rule.role_type == str(role_type) and rule.action == normalized
            ),
            key=lambda rule: rule.rule_id,
        )
    )


def _resolve_niche(
    *,
    niche: Niche | None,
    role_type: str | None,
    action: RuleAction | Any | None,
) -> Niche:
    if niche is not None:
        if role_type is not None or action is not None:
            raise ValueError("provide niche or role_type/action, not both")
        return (str(niche[0]), RuleAction.from_candidate(niche[1]))
    if role_type is None or action is None:
        raise ValueError("parent selection requires an exact role/action niche")
    return (str(role_type), RuleAction.from_candidate(action))


def select_parent(
    rules: Iterable[MutableRule],
    *,
    niche: Niche | None = None,
    role_type: str | None = None,
    action: RuleAction | Any | None = None,
    rng: RngLike = None,
    min_experience: int = 2,
    z: float = 1.96,
    variance_floor: float = 1.0e-6,
    exclude_rule_ids: Iterable[str] = (),
) -> MutableRule:
    """Sample reproducibly within one niche using rival-ledger LCB weights."""

    if isinstance(min_experience, bool) or int(min_experience) < 1:
        raise ValueError("min_experience must be positive")
    resolved = _resolve_niche(niche=niche, role_type=role_type, action=action)
    excluded = {str(rule_id) for rule_id in exclude_rule_ids}
    candidates = tuple(
        rule
        for rule in rules_in_niche(rules, resolved)
        if rule.evolution.n >= int(min_experience) and rule.rule_id not in excluded
    )
    if not candidates:
        raise ValueError("niche has no parent with the required rival experience")

    scores = np.asarray(
        [rule.evolution.lcb(z, variance_floor) for rule in candidates],
        dtype=np.float64,
    )
    shifted = scores - float(np.min(scores))
    if float(np.sum(shifted)) <= 1.0e-15:
        weights = np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
    else:
        epsilon = max(float(np.max(shifted)) * 1.0e-12, 1.0e-12)
        weights = shifted + epsilon
        weights /= float(np.sum(weights))
    index = int(_rng(rng).choice(len(candidates), p=weights))
    return candidates[index]


def _schema_bounds(
    schema: FeatureSchema | Mapping[str, Interval | Sequence[float | None]],
    *,
    expected_schema_hash: str,
) -> dict[str, Interval]:
    if isinstance(schema, FeatureSchema):
        if schema.schema_hash != expected_schema_hash:
            raise ValueError("feature schema hash does not match the rule condition")
        return {
            field.name: Interval(field.lower_bound, field.upper_bound)
            for field in schema.fields
        }
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be FeatureSchema or a feature-bound mapping")
    return {
        str(name): (
            Interval(value.lower_bound, value.upper_bound)
            if hasattr(value, "lower_bound") and hasattr(value, "upper_bound")
            else Interval.coerce(value)
        )
        for name, value in schema.items()
    }


def _current_values(
    current_state: Mapping[str, float] | Any | None,
) -> dict[str, float] | None:
    if current_state is None:
        return None
    named = getattr(current_state, "named", current_state)
    if not isinstance(named, Mapping):
        raise TypeError("current_state must be a feature mapping or FeatureVector")
    normalized = {str(name): float(value) for name, value in named.items()}
    if not all(math.isfinite(value) for value in normalized.values()):
        raise ValueError("current_state features must be finite")
    return normalized


def _span(
    bounds: Interval,
    interval: Interval | None,
    current: float | None,
) -> float:
    if bounds.lower is not None and bounds.upper is not None:
        return max(bounds.upper - bounds.lower, 1.0e-9)
    if (
        interval is not None
        and interval.lower is not None
        and interval.upper is not None
    ):
        return max(interval.upper - interval.lower, 1.0e-9)
    magnitude = 1.0 if current is None else max(1.0, abs(current))
    return 2.0 * magnitude


def _clip_interval(
    interval: Interval,
    bounds: Interval,
    *,
    current: float | None,
) -> Interval:
    lower = interval.lower
    upper = interval.upper
    if bounds.lower is not None:
        lower = bounds.lower if lower is None else max(lower, bounds.lower)
    if bounds.upper is not None:
        upper = bounds.upper if upper is None else min(upper, bounds.upper)

    if current is not None:
        if bounds.lower is not None and current < bounds.lower - 1.0e-12:
            raise ValueError("current feature is below its schema bound")
        if bounds.upper is not None and current > bounds.upper + 1.0e-12:
            raise ValueError("current feature is above its schema bound")
        lower = current if lower is not None and lower > current else lower
        upper = current if upper is not None and upper < current else upper

    if lower is not None and upper is not None and lower > upper:
        pivot = current
        if pivot is None:
            if bounds.upper is not None and lower > bounds.upper:
                pivot = bounds.upper
            elif bounds.lower is not None and upper < bounds.lower:
                pivot = bounds.lower
            else:
                pivot = 0.5 * (lower + upper)
        lower = upper = float(pivot)
    return Interval(lower, upper)


def mutate_condition(
    condition: RuleCondition,
    schema: FeatureSchema | Mapping[str, Interval | Sequence[float | None]],
    *,
    rng: RngLike = None,
    operation: MutationOperation | None = None,
    feature_name: str | None = None,
    current_state: Mapping[str, float] | Any | None = None,
    amount_fraction: float = 0.10,
) -> RuleCondition:
    """Mutate one predicate while respecting schema bounds and the live match."""

    if not isinstance(condition, RuleCondition):
        raise TypeError("condition must be RuleCondition")
    fraction = float(amount_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("amount_fraction must lie in (0, 1]")
    bounds_by_name = _schema_bounds(schema, expected_schema_hash=condition.schema_hash)
    current = _current_values(current_state)
    if current is not None and not condition.matches(
        current,
        role_type=condition.role_type,
        schema_hash=condition.schema_hash,
    ):
        raise ValueError("current_state must match the parent condition")

    existing_names = tuple(sorted(condition.intervals))
    add_names = tuple(
        name
        for name in sorted(bounds_by_name)
        if name not in condition.intervals and (current is None or name in current)
    )
    eligible: list[MutationOperation] = []
    if existing_names:
        eligible.extend(("widen", "narrow", "shift", "remove"))
    if add_names:
        eligible.append("add")
    if not eligible:
        return condition

    generator = _rng(rng)
    selected_operation = (
        str(generator.choice(eligible)) if operation is None else str(operation)
    )
    if selected_operation not in _MUTATION_OPERATIONS:
        raise ValueError(f"unknown mutation operation {selected_operation!r}")
    if selected_operation == "add":
        candidates = add_names
    else:
        candidates = existing_names
    if not candidates:
        raise ValueError(
            f"mutation operation {selected_operation!r} has no eligible feature"
        )
    selected_name = (
        str(generator.choice(candidates)) if feature_name is None else str(feature_name)
    )
    if selected_name not in candidates:
        raise ValueError(
            f"feature {selected_name!r} is not eligible for {selected_operation} mutation"
        )

    if selected_operation == "remove":
        result = condition.without_interval(selected_name)
    else:
        bounds = bounds_by_name[selected_name]
        value = None if current is None else current.get(selected_name)
        span = _span(bounds, condition.intervals.get(selected_name), value)
        amount = fraction * span * float(generator.uniform(0.25, 1.0))
        existing = condition.intervals.get(selected_name)

        if selected_operation == "add":
            if value is not None:
                center = value
            elif bounds.lower is not None and bounds.upper is not None:
                center = float(generator.uniform(bounds.lower, bounds.upper))
            elif bounds.lower is not None:
                center = bounds.lower + span / 2.0
            elif bounds.upper is not None:
                center = bounds.upper - span / 2.0
            else:
                center = 0.0
            candidate = Interval(center - amount, center + amount)
        elif selected_operation == "widen":
            assert existing is not None
            candidate = Interval(
                None if existing.lower is None else existing.lower - amount,
                None if existing.upper is None else existing.upper + amount,
            )
        elif selected_operation == "shift":
            assert existing is not None
            delta = float(generator.uniform(-amount, amount))
            candidate = existing.shifted(delta)
        else:
            assert selected_operation == "narrow" and existing is not None
            center = value
            if center is None:
                if existing.lower is not None and existing.upper is not None:
                    center = 0.5 * (existing.lower + existing.upper)
                elif existing.lower is not None:
                    center = existing.lower + span / 2.0
                elif existing.upper is not None:
                    center = existing.upper - span / 2.0
                else:
                    center = 0.0
            lower = (
                center - span / 2.0
                if existing.lower is None
                else min(center, existing.lower + amount)
            )
            upper = (
                center + span / 2.0
                if existing.upper is None
                else max(center, existing.upper - amount)
            )
            candidate = Interval(lower, upper)

        result = condition.with_interval(
            selected_name,
            _clip_interval(candidate, bounds, current=value),
        )

    if current is not None and not result.matches(
        current,
        role_type=condition.role_type,
        schema_hash=condition.schema_hash,
    ):
        raise AssertionError("condition mutation lost the supplied current-state match")
    return result


def crossover_conditions(
    left: RuleCondition,
    right: RuleCondition,
    *,
    rng: RngLike = None,
    current_state: Mapping[str, float] | Any | None = None,
) -> tuple[RuleCondition, RuleCondition]:
    """Exchange predicates between compatible same-schema conditions."""

    if left.role_type != right.role_type or left.schema_hash != right.schema_hash:
        raise ValueError("crossover conditions must share role and feature schema")
    current = _current_values(current_state)
    if current is not None:
        for condition in (left, right):
            if not condition.matches(
                current,
                role_type=condition.role_type,
                schema_hash=condition.schema_hash,
            ):
                raise ValueError("current_state must match both crossover parents")

    generator = _rng(rng)
    first: dict[str, Interval] = {}
    second: dict[str, Interval] = {}
    for name in sorted(set(left.intervals) | set(right.intervals)):
        left_interval = left.intervals.get(name)
        right_interval = right.intervals.get(name)
        if left_interval is not None and right_interval is not None:
            if bool(generator.integers(0, 2)):
                first[name], second[name] = left_interval, right_interval
            else:
                first[name], second[name] = right_interval, left_interval
        elif left_interval is not None:
            (first if bool(generator.integers(0, 2)) else second)[name] = left_interval
        elif right_interval is not None:
            (first if bool(generator.integers(0, 2)) else second)[name] = right_interval

    children = (
        RuleCondition(left.role_type, left.schema_hash, first, left.categories),
        RuleCondition(left.role_type, left.schema_hash, second, right.categories),
    )
    if current is not None and not all(
        child.matches(
            current,
            role_type=left.role_type,
            schema_hash=left.schema_hash,
        )
        for child in children
    ):
        raise AssertionError(
            "condition crossover lost the supplied current-state match"
        )
    return children


def discounted_moments(
    moments: OnlineMoments,
    discount: float,
    *,
    max_samples: int,
) -> OnlineMoments:
    """Discount sufficient statistics while preserving mean and sample variance."""

    factor = float(discount)
    if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
        raise ValueError("evidence discount must lie in [0, 1]")
    if isinstance(max_samples, bool) or int(max_samples) < 0:
        raise ValueError("max_samples cannot be negative")
    target_n = min(int(max_samples), int(math.floor(moments.n * factor)))
    if target_n == 0:
        return OnlineMoments()
    if target_n == 1:
        return OnlineMoments(n=1, mean=moments.mean, m2=0.0)
    return OnlineMoments(
        n=target_n,
        mean=moments.mean,
        m2=moments.sample_variance * float(target_n - 1),
    )


def _combined_ledgers(
    parents: Sequence[MutableRule],
) -> tuple[OnlineMoments, OnlineMoments]:
    evolution = OnlineMoments()
    deployment = OnlineMoments()
    for parent in parents:
        evolution.combine(parent.evolution)
        deployment.combine(parent.deployment)
    return evolution, deployment


def make_offspring(
    parents: Sequence[MutableRule],
    condition: RuleCondition,
    *,
    epoch: int,
    operator: str,
    evidence_discount: float = 0.50,
    action_certification_min_samples: int = 30,
    veto_certification_min_samples: int = 60,
    provenance: Mapping[str, Any] | None = None,
) -> MutableRule:
    """Create a detached, uncertified child with audited parent provenance."""

    if isinstance(parents, (str, bytes)) or not parents:
        raise ValueError("offspring requires at least one parent")
    normalized = tuple(parents)
    if any(not isinstance(parent, MutableRule) for parent in normalized):
        raise TypeError("offspring parents must be MutableRule instances")
    first = normalized[0]
    if any(not same_niche(first, parent) for parent in normalized[1:]):
        raise ValueError("offspring parents must occupy the same exact niche")
    if any(parent.schema_hash != first.schema_hash for parent in normalized[1:]):
        raise ValueError("offspring parents must share a feature schema")
    if (
        condition.role_type != first.role_type
        or condition.schema_hash != first.schema_hash
    ):
        raise ValueError("offspring condition must preserve parent role and schema")
    if (
        isinstance(action_certification_min_samples, bool)
        or isinstance(veto_certification_min_samples, bool)
        or action_certification_min_samples < 1
        or veto_certification_min_samples < 1
    ):
        raise ValueError("certification thresholds must be positive")

    combined_evolution, combined_deployment = _combined_ledgers(normalized)
    child_evolution = discounted_moments(
        combined_evolution,
        evidence_discount,
        max_samples=veto_certification_min_samples - 1,
    )
    child_deployment = discounted_moments(
        combined_deployment,
        evidence_discount,
        max_samples=action_certification_min_samples - 1,
    )
    parent_ids = tuple(sorted({parent.rule_id for parent in normalized}))
    details = {
        "operator": str(operator),
        "evidence_discount": float(evidence_discount),
        "birth_epoch": int(epoch),
    }
    if provenance:
        if OFFSPRING_EVIDENCE_PROVENANCE_KEY in provenance:
            raise ValueError(
                f"offspring provenance cannot override reserved key "
                f"{OFFSPRING_EVIDENCE_PROVENANCE_KEY!r}"
            )
        details.update(dict(provenance))
    details[OFFSPRING_EVIDENCE_PROVENANCE_KEY] = {
        "artifact_version": OFFSPRING_EVIDENCE_BASELINE_VERSION,
        "evolution": child_evolution.to_dict(),
        "deployment": child_deployment.to_dict(),
    }
    return MutableRule(
        condition=RuleCondition.from_dict(condition.to_dict()),
        action=RuleAction.from_dict(first.action.to_dict()),
        evolution=child_evolution,
        deployment=child_deployment,
        numerosity=1,
        creation_epoch=int(epoch),
        last_ga_epoch=int(epoch),
        parent_ids=parent_ids,
        provenance=details,
    )


def mutate_rule(
    parent: MutableRule,
    schema: FeatureSchema | Mapping[str, Interval | Sequence[float | None]],
    *,
    epoch: int,
    rng: RngLike = None,
    operation: MutationOperation | None = None,
    feature_name: str | None = None,
    current_state: Mapping[str, float] | Any | None = None,
    amount_fraction: float = 0.10,
    evidence_discount: float = 0.50,
    action_certification_min_samples: int = 30,
    veto_certification_min_samples: int = 60,
) -> MutableRule:
    condition = mutate_condition(
        parent.condition,
        schema,
        rng=rng,
        operation=operation,
        feature_name=feature_name,
        current_state=current_state,
        amount_fraction=amount_fraction,
    )
    return make_offspring(
        (parent,),
        condition,
        epoch=epoch,
        operator="mutation",
        evidence_discount=evidence_discount,
        action_certification_min_samples=action_certification_min_samples,
        veto_certification_min_samples=veto_certification_min_samples,
        provenance={
            "mutation_operation": operation or "sampled",
            "mutation_feature": feature_name,
        },
    )


def _mutate_newborn(
    child: MutableRule,
    schema: FeatureSchema | Mapping[str, Interval | Sequence[float | None]],
    *,
    rng: RngLike,
    current_state: Mapping[str, float] | Any | None,
) -> MutableRule:
    """Mutate one GA newborn without creating a second generation.

    Crossover/reproduction already applies the configured evidence discount and
    records the population parents. Treating that newborn as the parent of a
    subsequent ``mutate_rule`` call would discount the same evidence again and
    replace the real parent IDs with the transient newborn ID. Mutation here is
    therefore another operator in the same birth: only the condition and audit
    provenance change, while ledgers and original parent IDs are copied exactly.
    """

    condition = mutate_condition(
        child.condition,
        schema,
        rng=rng,
        current_state=current_state,
    )
    provenance = dict(child.provenance)
    birth_operator = str(provenance.get("operator", "offspring"))
    provenance.update(
        {
            "operator": "mutation",
            "birth_operator": birth_operator,
            "mutation_operation": "sampled",
            "mutation_feature": None,
        }
    )
    return MutableRule(
        condition=condition,
        action=RuleAction.from_dict(child.action.to_dict()),
        evolution=child.evolution.copy(),
        deployment=child.deployment.copy(),
        numerosity=child.numerosity,
        creation_epoch=child.creation_epoch,
        last_ga_epoch=child.last_ga_epoch,
        parent_ids=child.parent_ids,
        provenance=provenance,
    )


def crossover_rules(
    left: MutableRule,
    right: MutableRule,
    *,
    epoch: int,
    rng: RngLike = None,
    current_state: Mapping[str, float] | Any | None = None,
    evidence_discount: float = 0.50,
    action_certification_min_samples: int = 30,
    veto_certification_min_samples: int = 60,
) -> tuple[MutableRule, MutableRule]:
    if not same_niche(left, right):
        raise ValueError("crossover is restricted to one exact role/action niche")
    if left.schema_hash != right.schema_hash:
        raise ValueError("crossover parents must share a feature schema")
    conditions = crossover_conditions(
        left.condition,
        right.condition,
        rng=rng,
        current_state=current_state,
    )
    parents = (left, right)
    first_child = make_offspring(
        parents,
        conditions[0],
        epoch=epoch,
        operator="crossover",
        evidence_discount=evidence_discount,
        action_certification_min_samples=action_certification_min_samples,
        veto_certification_min_samples=veto_certification_min_samples,
        provenance={"child_index": 0},
    )
    second_child = make_offspring(
        parents,
        conditions[1],
        epoch=epoch,
        operator="crossover",
        evidence_discount=evidence_discount,
        action_certification_min_samples=action_certification_min_samples,
        veto_certification_min_samples=veto_certification_min_samples,
        provenance={"child_index": 1},
    )
    return first_child, second_child


def deletion_score(
    rule: MutableRule,
    *,
    epoch: int,
    z: float = 1.96,
    variance_floor: float = 1.0e-6,
    weakness_weight: float = 1.0,
    age_weight: float = 0.05,
    numerosity_weight: float = 0.25,
) -> float:
    """Higher scores indicate greater deletion pressure."""

    age = max(0, int(epoch) - rule.creation_epoch)
    weakness = -rule.evolution.lcb(z, variance_floor)
    return float(
        weakness_weight * weakness
        + age_weight * math.log1p(age)
        + numerosity_weight * math.log1p(rule.numerosity)
    )


def select_deletion_candidate(
    rules: Iterable[MutableRule],
    *,
    epoch: int,
    young_protection_epochs: int = 100,
    z: float = 1.96,
    variance_floor: float = 1.0e-6,
) -> MutableRule:
    candidates = tuple(rules)
    if not candidates:
        raise ValueError("cannot select deletion from an empty population")
    if young_protection_epochs < 0:
        raise ValueError("young_protection_epochs cannot be negative")
    mature = tuple(
        rule
        for rule in candidates
        if int(epoch) - rule.creation_epoch >= young_protection_epochs
    )
    pool = mature or candidates
    return min(
        pool,
        key=lambda rule: (
            -deletion_score(
                rule,
                epoch=epoch,
                z=z,
                variance_floor=variance_floor,
            ),
            rule.rule_id,
        ),
    )


def enforce_population_bound(
    population: Population,
    *,
    epoch: int,
    max_size: int | None = None,
    young_protection_epochs: int = 100,
    z: float = 1.96,
    variance_floor: float = 1.0e-6,
) -> tuple[str, ...]:
    """Delete classifier numerosities until the requested bound is satisfied."""

    if not isinstance(population, Population):
        raise TypeError("population must be Population")
    limit = population.max_size if max_size is None else int(max_size)
    if limit < 1:
        raise ValueError("population bound must be positive")
    removed: list[str] = []
    while population.total_numerosity > limit:
        candidate = select_deletion_candidate(
            population.rules,
            epoch=epoch,
            young_protection_epochs=young_protection_epochs,
            z=z,
            variance_floor=variance_floor,
        )
        removed.append(candidate.rule_id)
        population.remove(candidate.rule_id)
    return tuple(removed)


def insert_offspring(
    population: Population,
    offspring: MutableRule,
    *,
    epoch: int,
    max_size: int | None = None,
    young_protection_epochs: int = 100,
    z: float = 1.96,
    variance_floor: float = 1.0e-6,
) -> MutableRule:
    """Insert one GA birth without treating inherited evidence as observations.

    Population.add deliberately combines sufficient statistics when callers
    import independent, equivalent classifiers. A GA child is different: its
    ledgers are discounted copies of evidence already owned by the population.
    Combining them into an equivalent macro-classifier would count the same
    observations again on every reproduction. Equivalent offspring therefore
    increase only numerosity and the last-GA timestamp.

    The returned value is always the detached birth, even when its numerosity is
    represented by an existing macro-classifier. This preserves exact parent
    provenance in GA audit events.
    """

    if max_size is not None and (isinstance(max_size, bool) or int(max_size) < 1):
        raise ValueError("population bound must be positive")
    limit = (
        population.max_size
        if max_size is None
        else min(population.max_size, int(max_size))
    )
    if offspring.numerosity > limit:
        raise OverflowError("offspring numerosity exceeds the population bound")

    equivalent = next(
        (
            rule
            for rule in population.rules
            if rule.condition == offspring.condition and rule.action == offspring.action
        ),
        None,
    )
    if equivalent is offspring:
        return offspring
    if equivalent is not None:
        equivalent.numerosity += offspring.numerosity
        equivalent.last_ga_epoch = max(
            equivalent.last_ga_epoch,
            offspring.last_ga_epoch,
            int(epoch),
        )
        enforce_population_bound(
            population,
            epoch=epoch,
            max_size=limit,
            young_protection_epochs=young_protection_epochs,
            z=z,
            variance_floor=variance_floor,
        )
        return offspring

    while population.total_numerosity + offspring.numerosity > limit:
        candidate = select_deletion_candidate(
            population.rules,
            epoch=epoch,
            young_protection_epochs=young_protection_epochs,
            z=z,
            variance_floor=variance_floor,
        )
        population.remove(candidate.rule_id)
    inserted = population.add(offspring)
    enforce_population_bound(
        population,
        epoch=epoch,
        max_size=limit,
        young_protection_epochs=young_protection_epochs,
        z=z,
        variance_floor=variance_floor,
    )
    return inserted


def can_subsume(
    general: MutableRule,
    specific: MutableRule,
    *,
    min_experience: int,
    material_lcb_tolerance: float = 0.0,
    z: float = 1.96,
    variance_floor: float = 1.0e-6,
) -> bool:
    """Check conservative same-niche interval subsumption."""

    if isinstance(min_experience, bool) or int(min_experience) < 1:
        raise ValueError("min_experience must be positive")
    tolerance = float(material_lcb_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("material_lcb_tolerance must be finite and non-negative")
    if general.rule_id == specific.rule_id or not same_niche(general, specific):
        return False
    if general.evolution.n < int(min_experience):
        return False
    if not general.condition.contains(specific.condition):
        return False
    general_lcb = general.evolution.lcb(z, variance_floor)
    specific_lcb = specific.evolution.lcb(z, variance_floor)
    return general_lcb >= specific_lcb - tolerance


def subsume_rule(
    population: Population,
    general: MutableRule,
    specific: MutableRule,
    *,
    min_experience: int,
    material_lcb_tolerance: float = 0.0,
    z: float = 1.96,
    variance_floor: float = 1.0e-6,
) -> bool:
    if not isinstance(population, Population):
        raise TypeError("population must be Population")
    if not isinstance(general, MutableRule) or not isinstance(specific, MutableRule):
        raise TypeError("subsumption candidates must be MutableRule instances")
    try:
        stored_general = population.get(general.rule_id)
        stored_specific = population.get(specific.rule_id)
    except KeyError as exc:
        raise ValueError(
            "subsumption candidates must both be live in population"
        ) from exc
    if stored_general is not general or stored_specific is not specific:
        raise ValueError(
            "subsumption candidates must be the exact live population objects"
        )
    if not can_subsume(
        stored_general,
        stored_specific,
        min_experience=min_experience,
        material_lcb_tolerance=material_lcb_tolerance,
        z=z,
        variance_floor=variance_floor,
    ):
        return False
    absorbed = stored_specific.numerosity
    population.remove(stored_specific.rule_id, numerosity=absorbed)
    stored_general.numerosity += absorbed
    return True


def evolve_niche(
    population: Population,
    *,
    role_type: str,
    action: RuleAction | Any,
    schema: FeatureSchema | Mapping[str, Interval | Sequence[float | None]],
    epoch: int,
    config: LearningConfig | None = None,
    rng: RngLike = None,
    current_state: Mapping[str, float] | Any | None = None,
    eligible_rule_ids: Iterable[str] | None = None,
    insert: bool = True,
) -> tuple[MutableRule, ...]:
    """Generate one deterministic action-set GA step and optionally insert it.

    ``insert=False`` freezes parent selection and inherited evidence without
    mutating the population. Coordinators with multiple action niches can
    therefore prepare every birth before any global deletion is allowed to
    invalidate another niche's frozen parent set.
    """

    settings = LearningConfig() if config is None else config
    if type(insert) is not bool:
        raise TypeError("insert must be a boolean")
    generator = _rng(settings.random_seed if rng is None else rng)
    niche = (str(role_type), RuleAction.from_candidate(action))
    if eligible_rule_ids is None:
        parent_pool = population.rules
    else:
        eligible_ids = {str(rule_id) for rule_id in eligible_rule_ids}
        if not eligible_ids or "" in eligible_ids:
            raise ValueError("eligible_rule_ids must contain non-empty rule IDs")
        parent_pool = tuple(
            rule for rule in population.rules if rule.rule_id in eligible_ids
        )
        missing = eligible_ids - {rule.rule_id for rule in parent_pool}
        if missing:
            raise KeyError(
                f"eligible GA parent IDs are absent from population: {sorted(missing)!r}"
            )

    first = select_parent(
        parent_pool,
        niche=niche,
        rng=generator,
        min_experience=settings.ga_min_experience,
        z=settings.contender_lcb_z,
        variance_floor=settings.variance_floor,
    )
    remaining = tuple(
        rule
        for rule in rules_in_niche(parent_pool, niche)
        if rule.rule_id != first.rule_id
        and rule.evolution.n >= settings.ga_min_experience
    )
    second = (
        select_parent(
            remaining,
            niche=niche,
            rng=generator,
            min_experience=settings.ga_min_experience,
            z=settings.contender_lcb_z,
            variance_floor=settings.variance_floor,
        )
        if remaining
        else first
    )

    if bool(generator.random() < settings.crossover_probability):
        children = list(
            crossover_rules(
                first,
                second,
                epoch=epoch,
                rng=generator,
                current_state=current_state,
                evidence_discount=settings.offspring_evidence_discount,
                action_certification_min_samples=settings.action_min_noop_samples,
                veto_certification_min_samples=settings.veto_min_rival_samples,
            )
        )
    else:
        children = [
            make_offspring(
                (first,),
                first.condition,
                epoch=epoch,
                operator="reproduction",
                evidence_discount=settings.offspring_evidence_discount,
                action_certification_min_samples=settings.action_min_noop_samples,
                veto_certification_min_samples=settings.veto_min_rival_samples,
            )
        ]

    for index, child in enumerate(tuple(children)):
        if bool(generator.random() < settings.mutation_probability):
            children[index] = _mutate_newborn(
                child,
                schema,
                rng=generator,
                current_state=current_state,
            )

    if not insert:
        return tuple(children)

    inserted: list[MutableRule] = []
    for child in children:
        inserted.append(
            insert_offspring(
                population,
                child,
                epoch=epoch,
                max_size=settings.population_limit,
                young_protection_epochs=settings.young_rule_protection_epochs,
                z=settings.contender_lcb_z,
                variance_floor=settings.variance_floor,
            )
        )
    return tuple(inserted)


__all__ = [
    "MutationOperation",
    "Niche",
    "OFFSPRING_EVIDENCE_BASELINE_VERSION",
    "OFFSPRING_EVIDENCE_PROVENANCE_KEY",
    "can_subsume",
    "crossover_conditions",
    "crossover_rules",
    "deletion_score",
    "discounted_moments",
    "enforce_population_bound",
    "evolve_niche",
    "insert_offspring",
    "make_offspring",
    "mutate_condition",
    "mutate_rule",
    "rule_niche",
    "rules_in_niche",
    "same_niche",
    "select_deletion_candidate",
    "select_parent",
    "subsume_rule",
]
