from __future__ import annotations

import pytest

from hailmary.actions.models import ActionLever
from hailmary.config import LearningConfig
from hailmary.features.schema import FeatureField, FeatureSchema
from hailmary.learning.certification import (
    CertificationThresholds,
    certify_rule,
    independent_evidence_moments,
)
from hailmary.learning.conditions import Interval, RuleCondition
from hailmary.learning.evolution import (
    can_subsume,
    crossover_rules,
    enforce_population_bound,
    evolve_niche,
    insert_offspring,
    make_offspring,
    mutate_condition,
    rule_niche,
    same_niche,
    select_deletion_candidate,
    select_parent,
    subsume_rule,
)
from hailmary.learning.population import Population
from hailmary.learning.rules import MutableRule, RuleAction
from hailmary.learning.statistics import OnlineMoments


SCHEMA = FeatureSchema(
    "test.learning.evolution.v1",
    (
        FeatureField("x", lower_bound=0.0, upper_bound=10.0),
        FeatureField("y", lower_bound=0.0, upper_bound=1.0),
    ),
)
SPEED = RuleAction(ActionLever.SPEED, "light")
STRETCH = RuleAction(ActionLever.PATH_STRETCH, "oracle_short_medium_long")


def _moments(n: int, mean: float, variance: float = 0.01) -> OnlineMoments:
    return OnlineMoments(
        n=n,
        mean=mean,
        m2=0.0 if n < 2 else variance * (n - 1),
    )


def _rule(
    label: str,
    *,
    interval: tuple[float, float] = (2.0, 8.0),
    action: RuleAction = SPEED,
    role_type: str = "leader_follower",
    evolution_n: int = 20,
    evolution_mean: float = 1.0,
    deployment_n: int = 20,
    deployment_mean: float = 1.0,
    numerosity: int = 1,
    creation_epoch: int = 0,
    schema_hash: str = SCHEMA.schema_hash,
) -> MutableRule:
    return MutableRule(
        condition=RuleCondition(
            role_type,
            schema_hash,
            {"x": Interval(*interval)},
        ),
        action=action,
        evolution=_moments(evolution_n, evolution_mean),
        deployment=_moments(deployment_n, deployment_mean),
        numerosity=numerosity,
        creation_epoch=creation_epoch,
        provenance={"label": label},
    )


def test_niche_is_exact_role_by_action() -> None:
    speed = _rule("speed")
    stretch = _rule("stretch", action=STRETCH)
    other_role = _rule("other-role", role_type="aircraft_resource")

    assert rule_niche(speed) == ("leader_follower", SPEED)
    assert not same_niche(speed, stretch)
    assert not same_niche(speed, other_role)
    assert same_niche(speed, _rule("same"))


def test_parent_selection_is_seeded_lcb_driven_and_experience_gated() -> None:
    strong = _rule("strong", evolution_mean=4.0)
    medium = _rule("medium", interval=(1.0, 7.0), evolution_mean=1.0)
    inexperienced = _rule(
        "inexperienced",
        interval=(3.0, 9.0),
        evolution_n=1,
        evolution_mean=100.0,
    )
    wrong_action = _rule("wrong", action=STRETCH, evolution_mean=100.0)
    rules = (strong, medium, inexperienced, wrong_action)

    first = select_parent(
        rules,
        niche=("leader_follower", SPEED),
        rng=123,
        min_experience=2,
    )
    second = select_parent(
        reversed(rules),
        role_type="leader_follower",
        action=SPEED,
        rng=123,
        min_experience=2,
    )

    assert first.rule_id == second.rule_id
    assert first in (strong, medium)
    assert first is not inexperienced
    assert first.action == SPEED

    with pytest.raises(ValueError, match="required rival experience"):
        select_parent(
            (inexperienced,), niche=rule_niche(inexperienced), min_experience=2
        )


@pytest.mark.parametrize(
    ("operation", "feature_name"),
    [
        ("widen", "x"),
        ("narrow", "x"),
        ("shift", "x"),
        ("add", "y"),
        ("remove", "x"),
    ],
)
def test_condition_mutations_clip_and_preserve_current_match(
    operation: str,
    feature_name: str,
) -> None:
    parent = RuleCondition(
        "leader_follower",
        SCHEMA.schema_hash,
        {"x": Interval(2.0, 8.0)},
    )
    current = {"x": 5.0, "y": 0.5}

    mutated = mutate_condition(
        parent,
        SCHEMA,
        operation=operation,
        feature_name=feature_name,
        rng=91,
        current_state=current,
        amount_fraction=0.40,
    )

    assert mutated.matches(
        current,
        role_type="leader_follower",
        schema_hash=SCHEMA.schema_hash,
    )
    for name, interval in mutated.intervals.items():
        field = next(item for item in SCHEMA.fields if item.name == name)
        assert interval.lower is None or interval.lower >= field.lower_bound
        assert interval.upper is None or interval.upper <= field.upper_bound
    if operation == "add":
        assert "y" in mutated.intervals
    if operation == "remove":
        assert "x" not in mutated.intervals


def test_crossover_stays_in_niche_preserves_action_and_current_match() -> None:
    left = _rule("left", interval=(0.0, 6.0), evolution_mean=2.0)
    left.condition = left.condition.with_interval("y", Interval(0.0, 0.7))
    right = _rule("right", interval=(4.0, 10.0), evolution_mean=3.0)
    current = {"x": 5.0, "y": 0.5}

    children = crossover_rules(
        left,
        right,
        epoch=40,
        rng=19,
        current_state=current,
    )

    assert len(children) == 2
    assert all(child.action == SPEED for child in children)
    assert all(
        child.parent_ids == tuple(sorted((left.rule_id, right.rule_id)))
        for child in children
    )
    assert all(
        child.condition.matches(
            current,
            role_type="leader_follower",
            schema_hash=SCHEMA.schema_hash,
        )
        for child in children
    )

    with pytest.raises(ValueError, match="exact role/action niche"):
        crossover_rules(left, _rule("stretch", action=STRETCH), epoch=40)


def test_offspring_evidence_is_discounted_capped_and_detached() -> None:
    left = _rule(
        "left",
        evolution_n=100,
        evolution_mean=2.0,
        deployment_n=80,
        deployment_mean=1.5,
    )
    right = _rule(
        "right",
        interval=(1.0, 9.0),
        evolution_n=90,
        evolution_mean=1.0,
        deployment_n=70,
        deployment_mean=0.5,
    )

    child = make_offspring(
        (left, right),
        left.condition,
        epoch=90,
        operator="test-crossover",
        evidence_discount=0.75,
        action_certification_min_samples=30,
        veto_certification_min_samples=60,
    )

    assert child.evolution.n == 59
    assert child.deployment.n == 29
    assert child.parent_ids == tuple(sorted((left.rule_id, right.rule_id)))
    assert child.provenance["operator"] == "test-crossover"
    assert certify_rule(child, CertificationThresholds()) is None

    original_parent_n = left.evolution.n
    child.evolution.update(99.0)
    assert left.evolution.n == original_parent_n


def test_inherited_threshold_minus_one_plus_one_new_sample_cannot_certify() -> None:
    parent = _rule(
        "parent",
        evolution_n=200,
        evolution_mean=3.0,
        deployment_n=200,
        deployment_mean=3.0,
    )
    thresholds = CertificationThresholds(
        action_min_noop_samples=30,
        veto_min_rival_samples=60,
    )
    action_child = make_offspring(
        (parent,),
        parent.condition,
        epoch=90,
        operator="reproduction",
        evidence_discount=1.0,
        action_certification_min_samples=30,
        veto_certification_min_samples=60,
    )
    assert action_child.deployment.n == 29
    action_child.deployment.update(3.0)
    assert action_child.deployment.n == 30
    assert certify_rule(action_child, thresholds) is None

    noop_parent = _rule(
        "noop-parent",
        action=RuleAction.no_op(),
        evolution_n=200,
        evolution_mean=3.0,
        deployment_n=0,
        deployment_mean=0.0,
    )
    noop_child = make_offspring(
        (noop_parent,),
        noop_parent.condition,
        epoch=90,
        operator="reproduction",
        evidence_discount=1.0,
        action_certification_min_samples=30,
        veto_certification_min_samples=60,
    )
    assert noop_child.evolution.n == 59
    noop_child.evolution.update(3.0)
    assert noop_child.evolution.n == 60
    assert certify_rule(noop_child, thresholds) is None


def test_population_rejects_merges_that_corrupt_offspring_evidence_baselines() -> None:
    parent = _rule(
        "merge-parent",
        evolution_n=200,
        evolution_mean=3.0,
        deployment_n=200,
        deployment_mean=3.0,
    )
    thresholds = CertificationThresholds(
        action_min_noop_samples=3,
        veto_min_rival_samples=3,
    )
    child = make_offspring(
        (parent,),
        parent.condition,
        epoch=90,
        operator="reproduction",
        evidence_discount=1.0,
        action_certification_min_samples=3,
        veto_certification_min_samples=3,
    )
    child.deployment.extend((3.0,) * (thresholds.action_min_noop_samples - 1))
    population = Population((child,), max_size=10)

    payload = population.to_dict()
    restored = Population.from_dict(payload)
    stored = restored.get(child.rule_id)
    independent = independent_evidence_moments(stored, "deployment")
    assert restored.to_dict() == payload
    assert stored.deployment.n == 4
    assert independent.n == thresholds.action_min_noop_samples - 1
    assert independent.mean == pytest.approx(3.0)
    assert certify_rule(stored, thresholds) is None

    preexisting = _rule(
        "preexisting-equivalent",
        evolution_n=3,
        evolution_mean=3.0,
        deployment_n=3,
        deployment_mean=3.0,
    )
    before = restored.to_dict()
    with pytest.raises(ValueError, match="post-birth evidence provenance"):
        restored.add(preexisting)
    assert restored.to_dict() == before

    base_target = _rule(
        "base-target",
        evolution_n=3,
        evolution_mean=3.0,
        deployment_n=3,
        deployment_mean=3.0,
    )
    reverse = Population((base_target,), max_size=10)
    with pytest.raises(ValueError, match="post-birth evidence provenance"):
        reverse.add(MutableRule.from_dict(stored.to_dict()))


def test_offspring_requires_full_independent_post_birth_certification_evidence() -> (
    None
):
    parent = _rule(
        "parent",
        evolution_n=200,
        evolution_mean=3.0,
        deployment_n=200,
        deployment_mean=3.0,
    )
    thresholds = CertificationThresholds(
        action_min_noop_samples=3,
        veto_min_rival_samples=3,
    )
    child = make_offspring(
        (parent,),
        parent.condition,
        epoch=90,
        operator="reproduction",
        evidence_discount=1.0,
        action_certification_min_samples=3,
        veto_certification_min_samples=3,
    )

    child.deployment.extend((3.0, 3.0))
    assert certify_rule(child, thresholds) is None
    child.deployment.update(3.0)
    assert certify_rule(child, thresholds) is not None

    restored = MutableRule.from_dict(child.to_dict())
    assert certify_rule(restored, thresholds) is not None


def test_offspring_certification_uses_only_post_birth_observations() -> None:
    parent = _rule(
        "strong-parent",
        evolution_n=200,
        evolution_mean=100.0,
        deployment_n=200,
        deployment_mean=100.0,
    )
    thresholds = CertificationThresholds(
        action_min_noop_samples=30,
        veto_min_rival_samples=60,
    )
    child = make_offspring(
        (parent,),
        parent.condition,
        epoch=90,
        operator="reproduction",
        evidence_discount=1.0,
        action_certification_min_samples=30,
        veto_certification_min_samples=60,
    )
    child.deployment.extend((-1.0,) * 30)
    child.evolution.extend((-2.0, -2.0))

    assert child.deployment.n == 59
    assert (
        child.deployment.lcb(
            thresholds.action_lcb_z,
            thresholds.variance_floor,
        )
        > 0.0
    )
    deployment = independent_evidence_moments(child, "deployment")
    rival = independent_evidence_moments(child, "evolution")
    assert deployment.n == 30
    assert deployment.mean == pytest.approx(-1.0)
    assert deployment.m2 == pytest.approx(0.0, abs=1.0e-9)
    assert rival.n == 2
    assert rival.mean == pytest.approx(-2.0)
    assert certify_rule(child, thresholds) is None

    restored = MutableRule.from_dict(child.to_dict())
    assert independent_evidence_moments(restored, "deployment") == deployment
    assert independent_evidence_moments(restored, "evolution") == rival


def test_deletion_is_weak_old_numerosity_aware_and_protects_young_rules() -> None:
    young_weak = _rule(
        "young",
        interval=(0.0, 2.0),
        evolution_mean=-100.0,
        creation_epoch=180,
    )
    mature_weak = _rule(
        "mature-weak",
        interval=(2.0, 4.0),
        evolution_mean=-4.0,
        creation_epoch=0,
    )
    mature_strong = _rule(
        "mature-strong",
        interval=(4.0, 6.0),
        evolution_mean=4.0,
        creation_epoch=0,
    )

    selected = select_deletion_candidate(
        (young_weak, mature_weak, mature_strong),
        epoch=200,
        young_protection_epochs=50,
    )
    assert selected is mature_weak

    low_numerosity = _rule(
        "low-num",
        interval=(6.0, 8.0),
        evolution_mean=0.0,
        numerosity=1,
    )
    high_numerosity = _rule(
        "high-num",
        interval=(8.0, 10.0),
        evolution_mean=0.0,
        numerosity=3,
    )
    assert (
        select_deletion_candidate(
            (low_numerosity, high_numerosity),
            epoch=200,
            young_protection_epochs=0,
        )
        is high_numerosity
    )


def test_insertion_and_explicit_enforcement_keep_population_bounded() -> None:
    first = _rule(
        "first",
        interval=(0.0, 2.0),
        evolution_mean=-2.0,
        numerosity=2,
    )
    second = _rule(
        "second",
        interval=(4.0, 6.0),
        evolution_mean=2.0,
        numerosity=2,
    )
    population = Population((first, second), max_size=4)
    offspring = _rule("offspring", interval=(8.0, 10.0), creation_epoch=100)

    insert_offspring(
        population,
        offspring,
        epoch=100,
        young_protection_epochs=0,
    )
    assert population.total_numerosity == 4
    assert population.total_numerosity <= population.max_size

    population.get(second.rule_id).numerosity += 2
    removed = enforce_population_bound(
        population,
        epoch=100,
        young_protection_epochs=0,
    )
    assert removed
    assert population.total_numerosity == population.max_size


def test_equivalent_ga_births_do_not_recombine_inherited_evidence() -> None:
    parent = _rule(
        "parent",
        evolution_n=20,
        evolution_mean=2.0,
        deployment_n=20,
        deployment_mean=3.0,
    )
    population = Population((parent,), max_size=100)
    original_evolution = parent.evolution.to_dict()
    original_deployment = parent.deployment.to_dict()
    thresholds = CertificationThresholds(
        action_min_noop_samples=30,
        veto_min_rival_samples=30,
    )

    births = []
    for epoch in range(100, 105):
        child = make_offspring(
            (parent,),
            parent.condition,
            epoch=epoch,
            operator="reproduction",
            evidence_discount=0.5,
            action_certification_min_samples=30,
            veto_certification_min_samples=30,
        )
        births.append(insert_offspring(population, child, epoch=epoch))

    stored = population.get(parent.rule_id)
    assert population.macro_size == 1
    assert stored.numerosity == 6
    assert stored.last_ga_epoch == 104
    assert stored.evolution.to_dict() == original_evolution
    assert stored.deployment.to_dict() == original_deployment
    assert all(birth.parent_ids == (parent.rule_id,) for birth in births)
    assert certify_rule(stored, thresholds) is None


def test_subsumption_requires_same_niche_containment_experience_and_lcb() -> None:
    general = _rule(
        "general",
        interval=(1.0, 9.0),
        evolution_n=40,
        evolution_mean=1.2,
        numerosity=2,
    )
    specific = _rule(
        "specific",
        interval=(3.0, 5.0),
        evolution_n=20,
        evolution_mean=1.0,
        numerosity=3,
    )
    population = Population((general, specific), max_size=10)
    before = population.total_numerosity

    assert can_subsume(general, specific, min_experience=30)
    assert subsume_rule(population, general, specific, min_experience=30)
    assert population.total_numerosity == before
    assert population.get(general.rule_id).numerosity == 5
    with pytest.raises(KeyError):
        population.get(specific.rule_id)

    different_action = _rule("different-action", action=STRETCH)
    assert not can_subsume(general, different_action, min_experience=30)
    inexperienced = _rule(
        "inexperienced-general",
        interval=(0.0, 10.0),
        evolution_n=2,
        evolution_mean=10.0,
    )
    assert not can_subsume(inexperienced, specific, min_experience=30)
    worse = _rule(
        "worse-general",
        interval=(0.0, 10.0),
        evolution_n=40,
        evolution_mean=-5.0,
    )
    assert not can_subsume(worse, specific, min_experience=30)


def test_subsumption_rejects_stale_or_fabricated_clones_with_live_ids() -> None:
    general = _rule(
        "general",
        interval=(1.0, 9.0),
        evolution_n=40,
        evolution_mean=2.0,
    )
    specific = _rule(
        "specific",
        interval=(3.0, 5.0),
        evolution_n=20,
        evolution_mean=1.0,
    )
    population = Population((general, specific), max_size=10)
    before = population.to_dict()

    stale_general = MutableRule.from_dict(general.to_dict())
    with pytest.raises(ValueError, match="exact live population objects"):
        subsume_rule(population, stale_general, specific, min_experience=30)
    assert population.to_dict() == before

    fabricated_specific = _rule(
        "fabricated-specific",
        interval=(4.0, 4.5),
        evolution_n=20,
        evolution_mean=-100.0,
    )
    fabricated_specific.rule_id = specific.rule_id
    with pytest.raises(ValueError, match="exact live population objects"):
        subsume_rule(population, general, fabricated_specific, min_experience=30)
    assert population.to_dict() == before


def test_evolve_niche_is_seeded_and_enforces_configured_limit() -> None:
    parents = (
        _rule("parent-a", interval=(0.0, 7.0), evolution_mean=2.0),
        _rule("parent-b", interval=(3.0, 10.0), evolution_mean=1.0),
    )
    population = Population(parents, max_size=4)
    config = LearningConfig(
        population_limit=4,
        crossover_probability=1.0,
        mutation_probability=1.0,
        random_seed=47,
    )

    children = evolve_niche(
        population,
        role_type="leader_follower",
        action=SPEED,
        schema=SCHEMA,
        epoch=200,
        config=config,
        current_state={"x": 5.0, "y": 0.5},
    )

    assert children
    assert all(child.action == SPEED for child in children)
    assert population.total_numerosity <= 4


def test_evolve_niche_mutation_is_one_birth_with_one_evidence_discount() -> None:
    parents = (
        _rule("parent-a", interval=(0.0, 7.0)),
        _rule("parent-b", interval=(3.0, 10.0)),
    )
    population = Population(parents, max_size=100)
    config = LearningConfig(
        population_limit=100,
        crossover_probability=1.0,
        mutation_probability=1.0,
        offspring_evidence_discount=0.5,
        random_seed=47,
    )

    children = evolve_niche(
        population,
        role_type="leader_follower",
        action=SPEED,
        schema=SCHEMA,
        epoch=200,
        config=config,
        rng=47,
        current_state={"x": 5.0, "y": 0.5},
    )

    original_parent_ids = tuple(sorted(parent.rule_id for parent in parents))
    assert len(children) == 2
    assert all(child.evolution.n == 20 for child in children)
    assert all(child.deployment.n == 20 for child in children)
    assert all(child.parent_ids == original_parent_ids for child in children)
    assert all(child.provenance["operator"] == "mutation" for child in children)
    assert all(child.provenance["birth_operator"] == "crossover" for child in children)
    assert {child.provenance["child_index"] for child in children} == {0, 1}


def test_evolve_niche_restricts_both_parents_to_frozen_action_set() -> None:
    matching = _rule(
        "matching",
        interval=(0.0, 4.0),
        evolution_mean=0.5,
    )
    disjoint = _rule(
        "disjoint-high-fitness",
        interval=(6.0, 10.0),
        evolution_mean=100.0,
    )
    population = Population((matching, disjoint), max_size=100)
    config = LearningConfig(
        population_limit=100,
        ga_min_experience=2,
        crossover_probability=1.0,
        mutation_probability=0.0,
        random_seed=31,
    )

    children = evolve_niche(
        population,
        role_type="leader_follower",
        action=SPEED,
        schema=SCHEMA,
        epoch=200,
        config=config,
        rng=31,
        current_state={"x": 2.0, "y": 0.5},
        eligible_rule_ids=(matching.rule_id,),
    )

    assert len(children) == 2
    assert all(child.parent_ids == (matching.rule_id,) for child in children)
    assert all(disjoint.rule_id not in child.parent_ids for child in children)


def test_evolve_niche_can_prepare_births_without_population_mutation() -> None:
    parent = _rule(
        "matching",
        interval=(0.0, 4.0),
        evolution_mean=0.5,
    )
    population = Population((parent,), max_size=1)
    before = population.to_dict()
    config = LearningConfig(
        population_limit=1,
        ga_min_experience=2,
        crossover_probability=0.0,
        mutation_probability=0.0,
        random_seed=31,
    )

    children = evolve_niche(
        population,
        role_type="leader_follower",
        action=SPEED,
        schema=SCHEMA,
        epoch=200,
        config=config,
        rng=31,
        current_state={"x": 2.0, "y": 0.5},
        eligible_rule_ids=(parent.rule_id,),
        insert=False,
    )

    assert len(children) == 1
    assert children[0].parent_ids == (parent.rule_id,)
    assert population.to_dict() == before
