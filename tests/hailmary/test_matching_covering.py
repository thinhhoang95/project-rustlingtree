from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from hailmary.actions.models import ActionLever
from hailmary.actions.vocabulary import PATH_STRETCH_MACRO_BAND
from hailmary.config import LearningConfig
from hailmary.features.schema import FeatureField, FeatureSchema
from hailmary.learning.conditions import Interval, RuleCondition
from hailmary.learning.covering import cover_missing_actions
from hailmary.learning.matching import AnchorContext, build_match_set
from hailmary.learning.population import Population
from hailmary.learning.rules import MutableRule, RuleAction


ROLE_TYPE = "leader_follower"


@dataclass(frozen=True, slots=True)
class Candidate:
    lever: ActionLever
    band: str
    feasible: bool = True


@dataclass(slots=True)
class MutableCandidate:
    lever: ActionLever
    band: str
    feasible: bool = True


def _schema() -> FeatureSchema:
    return FeatureSchema(
        schema_version="test.matching.v1",
        fields=(
            FeatureField(
                "spacing_deviation_s",
                "s",
                "identity",
                -100.0,
                100.0,
            ),
            FeatureField(
                "abs_spacing_deviation_s",
                "s",
                "identity",
                0.0,
                100.0,
            ),
            FeatureField(
                "commitment_fraction",
                "1",
                "identity",
                0.0,
                1.0,
            ),
            FeatureField(
                "pressure_ratio",
                "1",
                "identity",
                0.0,
                2.0,
            ),
        ),
    )


def _context(*candidates: Candidate) -> AnchorContext:
    schema = _schema()
    vector = schema.encode(
        {
            "spacing_deviation_s": 0.0,
            "abs_spacing_deviation_s": 50.0,
            "commitment_fraction": 0.5,
            "pressure_ratio": 1.0,
        }
    )
    return AnchorContext(
        role_type=ROLE_TYPE,
        schema_hash=schema.schema_hash,
        anchor_id="anchor-1",
        vector=vector,
        candidates=tuple(candidates),
    )


def _rule(
    context: AnchorContext,
    action: RuleAction,
    *,
    rule_id: str,
    spacing_interval: Interval = Interval(-10.0, 10.0),
) -> MutableRule:
    return MutableRule(
        condition=RuleCondition(
            role_type=context.role_type,
            schema_hash=context.schema_hash,
            intervals={"spacing_deviation_s": spacing_interval},
        ),
        action=action,
        rule_id=rule_id,
    )


def test_match_set_freezes_rule_ids_grouped_by_exact_feasible_action() -> None:
    no_op = RuleAction.no_op()
    light = RuleAction(ActionLever.SPEED, "light")
    context = _context(
        Candidate(ActionLever.NO_OP, "no_op"),
        Candidate(ActionLever.SPEED, "light"),
        Candidate(
            ActionLever.PATH_STRETCH,
            PATH_STRETCH_MACRO_BAND,
            feasible=False,
        ),
    )
    no_op_rule = _rule(context, no_op, rule_id="rule-no-op")
    light_rule = _rule(
        context,
        light,
        rule_id="rule-light",
        spacing_interval=Interval(-20.0, 20.0),
    )
    ignored_medium = _rule(
        context,
        RuleAction(ActionLever.SPEED, "medium"),
        rule_id="rule-medium",
    )
    population = Population((no_op_rule, light_rule, ignored_medium))

    match_set = build_match_set(context, population)

    assert match_set.role_type == ROLE_TYPE
    assert match_set.schema_hash == context.schema_hash
    assert match_set.anchor_id == "anchor-1"
    assert match_set.vector is context.vector
    assert match_set.candidate_actions == (no_op, light)
    assert match_set.advocates(no_op) == ("rule-no-op",)
    assert match_set.advocates(light) == ("rule-light",)
    assert match_set.matched_rule_ids == ("rule-light", "rule-no-op")
    assert all(
        candidate.lever is not ActionLever.PATH_STRETCH
        for candidate in match_set.candidates
    )

    population.remove(light_rule.rule_id)
    no_op_rule.update_rival(3.0)
    assert match_set.advocates(light) == ("rule-light",)
    with pytest.raises(TypeError):
        match_set.rule_ids_by_action[no_op] = ()  # type: ignore[index]


def test_anchor_context_rejects_mutable_candidate_records() -> None:
    schema = _schema()
    vector = schema.encode(
        {
            "spacing_deviation_s": 0.0,
            "abs_spacing_deviation_s": 50.0,
            "commitment_fraction": 0.5,
            "pressure_ratio": 1.0,
        }
    )

    with pytest.raises(TypeError, match="frozen dataclass"):
        AnchorContext(
            role_type=ROLE_TYPE,
            schema_hash=schema.schema_hash,
            anchor_id="anchor-1",
            vector=vector,
            candidates=(MutableCandidate(ActionLever.NO_OP, "no_op"),),
        )


def test_covering_creates_seeded_empty_rules_only_for_missing_feasible_actions() -> (
    None
):
    no_op = RuleAction.no_op()
    light = RuleAction(ActionLever.SPEED, "light")
    context = _context(
        Candidate(ActionLever.NO_OP, "no_op"),
        Candidate(ActionLever.SPEED, "light"),
        Candidate(
            ActionLever.PATH_STRETCH,
            PATH_STRETCH_MACRO_BAND,
            feasible=False,
        ),
    )
    population = Population((_rule(context, no_op, rule_id="rule-no-op"),))
    match_set = build_match_set(context, population)
    config = replace(
        LearningConfig(),
        covering_width_min_fraction=0.1,
        covering_width_max_fraction=0.2,
    )

    covered = cover_missing_actions(
        match_set,
        config=config,
        creation_epoch=7,
        seed=41,
    )
    repeated = cover_missing_actions(
        match_set,
        config=config,
        creation_epoch=7,
        seed=41,
    )

    assert len(covered) == 1
    rule = covered[0]
    assert rule.action == light
    assert rule.creation_epoch == 7
    assert rule.last_ga_epoch == 7
    assert rule.evolution.n == 0
    assert rule.deployment.n == 0
    assert rule.parent_ids == ()
    assert rule.provenance["kind"] == "covering"
    assert rule.provenance["anchor_id"] == context.anchor_id
    assert rule.provenance["action_key"] == light.key
    assert rule.provenance["covering_seed"] == 41
    assert rule.matches(
        context.vector,
        role_type=context.role_type,
        schema_hash=context.schema_hash,
    )
    assert rule.to_dict() == repeated[0].to_dict()
    assert match_set.advocates(light) == ()

    population.add(rule)
    rollout_match_set = build_match_set(context, population)
    assert rollout_match_set.advocates(light) == (rule.rule_id,)

    width_fractions = rule.provenance["width_fractions"]
    for field in context.vector.schema.fields:
        interval = rule.condition.intervals[field.name]
        value = context.vector.value(field.name)
        assert interval.matches(value)
        assert interval.lower is not None
        assert interval.upper is not None
        assert field.lower_bound is not None
        assert field.upper_bound is not None
        assert interval.lower >= field.lower_bound
        assert interval.upper <= field.upper_bound
        fraction = float(width_fractions[field.name])
        assert config.covering_width_min_fraction <= fraction
        assert fraction <= config.covering_width_max_fraction
        span = field.upper_bound - field.lower_bound
        assert interval.upper - interval.lower == pytest.approx(fraction * span)

    assert all(
        new_rule.action.lever is not ActionLever.PATH_STRETCH for new_rule in covered
    )
