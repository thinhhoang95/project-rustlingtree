from __future__ import annotations

from dataclasses import dataclass, replace
import json

import pytest

from hailmary.actions.models import ActionLever
from hailmary.actions.vocabulary import PATH_STRETCH_MACRO_BAND
from hailmary.config import LearningConfig
from hailmary.features.schema import FeatureField, FeatureSchema
from hailmary.learning.exploration import RegionCell, RegionExplorationScheduler
from hailmary.learning.matching import AnchorContext
from hailmary.learning.rules import RuleAction


@dataclass(frozen=True, slots=True)
class Candidate:
    lever: ActionLever
    band: str
    feasible: bool = True


def _context(
    *candidates: Candidate,
    commitment: float = 0.5,
    pressure: float = 1.0,
    abs_error_s: float = 50.0,
) -> AnchorContext:
    schema = FeatureSchema(
        schema_version="test.exploration.v1",
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
    vector = schema.encode(
        {
            "spacing_deviation_s": -abs_error_s,
            "abs_spacing_deviation_s": abs_error_s,
            "commitment_fraction": commitment,
            "pressure_ratio": pressure,
        }
    )
    return AnchorContext(
        role_type="leader_follower",
        schema_hash=schema.schema_hash,
        anchor_id="anchor-exploration",
        vector=vector,
        candidates=tuple(candidates),
    )


def _three_action_context() -> AnchorContext:
    return _context(
        Candidate(ActionLever.NO_OP, "no_op"),
        Candidate(ActionLever.SPEED, "light"),
        Candidate(ActionLever.PATH_STRETCH, PATH_STRETCH_MACRO_BAND),
    )


def _complete(
    scheduler: RegionExplorationScheduler,
    context: AnchorContext,
    *,
    exploit_action: RuleAction | None = None,
):
    decision = scheduler.choose(context, exploit_action=exploit_action)
    assert scheduler.record_experiment(decision) == (
        decision.action_experiment_count + 1
    )
    return decision


def test_coverage_floor_balances_all_feasible_actions_including_no_op() -> None:
    context = _three_action_context()
    config = replace(
        LearningConfig(),
        exploration_coverage_floor=6,
        base_exploration_rate=0.0,
    )
    scheduler = RegionExplorationScheduler(config, seed=7)

    decisions = [_complete(scheduler, context) for _ in range(6)]

    assert all(decision.reason == "coverage_floor" for decision in decisions)
    assert all(decision.exploratory for decision in decisions)
    assert any(decision.action.is_no_op for decision in decisions)
    cell = scheduler.cell_for(context)
    assert cell == RegionCell("medium", "nominal", "medium")
    assert scheduler.visit_count(cell) == 6
    assert {
        action: scheduler.experiment_count(cell, action)
        for action in context.candidate_actions
    } == {action: 2 for action in context.candidate_actions}
    assert all(
        decision.candidate is context.candidate_for(decision.action)
        for decision in decisions
    )


def test_base_exploration_runs_only_after_the_region_coverage_floor() -> None:
    context = _three_action_context()
    config = replace(
        LearningConfig(),
        exploration_coverage_floor=3,
        base_exploration_rate=1.0,
    )
    scheduler = RegionExplorationScheduler(config, seed=11)

    floor_decisions = [_complete(scheduler, context) for _ in range(3)]
    base_decision = _complete(scheduler, context)

    assert all(decision.reason == "coverage_floor" for decision in floor_decisions)
    assert base_decision.reason == "base_exploration"
    assert base_decision.exploratory
    assert base_decision.action in context.candidate_actions


def test_exploit_selection_has_a_deterministic_feasible_fallback() -> None:
    context = _context(
        Candidate(ActionLever.NO_OP, "no_op"),
        Candidate(ActionLever.SPEED, "light"),
    )
    config = replace(
        LearningConfig(),
        exploration_coverage_floor=2,
        base_exploration_rate=0.0,
    )
    scheduler = RegionExplorationScheduler(config, seed=5)
    for _ in context.candidate_actions:
        _complete(scheduler, context)

    speed = RuleAction(ActionLever.SPEED, "light")
    exploit = _complete(scheduler, context, exploit_action=speed)
    fallback = _complete(
        scheduler,
        context,
        exploit_action=RuleAction(
            ActionLever.PATH_STRETCH,
            PATH_STRETCH_MACRO_BAND,
        ),
    )

    assert exploit.reason == "exploit"
    assert exploit.action == speed
    assert not exploit.exploratory
    assert fallback.reason == "deterministic_fallback"
    assert fallback.action == RuleAction.no_op()
    assert not fallback.exploratory


def test_selection_does_not_count_an_experiment_until_recorded() -> None:
    context = _three_action_context()
    scheduler = RegionExplorationScheduler(
        replace(
            LearningConfig(),
            exploration_coverage_floor=3,
            base_exploration_rate=0.0,
        ),
        seed=9,
    )
    cell = scheduler.cell_for(context)

    decision = scheduler.choose(context)

    assert scheduler.visit_count(cell) == 1
    assert scheduler.experiment_count(cell, decision.action) == 0
    assert decision.action_experiment_count == 0
    assert scheduler.record_experiment(decision) == 1
    with pytest.raises(ValueError, match="already recorded"):
        scheduler.record_experiment(decision)


def test_newly_feasible_action_gets_coverage_after_visit_floor() -> None:
    initial = _context(
        Candidate(ActionLever.NO_OP, "no_op"),
        Candidate(ActionLever.SPEED, "light"),
    )
    expanded = _three_action_context()
    config = replace(
        LearningConfig(),
        exploration_coverage_floor=2,
        base_exploration_rate=0.0,
    )
    scheduler = RegionExplorationScheduler(config, seed=3)
    speed = RuleAction(ActionLever.SPEED, "light")
    for _ in range(6):
        _complete(scheduler, initial, exploit_action=speed)

    cell = scheduler.cell_for(initial)
    assert scheduler.visit_count(cell) == 6
    stretch = RuleAction(ActionLever.PATH_STRETCH, PATH_STRETCH_MACRO_BAND)
    assert scheduler.experiment_count(cell, stretch) == 0

    late = scheduler.choose(expanded, exploit_action=speed)

    assert late.action == stretch
    assert late.reason == "coverage_floor"
    assert late.exploratory
    assert scheduler.experiment_count(cell, stretch) == 0
    scheduler.record_experiment(late)
    assert scheduler.experiment_count(cell, stretch) == 1


@pytest.mark.parametrize("section", ("visits", "experiments"))
@pytest.mark.parametrize("invalid_count", (True, 1.5))
def test_restored_counts_require_exact_integers(
    section: str,
    invalid_count: object,
) -> None:
    context = _three_action_context()
    scheduler = RegionExplorationScheduler(
        replace(LearningConfig(), exploration_coverage_floor=3),
        seed=5,
    )
    _complete(scheduler, context)
    payload = scheduler.to_dict()
    payload[section][0]["count"] = invalid_count

    with pytest.raises(ValueError, match="non-negative integers"):
        RegionExplorationScheduler.from_dict(payload)


def test_seed_and_json_state_round_trip_replay_future_choices_exactly() -> None:
    context = _three_action_context()
    config = replace(
        LearningConfig(),
        exploration_coverage_floor=1,
        base_exploration_rate=0.4,
        random_seed=73,
    )
    original = RegionExplorationScheduler(config)
    same_seed = RegionExplorationScheduler(config)

    original_prefix = [_complete(original, context) for _ in range(8)]
    same_seed_prefix = [_complete(same_seed, context) for _ in range(8)]
    assert [(decision.action, decision.reason) for decision in original_prefix] == [
        (decision.action, decision.reason) for decision in same_seed_prefix
    ]

    serialized = json.loads(json.dumps(original.state_dict()))
    restored = RegionExplorationScheduler.from_state_dict(serialized)
    for _ in range(20):
        expected = _complete(
            original,
            context,
            exploit_action=RuleAction(ActionLever.SPEED, "medium"),
        )
        actual = _complete(
            restored,
            context,
            exploit_action=RuleAction(ActionLever.SPEED, "medium"),
        )
        assert (
            actual.cell,
            actual.action,
            actual.reason,
            actual.exploratory,
            actual.visit_count,
            actual.action_experiment_count,
        ) == (
            expected.cell,
            expected.action,
            expected.reason,
            expected.exploratory,
            expected.visit_count,
            expected.action_experiment_count,
        )

    assert restored.state_dict() == original.state_dict()
