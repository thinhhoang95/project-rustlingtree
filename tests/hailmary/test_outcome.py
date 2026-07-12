from __future__ import annotations

import pytest

from hailmary.config import OutcomeConfig
from hailmary.evaluation.outcome import (
    InterventionSummary,
    freeze_outcome_cohort,
    rollout_horizon_s,
    score_semi_local_outcome,
)
from hailmary.evaluation.spacing import spacing_ratio_score
from hailmary.features.anchors import LeaderFollowerAnchor


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (-0.1, -1.0),
        (0.0, -1.0),
        (0.5, 0.0),
        (1.0, 1.0),
        (1.25, 1.0),
        (1.5, 0.5),
        (2.0, -0.5),
        (3.0, -0.5),
    ],
)
def test_spacing_score_exact_piecewise_boundaries(ratio: float, expected: float) -> None:
    assert spacing_ratio_score(90.0 * ratio, 90.0) == pytest.approx(expected)


def test_infeasible_completion_receives_negative_one() -> None:
    assert spacing_ratio_score(90.0, 90.0, dynamically_feasible=False) == -1.0


def _anchor() -> LeaderFollowerAnchor:
    return LeaderFollowerAnchor("RW17L", "L", "F", "state-v1", 3)


def test_cohort_freezes_next_three_trailers_and_horizon() -> None:
    cohort = freeze_outcome_cohort(
        _anchor(),
        ("L", "F", "T1", "T2", "T3", "T4"),
        trailer_count=3,
    )
    crossings = {"L": 100.0, "F": 190.0, "T1": 280.0, "T2": 370.0, "T3": 460.0}

    assert cohort.trailer_ids == ("T1", "T2", "T3")
    assert rollout_horizon_s(crossings, cohort) == 550.0


def test_short_flow_horizon_uses_follower_when_no_trailer_exists() -> None:
    cohort = freeze_outcome_cohort(_anchor(), ("L", "F"), trailer_count=3)
    assert cohort.effective_trailer_count == 0
    assert rollout_horizon_s({"L": 100.0, "F": 190.0}, cohort) == 280.0


def test_semi_local_outcome_uses_pair_and_mean_trailer_propagation() -> None:
    cohort = freeze_outcome_cohort(_anchor(), ("L", "F", "T1", "T2", "T3"))
    crossings = {"L": 100.0, "F": 190.0, "T1": 280.0, "T2": 370.0, "T3": 460.0}

    outcome = score_semi_local_outcome(
        cohort,
        crossing_times_s=crossings,
        intervention=InterventionSummary(),
        config=OutcomeConfig(),
    )

    assert outcome.pair_score == 1.0
    assert outcome.propagation_score == 1.0
    assert outcome.throughput_score == 0.0
    assert outcome.intervention_penalty == 0.0
    assert outcome.score == 2.0
    assert outcome.effective_trailer_count == 3
    assert outcome.horizon_s == 550.0


def test_beneficial_spacing_has_higher_outcome_than_compression() -> None:
    cohort = freeze_outcome_cohort(_anchor(), ("L", "F"))
    separated = score_semi_local_outcome(cohort, crossing_times_s={"L": 100.0, "F": 190.0})
    compressed = score_semi_local_outcome(cohort, crossing_times_s={"L": 100.0, "F": 145.0})

    assert separated.score > compressed.score
