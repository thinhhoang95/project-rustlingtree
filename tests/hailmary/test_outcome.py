from __future__ import annotations

from types import SimpleNamespace

import pytest

import hailmary.evaluation.outcome as outcome_module
from hailmary.config import OutcomeConfig
from hailmary.evaluation.outcome import (
    InterventionSummary,
    SimulatorOutcomePlan,
    freeze_outcome_cohort,
    rollout_horizon_s,
    score_semi_local_outcome,
    score_simulator_outcome,
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
def test_spacing_score_exact_piecewise_boundaries(
    ratio: float, expected: float
) -> None:
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


def test_intervention_summary_difference_returns_rollout_only_values() -> None:
    baseline = InterventionSummary(
        action_count=5,
        speed_action_count=3,
        total_speed_reduction_kts=30.0,
        stretch_action_count=2,
        total_stretch_added_distance_nm=4.0,
    )
    final = InterventionSummary(
        action_count=7,
        speed_action_count=4,
        total_speed_reduction_kts=50.0,
        stretch_action_count=3,
        total_stretch_added_distance_nm=9.0,
    )

    assert final.difference(baseline) == InterventionSummary(
        action_count=2,
        speed_action_count=1,
        total_speed_reduction_kts=20.0,
        stretch_action_count=1,
        total_stretch_added_distance_nm=5.0,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_count": True},
        {"action_count": 1.0},
        {"speed_action_count": False},
        {"stretch_action_count": 0.5},
    ],
)
def test_intervention_summary_requires_exact_non_bool_integer_counts(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match="exact non-bool integers"):
        InterventionSummary(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("final", "baseline"),
    [
        (
            InterventionSummary(
                action_count=1,
                speed_action_count=1,
                total_speed_reduction_kts=10.0,
            ),
            InterventionSummary(
                action_count=2,
                speed_action_count=1,
                total_speed_reduction_kts=10.0,
            ),
        ),
        (
            InterventionSummary(action_count=1),
            InterventionSummary(action_count=1, speed_action_count=1),
        ),
        (
            InterventionSummary(
                action_count=1,
                speed_action_count=1,
                total_speed_reduction_kts=10.0,
            ),
            InterventionSummary(
                action_count=1,
                speed_action_count=1,
                total_speed_reduction_kts=11.0,
            ),
        ),
        (
            InterventionSummary(action_count=1),
            InterventionSummary(action_count=1, stretch_action_count=1),
        ),
        (
            InterventionSummary(
                action_count=1,
                stretch_action_count=1,
                total_stretch_added_distance_nm=2.0,
            ),
            InterventionSummary(
                action_count=1,
                stretch_action_count=1,
                total_stretch_added_distance_nm=3.0,
            ),
        ),
    ],
)
def test_intervention_summary_difference_rejects_negative_deltas(
    final: InterventionSummary,
    baseline: InterventionSummary,
) -> None:
    with pytest.raises(ValueError, match="negative rollout deltas"):
        final.difference(baseline)


def test_simulator_outcome_plan_excludes_historical_interventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = freeze_outcome_cohort(_anchor(), ("L", "F"))
    historical = InterventionSummary(
        action_count=5,
        speed_action_count=3,
        total_speed_reduction_kts=30.0,
        stretch_action_count=2,
        total_stretch_added_distance_nm=4.0,
    )
    plan = SimulatorOutcomePlan(
        cohort=cohort,
        root_dynamic_content_hash="root-dynamic-content-hash",
        horizon_s=280.0,
        baseline_crossing_times_s=(("L", 100.0), ("F", 190.0)),
        baseline_intervention_summary=historical,
        root_time_s=12.0,
    )
    variant = SimpleNamespace(diagnostics=SimpleNamespace(feasible=True))
    dynamics = {
        flight_id: SimpleNamespace(current_variant_id="variant")
        for flight_id in cohort.ordered_flight_ids
    }
    state = SimpleNamespace(
        definition=SimpleNamespace(variant=lambda _variant_id: variant),
        flight=lambda flight_id: dynamics[flight_id],
    )
    simulator = SimpleNamespace(
        state=state,
        dynamic_content_hash=plan.root_dynamic_content_hash,
    )
    crossing_times = plan.baseline_crossing_times
    monkeypatch.setattr(
        "hailmary.features.anchors.resource_eta_s",
        lambda _simulator, flight_id, _resource_id: crossing_times[flight_id],
    )
    current = {"summary": historical}
    monkeypatch.setattr(
        outcome_module,
        "simulator_intervention_summary",
        lambda _simulator: current["summary"],
    )

    no_op_outcome = score_simulator_outcome(simulator, outcome_plan=plan)
    assert no_op_outcome.intervention_penalty == 0.0

    current["summary"] = InterventionSummary(
        action_count=6,
        speed_action_count=4,
        total_speed_reduction_kts=50.0,
        stretch_action_count=2,
        total_stretch_added_distance_nm=4.0,
    )
    speed_outcome = score_simulator_outcome(simulator, plan)
    assert speed_outcome.intervention_penalty == pytest.approx(4.0 / 9.0)

    cumulative_outcome = score_simulator_outcome(
        simulator,
        cohort,
        horizon_s=plan.horizon_s,
    )
    assert cumulative_outcome.intervention_penalty > speed_outcome.intervention_penalty


def test_historical_interventions_do_not_change_selected_vs_noop_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = freeze_outcome_cohort(_anchor(), ("L", "F"))
    crossings = (("L", 100.0), ("F", 190.0))
    clean = InterventionSummary()
    historical = InterventionSummary(
        action_count=2,
        speed_action_count=1,
        total_speed_reduction_kts=10.0,
        stretch_action_count=1,
        total_stretch_added_distance_nm=4.0,
    )
    clean_plan = SimulatorOutcomePlan(
        cohort=cohort,
        horizon_s=280.0,
        baseline_crossing_times_s=crossings,
        root_dynamic_content_hash="shared-physical-root",
        baseline_intervention_summary=clean,
    )
    historical_plan = SimulatorOutcomePlan(
        cohort=cohort,
        horizon_s=280.0,
        baseline_crossing_times_s=crossings,
        root_dynamic_content_hash="shared-physical-root",
        baseline_intervention_summary=historical,
    )
    variant = SimpleNamespace(diagnostics=SimpleNamespace(feasible=True))
    dynamics = {
        flight_id: SimpleNamespace(current_variant_id="variant")
        for flight_id in cohort.ordered_flight_ids
    }
    simulator = SimpleNamespace(
        state=SimpleNamespace(
            definition=SimpleNamespace(variant=lambda _variant_id: variant),
            flight=lambda flight_id: dynamics[flight_id],
        ),
        dynamic_content_hash="shared-physical-root",
    )
    crossing_lookup = dict(crossings)
    monkeypatch.setattr(
        "hailmary.features.anchors.resource_eta_s",
        lambda _simulator, flight_id, _resource_id: crossing_lookup[flight_id],
    )
    current = {"summary": clean}
    monkeypatch.setattr(
        outcome_module,
        "simulator_intervention_summary",
        lambda _simulator: current["summary"],
    )

    post_root_speed = InterventionSummary(
        action_count=1,
        speed_action_count=1,
        total_speed_reduction_kts=15.0,
    )
    current["summary"] = post_root_speed
    clean_selected = score_simulator_outcome(simulator, outcome_plan=clean_plan).score
    current["summary"] = clean
    clean_no_op = score_simulator_outcome(simulator, outcome_plan=clean_plan).score

    current["summary"] = InterventionSummary(
        action_count=historical.action_count + post_root_speed.action_count,
        speed_action_count=(
            historical.speed_action_count + post_root_speed.speed_action_count
        ),
        total_speed_reduction_kts=(
            historical.total_speed_reduction_kts
            + post_root_speed.total_speed_reduction_kts
        ),
        stretch_action_count=historical.stretch_action_count,
        total_stretch_added_distance_nm=historical.total_stretch_added_distance_nm,
    )
    historical_selected = score_simulator_outcome(
        simulator, outcome_plan=historical_plan
    ).score
    current["summary"] = historical
    historical_no_op = score_simulator_outcome(
        simulator, outcome_plan=historical_plan
    ).score

    assert historical_selected - historical_no_op == pytest.approx(
        clean_selected - clean_no_op
    )


def test_outcome_plan_rejects_wrong_root_before_baseline_subtraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = SimulatorOutcomePlan(
        cohort=freeze_outcome_cohort(_anchor(), ("L", "F")),
        horizon_s=280.0,
        baseline_crossing_times_s=(("L", 100.0), ("F", 190.0)),
        root_dynamic_content_hash="expected-root",
    )
    simulator = SimpleNamespace(dynamic_content_hash="another-root")
    monkeypatch.setattr(
        outcome_module,
        "simulator_intervention_summary",
        lambda _simulator: pytest.fail("baseline subtraction was reached"),
    )

    with pytest.raises(ValueError, match="root dynamic-content hash"):
        score_simulator_outcome(simulator, outcome_plan=plan)


def test_beneficial_spacing_has_higher_outcome_than_compression() -> None:
    cohort = freeze_outcome_cohort(_anchor(), ("L", "F"))
    separated = score_semi_local_outcome(
        cohort, crossing_times_s={"L": 100.0, "F": 190.0}
    )
    compressed = score_semi_local_outcome(
        cohort, crossing_times_s={"L": 100.0, "F": 145.0}
    )

    assert separated.score > compressed.score
