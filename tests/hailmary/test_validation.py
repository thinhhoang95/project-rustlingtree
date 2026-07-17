from __future__ import annotations

from dataclasses import replace
import json

import pytest

from hailmary.actions.models import ActionLever
from hailmary.ids import canonical_json, content_hash
from hailmary.learning.certification import FrozenActionRule
from hailmary.learning.conditions import RuleCondition
from hailmary.learning.rulebook import FrozenRulebookPolicy
from hailmary.learning.rules import RuleAction
from hailmary.learning.validation import (
    ActionCertificationEvidence,
    AuthenticatedRuntimeManifest,
    HeldOutComparison,
    HeldOutOutcomeDetails,
    HeldOutRolloutProvenance,
    PATH_STRETCH_ACTION_KEY,
    PHASE0_ACTION_KEYS,
    PathStretchAblationResult,
    Phase0AcceptanceEvaluator,
    Phase0ValidationInputs,
    RefreshIntervalResult,
    RuleHyperrectangle,
    RuleRegion,
    SeededRuleRegions,
    TrainDeployDecision,
    evaluate_phase0_acceptance,
)
from hailmary.rollout.policy import NoOpPolicy, policy_fingerprint


SCHEMA_HASH = "phase0-validation-schema"


def _speed_action() -> RuleAction:
    return RuleAction(ActionLever.SPEED, "light")


def _path_action() -> RuleAction:
    return RuleAction(
        ActionLever.PATH_STRETCH,
        "oracle_short_medium_long",
    )


def _runtime_manifest(selector: str) -> AuthenticatedRuntimeManifest:
    manifest = {
        "schema_version": "hailmary.action_runtime.v1",
        "action_vocabulary": {"version": 1},
        "stretch": {"selector": selector, "config": {"max_turn_deg": 90.0}},
    }
    return AuthenticatedRuntimeManifest.from_configuration(
        manifest,
        content_hash(manifest, namespace="hailmary.action_runtime.v1"),
    )


def _outcome_details(
    score: float, *, horizon_s: float = 1_000.0
) -> HeldOutOutcomeDetails:
    return HeldOutOutcomeDetails(
        score=score,
        pair_score=score,
        propagation_score=score / 2.0,
        intervention_penalty=0.1,
        throughput_score=0.5,
        effective_trailer_count=0,
        horizon_s=horizon_s,
    )


def _held_out(
    scenario_id: str,
    *,
    policy_score: float,
    permanent_no_op_score: float,
    learned_policy_fingerprint: str = "learned-policy",
) -> HeldOutComparison:
    root = f"root-{scenario_id}"
    outcome_plan = {
        "schema_version": "hailmary.phase0.outcome_plan.v1",
        "root_dynamic_content_hash": root,
        "root_time_s": 100.0,
        "horizon_s": 1_000.0,
        "cohort": {"anchor_id": f"anchor-{scenario_id}"},
    }
    provenance = HeldOutRolloutProvenance(
        parent_dynamic_content_hash=root,
        outcome_plan_json=canonical_json(outcome_plan),
        outcome_plan_hash=content_hash(
            outcome_plan,
            namespace="hailmary.phase0.outcome_plan.v1",
        ),
        runtime=_runtime_manifest("semi_local_outcome"),
        learned_policy_fingerprint=learned_policy_fingerprint,
        permanent_no_op_policy_fingerprint=policy_fingerprint(NoOpPolicy()),
        policy_arm_initial_dynamic_content_hash=root,
        policy_arm_final_dynamic_content_hash=f"policy-final-{scenario_id}",
        permanent_no_op_initial_dynamic_content_hash=root,
        permanent_no_op_final_dynamic_content_hash=f"noop-final-{scenario_id}",
        policy_outcome=_outcome_details(policy_score),
        permanent_no_op_outcome=_outcome_details(permanent_no_op_score),
    )
    return HeldOutComparison(
        scenario_id,
        policy_score=policy_score,
        permanent_no_op_score=permanent_no_op_score,
        provenance=provenance,
    )


def _rectangle(
    action: RuleAction,
    **bounds: tuple[float, float],
) -> RuleHyperrectangle:
    return RuleHyperrectangle(
        action,
        tuple(
            RuleRegion(action, name, lower, upper)
            for name, (lower, upper) in bounds.items()
        ),
    )


def _accepted_rulebook() -> FrozenRulebookPolicy:
    speed = FrozenActionRule(
        source_rule_id="speed-rule",
        condition=RuleCondition(
            "leader_follower",
            SCHEMA_HASH,
            {"required_delay_over_speed_capacity": (0.0, 1.0)},
        ),
        action=_speed_action(),
        w=0.5,
        precision=10.0,
        rival_lcb=0.1,
        rival_samples=20,
    )
    path = FrozenActionRule(
        source_rule_id="path-rule",
        condition=RuleCondition(
            "leader_follower",
            SCHEMA_HASH,
            {"required_delay_over_path_capacity": (1.0, 10.0)},
        ),
        action=_path_action(),
        w=0.8,
        precision=8.0,
        rival_lcb=0.2,
        rival_samples=20,
    )
    return FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        certification_generation=4,
        action_rules=(speed, path),
    )


def _accepted_certification_evidence() -> dict[str, ActionCertificationEvidence]:
    return {
        "speed-rule": ActionCertificationEvidence(
            "speed-rule", noop_samples=30, noop_lcb=0.1
        ),
        "path-rule": ActionCertificationEvidence(
            "path-rule", noop_samples=31, noop_lcb=0.2
        ),
    }


def _accepted_inputs() -> Phase0ValidationInputs:
    concept = {
        "small_within_speed_capacity": _speed_action(),
        "beyond_speed_within_path_capacity": _path_action(),
    }
    return Phase0ValidationInputs(
        train_deploy_decisions=(
            TrainDeployDecision(
                "held-1",
                training_action=_speed_action(),
                deployment_action=_speed_action(),
            ),
            TrainDeployDecision(
                "held-2",
                training_action=None,
                deployment_action=RuleAction.no_op(),
            ),
        ),
        held_out_comparisons=(
            _held_out("held-1", policy_score=2.0, permanent_no_op_score=1.0),
            _held_out("held-2", policy_score=1.5, permanent_no_op_score=1.0),
        ),
        path_stretch_ablation=PathStretchAblationResult(
            action_key=PATH_STRETCH_ACTION_KEY,
            oracle_score=1.2,
            geometry_only_score=0.8,
            oracle_runtime=_runtime_manifest("semi_local_outcome"),
            geometry_only_runtime=_runtime_manifest("geometry_clearance"),
        ),
        refresh_interval_results=(
            RefreshIntervalResult(100, concept, (100, 200)),
            RefreshIntervalResult(500, concept, (500, 1_000)),
        ),
        seeded_rule_regions=(
            SeededRuleRegions(
                17,
                (
                    _rectangle(
                        _speed_action(),
                        required_delay_over_speed_capacity=(0.0, 1.0),
                    ),
                    _rectangle(
                        _path_action(),
                        required_delay_over_path_capacity=(1.0, 3.0),
                    ),
                ),
            ),
            SeededRuleRegions(
                29,
                (
                    _rectangle(
                        _speed_action(),
                        required_delay_over_speed_capacity=(0.1, 1.1),
                    ),
                    _rectangle(
                        _path_action(),
                        required_delay_over_path_capacity=(1.2, 3.2),
                    ),
                ),
            ),
        ),
        action_certification_evidence=_accepted_certification_evidence(),
    )


def test_phase0_action_contract_is_the_exact_current_five_identities() -> None:
    assert PHASE0_ACTION_KEYS == (
        "no_op/no_op",
        "speed/light",
        "speed/medium",
        "speed/heavy",
        "path_stretch/oracle_short_medium_long",
    )
    assert PATH_STRETCH_ACTION_KEY == ("path_stretch/oracle_short_medium_long")


def test_phase0_acceptance_report_passes_complete_synthetic_evidence() -> None:
    rulebook = _accepted_rulebook()
    inputs = _accepted_inputs()

    report = evaluate_phase0_acceptance(rulebook, inputs)

    assert report.passed
    assert report.rulebook_fingerprint == rulebook.policy_fingerprint()
    assert len(report.checks) == 8
    assert not report.failures
    assert report.check("capacity_ratio_predicates").passed
    assert (
        report.check("capacity_ratio_predicates").metrics["distance_only_rule_count"]
        == 0
    )
    certification = report.check("positive_noop_grounded_certification")
    assert certification.passed
    assert certification.metrics["minimum_noop_samples"] == 30
    assert certification.metrics["minimum_noop_lcb"] == pytest.approx(0.1)
    assert certification.metrics["missing_evidence_count"] == 0
    assert certification.metrics["unexpected_evidence_count"] == 0
    assert report.check("exploit_deployment_agreement").passed
    assert report.check("held_out_beats_permanent_no_op").metrics[
        "mean_improvement"
    ] == pytest.approx(0.75)
    assert report.check("path_stretch_oracle_and_geometry_ablation").passed
    assert report.check("refresh_interval_robustness").passed
    assert report.check("seeded_rule_region_comparability").passed
    json.dumps(report.to_dict())


def test_held_out_provenance_rejects_tampered_root_policy_and_scores() -> None:
    comparison = _held_out(
        "held-tamper",
        policy_score=2.0,
        permanent_no_op_score=1.0,
    )
    provenance = comparison.provenance
    assert provenance is not None

    with pytest.raises(ValueError, match="learned arm initial hash"):
        replace(provenance, policy_arm_initial_dynamic_content_hash="other-root")
    with pytest.raises(ValueError, match="canonical NoOpPolicy"):
        replace(provenance, permanent_no_op_policy_fingerprint="forged-policy")
    with pytest.raises(ValueError, match="policy score"):
        replace(comparison, policy_score=99.0)

    tampered_plan = dict(provenance.outcome_plan)
    tampered_plan["horizon_s"] = 2_000.0
    with pytest.raises(ValueError, match="outcome-plan hash"):
        replace(provenance, outcome_plan_json=canonical_json(tampered_plan))


def test_causal_and_vanilla_controls_require_full_permanent_identity() -> None:
    inputs = _accepted_inputs()
    causal = inputs.held_out_comparisons[0]
    assert causal.provenance is not None
    forged_provenance = replace(
        causal.provenance,
        permanent_no_op_final_dynamic_content_hash="different-final-state",
    )
    forged_vanilla = replace(
        causal,
        provenance=forged_provenance,
    )

    with pytest.raises(ValueError, match="auditable permanent no-op provenance"):
        replace(
            inputs,
            held_out_comparisons=(causal,),
            vanilla_held_out_comparisons=(forged_vanilla,),
        )

    valid_vanilla = _held_out(
        causal.scenario_id,
        policy_score=causal.policy_score,
        permanent_no_op_score=causal.permanent_no_op_score,
        learned_policy_fingerprint="vanilla-policy",
    )
    paired = replace(
        inputs,
        held_out_comparisons=(causal,),
        vanilla_held_out_comparisons=(valid_vanilla,),
    )
    assert paired.vanilla_held_out_comparisons == (valid_vanilla,)


def test_scientific_gates_require_complete_authenticated_provenance() -> None:
    inputs = _accepted_inputs()
    unaudited = tuple(
        HeldOutComparison(
            item.scenario_id,
            policy_score=item.policy_score,
            permanent_no_op_score=item.permanent_no_op_score,
        )
        for item in inputs.held_out_comparisons
    )
    held_out_check = evaluate_phase0_acceptance(
        _accepted_rulebook(),
        replace(inputs, held_out_comparisons=unaudited),
    ).check("held_out_beats_permanent_no_op")
    assert not held_out_check.passed
    assert held_out_check.metrics["provenance_complete"] is False

    path_check = evaluate_phase0_acceptance(
        _accepted_rulebook(),
        replace(
            inputs,
            path_stretch_ablation=PathStretchAblationResult(
                PATH_STRETCH_ACTION_KEY,
                oracle_score=1.2,
                geometry_only_score=0.8,
            ),
        ),
    ).check("path_stretch_oracle_and_geometry_ablation")
    assert not path_check.passed
    assert path_check.metrics["runtime_provenance_authenticated"] is False


@pytest.mark.parametrize("value", (2.0, True, "2"))
def test_refresh_interval_requires_an_exact_integer(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RefreshIntervalResult(
            value,  # type: ignore[arg-type]
            {"regime": _speed_action()},
            (2,),
        )


def test_refresh_robustness_requires_real_rulebook_publications() -> None:
    inputs = _accepted_inputs()
    unpublished = tuple(
        replace(result, publication_generations=())
        for result in inputs.refresh_interval_results
    )

    check = evaluate_phase0_acceptance(
        _accepted_rulebook(),
        replace(inputs, refresh_interval_results=unpublished),
    ).check("refresh_interval_robustness")

    assert not check.passed
    assert check.metrics["publication_backed"] is False
    assert check.metrics["minimum_publication_count"] == 0
    assert check.metrics["minimum_latest_generation"] == 0


def test_seed_comparison_rejects_diagonal_vs_aligned_joint_geometry() -> None:
    speed = "required_delay_over_speed_capacity"
    path = "required_delay_over_path_capacity"
    snapshots = (
        SeededRuleRegions(
            17,
            (
                _rectangle(_speed_action(), **{speed: (0.0, 1.0), path: (0.0, 1.0)}),
                _rectangle(_speed_action(), **{speed: (9.0, 10.0), path: (9.0, 10.0)}),
            ),
        ),
        SeededRuleRegions(
            29,
            (
                _rectangle(_speed_action(), **{speed: (0.0, 1.0), path: (9.0, 10.0)}),
                _rectangle(_speed_action(), **{speed: (9.0, 10.0), path: (0.0, 1.0)}),
            ),
        ),
    )

    check = (
        Phase0AcceptanceEvaluator(minimum_region_iou=0.5)
        .evaluate(
            _accepted_rulebook(),
            replace(_accepted_inputs(), seeded_rule_regions=snapshots),
        )
        .check("seeded_rule_region_comparability")
    )

    assert not check.passed
    assert check.metrics["joint_geometry_comparable"] is True
    assert check.metrics["minimum_pairwise_mean_iou"] == 0.0


def test_seed_comparison_accepts_genuinely_overlapping_hyperrectangles() -> None:
    speed = "required_delay_over_speed_capacity"
    path = "required_delay_over_path_capacity"
    snapshots = (
        SeededRuleRegions(
            17,
            (
                _rectangle(_speed_action(), **{speed: (0.0, 2.0), path: (0.0, 2.0)}),
                _rectangle(_speed_action(), **{speed: (5.0, 7.0), path: (5.0, 7.0)}),
            ),
        ),
        SeededRuleRegions(
            29,
            (
                _rectangle(_speed_action(), **{speed: (0.2, 2.2), path: (0.2, 2.2)}),
                _rectangle(_speed_action(), **{speed: (5.2, 7.2), path: (5.2, 7.2)}),
            ),
        ),
    )

    check = (
        Phase0AcceptanceEvaluator(minimum_region_iou=0.5)
        .evaluate(
            _accepted_rulebook(),
            replace(_accepted_inputs(), seeded_rule_regions=snapshots),
        )
        .check("seeded_rule_region_comparability")
    )

    assert check.passed
    assert check.metrics["minimum_pairwise_mean_iou"] > 0.5


def test_seed_comparison_handles_ten_same_key_rectangles_without_factorial_matching() -> (
    None
):
    speed = "required_delay_over_speed_capacity"
    path = "required_delay_over_path_capacity"
    rectangles = tuple(
        _rectangle(
            _speed_action(),
            **{speed: (float(index), index + 0.8), path: (float(index), index + 0.8)},
        )
        for index in range(10)
    )
    snapshots = (
        SeededRuleRegions(17, rectangles),
        SeededRuleRegions(29, tuple(reversed(rectangles))),
    )

    check = (
        Phase0AcceptanceEvaluator(minimum_region_iou=0.99)
        .evaluate(
            _accepted_rulebook(),
            replace(_accepted_inputs(), seeded_rule_regions=snapshots),
        )
        .check("seeded_rule_region_comparability")
    )

    assert check.passed
    assert check.metrics["minimum_pairwise_mean_iou"] == 1.0


def test_capacity_check_rejects_distance_confound_even_with_capacity_ratio() -> None:
    accepted = _accepted_rulebook()
    confounded_speed = replace(
        accepted.action_rules[1],
        condition=RuleCondition(
            "leader_follower",
            SCHEMA_HASH,
            {
                "required_delay_over_speed_capacity": (0.0, 1.0),
                "follower_distance_to_resource_m": (0.0, 30_000.0),
            },
        ),
    )
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        certification_generation=accepted.certification_generation,
        action_rules=(accepted.action_rules[0], confounded_speed),
        action_configuration=accepted.action_configuration,
    )

    report = Phase0AcceptanceEvaluator().evaluate(
        rulebook,
        _accepted_inputs(),
    )

    check = report.check("capacity_ratio_predicates")
    assert not check.passed
    assert check.metrics["action_rule_count"] == 2
    assert check.metrics["capacity_ratio_rule_count"] == 2
    assert check.metrics["distance_confound_rule_count"] == 1
    assert check.metrics["distance_only_rule_count"] == 0


@pytest.mark.parametrize(
    ("noop_samples", "noop_lcb", "expected"),
    (
        pytest.param(29, 0.1, False, id="below-sample-gate"),
        pytest.param(30, 0.0, False, id="zero-lcb"),
        pytest.param(30, -0.1, False, id="negative-lcb"),
        pytest.param(30, 1.0e-9, True, id="positive-boundary"),
    ),
)
def test_certification_gate_requires_threshold_samples_and_positive_lcb(
    noop_samples: int,
    noop_lcb: float,
    expected: bool,
) -> None:
    inputs = _accepted_inputs()
    evidence = dict(inputs.action_certification_evidence)
    evidence["speed-rule"] = ActionCertificationEvidence(
        "speed-rule",
        noop_samples=noop_samples,
        noop_lcb=noop_lcb,
    )

    report = evaluate_phase0_acceptance(
        _accepted_rulebook(),
        replace(inputs, action_certification_evidence=evidence),
    )

    assert report.check("positive_noop_grounded_certification").passed is expected


def test_certification_gate_rejects_missing_stale_and_configured_threshold() -> None:
    inputs = _accepted_inputs()
    evidence = dict(inputs.action_certification_evidence)
    evidence.pop("speed-rule")
    missing = evaluate_phase0_acceptance(
        _accepted_rulebook(),
        replace(inputs, action_certification_evidence=evidence),
    ).check("positive_noop_grounded_certification")
    assert not missing.passed
    assert missing.metrics["missing_evidence_count"] == 1

    stale_evidence = _accepted_certification_evidence()
    stale_evidence["removed-rule"] = ActionCertificationEvidence(
        "removed-rule",
        noop_samples=100,
        noop_lcb=1.0,
    )
    stale = evaluate_phase0_acceptance(
        _accepted_rulebook(),
        replace(inputs, action_certification_evidence=stale_evidence),
    ).check("positive_noop_grounded_certification")
    assert not stale.passed
    assert stale.metrics["unexpected_evidence_count"] == 1

    configured = evaluate_phase0_acceptance(
        _accepted_rulebook(),
        inputs,
        action_min_noop_samples=32,
    ).check("positive_noop_grounded_certification")
    assert not configured.passed
    assert configured.metrics["required_noop_samples"] == 32
    assert configured.metrics["insufficient_sample_count"] == 2


def test_acceptance_report_surfaces_each_failed_scientific_gate() -> None:
    distance_only_negative = FrozenActionRule(
        source_rule_id="confounded-rule",
        condition=RuleCondition(
            "leader_follower",
            SCHEMA_HASH,
            {"follower_distance_to_resource_m": (0.0, 30_000.0)},
        ),
        action=_speed_action(),
        w=-0.2,
        precision=1.0,
    )
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        action_rules=(distance_only_negative,),
        action_configuration={
            "actions": [{"lever": "no_op", "band": "no_op"}],
        },
    )
    inputs = Phase0ValidationInputs(
        train_deploy_decisions=(
            TrainDeployDecision(
                "disagree",
                training_action=_speed_action(),
                deployment_action=RuleAction.no_op(),
                exploration_rate=0.1,
            ),
        ),
        held_out_comparisons=(
            HeldOutComparison(
                "worse",
                policy_score=0.0,
                permanent_no_op_score=1.0,
            ),
        ),
        path_stretch_ablation=PathStretchAblationResult(
            action_key="path_stretch/geometry_only",
            oracle_score=0.5,
            geometry_only_score=None,
        ),
        refresh_interval_results=(
            RefreshIntervalResult(
                100,
                {"small": _speed_action(), "large": _path_action()},
                (100,),
            ),
            RefreshIntervalResult(
                500,
                {"small": _path_action(), "large": _speed_action()},
                (500,),
            ),
        ),
        seeded_rule_regions=(
            SeededRuleRegions(
                1,
                (
                    _rectangle(
                        _speed_action(),
                        required_delay_over_speed_capacity=(0.0, 1.0),
                    ),
                ),
            ),
            SeededRuleRegions(
                2,
                (
                    _rectangle(
                        _speed_action(),
                        required_delay_over_speed_capacity=(2.0, 3.0),
                    ),
                ),
            ),
        ),
    )

    report = Phase0AcceptanceEvaluator(
        minimum_region_iou=0.5,
    ).evaluate(rulebook, inputs)

    assert not report.passed
    assert {check.name for check in report.failures} == {
        "exact_five_action_vocabulary",
        "capacity_ratio_predicates",
        "positive_noop_grounded_certification",
        "exploit_deployment_agreement",
        "held_out_beats_permanent_no_op",
        "path_stretch_oracle_and_geometry_ablation",
        "refresh_interval_robustness",
        "seeded_rule_region_comparability",
    }
    assert (
        report.check("capacity_ratio_predicates").metrics["distance_only_rule_count"]
        == 1
    )
    assert (
        report.check("capacity_ratio_predicates").metrics[
            "distance_confound_rule_count"
        ]
        == 1
    )
    assert (
        report.check("seeded_rule_region_comparability").metrics[
            "minimum_pairwise_mean_iou"
        ]
        == 0.0
    )


def test_evidence_records_and_report_metrics_are_defensively_immutable() -> None:
    inputs = _accepted_inputs()
    refresh = inputs.refresh_interval_results[0]
    with pytest.raises(TypeError):
        refresh.regime_actions["new"] = _speed_action()  # type: ignore[index]

    with pytest.raises(TypeError):
        inputs.action_certification_evidence["new"] = (  # type: ignore[index]
            ActionCertificationEvidence("new", 30, 0.1)
        )

    report = evaluate_phase0_acceptance(_accepted_rulebook(), inputs)
    with pytest.raises(TypeError):
        report.check("capacity_ratio_predicates").metrics["action_rule_count"] = 999  # type: ignore[index]

    altered = replace(
        inputs,
        held_out_comparisons=(),
        path_stretch_ablation=None,
    )
    failed = evaluate_phase0_acceptance(_accepted_rulebook(), altered)
    assert not failed.check("held_out_beats_permanent_no_op").passed
    assert not failed.check("path_stretch_oracle_and_geometry_ablation").passed
