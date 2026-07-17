from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hailmary.learning as learning_api
from hailmary.actions.models import ActionLever
from hailmary.features.schema import FeatureField, FeatureSchema
from hailmary.ids import content_hash
from hailmary.learning.artifacts import (
    EPOCH_TRACE_VERSION,
    EVALUATION_SNAPSHOT_VERSION,
    EXPORTED_RULEBOOK_VERSION,
    PUBLISHED_CERTIFICATION_EVIDENCE_VERSION,
    TRAINING_CHECKPOINT_VERSION,
    EpochTrace,
    EvaluationSnapshot,
    ExportedRulebook,
    PublishedCertificationEvidence,
    TrainingCheckpoint,
    artifact_from_dict,
    capture_rng_state,
    load_artifact,
    load_epoch_trace,
    load_evaluation_snapshot,
    load_exported_rulebook,
    load_training_checkpoint,
    restore_rng_state,
    save_artifact,
    save_epoch_trace,
    save_evaluation_snapshot,
    save_training_checkpoint,
    save_exported_rulebook,
)
from hailmary.learning.certification import (
    CertificationThresholds,
    FrozenActionRule,
    FrozenVetoRule,
)
from hailmary.learning.conditions import Interval, RuleCondition
from hailmary.learning.population import POPULATION_VERSION, Population
from hailmary.learning.rulebook import (
    FROZEN_RULEBOOK_VERSION,
    FrozenRulebookPolicy,
    RulebookDecisionRecord,
)
from hailmary.learning.rules import MutableRule, RuleAction
from hailmary.learning.statistics import OnlineMoments


SCHEMA = FeatureSchema(
    "test.learning.artifacts.v1",
    (FeatureField("x", lower_bound=0.0, upper_bound=10.0),),
)
SPEED = RuleAction(ActionLever.SPEED, "light")
SCENARIO_HASH = "scenario-definition-hash"
CONFIG_HASH = "learning-config-hash"


def _rule() -> MutableRule:
    return MutableRule(
        condition=RuleCondition(
            "leader_follower",
            SCHEMA.schema_hash,
            {"x": Interval(1.0, 9.0)},
        ),
        action=SPEED,
        evolution=OnlineMoments(n=3, mean=1.0, m2=0.5),
        deployment=OnlineMoments(n=4, mean=2.0, m2=0.75),
        provenance={"fixture": "artifact"},
    )


def _evaluation_snapshot(
    ranking: dict[str, object] | None = None,
) -> EvaluationSnapshot:
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA.schema_hash,
        certification_generation=12,
    )
    return EvaluationSnapshot(
        rulebook,
        publication_epoch=12,
        contender_ranking_fields=ranking
        or {
            "order": ["lcb_desc", "rule_id"],
            "min_samples": 2,
        },
        config_hash=CONFIG_HASH,
        action_vocabulary_hash=rulebook.action_config_hash,
    )


def _published_evidence() -> PublishedCertificationEvidence:
    return PublishedCertificationEvidence(
        source_rule_id="published-speed",
        credit_mode="causal",
        rule_kind="action",
        action=SPEED,
        independent_evolution=OnlineMoments(n=2, mean=0.5, m2=0.02),
        independent_deployment=OnlineMoments(n=3, mean=2.0, m2=0.2),
        thresholds=CertificationThresholds(
            action_min_noop_samples=2,
            action_lcb_z=0.5,
        ),
    )


def _exported_rulebook() -> ExportedRulebook:
    condition = _rule().condition
    policy = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA.schema_hash,
        action_rules=(
            FrozenActionRule(
                source_rule_id="population-speed-source",
                condition=condition,
                action=SPEED,
                w=2.5,
                precision=4.0,
                rival_lcb=91.0,
                rival_samples=700,
            ),
        ),
        veto_rules=(
            FrozenVetoRule(
                source_rule_id="population-veto-source",
                condition=RuleCondition(
                    "leader_follower",
                    SCHEMA.schema_hash,
                    {"x": Interval(9.5, 10.0)},
                ),
            ),
        ),
    )
    return ExportedRulebook(policy)


def _rehash(payload: dict[str, object]) -> None:
    payload.pop("content_hash", None)
    payload["content_hash"] = content_hash(
        payload,
        namespace=str(payload["artifact_version"]),
    )


def _checkpoint(
    *,
    rule: MutableRule | None = None,
    simulator_snapshot: dict[str, object] | None = None,
    rng_state: dict[str, object] | None = None,
) -> TrainingCheckpoint:
    population = Population((_rule() if rule is None else rule,), max_size=20)
    evaluation = _evaluation_snapshot()
    state = (
        {
            "definition_hash": SCENARIO_HASH,
            "state_id": "root-state",
            "nested": {"events": [1, 2, 3]},
        }
        if simulator_snapshot is None
        else simulator_snapshot
    )
    generator = np.random.default_rng(44)
    return TrainingCheckpoint(
        scenario_definition_hash=SCENARIO_HASH,
        simulator_snapshot=state,
        population=population,
        evaluation_snapshot=evaluation,
        epoch=72,
        rng_state=capture_rng_state(generator) if rng_state is None else rng_state,
        exploration_counts={"global": 7, "sparse-region": 2},
    )


def _trace() -> EpochTrace:
    return EpochTrace(
        scenario_definition_hash=SCENARIO_HASH,
        epoch=72,
        root_parent_hash="root-content-hash",
        committed_state_hash="committed-content-hash",
        outcome_plan_hash="outcome-plan-hash",
        selected_anchor_id="UAL88/AAL12",
        selected_action=SPEED,
        contender_action=RuleAction(
            ActionLever.PATH_STRETCH,
            "oracle_short_medium_long",
        ),
        no_op_action=RuleAction.no_op(),
        committed_action=SPEED,
        matched_rule_ids=("rule-a", "rule-b"),
        coadvocate_rule_ids=("rule-a",),
        exploration_reason="coverage-floor",
        contender_selection={
            "ranking_fields": _evaluation_snapshot().contender_ranking_fields,
            "ranked_candidates": [
                {
                    "rank": 1,
                    "action": RuleAction(
                        ActionLever.PATH_STRETCH,
                        "oracle_short_medium_long",
                    ).to_dict(),
                    "score": 0.75,
                    "source_rule_ids": ["rule-b"],
                }
            ],
            "selected_rank": 1,
            "fallback": None,
        },
        rulebook_hash="rulebook-content-hash",
        arm_initial_hashes={
            "selected": "root-content-hash",
            "contender": "root-content-hash",
            "no_op": "root-content-hash",
        },
        arm_final_hashes={
            "selected": "selected-final",
            "contender": "contender-final",
            "no_op": "noop-final",
        },
        arm_scores={"selected": 1.49, "contender": 0.82, "no_op": 0.11},
        delta_rival=0.67,
        delta_selected_noop=1.38,
        delta_contender_noop=0.71,
        delta_veto=-1.38,
        ledger_recipients={
            "evolution": ["rule-a"],
            "deployment": ["rule-a", "rule-b"],
        },
        ga_events=({"operator": "crossover", "offspring_ids": ["child-a"]},),
        certification_events=({"generation": 12, "published": True},),
        config_hash=CONFIG_HASH,
        action_vocabulary_hash=_evaluation_snapshot().action_vocabulary_hash,
        feature_schema_hash=SCHEMA.schema_hash,
    )


def test_evaluation_snapshot_is_detached_versioned_and_canonical() -> None:
    ranking: dict[str, object] = {
        "order": ["lcb_desc", "rule_id"],
        "nested": {"weights": [1, 2]},
    }
    snapshot = _evaluation_snapshot(ranking)
    before = snapshot.to_dict()

    ranking["order"].append("mutated")  # type: ignore[union-attr]
    nested = ranking["nested"]
    assert isinstance(nested, dict)
    nested["weights"].append(3)  # type: ignore[union-attr]

    assert snapshot.artifact_version == EVALUATION_SNAPSHOT_VERSION
    with pytest.raises(AttributeError, match="immutable"):
        snapshot.config_hash = "mutated"
    assert snapshot.to_dict() == before
    assert snapshot.rulebook.content_hash == snapshot.rulebook_hash
    assert (
        EvaluationSnapshot.from_dict(snapshot.to_dict()).to_dict() == snapshot.to_dict()
    )
    assert EvaluationSnapshot.from_json(snapshot.to_json()) == snapshot


def test_published_evidence_is_strict_detached_and_content_hashed() -> None:
    assert (
        learning_api.PUBLISHED_CERTIFICATION_EVIDENCE_VERSION
        == PUBLISHED_CERTIFICATION_EVIDENCE_VERSION
    )
    assert learning_api.PublishedCertificationEvidence is PublishedCertificationEvidence
    evolution = OnlineMoments(n=2, mean=0.5, m2=0.02)
    deployment = OnlineMoments(n=3, mean=2.0, m2=0.2)
    thresholds = CertificationThresholds(
        action_min_noop_samples=2,
        action_lcb_z=0.5,
    )
    evidence = PublishedCertificationEvidence(
        source_rule_id="published-speed",
        credit_mode="causal",
        rule_kind="action",
        action=SPEED,
        independent_evolution=evolution,
        independent_deployment=deployment,
        thresholds=thresholds,
    )
    before = evidence.to_dict()

    evolution.update(99.0)
    deployment.update(-99.0)
    assert evidence.to_dict() == before
    assert evidence.noop_samples == 3
    assert evidence.noop_lcb == pytest.approx(
        OnlineMoments(n=3, mean=2.0, m2=0.2).lcb(
            thresholds.action_lcb_z,
            thresholds.variance_floor,
        )
    )
    assert PublishedCertificationEvidence.from_dict(before) == evidence
    assert PublishedCertificationEvidence.from_json(evidence.to_json()) == evidence
    with pytest.raises(AttributeError, match="immutable"):
        evidence.credit_mode = "vanilla_accuracy"

    tampered = evidence.to_dict()
    deployment_payload = tampered["independent_deployment"]
    assert isinstance(deployment_payload, dict)
    deployment_payload["mean"] = 4.0
    with pytest.raises(ValueError, match="content hash"):
        PublishedCertificationEvidence.from_dict(tampered)

    malformed = evidence.to_dict()
    malformed.pop("independent_evolution")
    _rehash(malformed)
    with pytest.raises(ValueError, match="exactly match"):
        PublishedCertificationEvidence.from_dict(malformed)

    malformed_thresholds = evidence.to_dict()
    threshold_payload = malformed_thresholds["thresholds"]
    assert isinstance(threshold_payload, dict)
    threshold_payload["unexpected"] = 1
    _rehash(malformed_thresholds)
    with pytest.raises(ValueError, match="threshold payload fields"):
        PublishedCertificationEvidence.from_dict(malformed_thresholds)


def test_exported_rulebook_artifact_is_available_from_public_learning_api() -> None:
    assert learning_api.EXPORTED_RULEBOOK_VERSION == EXPORTED_RULEBOOK_VERSION
    assert learning_api.ExportedRulebook is ExportedRulebook
    assert learning_api.FROZEN_RULEBOOK_VERSION == FROZEN_RULEBOOK_VERSION
    assert learning_api.POPULATION_VERSION == POPULATION_VERSION
    assert learning_api.save_exported_rulebook is save_exported_rulebook
    assert learning_api.load_exported_rulebook is load_exported_rulebook


def test_exported_rulebook_contains_only_inference_fields_and_is_equivalent() -> None:
    exported = _exported_rulebook()
    payload = exported.to_dict()

    assert exported.artifact_version == EXPORTED_RULEBOOK_VERSION
    assert set(payload["action_rules"][0]) == {
        "condition",
        "action",
        "w",
        "precision",
    }
    assert set(payload["veto_rules"][0]) == {"condition", "veto"}
    serialized = exported.to_json()
    assert "source_rule_id" not in serialized
    assert "rival_lcb" not in serialized
    assert "rival_samples" not in serialized
    assert "contender_min_samples" not in serialized
    assert ExportedRulebook.from_dict(payload) == exported
    assert ExportedRulebook.from_json(serialized) == exported

    no_op = RuleAction.no_op()
    ordinary = RulebookDecisionRecord(
        anchor_id="ordinary",
        features={"x": 5.0},
        candidates=(no_op, SPEED),
        schema_hash=SCHEMA.schema_hash,
    )
    vetoed = RulebookDecisionRecord(
        anchor_id="vetoed",
        features={"x": 9.75},
        candidates=(no_op, SPEED),
        schema_hash=SCHEMA.schema_hash,
    )
    assert exported.decide((ordinary,)) == SPEED
    assert exported.decide((vetoed,)) == no_op
    with pytest.raises(AttributeError, match="immutable"):
        exported.role_type = "mutated"


def test_exported_vanilla_rulebook_preserves_learned_no_op_semantics() -> None:
    condition = _rule().condition
    no_op = RuleAction.no_op()
    vanilla = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA.schema_hash,
        credit_mode="vanilla_accuracy",
        action_rules=(
            FrozenActionRule(
                source_rule_id="vanilla-no-op",
                condition=condition,
                action=no_op,
                w=2.0,
                precision=0.75,
                rival_lcb=-100.0,
                rival_samples=30,
            ),
            FrozenActionRule(
                source_rule_id="vanilla-speed",
                condition=condition,
                action=SPEED,
                w=-1.0,
                precision=1.0,
                rival_lcb=100.0,
                rival_samples=30,
            ),
        ),
    )
    exported = ExportedRulebook(vanilla)
    payload = exported.to_dict()
    restored = ExportedRulebook.from_dict(payload)
    record = RulebookDecisionRecord(
        anchor_id="vanilla",
        features={"x": 5.0},
        candidates=(no_op, SPEED),
        schema_hash=SCHEMA.schema_hash,
    )

    assert exported.credit_mode == "vanilla_accuracy"
    assert payload["credit_mode"] == "vanilla_accuracy"
    assert restored.rulebook.credit_mode == "vanilla_accuracy"
    assert restored.decide((record,)) == no_op
    assert restored.content_hash == exported.content_hash

    malformed = {**payload, "credit_mode": "causal"}
    _rehash(malformed)
    with pytest.raises(ValueError, match="scoped vetoes"):
        ExportedRulebook.from_dict(malformed)


def test_evaluation_snapshot_rejects_redundant_hash_or_membership_mismatch() -> None:
    frozen = FrozenActionRule(
        source_rule_id="certified-speed",
        condition=_rule().condition,
        action=SPEED,
        w=1.0,
        precision=2.0,
    )
    evidence = PublishedCertificationEvidence(
        source_rule_id=frozen.source_rule_id,
        credit_mode="causal",
        rule_kind="action",
        action=SPEED,
        independent_evolution=OnlineMoments(),
        independent_deployment=OnlineMoments(n=2, mean=1.0, m2=1.0),
        thresholds=CertificationThresholds(
            action_min_noop_samples=2,
            action_lcb_z=0.0,
            contender_lcb_z=0.0,
        ),
    )
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA.schema_hash,
        action_rules=(frozen,),
    )
    snapshot = EvaluationSnapshot(
        rulebook,
        config_hash=CONFIG_HASH,
        certified_source_rule_ids=(frozen.source_rule_id,),
        action_vocabulary_hash=rulebook.action_config_hash,
        certification_evidence={frozen.source_rule_id: evidence},
    )
    assert snapshot.certified_source_rule_ids == (frozen.source_rule_id,)

    with pytest.raises(ValueError, match="certification evidence keys"):
        EvaluationSnapshot(
            rulebook,
            config_hash=CONFIG_HASH,
            action_vocabulary_hash=rulebook.action_config_hash,
        )

    with pytest.raises(ValueError, match="action vocabulary"):
        EvaluationSnapshot(
            rulebook,
            config_hash=CONFIG_HASH,
            action_vocabulary_hash="different-action-vocabulary",
        )
    with pytest.raises(ValueError, match="exactly match"):
        EvaluationSnapshot(
            rulebook,
            config_hash=CONFIG_HASH,
            certified_source_rule_ids=(),
        )
    with pytest.raises(ValueError, match="exactly match"):
        EvaluationSnapshot(
            rulebook,
            config_hash=CONFIG_HASH,
            certified_source_rule_ids=(frozen.source_rule_id, "extra-rule"),
            certification_evidence={frozen.source_rule_id: evidence},
        )

    vanilla_rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA.schema_hash,
        credit_mode="vanilla_accuracy",
        action_rules=(frozen,),
    )
    with pytest.raises(ValueError, match="credit_mode"):
        EvaluationSnapshot(
            vanilla_rulebook,
            config_hash=CONFIG_HASH,
            certification_evidence={frozen.source_rule_id: evidence},
        )

    tampered = snapshot.to_dict()
    evidence_payloads = tampered["certification_evidence"]
    assert isinstance(evidence_payloads, dict)
    evidence_payload = evidence_payloads[frozen.source_rule_id]
    assert isinstance(evidence_payload, dict)
    deployment_payload = evidence_payload["independent_deployment"]
    assert isinstance(deployment_payload, dict)
    deployment_payload["mean"] = 1.25
    _rehash(evidence_payload)
    _rehash(tampered)
    with pytest.raises(ValueError, match="reproduce frozen rule weight"):
        EvaluationSnapshot.from_dict(tampered)

    missing_key = snapshot.to_dict()
    missing_key["certification_evidence"] = {}
    _rehash(missing_key)
    with pytest.raises(ValueError, match="certification evidence keys"):
        EvaluationSnapshot.from_dict(missing_key)


def test_checkpoint_captures_detached_population_simulator_and_hashes() -> None:
    rule = _rule()
    simulator_snapshot: dict[str, object] = {
        "definition_hash": SCENARIO_HASH,
        "state_id": "root-state",
        "nested": {"events": [1, 2, 3]},
    }
    checkpoint = _checkpoint(rule=rule, simulator_snapshot=simulator_snapshot)
    captured_evolution_n = checkpoint.population.rules[0].evolution.n

    rule.evolution.update(99.0)
    nested = simulator_snapshot["nested"]
    assert isinstance(nested, dict)
    nested["events"].append(4)  # type: ignore[union-attr]

    assert checkpoint.artifact_version == TRAINING_CHECKPOINT_VERSION
    assert checkpoint.scenario_hash == SCENARIO_HASH
    assert checkpoint.config_hash == CONFIG_HASH
    assert checkpoint.feature_schema_hash == SCHEMA.schema_hash
    assert checkpoint.population.rules[0].evolution.n == captured_evolution_n
    assert checkpoint.simulator_snapshot["nested"]["events"] == [1, 2, 3]

    restored_population = checkpoint.population
    restored_population.rules[0].evolution.update(-100.0)
    assert checkpoint.population.rules[0].evolution.n == captured_evolution_n

    assert TrainingCheckpoint.from_dict(checkpoint.to_dict()) == checkpoint
    assert TrainingCheckpoint.from_json(checkpoint.to_json()) == checkpoint


def test_rng_state_round_trip_restores_exact_sequence() -> None:
    generator = np.random.default_rng(17)
    state = capture_rng_state(generator)
    expected = generator.integers(0, 1_000_000, size=12)

    restored = restore_rng_state(np.random.default_rng(999), state)
    actual = restored.integers(0, 1_000_000, size=12)

    np.testing.assert_array_equal(actual, expected)

    checkpoint = _checkpoint(rng_state=dict(state))
    via_checkpoint = checkpoint.restore_rng(np.random.default_rng(1))
    np.testing.assert_array_equal(
        via_checkpoint.integers(0, 1_000_000, size=12),
        expected,
    )

    trainer_generator = np.random.default_rng(29)
    trainer_state = capture_rng_state(trainer_generator)
    trainer_expected = trainer_generator.integers(0, 1_000_000, size=12)
    composite = _checkpoint(
        rng_state={
            "schema_version": "hailmary.causal_trainer.v1",
            "trainer_rng": trainer_state,
            "exploration_scheduler": {"visits": []},
        }
    )
    composite_restored = composite.restore_rng(np.random.default_rng(1))
    np.testing.assert_array_equal(
        composite_restored.integers(0, 1_000_000, size=12),
        trainer_expected,
    )


def test_epoch_trace_round_trip_records_all_three_arm_evidence() -> None:
    trace = _trace()

    assert trace.artifact_version == EPOCH_TRACE_VERSION
    assert trace.arm_scores == {
        "contender": 0.82,
        "no_op": 0.11,
        "selected": 1.49,
    }
    assert trace.selected_action == {"band": "light", "lever": "speed"}
    assert trace.delta_rival == pytest.approx(0.67)
    assert trace.delta_selected_noop == pytest.approx(1.38)
    assert trace.delta_contender_noop == pytest.approx(0.71)
    assert trace.delta_veto == pytest.approx(-1.38)
    assert trace.contender_selection["selected_rank"] == 1
    assert trace.contender_selection["fallback"] is None
    assert trace.contender_selection["ranked_candidates"][0]["action"] == {
        "lever": "path_stretch",
        "band": "oracle_short_medium_long",
    }
    assert EpochTrace.from_dict(trace.to_dict()) == trace
    assert EpochTrace.from_json(trace.to_json()) == trace


def test_epoch_trace_rejects_incomplete_or_inconsistent_arm_evidence() -> None:
    incomplete = _trace().to_dict()
    incomplete["arm_scores"].pop("no_op")
    _rehash(incomplete)
    with pytest.raises(ValueError, match="exact selected/contender/no_op"):
        EpochTrace.from_dict(incomplete)

    wrong_root = _trace().to_dict()
    wrong_root["arm_initial_hashes"]["selected"] = "different-root"
    _rehash(wrong_root)
    with pytest.raises(ValueError, match="root_parent_hash"):
        EpochTrace.from_dict(wrong_root)

    wrong_delta = _trace().to_dict()
    wrong_delta["delta_rival"] = 99.0
    _rehash(wrong_delta)
    with pytest.raises(ValueError, match="delta_rival"):
        EpochTrace.from_dict(wrong_delta)

    missing_required = _trace().to_dict()
    missing_required["outcome_plan_hash"] = None
    _rehash(missing_required)
    with pytest.raises(ValueError, match="completed traces require outcome_plan_hash"):
        EpochTrace.from_dict(missing_required)

    wrong_commit = _trace().to_dict()
    wrong_commit["committed_action"] = RuleAction(ActionLever.SPEED, "medium").to_dict()
    _rehash(wrong_commit)
    with pytest.raises(ValueError, match="exactly equal selected_action"):
        EpochTrace.from_dict(wrong_commit)

    wrong_arm_c = _trace().to_dict()
    wrong_arm_c["no_op_action"] = SPEED.to_dict()
    _rehash(wrong_arm_c)
    with pytest.raises(ValueError, match="Arm C action"):
        EpochTrace.from_dict(wrong_arm_c)

    wrong_rank = _trace().to_dict()
    wrong_rank["contender_selection"]["ranked_candidates"][0]["action"] = (
        RuleAction.no_op().to_dict()
    )
    _rehash(wrong_rank)
    with pytest.raises(ValueError, match="does not match contender_action"):
        EpochTrace.from_dict(wrong_rank)

    invalid_fallback = _trace().to_dict()
    invalid_fallback["contender_selection"] = {
        "ranking_fields": {"score": "rival_lcb"},
        "ranked_candidates": [],
        "selected_rank": None,
        "fallback": "mandatory_no_op_no_certified_rival",
    }
    _rehash(invalid_fallback)
    with pytest.raises(ValueError, match="fallback contender"):
        EpochTrace.from_dict(invalid_fallback)

    mismatched_duplicate_arm = _trace().to_dict()
    mismatched_duplicate_arm["contender_action"] = RuleAction.no_op().to_dict()
    mismatched_duplicate_arm["contender_selection"] = {
        "ranking_fields": {"score": "rival_lcb"},
        "ranked_candidates": [],
        "selected_rank": None,
        "fallback": "mandatory_no_op_no_certified_rival",
    }
    _rehash(mismatched_duplicate_arm)
    with pytest.raises(ValueError, match="identical-action arms"):
        EpochTrace.from_dict(mismatched_duplicate_arm)

    skipped = EpochTrace(
        scenario_definition_hash=SCENARIO_HASH,
        epoch=1,
        root_parent_hash="root",
        rulebook_hash="rulebook",
        config_hash=CONFIG_HASH,
        action_vocabulary_hash="actions",
        feature_schema_hash=SCHEMA.schema_hash,
    )
    assert skipped.arm_scores == {}
    with pytest.raises(ValueError, match="zero deltas"):
        EpochTrace(
            scenario_definition_hash=SCENARIO_HASH,
            epoch=1,
            root_parent_hash="root",
            rulebook_hash="rulebook",
            delta_veto=1.0,
            config_hash=CONFIG_HASH,
            action_vocabulary_hash="actions",
            feature_schema_hash=SCHEMA.schema_hash,
        )


def test_serialized_artifacts_and_rulebooks_require_content_hash() -> None:
    artifacts = (
        _published_evidence(),
        _exported_rulebook(),
        _evaluation_snapshot(),
        _checkpoint(),
        _trace(),
    )
    for artifact in artifacts:
        payload = artifact.to_dict()
        payload.pop("content_hash")
        with pytest.raises(ValueError, match="requires content_hash"):
            artifact_from_dict(payload)

    rulebook_payload = _exported_rulebook().rulebook.to_dict()
    rulebook_payload.pop("content_hash")
    with pytest.raises(ValueError, match="requires content_hash"):
        FrozenRulebookPolicy.from_dict(rulebook_payload)


@pytest.mark.parametrize(
    "artifact",
    [_exported_rulebook(), _evaluation_snapshot(), _checkpoint(), _trace()],
)
def test_artifact_content_hash_detects_tampering(
    artifact: ExportedRulebook | EvaluationSnapshot | TrainingCheckpoint | EpochTrace,
) -> None:
    payload = artifact.to_dict()
    payload["feature_schema_hash"] = "tampered"

    with pytest.raises(ValueError, match="content hash"):
        artifact_from_dict(payload)


@pytest.mark.parametrize("invalid_count", [1.0, "1", True, -1])
def test_checkpoint_exploration_counts_require_exact_nonnegative_ints(
    invalid_count: object,
) -> None:
    payload = _checkpoint().to_dict()
    payload["exploration_counts"] = {"global": invalid_count}
    _rehash(payload)

    with pytest.raises(ValueError, match="non-negative integers"):
        TrainingCheckpoint.from_dict(payload)


@pytest.mark.parametrize(
    ("artifact", "expected_type"),
    [
        (_published_evidence(), PublishedCertificationEvidence),
        (_exported_rulebook(), ExportedRulebook),
        (_evaluation_snapshot(), EvaluationSnapshot),
        (_checkpoint(), TrainingCheckpoint),
        (_trace(), EpochTrace),
    ],
)
def test_generic_artifact_dispatch(
    artifact: PublishedCertificationEvidence
    | ExportedRulebook
    | EvaluationSnapshot
    | TrainingCheckpoint
    | EpochTrace,
    expected_type: type[PublishedCertificationEvidence]
    | type[ExportedRulebook]
    | type[EvaluationSnapshot]
    | type[TrainingCheckpoint]
    | type[EpochTrace],
) -> None:
    assert isinstance(artifact_from_dict(artifact.to_dict()), expected_type)


def test_atomic_save_load_helpers_replace_without_partial_files(
    tmp_path: Path,
) -> None:
    evidence = _published_evidence()
    evaluation = _evaluation_snapshot()
    exported = _exported_rulebook()
    checkpoint = _checkpoint()
    trace = _trace()

    exported_path = tmp_path / "deployment-rulebook.json"
    evaluation_path = tmp_path / "evaluation.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    trace_path = tmp_path / "trace.json"
    evidence_path = tmp_path / "certification-evidence.json"

    assert save_artifact(evidence_path, evidence) == evidence_path
    assert save_exported_rulebook(exported_path, exported) == exported_path
    assert save_evaluation_snapshot(evaluation_path, evaluation) == evaluation_path
    assert save_training_checkpoint(checkpoint_path, checkpoint) == checkpoint_path
    assert save_epoch_trace(trace_path, trace) == trace_path

    assert load_exported_rulebook(exported_path) == exported
    assert load_evaluation_snapshot(evaluation_path) == evaluation
    assert load_training_checkpoint(checkpoint_path) == checkpoint
    assert load_epoch_trace(trace_path) == trace
    assert load_artifact(evidence_path) == evidence
    assert isinstance(load_artifact(checkpoint_path), TrainingCheckpoint)
    assert checkpoint_path.read_text(encoding="utf-8") == checkpoint.to_json()

    save_artifact(checkpoint_path, trace)
    assert load_artifact(checkpoint_path) == trace
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_rejects_unknown_versions_nonfinite_trace_and_mismatched_scenario() -> None:
    evaluation_payload = _evaluation_snapshot().to_dict()
    evaluation_payload["artifact_version"] = "hailmary.evaluation_snapshot.v999"
    with pytest.raises(ValueError, match="unsupported"):
        EvaluationSnapshot.from_dict(evaluation_payload)

    with pytest.raises(ValueError, match="finite"):
        EpochTrace(
            scenario_definition_hash=SCENARIO_HASH,
            epoch=1,
            root_parent_hash="root",
            rulebook_hash="rulebook",
            arm_scores={"selected": float("nan")},
            config_hash=CONFIG_HASH,
            action_vocabulary_hash="actions",
            feature_schema_hash=SCHEMA.schema_hash,
        )

    with pytest.raises(ValueError, match="definition hash"):
        TrainingCheckpoint(
            scenario_definition_hash=SCENARIO_HASH,
            simulator_snapshot={"definition_hash": "different"},
            population=Population((_rule(),)),
            evaluation_snapshot=_evaluation_snapshot(),
            epoch=1,
            rng_state=capture_rng_state(np.random.default_rng(1)),
        )


def test_artifact_epochs_reject_bool_fractional_and_string_values() -> None:
    artifacts = (
        (_evaluation_snapshot(), "publication_epoch"),
        (_checkpoint(), "epoch"),
        (_trace(), "epoch"),
    )
    for artifact, field_name in artifacts:
        for invalid_epoch in (True, 1.5, "1"):
            payload = artifact.to_dict()
            payload[field_name] = invalid_epoch
            _rehash(payload)
            with pytest.raises(ValueError, match="non-negative integer"):
                type(artifact).from_dict(payload)
