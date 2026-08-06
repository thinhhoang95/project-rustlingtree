from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import hailmary.features as feature_api
from hailmary.actions import ActionCatalog, ActionLever, action_vocabulary
from hailmary.features import leader_follower_feature_schema
from hailmary.ids import content_hash
from hailmary.learning import (
    CertificationThresholds,
    FrozenActionRule,
    FrozenRulebookPolicy,
    FrozenVetoRule,
    MutableRule,
    RuleAction,
    RuleCondition,
    RulebookDecisionRecord,
    SimulatorRulebookPolicy,
)
from hailmary.learning.rulebook import FROZEN_RULEBOOK_VERSION


SCHEMA_HASH = "schema-rulebook-v1"


def _condition(
    lower: float | None = None,
    upper: float | None = None,
    *,
    schema_hash: str = SCHEMA_HASH,
) -> RuleCondition:
    return RuleCondition(
        "leader_follower",
        schema_hash,
        {"x": (lower, upper)},
    )


def _frozen(
    source: str,
    action: RuleAction,
    *,
    w: float,
    precision: float,
    condition: RuleCondition | None = None,
    rival_lcb: float = 0.0,
    rival_samples: int = 2,
) -> FrozenActionRule:
    return FrozenActionRule(
        source_rule_id=source,
        condition=_condition() if condition is None else condition,
        action=action,
        w=w,
        precision=precision,
        rival_lcb=rival_lcb,
        rival_samples=rival_samples,
    )


def _record(
    anchor_id: str,
    x: float,
    *candidates: RuleAction,
    schema_hash: str = SCHEMA_HASH,
) -> RulebookDecisionRecord:
    return RulebookDecisionRecord(
        anchor_id=anchor_id,
        features={"x": x},
        candidates=candidates,
        schema_hash=schema_hash,
    )


def test_rulebook_precision_pools_and_filters_unavailable_actions() -> None:
    no_op = RuleAction.no_op()
    light = RuleAction(ActionLever.SPEED, "light")
    heavy = RuleAction(ActionLever.SPEED, "heavy")
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        action_rules=(
            _frozen("light-1", light, w=1.0, precision=1.0),
            _frozen("light-2", light, w=3.0, precision=3.0),
            _frozen("heavy", heavy, w=2.4, precision=1.0),
        ),
    )

    decision = rulebook.evaluate((_record("A", 0.5, no_op, heavy, light),))
    assert decision.candidate == light
    assert decision.score == pytest.approx(2.5)

    unavailable = rulebook.evaluate((_record("A", 0.5, no_op, heavy),))
    assert unavailable.candidate == heavy
    only_light = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        action_rules=(_frozen("light", light, w=1.0, precision=1.0),),
    )
    assert only_light.decide((_record("A", 0.5, no_op, heavy),)) == no_op


def test_vanilla_rulebook_weights_predictions_by_accuracy_without_vetoes() -> None:
    light = RuleAction(ActionLever.SPEED, "light")
    heavy = RuleAction(ActionLever.SPEED, "heavy")
    no_op = RuleAction.no_op()

    def vanilla_rule(
        rule_id: str,
        action: RuleAction,
        *,
        prediction: float,
        accuracy: float,
    ) -> MutableRule:
        rule = MutableRule(
            _condition(),
            action,
            rule_id=rule_id,
        )
        rule.evolution.extend((accuracy, accuracy))
        rule.deployment.extend((prediction, prediction))
        return rule

    rules = (
        vanilla_rule(
            "light-low-accuracy",
            light,
            prediction=2.0,
            accuracy=0.2,
        ),
        vanilla_rule(
            "light-high-accuracy",
            light,
            prediction=4.0,
            accuracy=0.8,
        ),
        vanilla_rule(
            "heavy",
            heavy,
            prediction=3.5,
            accuracy=1.0,
        ),
        vanilla_rule(
            "trained-no-op",
            no_op,
            prediction=10.0,
            accuracy=1.0,
        ),
    )
    rulebook = FrozenRulebookPolicy.from_population(
        rules,
        feature_schema_hash=SCHEMA_HASH,
        certification_generation=2,
        thresholds=CertificationThresholds(
            action_min_noop_samples=2,
            veto_min_rival_samples=2,
            contender_min_samples=2,
        ),
        credit_mode="vanilla_accuracy",
    )

    by_id = {rule.source_rule_id: rule for rule in rulebook.action_rules}
    assert rulebook.credit_mode == "vanilla_accuracy"
    assert rulebook.veto_rules == ()
    assert by_id["light-high-accuracy"].w == 4.0
    assert by_id["light-high-accuracy"].precision == pytest.approx(0.8)
    assert by_id["trained-no-op"].action == no_op
    decision = rulebook.evaluate((_record("A", 0.0, no_op, light, heavy),))
    assert decision.action == no_op
    assert decision.score == pytest.approx(10.0)
    ranked = rulebook.rank_rivals(
        _record("A", 0.0, no_op, light, heavy),
        selected_action=heavy,
    )
    assert ranked[0].action == no_op
    assert ranked[0].score == pytest.approx(10.0)


def test_vanilla_negative_predictions_use_learned_or_implicit_no_op() -> None:
    no_op = RuleAction.no_op()
    light = RuleAction(ActionLever.SPEED, "light")
    record = _record("A", 0.0, no_op, light)
    unmatched_no_op = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        credit_mode="vanilla_accuracy",
        action_rules=(_frozen("negative-light", light, w=-1.0, precision=1.0),),
    )

    fallback = unmatched_no_op.evaluate((record,))
    assert fallback.action == no_op
    assert fallback.score == 0.0

    learned_no_op = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        credit_mode="vanilla_accuracy",
        action_rules=(
            _frozen("negative-light", light, w=-1.0, precision=1.0),
            _frozen("more-negative-no-op", no_op, w=-2.0, precision=1.0),
        ),
    )
    learned = learned_no_op.evaluate((record,))
    assert learned.action == light
    assert learned.score == -1.0


def test_veto_is_anchor_local_and_global_ties_are_deterministic() -> None:
    no_op = RuleAction.no_op()
    light = RuleAction("speed", "light")
    heavy = RuleAction("speed", "heavy")
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        action_rules=(
            _frozen("light", light, w=2.0, precision=1.0),
            _frozen(
                "heavy",
                heavy,
                w=4.0,
                precision=1.0,
                condition=_condition(1.0, 1.0),
            ),
        ),
        veto_rules=(FrozenVetoRule("veto-b", _condition(1.0, 1.0)),),
    )
    records = (
        _record("B", 1.0, no_op, light, heavy),
        _record("A", 0.0, no_op, light, heavy),
    )

    decision = rulebook.evaluate(records)
    assert decision.anchor_id == "A"
    assert decision.candidate == light
    assert decision.vetoed_anchor_ids == ("B",)

    tied = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        action_rules=(_frozen("light", light, w=2.0, precision=1.0),),
    )
    tied_decision = tied.evaluate(
        (
            _record("B", 0.0, no_op, light),
            _record("A", 0.0, no_op, light),
        )
    )
    assert tied_decision.anchor_id == "A"


def test_veto_blocks_deployment_but_not_experimental_rival_ranking() -> None:
    no_op = RuleAction.no_op()
    light = RuleAction("speed", "light")
    heavy = RuleAction("speed", "heavy")
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        action_rules=(
            _frozen(
                "light",
                light,
                w=2.0,
                precision=1.0,
                rival_lcb=1.0,
            ),
            _frozen(
                "heavy",
                heavy,
                w=4.0,
                precision=1.0,
                rival_lcb=3.0,
            ),
        ),
        veto_rules=(FrozenVetoRule("veto-a", _condition()),),
    )
    record = _record("A", 0.0, no_op, light, heavy)

    assert rulebook.evaluate((record,)).candidate == no_op
    assert rulebook.strongest_rival(record, selected_action=no_op) == heavy


def test_rulebook_round_trip_hash_and_nested_configuration_are_immutable() -> None:
    light = RuleAction("speed", "light")
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        certification_generation=7,
        contender_min_samples=2,
        action_rules=(_frozen("light", light, w=1.0, precision=2.0),),
    )
    original_hash = rulebook.content_hash
    payload = json.loads(json.dumps(rulebook.to_dict()))
    restored = FrozenRulebookPolicy.from_dict(payload)

    assert restored.content_hash == original_hash
    assert restored.to_dict() == rulebook.to_dict()
    assert restored.contender_min_samples == 2
    assert restored.credit_mode == "causal"
    assert (
        FrozenRulebookPolicy(
            feature_schema_hash=SCHEMA_HASH,
            credit_mode="vanilla_accuracy",
        ).content_hash
        != rulebook.content_hash
    )

    configuration = rulebook.action_configuration
    assert not isinstance(configuration, str)
    actions = configuration["actions"]
    with pytest.raises(TypeError):
        actions[0]["band"] = "mutated"  # type: ignore[index]
    with pytest.raises(AttributeError):
        actions.append({})  # type: ignore[attr-defined]
    assert rulebook.content_hash == original_hash

    different_threshold = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        certification_generation=7,
        contender_min_samples=3,
        action_rules=(_frozen("light", light, w=1.0, precision=2.0),),
    )
    assert different_threshold.content_hash != original_hash


def test_rival_ranking_is_frozen_until_rulebook_republication() -> None:
    condition = _condition()
    light = MutableRule(condition, RuleAction("speed", "light"))
    heavy = MutableRule(condition, RuleAction("speed", "heavy"))
    for rule in (light, heavy):
        rule.deployment.extend([1.0] * 30)
    light.evolution.extend([0.2, 0.2])
    heavy.evolution.extend([0.5, 0.5])
    thresholds = CertificationThresholds(variance_floor=1.0e-6)
    snapshot = FrozenRulebookPolicy.from_population(
        (light, heavy),
        feature_schema_hash=SCHEMA_HASH,
        certification_generation=1,
        thresholds=thresholds,
    )
    record = _record(
        "A",
        0.0,
        RuleAction.no_op(),
        light.action,
        heavy.action,
    )

    assert (
        snapshot.strongest_rival(record, selected_action=RuleAction.no_op())
        == heavy.action
    )
    light.evolution.extend([10.0] * 100)
    assert (
        snapshot.strongest_rival(record, selected_action=RuleAction.no_op())
        == heavy.action
    )

    republished = FrozenRulebookPolicy.from_population(
        (light, heavy),
        feature_schema_hash=SCHEMA_HASH,
        certification_generation=2,
        thresholds=thresholds,
    )
    assert (
        republished.strongest_rival(record, selected_action=RuleAction.no_op())
        == light.action
    )


def test_action_configuration_hash_matches_vocabulary_or_preserves_hash() -> None:
    vocabulary = action_vocabulary()
    embedded = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        action_configuration=vocabulary.payload,
    )
    assert embedded.action_config_hash == vocabulary.content_hash

    supplied_hash = "supplied-action-vocabulary-hash"
    hash_only = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        action_configuration=supplied_hash,
    )
    assert hash_only.action_config_hash == supplied_hash


def test_rival_ranking_enforces_frozen_minimum_samples() -> None:
    no_op = RuleAction.no_op()
    light = RuleAction("speed", "light")
    heavy = RuleAction("speed", "heavy")
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA_HASH,
        contender_min_samples=2,
        action_rules=(
            _frozen(
                "light-one-sample",
                light,
                w=1.0,
                precision=1.0,
                rival_lcb=100.0,
                rival_samples=1,
            ),
            _frozen(
                "heavy-two-samples",
                heavy,
                w=1.0,
                precision=1.0,
                rival_lcb=0.1,
                rival_samples=2,
            ),
        ),
    )
    ranked = rulebook.rank_rivals(
        _record("A", 0.0, no_op, light, heavy),
        selected_action=no_op,
    )
    assert tuple(item.action for item in ranked) == (heavy,)


def test_simulator_wrapper_builds_records_and_diverged_contexts_choose_differently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = leader_follower_feature_schema()
    light = RuleAction("speed", "light")
    heavy = RuleAction("speed", "heavy")
    no_op = RuleAction.no_op()
    rulebook = FrozenRulebookPolicy(
        feature_schema_hash=schema.schema_hash,
        action_rules=(
            _frozen(
                "light-negative-x",
                light,
                w=1.0,
                precision=1.0,
                condition=_condition(None, -0.01, schema_hash=schema.schema_hash),
            ),
            _frozen(
                "heavy-positive-x",
                heavy,
                w=1.0,
                precision=1.0,
                condition=_condition(0.01, None, schema_hash=schema.schema_hash),
            ),
        ),
    )
    runtime_hash = "runtime-config-v1"
    policy = SimulatorRulebookPolicy(
        rulebook,
        schema=schema,
        runtime_configuration_hash=runtime_hash,
    )

    def fake_anchors(simulator: object) -> SimpleNamespace:
        return SimpleNamespace(leader_follower=(simulator.anchor,))  # type: ignore[attr-defined]

    def fake_candidates(
        self: ActionCatalog,
        simulator: object,
        batch: object,
        *,
        anchor_id: str,
        bound_flight_id: str,
        resource_id: str,
        segment_id: str,
    ) -> tuple[RuleAction, ...]:
        del self, batch, anchor_id, bound_flight_id, resource_id, segment_id
        return simulator.candidates  # type: ignore[attr-defined, no-any-return]

    def fake_vector(
        simulator: object, anchor: object, **kwargs: object
    ) -> SimpleNamespace:
        del anchor, kwargs
        return SimpleNamespace(
            schema_hash=schema.schema_hash,
            named={"x": simulator.x},  # type: ignore[attr-defined]
        )

    monkeypatch.setattr(
        feature_api, "build_current_segment_anchors", fake_anchors
    )
    monkeypatch.setattr(feature_api, "simulator_state_vector", fake_vector)
    monkeypatch.setattr(ActionCatalog, "enumerate_for_batch", fake_candidates)

    anchor = SimpleNamespace(
        anchor_id="A",
        follower_id="F",
        resource_id="S:exit",
        segment_id="S",
    )
    left_context = SimpleNamespace(
        simulator=SimpleNamespace(
            anchor=anchor,
            x=-1.0,
            candidates=(no_op, light, heavy),
            runtime_configuration_hash=runtime_hash,
        ),
        event_batch=object(),
    )
    right_context = SimpleNamespace(
        simulator=SimpleNamespace(
            anchor=anchor,
            x=1.0,
            candidates=(no_op, light, heavy),
            runtime_configuration_hash=runtime_hash,
        ),
        event_batch=object(),
    )

    assert policy.select_action(left_context) == light
    assert policy.select_action(right_context) == heavy
    assert policy.policy_fingerprint() == policy.content_hash


def test_simulator_wrapper_requires_explicit_nonblank_runtime_hash() -> None:
    schema = leader_follower_feature_schema()
    rulebook = FrozenRulebookPolicy(feature_schema_hash=schema.schema_hash)

    with pytest.raises(
        ValueError,
        match="requires runtime_configuration_hash",
    ):
        SimulatorRulebookPolicy(rulebook, schema=schema)
    with pytest.raises(
        ValueError,
        match="requires runtime_configuration_hash",
    ):
        SimulatorRulebookPolicy(
            rulebook,
            schema=schema,
            runtime_configuration_hash="   ",
        )


def test_simulator_wrapper_rejects_missing_or_mismatched_context_runtime() -> None:
    schema = leader_follower_feature_schema()
    rulebook = FrozenRulebookPolicy(feature_schema_hash=schema.schema_hash)
    policy = SimulatorRulebookPolicy(
        rulebook,
        schema=schema,
        runtime_configuration_hash="runtime-config-v1",
    )

    missing_context = SimpleNamespace(
        simulator=SimpleNamespace(),
        event_batch=object(),
    )
    with pytest.raises(ValueError, match="must expose a nonblank"):
        policy.rulebook_records(missing_context)

    mismatched_context = SimpleNamespace(
        simulator=SimpleNamespace(runtime_configuration_hash="runtime-config-v2"),
        event_batch=object(),
    )
    with pytest.raises(ValueError, match="does not match"):
        policy.select_action(mismatched_context)


def _rehash_rulebook(payload: dict[str, object]) -> None:
    payload.pop("content_hash", None)
    payload["content_hash"] = content_hash(
        payload,
        namespace=FROZEN_RULEBOOK_VERSION,
    )


def test_rulebook_constructor_rejects_coercive_identity_and_integer_values() -> None:
    for invalid_generation in (True, 1.5, "1"):
        with pytest.raises(ValueError, match="non-negative integer"):
            FrozenRulebookPolicy(
                feature_schema_hash=SCHEMA_HASH,
                certification_generation=invalid_generation,  # type: ignore[arg-type]
            )

    for invalid_min_samples in (True, 2.5, "2"):
        with pytest.raises(ValueError, match="integer of at least two"):
            FrozenRulebookPolicy(
                feature_schema_hash=SCHEMA_HASH,
                contender_min_samples=invalid_min_samples,  # type: ignore[arg-type]
            )

    with pytest.raises(TypeError, match="must be strings"):
        FrozenRulebookPolicy(feature_schema_hash=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be strings"):
        FrozenRulebookPolicy(
            feature_schema_hash=SCHEMA_HASH,
            role_type=123,  # type: ignore[arg-type]
        )


def test_rulebook_deserialization_requires_exact_canonical_schema() -> None:
    canonical = FrozenRulebookPolicy(feature_schema_hash=SCHEMA_HASH).to_dict()

    for field_name in tuple(canonical):
        missing = dict(canonical)
        missing.pop(field_name)
        if field_name != "content_hash":
            _rehash_rulebook(missing)
        with pytest.raises(ValueError, match="canonical schema"):
            FrozenRulebookPolicy.from_dict(missing)

    unknown = {**canonical, "extra": None}
    _rehash_rulebook(unknown)
    with pytest.raises(ValueError, match="canonical schema"):
        FrozenRulebookPolicy.from_dict(unknown)


def test_rulebook_deserialization_rejects_coercive_integer_fields() -> None:
    canonical = FrozenRulebookPolicy(feature_schema_hash=SCHEMA_HASH).to_dict()
    cases = (
        (
            "certification_generation",
            (True, 1.5, "1"),
            "non-negative integer",
        ),
        (
            "contender_min_samples",
            (True, 2.5, "2"),
            "integer of at least two",
        ),
    )
    for field_name, invalid_values, message in cases:
        for invalid in invalid_values:
            malformed = {**canonical, field_name: invalid}
            _rehash_rulebook(malformed)
            with pytest.raises(ValueError, match=message):
                FrozenRulebookPolicy.from_dict(malformed)


def test_rulebook_deserialization_rejects_invalid_sequence_fields() -> None:
    canonical = FrozenRulebookPolicy(feature_schema_hash=SCHEMA_HASH).to_dict()

    for field_name in ("action_rules", "veto_rules", "tie_break"):
        for invalid in ("not-a-sequence", {"bad": "shape"}, 7):
            malformed = {**canonical, field_name: invalid}
            _rehash_rulebook(malformed)
            with pytest.raises(TypeError, match="must be a sequence"):
                FrozenRulebookPolicy.from_dict(malformed)

    tie_break = list(canonical["tie_break"])
    tie_break[0] = 7
    malformed_tie_break = {**canonical, "tie_break": tie_break}
    _rehash_rulebook(malformed_tie_break)
    with pytest.raises(TypeError, match="entries must be strings"):
        FrozenRulebookPolicy.from_dict(malformed_tie_break)


def test_rulebook_deserialization_rejects_noncanonical_identity_and_hash() -> None:
    canonical = FrozenRulebookPolicy(feature_schema_hash=SCHEMA_HASH).to_dict()

    for field_name in ("feature_schema_hash", "role_type"):
        malformed = {**canonical, field_name: 123}
        _rehash_rulebook(malformed)
        with pytest.raises(TypeError, match="must be strings"):
            FrozenRulebookPolicy.from_dict(malformed)

    for field_name in ("feature_schema_hash", "role_type"):
        malformed = {**canonical, field_name: "   "}
        _rehash_rulebook(malformed)
        with pytest.raises(ValueError, match="cannot be empty"):
            FrozenRulebookPolicy.from_dict(malformed)

    wrong_version = {
        **canonical,
        "artifact_version": "hailmary.frozen_rulebook.v999",
    }
    _rehash_rulebook(wrong_version)
    with pytest.raises(ValueError, match="unsupported"):
        FrozenRulebookPolicy.from_dict(wrong_version)

    malformed_hash = {**canonical, "content_hash": 123}
    with pytest.raises(TypeError, match="content_hash"):
        FrozenRulebookPolicy.from_dict(malformed_hash)


def test_rulebook_mode_is_strict_and_enforces_ledger_semantics() -> None:
    no_op = RuleAction.no_op()
    no_op_action = _frozen("no-op-action", no_op, w=1.0, precision=1.0)
    veto = FrozenVetoRule("veto", _condition())

    with pytest.raises(ValueError, match="scoped vetoes"):
        FrozenRulebookPolicy(
            feature_schema_hash=SCHEMA_HASH,
            action_rules=(no_op_action,),
        )
    with pytest.raises(ValueError, match="cannot contain causal vetoes"):
        FrozenRulebookPolicy(
            feature_schema_hash=SCHEMA_HASH,
            credit_mode="vanilla_accuracy",
            veto_rules=(veto,),
        )
    for invalid in (None, True, "", " causal", "unsupported"):
        expected = TypeError if type(invalid) is not str else ValueError
        with pytest.raises(expected, match="credit_mode"):
            FrozenRulebookPolicy(
                feature_schema_hash=SCHEMA_HASH,
                credit_mode=invalid,  # type: ignore[arg-type]
            )

    payload = FrozenRulebookPolicy(feature_schema_hash=SCHEMA_HASH).to_dict()
    payload["credit_mode"] = "vanilla_accuracy"
    _rehash_rulebook(payload)
    restored = FrozenRulebookPolicy.from_dict(payload)
    assert restored.credit_mode == "vanilla_accuracy"
