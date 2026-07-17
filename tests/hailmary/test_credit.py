from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace

import pytest

from hailmary.actions.models import ActionLever
from hailmary.actions.vocabulary import action_vocabulary
from hailmary.features.schema import FeatureField, FeatureSchema
from hailmary.learning.conditions import Interval, RuleCondition
from hailmary.learning.credit import (
    CausalCreditAssigner,
    CreditSignals,
    VanillaAccuracyCreditAssigner,
    credit_assigner_configuration,
    credit_assigner_from_configuration,
)
from hailmary.learning.modes import (
    CAUSAL_CREDIT_MODE,
    VANILLA_ACCURACY_CREDIT_MODE,
)
from hailmary.learning.matching import AnchorContext, build_match_set
from hailmary.learning.population import Population
from hailmary.learning.rules import MutableRule, RuleAction


@dataclass(frozen=True, slots=True)
class Candidate:
    lever: ActionLever
    band: str
    feasible: bool = True


def _learning_fixture() -> tuple[
    AnchorContext,
    Population,
    object,
    dict[str, MutableRule],
]:
    return _learning_fixture_for_mode(CAUSAL_CREDIT_MODE)


def _learning_fixture_for_mode(
    credit_mode: str,
) -> tuple[
    AnchorContext,
    Population,
    object,
    dict[str, MutableRule],
]:
    schema = FeatureSchema(
        "test.credit.v1",
        (FeatureField("x", lower_bound=-10.0, upper_bound=10.0),),
    )
    vector = schema.encode({"x": 0.0})
    candidates = tuple(
        Candidate(identity.lever, identity.band)
        for identity in action_vocabulary().identities
    )
    context = AnchorContext(
        role_type="leader_follower",
        schema_hash=schema.schema_hash,
        anchor_id="anchor-credit",
        vector=vector,
        candidates=candidates,
    )

    def rule(
        name: str,
        action: RuleAction,
        interval: Interval = Interval(-1.0, 1.0),
    ) -> MutableRule:
        return MutableRule(
            RuleCondition(
                context.role_type,
                context.schema_hash,
                {"x": interval},
            ),
            action,
            rule_id=name,
        )

    rules = {
        "noop": rule("rule-noop", RuleAction.no_op()),
        "light_a": rule(
            "rule-light-a",
            RuleAction(ActionLever.SPEED, "light"),
        ),
        "light_b": rule(
            "rule-light-b",
            RuleAction(ActionLever.SPEED, "light"),
            Interval(-2.0, 2.0),
        ),
        "medium": rule(
            "rule-medium",
            RuleAction(ActionLever.SPEED, "medium"),
        ),
        "heavy": rule(
            "rule-heavy",
            RuleAction(ActionLever.SPEED, "heavy"),
        ),
        "path": rule(
            "rule-path",
            RuleAction(
                ActionLever.PATH_STRETCH,
                "oracle_short_medium_long",
            ),
        ),
    }
    population = Population(tuple(rules.values()), credit_mode=credit_mode)
    return context, population, build_match_set(context, population), rules


def test_credit_signals_are_exact_finite_three_arm_deltas() -> None:
    signals = CreditSignals.from_arm_scores(
        selected=5.0,
        contender=3.0,
        no_op=2.0,
    )

    assert signals == CreditSignals(
        delta_rival=2.0,
        delta_selected_noop=3.0,
        delta_contender_noop=1.0,
        delta_veto=-3.0,
    )
    assert (
        CreditSignals.from_rollout(
            SimpleNamespace(
                delta_rival=2.0,
                delta_selected_noop=3.0,
                delta_contender_noop=1.0,
                delta_veto=-3.0,
            )
        )
        == signals
    )
    with pytest.raises(ValueError, match="finite"):
        CreditSignals(float("nan"), 0.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        signals.delta_rival = 9.0  # type: ignore[misc]


def test_causal_credit_routes_only_frozen_root_coadvocates() -> None:
    _, population, match_set, rules = _learning_fixture()
    selected = RuleAction(ActionLever.SPEED, "light")
    contender = RuleAction(
        ActionLever.PATH_STRETCH,
        "oracle_short_medium_long",
    )

    recipients = CausalCreditAssigner().assign(
        population,
        match_set,
        selected_action=selected,
        contender_action=contender,
        signals=CreditSignals.from_arm_scores(
            selected=5.0,
            contender=3.0,
            no_op=2.0,
        ),
    )

    assert recipients.anchor_id == "anchor-credit"
    assert recipients.selected_rule_ids == (
        "rule-light-a",
        "rule-light-b",
    )
    assert recipients.contender_rule_ids == ("rule-path",)
    assert recipients.no_op_rule_ids == ("rule-noop",)
    assert recipients.all_rule_ids == (
        "rule-light-a",
        "rule-light-b",
        "rule-noop",
        "rule-path",
    )
    for key in ("light_a", "light_b"):
        assert rules[key].evolution.to_dict() == {
            "n": 1,
            "mean": 2.0,
            "m2": 0.0,
        }
        assert rules[key].deployment.to_dict() == {
            "n": 1,
            "mean": 3.0,
            "m2": 0.0,
        }
    assert rules["path"].evolution.mean == -2.0
    assert rules["path"].deployment.mean == 1.0
    assert rules["noop"].evolution.mean == -3.0
    assert rules["noop"].deployment.n == 0
    assert rules["medium"].evolution.n == 0
    assert rules["heavy"].evolution.n == 0


def test_no_op_arm_roles_never_receive_ordinary_deployment_credit() -> None:
    _, population, match_set, rules = _learning_fixture()
    no_op = RuleAction.no_op()
    light = RuleAction(ActionLever.SPEED, "light")

    CausalCreditAssigner().assign(
        population,
        match_set,
        selected_action=light,
        contender_action=no_op,
        signals=CreditSignals.from_arm_scores(
            selected=4.0,
            contender=2.0,
            no_op=2.0,
        ),
    )

    assert rules["light_a"].evolution.mean == 2.0
    assert rules["light_a"].deployment.mean == 2.0
    assert rules["noop"].evolution.mean == -2.0
    assert rules["noop"].evolution.n == 1
    assert rules["noop"].deployment.n == 0

    _, population, match_set, rules = _learning_fixture()
    CausalCreditAssigner().assign(
        population,
        match_set,
        selected_action=no_op,
        contender_action=light,
        signals=CreditSignals.from_arm_scores(
            selected=2.0,
            contender=1.0,
            no_op=2.0,
        ),
    )

    assert rules["noop"].evolution.mean == 0.0
    assert rules["noop"].evolution.n == 1
    assert rules["noop"].deployment.n == 0
    assert rules["light_a"].evolution.mean == -1.0
    assert rules["light_a"].deployment.mean == -1.0


def test_causal_credit_resolves_every_frozen_id_before_any_mutation() -> None:
    _, population, match_set, rules = _learning_fixture()
    population.remove(rules["path"].rule_id)

    with pytest.raises(KeyError, match="unknown rule"):
        CausalCreditAssigner().assign(
            population,
            match_set,
            selected_action=RuleAction(ActionLever.SPEED, "light"),
            contender_action=RuleAction(
                ActionLever.PATH_STRETCH,
                "oracle_short_medium_long",
            ),
            signals=CreditSignals.from_arm_scores(
                selected=5.0,
                contender=3.0,
                no_op=2.0,
            ),
        )

    assert rules["light_a"].evolution.n == 0
    assert rules["light_a"].deployment.n == 0
    assert rules["noop"].evolution.n == 0


def test_vanilla_control_updates_only_selected_action_prediction_and_accuracy() -> None:
    _, population, match_set, rules = _learning_fixture_for_mode(
        VANILLA_ACCURACY_CREDIT_MODE
    )
    rules["light_a"].deployment.update(2.0)
    assigner = VanillaAccuracyCreditAssigner(
        initial_prediction=0.0,
        accuracy_scale=1.0,
    )

    updates = assigner.assign(
        population,
        match_set,
        selected_action=RuleAction(ActionLever.SPEED, "light"),
        selected_outcome=3.0,
    )
    updates_by_id = {update.rule_id: update for update in updates}

    assert updates_by_id["rule-light-a"].pre_update_prediction == 2.0
    assert updates_by_id["rule-light-a"].absolute_error == 1.0
    assert updates_by_id["rule-light-a"].accuracy == pytest.approx(0.5)
    assert updates_by_id["rule-light-b"].pre_update_prediction == 0.0
    assert updates_by_id["rule-light-b"].absolute_error == 3.0
    assert updates_by_id["rule-light-b"].accuracy == pytest.approx(0.25)
    assert rules["light_a"].evolution.mean == pytest.approx(0.5)
    assert rules["light_a"].deployment.mean == pytest.approx(2.5)
    assert rules["light_b"].evolution.mean == pytest.approx(0.25)
    assert rules["light_b"].deployment.mean == pytest.approx(3.0)
    for key in ("noop", "medium", "heavy", "path"):
        assert rules[key].evolution.n == 0
        assert rules[key].deployment.n == 0


def test_vanilla_control_treats_selected_no_op_as_an_ordinary_action() -> None:
    _, population, match_set, rules = _learning_fixture_for_mode(
        VANILLA_ACCURACY_CREDIT_MODE
    )

    updates = VanillaAccuracyCreditAssigner().assign(
        population,
        match_set,
        selected_action=RuleAction.no_op(),
        selected_outcome=1.0,
    )

    assert len(updates) == 1
    assert updates[0].accuracy == pytest.approx(0.5)
    assert rules["noop"].evolution.mean == pytest.approx(0.5)
    assert rules["noop"].deployment.mean == 1.0
    for key in ("light_a", "light_b", "medium", "heavy", "path"):
        assert rules[key].evolution.n == 0
        assert rules[key].deployment.n == 0


def test_credit_assigners_reject_cross_mode_population_reuse() -> None:
    _, causal, causal_match, _ = _learning_fixture()
    _, vanilla, vanilla_match, _ = _learning_fixture_for_mode(
        VANILLA_ACCURACY_CREDIT_MODE
    )
    light = RuleAction(ActionLever.SPEED, "light")

    with pytest.raises(ValueError, match="vanilla_accuracy-ledger"):
        VanillaAccuracyCreditAssigner().assign(
            causal,
            causal_match,
            selected_action=light,
            selected_outcome=1.0,
        )
    with pytest.raises(ValueError, match="causal-ledger"):
        CausalCreditAssigner().assign(
            vanilla,
            vanilla_match,
            selected_action=light,
            contender_action=RuleAction.no_op(),
            signals=CreditSignals.from_arm_scores(
                selected=1.0,
                contender=0.0,
                no_op=0.0,
            ),
        )


def test_vanilla_credit_configuration_is_immutable_and_replayable() -> None:
    assigner = VanillaAccuracyCreditAssigner(
        initial_prediction=1.25,
        accuracy_scale=4.5,
    )
    configuration = credit_assigner_configuration(assigner)
    restored = credit_assigner_from_configuration(configuration)

    assert configuration == {
        "mode": "vanilla_accuracy",
        "settings": {
            "initial_prediction": 1.25,
            "accuracy_scale": 4.5,
        },
    }
    assert restored == assigner
    with pytest.raises(FrozenInstanceError):
        assigner.accuracy_scale = 99.0  # type: ignore[misc]
