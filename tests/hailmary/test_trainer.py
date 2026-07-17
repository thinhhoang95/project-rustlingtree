from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest

from hailmary.actions.models import ActionLever
from hailmary.config import LearningConfig, StretchConfig
from hailmary.features.schema import FeatureField, FeatureSchema
from hailmary.ids import canonical_data, content_hash
from hailmary.learning.artifacts import EvaluationSnapshot
from hailmary.learning.certification import CertificationThresholds
from hailmary.learning.conditions import Interval, RuleCondition
from hailmary.learning.evolution import make_offspring
from hailmary.learning.credit import VanillaAccuracyCreditAssigner
from hailmary.learning.exploration import RegionExplorationScheduler
from hailmary.learning.matching import AnchorContext, build_match_set
from hailmary.learning.population import Population
from hailmary.learning.rulebook import FrozenRulebookPolicy
from hailmary.learning.rules import MutableRule, RuleAction
from hailmary.learning.statistics import OnlineMoments
from hailmary.learning.trainer import CausalTrainer
from hailmary.rollout.paired import PairedArmTrace, ThreeArmRolloutResult
from hailmary.runtime import build_action_runtime
from hailmary.simulator.events import DecisionEpoch, EventBatchResult


SCHEMA = FeatureSchema(
    "test.trainer.v1",
    (
        FeatureField(
            "commitment_fraction",
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        FeatureField("pressure_ratio", lower_bound=0.0, upper_bound=3.0),
        FeatureField(
            "abs_spacing_deviation_s",
            lower_bound=0.0,
            upper_bound=300.0,
        ),
    ),
)
VECTOR = SCHEMA.encode(
    {
        "commitment_fraction": 0.4,
        "pressure_ratio": 1.0,
        "abs_spacing_deviation_s": 65.0,
    }
)
NO_OP = RuleAction.no_op()
SPEED = RuleAction(ActionLever.SPEED, "light")
STRETCH = RuleAction(ActionLever.PATH_STRETCH, "oracle_short_medium_long")


@dataclass(frozen=True, slots=True)
class Candidate:
    anchor_id: str
    lever: ActionLever
    band: str
    feasible: bool = True


@dataclass(frozen=True, slots=True)
class SourceAnchor:
    anchor_id: str


@dataclass(frozen=True, slots=True)
class FakePlan:
    anchor_id: str
    horizon_s: float = 120.0


@dataclass(frozen=True, slots=True)
class FakeDefinition:
    definition_hash: str = "test-scenario-definition"


@dataclass(frozen=True, slots=True)
class FakeCheckpointState:
    state_id: str
    parent_state_id: str
    sim_time_s: float
    version: int
    decision_epoch_index: int


class FakeSimulator:
    def __init__(
        self,
        definition: FakeDefinition,
        *,
        dynamic_hash: str = "root-dynamic-content",
        commits: list[dict[str, Any]] | None = None,
    ) -> None:
        self.definition = definition
        self.dynamic_content_hash = dynamic_hash
        self.commits = [] if commits is None else list(commits)

    def apply(self, candidate: Candidate) -> "FakeSimulator":
        self.commits.append(canonical_data(candidate))
        self.dynamic_content_hash = content_hash(
            {
                "root": "root-dynamic-content",
                "commits": self.commits,
            },
            namespace="test.fake_simulator",
        )
        return self

    apply_action = apply

    def snapshot(self) -> dict[str, Any]:
        return {
            "definition_hash": self.definition.definition_hash,
            "dynamic_content_hash": self.dynamic_content_hash,
            "commits": list(self.commits),
        }


class FakeRuntime:
    def __init__(self, *, stretch_config: StretchConfig | None = None) -> None:
        base = build_action_runtime(stretch_config=stretch_config)
        self.catalog = base.catalog
        self.template_config = base.template_config
        self.vocabulary = base.vocabulary
        self.action_vocabulary_hash = base.action_vocabulary_hash
        self.runtime_configuration_hash = base.runtime_configuration_hash
        self.action_applier = base.action_applier

    def resume_simulator(
        self,
        definition: FakeDefinition,
        snapshot: dict[str, Any],
    ) -> FakeSimulator:
        return FakeSimulator(
            definition,
            dynamic_hash=str(snapshot["dynamic_content_hash"]),
            commits=list(snapshot.get("commits", ())),
        )


def _candidate(action: RuleAction, anchor_id: str = "anchor-1") -> Candidate:
    return Candidate(anchor_id, action.lever, action.band)


def _context(*actions: RuleAction, anchor_id: str = "anchor-1") -> AnchorContext:
    return AnchorContext(
        role_type="leader_follower",
        schema_hash=SCHEMA.schema_hash,
        anchor_id=anchor_id,
        vector=VECTOR,
        candidates=tuple(_candidate(action, anchor_id) for action in actions),
    )


def _context_builder(context: AnchorContext):
    source = SourceAnchor(context.anchor_id)

    def build(_simulator: Any, _batch: Any):
        return ((source, context),)

    return build


def _rule(
    action: RuleAction,
    *,
    rival_mean: float = 0.0,
    deployment_mean: float = 0.0,
    certified: bool = False,
) -> MutableRule:
    return MutableRule(
        condition=RuleCondition("leader_follower", SCHEMA.schema_hash, {}),
        action=action,
        evolution=(
            OnlineMoments(n=3, mean=rival_mean, m2=0.0)
            if certified
            else OnlineMoments()
        ),
        deployment=(
            OnlineMoments(n=3, mean=deployment_mean, m2=0.0)
            if certified and not action.is_no_op
            else OnlineMoments()
        ),
        provenance={"fixture": action.key},
    )


def _config(**changes: Any) -> LearningConfig:
    values = {
        "population_limit": 100,
        "ga_interval": 999,
        "ga_min_experience": 2,
        "certification_interval": 999,
        "action_min_noop_samples": 2,
        "veto_min_rival_samples": 2,
        "contender_min_samples": 2,
        "base_exploration_rate": 0.0,
        "exploration_coverage_floor": 1,
        "random_seed": 13,
    }
    values.update(changes)
    return LearningConfig(**values)


def _rollout_hook(scores: dict[str, float], calls: list[dict[str, str]]):
    def run(
        parent: FakeSimulator,
        *,
        selected_action: Candidate,
        contender_action: Candidate,
        no_op_action: Candidate,
        frozen_policy: Any,
        outcome_plan: FakePlan,
        outcome_config: Any,
    ) -> ThreeArmRolloutResult:
        del outcome_config
        parent_hash = parent.dynamic_content_hash

        def arm(label: str, candidate: Candidate) -> PairedArmTrace:
            action = RuleAction.from_candidate(candidate)
            score = float(scores[action.key])
            return PairedArmTrace(
                label=label,
                initial_action=candidate,
                applied_action=candidate,
                action_audit=None,
                initial_dynamic_content_hash=parent_hash,
                final_dynamic_content_hash=content_hash(
                    {"parent": parent_hash, "action": action.to_dict()},
                    namespace="test.fake_arm",
                ),
                score=score,
                outcome=score,
            )

        selected = arm("selected", selected_action)
        contender = arm("contender", contender_action)
        no_op = arm("no_op", no_op_action)
        calls.append(
            {
                "selected": RuleAction.from_candidate(selected_action).key,
                "contender": RuleAction.from_candidate(contender_action).key,
                "no_op": RuleAction.from_candidate(no_op_action).key,
            }
        )
        return ThreeArmRolloutResult(
            selected=selected,
            contender=contender,
            no_op=no_op,
            delta_rival=selected.score - contender.score,
            delta_selected_noop=selected.score - no_op.score,
            delta_contender_noop=contender.score - no_op.score,
            delta_veto=no_op.score - max(selected.score, contender.score),
            parent_dynamic_content_hash=parent_hash,
            policy_fingerprint=frozen_policy.policy_fingerprint(),
            horizon_s=outcome_plan.horizon_s,
        )

    return run


def _corrupt_rollout(
    rollout: ThreeArmRolloutResult,
    fault: str,
) -> ThreeArmRolloutResult:
    if fault == "parent_hash":
        return replace(rollout, parent_dynamic_content_hash="wrong-parent")
    if fault == "policy_fingerprint":
        return replace(rollout, policy_fingerprint="wrong-policy")
    if fault == "horizon":
        return replace(rollout, horizon_s=rollout.horizon_s + 1.0)
    if fault == "label":
        return replace(
            rollout,
            selected=replace(rollout.selected, label="wrong-label"),
        )
    if fault == "initial_hash":
        return replace(
            rollout,
            contender=replace(
                rollout.contender,
                initial_dynamic_content_hash="wrong-parent",
            ),
        )
    if fault == "selected_action":
        return replace(
            rollout,
            selected=replace(
                rollout.selected,
                initial_action=replace(
                    rollout.selected.initial_action,
                    feasible=False,
                ),
            ),
        )
    if fault == "arm_c_action":
        return replace(
            rollout,
            no_op=replace(
                rollout.no_op,
                initial_action=rollout.selected.initial_action,
            ),
        )
    if fault == "score":
        return replace(
            rollout,
            selected=replace(rollout.selected, score=float("nan")),
        )
    if fault in {
        "delta_rival",
        "delta_selected_noop",
        "delta_contender_noop",
        "delta_veto",
    }:
        return replace(
            rollout,
            **{fault: getattr(rollout, fault) + 0.25},
        )
    raise AssertionError(f"unknown rollout corruption {fault!r}")


def _corrupting_rollout_hook(
    scores: dict[str, float],
    calls: list[dict[str, str]],
    fault: str,
):
    valid_hook = _rollout_hook(scores, calls)

    def run(parent: FakeSimulator, **kwargs: Any) -> ThreeArmRolloutResult:
        return _corrupt_rollout(valid_hook(parent, **kwargs), fault)

    return run


def _plan(_simulator: Any, anchor: SourceAnchor, *, config: Any) -> FakePlan:
    del config
    return FakePlan(anchor.anchor_id)


def _test_hook_fingerprints(
    context: Any,
    scores: dict[str, float],
    *,
    rollout_variant: str = "three_arm",
) -> dict[str, str]:
    return {
        "context_builder": content_hash(
            context,
            namespace="test.trainer.context_builder.v1",
        ),
        "outcome_plan_factory": "test.trainer.fake_plan.v1",
        "rollout_hook": content_hash(
            {"variant": rollout_variant, "scores": scores},
            namespace="test.trainer.rollout_hook.v1",
        ),
    }


def _trainer(
    context: AnchorContext,
    population: Population,
    calls: list[dict[str, str]],
    *,
    scores: dict[str, float],
    config: LearningConfig | None = None,
    credit_assigner: Any | None = None,
) -> CausalTrainer:
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        population=population,
        config=_config() if config is None else config,
        credit_assigner=credit_assigner,
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook(scores, calls),
        hook_fingerprints=_test_hook_fingerprints(context, scores),
    )
    # Publish the supplied pre-trained rules at a real certification boundary.
    trainer._publish_if_due(epoch=trainer.config.certification_interval)
    trainer.epoch = trainer.config.certification_interval
    return trainer


def _prime_scheduler(
    trainer: CausalTrainer,
    context: AnchorContext,
    *,
    undercovered: RuleAction | None = None,
) -> None:
    payload = trainer.scheduler.to_dict()
    cell = trainer.scheduler.cell_for(context)
    completed = tuple(
        action for action in context.candidate_actions if action != undercovered
    )
    payload["visits"] = [{"cell": cell.to_dict(), "count": len(completed)}]
    payload["experiments"] = [
        {
            "cell": cell.to_dict(),
            "action": action.to_dict(),
            "count": 1,
        }
        for action in completed
    ]
    trainer.scheduler = RegionExplorationScheduler.from_dict(payload)


def _added_sample(before: OnlineMoments, after: OnlineMoments) -> float:
    assert after.n == before.n + 1
    return after.mean * after.n - before.mean * before.n


def _copy(moment: OnlineMoments) -> OnlineMoments:
    return OnlineMoments.from_dict(moment.to_dict())


def test_routes_ordinary_selected_contender_and_veto_credit_once() -> None:
    speed = _rule(SPEED, rival_mean=1.0, deployment_mean=2.0, certified=True)
    stretch = _rule(
        STRETCH,
        rival_mean=1.5,
        deployment_mean=1.0,
        certified=True,
    )
    no_op = _rule(NO_OP)
    population = Population((speed, stretch, no_op), max_size=100)
    context = _context(NO_OP, SPEED, STRETCH)
    calls: list[dict[str, str]] = []
    trainer = _trainer(
        context,
        population,
        calls,
        scores={SPEED.key: 4.0, STRETCH.key: 2.0, NO_OP.key: 1.0},
    )
    _prime_scheduler(trainer, context)
    before = {
        rule.rule_id: (_copy(rule.evolution), _copy(rule.deployment))
        for rule in population.rules
    }
    simulator = FakeSimulator(FakeDefinition())

    result = trainer.process_epoch(simulator, object())

    assert calls == [
        {"selected": SPEED.key, "contender": STRETCH.key, "no_op": NO_OP.key}
    ]
    assert result.trace.delta_rival == pytest.approx(2.0)
    assert _added_sample(before[speed.rule_id][0], speed.evolution) == pytest.approx(
        2.0
    )
    assert _added_sample(before[speed.rule_id][1], speed.deployment) == pytest.approx(
        3.0
    )
    assert _added_sample(
        before[stretch.rule_id][0], stretch.evolution
    ) == pytest.approx(-2.0)
    assert _added_sample(
        before[stretch.rule_id][1], stretch.deployment
    ) == pytest.approx(1.0)
    assert _added_sample(before[no_op.rule_id][0], no_op.evolution) == pytest.approx(
        -3.0
    )
    assert no_op.deployment.n == 0
    assert result.trace.ledger_recipients["veto_evolution"] == [no_op.rule_id]
    assert set(result.trace.coadvocate_rule_ids) == {
        speed.rule_id,
        stretch.rule_id,
        no_op.rule_id,
    }
    assert len(simulator.commits) == 1
    assert simulator.commits[0]["lever"] == ActionLever.SPEED.value


def test_vanilla_mode_uses_same_rollout_and_accuracy_fitness_credit() -> None:
    speed = _rule(
        SPEED,
        rival_mean=0.8,
        deployment_mean=2.0,
        certified=True,
    )
    no_op = _rule(NO_OP)
    population = Population(
        (speed, no_op), max_size=100, credit_mode="vanilla_accuracy"
    )
    context = _context(NO_OP, SPEED)
    calls: list[dict[str, str]] = []
    trainer = _trainer(
        context,
        population,
        calls,
        scores={SPEED.key: 5.0, NO_OP.key: 1.0},
        credit_assigner=VanillaAccuracyCreditAssigner(
            initial_prediction=0.0,
            accuracy_scale=2.0,
        ),
    )
    _prime_scheduler(trainer, context)
    before_no_op = no_op.to_dict()
    simulator = FakeSimulator(FakeDefinition())

    result = trainer.process_epoch(simulator, object())

    assert calls == [
        {"selected": SPEED.key, "contender": NO_OP.key, "no_op": NO_OP.key}
    ]
    assert trainer.credit_mode == "vanilla_accuracy"
    assert speed.evolution.n == 4
    assert speed.evolution.mean == pytest.approx(0.7)
    assert speed.deployment.n == 4
    assert speed.deployment.mean == pytest.approx(2.75)
    assert no_op.to_dict() == before_no_op
    assert result.trace.coadvocate_rule_ids == (speed.rule_id,)
    assert result.trace.ledger_recipients == {
        "credit_mode": "vanilla_accuracy",
        "anchor_id": context.anchor_id,
        "selected_accuracy_evolution": [speed.rule_id],
        "selected_prediction_deployment": [speed.rule_id],
        "contender_prediction": [],
        "veto_evolution": [],
        "updates": [
            {
                "rule_id": speed.rule_id,
                "pre_update_prediction": 2.0,
                "selected_outcome": 5.0,
                "absolute_error": 3.0,
                "accuracy": 0.4,
            }
        ],
    }
    assert trainer.current_rulebook.veto_rules == ()
    assert trainer.current_rulebook.credit_mode == "vanilla_accuracy"
    assert len(simulator.commits) == 1


def test_trainer_rejects_population_from_the_other_credit_mode() -> None:
    causal_population = Population(max_size=100)
    vanilla_population = Population(
        max_size=100,
        credit_mode="vanilla_accuracy",
    )

    with pytest.raises(ValueError, match="population credit_mode"):
        CausalTrainer(
            FakeRuntime(),  # type: ignore[arg-type]
            population=causal_population,
            config=_config(),
            schema=SCHEMA,
            credit_assigner=VanillaAccuracyCreditAssigner(),
        )
    with pytest.raises(ValueError, match="population credit_mode"):
        CausalTrainer(
            FakeRuntime(),  # type: ignore[arg-type]
            population=vanilla_population,
            config=_config(),
            schema=SCHEMA,
        )

    created = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=_config(),
        schema=SCHEMA,
        credit_assigner=VanillaAccuracyCreditAssigner(),
    )
    assert created.population.credit_mode == "vanilla_accuracy"
    assert created.current_rulebook.credit_mode == "vanilla_accuracy"


def test_contender_falls_back_to_mandatory_no_op() -> None:
    speed = _rule(SPEED, rival_mean=1.0, deployment_mean=2.0, certified=True)
    no_op = _rule(NO_OP)
    population = Population((speed, no_op), max_size=100)
    context = _context(NO_OP, SPEED)
    calls: list[dict[str, str]] = []
    trainer = _trainer(
        context,
        population,
        calls,
        scores={SPEED.key: 3.0, NO_OP.key: 1.0},
    )
    _prime_scheduler(trainer, context)
    simulator = FakeSimulator(FakeDefinition())

    result = trainer.process_epoch(simulator, object())

    assert calls[0]["contender"] == NO_OP.key
    assert result.trace.delta_contender_noop == 0.0
    assert result.trace.ledger_recipients["contender_evolution"] == []
    assert result.trace.contender_selection["fallback"] == (
        "mandatory_no_op_no_certified_rival"
    )
    assert result.trace.contender_selection["ranked_candidates"] == []
    assert len(simulator.commits) == 1


def test_forced_causal_all_no_op_skip_does_not_record_an_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speed = _rule(SPEED)
    no_op = _rule(NO_OP)
    population = Population((speed, no_op), max_size=100)
    context = _context(NO_OP, SPEED)
    calls: list[dict[str, str]] = []
    trainer = _trainer(
        context,
        population,
        calls,
        scores={SPEED.key: 3.0, NO_OP.key: 1.0},
    )
    _prime_scheduler(trainer, context, undercovered=NO_OP)
    cell = trainer.scheduler.cell_for(context)
    before_visits = trainer.scheduler.visit_count(cell)
    simulator = FakeSimulator(FakeDefinition())

    def force_all_no_op(match_sets: Any, _rulebook: Any) -> Any:
        prepared, match_set = match_sets[0]
        decision = trainer.scheduler.choose(match_set, exploit_action=NO_OP)
        assert decision.action == NO_OP
        return prepared, match_set, decision

    monkeypatch.setattr(trainer, "_select_root", force_all_no_op)

    result = trainer.process_epoch(simulator, object())

    assert result.skipped_reason == "all_no_op_experiment"
    assert result.rollout is None
    assert calls == []
    assert simulator.commits == []
    assert trainer.scheduler.visit_count(cell) == before_visits + 1
    assert trainer.scheduler.experiment_count(cell, NO_OP) == 0
    assert trainer.scheduler.experiment_count(cell, SPEED) == 1
    assert result.trace.contender_selection["fallback"] == (
        "mandatory_no_op_no_certified_rival"
    )


def test_causal_cold_start_excludes_unexperimentable_no_op_from_coverage() -> None:
    speed = _rule(SPEED)
    no_op = _rule(NO_OP)
    population = Population((speed, no_op), max_size=100)
    context = _context(NO_OP, SPEED)
    calls: list[dict[str, str]] = []
    trainer = _trainer(
        context,
        population,
        calls,
        scores={SPEED.key: 3.0, NO_OP.key: 1.0},
    )
    _prime_scheduler(trainer, context, undercovered=NO_OP)
    cell = trainer.scheduler.cell_for(context)
    simulator = FakeSimulator(FakeDefinition())

    result = trainer.process_epoch(simulator, object())

    assert result.committed
    assert result.skipped_reason is None
    assert calls == [
        {"selected": SPEED.key, "contender": NO_OP.key, "no_op": NO_OP.key}
    ]
    assert trainer.scheduler.experiment_count(cell, SPEED) == 2
    assert trainer.scheduler.experiment_count(cell, NO_OP) == 0
    assert result.trace.exploration_reason == "deterministic_fallback"
    assert result.trace.contender_selection["fallback"] == (
        "mandatory_no_op_no_certified_rival"
    )


def test_vanilla_all_no_op_epoch_updates_selected_no_op_accuracy() -> None:
    speed = _rule(SPEED)
    no_op = _rule(NO_OP)
    population = Population(
        (speed, no_op),
        max_size=100,
        credit_mode="vanilla_accuracy",
    )
    context = _context(NO_OP, SPEED)
    calls: list[dict[str, str]] = []
    trainer = _trainer(
        context,
        population,
        calls,
        scores={SPEED.key: 3.0, NO_OP.key: 1.0},
        credit_assigner=VanillaAccuracyCreditAssigner(),
    )
    _prime_scheduler(trainer, context, undercovered=NO_OP)
    cell = trainer.scheduler.cell_for(context)
    simulator = FakeSimulator(FakeDefinition())

    result = trainer.process_epoch(simulator, object())

    assert result.committed
    assert result.skipped_reason is None
    assert calls == [
        {"selected": NO_OP.key, "contender": NO_OP.key, "no_op": NO_OP.key}
    ]
    assert no_op.evolution.n == 1
    assert no_op.evolution.mean == pytest.approx(0.5)
    assert no_op.deployment.n == 1
    assert no_op.deployment.mean == pytest.approx(1.0)
    assert speed.evolution.n == 0
    assert speed.deployment.n == 0
    assert trainer.scheduler.experiment_count(cell, NO_OP) == 1
    assert simulator.commits[0]["lever"] == ActionLever.NO_OP.value


def test_selected_no_op_uses_certified_ordinary_contender() -> None:
    speed = _rule(SPEED, rival_mean=1.0, deployment_mean=2.0, certified=True)
    no_op = _rule(NO_OP)
    population = Population((speed, no_op), max_size=100)
    context = _context(NO_OP, SPEED)
    calls: list[dict[str, str]] = []
    trainer = _trainer(
        context,
        population,
        calls,
        scores={SPEED.key: 3.0, NO_OP.key: 1.0},
        config=_config(exploration_coverage_floor=2),
    )
    _prime_scheduler(trainer, context, undercovered=NO_OP)
    speed_evolution = _copy(speed.evolution)
    speed_deployment = _copy(speed.deployment)
    simulator = FakeSimulator(FakeDefinition())

    result = trainer.process_epoch(simulator, object())

    assert calls == [
        {"selected": NO_OP.key, "contender": SPEED.key, "no_op": NO_OP.key}
    ]
    assert result.trace.delta_selected_noop == 0.0
    assert _added_sample(speed_evolution, speed.evolution) == pytest.approx(2.0)
    assert _added_sample(speed_deployment, speed.deployment) == pytest.approx(2.0)
    assert result.trace.ledger_recipients["selected_evolution"] == []
    assert result.trace.contender_selection["selected_rank"] == 1
    assert result.trace.contender_selection["fallback"] is None
    ranked = result.trace.contender_selection["ranked_candidates"][0]
    assert ranked["rank"] == 1
    assert ranked["action"] == SPEED.to_dict()
    assert ranked["source_rule_ids"] == [speed.rule_id]
    assert ranked["score"] == pytest.approx(
        speed_evolution.lcb(
            trainer.config.contender_lcb_z,
            trainer.config.variance_floor,
        )
    )
    assert len(simulator.commits) == 1
    assert simulator.commits[0]["lever"] == ActionLever.NO_OP.value


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        pytest.param("parent_hash", "current real parent", id="parent-hash"),
        pytest.param("policy_fingerprint", "frozen continuation policy", id="policy"),
        pytest.param("horizon", "configured horizon", id="horizon"),
        pytest.param("label", "arm label", id="arm-label"),
        pytest.param("initial_hash", "common parent", id="arm-parent"),
        pytest.param("selected_action", "unexpected initial action", id="exact-action"),
        pytest.param("arm_c_action", "Arm C is not canonical no-op", id="arm-c"),
        pytest.param("score", "invalid arm scores", id="finite-scores"),
        pytest.param("delta_rival", "inconsistent delta_rival", id="rival-delta"),
        pytest.param(
            "delta_selected_noop",
            "inconsistent delta_selected_noop",
            id="selected-noop-delta",
        ),
        pytest.param(
            "delta_contender_noop",
            "inconsistent delta_contender_noop",
            id="contender-noop-delta",
        ),
        pytest.param("delta_veto", "inconsistent delta_veto", id="veto-delta"),
    ),
)
def test_rejects_malformed_rollout_hook_evidence_before_credit_or_commit(
    fault: str,
    message: str,
) -> None:
    speed = _rule(SPEED, rival_mean=1.0, deployment_mean=2.0, certified=True)
    no_op = _rule(NO_OP)
    population = Population((speed, no_op), max_size=100)
    context = _context(NO_OP, SPEED)
    calls: list[dict[str, str]] = []
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        population=population,
        config=_config(),
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_corrupting_rollout_hook(
            {SPEED.key: 3.0, NO_OP.key: 1.0},
            calls,
            fault,
        ),
        hook_fingerprints=_test_hook_fingerprints(
            context,
            {SPEED.key: 3.0, NO_OP.key: 1.0},
            rollout_variant=f"corrupt:{fault}",
        ),
    )
    trainer._publish_if_due(epoch=trainer.config.certification_interval)
    trainer.epoch = trainer.config.certification_interval
    before_epoch = trainer.epoch
    _prime_scheduler(trainer, context)
    before_population = population.to_dict()
    simulator = FakeSimulator(FakeDefinition())

    with pytest.raises(RuntimeError, match=message):
        trainer.process_epoch(simulator, object())

    assert population.to_dict() == before_population
    assert simulator.commits == []
    assert trainer.epoch == before_epoch


def test_no_physical_action_skips_rollout_and_commit() -> None:
    context = _context(NO_OP)
    calls: list[dict[str, str]] = []
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=_config(),
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook({NO_OP.key: 1.0}, calls),
        hook_fingerprints=_test_hook_fingerprints(
            context,
            {NO_OP.key: 1.0},
        ),
    )
    simulator = FakeSimulator(FakeDefinition())

    result = trainer.process_epoch(simulator, object())

    assert result.skipped_reason == "no_non_noop_action"
    assert result.rollout is None
    assert calls == []
    assert simulator.commits == []


def test_rulebook_changes_only_on_certification_tick() -> None:
    context = _context(NO_OP)
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=_config(certification_interval=3),
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook({NO_OP.key: 0.0}, []),
        hook_fingerprints=_test_hook_fingerprints(
            context,
            {NO_OP.key: 0.0},
        ),
    )
    simulator = FakeSimulator(FakeDefinition())
    initial = trainer.rulebook_hash

    first = trainer.process_epoch(simulator, object())
    second = trainer.process_epoch(simulator, object())
    third = trainer.process_epoch(simulator, object())

    assert first.evaluation_snapshot.rulebook_hash == initial
    assert second.evaluation_snapshot.rulebook_hash == initial
    assert third.trace.rulebook_hash == initial
    assert third.evaluation_snapshot.rulebook_hash != initial
    assert len(third.trace.certification_events) == 1


def test_publication_freezes_complete_evidence_and_checkpoint_round_trips() -> None:
    speed = _rule(
        SPEED,
        rival_mean=1.5,
        deployment_mean=2.0,
        certified=True,
    )
    veto = _rule(
        NO_OP,
        rival_mean=2.5,
        certified=True,
    )
    population = Population((speed, veto), max_size=100)
    config = _config(certification_interval=5)
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        population=population,
        config=config,
        schema=SCHEMA,
    )

    events = trainer._publish_if_due(epoch=config.certification_interval)
    trainer.epoch = config.certification_interval
    published = trainer.evaluation_snapshot
    evidence = published.certification_evidence[speed.rule_id]
    serialized = published.to_dict()["certification_evidence"][speed.rule_id]

    assert evidence.thresholds == CertificationThresholds.from_config(config)
    assert evidence.independent_evolution.to_dict() == speed.evolution.to_dict()
    assert evidence.independent_deployment.to_dict() == speed.deployment.to_dict()
    assert evidence.noop_samples == speed.deployment.n
    assert evidence.noop_lcb > 0.0
    assert set(serialized) == {
        "artifact_version",
        "source_rule_id",
        "credit_mode",
        "rule_kind",
        "action",
        "independent_evolution",
        "independent_deployment",
        "thresholds",
        "content_hash",
    }
    assert events[0]["evaluation_snapshot_hash"] == published.content_hash
    assert events[0]["certification_evidence_hashes"] == {
        speed.rule_id: evidence.content_hash,
        veto.rule_id: published.certification_evidence[veto.rule_id].content_hash,
    }
    veto_evidence = published.certification_evidence[veto.rule_id]
    assert veto_evidence.rule_kind == "veto"
    assert veto_evidence.action == NO_OP
    assert veto_evidence.independent_evolution.to_dict() == veto.evolution.to_dict()
    with pytest.raises(ValueError, match="only for causal action evidence"):
        _ = veto_evidence.noop_lcb

    published_payload = published.to_dict()
    speed.evolution.update(-99.0)
    speed.deployment.update(99.0)
    assert published.to_dict() == published_payload
    veto.evolution.update(-99.0)
    assert trainer.evaluation_snapshot.to_dict() == published_payload

    checkpoint = trainer.checkpoint(FakeSimulator(FakeDefinition()))
    restored = type(checkpoint).from_dict(checkpoint.to_dict())
    assert restored.evaluation_snapshot.to_dict() == published_payload
    assert (
        restored.evaluation_snapshot.certification_evidence[speed.rule_id].to_dict()
        == evidence.to_dict()
    )


def test_publication_evidence_excludes_offspring_birth_baselines() -> None:
    parent = _rule(
        SPEED,
        rival_mean=50.0,
        deployment_mean=50.0,
        certified=True,
    )
    child = make_offspring(
        (parent,),
        parent.condition,
        epoch=3,
        operator="reproduction",
        evidence_discount=1.0,
        action_certification_min_samples=2,
        veto_certification_min_samples=2,
    )
    child.evolution.extend((1.0, 1.0))
    child.deployment.extend((2.0, 2.0))
    assert child.evolution.n == 3
    assert child.deployment.n == 3

    config = _config(certification_interval=5)
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        population=Population((child,), max_size=100),
        config=config,
        schema=SCHEMA,
    )
    trainer._publish_if_due(epoch=config.certification_interval)
    evidence = trainer.evaluation_snapshot.certification_evidence[child.rule_id]

    assert evidence.independent_evolution.n == 2
    assert evidence.independent_evolution.mean == pytest.approx(1.0)
    assert evidence.independent_deployment.n == 2
    assert evidence.independent_deployment.mean == pytest.approx(2.0)
    assert evidence.noop_samples == 2
    assert evidence.noop_lcb > 0.0


def test_exploration_disabled_decision_ignores_live_population_between_publications() -> (
    None
):
    speed = _rule(
        SPEED,
        rival_mean=1.0,
        deployment_mean=2.0,
        certified=True,
    )
    population = Population((speed,), max_size=100)
    context = _context(NO_OP, SPEED, STRETCH)
    trainer = _trainer(
        context,
        population,
        [],
        scores={NO_OP.key: 0.0, SPEED.key: 2.0, STRETCH.key: 9.0},
    )
    simulator = FakeSimulator(FakeDefinition())
    simulator.state = FakeCheckpointState(
        state_id="after",
        parent_state_id="before",
        sim_time_s=12.0,
        version=7,
        decision_epoch_index=3,
    )
    batch = EventBatchResult(
        time_s=12.0,
        events=(),
        decision_epoch=DecisionEpoch(
            epoch_index=3,
            time_s=12.0,
            state_version=7,
            state_id="after",
            trigger_event_ids=("event-1",),
        ),
        state_id_before="before",
        state_id_after="after",
    )
    published_payload = trainer.evaluation_snapshot.to_dict()
    parent_hash = simulator.dynamic_content_hash

    before = trainer.exploration_disabled_decision(simulator, batch)
    assert before.action == SPEED

    population.remove(speed.rule_id, numerosity=speed.numerosity)
    population.add(
        _rule(
            STRETCH,
            rival_mean=100.0,
            deployment_mean=100.0,
            certified=True,
        )
    )
    after = trainer.exploration_disabled_decision(simulator, batch)

    assert after == before
    assert trainer.evaluation_snapshot.to_dict() == published_payload
    assert simulator.dynamic_content_hash == parent_hash


def test_checkpoint_replay_restores_rng_scheduler_population_and_commit() -> None:
    speed = _rule(SPEED, rival_mean=1.0, deployment_mean=2.0, certified=True)
    no_op = _rule(NO_OP)
    population = Population((speed, no_op), max_size=100)
    context = _context(NO_OP, SPEED)
    calls: list[dict[str, str]] = []
    config = _config(certification_interval=50)
    trainer = _trainer(
        context,
        population,
        calls,
        scores={SPEED.key: 3.0, NO_OP.key: 1.0},
        config=config,
    )
    _prime_scheduler(trainer, context)
    definition = FakeDefinition()
    simulator = FakeSimulator(definition)
    decision_reference = {"adapter_batch_id": "causal-epoch-1"}
    checkpoint = trainer.checkpoint(simulator, decision_reference=decision_reference)
    round_tripped = type(checkpoint).from_dict(checkpoint.to_dict())

    assert (
        checkpoint.pending_decision_reference
        == checkpoint.rng_state["pending_decision"]
    )
    assert (
        round_tripped.pending_decision_reference
        == checkpoint.pending_decision_reference
    )

    original = trainer.process_epoch(simulator, object())
    replay_calls: list[dict[str, str]] = []
    resumed, resumed_simulator = CausalTrainer.resume_from_checkpoint(
        checkpoint,
        definition=definition,
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook(
            {SPEED.key: 3.0, NO_OP.key: 1.0},
            replay_calls,
        ),
        hook_fingerprints=_test_hook_fingerprints(
            context,
            {SPEED.key: 3.0, NO_OP.key: 1.0},
        ),
    )
    assert resumed.pending_decision_reference == checkpoint.pending_decision_reference
    replay = resumed.process_epoch(
        resumed_simulator,
        object(),
        decision_reference=decision_reference,
    )
    assert resumed.pending_decision_reference is None

    assert original.trace.to_dict() == replay.trace.to_dict()
    assert trainer.population.to_dict() == resumed.population.to_dict()
    assert simulator.dynamic_content_hash == resumed_simulator.dynamic_content_hash
    assert calls == replay_calls
    assert len(simulator.commits) == len(resumed_simulator.commits) == 1


def test_custom_hooks_require_explicit_nonblank_fingerprints() -> None:
    context = _context(NO_OP)
    scores = {NO_OP.key: 0.0}

    with pytest.raises(ValueError, match="custom context_builder"):
        CausalTrainer(
            FakeRuntime(),  # type: ignore[arg-type]
            config=_config(),
            schema=SCHEMA,
            context_builder=_context_builder(context),
        )
    with pytest.raises(ValueError, match="custom outcome_plan_factory"):
        CausalTrainer(
            FakeRuntime(),  # type: ignore[arg-type]
            config=_config(),
            schema=SCHEMA,
            outcome_plan_factory=_plan,
        )
    with pytest.raises(ValueError, match="custom rollout_hook"):
        CausalTrainer(
            FakeRuntime(),  # type: ignore[arg-type]
            config=_config(),
            schema=SCHEMA,
            rollout_hook=_rollout_hook(scores, []),
        )
    with pytest.raises(ValueError, match="non-empty string"):
        CausalTrainer(
            FakeRuntime(),  # type: ignore[arg-type]
            config=_config(),
            schema=SCHEMA,
            context_builder=_context_builder(context),
            outcome_plan_factory=_plan,
            rollout_hook=_rollout_hook(scores, []),
            hook_fingerprints={
                **_test_hook_fingerprints(context, scores),
                "rollout_hook": " ",
            },
        )


def test_resume_rejects_changed_hook_identity_or_configuration() -> None:
    context = _context(NO_OP)
    scores = {NO_OP.key: 0.0}
    fingerprints = _test_hook_fingerprints(context, scores)
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=_config(),
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook(scores, []),
        hook_fingerprints=fingerprints,
    )
    definition = FakeDefinition()
    checkpoint = trainer.checkpoint(FakeSimulator(definition))
    changed = dict(fingerprints)
    changed["rollout_hook"] = content_hash(
        {"variant": "three_arm", "scores": {NO_OP.key: 99.0}},
        namespace="test.trainer.rollout_hook.v1",
    )

    assert checkpoint.rng_state["hook_fingerprints"] == fingerprints
    with pytest.raises(ValueError, match="hook fingerprints"):
        CausalTrainer.resume_from_checkpoint(
            checkpoint,
            definition=definition,
            runtime=FakeRuntime(),  # type: ignore[arg-type]
            schema=SCHEMA,
            context_builder=_context_builder(context),
            outcome_plan_factory=_plan,
            rollout_hook=_rollout_hook({NO_OP.key: 99.0}, []),
            hook_fingerprints=changed,
        )


def test_resumed_decision_reference_is_exact_and_consumed_once() -> None:
    context = _context(NO_OP)
    scores = {NO_OP.key: 0.0}
    fingerprints = _test_hook_fingerprints(context, scores)
    definition = FakeDefinition()
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=_config(),
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook(scores, []),
        hook_fingerprints=fingerprints,
    )
    checkpoint = trainer.checkpoint(
        FakeSimulator(definition),
        decision_reference={"batch_id": "expected"},
    )
    resumed, simulator = CausalTrainer.resume_from_checkpoint(
        checkpoint,
        definition=definition,
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook(scores, []),
        hook_fingerprints=fingerprints,
    )

    with pytest.raises(ValueError, match="requires the pending"):
        resumed.process_epoch(simulator, object())
    with pytest.raises(ValueError, match="does not match checkpoint"):
        resumed.process_epoch(
            simulator,
            object(),
            decision_reference={"batch_id": "different"},
        )
    result = resumed.process_epoch(
        simulator,
        object(),
        decision_reference={"batch_id": "expected"},
    )

    assert result.skipped_reason == "no_non_noop_action"
    assert resumed.pending_decision_reference is None
    with pytest.raises(ValueError, match="without a pending"):
        resumed.process_epoch(
            simulator,
            object(),
            decision_reference={"batch_id": "expected"},
        )


def test_real_event_batch_binding_requires_exact_checkpoint_state() -> None:
    context = _context(NO_OP)
    scores = {NO_OP.key: 0.0}
    fingerprints = _test_hook_fingerprints(context, scores)
    definition = FakeDefinition()
    simulator = FakeSimulator(definition)
    simulator.state = FakeCheckpointState(
        state_id="after",
        parent_state_id="before",
        sim_time_s=12.0,
        version=7,
        decision_epoch_index=3,
    )
    batch = EventBatchResult(
        time_s=12.0,
        events=(),
        decision_epoch=DecisionEpoch(
            epoch_index=3,
            time_s=12.0,
            state_version=7,
            state_id="after",
            trigger_event_ids=("event-1",),
        ),
        state_id_before="before",
        state_id_after="after",
    )
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=_config(),
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook(scores, []),
        hook_fingerprints=fingerprints,
    )

    with pytest.raises(ValueError, match="does not end"):
        trainer.checkpoint(
            simulator,
            event_batch=replace(batch, state_id_after="unrelated"),
        )
    checkpoint = trainer.checkpoint(simulator, event_batch=batch)
    resumed, resumed_simulator = CausalTrainer.resume_from_checkpoint(
        checkpoint,
        definition=definition,
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook(scores, []),
        hook_fingerprints=fingerprints,
    )

    with pytest.raises(ValueError, match="does not match checkpoint"):
        resumed.process_epoch(
            resumed_simulator,
            replace(batch, state_id_before="wrong"),
        )
    result = resumed.process_epoch(resumed_simulator, batch)
    assert result.skipped_reason == "no_non_noop_action"


@pytest.mark.parametrize("epoch", (True, 1.5, "1"))
def test_trainer_epoch_requires_exact_integer(epoch: Any) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        CausalTrainer(
            FakeRuntime(),  # type: ignore[arg-type]
            config=_config(),
            schema=SCHEMA,
            epoch=epoch,
        )


def test_snapshot_rejects_ranking_drift_and_impossible_chronology() -> None:
    runtime = FakeRuntime()
    config = _config(certification_interval=3)
    base = CausalTrainer(
        runtime,  # type: ignore[arg-type]
        config=config,
        schema=SCHEMA,
    )
    wrong_ranking = EvaluationSnapshot(
        base.current_rulebook,
        publication_epoch=0,
        contender_ranking_fields={"score": "tampered"},
        config_hash=base.config_hash,
        action_vocabulary_hash=runtime.action_vocabulary_hash,
    )
    with pytest.raises(ValueError, match="ranking metadata"):
        CausalTrainer(
            runtime,  # type: ignore[arg-type]
            config=config,
            schema=SCHEMA,
            evaluation_snapshot=wrong_ranking,
        )

    future_rulebook = FrozenRulebookPolicy(
        feature_schema_hash=SCHEMA.schema_hash,
        certification_generation=3,
        contender_min_samples=config.contender_min_samples,
        action_configuration=runtime.vocabulary.payload,
    )
    future = EvaluationSnapshot(
        future_rulebook,
        publication_epoch=3,
        contender_ranking_fields=base.evaluation_snapshot.contender_ranking_fields,
        config_hash=base.config_hash,
        action_vocabulary_hash=runtime.action_vocabulary_hash,
    )
    with pytest.raises(ValueError, match="cannot exceed trainer epoch"):
        CausalTrainer(
            runtime,  # type: ignore[arg-type]
            config=config,
            schema=SCHEMA,
            evaluation_snapshot=future,
            epoch=2,
        )

    mismatched_generation = EvaluationSnapshot(
        future_rulebook,
        publication_epoch=0,
        contender_ranking_fields=base.evaluation_snapshot.contender_ranking_fields,
        config_hash=base.config_hash,
        action_vocabulary_hash=runtime.action_vocabulary_hash,
    )
    with pytest.raises(ValueError, match="does not match its rulebook"):
        CausalTrainer(
            runtime,  # type: ignore[arg-type]
            config=config,
            schema=SCHEMA,
            evaluation_snapshot=mismatched_generation,
            epoch=3,
        )


def test_vanilla_checkpoint_restores_mode_settings_and_exact_replay() -> None:
    speed = _rule(
        SPEED,
        rival_mean=0.8,
        deployment_mean=2.0,
        certified=True,
    )
    no_op = _rule(NO_OP)
    population = Population(
        (speed, no_op), max_size=100, credit_mode="vanilla_accuracy"
    )
    context = _context(NO_OP, SPEED)
    calls: list[dict[str, str]] = []
    assigner = VanillaAccuracyCreditAssigner(
        initial_prediction=1.25,
        accuracy_scale=2.5,
    )
    config = _config(certification_interval=50)
    trainer = _trainer(
        context,
        population,
        calls,
        scores={SPEED.key: 5.0, NO_OP.key: 1.0},
        config=config,
        credit_assigner=assigner,
    )
    _prime_scheduler(trainer, context)
    definition = FakeDefinition()
    simulator = FakeSimulator(definition)
    decision_reference = {"adapter_batch_id": "vanilla-epoch-1"}
    checkpoint = trainer.checkpoint(simulator, decision_reference=decision_reference)
    assert checkpoint.population.credit_mode == "vanilla_accuracy"
    assert checkpoint.evaluation_snapshot.rulebook.credit_mode == "vanilla_accuracy"
    assert checkpoint.to_dict()["population"]["credit_mode"] == "vanilla_accuracy"
    vanilla_evidence = checkpoint.evaluation_snapshot.certification_evidence[
        speed.rule_id
    ]
    assert vanilla_evidence.credit_mode == "vanilla_accuracy"
    assert vanilla_evidence.rule_kind == "action"
    assert vanilla_evidence.independent_evolution.mean == pytest.approx(0.8)
    assert vanilla_evidence.independent_deployment.mean == pytest.approx(2.0)
    with pytest.raises(ValueError, match="only for causal action evidence"):
        _ = vanilla_evidence.noop_lcb

    causal = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=config,
        schema=SCHEMA,
    )
    changed_settings = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=config,
        schema=SCHEMA,
        credit_assigner=VanillaAccuracyCreditAssigner(
            initial_prediction=1.25,
            accuracy_scale=3.0,
        ),
    )
    assert trainer.config_hash != causal.config_hash
    assert trainer.config_hash != changed_settings.config_hash
    assert checkpoint.rng_state["credit"] == {
        "mode": "vanilla_accuracy",
        "settings": {
            "initial_prediction": 1.25,
            "accuracy_scale": 2.5,
        },
    }

    original = trainer.process_epoch(simulator, object())
    replay_calls: list[dict[str, str]] = []
    resumed, resumed_simulator = CausalTrainer.resume_from_checkpoint(
        checkpoint,
        definition=definition,
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook(
            {SPEED.key: 5.0, NO_OP.key: 1.0},
            replay_calls,
        ),
        hook_fingerprints=_test_hook_fingerprints(
            context,
            {SPEED.key: 5.0, NO_OP.key: 1.0},
        ),
    )
    replay = resumed.process_epoch(
        resumed_simulator,
        object(),
        decision_reference=decision_reference,
    )

    assert resumed.credit_mode == "vanilla_accuracy"
    assert resumed.population.credit_mode == "vanilla_accuracy"
    assert resumed.current_rulebook.credit_mode == "vanilla_accuracy"
    assert resumed.credit_assigner == assigner
    assert original.trace.to_dict() == replay.trace.to_dict()
    assert trainer.population.to_dict() == resumed.population.to_dict()
    assert simulator.dynamic_content_hash == resumed_simulator.dynamic_content_hash
    assert calls == replay_calls

    with pytest.raises(ValueError, match="credit mode/settings"):
        CausalTrainer.resume_from_checkpoint(
            checkpoint,
            definition=definition,
            runtime=FakeRuntime(),  # type: ignore[arg-type]
            schema=SCHEMA,
            credit_assigner=VanillaAccuracyCreditAssigner(
                initial_prediction=1.25,
                accuracy_scale=9.0,
            ),
            context_builder=_context_builder(context),
            outcome_plan_factory=_plan,
            rollout_hook=_rollout_hook(
                {SPEED.key: 5.0, NO_OP.key: 1.0},
                [],
            ),
            hook_fingerprints=_test_hook_fingerprints(
                context,
                {SPEED.key: 5.0, NO_OP.key: 1.0},
            ),
        )


def test_checkpoint_resume_rejects_changed_stretch_runtime_configuration() -> None:
    context = _context(NO_OP)
    original_runtime = FakeRuntime()
    trainer = CausalTrainer(
        original_runtime,  # type: ignore[arg-type]
        config=_config(),
        schema=SCHEMA,
        context_builder=_context_builder(context),
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook({NO_OP.key: 0.0}, []),
        hook_fingerprints=_test_hook_fingerprints(
            context,
            {NO_OP.key: 0.0},
        ),
    )
    definition = FakeDefinition()
    checkpoint = trainer.checkpoint(FakeSimulator(definition))
    changed_runtime = FakeRuntime(
        stretch_config=StretchConfig(candidate_azimuth_count=32),
    )

    assert (
        changed_runtime.runtime_configuration_hash
        != original_runtime.runtime_configuration_hash
    )
    with pytest.raises(
        ValueError,
        match="evaluation snapshot configuration does not match trainer",
    ):
        CausalTrainer.resume_from_checkpoint(
            checkpoint,
            definition=definition,
            runtime=changed_runtime,  # type: ignore[arg-type]
            schema=SCHEMA,
            context_builder=_context_builder(context),
            outcome_plan_factory=_plan,
            rollout_hook=_rollout_hook({NO_OP.key: 0.0}, []),
            hook_fingerprints=_test_hook_fingerprints(
                context,
                {NO_OP.key: 0.0},
            ),
        )


def test_trainer_hash_includes_exploration_partition_boundaries() -> None:
    config = _config()
    default = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=config,
        schema=SCHEMA,
        scheduler=RegionExplorationScheduler(config),
    )
    changed = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=config,
        schema=SCHEMA,
        scheduler=RegionExplorationScheduler(
            config,
            commitment_thresholds=(0.2, 0.8),
        ),
    )

    assert (
        default.scheduler.partition_configuration
        != changed.scheduler.partition_configuration
    )
    assert default.config_hash != changed.config_hash


def test_fallback_root_ignores_no_op_only_anchor() -> None:
    no_op_only = _context(NO_OP, anchor_id="anchor-a")
    physical = _context(NO_OP, SPEED, anchor_id="anchor-b")
    calls: list[dict[str, str]] = []

    def contexts(_simulator: Any, _batch: Any):
        return (
            (SourceAnchor(no_op_only.anchor_id), no_op_only),
            (SourceAnchor(physical.anchor_id), physical),
        )

    config = _config(exploration_coverage_floor=2)
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        config=config,
        schema=SCHEMA,
        context_builder=contexts,
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook(
            {NO_OP.key: 1.0, SPEED.key: 3.0},
            calls,
        ),
        hook_fingerprints=_test_hook_fingerprints(
            (no_op_only, physical),
            {NO_OP.key: 1.0, SPEED.key: 3.0},
        ),
    )
    _prime_scheduler(trainer, physical, undercovered=SPEED)
    simulator = FakeSimulator(FakeDefinition())

    result = trainer.process_epoch(simulator, object())

    assert result.skipped_reason is None
    assert result.trace.selected_anchor_id == physical.anchor_id
    assert calls[0]["selected"] == SPEED.key
    assert len(simulator.commits) == 1


def test_low_capacity_multi_anchor_covering_selects_only_a_live_advocate() -> None:
    first = _context(NO_OP, SPEED, anchor_id="anchor-a")
    second_vector = SCHEMA.encode(
        {
            "commitment_fraction": 0.9,
            "pressure_ratio": 2.5,
            "abs_spacing_deviation_s": 250.0,
        }
    )
    second = AnchorContext(
        role_type="leader_follower",
        schema_hash=SCHEMA.schema_hash,
        anchor_id="anchor-b",
        vector=second_vector,
        candidates=(
            _candidate(NO_OP, "anchor-b"),
            _candidate(SPEED, "anchor-b"),
        ),
    )
    calls: list[dict[str, str]] = []

    def contexts(_simulator: Any, _batch: Any):
        return (
            (SourceAnchor(first.anchor_id), first),
            (SourceAnchor(second.anchor_id), second),
        )

    config = _config(
        population_limit=1,
        exploration_coverage_floor=2,
    )
    population = Population(max_size=1)
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        population=population,
        config=config,
        schema=SCHEMA,
        context_builder=contexts,
        outcome_plan_factory=_plan,
        rollout_hook=_rollout_hook(
            {NO_OP.key: 1.0, SPEED.key: 3.0},
            calls,
        ),
        hook_fingerprints=_test_hook_fingerprints(
            (first, second),
            {NO_OP.key: 1.0, SPEED.key: 3.0},
        ),
    )
    _prime_scheduler(trainer, first, undercovered=SPEED)

    result = trainer.process_epoch(FakeSimulator(FakeDefinition()), object())

    assert result.skipped_reason is None
    assert result.trace.selected_anchor_id == second.anchor_id
    assert calls == [
        {
            "selected": SPEED.key,
            "contender": NO_OP.key,
            "no_op": NO_OP.key,
        }
    ]
    assert build_match_set(first, population).advocates(SPEED) == ()
    second_advocates = build_match_set(second, population).advocates(SPEED)
    assert second_advocates == result.trace.coadvocate_rule_ids
    assert population.total_numerosity == 1


def test_at_capacity_multi_niche_ga_prepares_all_births_before_deletion() -> None:
    speed = _rule(SPEED, rival_mean=-100.0, certified=True)
    stretch = _rule(STRETCH, rival_mean=100.0, certified=True)
    population = Population((speed, stretch), max_size=2)
    context = _context(SPEED, STRETCH)
    config = _config(
        population_limit=2,
        ga_interval=1,
        mutation_probability=0.0,
        crossover_probability=0.0,
    )
    trainer = CausalTrainer(
        FakeRuntime(),  # type: ignore[arg-type]
        population=population,
        config=config,
        schema=SCHEMA,
    )

    events = trainer._run_ga(
        build_match_set(context, population),
        selected_action=SPEED,
        contender_action=STRETCH,
        epoch=1,
    )

    assert [event["action"] for event in events] == [
        STRETCH.to_dict(),
        SPEED.to_dict(),
    ]
    assert events[0]["eligible_parent_ids"] == [stretch.rule_id]
    assert events[1]["eligible_parent_ids"] == [speed.rule_id]
    assert population.total_numerosity == 2
    with pytest.raises(KeyError):
        population.get(speed.rule_id)
    surviving_speed = population.by_action(SPEED)
    assert len(surviving_speed) == 1
    assert surviving_speed[0].rule_id == events[1]["offspring_rule_ids"][0]
    assert surviving_speed[0].evolution.n == 1


def test_ga_tick_runs_conservative_same_niche_subsumption() -> None:
    general = _rule(
        SPEED,
        rival_mean=5.0,
        deployment_mean=2.0,
        certified=True,
    )
    specific = MutableRule(
        condition=RuleCondition(
            "leader_follower",
            SCHEMA.schema_hash,
            {"commitment_fraction": Interval(0.3, 0.5)},
        ),
        action=SPEED,
        evolution=OnlineMoments(n=3, mean=1.0, m2=0.0),
        deployment=OnlineMoments(n=3, mean=1.0, m2=0.0),
        provenance={"fixture": "specific-speed"},
    )
    no_op = _rule(NO_OP)
    population = Population((general, specific, no_op), max_size=100)
    context = _context(NO_OP, SPEED)
    calls: list[dict[str, str]] = []
    trainer = _trainer(
        context,
        population,
        calls,
        scores={SPEED.key: 3.0, NO_OP.key: 1.0},
        config=_config(
            ga_interval=1,
            mutation_probability=0.0,
            crossover_probability=0.0,
        ),
    )
    _prime_scheduler(trainer, context)

    result = trainer.process_epoch(FakeSimulator(FakeDefinition()), object())

    assert len(result.trace.ga_events) == 1
    assert specific.rule_id in result.trace.ga_events[0]["absorbed_rule_ids"]
    with pytest.raises(KeyError):
        population.get(specific.rule_id)
    assert population.get(general.rule_id).numerosity >= 2
