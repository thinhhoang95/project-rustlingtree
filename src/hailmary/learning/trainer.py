"""Explicit coordinator for one causal SEQD training decision.

The trainer deliberately keeps root experimentation separate from rollout
continuation.  Mutable population rules may advocate the root action, while
all temporary arms continue under one detached ``EvaluationSnapshot``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from hailmary.config import (
    FeatureConfig,
    LearningConfig,
    OutcomeConfig,
    ScenarioConfig,
)
from hailmary.evaluation.outcome import simulator_outcome_plan
from hailmary.features import (
    build_current_segment_anchors,
    leader_follower_feature_schema,
    simulator_state_vector,
)
from hailmary.features.schema import FeatureSchema
from hailmary.ids import canonical_data, content_hash
from hailmary.learning.artifacts import (
    EpochTrace,
    EvaluationSnapshot,
    PublishedCertificationEvidence,
    TrainingCheckpoint,
    capture_rng_state,
)
from hailmary.learning.certification import CertificationThresholds
from hailmary.learning.covering import cover_missing_actions
from hailmary.learning.credit import (
    CAUSAL_CREDIT_MODE,
    VANILLA_ACCURACY_CREDIT_MODE,
    CausalCreditAssigner,
    CreditAssigner,
    CreditRecipients,
    CreditSignals,
    VanillaAccuracyCreditAssigner,
    credit_assigner_configuration,
    credit_assigner_from_configuration,
)
from hailmary.learning.evolution import (
    can_subsume,
    evolve_niche,
    insert_offspring,
    subsume_rule,
)
from hailmary.learning.exploration import (
    ExplorationDecision,
    RegionExplorationScheduler,
)
from hailmary.learning.matching import AnchorContext, MatchSet, build_match_set
from hailmary.learning.population import Population
from hailmary.learning.rulebook import (
    FrozenRulebookPolicy,
    RulebookDecision,
    RulebookDecisionRecord,
    SimulatorRulebookPolicy,
)
from hailmary.learning.rules import RuleAction
from hailmary.rollout.paired import (
    PairedArmTrace,
    ThreeArmRolloutResult,
    dynamic_content_fingerprint,
    three_arm_simulator_rollout,
)
from hailmary.runtime import ActionRuntime
from hailmary.simulator.events import EventBatchResult


TRAINER_STATE_SCHEMA_VERSION = "hailmary.causal_trainer.v3"
_HOOK_NAMES = (
    "context_builder",
    "outcome_plan_factory",
    "rollout_hook",
)
_DEFAULT_HOOK_FINGERPRINTS = {
    "context_builder": "hailmary.context_builder.default.v1",
    "outcome_plan_factory": "hailmary.simulator_outcome_plan.v1",
    "rollout_hook": "hailmary.three_arm_simulator_rollout.v1",
}
_DECISION_REFERENCE_NAMESPACE = "hailmary.trainer.pending_decision.v1"


ContextBuilder = Callable[[Any, Any], Sequence[tuple[Any, AnchorContext]]]
OutcomePlanFactory = Callable[..., Any]
RolloutHook = Callable[..., ThreeArmRolloutResult]


@dataclass(frozen=True, slots=True)
class TrainingEpochResult:
    """Concise result plus the complete content-addressed audit trace."""

    epoch: int
    trace: EpochTrace
    rollout: ThreeArmRolloutResult | None
    evaluation_snapshot: EvaluationSnapshot
    skipped_reason: str | None = None

    @property
    def committed(self) -> bool:
        return self.trace.committed_action is not None


@dataclass(frozen=True, slots=True)
class _PreparedAnchor:
    source: Any
    context: AnchorContext


class CausalTrainer:
    """Run three-arm causal epochs without exposing mutable rollout policy."""

    def __init__(
        self,
        runtime: ActionRuntime,
        *,
        population: Population | None = None,
        config: LearningConfig | None = None,
        outcome_config: OutcomeConfig | None = None,
        feature_config: FeatureConfig | None = None,
        scenario_config: ScenarioConfig | None = None,
        schema: FeatureSchema | None = None,
        scheduler: RegionExplorationScheduler | None = None,
        evaluation_snapshot: EvaluationSnapshot | None = None,
        epoch: int = 0,
        rng: np.random.Generator | None = None,
        credit_assigner: CreditAssigner | None = None,
        context_builder: ContextBuilder | None = None,
        outcome_plan_factory: OutcomePlanFactory = simulator_outcome_plan,
        rollout_hook: RolloutHook = three_arm_simulator_rollout,
        hook_fingerprints: Mapping[str, str] | None = None,
        pending_decision_reference: Mapping[str, Any] | None = None,
    ) -> None:
        self._validate_runtime(runtime)
        if type(epoch) is not int or epoch < 0:
            raise ValueError("trainer epoch must be a non-negative integer")
        self.runtime = runtime
        self.config = LearningConfig() if config is None else config
        self.outcome_config = (
            OutcomeConfig() if outcome_config is None else outcome_config
        )
        self.feature_config = (
            FeatureConfig() if feature_config is None else feature_config
        )
        self.scenario_config = (
            ScenarioConfig() if scenario_config is None else scenario_config
        )
        self.schema = (
            leader_follower_feature_schema(self.feature_config.schema_version)
            if schema is None
            else schema
        )
        if not isinstance(self.schema, FeatureSchema):
            raise TypeError("trainer schema must be FeatureSchema")

        self.credit_assigner = (
            CausalCreditAssigner() if credit_assigner is None else credit_assigner
        )
        self._credit_configuration = credit_assigner_configuration(self.credit_assigner)
        resolved_credit_mode = str(self._credit_configuration["mode"])
        self.population = (
            Population(
                max_size=self.config.population_limit,
                credit_mode=resolved_credit_mode,
            )
            if population is None
            else population
        )
        if not isinstance(self.population, Population):
            raise TypeError("trainer population must be Population")
        if self.population.total_numerosity > self.config.population_limit:
            raise ValueError("population already exceeds the configured limit")
        if self.population.credit_mode != resolved_credit_mode:
            raise ValueError(
                "population credit_mode does not match trainer credit mode"
            )

        self.rng = (
            np.random.default_rng(self.config.random_seed) if rng is None else rng
        )
        if not isinstance(self.rng, np.random.Generator):
            raise TypeError("trainer rng must be numpy.random.Generator")
        self.scheduler = (
            RegionExplorationScheduler(self.config, seed=self.config.random_seed)
            if scheduler is None
            else scheduler
        )
        if not isinstance(self.scheduler, RegionExplorationScheduler):
            raise TypeError("trainer scheduler must be RegionExplorationScheduler")
        if self.scheduler.config != self.config:
            raise ValueError(
                "scheduler configuration does not match trainer configuration"
            )

        self.epoch = epoch
        self._context_builder = context_builder
        if self._context_builder is not None and not callable(self._context_builder):
            raise TypeError("context_builder must be callable or None")
        self._outcome_plan_factory = outcome_plan_factory
        self._rollout_hook = rollout_hook
        if not callable(self._outcome_plan_factory) or not callable(self._rollout_hook):
            raise TypeError("outcome plan and rollout hooks must be callable")
        self._hook_fingerprints = self._resolve_hook_fingerprints(
            context_builder=context_builder,
            outcome_plan_factory=outcome_plan_factory,
            rollout_hook=rollout_hook,
            supplied=hook_fingerprints,
        )
        self._pending_decision_reference = self._validate_pending_decision_reference(
            pending_decision_reference,
            expected_next_epoch=self.epoch + 1,
        )

        self.config_hash = self._configuration_hash()
        self._evaluation_snapshot = (
            self._initial_evaluation_snapshot()
            if evaluation_snapshot is None
            else EvaluationSnapshot.from_dict(evaluation_snapshot.to_dict())
        )
        self._validate_evaluation_snapshot(self._evaluation_snapshot)

    @staticmethod
    def _validate_runtime(runtime: Any) -> None:
        required = (
            "catalog",
            "template_config",
            "vocabulary",
            "action_vocabulary_hash",
            "runtime_configuration_hash",
            "action_applier",
            "resume_simulator",
        )
        missing = tuple(name for name in required if not hasattr(runtime, name))
        if missing:
            raise TypeError("trainer runtime is missing " + ", ".join(sorted(missing)))

    @staticmethod
    def _resolve_hook_fingerprints(
        *,
        context_builder: ContextBuilder | None,
        outcome_plan_factory: OutcomePlanFactory,
        rollout_hook: RolloutHook,
        supplied: Mapping[str, str] | None,
    ) -> Mapping[str, str]:
        if supplied is None:
            raw: Mapping[str, str] = {}
        elif isinstance(supplied, Mapping):
            raw = supplied
        else:
            raise TypeError("hook_fingerprints must be a mapping")
        unknown = set(raw) - set(_HOOK_NAMES)
        if unknown:
            raise ValueError(
                "hook_fingerprints contains unknown hooks: "
                + ", ".join(sorted(str(name) for name in unknown))
            )

        hooks = {
            "context_builder": context_builder,
            "outcome_plan_factory": outcome_plan_factory,
            "rollout_hook": rollout_hook,
        }
        defaults = {
            "context_builder": context_builder is None,
            "outcome_plan_factory": outcome_plan_factory is simulator_outcome_plan,
            "rollout_hook": rollout_hook is three_arm_simulator_rollout,
        }
        resolved: dict[str, str] = {}
        for name in _HOOK_NAMES:
            if name in raw:
                value = raw[name]
                if type(value) is not str or not value.strip():
                    raise ValueError(
                        f"{name} hook fingerprint must be a non-empty string"
                    )
                resolved[name] = value.strip()
                continue
            if defaults[name]:
                resolved[name] = _DEFAULT_HOOK_FINGERPRINTS[name]
                continue
            if not callable(hooks[name]):
                raise TypeError(f"{name} must be callable")
            raise ValueError(f"custom {name} requires an explicit hook fingerprint")
        return canonical_data(resolved)

    @staticmethod
    def _validate_event_batch_against_simulator(
        event_batch: Any,
        simulator: Any,
    ) -> None:
        if not isinstance(event_batch, EventBatchResult):
            raise TypeError(
                "event_batch checkpoint binding requires EventBatchResult; "
                "use decision_reference for a custom event adapter"
            )
        state = getattr(simulator, "state", None)
        if state is None:
            raise TypeError(
                "EventBatchResult checkpoint binding requires simulator.state"
            )
        if event_batch.state_id_after != getattr(state, "state_id", None):
            raise ValueError(
                "checkpoint event batch does not end at the simulator state"
            )
        if event_batch.state_id_before != getattr(state, "parent_state_id", None):
            raise ValueError(
                "checkpoint event batch does not begin at the simulator parent state"
            )
        if event_batch.time_s != getattr(state, "sim_time_s", None):
            raise ValueError(
                "checkpoint event batch time does not match simulator state"
            )
        decision = event_batch.decision_epoch
        if decision is None:
            raise ValueError("checkpoint event batch must contain a decision epoch")
        expected = (
            getattr(state, "decision_epoch_index", None),
            getattr(state, "sim_time_s", None),
            getattr(state, "version", None),
            getattr(state, "state_id", None),
        )
        actual = (
            decision.epoch_index,
            decision.time_s,
            decision.state_version,
            decision.state_id,
        )
        if actual != expected:
            raise ValueError("checkpoint decision epoch does not match simulator state")

    @staticmethod
    def _build_pending_decision_reference(
        *,
        next_epoch: int,
        event_batch: Any | None,
        decision_reference: Any | None,
    ) -> Mapping[str, Any] | None:
        if event_batch is not None and decision_reference is not None:
            raise ValueError(
                "checkpoint accepts either event_batch or decision_reference, not both"
            )
        if event_batch is None and decision_reference is None:
            return None
        kind = "event_batch" if event_batch is not None else "decision_reference"
        value = event_batch if event_batch is not None else decision_reference
        try:
            payload = canonical_data(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{kind} must be canonical; use an explicit canonical "
                "decision_reference for custom event adapters"
            ) from exc
        if payload in (None, "", [], {}):
            raise ValueError(f"{kind} cannot be empty")
        fingerprint = content_hash(
            {"kind": kind, "payload": payload},
            namespace=_DECISION_REFERENCE_NAMESPACE,
        )
        return {
            "kind": kind,
            "next_epoch": next_epoch,
            "payload": payload,
            "fingerprint": fingerprint,
        }

    @staticmethod
    def _validate_pending_decision_reference(
        value: Mapping[str, Any] | None,
        *,
        expected_next_epoch: int,
    ) -> Mapping[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("pending decision reference must be a mapping")
        required = {"kind", "next_epoch", "payload", "fingerprint"}
        if set(value) != required:
            raise ValueError(
                "pending decision reference must contain exactly "
                + ", ".join(sorted(required))
            )
        kind = value["kind"]
        if kind not in {"event_batch", "decision_reference"}:
            raise ValueError("pending decision reference has an unsupported kind")
        next_epoch = value["next_epoch"]
        if type(next_epoch) is not int or next_epoch != expected_next_epoch:
            raise ValueError(
                "pending decision reference does not target the next trainer epoch"
            )
        payload = canonical_data(value["payload"])
        if payload in (None, "", [], {}):
            raise ValueError("pending decision reference payload cannot be empty")
        fingerprint = value["fingerprint"]
        if type(fingerprint) is not str or not fingerprint.strip():
            raise ValueError("pending decision fingerprint must be a non-empty string")
        expected = content_hash(
            {"kind": kind, "payload": payload},
            namespace=_DECISION_REFERENCE_NAMESPACE,
        )
        if fingerprint != expected:
            raise ValueError("pending decision fingerprint does not match its payload")
        return canonical_data(
            {
                "kind": kind,
                "next_epoch": next_epoch,
                "payload": payload,
                "fingerprint": fingerprint,
            }
        )

    def _configuration_hash(self) -> str:
        return content_hash(
            {
                "action_runtime_hash": self.runtime.runtime_configuration_hash,
                "learning": asdict(self.config),
                "outcome": asdict(self.outcome_config),
                "features": asdict(self.feature_config),
                "scenario": asdict(self.scenario_config),
                "exploration_partition": self.scheduler.partition_configuration,
                "credit": self._credit_configuration,
                "hooks": self._hook_fingerprints,
            },
            namespace="hailmary.causal_trainer.config.v3",
        )

    @property
    def hook_fingerprints(self) -> Mapping[str, str]:
        return dict(self._hook_fingerprints)

    @property
    def pending_decision_reference(self) -> Mapping[str, Any] | None:
        return canonical_data(self._pending_decision_reference)

    @property
    def evaluation_snapshot(self) -> EvaluationSnapshot:
        return EvaluationSnapshot.from_dict(self._evaluation_snapshot.to_dict())

    @property
    def current_rulebook(self) -> FrozenRulebookPolicy:
        return self._evaluation_snapshot.rulebook

    def exploration_disabled_decision(
        self,
        simulator: Any,
        event_batch: Any,
    ) -> RulebookDecision:
        """Select from the published snapshot without exploration or mutation."""

        self._validate_event_batch_against_simulator(event_batch, simulator)
        parent_hash = dynamic_content_fingerprint(simulator)
        rulebook = self.current_rulebook
        rulebook_hash = self.rulebook_hash
        records = tuple(
            RulebookDecisionRecord(
                anchor_id=prepared.context.anchor_id,
                features=prepared.context.vector,
                candidates=prepared.context.candidates,
                role_type=prepared.context.role_type,
                schema_hash=prepared.context.schema_hash,
            )
            for prepared in self._prepare_contexts(simulator, event_batch)
        )
        decision = rulebook.evaluate(records)
        if dynamic_content_fingerprint(simulator) != parent_hash:
            raise RuntimeError(
                "exploration-disabled exploit selection mutated the simulator"
            )
        if self.rulebook_hash != rulebook_hash:
            raise RuntimeError(
                "exploration-disabled exploit selection mutated the published rulebook"
            )
        return decision

    @property
    def rulebook_hash(self) -> str:
        return self._evaluation_snapshot.rulebook_hash

    @property
    def credit_mode(self) -> str:
        return str(self._credit_configuration["mode"])

    def _ranking_fields(self) -> Mapping[str, Any]:
        if self.credit_mode == VANILLA_ACCURACY_CREDIT_MODE:
            return {
                "score": "accuracy_weighted_predicted_reward",
                "minimum_samples": self.config.action_min_noop_samples,
                "prediction": "mean_selected_arm_outcome",
                "weight": "mean_accuracy",
                "tie_break": ["score_desc", "lever", "band", "source_rule_id"],
            }
        return {
            "score": "rival_lcb",
            "minimum_samples": self.config.contender_min_samples,
            "lcb_z": self.config.contender_lcb_z,
            "variance_floor": self.config.variance_floor,
            "tie_break": ["score_desc", "lever", "band", "source_rule_id"],
        }

    def _initial_evaluation_snapshot(self) -> EvaluationSnapshot:
        rulebook = FrozenRulebookPolicy(
            feature_schema_hash=self.schema.schema_hash,
            credit_mode=self.credit_mode,
            certification_generation=0,
            contender_min_samples=self.config.contender_min_samples,
            action_configuration=self.runtime.vocabulary.payload,
        )
        return EvaluationSnapshot(
            rulebook,
            publication_epoch=0,
            certified_source_rule_ids=(),
            contender_ranking_fields=self._ranking_fields(),
            config_hash=self.config_hash,
            action_vocabulary_hash=self.runtime.action_vocabulary_hash,
        )

    def _validate_evaluation_snapshot(self, snapshot: EvaluationSnapshot) -> None:
        if not isinstance(snapshot, EvaluationSnapshot):
            raise TypeError("evaluation_snapshot must be EvaluationSnapshot")
        if snapshot.feature_schema_hash != self.schema.schema_hash:
            raise ValueError(
                "evaluation snapshot feature schema does not match trainer"
            )
        if snapshot.action_vocabulary_hash != self.runtime.action_vocabulary_hash:
            raise ValueError(
                "evaluation snapshot action vocabulary does not match runtime"
            )
        if snapshot.config_hash != self.config_hash:
            raise ValueError("evaluation snapshot configuration does not match trainer")
        expected_thresholds = CertificationThresholds.from_config(self.config)
        if any(
            record.thresholds != expected_thresholds
            for record in snapshot.certification_evidence.values()
        ):
            raise ValueError(
                "evaluation snapshot certification thresholds do not match trainer"
            )
        if canonical_data(snapshot.contender_ranking_fields) != canonical_data(
            self._ranking_fields()
        ):
            raise ValueError(
                "evaluation snapshot ranking metadata does not match trainer"
            )
        if snapshot.publication_epoch != snapshot.rulebook.certification_generation:
            raise ValueError(
                "evaluation snapshot publication epoch does not match its rulebook"
            )
        if snapshot.publication_epoch > self.epoch:
            raise ValueError(
                "evaluation snapshot publication epoch cannot exceed trainer epoch"
            )
        if (
            snapshot.publication_epoch
            and snapshot.publication_epoch % self.config.certification_interval
        ):
            raise ValueError(
                "evaluation snapshot publication epoch is not a certification epoch"
            )
        if snapshot.rulebook.credit_mode != self.credit_mode:
            raise ValueError("evaluation snapshot credit_mode does not match trainer")
        # Constructing the wrapper also verifies the frozen physical action
        # configuration against the runtime vocabulary.
        self._simulator_policy(snapshot.rulebook)

    def _simulator_policy(
        self,
        rulebook: FrozenRulebookPolicy,
    ) -> SimulatorRulebookPolicy:
        return SimulatorRulebookPolicy(
            rulebook,
            template_config=self.runtime.template_config,
            feature_config=self.feature_config,
            scenario_config=self.scenario_config,
            schema=self.schema,
            runtime_configuration_hash=self.runtime.runtime_configuration_hash,
        )

    def _default_contexts(
        self,
        simulator: Any,
        event_batch: Any,
    ) -> tuple[tuple[Any, AnchorContext], ...]:
        anchors = build_current_segment_anchors(simulator)
        prepared: list[tuple[Any, AnchorContext]] = []
        for anchor in anchors.leader_follower:
            candidates = self.runtime.catalog.enumerate_for_batch(
                simulator,
                event_batch,
                anchor_id=anchor.anchor_id,
                bound_flight_id=anchor.follower_id,
                resource_id=anchor.resource_id,
                segment_id=anchor.segment_id,
            )
            if not candidates:
                continue
            vector = simulator_state_vector(
                simulator,
                anchor,
                action_candidates=candidates,
                action_applier=self.runtime.action_applier,
                feature_config=self.feature_config,
                scenario_config=self.scenario_config,
                template_config=self.runtime.template_config,
                schema=self.schema,
            )
            prepared.append(
                (
                    anchor,
                    AnchorContext(
                        role_type="leader_follower",
                        schema_hash=vector.schema_hash,
                        anchor_id=anchor.anchor_id,
                        vector=vector,
                        candidates=tuple(candidates),
                    ),
                )
            )
        return tuple(prepared)

    def _prepare_contexts(
        self,
        simulator: Any,
        event_batch: Any,
    ) -> tuple[_PreparedAnchor, ...]:
        raw = (
            self._default_contexts(simulator, event_batch)
            if self._context_builder is None
            else self._context_builder(simulator, event_batch)
        )
        prepared: list[_PreparedAnchor] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise TypeError(
                    "context builder entries must be (source, AnchorContext)"
                )
            source, context = item
            if not isinstance(context, AnchorContext):
                raise TypeError("context builder must return AnchorContext values")
            if context.role_type != "leader_follower":
                raise ValueError(
                    "Phase-0 trainer supports only leader_follower contexts"
                )
            if context.schema_hash != self.schema.schema_hash:
                raise ValueError("anchor context schema does not match trainer schema")
            if context.anchor_id in seen:
                raise ValueError("training contexts must have unique anchor IDs")
            seen.add(context.anchor_id)
            prepared.append(_PreparedAnchor(source=source, context=context))
        return tuple(sorted(prepared, key=lambda item: item.context.anchor_id))

    def _freeze_match_sets(
        self,
        prepared: Sequence[_PreparedAnchor],
        *,
        epoch: int,
    ) -> tuple[tuple[_PreparedAnchor, MatchSet], ...]:
        # Cover from the pre-cover match set, insert under the global bound,
        # and only then rebuild/freeze the recipients used for this epoch.
        for item in prepared:
            initial = build_match_set(item.context, self.population)
            for rule in cover_missing_actions(
                initial,
                config=self.config,
                creation_epoch=epoch,
                rng=self.rng,
            ):
                insert_offspring(
                    self.population,
                    rule,
                    epoch=epoch,
                    max_size=self.config.population_limit,
                    young_protection_epochs=self.config.young_rule_protection_epochs,
                    z=self.config.contender_lcb_z,
                    variance_floor=self.config.variance_floor,
                )
        return tuple(
            (item, build_match_set(item.context, self.population)) for item in prepared
        )

    @staticmethod
    def _record(match_set: MatchSet) -> RulebookDecisionRecord:
        candidates = tuple(
            candidate
            for candidate in match_set.candidates
            if match_set.advocates(RuleAction.from_candidate(candidate))
        )
        return RulebookDecisionRecord(
            anchor_id=match_set.anchor_id,
            features=match_set.vector,
            candidates=candidates,
            role_type=match_set.role_type,
            schema_hash=match_set.schema_hash,
        )

    @staticmethod
    def _advocated_context(match_set: MatchSet) -> AnchorContext | None:
        candidates = tuple(
            candidate
            for candidate in match_set.candidates
            if match_set.advocates(RuleAction.from_candidate(candidate))
        )
        if not candidates:
            return None
        return AnchorContext(
            role_type=match_set.role_type,
            schema_hash=match_set.schema_hash,
            anchor_id=match_set.anchor_id,
            vector=match_set.vector,
            candidates=candidates,
        )

    def _experimentable_context(
        self,
        match_set: MatchSet,
        rulebook: FrozenRulebookPolicy,
    ) -> AnchorContext | None:
        """Return root actions that can produce a credit-bearing experiment.

        In causal mode a selected no-op needs a certified ordinary contender.
        Before the first publication no such contender exists, so allowing
        no-op into the scheduler's coverage pool would leave its completed
        experiment count at zero forever and eventually starve every physical
        action. Vanilla accuracy credit does not have that restriction because
        an all-no-op epoch still supplies a selected-outcome accuracy sample.
        """

        context = self._advocated_context(match_set)
        if context is None or self.credit_mode != CAUSAL_CREDIT_MODE:
            return context
        no_op = RuleAction.no_op()
        if no_op not in context.candidate_actions:
            return context
        if rulebook.rank_rivals(
            self._record(match_set),
            selected_action=no_op,
        ):
            return context
        candidates = tuple(
            candidate
            for candidate in context.candidates
            if not RuleAction.from_candidate(candidate).is_no_op
        )
        if not candidates:
            return None
        return AnchorContext(
            role_type=context.role_type,
            schema_hash=context.schema_hash,
            anchor_id=context.anchor_id,
            vector=context.vector,
            candidates=candidates,
        )

    def _select_root(
        self,
        match_sets: Sequence[tuple[_PreparedAnchor, MatchSet]],
        rulebook: FrozenRulebookPolicy,
    ) -> tuple[_PreparedAnchor, MatchSet, ExplorationDecision]:
        selectable = tuple(
            (prepared, match_set, context)
            for prepared, match_set in match_sets
            if (context := self._experimentable_context(match_set, rulebook))
            is not None
        )
        if not selectable:
            raise RuntimeError("root selection requires an advocated action")
        records = tuple(self._record(match_set) for _, match_set, _ in selectable)
        global_decision = rulebook.evaluate(records)
        by_anchor = {
            match_set.anchor_id: (prepared, match_set, context)
            for prepared, match_set, context in selectable
        }
        if (
            global_decision.anchor_id in by_anchor
            and global_decision.action is not None
            and not global_decision.action.is_no_op
        ):
            prepared, match_set, selection_context = by_anchor[
                str(global_decision.anchor_id)
            ]
            exploit_action = global_decision.action
        else:
            physical_match_sets = tuple(
                item
                for item in selectable
                if any(not action.is_no_op for action in item[2].candidate_actions)
            )
            prepared, match_set, selection_context = min(
                physical_match_sets or selectable,
                key=lambda item: (
                    self.scheduler.visit_count(self.scheduler.cell_for(item[2])),
                    item[1].anchor_id,
                ),
            )
            local = rulebook.evaluate((self._record(match_set),))
            exploit_action = local.action
        decision = self.scheduler.choose(
            selection_context,
            exploit_action=exploit_action,
        )
        return prepared, match_set, decision

    def _subsume_niche(
        self,
        *,
        role_type: str,
        action: RuleAction,
    ) -> tuple[Mapping[str, Any], ...]:
        """Deterministically absorb conservative same-niche specializations."""

        absorbed: list[Mapping[str, Any]] = []
        while True:
            niche = tuple(
                rule
                for rule in self.population.rules
                if rule.role_type == role_type and rule.action == action
            )
            candidates = tuple(
                (general, specific)
                for general in niche
                for specific in niche
                if can_subsume(
                    general,
                    specific,
                    min_experience=self.config.ga_min_experience,
                    material_lcb_tolerance=0.0,
                    z=self.config.contender_lcb_z,
                    variance_floor=self.config.variance_floor,
                )
            )
            if not candidates:
                break
            general, specific = min(
                candidates,
                key=lambda pair: (
                    -pair[0].evolution.lcb(
                        self.config.contender_lcb_z,
                        self.config.variance_floor,
                    ),
                    pair[0].rule_id,
                    pair[1].rule_id,
                ),
            )
            absorbed_numerosity = specific.numerosity
            if not subsume_rule(
                self.population,
                general,
                specific,
                min_experience=self.config.ga_min_experience,
                material_lcb_tolerance=0.0,
                z=self.config.contender_lcb_z,
                variance_floor=self.config.variance_floor,
            ):
                raise RuntimeError("validated subsumption unexpectedly failed")
            absorbed.append(
                {
                    "general_rule_id": general.rule_id,
                    "absorbed_rule_id": specific.rule_id,
                    "absorbed_numerosity": absorbed_numerosity,
                }
            )
        return tuple(absorbed)

    def _run_ga(
        self,
        match_set: MatchSet,
        *,
        selected_action: RuleAction,
        contender_action: RuleAction,
        epoch: int,
    ) -> tuple[Mapping[str, Any], ...]:
        if epoch % self.config.ga_interval:
            return ()
        actions = tuple(
            sorted(
                {selected_action, contender_action},
                key=lambda action: action.key,
            )
        )
        frozen_niches: list[tuple[RuleAction, tuple[Any, ...]]] = []
        for action in actions:
            eligible = tuple(
                self.population.get(rule_id)
                for rule_id in match_set.advocates(action)
                if self.population.get(rule_id).evolution.n
                >= self.config.ga_min_experience
            )
            if eligible:
                frozen_niches.append((action, eligible))

        prepared_births: list[tuple[RuleAction, tuple[str, ...], tuple[Any, ...]]] = []
        for action, eligible in frozen_niches:
            eligible_ids = tuple(rule.rule_id for rule in eligible)
            children = evolve_niche(
                self.population,
                role_type=match_set.role_type,
                action=action,
                schema=self.schema,
                epoch=epoch,
                config=self.config,
                rng=self.rng,
                current_state=match_set.vector.named,
                eligible_rule_ids=eligible_ids,
                insert=False,
            )
            prepared_births.append((action, eligible_ids, children))

        events: list[Mapping[str, Any]] = []
        for action, eligible_ids, children in prepared_births:
            for child in children:
                insert_offspring(
                    self.population,
                    child,
                    epoch=epoch,
                    max_size=self.config.population_limit,
                    young_protection_epochs=self.config.young_rule_protection_epochs,
                    z=self.config.contender_lcb_z,
                    variance_floor=self.config.variance_floor,
                )
            subsumptions = self._subsume_niche(
                role_type=match_set.role_type,
                action=action,
            )
            events.append(
                {
                    "epoch": epoch,
                    "role_type": match_set.role_type,
                    "action": action.to_dict(),
                    "eligible_parent_ids": list(eligible_ids),
                    "offspring_rule_ids": [rule.rule_id for rule in children],
                    "absorbed_rule_ids": [
                        event["absorbed_rule_id"] for event in subsumptions
                    ],
                    "subsumptions": list(subsumptions),
                }
            )
        return tuple(events)

    def _publish_if_due(
        self,
        *,
        epoch: int,
    ) -> tuple[Mapping[str, Any], ...]:
        if epoch % self.config.certification_interval:
            return ()
        previous_hash = self._evaluation_snapshot.rulebook_hash
        thresholds = CertificationThresholds.from_config(self.config)
        rulebook = FrozenRulebookPolicy.from_population(
            self.population.rules,
            feature_schema_hash=self.schema.schema_hash,
            certification_generation=epoch,
            contender_min_samples=self.config.contender_min_samples,
            thresholds=thresholds,
            credit_mode=self.credit_mode,
            action_configuration=self.runtime.vocabulary.payload,
            role_type="leader_follower",
        )
        certification_evidence = {
            frozen.source_rule_id: PublishedCertificationEvidence.from_rule(
                self.population.get(frozen.source_rule_id),
                frozen,
                thresholds=thresholds,
                credit_mode=self.credit_mode,
            )
            for frozen in (*rulebook.action_rules, *rulebook.veto_rules)
        }
        next_snapshot = EvaluationSnapshot(
            rulebook,
            publication_epoch=epoch,
            contender_ranking_fields=self._ranking_fields(),
            certification_evidence=certification_evidence,
            config_hash=self.config_hash,
            action_vocabulary_hash=self.runtime.action_vocabulary_hash,
        )
        # This single assignment is the publication boundary.  The old local
        # snapshot remains valid for every arm that used it this epoch.
        self._evaluation_snapshot = next_snapshot
        return (
            {
                "publication_epoch": epoch,
                "previous_rulebook_hash": previous_hash,
                "rulebook_hash": next_snapshot.rulebook_hash,
                "evaluation_snapshot_hash": next_snapshot.content_hash,
                "certification_evidence_hashes": {
                    source_rule_id: record.content_hash
                    for source_rule_id, record in (
                        next_snapshot.certification_evidence.items()
                    )
                },
                "credit_mode": self.credit_mode,
                "certified_source_rule_ids": list(
                    next_snapshot.certified_source_rule_ids
                ),
            },
        )

    @staticmethod
    def _recipient_payload(recipients: CreditRecipients) -> Mapping[str, Any]:
        return {
            "credit_mode": CAUSAL_CREDIT_MODE,
            "anchor_id": recipients.anchor_id,
            "selected_evolution": list(recipients.selected_rule_ids),
            "selected_deployment": list(recipients.selected_rule_ids),
            "contender_evolution": list(recipients.contender_rule_ids),
            "contender_deployment": list(recipients.contender_rule_ids),
            "veto_evolution": list(recipients.no_op_rule_ids),
        }

    def _assign_credit(
        self,
        match_set: MatchSet,
        *,
        selected_action: RuleAction,
        contender_action: RuleAction,
        selected_outcome: float,
        signals: CreditSignals,
    ) -> tuple[tuple[str, ...], Mapping[str, Any]]:
        """Route one validated rollout under the configured control mode."""

        if self.credit_mode == CAUSAL_CREDIT_MODE:
            if type(self.credit_assigner) is not CausalCreditAssigner:
                raise RuntimeError("trainer causal credit configuration drifted")
            recipients = self.credit_assigner.assign(
                self.population,
                match_set,
                selected_action=selected_action,
                contender_action=contender_action,
                signals=signals,
            )
            return recipients.all_rule_ids, self._recipient_payload(recipients)

        if self.credit_mode != VANILLA_ACCURACY_CREDIT_MODE:
            raise RuntimeError("trainer has an unsupported credit mode")
        if type(self.credit_assigner) is not VanillaAccuracyCreditAssigner:
            raise RuntimeError("trainer vanilla credit configuration drifted")
        updates = self.credit_assigner.assign(
            self.population,
            match_set,
            selected_action=selected_action,
            selected_outcome=selected_outcome,
        )
        rule_ids = tuple(sorted(update.rule_id for update in updates))
        return rule_ids, {
            "credit_mode": VANILLA_ACCURACY_CREDIT_MODE,
            "anchor_id": match_set.anchor_id,
            "selected_accuracy_evolution": list(rule_ids),
            "selected_prediction_deployment": list(rule_ids),
            "contender_prediction": [],
            "veto_evolution": [],
            "updates": [asdict(update) for update in updates],
        }

    @staticmethod
    def _actions_equal(left: Any, right: Any) -> bool:
        """Compare complete requested actions without trusting object equality."""

        try:
            return canonical_data(left) == canonical_data(right)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "three-arm rollout contains non-canonical action evidence"
            ) from exc

    @classmethod
    def _validate_rollout_evidence(
        cls,
        rollout: ThreeArmRolloutResult,
        *,
        selected_action: Any,
        contender_action: Any,
        no_op_action: Any,
        parent_hash: str,
        policy_fingerprint: str,
        horizon_s: float,
    ) -> CreditSignals:
        """Validate hook-produced evidence before it can receive causal credit."""

        if rollout.parent_dynamic_content_hash != parent_hash:
            raise RuntimeError("three-arm rollout did not use the current real parent")
        if rollout.policy_fingerprint != policy_fingerprint:
            raise RuntimeError(
                "three-arm rollout did not use the frozen continuation policy"
            )
        if isinstance(rollout.horizon_s, bool):
            raise RuntimeError("three-arm rollout did not use the configured horizon")
        try:
            actual_horizon = float(rollout.horizon_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "three-arm rollout did not use the configured horizon"
            ) from exc
        if not np.isfinite(actual_horizon) or actual_horizon != horizon_s:
            raise RuntimeError("three-arm rollout did not use the configured horizon")

        try:
            expected_no_op = RuleAction.from_candidate(no_op_action)
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeError(
                "three-arm rollout Arm C request is not canonical no-op"
            ) from exc
        if not expected_no_op.is_no_op:
            raise RuntimeError("three-arm rollout Arm C request is not canonical no-op")

        arms = (
            ("selected", rollout.selected, selected_action),
            ("contender", rollout.contender, contender_action),
            ("no_op", rollout.no_op, no_op_action),
        )
        for expected_label, arm, _ in arms:
            if not isinstance(arm, PairedArmTrace):
                raise TypeError(
                    f"three-arm rollout {expected_label} arm must be PairedArmTrace"
                )

        try:
            returned_no_op = RuleAction.from_candidate(rollout.no_op.initial_action)
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeError(
                "three-arm rollout Arm C is not canonical no-op"
            ) from exc
        if not returned_no_op.is_no_op:
            raise RuntimeError("three-arm rollout Arm C is not canonical no-op")

        for expected_label, arm, expected_action in arms:
            if arm.label != expected_label:
                raise RuntimeError(
                    f"three-arm rollout arm label must be {expected_label!r}"
                )
            if arm.initial_dynamic_content_hash != parent_hash:
                raise RuntimeError(
                    f"three-arm rollout {expected_label} arm did not fork the common parent"
                )
            if not cls._actions_equal(arm.initial_action, expected_action):
                raise RuntimeError(
                    f"three-arm rollout {expected_label} arm used an unexpected initial action"
                )

        try:
            signals = CreditSignals.from_arm_scores(
                selected=rollout.selected.score,
                contender=rollout.contender.score,
                no_op=rollout.no_op.score,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("three-arm rollout contains invalid arm scores") from exc
        for name in (
            "delta_rival",
            "delta_selected_noop",
            "delta_contender_noop",
            "delta_veto",
        ):
            if getattr(rollout, name) != getattr(signals, name):
                raise RuntimeError(f"three-arm rollout stored inconsistent {name}")
        return signals

    def _skip_result(
        self,
        *,
        simulator: Any,
        epoch: int,
        reason: str,
        rulebook_hash: str,
        matched_rule_ids: Sequence[str] = (),
        selected_anchor_id: str | None = None,
        selected_action: Any | None = None,
        contender_action: Any | None = None,
        no_op_action: Any | None = None,
        exploration_reason: str | None = None,
        contender_selection: Mapping[str, Any] | None = None,
    ) -> TrainingEpochResult:
        root_hash = dynamic_content_fingerprint(simulator)
        certification_events = self._publish_if_due(epoch=epoch)
        self.epoch = epoch
        trace = EpochTrace(
            scenario_definition_hash=simulator.definition.definition_hash,
            epoch=epoch,
            root_parent_hash=root_hash,
            committed_state_hash=root_hash,
            selected_anchor_id=selected_anchor_id,
            selected_action=selected_action,
            contender_action=contender_action,
            no_op_action=no_op_action,
            matched_rule_ids=tuple(matched_rule_ids),
            coadvocate_rule_ids=(),
            exploration_reason=(
                reason
                if exploration_reason is None
                else f"{exploration_reason}:{reason}"
            ),
            contender_selection=contender_selection,
            rulebook_hash=rulebook_hash,
            certification_events=certification_events,
            config_hash=self.config_hash,
            action_vocabulary_hash=self.runtime.action_vocabulary_hash,
            feature_schema_hash=self.schema.schema_hash,
        )
        return TrainingEpochResult(
            epoch=epoch,
            trace=trace,
            rollout=None,
            evaluation_snapshot=self.evaluation_snapshot,
            skipped_reason=reason,
        )

    def _consume_pending_decision(
        self,
        *,
        event_batch: Any,
        decision_reference: Any | None,
    ) -> None:
        pending = self._pending_decision_reference
        if pending is None:
            if decision_reference is not None:
                raise ValueError(
                    "decision_reference was supplied without a pending checkpoint decision"
                )
            return
        kind = str(pending["kind"])
        if kind == "event_batch":
            if decision_reference is not None:
                raise ValueError(
                    "pending event batch must be replayed with the event batch itself"
                )
            candidate = self._build_pending_decision_reference(
                next_epoch=self.epoch + 1,
                event_batch=event_batch,
                decision_reference=None,
            )
        else:
            if decision_reference is None:
                raise ValueError(
                    "resumed trainer requires the pending decision_reference"
                )
            candidate = self._build_pending_decision_reference(
                next_epoch=self.epoch + 1,
                event_batch=None,
                decision_reference=decision_reference,
            )
        if canonical_data(candidate) != canonical_data(pending):
            raise ValueError(
                "resumed event batch/decision reference does not match checkpoint"
            )
        self._pending_decision_reference = None

    def process_epoch(
        self,
        simulator: Any,
        event_batch: Any,
        *,
        decision_reference: Any | None = None,
    ) -> TrainingEpochResult:
        """Perform one complete root experiment and commit only Arm A."""

        self._consume_pending_decision(
            event_batch=event_batch,
            decision_reference=decision_reference,
        )
        epoch = self.epoch + 1
        scenario_definition_hash = simulator.definition.definition_hash
        frozen_snapshot = self._evaluation_snapshot
        rulebook = frozen_snapshot.rulebook
        rulebook_hash = frozen_snapshot.rulebook_hash
        prepared = self._prepare_contexts(simulator, event_batch)
        match_sets = self._freeze_match_sets(prepared, epoch=epoch)
        all_matched_ids = tuple(
            sorted(
                {
                    rule_id
                    for _, match_set in match_sets
                    for rule_id in match_set.matched_rule_ids
                }
            )
        )
        if not match_sets or not any(
            not action.is_no_op
            for _, match_set in match_sets
            for action in match_set.candidate_actions
        ):
            return self._skip_result(
                simulator=simulator,
                epoch=epoch,
                reason="no_non_noop_action",
                rulebook_hash=rulebook_hash,
                matched_rule_ids=all_matched_ids,
            )

        prepared_root, root_match, exploration_decision = self._select_root(
            match_sets, rulebook
        )
        selected_candidate = exploration_decision.candidate
        exploration_reason = exploration_decision.reason
        selected_action = RuleAction.from_candidate(selected_candidate)
        record = self._record(root_match)
        ranked_rivals = rulebook.rank_rivals(
            record,
            selected_action=selected_action,
        )
        no_op_candidate = root_match.context.candidate_for(RuleAction.no_op())
        if ranked_rivals:
            contender_candidate = ranked_rivals[0].candidate
            selected_rank: int | None = 1
            fallback: str | None = None
        else:
            contender_candidate = no_op_candidate
            selected_rank = None
            fallback = "mandatory_no_op_no_certified_rival"
        contender_selection = {
            "ranking_fields": frozen_snapshot.contender_ranking_fields,
            "ranked_candidates": [
                {
                    "rank": rank,
                    "action": rival.action.to_dict(),
                    "score": rival.score,
                    "source_rule_ids": list(rival.source_rule_ids),
                }
                for rank, rival in enumerate(ranked_rivals, start=1)
            ],
            "selected_rank": selected_rank,
            "fallback": fallback,
        }
        contender_action = RuleAction.from_candidate(contender_candidate)
        if (
            self.credit_mode == CAUSAL_CREDIT_MODE
            and selected_action.is_no_op
            and contender_action.is_no_op
        ):
            return self._skip_result(
                simulator=simulator,
                epoch=epoch,
                reason="all_no_op_experiment",
                rulebook_hash=rulebook_hash,
                matched_rule_ids=root_match.matched_rule_ids,
                selected_anchor_id=root_match.anchor_id,
                selected_action=selected_candidate,
                contender_action=contender_candidate,
                no_op_action=no_op_candidate,
                exploration_reason=exploration_reason,
                contender_selection=contender_selection,
            )

        outcome_plan = self._outcome_plan_factory(
            simulator,
            prepared_root.source,
            config=self.outcome_config,
        )
        outcome_plan_hash = content_hash(
            outcome_plan,
            namespace="hailmary.simulator_outcome_plan.v1",
        )
        continuation_policy = self._simulator_policy(rulebook)
        frozen_policy_fingerprint = continuation_policy.policy_fingerprint()
        try:
            configured_horizon = float(getattr(outcome_plan, "horizon_s"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("outcome plan horizon_s must be numeric") from exc
        if not np.isfinite(configured_horizon):
            raise ValueError("outcome plan horizon_s must be finite")
        parent_hash = dynamic_content_fingerprint(simulator)
        rollout = self._rollout_hook(
            simulator,
            selected_action=selected_candidate,
            contender_action=contender_candidate,
            no_op_action=no_op_candidate,
            frozen_policy=continuation_policy,
            outcome_plan=outcome_plan,
            outcome_config=self.outcome_config,
        )
        if not isinstance(rollout, ThreeArmRolloutResult):
            raise TypeError("rollout hook must return ThreeArmRolloutResult")
        if dynamic_content_fingerprint(simulator) != parent_hash:
            raise RuntimeError("temporary rollout mutated the real simulator")
        if continuation_policy.policy_fingerprint() != frozen_policy_fingerprint:
            raise RuntimeError(
                "temporary rollout mutated the frozen continuation policy"
            )

        signals = self._validate_rollout_evidence(
            rollout,
            selected_action=selected_candidate,
            contender_action=contender_candidate,
            no_op_action=no_op_candidate,
            parent_hash=parent_hash,
            policy_fingerprint=frozen_policy_fingerprint,
            horizon_s=configured_horizon,
        )
        credit_rule_ids, ledger_recipients = self._assign_credit(
            root_match,
            selected_action=selected_action,
            contender_action=contender_action,
            selected_outcome=rollout.selected.score,
            signals=signals,
        )

        apply_root = getattr(simulator, "apply", None)
        if not callable(apply_root):
            apply_root = getattr(simulator, "apply_action", None)
        if not callable(apply_root):
            raise TypeError("real simulator must implement apply(action)")
        apply_root(selected_candidate)
        self.scheduler.record_experiment(exploration_decision)
        committed_hash = dynamic_content_fingerprint(simulator)

        ga_events = self._run_ga(
            root_match,
            selected_action=selected_action,
            contender_action=contender_action,
            epoch=epoch,
        )
        certification_events = self._publish_if_due(epoch=epoch)
        self.epoch = epoch

        arms = {
            "selected": rollout.selected,
            "contender": rollout.contender,
            "no_op": rollout.no_op,
        }
        trace = EpochTrace(
            scenario_definition_hash=scenario_definition_hash,
            epoch=epoch,
            root_parent_hash=parent_hash,
            committed_state_hash=committed_hash,
            outcome_plan_hash=outcome_plan_hash,
            selected_anchor_id=root_match.anchor_id,
            selected_action=selected_candidate,
            contender_action=contender_candidate,
            no_op_action=no_op_candidate,
            committed_action=selected_candidate,
            matched_rule_ids=root_match.matched_rule_ids,
            coadvocate_rule_ids=credit_rule_ids,
            exploration_reason=exploration_reason,
            contender_selection=contender_selection,
            rulebook_hash=rulebook_hash,
            arm_initial_hashes={
                name: arm.initial_dynamic_content_hash for name, arm in arms.items()
            },
            arm_final_hashes={
                name: arm.final_dynamic_content_hash for name, arm in arms.items()
            },
            arm_scores={name: arm.score for name, arm in arms.items()},
            delta_rival=signals.delta_rival,
            delta_selected_noop=signals.delta_selected_noop,
            delta_contender_noop=signals.delta_contender_noop,
            delta_veto=signals.delta_veto,
            ledger_recipients=ledger_recipients,
            ga_events=ga_events,
            certification_events=certification_events,
            config_hash=self.config_hash,
            action_vocabulary_hash=self.runtime.action_vocabulary_hash,
            feature_schema_hash=self.schema.schema_hash,
        )
        return TrainingEpochResult(
            epoch=epoch,
            trace=trace,
            rollout=rollout,
            evaluation_snapshot=self.evaluation_snapshot,
        )

    def checkpoint(
        self,
        simulator: Any,
        *,
        event_batch: Any | None = None,
        decision_reference: Any | None = None,
    ) -> TrainingCheckpoint:
        """Detach state, optionally binding the next decision for exact replay."""

        self._validate_evaluation_snapshot(self._evaluation_snapshot)
        if event_batch is not None:
            self._validate_event_batch_against_simulator(event_batch, simulator)
        supplied_pending = self._build_pending_decision_reference(
            next_epoch=self.epoch + 1,
            event_batch=event_batch,
            decision_reference=decision_reference,
        )
        if self._pending_decision_reference is not None:
            if supplied_pending is not None and canonical_data(
                supplied_pending
            ) != canonical_data(self._pending_decision_reference):
                raise ValueError(
                    "checkpoint decision reference does not match pending replay"
                )
            pending_decision = self._pending_decision_reference
        else:
            pending_decision = supplied_pending
        scheduler_state = self.scheduler.to_dict()
        trainer_state = {
            "schema_version": TRAINER_STATE_SCHEMA_VERSION,
            "trainer_rng": capture_rng_state(self.rng),
            "exploration_scheduler": scheduler_state,
            "hook_fingerprints": self._hook_fingerprints,
            "pending_decision": pending_decision,
            "learning_config": asdict(self.config),
            "outcome_config": asdict(self.outcome_config),
            "feature_config": asdict(self.feature_config),
            "scenario_config": asdict(self.scenario_config),
            "credit": self._credit_configuration,
        }
        visits = sum(int(item["count"]) for item in scheduler_state["visits"])
        experiments = sum(int(item["count"]) for item in scheduler_state["experiments"])
        return TrainingCheckpoint(
            scenario_definition_hash=simulator.definition.definition_hash,
            simulator_snapshot=simulator.snapshot(),
            population=self.population,
            evaluation_snapshot=self._evaluation_snapshot,
            epoch=self.epoch,
            rng_state=trainer_state,
            exploration_counts={
                "region_visits": visits,
                "action_experiments": experiments,
            },
            config_hash=self.config_hash,
            action_vocabulary_hash=self.runtime.action_vocabulary_hash,
            feature_schema_hash=self.schema.schema_hash,
        )

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint: TrainingCheckpoint,
        *,
        definition: Any,
        runtime: ActionRuntime,
        schema: FeatureSchema | None = None,
        credit_assigner: CreditAssigner | None = None,
        context_builder: ContextBuilder | None = None,
        outcome_plan_factory: OutcomePlanFactory = simulator_outcome_plan,
        rollout_hook: RolloutHook = three_arm_simulator_rollout,
        hook_fingerprints: Mapping[str, str] | None = None,
    ) -> tuple["CausalTrainer", Any]:
        """Restore both owners; simulator resumption always goes through runtime."""

        if not isinstance(checkpoint, TrainingCheckpoint):
            raise TypeError("checkpoint must be TrainingCheckpoint")
        state = checkpoint.rng_state
        if state.get("schema_version") != TRAINER_STATE_SCHEMA_VERSION:
            raise ValueError(
                "checkpoint does not contain a compatible SEQD trainer state"
            )
        learning = LearningConfig(**dict(state["learning_config"]))
        outcome = OutcomeConfig(**dict(state["outcome_config"]))
        features = FeatureConfig(**dict(state["feature_config"]))
        scenario = ScenarioConfig(**dict(state["scenario_config"]))
        scheduler = RegionExplorationScheduler.from_dict(state["exploration_scheduler"])
        stored_credit = state.get(
            "credit",
            {"mode": CAUSAL_CREDIT_MODE, "settings": {}},
        )
        restored_credit_assigner = credit_assigner_from_configuration(stored_credit)
        if credit_assigner is None:
            resolved_credit_assigner = restored_credit_assigner
        else:
            if credit_assigner_configuration(credit_assigner) != canonical_data(
                stored_credit
            ):
                raise ValueError(
                    "checkpoint credit mode/settings do not match resume input"
                )
            resolved_credit_assigner = credit_assigner

        stored_hooks = state.get("hook_fingerprints")
        if not isinstance(stored_hooks, Mapping):
            raise ValueError("checkpoint does not contain hook fingerprints")
        resolved_hooks = cls._resolve_hook_fingerprints(
            context_builder=context_builder,
            outcome_plan_factory=outcome_plan_factory,
            rollout_hook=rollout_hook,
            supplied=hook_fingerprints,
        )
        if canonical_data(stored_hooks) != canonical_data(resolved_hooks):
            raise ValueError("checkpoint hook fingerprints do not match resume input")
        raw_pending = state.get("pending_decision")
        if raw_pending is not None and not isinstance(raw_pending, Mapping):
            raise TypeError("checkpoint pending_decision must be a mapping or null")
        pending_decision = cls._validate_pending_decision_reference(
            raw_pending,
            expected_next_epoch=checkpoint.epoch + 1,
        )

        rng = np.random.default_rng()
        rng.bit_generator.state = dict(state["trainer_rng"])
        trainer = cls(
            runtime,
            population=checkpoint.population,
            config=learning,
            outcome_config=outcome,
            feature_config=features,
            scenario_config=scenario,
            schema=schema,
            scheduler=scheduler,
            evaluation_snapshot=checkpoint.evaluation_snapshot,
            epoch=checkpoint.epoch,
            rng=rng,
            credit_assigner=resolved_credit_assigner,
            context_builder=context_builder,
            outcome_plan_factory=outcome_plan_factory,
            rollout_hook=rollout_hook,
            hook_fingerprints=resolved_hooks,
            pending_decision_reference=pending_decision,
        )
        if definition.definition_hash != checkpoint.scenario_definition_hash:
            raise ValueError(
                "checkpoint scenario definition does not match resume input"
            )
        simulator = runtime.resume_simulator(
            definition,
            checkpoint.simulator_snapshot,
        )
        return trainer, simulator


SEQDTrainer = CausalTrainer


__all__ = [
    "CausalTrainer",
    "SEQDTrainer",
    "TRAINER_STATE_SCHEMA_VERSION",
    "TrainingEpochResult",
]
