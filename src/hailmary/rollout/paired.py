"""Coupled temporary branches for paired and three-arm rollouts."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from hailmary.errors import SimulationError
from hailmary.ids import canonical_json, content_hash
from hailmary.rollout.policy import FrozenPolicy, NoOpPolicy, policy_fingerprint


@runtime_checkable
class ForkableRollout(Protocol):
    def fork(self, *, label: str) -> Any: ...


ActionApplier = Callable[[Any, Any], Any | None]
RolloutRunner = Callable[[Any, float, FrozenPolicy], Any | None]


def _finite_delta(value: Any, *, name: str) -> float:
    normalized = float(value)
    if not np.isfinite(normalized):
        raise SimulationError(f"{name} delta is non-finite")
    return normalized


@dataclass(frozen=True)
class PairedArmTrace:
    label: str
    initial_action: Any
    applied_action: Any
    action_audit: Any
    initial_dynamic_content_hash: str
    final_dynamic_content_hash: str
    score: float
    outcome: Any


@dataclass(frozen=True)
class PairedRolloutResult:
    selected: PairedArmTrace
    contender: PairedArmTrace
    delta: float
    parent_dynamic_content_hash: str
    policy_fingerprint: str
    horizon_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta", _finite_delta(self.delta, name="paired"))


@dataclass(frozen=True)
class PolicyVsPermanentNoOpResult:
    """Held-out learned policy versus an always-no-op policy from one root."""

    policy_arm: PairedArmTrace
    permanent_no_op: PairedArmTrace
    delta: float
    parent_dynamic_content_hash: str
    learned_policy_fingerprint: str
    permanent_no_op_policy_fingerprint: str
    horizon_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta",
            _finite_delta(self.delta, name="policy-vs-permanent-no-op"),
        )


@dataclass(frozen=True)
class ThreeArmRolloutResult:
    """Selected (A), rival (B), and mandatory no-op (C) outcomes."""

    selected: PairedArmTrace
    contender: PairedArmTrace
    no_op: PairedArmTrace
    delta_rival: float
    delta_selected_noop: float
    delta_contender_noop: float
    delta_veto: float
    parent_dynamic_content_hash: str
    policy_fingerprint: str
    horizon_s: float

    def __post_init__(self) -> None:
        for name in (
            "delta_rival",
            "delta_selected_noop",
            "delta_contender_noop",
            "delta_veto",
        ):
            object.__setattr__(
                self,
                name,
                _finite_delta(getattr(self, name), name=name),
            )


def rebind_action_to_branch(action: Any, branch: Any) -> Any:
    """Rebind an epoch candidate to a fork's distinct provenance state ID.

    Forks initially have identical dynamic content but deliberately distinct
    ``state_id`` values. Action freshness must therefore be rebound without
    changing its anchor, lever, band, station, or parent decision epoch.
    """

    state = getattr(branch, "state", None)
    if state is None:
        return action
    if dataclasses.is_dataclass(action) and not isinstance(action, type):
        field_names = {field.name for field in dataclasses.fields(action)}
        changes: dict[str, Any] = {}
        if "state_id" in field_names:
            changes["state_id"] = str(state.state_id)
        if "state_version" in field_names:
            changes["state_version"] = int(state.version)
        if "epoch_index" in field_names:
            changes["epoch_index"] = int(state.decision_epoch_index)
        if "action_id" in field_names:
            changes["action_id"] = ""
        if changes:
            return dataclasses.replace(action, **changes)
    if isinstance(action, dict) and any(
        key in action for key in ("state_id", "state_version", "epoch_index")
    ):
        rebound = dict(action)
        rebound.update(
            state_id=str(state.state_id),
            state_version=int(state.version),
            epoch_index=int(state.decision_epoch_index),
        )
        if "action_id" in rebound:
            rebound["action_id"] = ""
        return rebound
    return action


def dynamic_content_fingerprint(subject: Any) -> str:
    """Read a branch/state hash without depending on concrete engine classes."""

    for attribute_name in ("dynamic_content_hash", "content_hash"):
        attribute = getattr(subject, attribute_name, None)
        if attribute is not None:
            value = attribute() if callable(attribute) else attribute
            if isinstance(value, str) and value:
                return value
    state = getattr(subject, "state", None)
    if state is not None and state is not subject:
        return dynamic_content_fingerprint(state)
    if dataclasses.is_dataclass(subject) and not isinstance(subject, type):
        return content_hash(subject, namespace="hailmary.rollout_state")
    if hasattr(subject, "__dict__"):
        try:
            return content_hash(vars(subject), namespace="hailmary.rollout_state")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "rollout objects must expose dynamic_content_hash/content_hash or canonical state"
            ) from exc
    raise TypeError(
        "rollout objects must expose dynamic_content_hash/content_hash or canonical state"
    )


def _apply_initial_action(
    branch: Any,
    action: Any,
    *,
    action_applier: ActionApplier | None,
) -> Any:
    if action_applier is not None:
        result = action_applier(branch, action)
        return result if callable(getattr(result, "run_until", None)) else branch
    method = getattr(branch, "apply", None)
    if not callable(method):
        method = getattr(branch, "apply_action", None)
    if not callable(method):
        raise TypeError(
            "forked rollout branch must implement apply(action) or apply_action(action)"
        )
    result = method(action)
    return result if callable(getattr(result, "run_until", None)) else branch


def _run_until(
    branch: Any,
    horizon_s: float,
    policy: FrozenPolicy,
    *,
    rollout_runner: RolloutRunner | None,
) -> Any:
    if rollout_runner is not None:
        result = rollout_runner(branch, float(horizon_s), policy)
        if result is None or isinstance(result, (tuple, list)):
            return branch
        return result
    method = getattr(branch, "run_until", None)
    if not callable(method):
        raise TypeError(
            "forked rollout branch must implement run_until(horizon, policy=...)"
        )
    result = method(float(horizon_s), policy=policy)
    if result is None or isinstance(result, (tuple, list)):
        return branch
    return result


def _score_branch(branch: Any, scorer: Callable[[Any], Any] | Any) -> tuple[float, Any]:
    if callable(scorer):
        outcome = scorer(branch)
    else:
        evaluate = getattr(scorer, "evaluate", None)
        if not callable(evaluate):
            raise TypeError("scorer must be callable or implement evaluate(branch)")
        outcome = evaluate(branch)
    raw_score: Any = getattr(outcome, "score", outcome)
    score = float(raw_score)
    if not np.isfinite(score):
        raise SimulationError("paired rollout scorer returned a non-finite value")
    return score, outcome


def _actions_equivalent(left: Any, right: Any) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        try:
            return bool(left == right)
        except Exception:
            return left is right


def _run_arm(
    parent: ForkableRollout,
    *,
    label: str,
    action: Any,
    horizon_s: float,
    policy: FrozenPolicy,
    scorer: Callable[[Any], Any] | Any,
    action_applier: ActionApplier | None,
    rollout_runner: RolloutRunner | None,
) -> PairedArmTrace:
    branch = parent.fork(label=label)
    initial_hash = dynamic_content_fingerprint(branch)
    applied_action = rebind_action_to_branch(action, branch)
    branch = _apply_initial_action(
        branch, applied_action, action_applier=action_applier
    )
    action_audit = getattr(branch, "last_action_result", None)
    branch = _run_until(branch, horizon_s, policy, rollout_runner=rollout_runner)
    score, outcome = _score_branch(branch, scorer)
    return PairedArmTrace(
        label=label,
        initial_action=action,
        applied_action=applied_action,
        action_audit=action_audit,
        initial_dynamic_content_hash=initial_hash,
        final_dynamic_content_hash=dynamic_content_fingerprint(branch),
        score=score,
        outcome=outcome,
    )


def _assert_arm_invariants(
    parent: ForkableRollout,
    arm: PairedArmTrace,
    *,
    parent_dynamic_content_hash: str,
    frozen_policy: FrozenPolicy,
    frozen_policy_fingerprint: str,
) -> None:
    if arm.initial_dynamic_content_hash != parent_dynamic_content_hash:
        raise SimulationError(
            f"{arm.label} temporary rollout did not fork the common parent content"
        )
    if dynamic_content_fingerprint(parent) != parent_dynamic_content_hash:
        raise SimulationError(f"{arm.label} temporary rollout mutated its parent")
    if policy_fingerprint(frozen_policy) != frozen_policy_fingerprint:
        raise SimulationError(
            f"frozen policy mutated during {arm.label} temporary rollout"
        )


def _arm_delta(
    left: PairedArmTrace,
    right: PairedArmTrace,
    *,
    left_action: Any,
    right_action: Any,
    equivalent_label: str,
) -> float:
    if not _actions_equivalent(left_action, right_action):
        return _finite_delta(
            left.score - right.score,
            name=equivalent_label,
        )
    if left.final_dynamic_content_hash != right.final_dynamic_content_hash:
        raise SimulationError(
            f"identical {equivalent_label} actions produced different dynamic-content hashes"
        )
    if left.score != right.score:
        raise SimulationError(
            f"identical {equivalent_label} actions produced different scores"
        )
    return 0.0


def paired_rollout(
    parent: ForkableRollout,
    *,
    selected_action: Any,
    contender_action: Any,
    frozen_policy: FrozenPolicy,
    horizon_s: float,
    scorer: Callable[[Any], Any] | Any,
    action_applier: ActionApplier | None = None,
    rollout_runner: RolloutRunner | None = None,
) -> PairedRolloutResult:
    """Evaluate two first actions while proving parent/policy immutability.

    Engines following the design's ``apply`` and policy-aware ``run_until``
    contract need no adapters. While those methods are supplied as free
    functions, callers can pass ``action_applier`` and ``rollout_runner``;
    their return values may be a replacement branch or ordinary audit/batch
    data when they mutate the forked driver in place.
    """

    horizon = float(horizon_s)
    if not np.isfinite(horizon):
        raise ValueError("horizon_s must be finite")
    parent_before = dynamic_content_fingerprint(parent)
    policy_before = policy_fingerprint(frozen_policy)

    selected = _run_arm(
        parent,
        label="selected",
        action=selected_action,
        horizon_s=horizon,
        policy=frozen_policy,
        scorer=scorer,
        action_applier=action_applier,
        rollout_runner=rollout_runner,
    )
    _assert_arm_invariants(
        parent,
        selected,
        parent_dynamic_content_hash=parent_before,
        frozen_policy=frozen_policy,
        frozen_policy_fingerprint=policy_before,
    )

    contender = _run_arm(
        parent,
        label="contender",
        action=contender_action,
        horizon_s=horizon,
        policy=frozen_policy,
        scorer=scorer,
        action_applier=action_applier,
        rollout_runner=rollout_runner,
    )
    _assert_arm_invariants(
        parent,
        contender,
        parent_dynamic_content_hash=parent_before,
        frozen_policy=frozen_policy,
        frozen_policy_fingerprint=policy_before,
    )

    delta = _arm_delta(
        selected,
        contender,
        left_action=selected_action,
        right_action=contender_action,
        equivalent_label="paired",
    )

    return PairedRolloutResult(
        selected=selected,
        contender=contender,
        delta=delta,
        parent_dynamic_content_hash=parent_before,
        policy_fingerprint=policy_before,
        horizon_s=horizon,
    )


def policy_vs_permanent_no_op_rollout(
    parent: ForkableRollout,
    *,
    policy_action: Any,
    no_op_action: Any,
    frozen_policy: FrozenPolicy,
    horizon_s: float,
    scorer: Callable[[Any], Any] | Any,
    action_applier: ActionApplier | None = None,
    rollout_runner: RolloutRunner | None = None,
) -> PolicyVsPermanentNoOpResult:
    """Compare a learned policy with a permanent no-op held-out control.

    The learned arm applies ``policy_action`` and then uses ``frozen_policy``.
    The control arm applies ``no_op_action`` and uses a freshly constructed,
    immutable ``NoOpPolicy`` for every later decision through the horizon.
    This differs intentionally from causal paired/three-arm rollouts, whose
    arms must share one continuation policy.
    """

    horizon = float(horizon_s)
    if not np.isfinite(horizon):
        raise ValueError("horizon_s must be finite")
    parent_before = dynamic_content_fingerprint(parent)
    learned_before = policy_fingerprint(frozen_policy)
    permanent_policy = NoOpPolicy()
    permanent_before = policy_fingerprint(permanent_policy)

    policy_arm = _run_arm(
        parent,
        label="policy",
        action=policy_action,
        horizon_s=horizon,
        policy=frozen_policy,
        scorer=scorer,
        action_applier=action_applier,
        rollout_runner=rollout_runner,
    )
    _assert_arm_invariants(
        parent,
        policy_arm,
        parent_dynamic_content_hash=parent_before,
        frozen_policy=frozen_policy,
        frozen_policy_fingerprint=learned_before,
    )

    permanent_no_op = _run_arm(
        parent,
        label="permanent_no_op",
        action=no_op_action,
        horizon_s=horizon,
        policy=permanent_policy,
        scorer=scorer,
        action_applier=action_applier,
        rollout_runner=rollout_runner,
    )
    _assert_arm_invariants(
        parent,
        permanent_no_op,
        parent_dynamic_content_hash=parent_before,
        frozen_policy=permanent_policy,
        frozen_policy_fingerprint=permanent_before,
    )
    if policy_fingerprint(frozen_policy) != learned_before:
        raise SimulationError(
            "learned frozen policy mutated during permanent no-op rollout"
        )

    return PolicyVsPermanentNoOpResult(
        policy_arm=policy_arm,
        permanent_no_op=permanent_no_op,
        delta=_finite_delta(
            policy_arm.score - permanent_no_op.score,
            name="policy-vs-permanent-no-op",
        ),
        parent_dynamic_content_hash=parent_before,
        learned_policy_fingerprint=learned_before,
        permanent_no_op_policy_fingerprint=permanent_before,
        horizon_s=horizon,
    )


def three_arm_rollout(
    parent: ForkableRollout,
    *,
    selected_action: Any,
    contender_action: Any,
    no_op_action: Any,
    frozen_policy: FrozenPolicy,
    horizon_s: float,
    scorer: Callable[[Any], Any] | Any,
    action_applier: ActionApplier | None = None,
    rollout_runner: RolloutRunner | None = None,
) -> ThreeArmRolloutResult:
    """Evaluate A/B/C from one parent under one immutable continuation policy."""

    horizon = float(horizon_s)
    if not np.isfinite(horizon):
        raise ValueError("horizon_s must be finite")
    parent_before = dynamic_content_fingerprint(parent)
    policy_before = policy_fingerprint(frozen_policy)

    arms: list[PairedArmTrace] = []
    for label, action in (
        ("selected", selected_action),
        ("contender", contender_action),
        ("no_op", no_op_action),
    ):
        arm = _run_arm(
            parent,
            label=label,
            action=action,
            horizon_s=horizon,
            policy=frozen_policy,
            scorer=scorer,
            action_applier=action_applier,
            rollout_runner=rollout_runner,
        )
        _assert_arm_invariants(
            parent,
            arm,
            parent_dynamic_content_hash=parent_before,
            frozen_policy=frozen_policy,
            frozen_policy_fingerprint=policy_before,
        )
        arms.append(arm)

    selected, contender, no_op = arms
    delta_rival = _arm_delta(
        selected,
        contender,
        left_action=selected_action,
        right_action=contender_action,
        equivalent_label="three-arm selected/contender",
    )
    delta_selected_noop = _arm_delta(
        selected,
        no_op,
        left_action=selected_action,
        right_action=no_op_action,
        equivalent_label="three-arm selected/no-op",
    )
    delta_contender_noop = _arm_delta(
        contender,
        no_op,
        left_action=contender_action,
        right_action=no_op_action,
        equivalent_label="three-arm contender/no-op",
    )
    delta_veto = _finite_delta(
        no_op.score - max(selected.score, contender.score),
        name="three-arm veto",
    )

    return ThreeArmRolloutResult(
        selected=selected,
        contender=contender,
        no_op=no_op,
        delta_rival=delta_rival,
        delta_selected_noop=delta_selected_noop,
        delta_contender_noop=delta_contender_noop,
        delta_veto=delta_veto,
        parent_dynamic_content_hash=parent_before,
        policy_fingerprint=policy_before,
        horizon_s=horizon,
    )


def _assert_canonical_simulator_no_op(no_op_action: Any) -> None:
    from hailmary.actions.models import ActionLever
    from hailmary.actions.vocabulary import NO_OP_BAND

    if isinstance(no_op_action, dict):
        lever = no_op_action.get("lever")
        band = no_op_action.get("band")
    else:
        lever = getattr(no_op_action, "lever", None)
        band = getattr(no_op_action, "band", None)
    try:
        normalized_lever = ActionLever(lever)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "simulator permanent no-op must be the canonical no-op action"
        ) from exc
    if normalized_lever is not ActionLever.NO_OP or band != NO_OP_BAND:
        raise ValueError("simulator permanent no-op must be the canonical no-op action")


def paired_simulator_rollout(
    parent: ForkableRollout,
    *,
    selected_action: Any,
    contender_action: Any,
    frozen_policy: FrozenPolicy,
    outcome_plan: Any,
    outcome_config: Any | None = None,
) -> PairedRolloutResult:
    """Run paired arms and score a real simulator's frozen outcome cohort."""

    from hailmary.evaluation.outcome import score_simulator_outcome

    parent_hash = dynamic_content_fingerprint(parent)
    if getattr(outcome_plan, "root_dynamic_content_hash", None) != parent_hash:
        raise SimulationError(
            "simulator outcome plan does not belong to the rollout parent root"
        )
    horizon_s = float(getattr(outcome_plan, "horizon_s"))
    return paired_rollout(
        parent,
        selected_action=selected_action,
        contender_action=contender_action,
        frozen_policy=frozen_policy,
        horizon_s=horizon_s,
        scorer=lambda branch: score_simulator_outcome(
            branch,
            outcome_plan=outcome_plan,
            config=outcome_config,
        ),
    )


def policy_vs_permanent_no_op_simulator_rollout(
    parent: ForkableRollout,
    *,
    policy_action: Any,
    no_op_action: Any,
    frozen_policy: FrozenPolicy,
    outcome_plan: Any,
    outcome_config: Any | None = None,
) -> PolicyVsPermanentNoOpResult:
    """Run a learned simulator policy against a permanent no-op control."""

    from hailmary.evaluation.outcome import score_simulator_outcome

    parent_hash = dynamic_content_fingerprint(parent)
    if getattr(outcome_plan, "root_dynamic_content_hash", None) != parent_hash:
        raise SimulationError(
            "simulator outcome plan does not belong to the rollout parent root"
        )
    _assert_canonical_simulator_no_op(no_op_action)
    horizon_s = float(getattr(outcome_plan, "horizon_s"))
    return policy_vs_permanent_no_op_rollout(
        parent,
        policy_action=policy_action,
        no_op_action=no_op_action,
        frozen_policy=frozen_policy,
        horizon_s=horizon_s,
        scorer=lambda branch: score_simulator_outcome(
            branch,
            outcome_plan=outcome_plan,
            config=outcome_config,
        ),
    )


def three_arm_simulator_rollout(
    parent: ForkableRollout,
    *,
    selected_action: Any,
    contender_action: Any,
    no_op_action: Any,
    frozen_policy: FrozenPolicy,
    outcome_plan: Any,
    outcome_config: Any | None = None,
) -> ThreeArmRolloutResult:
    """Run selected, rival, and no-op arms against one frozen simulator plan."""

    from hailmary.evaluation.outcome import score_simulator_outcome

    parent_hash = dynamic_content_fingerprint(parent)
    if getattr(outcome_plan, "root_dynamic_content_hash", None) != parent_hash:
        raise SimulationError(
            "simulator outcome plan does not belong to the rollout parent root"
        )
    _assert_canonical_simulator_no_op(no_op_action)

    horizon_s = float(getattr(outcome_plan, "horizon_s"))
    return three_arm_rollout(
        parent,
        selected_action=selected_action,
        contender_action=contender_action,
        no_op_action=no_op_action,
        frozen_policy=frozen_policy,
        horizon_s=horizon_s,
        scorer=lambda branch: score_simulator_outcome(
            branch,
            outcome_plan=outcome_plan,
            config=outcome_config,
        ),
    )


__all__ = [
    "ActionApplier",
    "ForkableRollout",
    "PairedArmTrace",
    "PairedRolloutResult",
    "PolicyVsPermanentNoOpResult",
    "RolloutRunner",
    "ThreeArmRolloutResult",
    "dynamic_content_fingerprint",
    "paired_rollout",
    "paired_simulator_rollout",
    "policy_vs_permanent_no_op_rollout",
    "policy_vs_permanent_no_op_simulator_rollout",
    "rebind_action_to_branch",
    "three_arm_rollout",
    "three_arm_simulator_rollout",
]
