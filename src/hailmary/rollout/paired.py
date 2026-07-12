"""Coupled temporary branches for selected-versus-contender rollouts."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from hailmary.errors import SimulationError
from hailmary.ids import canonical_json, content_hash
from hailmary.rollout.policy import FrozenPolicy, policy_fingerprint


@runtime_checkable
class ForkableRollout(Protocol):
    def fork(self, *, label: str) -> Any: ...


ActionApplier = Callable[[Any, Any], Any | None]
RolloutRunner = Callable[[Any, float, FrozenPolicy], Any | None]


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
    raise TypeError("rollout objects must expose dynamic_content_hash/content_hash or canonical state")


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
        raise TypeError("forked rollout branch must implement apply(action) or apply_action(action)")
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
        raise TypeError("forked rollout branch must implement run_until(horizon, policy=...)")
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
    branch = _apply_initial_action(branch, applied_action, action_applier=action_applier)
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
    if dynamic_content_fingerprint(parent) != parent_before:
        raise SimulationError("selected temporary rollout mutated its parent")
    if policy_fingerprint(frozen_policy) != policy_before:
        raise SimulationError("frozen policy mutated during selected temporary rollout")

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
    if dynamic_content_fingerprint(parent) != parent_before:
        raise SimulationError("contender temporary rollout mutated its parent")
    if policy_fingerprint(frozen_policy) != policy_before:
        raise SimulationError("frozen policy mutated during contender temporary rollout")

    equivalent_actions = _actions_equivalent(selected_action, contender_action)
    if equivalent_actions:
        if selected.final_dynamic_content_hash != contender.final_dynamic_content_hash:
            raise SimulationError("identical paired actions produced different dynamic-content hashes")
        if selected.score != contender.score:
            raise SimulationError("identical paired actions produced different scores")
        delta = 0.0
    else:
        delta = float(selected.score - contender.score)

    return PairedRolloutResult(
        selected=selected,
        contender=contender,
        delta=delta,
        parent_dynamic_content_hash=parent_before,
        policy_fingerprint=policy_before,
        horizon_s=horizon,
    )


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

    cohort = getattr(outcome_plan, "cohort")
    horizon_s = float(getattr(outcome_plan, "horizon_s"))
    return paired_rollout(
        parent,
        selected_action=selected_action,
        contender_action=contender_action,
        frozen_policy=frozen_policy,
        horizon_s=horizon_s,
        scorer=lambda branch: score_simulator_outcome(
            branch,
            cohort,
            config=outcome_config,
            horizon_s=horizon_s,
        ),
    )


__all__ = [
    "ActionApplier",
    "ForkableRollout",
    "PairedArmTrace",
    "PairedRolloutResult",
    "RolloutRunner",
    "dynamic_content_fingerprint",
    "paired_rollout",
    "paired_simulator_rollout",
    "rebind_action_to_branch",
]
