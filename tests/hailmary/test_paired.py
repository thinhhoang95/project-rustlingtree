from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from hailmary.errors import SimulationError
from hailmary.ids import content_hash
from hailmary.rollout import (
    CallableFrozenPolicy,
    paired_rollout,
    paired_simulator_rollout,
    policy_fingerprint,
    policy_vs_permanent_no_op_rollout,
    three_arm_rollout,
)
from hailmary.rollout.policy import NoOpPolicy


_GLOBAL_POLICY_STATE = {"action": 1}
_DELEGATED_POLICY_STATE = {"action": 1}


def _global_state_selector(_context: object) -> int:
    return _GLOBAL_POLICY_STATE["action"]


def _delegated_state_selector(_context: object) -> int:
    return _DELEGATED_POLICY_STATE["action"]


def _delegating_selector(context: object) -> int:
    return _delegated_state_selector(context)


def _same_qualname_literal_selectors():
    def selector(_context: object) -> str:
        return "alpha"

    first = selector

    def selector(_context: object) -> str:
        return "beta"

    return first, selector


class CallableSelector:
    def __init__(self, action: int) -> None:
        self.action = action

    def __call__(self, _context: object) -> int:
        return self.action


class ClassOwnedPolicyState:
    action = 1


def _class_owned_state_selector(_context: object) -> int:
    return ClassOwnedPolicyState.action


@dataclass(frozen=True)
class SameStateDifferentPolicy:
    action: object | None = None

    def select_action(self, _context: object) -> int:
        return 99


class ClassAttributeSelectPolicy:
    action = 1

    def select_action(self, _context: object) -> int:
        return self.action


class ClassAttributeCallablePolicy:
    action = 1

    def __call__(self, _context: object) -> int:
        return type(self).action


class FakeBranch:
    def __init__(self, value: int = 0, time_s: float = 0.0) -> None:
        self.value = value
        self.time_s = time_s

    @property
    def dynamic_content_hash(self) -> str:
        return content_hash(
            {"value": self.value, "time_s": self.time_s},
            namespace="test.fake_branch",
        )

    def fork(self, *, label: str) -> "FakeBranch":
        del label
        return FakeBranch(self.value, self.time_s)

    def apply(self, action: int) -> None:
        self.value += int(action)

    def run_until(self, horizon_s: float, *, policy: object) -> None:
        self.time_s = float(horizon_s)
        selected = policy.select_action(self)  # type: ignore[attr-defined]
        if selected is not None:
            self.apply(int(selected))


def test_paired_rollout_leaves_parent_unchanged() -> None:
    parent = FakeBranch(value=10)
    before = parent.dynamic_content_hash

    result = paired_rollout(
        parent,
        selected_action=3,
        contender_action=1,
        frozen_policy=NoOpPolicy(),
        horizon_s=500.0,
        scorer=lambda branch: float(branch.value),
    )

    assert result.delta == 2.0
    assert result.selected.score == 13.0
    assert result.contender.score == 11.0
    assert parent.dynamic_content_hash == before
    assert parent.value == 10
    assert parent.time_s == 0.0


def test_three_arm_rollout_computes_all_causal_deltas_from_one_parent() -> None:
    parent = FakeBranch(value=10)
    before = parent.dynamic_content_hash

    result = three_arm_rollout(
        parent,
        selected_action=3,
        contender_action=1,
        no_op_action=0,
        frozen_policy=NoOpPolicy(),
        horizon_s=500.0,
        scorer=lambda branch: float(branch.value),
    )

    assert result.selected.score == 13.0
    assert result.contender.score == 11.0
    assert result.no_op.score == 10.0
    assert result.delta_rival == 2.0
    assert result.delta_selected_noop == 3.0
    assert result.delta_contender_noop == 1.0
    assert result.delta_veto == -3.0
    assert {
        result.selected.initial_dynamic_content_hash,
        result.contender.initial_dynamic_content_hash,
        result.no_op.initial_dynamic_content_hash,
    } == {before}
    assert parent.dynamic_content_hash == before
    assert parent.value == 10
    assert parent.time_s == 0.0


@pytest.mark.parametrize(
    ("selected_action", "contender_action", "equal_arm_names", "zero_delta_name"),
    [
        (2, 2, ("selected", "contender"), "delta_rival"),
        (0, 2, ("selected", "no_op"), "delta_selected_noop"),
        (2, 0, ("contender", "no_op"), "delta_contender_noop"),
    ],
)
def test_three_arm_rollout_enforces_exact_equality_for_equivalent_arms(
    selected_action: int,
    contender_action: int,
    equal_arm_names: tuple[str, str],
    zero_delta_name: str,
) -> None:
    result = three_arm_rollout(
        FakeBranch(value=10),
        selected_action=selected_action,
        contender_action=contender_action,
        no_op_action=0,
        frozen_policy=NoOpPolicy(),
        horizon_s=500.0,
        scorer=lambda branch: float(branch.value),
    )

    left = getattr(result, equal_arm_names[0])
    right = getattr(result, equal_arm_names[1])
    assert left.final_dynamic_content_hash == right.final_dynamic_content_hash
    assert left.score == right.score
    assert getattr(result, zero_delta_name) == 0.0


@dataclass(frozen=True)
class StateDependentPolicy:
    def select_action(self, context: FakeBranch) -> int:
        return context.value


def test_three_arms_may_take_different_later_actions_under_one_policy() -> None:
    result = three_arm_rollout(
        FakeBranch(),
        selected_action=3,
        contender_action=1,
        no_op_action=0,
        frozen_policy=StateDependentPolicy(),
        horizon_s=100.0,
        scorer=lambda branch: float(branch.value),
    )

    assert (result.selected.score, result.contender.score, result.no_op.score) == (
        6.0,
        2.0,
        0.0,
    )
    assert result.delta_rival == 4.0
    assert result.delta_selected_noop == 6.0
    assert result.delta_contender_noop == 2.0


def test_identical_arms_have_identical_hash_and_exact_zero_delta() -> None:
    parent = FakeBranch(value=10)

    result = paired_rollout(
        parent,
        selected_action=2,
        contender_action=2,
        frozen_policy=NoOpPolicy(),
        horizon_s=500.0,
        scorer=lambda branch: float(branch.value),
    )

    assert result.delta == 0.0
    assert (
        result.selected.final_dynamic_content_hash
        == result.contender.final_dynamic_content_hash
    )
    assert result.selected.score == result.contender.score


def test_free_function_engine_adapters_may_return_audit_and_batch_data() -> None:
    parent = FakeBranch(value=4)

    def apply_action(branch: FakeBranch, action: int) -> dict[str, int]:
        branch.apply(action)
        return {"realized": action}

    def run_branch(branch: FakeBranch, horizon_s: float, policy: object) -> tuple[str]:
        branch.run_until(horizon_s, policy=policy)
        return ("batch",)

    result = paired_rollout(
        parent,
        selected_action=2,
        contender_action=1,
        frozen_policy=NoOpPolicy(),
        horizon_s=80.0,
        scorer=lambda branch: float(branch.value),
        action_applier=apply_action,
        rollout_runner=run_branch,
    )

    assert result.delta == 1.0
    assert parent.value == 4


@dataclass
class MutatingPolicy:
    calls: int = 0

    def select_action(self, context: object) -> None:
        del context
        self.calls += 1
        return None


def test_three_arm_rollout_rejects_policy_mutation() -> None:
    with pytest.raises(SimulationError, match="frozen policy mutated"):
        three_arm_rollout(
            FakeBranch(),
            selected_action=2,
            contender_action=1,
            no_op_action=0,
            frozen_policy=MutatingPolicy(),
            horizon_s=100.0,
            scorer=lambda branch: float(branch.value),
        )


def test_paired_rollout_rejects_policy_mutation() -> None:
    with pytest.raises(SimulationError, match="frozen policy mutated"):
        paired_rollout(
            FakeBranch(),
            selected_action=0,
            contender_action=0,
            frozen_policy=MutatingPolicy(),
            horizon_s=100.0,
            scorer=lambda branch: float(branch.value),
        )


def test_callable_policy_fingerprint_tracks_captured_mutable_state() -> None:
    calls: list[int] = []

    def selector(_context: object) -> None:
        calls.append(len(calls))

    policy = CallableFrozenPolicy(selector, name="captured-mutation")
    before = policy.policy_fingerprint()

    with pytest.raises(SimulationError, match="policy mutated"):
        three_arm_rollout(
            FakeBranch(),
            selected_action=1,
            contender_action=2,
            no_op_action=0,
            frozen_policy=policy,
            horizon_s=1.0,
            scorer=lambda branch: float(branch.value),
        )

    assert calls
    assert policy.policy_fingerprint() != before


def test_callable_policy_fingerprint_tracks_referenced_global_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = policy_fingerprint(_global_state_selector)
    monkeypatch.setitem(_GLOBAL_POLICY_STATE, "action", 2)

    assert _global_state_selector(object()) == 2
    assert policy_fingerprint(_global_state_selector) != before


def test_callable_policy_fingerprint_tracks_delegated_function_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = policy_fingerprint(_delegating_selector)
    monkeypatch.setitem(_DELEGATED_POLICY_STATE, "action", 2)

    assert _delegating_selector(object()) == 2
    assert policy_fingerprint(_delegating_selector) != before


def test_callable_policy_fingerprint_rejects_opaque_rng_state() -> None:
    rng = np.random.default_rng(20260717)

    def selector(_context: object) -> float:
        return float(rng.random())

    with pytest.raises(TypeError, match="explicit policy_fingerprint"):
        policy_fingerprint(selector)


def test_recursive_function_fingerprint_is_finite_and_deterministic() -> None:
    def selector(remaining: int) -> int:
        if remaining <= 0:
            return 0
        return selector(remaining - 1)

    assert selector(3) == 0
    assert policy_fingerprint(selector) == policy_fingerprint(selector)


def test_callable_policy_fingerprint_distinguishes_same_qualname_constants() -> None:
    first, second = _same_qualname_literal_selectors()

    assert first.__qualname__ == second.__qualname__
    assert first.__code__.co_code == second.__code__.co_code
    assert policy_fingerprint(first) != policy_fingerprint(second)


def test_callable_policy_fingerprint_tracks_callable_instance_state() -> None:
    selector = CallableSelector(1)
    policy = CallableFrozenPolicy(selector, name="callable-instance")
    before = policy.policy_fingerprint()
    selector.action = 2

    assert selector(object()) == 2
    assert policy.policy_fingerprint() != before


def test_callable_policy_fingerprint_rejects_module_owned_mutable_state() -> None:
    policy_module = ModuleType("mutable_policy_fixture")
    policy_module.state = {"action": 1}  # type: ignore[attr-defined]

    def selector(_context: object) -> int:
        return policy_module.state["action"]  # type: ignore[attr-defined, no-any-return]

    assert selector(object()) == 1
    with pytest.raises(TypeError, match="explicit policy_fingerprint"):
        policy_fingerprint(selector)


def test_callable_policy_fingerprint_rejects_class_owned_mutable_state() -> None:
    assert _class_owned_state_selector(object()) == 1
    with pytest.raises(TypeError, match="explicit policy_fingerprint"):
        policy_fingerprint(_class_owned_state_selector)


def test_policy_fingerprint_includes_dataclass_policy_type_and_behavior() -> None:
    assert policy_fingerprint(NoOpPolicy()) != policy_fingerprint(
        SameStateDifferentPolicy()
    )


def test_policy_fingerprint_tracks_class_attribute_read_through_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = ClassAttributeSelectPolicy()
    before = policy_fingerprint(policy)

    monkeypatch.setattr(ClassAttributeSelectPolicy, "action", 2)

    assert policy.select_action(object()) == 2
    assert policy_fingerprint(policy) != before


def test_callable_policy_fingerprint_tracks_class_attribute_read_through_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = ClassAttributeCallablePolicy()
    before = policy_fingerprint(policy)

    monkeypatch.setattr(ClassAttributeCallablePolicy, "action", 2)

    assert policy(object()) == 2
    assert policy_fingerprint(policy) != before


def test_simulator_rollout_rejects_wrong_plan_root_before_forking() -> None:
    parent = FakeBranch()

    with pytest.raises(SimulationError, match="does not belong"):
        paired_simulator_rollout(
            parent,
            selected_action=1,
            contender_action=0,
            frozen_policy=NoOpPolicy(),
            outcome_plan=SimpleNamespace(
                root_dynamic_content_hash="not-the-parent",
                horizon_s=1.0,
            ),
        )

    assert parent.value == 0
    assert parent.time_s == 0.0


def test_paired_rollout_rejects_delta_overflow_from_finite_scores() -> None:
    def extreme_score(branch: FakeBranch) -> float:
        return {
            -1: -1.0e308,
            1: 1.0e308,
        }[branch.value]

    with pytest.raises(SimulationError, match="paired delta is non-finite"):
        paired_rollout(
            FakeBranch(),
            selected_action=1,
            contender_action=-1,
            frozen_policy=NoOpPolicy(),
            horizon_s=1.0,
            scorer=extreme_score,
        )


def test_three_arm_rollout_rejects_delta_overflow_from_finite_scores() -> None:
    def extreme_score(branch: FakeBranch) -> float:
        return {
            -1: -1.0e308,
            0: 0.0,
            1: 1.0e308,
        }[branch.value]

    with pytest.raises(
        SimulationError,
        match="three-arm selected/no-op delta is non-finite",
    ):
        three_arm_rollout(
            FakeBranch(),
            selected_action=1,
            contender_action=0,
            no_op_action=-1,
            frozen_policy=NoOpPolicy(),
            horizon_s=1.0,
            scorer=extreme_score,
        )


@pytest.mark.parametrize(
    ("selected_action", "contender_action"),
    [
        (-1, 0),
        (0, -1),
        (0, 0),
    ],
)
def test_veto_delta_uses_literal_max_when_an_arm_duplicates_no_op(
    selected_action: int,
    contender_action: int,
) -> None:
    result = three_arm_rollout(
        FakeBranch(),
        selected_action=selected_action,
        contender_action=contender_action,
        no_op_action=0,
        frozen_policy=NoOpPolicy(),
        horizon_s=1.0,
        scorer=lambda branch: float(branch.value),
    )

    assert result.delta_veto == (
        result.no_op.score - max(result.selected.score, result.contender.score)
    )
    assert result.delta_veto == 0.0


@dataclass(frozen=True)
class AlwaysOnePolicy:
    def select_action(self, context: FakeBranch) -> int:
        del context
        return 1


def test_policy_vs_permanent_no_op_uses_no_op_policy_for_whole_horizon() -> None:
    parent = FakeBranch(value=5)
    parent_before = parent.dynamic_content_hash
    learned_policy = AlwaysOnePolicy()

    def run_three_later_decisions(
        branch: FakeBranch,
        horizon_s: float,
        policy: object,
    ) -> None:
        branch.time_s = horizon_s
        for _ in range(3):
            action = policy.select_action(branch)  # type: ignore[attr-defined]
            if action is not None:
                branch.apply(action)

    result = policy_vs_permanent_no_op_rollout(
        parent,
        policy_action=0,
        no_op_action=0,
        frozen_policy=learned_policy,
        horizon_s=100.0,
        scorer=lambda branch: float(branch.value),
        rollout_runner=run_three_later_decisions,
    )

    assert result.policy_arm.initial_action == result.permanent_no_op.initial_action
    assert result.policy_arm.score == 8.0
    assert result.permanent_no_op.score == 5.0
    assert result.delta == 3.0
    assert result.policy_arm.initial_dynamic_content_hash == parent_before
    assert result.permanent_no_op.initial_dynamic_content_hash == parent_before
    assert result.parent_dynamic_content_hash == parent_before
    assert result.learned_policy_fingerprint == policy_fingerprint(learned_policy)
    assert result.permanent_no_op_policy_fingerprint == policy_fingerprint(NoOpPolicy())
    assert parent.dynamic_content_hash == parent_before
    assert parent.value == 5
    assert parent.time_s == 0.0
