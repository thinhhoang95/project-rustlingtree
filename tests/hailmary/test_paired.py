from __future__ import annotations

from dataclasses import dataclass

import pytest

from hailmary.errors import SimulationError
from hailmary.ids import content_hash
from hailmary.rollout.paired import paired_rollout
from hailmary.rollout.policy import NoOpPolicy


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
    assert result.selected.final_dynamic_content_hash == result.contender.final_dynamic_content_hash
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
