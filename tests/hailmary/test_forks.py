from __future__ import annotations

import json

import pytest

from hailmary.simulator import Simulator
from hailmary.rollout import NoOpPolicy, paired_rollout

from .test_engine import _definition


def test_forks_share_definition_but_have_independent_lineage_and_collections() -> None:
    parent = Simulator(_definition())

    left = parent.fork(label="selected")
    right = parent.fork(label="contender")

    assert left.state.definition is parent.state.definition
    assert right.state.definition is parent.state.definition
    assert left.state.definition.variants[0] is parent.state.definition.variants[0]
    assert left.state.dynamic_content_hash == parent.state.dynamic_content_hash
    assert right.state.dynamic_content_hash == parent.state.dynamic_content_hash
    assert len({parent.state.state_id, left.state.state_id, right.state.state_id}) == 3
    assert left.state.event_heap == parent.state.event_heap
    assert left.state.event_heap is not parent.state.event_heap


def test_child_progress_leaves_parent_byte_equivalent() -> None:
    parent = Simulator(_definition())
    parent_snapshot = json.dumps(
        parent.snapshot(), sort_keys=True, separators=(",", ":")
    )
    child = parent.fork(label="temporary")

    child.run()

    assert (
        json.dumps(parent.snapshot(), sort_keys=True, separators=(",", ":"))
        == parent_snapshot
    )
    assert parent.state.flight("F1").lifecycle.value == "scheduled"
    assert child.state.flight("F1").lifecycle.value == "completed"


def test_identical_branch_dynamics_have_same_content_hash_but_distinct_state_ids() -> (
    None
):
    parent = Simulator(_definition())
    left = parent.fork(label="left")
    right = parent.fork(label="right")

    left.run()
    right.run()

    assert left.state.dynamic_content_hash == right.state.dynamic_content_hash
    assert left.state.state_id != right.state.state_id
    assert parent.state.dynamic_content_hash != left.state.dynamic_content_hash


def test_rng_state_is_coupled_initially_and_branch_local_after_draws() -> None:
    parent = Simulator(_definition())
    left = parent.fork(label="left")
    right = parent.fork(label="right")
    parent_hash = parent.state.dynamic_content_hash

    left_first = left.random_uniform()
    right_first = right.random_uniform()

    assert left_first == right_first
    assert left.state.dynamic_content_hash == right.state.dynamic_content_hash
    assert parent.state.dynamic_content_hash == parent_hash
    left.random_uniform()
    assert left.state.dynamic_content_hash != right.state.dynamic_content_hash


def test_branch_metrics_and_action_log_do_not_escape_to_parent() -> None:
    parent = Simulator(_definition())
    child = parent.fork(label="metrics")

    child.add_metric("intervention_cost", 2.5)
    child.record_action({"action_id": "A1", "lever": "speed"})

    assert child.state.metrics_dict == {"intervention_cost": 2.5}
    assert child.state.action_log_records == ({"action_id": "A1", "lever": "speed"},)
    assert parent.state.metrics == ()
    assert parent.state.action_log == ()


def test_snapshot_resume_preserves_hash_and_exact_future_behavior() -> None:
    original = Simulator(_definition())
    original.advance_next()
    snapshot = original.snapshot()

    resumed = Simulator.resume(original.definition, snapshot)

    assert resumed.state.state_id == original.state.state_id
    assert resumed.state.dynamic_content_hash == original.state.dynamic_content_hash
    assert resumed.state.event_heap == original.state.event_heap
    original.run()
    resumed.run()
    assert resumed.state.dynamic_content_hash == original.state.dynamic_content_hash
    assert resumed.state.state_id == original.state.state_id


def test_snapshot_schema_is_strict_and_legacy_resume_is_bare_runtime_only() -> None:
    simulator = Simulator(_definition())
    snapshot = simulator.snapshot()

    with pytest.raises(
        ValueError,
        match="action_applier requires a non-empty runtime_configuration_hash",
    ):
        Simulator.resume(
            simulator.definition,
            snapshot,
            action_applier=lambda _simulator, _action: None,
        )

    legacy = dict(snapshot)
    legacy["schema_version"] = "hailmary.simulation-snapshot.v1"
    legacy.pop("runtime_configuration_hash")
    resumed = Simulator.resume(simulator.definition, legacy)
    assert resumed.dynamic_content_hash == simulator.dynamic_content_hash

    with pytest.raises(ValueError, match="cannot prove the configured resume runtime"):
        Simulator.resume(
            simulator.definition,
            legacy,
            runtime_configuration_hash="runtime-v1",
        )

    missing_hash = dict(snapshot)
    missing_hash.pop("runtime_configuration_hash")
    with pytest.raises(ValueError, match="requires runtime_configuration_hash"):
        Simulator.resume(simulator.definition, missing_hash)

    unsupported = dict(snapshot)
    unsupported["schema_version"] = "hailmary.simulation-snapshot.v999"
    with pytest.raises(ValueError, match="unsupported simulation snapshot"):
        Simulator.resume(simulator.definition, unsupported)


def test_simulator_satisfies_paired_rollout_protocol_for_identical_no_ops() -> None:
    parent = Simulator(_definition())

    result = paired_rollout(
        parent,
        selected_action={"lever": "no_op", "band": "no_op"},
        contender_action={"lever": "no_op", "band": "no_op"},
        frozen_policy=NoOpPolicy(),
        horizon_s=120.0,
        scorer=lambda branch: float(
            branch.state.flight("F1").lifecycle.value == "completed"
        ),
    )

    assert result.delta == 0.0
    assert (
        result.selected.final_dynamic_content_hash
        == result.contender.final_dynamic_content_hash
    )
    assert parent.state.flight("F1").lifecycle.value == "scheduled"
