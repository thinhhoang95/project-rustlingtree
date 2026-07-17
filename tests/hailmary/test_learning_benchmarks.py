from __future__ import annotations

import math
from pathlib import Path
import runpy

import pytest


MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "hailmary"
        / "benchmark_learning.py"
    )
)
RUN_CASES = MODULE["run_cases"]
LIST_CASES = MODULE["list_cases"]
BENCHMARK_CASES = MODULE["BENCHMARK_CASES"]
PRODUCTION_CASES = (
    "fork_latency_100_flights",
    "feature_vector_cache_hit",
    "feature_vector_cache_miss",
    "three_arm_no_op",
    "three_arm_speed",
    "three_arm_stretch_oracle",
    "complete_training_epochs_per_second",
    "temporary_fork_memory_growth",
)


def test_benchmark_registry_contains_every_required_production_contract() -> None:
    records = {item["name"]: item for item in LIST_CASES()}

    assert set(PRODUCTION_CASES) <= set(records)
    assert all(records[name]["default_status"] == "ok" for name in PRODUCTION_CASES)
    assert records["fork_latency_100_flights"]["proposed_targets"] == {
        "average_ms_lt": 1.0
    }
    assert records["feature_vector_cache_hit"]["proposed_targets"] == {
        "average_ms_lt": 2.0
    }


def test_default_production_benchmarks_execute_real_operations() -> None:
    results = RUN_CASES(
        PRODUCTION_CASES,
        population_size=10,
        iterations=1,
    )
    by_name = {result["name"]: result for result in results}

    assert set(by_name) == set(PRODUCTION_CASES)
    assert all(result["status"] == "ok" for result in results)
    assert all(
        result["metrics"]
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in result["metrics"].values()
        )
        for result in results
    )
    for name in (
        "fork_latency_100_flights",
        "feature_vector_cache_hit",
        "feature_vector_cache_miss",
        "temporary_fork_memory_growth",
    ):
        assert by_name[name]["metrics"]["active_flights"] == 100
    for name in ("three_arm_no_op", "three_arm_speed", "three_arm_stretch_oracle"):
        assert by_name[name]["metrics"]["arms"] == 3
        assert by_name[name]["metrics"]["active_flights"] == 5
    assert (
        by_name["complete_training_epochs_per_second"]["metrics"]["committed_epochs"]
        == 1
    )
    assert (
        by_name["complete_training_epochs_per_second"]["metrics"]["exploitation_epochs"]
        == 1
    )
    assert by_name["temporary_fork_memory_growth"]["metrics"]["temporary_forks"] == 1


def test_site_fixture_runner_overrides_the_executable_representative() -> None:
    default_runner = BENCHMARK_CASES["fork_latency_100_flights"].runner
    assert default_runner is not None

    def site_runner(context: object) -> dict[str, float | int]:
        metrics = dict(default_runner(context))
        metrics["site_override"] = 1
        return metrics

    (result,) = RUN_CASES(
        ("fork_latency_100_flights",),
        population_size=20,
        iterations=1,
        fixture_runners={"fork_latency_100_flights": site_runner},
    )

    assert result["status"] == "ok"
    assert result["metrics"]["active_flights"] == 100
    assert result["metrics"]["site_override"] == 1
    assert result["fixture_requirement"]


@pytest.mark.parametrize("bad_metric", [float("nan"), True, "slow"])
def test_fixture_runner_rejects_non_numeric_or_nonfinite_metrics(
    bad_metric: object,
) -> None:
    with pytest.raises(ValueError, match="finite number"):
        RUN_CASES(
            ("three_arm_speed",),
            iterations=1,
            fixture_runners={
                "three_arm_speed": lambda _context: {"average_ms": bad_metric}
            },
        )
