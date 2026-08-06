from __future__ import annotations

import pytest

from hailmary.learning import Phase0ExperimentRunner
from hailmary.scenario import (
    ArrivalClusterKey,
    DemandWindow,
    TrafficScaleConfig,
    TrafficScenarioBatch,
)

from .test_phase01_traffic import _arrival, _builder


def test_phase0_batch_contract_accepts_only_exact_scale_one_traffic() -> None:
    key = ArrivalClusterKey("KATL", "RW18R", "C1")
    builder = _builder((_arrival(0, key, 100.0),))
    scenario = builder.build_scenario(
        DemandWindow(0.0, 3_600.0),
        scale_config=TrafficScaleConfig(global_scale=1.0),
    )
    batch = TrafficScenarioBatch(
        dataset_id="phase01-test",
        scale_config=TrafficScaleConfig(global_scale=1.0),
        scenarios=(scenario,),
        source_partition="2026-04-01",
    )
    runner = object.__new__(Phase0ExperimentRunner)

    assert runner._validate_batch(batch, name="training_batch") == (scenario,)

    scaled_scenario = builder.build_scenario(
        DemandWindow(0.0, 3_600.0),
        scale_config=TrafficScaleConfig(global_scale=2.0),
    )
    scaled_batch = TrafficScenarioBatch(
        dataset_id="phase01-test",
        scale_config=TrafficScaleConfig(global_scale=2.0),
        scenarios=(scaled_scenario,),
        source_partition="2026-04-01",
    )
    with pytest.raises(ValueError, match="global scale 1.0"):
        runner._validate_batch(scaled_batch, name="training_batch")


def test_phase1_batch_contract_cannot_mix_window_scales() -> None:
    key = ArrivalClusterKey("KATL", "RW18R", "C1")
    builder = _builder((_arrival(0, key, 100.0),))
    scale_one = builder.build_scenario(
        DemandWindow(0.0, 3_600.0),
        scale_config=TrafficScaleConfig(global_scale=1.0),
    )

    with pytest.raises(ValueError, match="batch global scale"):
        TrafficScenarioBatch(
            dataset_id="phase01-test",
            scale_config=TrafficScaleConfig(global_scale=1.5),
            scenarios=(scale_one,),
        )

    with pytest.raises(ValueError, match="batch dataset"):
        TrafficScenarioBatch(
            dataset_id="other-dataset",
            scale_config=TrafficScaleConfig(global_scale=1.0),
            scenarios=(scale_one,),
        )
