from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from hailmary.scenario import (
    ArrivalClusterKey,
    DemandWindow,
    DemandWindowConfig,
    ObservedArrival,
    TerminalEntryCorpus,
    TrafficScaleConfig,
    TrafficScenarioBuilder,
    iter_demand_windows,
)
from hailmary.templates import ActionStation, ClusterTemplate, TrajectoryVariant


def _variant(cluster_id: str) -> TrajectoryVariant:
    stations = np.asarray([0.0, 5_000.0])
    speed = np.asarray([100.0, 100.0])
    return TrajectoryVariant.from_kinematic_profile(
        template_id=f"template:{cluster_id}",
        cluster_id=cluster_id,
        s_m=stations,
        east_m=stations,
        north_m=np.zeros(2),
        altitude_m=np.asarray([0.0, 1_500.0]),
        cas_mps=speed,
        lower_cas_mps=np.asarray([70.0, 70.0]),
        upper_cas_mps=np.asarray([130.0, 130.0]),
        threshold_resource_id="KATL:RW18R:threshold",
    )


def _template(key: ArrivalClusterKey) -> ClusterTemplate:
    variant = _variant(key.cluster)
    station = dict(
        entry_order=0,
        grid_index=1,
        s_m=float(variant.s_m[1]),
        east_m=float(variant.east_m[1]),
        north_m=float(variant.north_m[1]),
    )
    return ClusterTemplate(
        cluster_id=key.cluster,
        medoid_flight_id=f"MEDOID:{key.cluster}",
        member_count=1,
        baseline_variant=variant,
        speed_action_stations=(ActionStation(kind="speed", **station),),
        path_stretch_stations=(ActionStation(kind="path_stretch", **station),),
        dataset_id="phase01-test",
        airport_id=key.airport,
        runway_id=key.runway,
        expected_speed_station_count=1,
        expected_path_station_count=1,
    )


def _arrival(index: int, key: ArrivalClusterKey, time_s: float) -> ObservedArrival:
    return ObservedArrival(
        dataset_id="phase01-test",
        key=key,
        flight_id=f"F{index:03d}",
        terminal_entry_time_s=time_s,
        terminal_entry_ground_speed_mps=90.0 + index,
        terminal_entry_altitude_m=1_000.0 + 10.0 * index,
        baseline_variant_id=_template(key).baseline_variant.variant_id,
        source_day="2026-04-01",
    )


def _builder(arrivals: tuple[ObservedArrival, ...]) -> TrafficScenarioBuilder:
    templates = {
        item.cluster_id: _template(item.key)
        for item in arrivals
    }
    return TrafficScenarioBuilder(arrivals, templates_by_cluster=templates)


def test_demand_windows_are_half_open_and_stride_every_twenty_minutes() -> None:
    windows = iter_demand_windows(
        0.0,
        2_400.0,
        config=DemandWindowConfig(width_s=3_600, stride_s=1_200),
    )

    assert [(item.start_s, item.end_s) for item in windows] == [
        (0.0, 3_600.0),
        (1_200.0, 4_800.0),
    ]
    assert windows[0].contains(0.0)
    assert windows[0].contains(3_599.999)
    assert not windows[0].contains(3_600.0)
    assert windows[1].contains(3_600.0)


def test_half_up_scaling_is_per_cluster_and_runway_total_is_derived() -> None:
    cfg = TrafficScaleConfig(global_scale=1.05)
    assert cfg.target_count(34) == 36
    assert cfg.target_count(0) == 0

    key_1 = ArrivalClusterKey("KATL", "18R", "C1")
    key_2 = ArrivalClusterKey("KATL", "18R", "C2")
    arrivals = (_arrival(0, key_1, 100.0), _arrival(1, key_2, 200.0))
    scenario = _builder(arrivals).build_scenario(
        DemandWindow(0.0, 3_600.0),
        scale_config=TrafficScaleConfig(global_scale=1.5, master_seed=9),
    )

    assert {item.key: item.count for item in scenario.target_cluster_counts} == {
        key_1: 2,
        key_2: 2,
    }
    assert scenario.target_runway_counts[("KATL", "RW18R")] == 4
    assert TrafficScaleConfig(global_scale=1.5).target_count(2) == 3


def test_scale_one_replays_ids_timestamps_and_cluster_counts_exactly() -> None:
    key_1 = ArrivalClusterKey("KATL", "RW18R", "C1")
    key_2 = ArrivalClusterKey("KATL", "RW18R", "C2")
    arrivals = tuple(
        [
            *(_arrival(index, key_1, 10.0 + 50.0 * index) for index in range(34)),
            *(_arrival(34 + index, key_2, 2_000.0 + 100.0 * index) for index in range(3)),
        ]
    )
    scenario = _builder(tuple(reversed(arrivals))).build_scenario(
        DemandWindow(0.0, 3_600.0),
        scale_config=TrafficScaleConfig(global_scale=1.0),
    )
    expected = sorted(
        (item.flight_id, item.terminal_entry_time_s, item.cluster_id)
        for item in arrivals
    )
    actual = sorted(
        (item.flight_id, item.release_time_s, item.cluster_id)
        for item in scenario.definition.flights
    )

    assert actual == expected
    assert scenario.definition.schema_version == "hailmary.scenario.v2"
    assert all(flight.action_stations for flight in scenario.definition.flights)
    for flight in scenario.definition.flights:
        variant = scenario.definition.variant(flight.baseline_variant_id)
        source = next(item for item in arrivals if item.flight_id == flight.flight_id)
        assert variant.ground_speed_mps[-1] == pytest.approx(
            source.terminal_entry_ground_speed_mps
        )
        assert variant.altitude_m[-1] == pytest.approx(
            source.terminal_entry_altitude_m
        )
    assert Counter(item.cluster_id for item in scenario.definition.flights) == {
        key_1.qualified_id: 34,
        key_2.qualified_id: 3,
    }


def test_scaling_is_deterministic_input_order_independent_and_jointly_donated() -> None:
    key_1 = ArrivalClusterKey("KATL", "RW18R", "C1")
    key_2 = ArrivalClusterKey("KATL", "RW18R", "C2")
    arrivals = tuple(
        [
            *(_arrival(index, key_1, 10.0 + 50.0 * index) for index in range(34)),
            *(_arrival(34 + index, key_2, 2_000.0 + 100.0 * index) for index in range(3)),
        ]
    )
    cfg = TrafficScaleConfig(global_scale=1.05, master_seed=41, replicate=2)
    first = _builder(arrivals).build_scenario(DemandWindow(0.0, 3_600.0), scale_config=cfg)
    second = _builder(tuple(reversed(arrivals))).build_scenario(
        DemandWindow(0.0, 3_600.0), scale_config=cfg
    )

    assert first.definition.definition_hash == second.definition.definition_hash
    assert {item.key: item.count for item in first.target_cluster_counts} == {
        key_1: 36,
        key_2: 3,
    }
    profiles = {
        item.flight_id: (
            item.cluster_id,
            item.terminal_entry_ground_speed_mps,
            item.terminal_entry_altitude_m,
        )
        for item in arrivals
    }
    synthetic = [
        item for item in first.definition.flights if item.metadata_dict["synthetic"]
    ]
    assert len(synthetic) == 2
    for flight in synthetic:
        metadata = flight.metadata_dict
        donor = profiles[metadata["donor_flight_id"]]
        assert donor == (
            flight.cluster_id,
            metadata["terminal_entry_ground_speed_mps"],
            metadata["terminal_entry_altitude_m"],
        )


def test_seeded_thinning_is_exact_without_replacement() -> None:
    key = ArrivalClusterKey("KATL", "RW18R", "C1")
    arrivals = tuple(_arrival(index, key, 100.0 + index) for index in range(10))
    cfg = TrafficScaleConfig(global_scale=0.54, master_seed=3)
    first = _builder(arrivals).build_scenario(DemandWindow(0.0, 3_600.0), scale_config=cfg)
    second = _builder(tuple(reversed(arrivals))).build_scenario(
        DemandWindow(0.0, 3_600.0), scale_config=cfg
    )
    ids = tuple(sorted(item.flight_id for item in first.definition.flights))

    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert ids == tuple(sorted(item.flight_id for item in second.definition.flights))
    assert set(ids).issubset({item.flight_id for item in arrivals})


def test_traffic_scale_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TrafficScaleConfig(global_scale=-1.0)
    with pytest.raises(ValueError, match="non-negative integer"):
        TrafficScaleConfig().target_count(-1)


def test_training_intensity_corpus_must_be_observed_and_partition_compatible() -> None:
    key = ArrivalClusterKey("KATL", "RW18R", "C1")
    arrival = _arrival(0, key, 100.0)
    synthetic = ObservedArrival(
        dataset_id=arrival.dataset_id,
        key=key,
        flight_id="SYNTHETIC",
        terminal_entry_time_s=200.0,
        terminal_entry_ground_speed_mps=90.0,
        terminal_entry_altitude_m=1_000.0,
        baseline_variant_id=arrival.baseline_variant_id,
        synthetic=True,
        donor_flight_id=arrival.flight_id,
    )

    with pytest.raises(ValueError, match="cannot contain synthetic"):
        TrafficScenarioBuilder(
            (arrival,),
            templates_by_cluster={key.qualified_id: _template(key)},
            training_arrivals=(synthetic,),
        )


def test_terminal_entry_corpus_round_trip_is_hashed(tmp_path: Path) -> None:
    key = ArrivalClusterKey("KATL", "RW18R", "C1")
    corpus = TerminalEntryCorpus(
        dataset_id="phase01-test",
        airport="KATL",
        arrivals=(_arrival(0, key, 100.0),),
        rejection_counts=(("missing_raw_track", 2),),
    )
    path = corpus.write(tmp_path / "traffic_corpus.json")

    assert TerminalEntryCorpus.read(path).to_dict() == corpus.to_dict()

    tampered = corpus.to_dict()
    tampered["arrivals"][0]["terminal_entry_time_s"] = 101.0
    with pytest.raises(ValueError, match="content hash"):
        TerminalEntryCorpus.from_dict(json.loads(json.dumps(tampered)))
