"""Small, reproducible checks for demand windows and traffic scaling."""

from __future__ import annotations

from collections import Counter
import json

import numpy as np

from hailmary.scenario import (
    ArrivalClusterKey,
    DemandWindow,
    DemandWindowConfig,
    ObservedArrival,
    TrafficScaleConfig,
    TrafficScenarioBuilder,
    iter_demand_windows,
)
from hailmary.templates import ActionStation, ClusterTemplate, TrajectoryVariant


def _variant(cluster_id: str) -> TrajectoryVariant:
    values = np.asarray([0.0, 5_000.0])
    speed = np.asarray([100.0, 100.0])
    return TrajectoryVariant.from_kinematic_profile(
        template_id=f"template:{cluster_id}",
        cluster_id=cluster_id,
        s_m=values,
        east_m=values,
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
        dataset_id="phase01-example",
        airport_id=key.airport,
        runway_id=key.runway,
        expected_speed_station_count=1,
        expected_path_station_count=1,
    )


def _arrival(index: int, key: ArrivalClusterKey, time_s: float) -> ObservedArrival:
    # The paired values make joint donor sampling directly auditable.
    speed = 90.0 + index
    altitude = 1_000.0 + 10.0 * index
    return ObservedArrival(
        dataset_id="phase01-example",
        key=key,
        flight_id=f"OBS{index:03d}",
        terminal_entry_time_s=time_s,
        terminal_entry_ground_speed_mps=speed,
        terminal_entry_altitude_m=altitude,
        baseline_variant_id=_template(key).baseline_variant.variant_id,
        source_day="2026-04-01",
    )


def main() -> None:
    key_1 = ArrivalClusterKey("KATL", "18R", "C1")
    key_2 = ArrivalClusterKey("KATL", "18R", "C2")
    arrivals = tuple(
        [
            *(_arrival(index, key_1, 10.0 + index * 50.0) for index in range(34)),
            *(
                _arrival(34 + index, key_2, 2_000.0 + index * 100.0)
                for index in range(3)
            ),
        ]
    )
    templates = {
        key_1.qualified_id: _template(key_1),
        key_2.qualified_id: _template(key_2),
    }
    builder = TrafficScenarioBuilder(arrivals, templates_by_cluster=templates)
    window = DemandWindow(0.0, 3_600.0)

    baseline = builder.build_scenario(
        window, scale_config=TrafficScaleConfig(global_scale=1.0)
    )
    scaled = builder.build_scenario(
        window,
        scale_config=TrafficScaleConfig(global_scale=1.05, master_seed=41),
    )
    flight_metadata = {
        flight.flight_id: flight.metadata_dict for flight in scaled.definition.flights
    }
    synthetic = [
        {
            "flight_id": flight.flight_id,
            "cluster": flight.cluster_id,
            "donor_flight_id": flight_metadata[flight.flight_id]["donor_flight_id"],
            "ground_speed_mps": flight_metadata[flight.flight_id][
                "terminal_entry_ground_speed_mps"
            ],
            "altitude_m": flight_metadata[flight.flight_id][
                "terminal_entry_altitude_m"
            ],
        }
        for flight in scaled.definition.flights
        if flight_metadata[flight.flight_id]["synthetic"]
    ]
    observed_profile = {
        item.flight_id: (
            item.terminal_entry_ground_speed_mps,
            item.terminal_entry_altitude_m,
        )
        for item in arrivals
    }
    donor_profiles_match = all(
        observed_profile[item["donor_flight_id"]]
        == (item["ground_speed_mps"], item["altitude_m"])
        for item in synthetic
    )

    boundary_times = (0.0, 1_199.999, 1_200.0, 3_599.999, 3_600.0, 4_799.999)
    windows = iter_demand_windows(
        0.0, 2_400.0, config=DemandWindowConfig(width_s=3_600, stride_s=1_200)
    )
    membership = {
        f"[{item.start_s:g},{item.end_s:g})": [
            value for value in boundary_times if item.contains(value)
        ]
        for item in windows
    }

    result = {
        "demand_windows": membership,
        "half_up_examples": {
            "34_x_1.05": TrafficScaleConfig(global_scale=1.05).target_count(34),
            "two_cluster_1_x_1.5_each": 2
            * TrafficScaleConfig(global_scale=1.5).target_count(1),
            "independent_runway_rounding_for_2_x_1.5": TrafficScaleConfig(
                global_scale=1.5
            ).target_count(2),
            "zero_x_3": TrafficScaleConfig(global_scale=3.0).target_count(0),
        },
        "scale_1": {
            "timestamps_exact": tuple(
                flight.release_time_s for flight in baseline.definition.flights
            )
            == tuple(item.terminal_entry_time_s for item in arrivals),
            "cluster_counts": dict(
                sorted(Counter(flight.cluster_id for flight in baseline.definition.flights).items())
            ),
        },
        "scale_1_05": {
            "observed_cluster_counts": {
                item.key.qualified_id: item.count for item in scaled.observed_cluster_counts
            },
            "target_cluster_counts": {
                item.key.qualified_id: item.count for item in scaled.target_cluster_counts
            },
            "derived_runway_total": next(iter(scaled.target_runway_counts.values())),
            "synthetic_addition_count": len(synthetic),
            "joint_donor_profiles_match": donor_profiles_match,
            "synthetic_provenance": synthetic,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
