from __future__ import annotations

import numpy as np
import pytest

from hailmary.config import M_PER_NM
from hailmary.geometry.frame import LocalFrame
from hailmary.templates import (
    ClusterTemplate,
    MedoidTrack,
    TemplateCompiler,
    TemplateStore,
    TrajectoryVariant,
)
from hailmary.templates.compiler import select_action_stations


def _straight_variant(
    *,
    variant_template_id: str = "SYNTHETIC_TEMPLATE",
    cluster_id: str = "CLUSTER_A",
    length_m: float = 100_000.0,
    speed_mps: float = 100.0,
    station_count: int = 501,
    lower_cas_mps: float = 70.0,
) -> TrajectoryVariant:
    stations = np.linspace(0.0, length_m, station_count, dtype=np.float64)
    east = stations.copy()
    north = np.zeros(station_count, dtype=np.float64)
    lat, lon = LocalFrame(0.0, 0.0).unproject(east, north)
    speed = np.full(station_count, speed_mps, dtype=np.float64)
    return TrajectoryVariant.from_kinematic_profile(
        template_id=variant_template_id,
        cluster_id=cluster_id,
        s_m=stations,
        lat_deg=lat,
        lon_deg=lon,
        east_m=east,
        north_m=north,
        altitude_m=np.zeros(station_count, dtype=np.float64),
        cas_mps=speed,
        tas_mps=speed,
        ground_speed_mps=speed,
        command_cas_mps=speed,
        reference_command_cas_mps=speed,
        lower_cas_mps=np.full(station_count, lower_cas_mps, dtype=np.float64),
        upper_cas_mps=np.full(station_count, 130.0, dtype=np.float64),
        threshold_resource_id="RWY",
        resource_stations_m=(("MERGE", 50_000.0),),
    )


def _template(variant: TrajectoryVariant | None = None) -> ClusterTemplate:
    baseline = _straight_variant() if variant is None else variant
    speed_stations = select_action_stations(
        baseline.s_m,
        baseline.east_m,
        baseline.north_m,
        count=16,
        commitment_gate_m=4.0 * M_PER_NM,
        kind="speed",
    )
    stretch_stations = select_action_stations(
        baseline.s_m,
        baseline.east_m,
        baseline.north_m,
        count=8,
        commitment_gate_m=4.0 * M_PER_NM,
        kind="path_stretch",
    )
    return ClusterTemplate(
        cluster_id=baseline.cluster_id,
        medoid_flight_id="MEDOID_1",
        member_count=12,
        baseline_variant=baseline,
        speed_action_stations=speed_stations,
        path_stretch_stations=stretch_stations,
        dataset_id="SYNTHETIC",
        airport_id="TEST",
        runway_id="RWY",
    )


def test_trajectory_variant_arrays_are_contiguous_float64_and_read_only() -> None:
    variant = _straight_variant()
    array_names = (
        "s_m",
        "lat_deg",
        "lon_deg",
        "east_m",
        "north_m",
        "altitude_m",
        "cas_mps",
        "tas_mps",
        "ground_speed_mps",
        "command_cas_mps",
        "reference_command_cas_mps",
        "lower_cas_mps",
        "upper_cas_mps",
        "elapsed_time_s",
    )

    for name in array_names:
        array = getattr(variant, name)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        variant.command_cas_mps[0] = 1.0


def test_constant_speed_variant_has_analytic_duration_and_resource_times() -> None:
    variant = _straight_variant(length_m=100_000.0, speed_mps=100.0)

    assert variant.duration_s == pytest.approx(1_000.0, abs=1e-10)
    assert variant.elapsed_time_s[0] == pytest.approx(1_000.0)
    assert variant.elapsed_time_s[-1] == 0.0
    assert variant.resource("RWY").elapsed_time_s == pytest.approx(1_000.0)
    assert variant.resource("MERGE").elapsed_time_s == pytest.approx(500.0)


def test_template_has_exactly_16_and_8_distinct_entry_ordered_stations() -> None:
    template = _template()

    assert len(template.speed_action_stations) == 16
    assert len(template.path_stretch_stations) == 8
    for stations, expected_kind in (
        (template.speed_action_stations, "speed"),
        (template.path_stretch_stations, "path_stretch"),
    ):
        assert tuple(item.entry_order for item in stations) == tuple(range(len(stations)))
        assert len({item.grid_index for item in stations}) == len(stations)
        assert all(item.kind == expected_kind for item in stations)
        assert all(item.s_m >= 4.0 * M_PER_NM for item in stations)
        assert all(left.s_m > right.s_m for left, right in zip(stations, stations[1:], strict=False))


def test_template_store_preserves_shared_variant_array_identity() -> None:
    template = _template()
    store = TemplateStore((template,))

    loaded = store.template(template.template_id)
    variant = store.variant(template.baseline_variant.variant_id)

    assert loaded is template
    assert variant is template.baseline_variant
    assert variant.s_m is template.baseline_variant.s_m


def test_compiler_builds_analytic_straight_track_with_required_station_counts() -> None:
    length_m = 50.0 * M_PER_NM
    speed_mps = 100.0
    duration_s = length_m / speed_mps
    sample_count = 41
    # A north/south geodesic keeps the AEQD projection exactly one-dimensional.
    latitude = np.linspace(length_m / 111_319.49079327357, 0.0, sample_count)
    track = MedoidTrack(
        flight_id="MEDOID_TRACK",
        time_s=np.linspace(0.0, duration_s, sample_count),
        lat_deg=latitude,
        lon_deg=np.zeros(sample_count),
        altitude_m=np.zeros(sample_count),
    )

    template = TemplateCompiler(station_count=256).compile(
        track,
        cluster_id="CLUSTER_COMPILED",
        member_count=9,
        dataset_id="SYNTHETIC",
        airport_id="TEST",
        runway_id="RWY",
        threshold_resource_id="RWY",
    )

    assert template.baseline_variant.duration_s == pytest.approx(duration_s, abs=1e-5)
    assert template.baseline_variant.diagnostics.absolute_timing_error_s == pytest.approx(0.0, abs=1e-5)
    assert len(template.speed_action_stations) == 16
    assert len(template.path_stretch_stations) == 8
    assert template.baseline_variant.threshold_resource_id == "RWY"
