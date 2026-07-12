from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hailmary.evaluation import (
    DEFAULT_LATERAL_SEPARATION_M,
    DEFAULT_VERTICAL_SEPARATION_M,
    TimedTrajectory,
    detect_conflicts,
    detect_pair_conflicts,
    timed_trajectory_from_variant,
)


def _track(
    flight_id: str,
    time_s,
    east_m,
    north_m,
    altitude_m,
) -> TimedTrajectory:
    return TimedTrajectory(
        flight_id=flight_id,
        time_s=np.asarray(time_s, dtype=float),
        east_m=np.asarray(east_m, dtype=float),
        north_m=np.asarray(north_m, dtype=float),
        altitude_m=np.asarray(altitude_m, dtype=float),
    )


def test_timed_trajectory_copies_contiguous_read_only_arrays() -> None:
    east = np.asarray([0.0, 100.0])
    trajectory = _track("A", [0.0, 10.0], east, [0.0, 0.0], [100.0, 100.0])
    east[0] = 999.0

    assert trajectory.east_m.tolist() == [0.0, 100.0]
    for array in (trajectory.time_s, trajectory.east_m, trajectory.north_m, trajectory.altitude_m):
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        trajectory.east_m[0] = 1.0


@dataclass(frozen=True)
class _CanonicalVariant:
    elapsed_time_s: np.ndarray
    east_m: np.ndarray
    north_m: np.ndarray
    altitude_m: np.ndarray


def test_variant_release_adapter_reverses_threshold_to_upstream_arrays() -> None:
    variant = _CanonicalVariant(
        elapsed_time_s=np.asarray([20.0, 10.0, 0.0]),
        east_m=np.asarray([0.0, 1_000.0, 2_000.0]),
        north_m=np.asarray([0.0, 0.0, 0.0]),
        altitude_m=np.asarray([100.0, 500.0, 900.0]),
    )

    trajectory = timed_trajectory_from_variant(
        variant,
        flight_id="ARRIVAL",
        release_time_s=100.0,
    )

    np.testing.assert_array_equal(trajectory.time_s, [100.0, 110.0, 120.0])
    np.testing.assert_array_equal(trajectory.east_m, [2_000.0, 1_000.0, 0.0])
    np.testing.assert_array_equal(trajectory.altitude_m, [900.0, 500.0, 100.0])


def test_default_is_five_nm_center_separation_not_two_envelope_radii() -> None:
    first = _track("A", [0.0, 60.0], [0.0, 6_000.0], [0.0, 0.0], [0.0, 0.0])
    inside = _track(
        "B",
        [0.0, 60.0],
        [0.0, 6_000.0],
        [4.0 * 1_852.0, 4.0 * 1_852.0],
        [0.0, 0.0],
    )
    outside = _track(
        "C",
        [0.0, 60.0],
        [0.0, 6_000.0],
        [5.01 * 1_852.0, 5.01 * 1_852.0],
        [0.0, 0.0],
    )

    hit = detect_pair_conflicts(first, inside)

    assert DEFAULT_LATERAL_SEPARATION_M == 5.0 * 1_852.0
    assert len(hit) == 1
    assert hit[0].start_time_s == 0.0
    assert hit[0].end_time_s == 60.0
    assert hit[0].minimum_lateral_separation_m == pytest.approx(4.0 * 1_852.0)
    assert detect_pair_conflicts(first, outside) == ()


@pytest.mark.parametrize(
    ("vertical_ft", "expected_count"),
    [(999.0, 1), (1_000.0, 1), (1_000.1, 0)],
)
def test_vertical_band_is_inclusive_at_1000_ft(vertical_ft: float, expected_count: int) -> None:
    first = _track("A", [0.0, 10.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    second = _track(
        "B",
        [0.0, 10.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [vertical_ft * 0.3048, vertical_ft * 0.3048],
    )

    result = detect_pair_conflicts(first, second)

    assert DEFAULT_VERTICAL_SEPARATION_M == 1_000.0 * 0.3048
    assert len(result) == expected_count


def test_continuous_crossing_finds_subsample_entry_and_exit_times() -> None:
    moving = _track("MOVING", [0.0, 20.0], [-1_000.0, 1_000.0], [0.0, 0.0], [0.0, 0.0])
    fixed = _track("FIXED", [0.0, 20.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])

    result = detect_pair_conflicts(
        moving,
        fixed,
        lateral_separation_m=100.0,
    )

    assert len(result) == 1
    assert result[0].start_time_s == pytest.approx(9.0)
    assert result[0].end_time_s == pytest.approx(11.0)
    assert result[0].minimum_lateral_separation_m == pytest.approx(0.0)
    assert result[0].minimum_vertical_separation_m == pytest.approx(0.0)
    assert result[0].pair_ids == ("FIXED", "MOVING")


def test_lateral_and_vertical_intervals_are_intersected_exactly() -> None:
    moving = _track(
        "A",
        [0.0, 20.0],
        [-1_000.0, 1_000.0],
        [0.0, 0.0],
        [2_000.0 * 0.3048, 0.0],
    )
    fixed = _track("B", [0.0, 20.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])

    result = detect_pair_conflicts(
        moving,
        fixed,
        lateral_separation_m=600.0,
        vertical_separation_m=1_000.0 * 0.3048,
    )

    assert len(result) == 1
    assert result[0].start_time_s == pytest.approx(10.0)
    assert result[0].end_time_s == pytest.approx(16.0)
    assert result[0].minimum_lateral_separation_m == pytest.approx(0.0)
    assert result[0].minimum_vertical_separation_m == pytest.approx(400.0 * 0.3048)


def test_tangent_contact_is_reported_as_an_instantaneous_conflict() -> None:
    moving = _track("A", [0.0, 20.0], [-1_000.0, 1_000.0], [100.0, 100.0], [0.0, 0.0])
    fixed = _track("B", [0.0, 20.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])

    result = detect_pair_conflicts(moving, fixed, lateral_separation_m=100.0)

    assert len(result) == 1
    assert result[0].start_time_s == pytest.approx(10.0)
    assert result[0].end_time_s == pytest.approx(10.0)
    assert result[0].minimum_lateral_separation_m == pytest.approx(100.0)


def test_adjacent_segment_hits_merge_but_separated_windows_remain_distinct() -> None:
    first = _track("A", [0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0] * 3)
    always_close = _track(
        "B",
        [0.0, 10.0, 20.0],
        [0.0, 0.0, 0.0],
        [50.0, 50.0, 50.0],
        [0.0] * 3,
    )
    leaves_and_returns = _track(
        "C",
        [0.0, 10.0, 20.0],
        [0.0, 1_000.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0] * 3,
    )

    merged = detect_pair_conflicts(first, always_close, lateral_separation_m=100.0)
    separated = detect_pair_conflicts(first, leaves_and_returns, lateral_separation_m=100.0)

    assert [(item.start_time_s, item.end_time_s) for item in merged] == [(0.0, 20.0)]
    assert [(item.start_time_s, item.end_time_s) for item in separated] == pytest.approx(
        [(0.0, 1.0), (19.0, 20.0)]
    )


def test_nonoverlapping_times_have_no_conflict() -> None:
    first = _track("A", [0.0, 10.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    second = _track("B", [11.0, 20.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])

    assert detect_pair_conflicts(first, second) == ()


def test_multi_flight_results_are_pair_sorted_independent_of_input_order() -> None:
    tracks = [
        _track(flight_id, [0.0, 10.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        for flight_id in ("Z", "A", "M")
    ]

    result = detect_conflicts(reversed(tracks))

    assert [item.pair_ids for item in result] == [("A", "M"), ("A", "Z"), ("M", "Z")]
