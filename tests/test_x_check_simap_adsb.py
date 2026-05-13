from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.x_check_simap_adsb.envelope_viz import _wait_atc_point_coordinates
from scripts.x_check_simap_adsb import (
    derive_speed_mps,
    parse_flight_key,
    sample_trajectory,
    trajectory_from_bichannel_result,
)


def test_parse_flight_key_extracts_segment_and_icao24() -> None:
    key = parse_flight_key("JIA5128M2,a7cb67")

    assert key.callsign_segment == "JIA5128M2"
    assert key.base_callsign == "JIA5128"
    assert key.segment_number == 2
    assert key.icao24 == "a7cb67"
    assert key.flight_id == "JIA5128M2a7cb67"


def test_parse_flight_key_accepts_multi_digit_segment() -> None:
    key = parse_flight_key("ABC123M10,ABCDEF")

    assert key.callsign_segment == "ABC123M10"
    assert key.base_callsign == "ABC123"
    assert key.segment_number == 10
    assert key.icao24 == "abcdef"


def test_sample_trajectory_interpolates_in_range_and_marks_out_of_range() -> None:
    trajectory = trajectory_from_bichannel_result(
        SimpleNamespace(
            t_s=np.asarray([0.0, 10.0], dtype=float),
            lat_deg=np.asarray([32.0, 32.1], dtype=float),
            lon_deg=np.asarray([-97.0, -97.2], dtype=float),
            h_m=np.asarray([1000.0, 800.0], dtype=float),
            v_cas_mps=np.asarray([130.0, 120.0], dtype=float),
        ),
        SimpleNamespace(time_s=100),
    )

    sample = sample_trajectory(trajectory, 105.0)
    before = sample_trajectory(trajectory, 99.0)

    assert sample.available is True
    assert sample.lat_deg == 32.05
    assert sample.lon_deg == -97.1
    assert sample.altitude_m == 900.0
    assert before.available is False
    assert before.altitude_m is None


def test_sample_trajectory_returns_none_for_unavailable_value() -> None:
    trajectory = trajectory_from_bichannel_result(
        SimpleNamespace(
            t_s=np.asarray([0.0, 0.0], dtype=float),
            lat_deg=np.asarray([32.0, 32.1], dtype=float),
            lon_deg=np.asarray([-97.0, -97.2], dtype=float),
            h_m=np.asarray([1000.0, 800.0], dtype=float),
            v_cas_mps=np.asarray([np.nan, np.nan], dtype=float),
        ),
        SimpleNamespace(time_s=100),
    )

    sample = sample_trajectory(trajectory, 100.0)

    assert sample.available is True
    assert sample.speed_mps is None


def test_derive_speed_mps_uses_neighboring_positions() -> None:
    times = np.asarray([0.0, 10.0, 20.0])
    lats = np.asarray([0.0, 0.0, 0.0])
    lons = np.asarray([0.0, 0.01, 0.03])

    speeds = derive_speed_mps(times, lats, lons)

    assert np.isfinite(speeds).all()
    assert speeds[1] == speeds[2]
    assert speeds[1] > speeds[0]


def test_wait_atc_point_coordinates_accepts_valid_payload_and_rejects_invalid_payload() -> None:
    point = {"lat": 32.123456, "lon": -97.654321}

    assert _wait_atc_point_coordinates(point) == (-97.654321, 32.123456)
    assert _wait_atc_point_coordinates(None) is None
    assert _wait_atc_point_coordinates({"lat": "nan", "lon": -97.0}) is None
