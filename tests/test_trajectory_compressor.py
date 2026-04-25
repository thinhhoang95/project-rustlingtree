from __future__ import annotations

import numpy as np
import pandas as pd

from scenario.trajectory_compressor.algorithms import (
    compress_breakpoints,
    douglas_peucker_series_indices,
    douglas_peucker_spatial_indices,
)
from scenario.trajectory_compressor.io import filter_tracks_to_flights, load_arrival_departure_flight_ids


def test_spatial_douglas_peucker_keeps_turning_point() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 0.0])

    indices = douglas_peucker_spatial_indices(x, y, epsilon_m=1.0)

    assert indices.tolist() == [0, 1, 2]


def test_series_douglas_peucker_uses_vertical_interpolation_error() -> None:
    times = np.array([0.0, 10.0, 20.0])
    altitudes = np.array([0.0, 100.0, 0.0])

    indices = douglas_peucker_series_indices(times, altitudes, epsilon_y=10.0)

    assert indices.tolist() == [0, 1, 2]


def test_compress_breakpoints_unions_lateral_and_altitude_timestamps() -> None:
    times = np.array([0, 10, 20, 30], dtype=float)
    latitudes = np.array([32.0, 32.0, 32.01, 32.02])
    longitudes = np.array([-97.0, -96.99, -96.99, -96.99])
    geoaltitudes = np.array([0.0, 0.0, 500.0, 0.0])

    result = compress_breakpoints(
        times=times,
        latitudes=latitudes,
        longitudes=longitudes,
        geoaltitudes=geoaltitudes,
        lateral_tolerance_m=10.0,
        altitude_tolerance_m=10.0,
    )

    assert set(result.lateral_indices.tolist()).issubset(set(result.minimal_indices.tolist()))
    assert set(result.altitude_indices.tolist()).issubset(set(result.minimal_indices.tolist()))
    assert result.minimal_indices.tolist() == sorted(result.minimal_indices.tolist())


def test_load_arrival_departure_flight_ids_filters_catalog_operations(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "flight_id,operation\n"
        "ARR1,arrival\n"
        "DEP1,departure\n"
        "OVR1,overflight\n",
        encoding="utf-8",
    )

    flight_ids = load_arrival_departure_flight_ids(catalog_path)

    assert flight_ids == {"ARR1", "DEP1"}


def test_filter_tracks_to_flights_keeps_only_catalog_flights() -> None:
    tracks = pd.DataFrame(
        {
            "flight_id": ["ARR1", "DEP1", "OVR1"],
            "time": [1, 2, 3],
        }
    )

    filtered = filter_tracks_to_flights(tracks, {"ARR1", "DEP1"})

    assert filtered["flight_id"].tolist() == ["ARR1", "DEP1"]
