from __future__ import annotations

import numpy as np
import pandas as pd

from vlm_ppe.clustering.residual_windows import (
    compute_cluster_residual_windows,
    compute_heading_dispersion,
    compute_residual_energy,
    detect_window_spans,
    robust_threshold,
)
from vlm_ppe.schemas import ClusterMedoid


def test_residual_energy_uses_median_squared_distance() -> None:
    template = np.column_stack([np.arange(3, dtype=float), np.zeros(3)])
    tracks = np.stack(
        [
            template,
            template + np.asarray([[0.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
            template + np.asarray([[0.0, 0.0], [0.0, 4.0], [0.0, 0.0]]),
        ],
        axis=0,
    )

    energy = compute_residual_energy(tracks, template)

    np.testing.assert_allclose(energy, [0.0, 4.0, 0.0])


def test_heading_dispersion_is_circular() -> None:
    east = np.column_stack([np.arange(4, dtype=float), np.zeros(4)])
    north = np.column_stack([np.zeros(4), np.arange(4, dtype=float)])

    dispersion = compute_heading_dispersion(np.stack([east, north], axis=0))

    np.testing.assert_allclose(dispersion, np.full(4, 1.0 - np.sqrt(0.5)), atol=1e-12)


def test_detect_window_spans_merges_nearby_runs_and_filters_short_windows() -> None:
    energy = np.asarray([0.0, 10.0, 10.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    heading = np.zeros_like(energy)
    s_nm = np.arange(len(energy), dtype=float)

    spans = detect_window_spans(
        energy,
        heading,
        residual_energy_lambda=3.0,
        heading_dispersion_threshold=0.25,
        station_s_nm=s_nm,
        min_window_length_nm=3.0,
        merge_windows_gap_nm=2.0,
    )

    assert spans == [(1, 5, ["residual_energy"])]
    assert robust_threshold(energy, 3.0) == 0.0


def test_compute_cluster_residual_windows_detects_synthetic_excursion() -> None:
    station_x = np.linspace(0.0, 10.0, 11)
    template = np.column_stack([station_x, np.zeros_like(station_x)])
    excursion = template.copy()
    excursion[3:7, 1] = 3.0
    rows: list[dict] = []
    for flight_id, points in {"T0": template, "T1": excursion, "T2": excursion}.items():
        for station_index, (x_nm, y_nm) in enumerate(points):
            rows.append(
                {
                    "flight_id": flight_id,
                    "station_index": station_index,
                    "x_nm": float(x_nm),
                    "y_nm": float(y_nm),
                }
            )
    resampled = pd.DataFrame(rows)
    labels = pd.DataFrame({"flight_id": ["T0", "T1", "T2"], "cluster_id": [0, 0, 0]})
    medoid = ClusterMedoid(
        cluster_id=0,
        medoid_track_id="T0",
        n_tracks=3,
        mean_distance_nm=0.0,
        max_distance_nm=0.0,
        template_points=[(float(x), float(y)) for x, y in template],
    )

    profiles, windows = compute_cluster_residual_windows(
        resampled,
        labels,
        [medoid],
        residual_energy_lambda=3.0,
        min_window_length_nm=2.0,
        merge_windows_gap_nm=1.0,
        heading_dispersion_threshold=0.25,
    )

    assert len(windows) == 1
    window = windows[0]
    assert window.window_id == "C0_W1"
    assert window.start_station_index == 3
    assert window.end_station_index == 6
    assert window.trigger_reasons == ["residual_energy"]
    assert float(profiles["residual_energy_nm2"].max()) == 9.0


def test_compute_cluster_residual_windows_returns_empty_when_no_candidate_window() -> None:
    station_x = np.linspace(0.0, 10.0, 11)
    template = np.column_stack([station_x, np.zeros_like(station_x)])
    rows: list[dict] = []
    for flight_id in ["T0", "T1", "T2"]:
        for station_index, (x_nm, y_nm) in enumerate(template):
            rows.append({"flight_id": flight_id, "station_index": station_index, "x_nm": float(x_nm), "y_nm": float(y_nm)})
    resampled = pd.DataFrame(rows)
    labels = pd.DataFrame({"flight_id": ["T0", "T1", "T2"], "cluster_id": [0, 0, 0]})
    medoid = ClusterMedoid(
        cluster_id=0,
        medoid_track_id="T0",
        n_tracks=3,
        mean_distance_nm=0.0,
        max_distance_nm=0.0,
        template_points=[(float(x), float(y)) for x, y in template],
    )

    _profiles, windows = compute_cluster_residual_windows(
        resampled,
        labels,
        [medoid],
        residual_energy_lambda=3.0,
        min_window_length_nm=2.0,
        merge_windows_gap_nm=1.0,
        heading_dispersion_threshold=0.25,
    )

    assert windows == []
