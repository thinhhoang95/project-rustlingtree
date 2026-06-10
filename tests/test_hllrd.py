from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd

from hllrd.candidates import backtrack_peak_rise_start, is_duplicate_interval
from hllrd.cli import candidates_main, fit_main, transform_main
from hllrd.data import filter_tracks_to_cluster, load_cluster_flights, trim_tracks_from_anchor
from hllrd.fit import (
    HLLRDV1Config,
    activation_threshold_for_length,
    fit_localized_low_rank,
    load_fit_result,
    quiet_window_energy_floor,
    save_fit_result,
)
from hllrd.matrix import (
    MatrixArtifact,
    MatrixBuildConfig,
    _normal_residuals_at_reference_stations,
    build_matrix_from_tracks,
    load_matrix_artifact,
    save_matrix_artifact,
)
from hllrd.geometry import resample_polyline_constant_speed
from hllrd.simplifier import simplify_local_deviation_block, simplify_series_by_gain


def test_load_cluster_flights_filters_artifact_jsonl(tmp_path) -> None:
    path = tmp_path / "arrivals.jsonl"
    rows = [
        {"flight_id": "ARR_NE", "callsign": "A", "icao24": "aaa", "runway": "RW17C", "wait_atc_point": {"arrival_cluster": "NE"}},
        {"flight_id": "ARR_SW", "callsign": "B", "icao24": "bbb", "runway": "RW35C", "wait_atc_point": {"arrival_cluster": "SW"}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    flights = load_cluster_flights(path, "ne")

    assert [flight.flight_id for flight in flights] == ["ARR_NE"]
    assert flights[0].cluster == "NE"


def test_filter_tracks_to_cluster_keeps_only_selected_flights() -> None:
    flights = load_cluster_flights_from_rows(
        [
            {"flight_id": "ARR1", "wait_atc_point": {"arrival_cluster": "SE"}},
        ],
        "SE",
    )
    tracks = pd.DataFrame({"flight_id": ["ARR1", "ARR2"], "time": [1, 1]})

    filtered = filter_tracks_to_cluster(tracks, flights)

    assert filtered["flight_id"].tolist() == ["ARR1"]


def test_trim_tracks_from_anchor_refines_anchor_and_starts_at_closest_point() -> None:
    tracks = pd.DataFrame(
        [
            {"flight_id": "A", "time": 0, "lat": 31.9, "lon": -96.0},
            {"flight_id": "A", "time": 1, "lat": 32.245, "lon": -96.245},
            {"flight_id": "A", "time": 2, "lat": 32.6, "lon": -96.8},
            {"flight_id": "A", "time": 3, "lat": 32.9, "lon": -97.0},
            {"flight_id": "B", "time": 0, "lat": 31.8, "lon": -95.8},
            {"flight_id": "B", "time": 1, "lat": 32.255, "lon": -96.255},
            {"flight_id": "B", "time": 2, "lat": 32.7, "lon": -96.9},
            {"flight_id": "B", "time": 3, "lat": 32.9, "lon": -97.0},
        ]
    )

    result = trim_tracks_from_anchor(
        tracks,
        anchor_lat_deg=32.25,
        anchor_lon_deg=-96.25,
        max_anchor_distance_nm=5.0,
        min_points_after_anchor=3,
        min_refinement_flights=2,
    )

    assert result.dropped_flights == ()
    assert result.tracks.groupby("flight_id")["time"].min().to_dict() == {"A": 1, "B": 1}
    assert abs(result.refined_anchor_lat_deg - 32.25) < 0.01
    assert abs(result.refined_anchor_lon_deg + 96.25) < 0.01
    assert set(result.metadata["status"]) == {"kept"}


def test_build_matrix_from_tracks_returns_centered_normal_residuals() -> None:
    tracks = _straight_track_frame(offsets=[-100.0, 0.0, 100.0])

    artifact = build_matrix_from_tracks(
        tracks,
        config=MatrixBuildConfig(station_count=20, min_points_per_flight=3),
        cluster="SE",
    )

    assert artifact.X.shape == (3, 20)
    assert artifact.X_centered.shape == (3, 20)
    assert artifact.flight_ids == ("FLT0", "FLT1", "FLT2")
    assert np.all(np.isfinite(artifact.X_centered))
    assert np.allclose(np.linalg.norm(artifact.normals_xy, axis=1), 1.0)
    assert artifact.cluster == "SE"


def test_resample_polyline_constant_speed_uses_uniform_arc_length() -> None:
    polyline = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 3.0],
        ]
    )

    sampled = resample_polyline_constant_speed(polyline, 5)
    distances = np.hypot(np.diff(sampled[:, 0]), np.diff(sampled[:, 1]))

    np.testing.assert_allclose(distances, np.ones(4), atol=1.0e-12)


def test_reference_station_residual_ignores_alongtrack_polyline_timing() -> None:
    reference = np.column_stack((np.zeros(6), np.linspace(0.0, 1_000.0, 6)))
    normals = np.tile(np.asarray([-1.0, 0.0]), (6, 1))
    polyline = np.asarray(
        [
            [100.0, 0.0],
            [100.0, 80.0],
            [160.0, 120.0],
            [100.0, 200.0],
            [100.0, 650.0],
            [100.0, 1_000.0],
        ]
    )

    residual = _normal_residuals_at_reference_stations(polyline, reference, normals)

    np.testing.assert_allclose(residual, -100.0, atol=1.0e-9)


def test_fit_localized_low_rank_recovers_planted_event() -> None:
    rng = np.random.default_rng(7)
    n = 50
    M = 90
    X = rng.normal(0.0, 0.05, size=(n, M))
    local = np.column_stack(
        [
            np.sin(np.linspace(0.0, np.pi, 14)),
            np.cos(np.linspace(0.0, np.pi, 14)),
        ]
    )
    local_basis, _ = np.linalg.qr(local)
    basis = np.zeros((M, 2))
    basis[35:49, :] = local_basis
    coefficients = rng.normal(0.0, 5.0, size=(n, 2))
    X += coefficients @ basis.T

    result = fit_localized_low_rank(
        X,
        HLLRDV1Config(kappa_peak=0.5, K_max=5, n_min=5, epsilon_gain=0.0, local_simplifier_enabled=True),
    )

    assert result.events
    assert result.explained_fraction > 0.9
    assert any(max(event.start, 35) < min(event.end, 49) for event in result.events)
    assert result.events[0].simplifier["enabled"]


def test_local_simplifier_reduces_dogleg_to_one_approximation_point() -> None:
    dogleg = np.asarray([0.0, 5.0, 10.0, 5.0, 0.0])

    result = simplify_series_by_gain(
        dogleg,
        min_gain_per_point_m2=1.0,
        max_approximation_points=4,
    )

    assert result.approximation_points == 1
    assert result.retained_indices.tolist() == [0, 2, 4]
    np.testing.assert_allclose(result.values, dogleg)


def test_local_simplifier_keeps_two_points_for_trombone_like_pattern() -> None:
    trombone = np.asarray([0.0, 1.0, 4.0, 8.0, 9.0, 5.0, 1.0, 0.0])

    result = simplify_series_by_gain(
        trombone,
        min_gain_per_point_m2=2.0,
        max_approximation_points=4,
    )

    assert result.approximation_points == 2
    assert result.retained_indices[0] == 0
    assert result.retained_indices[-1] == len(trombone) - 1
    assert result.residual_error_m2 < result.initial_error_m2


def test_local_simplifier_only_simplifies_active_local_rows() -> None:
    block = np.asarray(
        [
            [0.0, 4.0, 8.0, 4.0, 0.0],
            [100.0, 50.0, 0.0, 50.0, 100.0],
        ]
    )

    result = simplify_local_deviation_block(
        block,
        active_mask=np.asarray([True, False]),
        min_gain_per_point_m2=1.0,
        max_approximation_points=4,
    )

    np.testing.assert_allclose(result.values[0], block[0])
    np.testing.assert_allclose(result.values[1], np.zeros(5))
    assert result.diagnostics["point_count_histogram"] == {"0": 0, "1": 1, "2": 0, "3": 0, "4": 0}


def test_fit_local_simplifier_can_be_disabled(tmp_path) -> None:
    X = np.zeros((12, 50), dtype=float)
    shape = np.asarray([0.0, 2.0, 6.0, 10.0, 6.0, 2.0, 0.0])
    for row in range(X.shape[0]):
        X[row, 20:27] = (row + 1) * shape

    enabled = fit_localized_low_rank(
        X,
        HLLRDV1Config(
            L_min=10,
            L_max=10,
            K_max=1,
            n_min=2,
            kappa_peak=0.0,
            c_null=0.0,
            epsilon_gain=0.0,
            local_simplifier_enabled=True,
            local_simplifier_gain_sigma=0.0,
            local_simplifier_max_relative_loss=1.0,
        ),
        already_centered=True,
    )
    disabled = fit_localized_low_rank(
        X,
        HLLRDV1Config(
            L_min=10,
            L_max=10,
            K_max=1,
            n_min=2,
            kappa_peak=0.0,
            c_null=0.0,
            epsilon_gain=0.0,
            local_simplifier_enabled=False,
        ),
        already_centered=True,
    )

    assert enabled.events
    assert enabled.events[0].simplifier["enabled"]
    assert disabled.events
    assert disabled.events[0].simplifier == {}

    model_path = tmp_path / "simplified_model.npz"
    save_fit_result(model_path, enabled)
    loaded = load_fit_result(model_path)
    assert loaded.config.local_simplifier_enabled
    assert loaded.events[0].simplifier["enabled"]
    assert loaded.events[0].simplifier["point_count_histogram"]


def test_fit_rejects_single_flight_outlier_when_n_min_is_high() -> None:
    X = np.zeros((20, 50), dtype=float)
    X[0, 20:25] = 100.0

    result = fit_localized_low_rank(
        X,
        HLLRDV1Config(kappa_peak=0.0, K_max=3, n_min=5, c_null=0.0),
        already_centered=True,
    )

    assert result.events == ()
    assert result.explained_fraction == 0.0


def test_empirical_null_threshold_exceeds_analytic_threshold_for_smooth_background() -> None:
    rng = np.random.default_rng(42)
    X = np.cumsum(rng.normal(size=(40, 60)), axis=1)
    for _ in range(4):
        X[:, 1:-1] = (X[:, :-2] + X[:, 1:-1] + X[:, 2:]) / 3.0
    X *= 100.0 / np.std(X)
    config = HLLRDV1Config(
        L_min=10,
        L_max=10,
        K_max=1,
        kappa_peak=0.0,
        n_min=5,
        local_simplifier_enabled=False,
    )

    analytic = fit_localized_low_rank(
        X,
        replace(config, empirical_null_repeats=0),
        already_centered=True,
    )
    empirical = fit_localized_low_rank(
        X,
        replace(config, empirical_null_repeats=12, empirical_null_quantile=0.95),
        already_centered=True,
    )

    analytic_threshold = analytic.metadata["null_thresholds_by_length"]["10"]
    empirical_threshold = empirical.metadata["null_thresholds_by_length"]["10"]
    assert empirical_threshold > analytic_threshold


def test_activation_threshold_uses_quiet_window_energy_floor() -> None:
    X = np.asarray(
        [
            [10.0, 10.0, 2.0, 2.0, 8.0],
            [-10.0, -10.0, -2.0, -2.0, -8.0],
        ]
    )

    floor = quiet_window_energy_floor(X, min_length=2)
    threshold = activation_threshold_for_length(8, floor, activation_scale=1.5)

    assert floor == 4.0
    assert threshold == 1.5 * np.sqrt(8 * 4.0)


def test_duplicate_interval_rule_allows_nested_but_rejects_near_identical() -> None:
    assert is_duplicate_interval((10, 30), (11, 31))
    assert not is_duplicate_interval((15, 20), (10, 30))


def test_peak_backtrack_finds_rising_shoulder() -> None:
    energy = np.asarray([1.0, 1.0, 1.2, 2.0, 5.0, 12.0, 20.0, 18.0, 4.0])

    start = backtrack_peak_rise_start(energy, 6, baseline=1.0, rise_fraction=0.10)

    assert start == 4


def test_fit_backtracks_late_peak_window_to_cover_rise() -> None:
    n = 24
    M = 80
    shape = np.zeros(M, dtype=float)
    shape[45:71] = np.linspace(0.0, 1.0, 26)
    shape[71:] = np.linspace(0.9, 0.1, 9)
    amplitudes = np.linspace(-80.0, 80.0, n)
    X = amplitudes[:, None] * shape[None, :]

    config = HLLRDV1Config(
        L_min=20,
        L_max=20,
        K_max=1,
        kappa_peak=0.0,
        n_min=5,
        c_null=0.0,
        epsilon_gain=0.0,
        peak_backtrack_rise_fraction=0.05,
        local_simplifier_enabled=False,
    )
    result = fit_localized_low_rank(
        X,
        config,
        already_centered=True,
    )
    centered = fit_localized_low_rank(
        X,
        replace(config, peak_backtrack_enabled=False),
        already_centered=True,
    )

    assert result.events
    assert centered.events
    event = result.events[0]
    centered_event = centered.events[0]
    assert event.peak_index >= 65
    assert event.start < centered_event.start
    assert event.start <= centered_event.start - 5
    assert event.end == centered_event.end


def test_matrix_artifact_round_trips(tmp_path) -> None:
    artifact = build_matrix_from_tracks(
        _straight_track_frame(offsets=[0.0, 50.0, 100.0]),
        config=MatrixBuildConfig(station_count=10),
        cluster="NW",
    )
    path = tmp_path / "matrix.npz"

    save_matrix_artifact(path, artifact)
    loaded = load_matrix_artifact(path)

    assert loaded.cluster == "NW"
    assert loaded.flight_ids == artifact.flight_ids
    np.testing.assert_allclose(loaded.X_centered, artifact.X_centered)


def test_fit_and_transform_clis_smoke(tmp_path) -> None:
    artifact = build_matrix_from_tracks(
        _straight_track_frame(offsets=[-250.0, -100.0, 100.0, 250.0, 400.0, -400.0]),
        config=MatrixBuildConfig(station_count=30),
        cluster="NE",
    )
    matrix_path = tmp_path / "matrix.npz"
    model_path = tmp_path / "model.npz"
    candidates_path = tmp_path / "candidates.csv"
    transform_path = tmp_path / "transform.npz"
    save_matrix_artifact(matrix_path, artifact)

    candidates_main(["--matrix", str(matrix_path), "--output", str(candidates_path), "--c-null", "0"])
    fit_main(["--matrix", str(matrix_path), "--output", str(model_path), "--c-null", "0", "--K-max", "2", "--n-min", "2"])
    transform_main(["--matrix", str(matrix_path), "--model", str(model_path), "--output", str(transform_path)])

    assert candidates_path.exists()
    assert model_path.exists()
    assert transform_path.exists()
    assert load_fit_result(model_path).dictionary.shape[0] == artifact.X.shape[1]


def load_cluster_flights_from_rows(rows: list[dict[str, object]], cluster: str):
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "arrivals.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return load_cluster_flights(path, cluster)


def _straight_track_frame(offsets: list[float]) -> pd.DataFrame:
    rows = []
    lat0 = 32.8
    lon0 = -97.2
    meters_per_deg_lat = 111_000.0
    meters_per_deg_lon = 111_000.0 * np.cos(np.radians(lat0))
    for flight_index, offset_m in enumerate(offsets):
        for point_index in range(6):
            north_m = point_index * 1_000.0
            east_m = offset_m
            rows.append(
                {
                    "flight_id": f"FLT{flight_index}",
                    "time": point_index,
                    "lat": lat0 + north_m / meters_per_deg_lat,
                    "lon": lon0 + east_m / meters_per_deg_lon,
                    "geoaltitude": 2_000.0 - point_index * 100.0,
                }
            )
    return pd.DataFrame(rows)
