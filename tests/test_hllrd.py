from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hllrd.candidates import is_duplicate_interval
from hllrd.cli import candidates_main, fit_main, transform_main
from hllrd.data import filter_tracks_to_cluster, load_cluster_flights
from hllrd.fit import HLLRDV1Config, fit_localized_low_rank, load_fit_result
from hllrd.matrix import MatrixArtifact, MatrixBuildConfig, build_matrix_from_tracks, load_matrix_artifact, save_matrix_artifact


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
        HLLRDV1Config(kappa_peak=0.5, K_max=5, n_min=5, epsilon_gain=0.0),
    )

    assert result.events
    assert result.explained_fraction > 0.9
    assert any(max(event.start, 35) < min(event.end, 49) for event in result.events)


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


def test_duplicate_interval_rule_allows_nested_but_rejects_near_identical() -> None:
    assert is_duplicate_interval((10, 30), (11, 31))
    assert not is_duplicate_interval((15, 20), (10, 30))


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
