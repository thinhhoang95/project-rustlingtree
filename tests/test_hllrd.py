from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd

from hllrd.candidates import backtrack_peak_rise_start, is_duplicate_interval
from hllrd.cli import candidates_main, evaluate_event_main, fit_main, transform_main
from hllrd.data import filter_tracks_to_cluster, load_cluster_flights, trim_tracks_from_anchor
from hllrd.evaluate import (
    evaluate_event_trace_match,
    _raw_polyline_window,
    _tangents_from_normals,
    _trace_lift_tangent_reconstruction,
)
from hllrd.fit import (
    HLLRDEvent,
    HLLRDFitResult,
    HLLRDV1Config,
    HLLRDV2Config,
    activation_threshold_for_length,
    augment_with_trace_tangent_lift,
    fit_lag_registered_low_rank,
    fit_localized_low_rank,
    load_fit_result,
    quiet_window_energy_floor,
    _row_event_basis,
    save_fit_result,
    transform_with_model,
)
from hllrd.geometry import LocalProjection
from hllrd.matrix import (
    MatrixArtifact,
    MatrixBuildConfig,
    _normal_residuals_at_reference_stations,
    build_matrix_from_tracks,
    load_matrix_artifact,
    save_matrix_artifact,
)
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
        HLLRDV1Config(kappa_peak=0.5, K_max=5, n_min=5, epsilon_gain=0.0),
    )

    assert result.events
    assert result.explained_fraction > 0.9
    assert any(max(event.start, 35) < min(event.end, 49) for event in result.events)
    assert result.events[0].simplifier["enabled"]


def test_lag_registered_v2_matches_v1_when_max_lag_is_zero() -> None:
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
    common = dict(
        kappa_peak=0.5,
        K_max=3,
        n_min=5,
        epsilon_gain=0.0,
        local_simplifier_enabled=False,
    )

    v1 = fit_localized_low_rank(X, HLLRDV1Config(**common))
    v2 = fit_lag_registered_low_rank(
        X,
        HLLRDV2Config(**common, max_lag_stations=0, lag_enabled=True),
    )

    assert [(event.start, event.end, event.peak_index) for event in v2.events] == [
        (event.start, event.end, event.peak_index) for event in v1.events
    ]
    np.testing.assert_allclose(v2.reconstruction, v1.reconstruction)
    assert all(event.lag_offsets is None for event in v2.events)


def test_lag_registered_v2_recovers_randomly_delayed_trombone() -> None:
    X, injected_delays = _lagged_trombone_matrix()
    L = 12
    common = dict(
        L_min=L,
        L_max=L,
        K_max=1,
        kappa_peak=0.0,
        n_min=5,
        c_null=0.0,
        epsilon_gain=0.0,
        activation_scale=0.0,
        peak_backtrack_enabled=False,
        local_simplifier_enabled=False,
    )

    v1 = fit_localized_low_rank(X, HLLRDV1Config(**common), already_centered=True)
    v2 = fit_lag_registered_low_rank(
        X,
        HLLRDV2Config(**common, max_lag_stations=8, lag_direction="both"),
        already_centered=True,
    )

    assert v2.events
    event = v2.events[0]
    assert event.lag_offsets is not None
    offset = int(round(float(np.median(event.lag_offsets - injected_delays))))
    lag_mae = float(np.mean(np.abs((event.lag_offsets - offset) - injected_delays)))
    assert lag_mae <= 1.0
    assert v2.explained_fraction > 0.98
    assert v2.explained_fraction > v1.explained_fraction + 0.10
    assert len(set(event.lag_offsets.tolist())) > 3


def test_extension_registered_v2_recovers_delayed_action_trombone() -> None:
    X, _injected_extensions = _extended_trombone_matrix()
    L = 12
    common = dict(
        L_min=L,
        L_max=L,
        K_max=1,
        kappa_peak=0.0,
        n_min=5,
        c_null=0.0,
        epsilon_gain=0.0,
        activation_scale=0.0,
        peak_backtrack_enabled=False,
        local_simplifier_enabled=False,
    )

    v1 = fit_localized_low_rank(X, HLLRDV1Config(**common), already_centered=True)
    v2 = fit_lag_registered_low_rank(
        X,
        HLLRDV2Config(
            **common,
            max_lag_stations=0,
            max_extend_stations=8,
            extend_direction="nonnegative",
        ),
        already_centered=True,
    )

    assert v2.events
    event = v2.events[0]
    assert event.extension_offsets is not None
    assert int(np.max(event.extension_offsets)) >= 5
    assert len(set(event.extension_offsets.tolist())) > 3
    assert v2.explained_fraction > 0.99
    assert v2.explained_fraction > v1.explained_fraction + 0.05


def test_lag_registered_v2_round_trips_and_transforms(tmp_path) -> None:
    X, _injected_delays = _lagged_trombone_matrix()
    result = fit_lag_registered_low_rank(
        X,
        HLLRDV2Config(
            L_min=12,
            L_max=12,
            K_max=1,
            kappa_peak=0.0,
            n_min=5,
            c_null=0.0,
            epsilon_gain=0.0,
            activation_scale=0.0,
            peak_backtrack_enabled=False,
            local_simplifier_enabled=False,
            max_lag_stations=8,
        ),
        already_centered=True,
    )
    model_path = tmp_path / "v2_model.npz"

    save_fit_result(model_path, result)
    loaded = load_fit_result(model_path)
    transformed = transform_with_model(X, loaded, already_centered=True)

    assert isinstance(loaded.config, HLLRDV2Config)
    assert loaded.events[0].lag_offsets is not None
    np.testing.assert_array_equal(loaded.events[0].lag_offsets, result.events[0].lag_offsets)
    assert transformed.lag_offsets is not None
    assert transformed.explained_fraction > 0.98


def test_extension_registered_v2_round_trips_and_transforms(tmp_path) -> None:
    X, _injected_extensions = _extended_trombone_matrix()
    result = fit_lag_registered_low_rank(
        X,
        HLLRDV2Config(
            L_min=12,
            L_max=12,
            K_max=1,
            kappa_peak=0.0,
            n_min=5,
            c_null=0.0,
            epsilon_gain=0.0,
            activation_scale=0.0,
            peak_backtrack_enabled=False,
            local_simplifier_enabled=False,
            max_lag_stations=0,
            max_extend_stations=8,
            extend_direction="nonnegative",
        ),
        already_centered=True,
    )
    model_path = tmp_path / "v2_extension_model.npz"

    save_fit_result(model_path, result)
    loaded = load_fit_result(model_path)
    transformed = transform_with_model(X, loaded, already_centered=True)

    assert isinstance(loaded.config, HLLRDV2Config)
    assert loaded.events[0].extension_offsets is not None
    np.testing.assert_array_equal(loaded.events[0].extension_offsets, result.events[0].extension_offsets)
    assert transformed.extension_offsets is not None
    assert transformed.explained_fraction > 0.99


def test_event_trace_evaluator_matches_exact_shifted_event(tmp_path) -> None:
    matrix, model = _exact_shifted_event_artifacts()

    result = evaluate_event_trace_match(matrix, model, event_index=0)
    raw_result = evaluate_event_trace_match(
        matrix,
        model,
        event_index=0,
        raw_tracks=_raw_tracks_from_matrix_paths(matrix),
    )

    assert result.summary["active_flight_count"] == 2
    assert result.summary["event_trace_rmse_m"] == 0.0
    assert result.summary["model_trace_rmse_m"] == 0.0
    assert raw_result.summary["event_trace_rmse_m"] < 1.0e-9
    assert raw_result.summary["model_trace_rmse_m"] < 1.0e-9
    assert result.summary["center_trace_rmse_m"] > 0.0
    assert result.summary["event_trace_rmse_reduction_fraction"] == 1.0
    assert result.summary["event_improves_center_fraction"] == 1.0
    assert result.summary["center_normal_rmse_m"] > 0.0
    assert result.summary["event_normal_rmse_m"] == 0.0
    assert result.summary["event_normal_rmse_reduction_fraction"] == 1.0

    matrix_path = tmp_path / "matrix.npz"
    model_path = tmp_path / "model.npz"
    rows_path = tmp_path / "event_rows.csv"
    summary_path = tmp_path / "event_summary.json"
    plot_path = tmp_path / "event_overlay.png"
    save_matrix_artifact(matrix_path, matrix)
    save_fit_result(model_path, model, flight_ids=matrix.flight_ids)
    evaluate_event_main(
        [
            "--matrix",
            str(matrix_path),
            "--model",
            str(model_path),
            "--event",
            "0",
            "--trace-source",
            "matrix",
            "--output",
            str(rows_path),
            "--summary",
            str(summary_path),
            "--plot",
            str(plot_path),
        ]
    )

    assert rows_path.exists()
    assert plot_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["event_trace_rmse_m"] == 0.0


def test_event_trace_evaluator_matches_exact_extended_event() -> None:
    matrix, model = _exact_extended_event_artifacts()

    result = evaluate_event_trace_match(matrix, model, event_index=0)
    raw_result = evaluate_event_trace_match(
        matrix,
        model,
        event_index=0,
        raw_tracks=_raw_tracks_from_matrix_paths(matrix),
    )

    assert result.summary["active_flight_count"] == 2
    assert result.summary["event_trace_rmse_m"] == 0.0
    assert result.summary["model_trace_rmse_m"] == 0.0
    assert raw_result.summary["event_trace_rmse_m"] < 1.0e-9
    assert raw_result.summary["model_trace_rmse_m"] < 1.0e-9
    assert {row.extension_offset for row in result.rows} == {0, 2}


def test_raw_polyline_window_keeps_contiguous_segment_between_matching_vertices() -> None:
    polyline = np.column_stack((np.arange(8, dtype=float), np.zeros(8, dtype=float)))
    stations = np.column_stack((np.arange(8, dtype=float), np.zeros(8, dtype=float)))
    stations[4] = [100.0, 0.0]

    window = _raw_polyline_window(polyline, stations, station_start=4, station_end=5)

    assert 4.0 in window[:, 0]
    np.testing.assert_allclose(np.diff(window[:, 0]), 1.0)


def test_trace_lift_recovers_tangential_residual_with_registered_dictionary() -> None:
    matrix, model = _exact_extended_event_artifacts()
    event = replace(
        model.events[0],
        extension_offsets=np.asarray([2, 2], dtype=int),
        simplifier={"extension_registered": True},
    )
    model = replace(model, events=(event,))
    center_xy = matrix.reference_xy_m.copy()
    tangents = _tangents_from_normals(matrix.normals_xy)
    tangent_residual = np.zeros_like(matrix.X)
    for row_index, extension in enumerate(event.extension_offsets):
        row_basis = _row_event_basis(event, matrix.X.shape[1], lag=0, extension=int(extension))
        tangent_residual[row_index, :] = event.coefficients[row_index] @ row_basis.T
    actual_xy_by_flight = {
        flight_id: center_xy + tangent_residual[row_index, :, None] * tangents
        for row_index, flight_id in enumerate(matrix.flight_ids)
    }

    lifted = _trace_lift_tangent_reconstruction(
        matrix,
        model,
        center_xy=center_xy,
        actual_xy_by_flight=actual_xy_by_flight,
        mode="registered-dictionary",
    )

    assert lifted is not None
    np.testing.assert_allclose(lifted, tangent_residual, atol=1.0e-5)


def test_trace_lift_round_trips_as_model_artifact(tmp_path) -> None:
    matrix, model = _exact_extended_event_artifacts()
    center_xy = matrix.reference_xy_m.copy()
    tangents = _tangents_from_normals(matrix.normals_xy)
    event = model.events[0]
    tangent_residual = np.zeros_like(matrix.X)
    for row_index, extension in enumerate(event.extension_offsets):
        row_basis = _row_event_basis(event, matrix.X.shape[1], lag=0, extension=int(extension))
        tangent_residual[row_index, :] = event.coefficients[row_index] @ row_basis.T
    augmented = augment_with_trace_tangent_lift(model, tangent_residual, center_method="none")
    model_path = tmp_path / "trace_augmented_model.npz"

    save_fit_result(model_path, augmented, flight_ids=matrix.flight_ids)
    loaded = load_fit_result(model_path)
    result = evaluate_event_trace_match(matrix, loaded, event_index=0, trace_lift="stored")

    assert loaded.trace_tangent_reconstruction is not None
    assert loaded.trace_tangent_center is not None
    assert loaded.trace_tangent_coefficients is not None
    np.testing.assert_allclose(loaded.trace_tangent_reconstruction, tangent_residual, atol=2.0e-5)
    assert loaded.metadata["trace_tangent_lift"]["enabled"]
    assert "lifted_model_trace_rmse_m" in result.summary


def test_trace_lift_extra_residual_events_reduce_unmodeled_tangent_signal() -> None:
    rng = np.random.default_rng(21)
    matrix, model = _exact_extended_event_artifacts()
    tangent_residual = rng.normal(0.0, 0.01, size=matrix.X.shape)
    tangent_residual[:, 14:19] += np.asarray([[8.0], [-6.0]]) * np.asarray([0.0, 1.0, 2.0, 1.0, 0.0])

    base = augment_with_trace_tangent_lift(model, tangent_residual, center_method="none")
    extra = augment_with_trace_tangent_lift(
        model,
        tangent_residual,
        center_method="none",
        extra_residual_config=HLLRDV1Config(
            L_min=5,
            L_max=5,
            K_max=1,
            kappa_peak=0.0,
            n_min=1,
            c_null=0.0,
            epsilon_gain=0.0,
            activation_scale=0.0,
            peak_backtrack_enabled=False,
            local_simplifier_enabled=False,
        ),
    )

    assert base.trace_tangent_reconstruction is not None
    assert extra.trace_tangent_reconstruction is not None
    base_rmse = float(np.sqrt(np.mean((tangent_residual - base.trace_tangent_reconstruction) ** 2)))
    extra_rmse = float(np.sqrt(np.mean((tangent_residual - extra.trace_tangent_reconstruction) ** 2)))
    assert extra_rmse < base_rmse * 0.75
    assert extra.metadata["trace_tangent_lift"]["extra_residual_events"]["enabled"]


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
            local_simplifier_gain_sigma=0.0,
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


def _lagged_trombone_matrix() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    n = 48
    M = 100
    start = 32
    length = 12
    max_delay = 8
    source_shape = np.asarray([0.0, 1.0, 4.0, 8.0, 9.0, 5.0, 1.0, 0.0])
    shape = np.interp(
        np.linspace(0.0, 1.0, length),
        np.linspace(0.0, 1.0, source_shape.size),
        source_shape,
    )
    shape = shape / np.linalg.norm(shape)
    X = rng.normal(0.0, 0.01, size=(n, M))
    delays = rng.integers(0, max_delay + 1, size=n)
    amplitudes = rng.normal(80.0, 8.0, size=n) * rng.choice([-1.0, 1.0], size=n)
    for row_index, (delay, amplitude) in enumerate(zip(delays, amplitudes, strict=True)):
        X[row_index, start + delay : start + delay + length] += amplitude * shape
    return X, delays


def _extended_trombone_matrix() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    n = 54
    M = 110
    start = 34
    length = 12
    max_extension = 8
    source_shape = np.asarray([0.0, 1.0, 4.0, 8.0, 9.0, 8.0, 5.0, 1.0, 0.0])
    shape = np.interp(
        np.linspace(0.0, 1.0, length),
        np.linspace(0.0, 1.0, source_shape.size),
        source_shape,
    )
    shape = shape / np.linalg.norm(shape)
    pivot = int(np.argmax(shape))
    X = rng.normal(0.0, 0.01, size=(n, M))
    extensions = rng.integers(0, max_extension + 1, size=n)
    amplitudes = rng.normal(90.0, 6.0, size=n) * rng.choice([-1.0, 1.0], size=n)
    for row_index, (extension, amplitude) in enumerate(zip(extensions, amplitudes, strict=True)):
        extended_shape = np.concatenate(
            [
                shape[: pivot + 1],
                np.repeat(shape[pivot], int(extension)),
                shape[pivot + 1 :],
            ]
        )
        X[row_index, start : start + extended_shape.size] += amplitude * extended_shape
    return X, extensions


def _exact_shifted_event_artifacts() -> tuple[MatrixArtifact, HLLRDFitResult]:
    n = 2
    M = 20
    start = 5
    end = 9
    reference_xy_m = np.column_stack((100.0 * np.arange(M, dtype=float), np.zeros(M, dtype=float)))
    normals_xy = np.tile(np.asarray([0.0, 1.0]), (M, 1))
    local_shape = np.asarray([1.0, 2.0, 2.0, 1.0])
    local_shape = local_shape / np.linalg.norm(local_shape)
    basis = np.zeros((M, 2), dtype=float)
    basis[start:end, 0] = local_shape
    coefficients = np.asarray([[30.0, 0.0], [-20.0, 0.0]])
    lag_offsets = np.asarray([2, -1], dtype=int)
    reconstruction = np.zeros((n, M), dtype=float)
    for row_index, lag in enumerate(lag_offsets):
        shifted = np.zeros_like(basis)
        if lag > 0:
            shifted[lag:, :] = basis[:-lag, :]
        elif lag < 0:
            shifted[:lag, :] = basis[-lag:, :]
        else:
            shifted = basis.copy()
        reconstruction[row_index, :] = coefficients[row_index] @ shifted.T
    event = HLLRDEvent(
        start=start,
        end=end,
        basis=basis,
        coefficients=coefficients,
        active_mask=np.asarray([True, True]),
        raw_gain=float(np.sum(coefficients * coefficients)),
        active_gain=float(np.sum(coefficients * coefficients)),
        score=float(np.sum(coefficients * coefficients)),
        threshold=0.0,
        peak_index=7,
        simplifier={"lag_registered": True},
        lag_offsets=lag_offsets,
    )
    matrix = MatrixArtifact(
        X=reconstruction.copy(),
        X_centered=reconstruction.copy(),
        column_center=np.zeros(M, dtype=float),
        flight_ids=("A", "B"),
        stations=np.linspace(0.0, 1.0, M),
        reference_xy_m=reference_xy_m,
        normals_xy=normals_xy,
        origin_lat_deg=0.0,
        origin_lon_deg=0.0,
        cluster="SE",
    )
    model = HLLRDFitResult(
        events=(event,),
        dictionary=np.column_stack((basis[:, 0], basis[:, 1])),
        coefficients=coefficients.copy(),
        reconstruction=reconstruction.copy(),
        residual=np.zeros_like(reconstruction),
        explained_fraction=1.0,
        sigma_hat=0.0,
        activation_energy_floor=0.0,
        column_center=np.zeros(M, dtype=float),
        config=HLLRDV2Config(max_lag_stations=4),
        metadata={},
    )
    return matrix, model


def _exact_extended_event_artifacts() -> tuple[MatrixArtifact, HLLRDFitResult]:
    n = 2
    M = 22
    start = 5
    end = 9
    reference_xy_m = np.column_stack((100.0 * np.arange(M, dtype=float), np.zeros(M, dtype=float)))
    normals_xy = np.tile(np.asarray([0.0, 1.0]), (M, 1))
    local_shape = np.asarray([1.0, 2.0, 2.0, 1.0])
    local_shape = local_shape / np.linalg.norm(local_shape)
    basis = np.zeros((M, 2), dtype=float)
    basis[start:end, 0] = local_shape
    coefficients = np.asarray([[30.0, 0.0], [-20.0, 0.0]])
    extension_offsets = np.asarray([2, 0], dtype=int)
    event = HLLRDEvent(
        start=start,
        end=end,
        basis=basis,
        coefficients=coefficients,
        active_mask=np.asarray([True, True]),
        raw_gain=float(np.sum(coefficients * coefficients)),
        active_gain=float(np.sum(coefficients * coefficients)),
        score=float(np.sum(coefficients * coefficients)),
        threshold=0.0,
        peak_index=6,
        simplifier={"extension_registered": True},
        extension_offsets=extension_offsets,
    )
    reconstruction = np.zeros((n, M), dtype=float)
    for row_index, extension in enumerate(extension_offsets):
        row_basis = _row_event_basis(event, M, lag=0, extension=int(extension))
        reconstruction[row_index, :] = coefficients[row_index] @ row_basis.T
    matrix = MatrixArtifact(
        X=reconstruction.copy(),
        X_centered=reconstruction.copy(),
        column_center=np.zeros(M, dtype=float),
        flight_ids=("A", "B"),
        stations=np.linspace(0.0, 1.0, M),
        reference_xy_m=reference_xy_m,
        normals_xy=normals_xy,
        origin_lat_deg=0.0,
        origin_lon_deg=0.0,
        cluster="SE",
    )
    model = HLLRDFitResult(
        events=(event,),
        dictionary=np.column_stack((basis[:, 0], basis[:, 1])),
        coefficients=coefficients.copy(),
        reconstruction=reconstruction.copy(),
        residual=np.zeros_like(reconstruction),
        explained_fraction=1.0,
        sigma_hat=0.0,
        activation_energy_floor=0.0,
        column_center=np.zeros(M, dtype=float),
        config=HLLRDV2Config(max_lag_stations=0, max_extend_stations=2),
        metadata={},
    )
    return matrix, model


def _raw_tracks_from_matrix_paths(matrix: MatrixArtifact) -> pd.DataFrame:
    projection = LocalProjection(matrix.origin_lat_deg, matrix.origin_lon_deg)
    rows: list[dict[str, object]] = []
    for row_index, flight_id in enumerate(matrix.flight_ids):
        xy_m = matrix.reference_xy_m + matrix.X[row_index, :, None] * matrix.normals_xy
        lat, lon = projection.unproject(xy_m[:, 0], xy_m[:, 1])
        for station_index, (lat_deg, lon_deg) in enumerate(zip(lat, lon, strict=True)):
            rows.append(
                {
                    "flight_id": str(flight_id),
                    "time": float(station_index),
                    "lat": float(lat_deg),
                    "lon": float(lon_deg),
                }
            )
    return pd.DataFrame(rows)


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
