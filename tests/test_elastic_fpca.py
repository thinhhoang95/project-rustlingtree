from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from hllrd.cli import elastic_fpca_main
from hllrd.elastic_fpca import (
    ElasticEventFPCA,
    ElasticFPCAConfig,
    ElasticFPCAResult,
    extract_event_functions,
    fit_elastic_event_fpca,
    horizontal_component_delta,
    load_elastic_fpca_result,
    save_elastic_fpca_result,
    vertical_component_delta,
    _function_sample_rank,
)
from hllrd.fit import HLLRDEvent, HLLRDFitResult, HLLRDV1Config, save_fit_result
from hllrd.matrix import MatrixArtifact, save_matrix_artifact


def test_extract_event_functions_uses_only_active_flights() -> None:
    event = _synthetic_hllrd_event(active_mask=np.asarray([True, False, True, False]))
    X_centered = np.arange(4 * 12, dtype=float).reshape(4, 12)

    extracted = extract_event_functions(event, 3, ("A", "B", "C", "D"), X_centered)

    assert extracted.event_index == 3
    assert extracted.functions.shape == (event.length, 2)
    assert extracted.active_rows.tolist() == [0, 2]
    assert extracted.active_flight_ids == ("A", "C")
    np.testing.assert_allclose(extracted.functions, X_centered[[0, 2], event.start : event.end].T)
    assert np.all(np.isfinite(extracted.functions))


def test_fit_elastic_event_fpca_synthetic_smoke() -> None:
    model = _synthetic_fit_result(flight_count=8, station_count=40)
    matrix = replace(
        _synthetic_matrix(flight_count=8, station_count=40),
        X=model.reconstruction.copy(),
        X_centered=model.reconstruction.copy(),
    )

    result = fit_elastic_event_fpca(
        matrix,
        model,
        config=ElasticFPCAConfig(components=2, std_grid=(-1.0, 0.0, 1.0), min_active_flights=4),
    )

    assert len(result.events) == 1
    event = result.events[0]
    assert event.vertical_f_pca.shape == (event.length, 3, 2)
    assert event.horizontal_gam_pca.shape == (3, event.length, 2)
    assert event.vertical_coefficients.shape == (8, 2)
    assert event.horizontal_coefficients.shape == (8, 2)
    np.testing.assert_allclose(vertical_component_delta(event, 0, 0.0, result.config.std_grid), 0.0)
    np.testing.assert_allclose(horizontal_component_delta(event, 0, 0.0, result.config.std_grid), 0.0)


def test_fit_elastic_event_fpca_caps_components_to_effective_data_rank() -> None:
    matrix = _synthetic_matrix(flight_count=8, station_count=20)
    rows = np.arange(8, dtype=float)
    shape = np.sin(np.linspace(0.0, np.pi, 10))
    X = np.zeros_like(matrix.X)
    X[:, 5:15] = rows[:, None] * shape[None, :]
    model = _synthetic_fit_result(flight_count=8, station_count=20)
    matrix = replace(matrix, X=X, X_centered=X.copy())

    result = fit_elastic_event_fpca(
        matrix,
        model,
        config=ElasticFPCAConfig(components=3, std_grid=(-1.0, 0.0, 1.0), min_active_flights=4),
    )

    assert len(result.events) == 1
    assert result.events[0].component_count == 1


def test_function_sample_rank_uses_centered_sample_rank() -> None:
    shape = np.sin(np.linspace(0.0, np.pi, 8))
    amplitudes = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    functions = shape[:, None] * amplitudes[None, :] + 100.0

    assert _function_sample_rank(functions) == 1


def test_elastic_fpca_artifact_round_trips(tmp_path: Path) -> None:
    result = _manual_elastic_result()
    path = tmp_path / "elastic.npz"

    save_elastic_fpca_result(path, result)
    loaded = load_elastic_fpca_result(path)

    assert loaded.config.std_grid == result.config.std_grid
    assert loaded.events[0].active_flight_ids == result.events[0].active_flight_ids
    np.testing.assert_allclose(loaded.events[0].vertical_f_pca, result.events[0].vertical_f_pca)
    np.testing.assert_allclose(loaded.events[0].horizontal_gam_pca, result.events[0].horizontal_gam_pca)


def test_horizontal_component_delta_synthesizes_with_inverse_warp() -> None:
    time = np.linspace(0.0, 1.0, 5)
    gamma = time**2
    horizontal = np.stack([time, time, gamma], axis=0)[:, :, None]
    event = ElasticEventFPCA(
        event_index=0,
        start=0,
        end=time.size,
        active_rows=np.asarray([0], dtype=int),
        active_flight_ids=("FLT0",),
        time=time,
        functions=np.zeros((time.size, 1), dtype=float),
        fmean=time.copy(),
        aligned_functions=np.zeros((time.size, 1), dtype=float),
        warps=time[:, None],
        vertical_f_pca=np.zeros((time.size, 3, 1), dtype=float),
        vertical_coefficients=np.zeros((1, 1), dtype=float),
        vertical_latent=np.asarray([1.0]),
        horizontal_gam_pca=horizontal,
        horizontal_coefficients=np.zeros((1, 1), dtype=float),
        horizontal_latent=np.asarray([1.0]),
    )

    delta = horizontal_component_delta(event, 0, 1.0, (-1.0, 0.0, 1.0))

    expected_inverse = np.interp(time, gamma, time)
    np.testing.assert_allclose(delta, expected_inverse - time)
    assert np.all(delta[1:-1] > 0.0)


def test_elastic_fpca_artifact_rejects_inconsistent_event_shapes(tmp_path: Path) -> None:
    result = _manual_elastic_result()
    bad_event = replace(result.events[0], start=3)
    bad_result = replace(result, events=(bad_event,))

    with pytest.raises(ValueError, match="time shape does not match"):
        save_elastic_fpca_result(tmp_path / "bad_elastic.npz", bad_result)


def test_elastic_fpca_cli_smoke(tmp_path: Path) -> None:
    model = _synthetic_fit_result(flight_count=8, station_count=40)
    matrix = replace(
        _synthetic_matrix(flight_count=8, station_count=40),
        X=model.reconstruction.copy(),
        X_centered=model.reconstruction.copy(),
    )
    matrix_path = tmp_path / "matrix.npz"
    model_path = tmp_path / "model.npz"
    output_path = tmp_path / "elastic_cli.npz"
    save_matrix_artifact(matrix_path, matrix)
    save_fit_result(model_path, model, flight_ids=matrix.flight_ids)

    elastic_fpca_main(
        [
            "--matrix",
            str(matrix_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--components",
            "1",
            "--std-grid",
            "-1,0,1",
            "--min-active-flights",
            "4",
        ]
    )

    loaded = load_elastic_fpca_result(output_path)
    assert len(loaded.events) == 1
    assert loaded.events[0].component_count == 1


def test_south_east_viewer_elastic_response_changes_only_event_window(tmp_path: Path) -> None:
    matplotlib.use("Agg")
    matrix = _synthetic_matrix(flight_count=3, station_count=10)
    model = _manual_fit_result_for_viewer()
    elastic = _manual_elastic_result()
    matrix_path = tmp_path / "matrix.npz"
    model_path = tmp_path / "model.npz"
    elastic_path = tmp_path / "elastic.npz"
    save_matrix_artifact(matrix_path, matrix)
    save_fit_result(model_path, model, flight_ids=matrix.flight_ids)
    save_elastic_fpca_result(elastic_path, elastic)
    viewer_module = _load_viewer_module()

    app = viewer_module.SouthEastInteractive(
        matrix_path,
        model_path,
        elastic_path,
        raw_adsb_dir=None,
        split_gap_seconds=1,
        processes=1,
        background_source="matrix",
        background_alpha=0.1,
        background_linewidth=0.3,
    )
    app.create()
    center = app.mean_xy_m.copy()
    np.testing.assert_allclose(app.event_response_xy_m(), center)

    app.amplitude_slider.set_val(1.0)
    response = app.event_response_xy_m()

    np.testing.assert_allclose(response[:2], center[:2])
    assert np.max(np.abs(response[2:7] - center[2:7])) > 0.0
    np.testing.assert_allclose(response[7:], center[7:])

    app.current_family = "horizontal"
    app.amplitude_slider.set_val(1.0)
    np.testing.assert_allclose(app.event_response_xy_m(), center)
    app.fig.canvas.draw()


def test_south_east_viewer_horizontal_response_spans_observed_score_range(tmp_path: Path) -> None:
    matplotlib.use("Agg")
    matrix = _synthetic_matrix(flight_count=3, station_count=10)
    arch = np.asarray([0.0, 0.0, 0.0, 10.0, 20.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    matrix.reference_xy_m[:, 1] = arch
    event_weight = arch / float(np.max(arch))
    matrix.X[0] = -10.0 * event_weight
    matrix.X[1] = 0.0
    matrix.X[2] = 20.0 * event_weight
    model = _manual_fit_result_for_viewer()
    elastic = _manual_elastic_result(horizontal_coefficients=np.asarray([[-1.0], [0.0], [1.0]]))
    matrix_path = tmp_path / "matrix.npz"
    model_path = tmp_path / "model.npz"
    elastic_path = tmp_path / "elastic.npz"
    save_matrix_artifact(matrix_path, matrix)
    save_fit_result(model_path, model, flight_ids=matrix.flight_ids)
    save_elastic_fpca_result(elastic_path, elastic)
    viewer_module = _load_viewer_module()

    app = viewer_module.SouthEastInteractive(
        matrix_path,
        model_path,
        elastic_path,
        raw_adsb_dir=None,
        split_gap_seconds=1,
        processes=1,
        background_source="matrix",
        background_alpha=0.1,
        background_linewidth=0.3,
    )
    app.create()
    app.current_family = "horizontal"
    center = app.mean_xy_m.copy()
    event = app.current_elastic_event()
    center_extent = app._event_trombone_extent_m(event, center[None, :, :])[0]

    np.testing.assert_allclose(app.event_response_xy_m(), center)
    np.testing.assert_allclose(center_extent, 20.0)

    app.amplitude_slider.set_val(2.0)
    high_response = app.event_response_xy_m()
    high_extent = app._event_trombone_extent_m(event, high_response[None, :, :])[0]
    np.testing.assert_allclose(high_extent, 40.0)

    app.amplitude_slider.set_val(-2.0)
    low_response = app.event_response_xy_m()
    low_extent = app._event_trombone_extent_m(event, low_response[None, :, :])[0]
    np.testing.assert_allclose(low_extent, 10.0)
    np.testing.assert_allclose(low_response[:2], center[:2])
    np.testing.assert_allclose(low_response[7:], center[7:])
    app.fig.canvas.draw()


def test_south_east_viewer_rejects_stale_elastic_artifact(tmp_path: Path) -> None:
    matplotlib.use("Agg")
    matrix = _synthetic_matrix(flight_count=3, station_count=10)
    model = _manual_fit_result_for_viewer()
    elastic = _manual_elastic_result()
    stale_event = replace(elastic.events[0], start=1, end=6)
    stale_elastic = replace(elastic, events=(stale_event,))
    matrix_path = tmp_path / "matrix.npz"
    model_path = tmp_path / "model.npz"
    elastic_path = tmp_path / "stale_elastic.npz"
    save_matrix_artifact(matrix_path, matrix)
    save_fit_result(model_path, model, flight_ids=matrix.flight_ids)
    save_elastic_fpca_result(elastic_path, stale_elastic)
    viewer_module = _load_viewer_module()

    with pytest.raises(RuntimeError, match="does not match model interval"):
        viewer_module.SouthEastInteractive(
            matrix_path,
            model_path,
            elastic_path,
            raw_adsb_dir=None,
            split_gap_seconds=1,
            processes=1,
            background_source="matrix",
            background_alpha=0.1,
            background_linewidth=0.3,
        )


def _synthetic_hllrd_event(active_mask: np.ndarray) -> HLLRDEvent:
    station_count = 12
    start = 2
    end = 10
    time = np.linspace(0.0, 1.0, end - start)
    basis = np.zeros((station_count, 2), dtype=float)
    basis[start:end, 0] = np.sin(np.pi * time)
    basis[start:end, 1] = np.cos(np.pi * time)
    coefficients = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.5],
            [0.0, 0.0],
        ]
    )
    return HLLRDEvent(
        start=start,
        end=end,
        basis=basis,
        coefficients=coefficients,
        active_mask=active_mask,
        raw_gain=1.0,
        active_gain=1.0,
        score=1.0,
        threshold=0.0,
        peak_index=5,
    )


def _synthetic_matrix(flight_count: int, station_count: int) -> MatrixArtifact:
    reference = np.column_stack((np.linspace(0.0, 1_000.0, station_count), np.zeros(station_count)))
    normals = np.tile(np.asarray([0.0, 1.0]), (station_count, 1))
    X = np.zeros((flight_count, station_count), dtype=float)
    return MatrixArtifact(
        X=X,
        X_centered=X.copy(),
        column_center=np.zeros(station_count, dtype=float),
        flight_ids=tuple(f"FLT{index}" for index in range(flight_count)),
        stations=np.linspace(0.0, 1.0, station_count),
        reference_xy_m=reference,
        normals_xy=normals,
        origin_lat_deg=32.0,
        origin_lon_deg=-97.0,
        cluster="SE",
    )


def _synthetic_fit_result(flight_count: int, station_count: int) -> HLLRDFitResult:
    start = 5
    end = station_count - 5
    local_time = np.linspace(0.0, 1.0, end - start)
    basis = np.zeros((station_count, 2), dtype=float)
    basis[start:end, 0] = np.sin(np.pi * local_time)
    basis[start:end, 1] = np.cos(np.pi * local_time)
    coefficients = np.zeros((flight_count, 2), dtype=float)
    for row in range(flight_count):
        amplitude = 1.0 + 0.08 * (row - flight_count / 2.0)
        phase = 0.10 * (row - flight_count / 2.0)
        coefficients[row] = [amplitude * np.cos(phase), amplitude * np.sin(phase)]
    event = HLLRDEvent(
        start=start,
        end=end,
        basis=basis,
        coefficients=coefficients,
        active_mask=np.ones(flight_count, dtype=bool),
        raw_gain=1.0,
        active_gain=1.0,
        score=1.0,
        threshold=0.0,
        peak_index=(start + end) // 2,
    )
    reconstruction = coefficients @ basis.T
    return HLLRDFitResult(
        events=(event,),
        dictionary=basis,
        coefficients=coefficients,
        reconstruction=reconstruction,
        residual=np.zeros_like(reconstruction),
        explained_fraction=1.0,
        sigma_hat=0.0,
        activation_energy_floor=0.0,
        column_center=np.zeros(station_count, dtype=float),
        config=HLLRDV1Config(),
        metadata={},
    )


def _manual_fit_result_for_viewer() -> HLLRDFitResult:
    matrix = _synthetic_matrix(flight_count=3, station_count=10)
    basis = np.zeros((10, 2), dtype=float)
    event = HLLRDEvent(
        start=2,
        end=7,
        basis=basis,
        coefficients=np.zeros((3, 2), dtype=float),
        active_mask=np.ones(3, dtype=bool),
        raw_gain=0.0,
        active_gain=0.0,
        score=0.0,
        threshold=0.0,
        peak_index=4,
    )
    return HLLRDFitResult(
        events=(event,),
        dictionary=np.zeros((10, 2), dtype=float),
        coefficients=np.zeros((3, 2), dtype=float),
        reconstruction=np.zeros((3, 10), dtype=float),
        residual=np.zeros((3, 10), dtype=float),
        explained_fraction=0.0,
        sigma_hat=0.0,
        activation_energy_floor=0.0,
        column_center=matrix.column_center,
        config=HLLRDV1Config(),
        metadata={},
    )


def _manual_elastic_result(horizontal_coefficients: np.ndarray | None = None) -> ElasticFPCAResult:
    time = np.linspace(0.0, 1.0, 5)
    shape = np.asarray([0.0, 1.0, 2.0, 1.0, 0.0])
    vertical = np.stack([-shape, np.zeros_like(shape), shape], axis=1)[:, :, None]
    horizontal = np.tile(time, (3, 1))[:, :, None]
    hcoef = (
        np.zeros((3, 1), dtype=float)
        if horizontal_coefficients is None
        else np.asarray(horizontal_coefficients, dtype=float)
    )
    event = ElasticEventFPCA(
        event_index=0,
        start=2,
        end=7,
        active_rows=np.asarray([0, 1, 2], dtype=int),
        active_flight_ids=("FLT0", "FLT1", "FLT2"),
        time=time,
        functions=np.zeros((5, 3), dtype=float),
        fmean=np.zeros(5, dtype=float),
        aligned_functions=np.zeros((5, 3), dtype=float),
        warps=np.tile(time[:, None], (1, 3)),
        vertical_f_pca=vertical,
        vertical_coefficients=np.zeros((3, 1), dtype=float),
        vertical_latent=np.asarray([1.0]),
        horizontal_gam_pca=horizontal,
        horizontal_coefficients=hcoef,
        horizontal_latent=np.asarray([1.0]),
    )
    return ElasticFPCAResult(
        events=(event,),
        config=ElasticFPCAConfig(components=1, std_grid=(-1.0, 0.0, 1.0), min_active_flights=1),
        metadata={},
    )


def _load_viewer_module():
    path = Path(__file__).resolve().parents[1] / "src" / "hllrd" / "examples" / "south-east-interactive.py"
    spec = importlib.util.spec_from_file_location("south_east_interactive_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
