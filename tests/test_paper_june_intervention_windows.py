from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ppe_evaluation.artifacts import GroundTruth, MedoidTrajectory, save_ground_truth


def _load_module():
    path = Path(__file__).resolve().parents[1] / "src" / "paper-june" / "intervention_windows" / "peak_detection_baseline.py"
    spec = importlib.util.spec_from_file_location("paper_june_intervention_peak_baseline", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _profile(cluster_id: int, energy: list[float]) -> pd.DataFrame:
    n_points = len(energy)
    station = np.arange(n_points, dtype=int)
    s_fraction = np.linspace(0.0, 1.0, n_points)
    return pd.DataFrame(
        {
            "cluster_id": cluster_id,
            "station_index": station,
            "s_fraction": s_fraction,
            "s_nm": station.astype(float),
            "template_x_nm": station.astype(float),
            "template_y_nm": np.zeros(n_points),
            "residual_energy_nm2": energy,
            "heading_dispersion": np.linspace(0.0, 0.5, n_points),
        }
    )


def test_peak_detector_ignores_flat_residual_curve() -> None:
    baseline = _load_module()
    profile = _profile(0, [3.0] * 15)

    windows = baseline.detect_peak_windows_for_cluster(profile, 0)

    assert windows == []


def test_peak_detector_emits_window_bounds_from_peak_width() -> None:
    baseline = _load_module()
    profile = _profile(2, [0.0, 0.1, 0.5, 3.0, 6.0, 3.0, 0.5, 0.1, 0.0])
    config = baseline.PeakDetectionConfig(
        min_normalized_height=0.3,
        min_normalized_prominence=0.2,
        min_distance_stations=1,
        smoothing_window=1,
        padding_stations=1,
    )

    windows = baseline.detect_peak_windows_for_cluster(profile, 2, config)

    assert len(windows) == 1
    window = windows[0]
    assert window.cluster_id == 2
    assert window.window_id == "C2_P1"
    assert window.class_name == "other"
    assert window.start_station_index <= 3
    assert window.end_station_index >= 5
    assert window.start_s_fraction < window.end_s_fraction
    assert window.peak_residual_energy_nm2 == 6.0


def test_peak_detector_processes_clusters_independently() -> None:
    baseline = _load_module()
    profiles = pd.concat(
        [
            _profile(0, [0.0, 0.2, 2.0, 0.2, 0.0]),
            _profile(1, [0.0, 0.0, 0.0, 0.0, 0.0]),
            _profile(2, [0.0, 0.1, 0.2, 0.1, 3.0, 0.1, 0.0]),
        ],
        ignore_index=True,
    )
    config = baseline.PeakDetectionConfig(
        min_normalized_height=0.3,
        min_normalized_prominence=0.2,
        min_distance_stations=1,
        smoothing_window=1,
        padding_stations=0,
    )

    windows = baseline.detect_peak_windows(profiles, config)

    assert [window.cluster_id for window in windows] == [0, 2]


def test_peak_baseline_uses_class_agnostic_evaluator_and_separate_classification(tmp_path: Path) -> None:
    baseline = _load_module()
    run_dir = tmp_path / "dataset" / "runs" / "source-run"
    (run_dir / "templates").mkdir(parents=True)
    (run_dir / "residuals").mkdir(parents=True)

    points = np.column_stack([np.arange(7, dtype=float), np.zeros(7)])
    medoid_rows = [
        {
            "cluster_id": 0,
            "medoid_track_id": "T0",
            "station_index": station_index,
            "x_nm": float(x_nm),
            "y_nm": float(y_nm),
        }
        for station_index, (x_nm, y_nm) in enumerate(points)
    ]
    pd.DataFrame(medoid_rows).to_parquet(run_dir / "templates" / "cluster_medoids.parquet", index=False)
    _profile(0, [0.0, 0.2, 1.0, 5.0, 1.0, 0.2, 0.0]).to_parquet(
        run_dir / "residuals" / "residual_profiles.parquet",
        index=False,
    )
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": "source-run",
                "config": {"dataset_id": "dataset"},
                "medoids_path": (run_dir / "templates" / "cluster_medoids.parquet").as_posix(),
                "residual_profiles_path": (run_dir / "residuals" / "residual_profiles.parquet").as_posix(),
            }
        ),
        encoding="utf-8",
    )

    ground_truth_dir = tmp_path / "dataset" / "ground_truth"
    save_ground_truth(
        GroundTruth(
            dataset_id="dataset",
            medoids=[MedoidTrajectory(medoid_id="GT000", points=points, cluster_id=0, medoid_track_id="T0")],
            windows=pd.DataFrame(
                [
                    {
                        "gt_medoid_id": "GT000",
                        "window_id": "GT000_W1",
                        "class_name": "dogleg",
                        "start_station_index": 2,
                        "end_station_index": 4,
                        "start_s_fraction": 2 / 6,
                        "end_s_fraction": 4 / 6,
                        "start_s_nm": 2.0,
                        "end_s_nm": 4.0,
                        "notes": "",
                    }
                ]
            ),
            manifest={"seed_run_id": "source-run"},
        ),
        ground_truth_dir,
    )

    config = baseline.PeakDetectionConfig(
        iou_threshold=0.1,
        min_normalized_height=0.3,
        min_normalized_prominence=0.2,
        min_distance_stations=1,
        smoothing_window=1,
        padding_stations=1,
    )
    result = baseline.run_peak_detection_for_run(
        run_dir,
        ground_truth_dir=ground_truth_dir,
        output_dir=tmp_path / "paper",
        config=config,
    )

    windows = pd.read_parquet(result.windows_path)
    assert windows["class_name"].tolist() == ["other"]
    assert result.report.window_summary["class_aware"] is False
    assert result.report.window_summary["primary_iou_threshold"] == 0.1
    assert result.report.window_summary["tp"] == 1
    assert result.report.window_classification_summary["tp"] == 0
    assert result.report.window_classification_summary["incorrect"] == 1
