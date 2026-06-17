from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ppe_evaluation.artifacts import GroundTruth, MedoidTrajectory, load_ground_truth, save_ground_truth
from ppe_evaluation.cli import main as eval_main
from ppe_evaluation.frechet import discrete_frechet_distance
from ppe_evaluation.gui import create_app
from ppe_evaluation.matching import detection_confusion_summary, match_medoids
from ppe_evaluation.metrics import evaluate_run, interval_iou


def _line(y: float = 0.0, n: int = 5) -> np.ndarray:
    return np.column_stack([np.arange(n, dtype=float), np.full(n, y, dtype=float)])


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "data" / "artifacts" / "ppe" / "fixture" / "runs" / "agent-run"
    (run_dir / "processed").mkdir(parents=True)
    (run_dir / "templates").mkdir(parents=True)
    (run_dir / "residuals").mkdir(parents=True)

    rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    for cluster_id, y_value in [(0, 0.0), (1, 5.0)]:
        for station_index, (x_nm, y_nm) in enumerate(_line(y_value)):
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "medoid_track_id": f"T{cluster_id}",
                    "station_index": station_index,
                    "x_nm": float(x_nm),
                    "y_nm": float(y_nm),
                }
            )
        for track_index, offset in enumerate([0.0, 0.2]):
            flight_id = f"T{cluster_id}-{track_index}"
            assignment_rows.append({"flight_id": flight_id, "cluster_id": cluster_id})
            for station_index, (x_nm, y_nm) in enumerate(_line(y_value + offset)):
                trace_rows.append(
                    {
                        "flight_id": flight_id,
                        "station_index": station_index,
                        "x_nm": float(x_nm),
                        "y_nm": float(y_nm),
                    }
                )
    pd.DataFrame(rows).to_parquet(run_dir / "templates" / "cluster_medoids.parquet", index=False)
    pd.DataFrame(trace_rows).to_parquet(run_dir / "processed" / "resampled_tracks.parquet", index=False)
    pd.DataFrame(assignment_rows).to_csv(run_dir / "templates" / "chosen_cluster_assignments.csv", index=False)
    pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "window_id": "C0_W1",
                "class_name": "dogleg",
                "confidence": 0.9,
                "visual_reason": "",
                "start_station_index": 1,
                "end_station_index": 3,
                "start_s_fraction": 0.25,
                "end_s_fraction": 0.75,
                "start_s_nm": 1.0,
                "end_s_nm": 3.0,
                "length_nm": 2.0,
                "peak_residual_energy_nm2": 1.0,
                "peak_heading_dispersion": 0.1,
                "track_ids": ["T0"],
            },
            {
                "cluster_id": 1,
                "window_id": "C1_W1",
                "class_name": "trombone",
                "confidence": 0.8,
                "visual_reason": "",
                "start_station_index": 1,
                "end_station_index": 2,
                "start_s_fraction": 0.25,
                "end_s_fraction": 0.5,
                "start_s_nm": 1.0,
                "end_s_nm": 2.0,
                "length_nm": 1.0,
                "peak_residual_energy_nm2": 1.0,
                "peak_heading_dispersion": 0.1,
                "track_ids": ["T1"],
            },
        ]
    ).to_parquet(run_dir / "residuals" / "intervention_windows.parquet", index=False)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "run_id": "agent-run",
                "config": {"dataset_id": "fixture"},
                "medoids_path": (run_dir / "templates" / "cluster_medoids.parquet").as_posix(),
                "intervention_windows_path": (run_dir / "residuals" / "intervention_windows.parquet").as_posix(),
                "resampled_tracks_path": (run_dir / "processed" / "resampled_tracks.parquet").as_posix(),
                "cluster_assignments_path": (run_dir / "templates" / "chosen_cluster_assignments.csv").as_posix(),
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _write_ground_truth(run_dir: Path) -> Path:
    gt_dir = run_dir.parent.parent / "ground_truth"
    save_ground_truth(
        GroundTruth(
            dataset_id="fixture",
            medoids=[
                MedoidTrajectory(
                    medoid_id="GT000",
                    points=_line(0.1),
                    cluster_id=0,
                    medoid_track_id="T0",
                    source_run_id="agent-run",
                )
            ],
            windows=pd.DataFrame(
                [
                    {
                        "gt_medoid_id": "GT000",
                        "window_id": "GT000_W1",
                        "class_name": "dogleg",
                        "start_station_index": 1,
                        "end_station_index": 3,
                        "start_s_fraction": 0.25,
                        "end_s_fraction": 0.75,
                        "start_s_nm": 1.0,
                        "end_s_nm": 3.0,
                        "notes": "",
                    }
                ]
            ),
            manifest={"seed_run_id": "agent-run"},
        ),
        gt_dir,
    )
    return gt_dir


def test_discrete_frechet_distance_handles_identical_and_shifted_lines() -> None:
    assert discrete_frechet_distance(_line(), _line()) == pytest.approx(0.0)
    assert discrete_frechet_distance(_line(), _line(2.0)) == pytest.approx(2.0)


def test_medoid_matching_uses_strict_threshold() -> None:
    result = match_medoids(
        [MedoidTrajectory(medoid_id="GT000", points=_line())],
        [MedoidTrajectory(medoid_id="C0", points=_line(0.75), cluster_id=0)],
        threshold_nm=0.75,
    )

    assert result.matches == []
    assert len(result.assignments) == 1
    assert result.assignments[0].matched is False
    assert result.unmatched_gt_ids == ["GT000"]
    assert result.unmatched_pred_cluster_ids == [0]


def test_detection_confusion_summary_uses_object_detection_denominator() -> None:
    summary = detection_confusion_summary(n_gt=2, n_pred=3, n_tp=1)

    assert summary["tp"] == 1
    assert summary["fp"] == 2
    assert summary["tn"] == 0
    assert summary["fn"] == 1
    assert summary["precision"] == pytest.approx(1 / 3)
    assert summary["recall"] == pytest.approx(1 / 2)
    assert summary["accuracy"] == pytest.approx(1 / 4)


def test_interval_iou_uses_fractional_window_overlap() -> None:
    assert interval_iou(0.2, 0.6, 0.4, 0.8) == pytest.approx(0.2 / 0.6)
    assert interval_iou(0.2, 0.6, 0.2, 0.6) == pytest.approx(1.0)


def test_evaluate_run_writes_summary_and_class_aware_window_metrics(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    gt_dir = _write_ground_truth(run_dir)

    report = evaluate_run(run_dir, gt_dir)

    assert report.medoid_summary["tp"] == 1
    assert report.medoid_summary["fp"] == 1
    assert report.medoid_summary["tn"] == 0
    assert report.medoid_summary["precision"] == pytest.approx(0.5)
    assert report.window_summary["tp"] == 1
    assert report.window_summary["fp"] == 1
    assert report.window_summary["tn"] == 0
    assert report.window_summary["map_at_0_5"] == pytest.approx(0.5)
    assert report.window_classification_summary["tp"] == 1
    assert report.window_classification_summary["fp"] == 1
    assert report.window_classification_summary["tn"] == 0
    assert report.window_classification_summary["fn"] == 0
    assert report.window_classification_summary["accuracy"] == pytest.approx(1.0)
    assert (run_dir / "evaluation" / "summary.json").exists()
    assert (run_dir / "evaluation" / "medoid_matches.csv").exists()
    assert (run_dir / "evaluation" / "window_matches.csv").exists()
    assert (run_dir / "evaluation" / "window_classification_matches.csv").exists()
    assert (run_dir / "evaluation" / "window_overlay.png").exists()
    log_text = (run_dir / "evaluation" / "evaluation_log.md").read_text(encoding="utf-8")
    assert "## Clusters" in log_text
    assert "## Windows" in log_text
    assert "GT000_W1: 1-3 (dogleg)" in log_text
    assert "C0_W1: 1-3 (dogleg)" in log_text
    assert "IoU >= 0.10 and class matches" in log_text


def test_window_metrics_report_multiple_iou_threshold_suites(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    gt_dir = _write_ground_truth(run_dir)
    windows_path = run_dir / "residuals" / "intervention_windows.parquet"
    windows = pd.read_parquet(windows_path)
    windows.loc[windows["window_id"] == "C0_W1", "end_station_index"] = 2
    windows.loc[windows["window_id"] == "C0_W1", "end_s_fraction"] = 0.55
    windows.loc[windows["window_id"] == "C0_W1", "end_s_nm"] = 2.2
    windows.to_parquet(windows_path, index=False)

    report = evaluate_run(run_dir, gt_dir)

    suites = {item["iou_threshold"]: item for item in report.window_summary["by_iou_threshold"]}
    assert sorted(suites) == [0.1, 0.25, 0.5, 0.75, 0.9]
    assert suites[0.1]["tp"] == 1
    assert suites[0.25]["tp"] == 1
    assert suites[0.5]["tp"] == 1
    assert suites[0.75]["tp"] == 0
    assert suites[0.9]["tp"] == 0
    assert suites[0.75]["fp"] == 2
    assert suites[0.75]["fn"] == 1
    assert report.window_summary["tp"] == suites[0.1]["tp"]
    assert report.window_summary["primary_iou_threshold"] == pytest.approx(0.1)
    assert set(report.window_matches["iou_threshold"]) == {0.1, 0.25, 0.5, 0.75, 0.9}
    rejected = report.window_matches.loc[
        (report.window_matches["iou_threshold"] == 0.75)
        & (report.window_matches["pred_window_id"] == "C0_W1")
        & (report.window_matches["reason"] == "unmatched_prediction")
    ].iloc[0]
    assert rejected["gt_window_id"] == "GT000_W1"
    assert rejected["gt_start_station_index"] == 1
    assert rejected["gt_end_station_index"] == 3


def test_class_agnostic_window_metrics_ignore_class_name(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    gt_dir = _write_ground_truth(run_dir)
    windows_path = run_dir / "residuals" / "intervention_windows.parquet"
    windows = pd.read_parquet(windows_path)
    windows.loc[windows["window_id"] == "C0_W1", "class_name"] = "trombone"
    windows.to_parquet(windows_path, index=False)

    class_aware = evaluate_run(run_dir, gt_dir, run_dir / "evaluation-aware")
    class_agnostic = evaluate_run(run_dir, gt_dir, run_dir / "evaluation-agnostic", class_aware=False)

    assert class_aware.window_summary["tp"] == 0
    assert class_aware.window_summary["fn"] == 1
    assert class_aware.window_summary["class_aware"] is True
    assert class_agnostic.window_summary["tp"] == 1
    assert class_agnostic.window_summary["fn"] == 0
    assert class_agnostic.window_summary["class_aware"] is False
    matched = class_agnostic.window_matches.loc[class_agnostic.window_matches["matched"]].iloc[0]
    assert matched["gt_class_name"] == "dogleg"
    assert matched["pred_class_name"] == "trombone"
    assert class_agnostic.window_classification_summary["tp"] == 0
    assert class_agnostic.window_classification_summary["fp"] == 2
    assert class_agnostic.window_classification_summary["fn"] == 1
    assert class_agnostic.window_classification_summary["accuracy"] == pytest.approx(0.0)
    classification_match = class_agnostic.window_classification_matches.loc[
        class_agnostic.window_classification_matches["classification_reason"] == "wrong_class"
    ].iloc[0]
    assert classification_match["gt_class_name"] == "dogleg"
    assert classification_match["pred_class_name"] == "trombone"


def test_cli_evaluate_uses_default_dataset_ground_truth(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    _write_ground_truth(run_dir)

    assert eval_main(["evaluate", "--run-dir", run_dir.as_posix()]) == 0
    summary = json.loads((run_dir / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    assert summary["medoid_summary"]["tp"] == 1


def test_cli_evaluate_supports_class_agnostic_window_matching(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    _write_ground_truth(run_dir)
    windows_path = run_dir / "residuals" / "intervention_windows.parquet"
    windows = pd.read_parquet(windows_path)
    windows.loc[windows["window_id"] == "C0_W1", "class_name"] = "trombone"
    windows.to_parquet(windows_path, index=False)

    assert eval_main(["evaluate", "--run-dir", run_dir.as_posix(), "--class-agnostic"]) == 0
    summary = json.loads((run_dir / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    assert summary["window_summary"]["class_aware"] is False
    assert summary["window_summary"]["tp"] == 1
    assert summary["window_classification_summary"]["tp"] == 0
    assert summary["window_classification_summary"]["accuracy"] == pytest.approx(0.0)
    log_text = (run_dir / "evaluation" / "evaluation_log.md").read_text(encoding="utf-8")
    assert "but class is trombone instead of dogleg" in log_text


def test_gui_save_endpoint_writes_ground_truth_artifacts(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    gt_dir = tmp_path / "gt"
    client = TestClient(create_app(run_dir, gt_dir))

    response = client.post(
        "/api/save",
        json={
            "accepted_cluster_ids": [0],
            "windows": [
                {
                    "cluster_id": 0,
                    "window_id": "manual-window",
                    "class_name": "dogleg",
                    "start_station_index": 1,
                    "end_station_index": 3,
                    "notes": "manual",
                }
            ],
        },
    )

    assert response.status_code == 200
    truth = load_ground_truth(gt_dir)
    assert len(truth.medoids) == 1
    assert truth.medoids[0].cluster_id == 0
    assert truth.windows.iloc[0]["window_id"] == "manual-window"
    assert (gt_dir / "medoids.npz").exists()
    assert (gt_dir / "windows.parquet").exists()
    assert (gt_dir / "manifest.json").exists()


def test_gui_run_payload_includes_adsb_traces_by_cluster(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    client = TestClient(create_app(run_dir, tmp_path / "gt"))

    response = client.get("/api/run")

    assert response.status_code == 200
    payload = response.json()
    cluster_zero = next(item for item in payload["medoids"] if item["cluster_id"] == 0)
    cluster_one = next(item for item in payload["medoids"] if item["cluster_id"] == 1)
    assert len(cluster_zero["traces"]) == 2
    assert len(cluster_one["traces"]) == 2
    assert cluster_zero["traces"][0]["points"][0] == [0.0, 0.0]
