from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GROUND_TRUTH_SCHEMA_VERSION = 1
WINDOW_COLUMNS = [
    "gt_medoid_id",
    "window_id",
    "class_name",
    "start_station_index",
    "end_station_index",
    "start_s_fraction",
    "end_s_fraction",
    "start_s_nm",
    "end_s_nm",
    "notes",
]


@dataclass(frozen=True)
class MedoidTrajectory:
    medoid_id: str
    points: np.ndarray
    cluster_id: int | None = None
    medoid_track_id: str | None = None
    source_run_id: str | None = None


@dataclass(frozen=True)
class GroundTruth:
    dataset_id: str
    medoids: list[MedoidTrajectory]
    windows: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PredictionArtifacts:
    run_dir: Path
    run_id: str
    dataset_id: str
    medoids: list[MedoidTrajectory]
    windows: pd.DataFrame
    state: dict[str, Any]


def default_ground_truth_dir(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    state = _read_state(run_path)
    dataset_id = _dataset_id_from_state(state)
    if dataset_id:
        return run_path.parents[1] / "ground_truth" if run_path.parent.name == "runs" else Path("data/artifacts/ppe") / dataset_id / "ground_truth"
    if run_path.parent.name == "runs":
        return run_path.parents[1] / "ground_truth"
    raise ValueError("ground truth directory is required when dataset_id cannot be inferred")


def load_run_artifacts(run_dir: str | Path) -> PredictionArtifacts:
    root = Path(run_dir)
    state = _read_state(root)
    run_id = str(state.get("run_id") or root.name)
    dataset_id = _dataset_id_from_state(state) or _dataset_id_from_path(root)
    if dataset_id is None:
        raise ValueError(f"could not infer dataset_id for run_dir={root}")

    medoids_path = _path_from_state_or_default(state, "medoids_path", root / "templates" / "cluster_medoids.parquet")
    medoids = _load_predicted_medoids(medoids_path)

    windows_path = _path_from_state_or_default(
        state,
        "intervention_windows_path",
        root / "residuals" / "intervention_windows.parquet",
    )
    windows = _read_parquet_or_empty(windows_path)
    return PredictionArtifacts(
        run_dir=root,
        run_id=run_id,
        dataset_id=dataset_id,
        medoids=medoids,
        windows=windows,
        state=state,
    )


def load_ground_truth(path: str | Path) -> GroundTruth:
    root = Path(path)
    manifest_path = root / "manifest.json"
    medoids_path = root / "medoids.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing ground-truth manifest: {manifest_path}")
    if not medoids_path.exists():
        raise FileNotFoundError(f"missing ground-truth medoids: {medoids_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = np.load(medoids_path, allow_pickle=False)
    medoid_array = np.asarray(data["medoids"], dtype=float)
    if medoid_array.ndim != 3 or medoid_array.shape[2] != 2:
        raise ValueError("ground-truth medoids must have shape (n_medoids, n_stations, 2)")

    ids = _string_list(data, "gt_medoid_ids", len(medoid_array), prefix="GT")
    source_run_ids = _optional_string_list(data, "source_run_ids", len(medoid_array))
    source_cluster_ids = _optional_int_list(data, "source_cluster_ids", len(medoid_array))
    source_track_ids = _optional_string_list(data, "source_medoid_track_ids", len(medoid_array))
    medoids = [
        MedoidTrajectory(
            medoid_id=ids[index],
            points=np.asarray(medoid_array[index], dtype=float),
            cluster_id=source_cluster_ids[index],
            medoid_track_id=source_track_ids[index],
            source_run_id=source_run_ids[index],
        )
        for index in range(len(medoid_array))
    ]
    windows = _read_parquet_or_empty(root / "windows.parquet", columns=WINDOW_COLUMNS)
    return GroundTruth(
        dataset_id=str(manifest.get("dataset_id") or root.parent.name),
        medoids=medoids,
        windows=_normalize_ground_truth_windows(windows),
        manifest=manifest,
    )


def save_ground_truth(ground_truth: GroundTruth, path: str | Path) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    medoids = ground_truth.medoids
    if medoids:
        medoid_array = np.stack([np.asarray(medoid.points, dtype=float) for medoid in medoids], axis=0)
    else:
        medoid_array = np.empty((0, 0, 2), dtype=float)
    manifest = {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "dataset_id": ground_truth.dataset_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "frechet_threshold_nm": 3.0,
        "window_iou_threshold": 0.1,
        **ground_truth.manifest,
    }

    np.savez(
        root / "medoids.npz",
        medoids=medoid_array,
        gt_medoid_ids=np.asarray([medoid.medoid_id for medoid in medoids], dtype=str),
        source_run_ids=np.asarray([medoid.source_run_id or "" for medoid in medoids], dtype=str),
        source_cluster_ids=np.asarray([medoid.cluster_id if medoid.cluster_id is not None else -1 for medoid in medoids], dtype=int),
        source_medoid_track_ids=np.asarray([medoid.medoid_track_id or "" for medoid in medoids], dtype=str),
    )
    windows = _normalize_ground_truth_windows(ground_truth.windows)
    windows.to_parquet(root / "windows.parquet", index=False)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def medoid_station_frame(medoid: MedoidTrajectory) -> pd.DataFrame:
    points = np.asarray(medoid.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("medoid points must have shape (n_points, 2)")
    if len(points) == 1:
        s_nm = np.asarray([0.0])
    else:
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        s_nm = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(s_nm[-1])
    s_fraction = np.zeros(len(points), dtype=float) if total <= 0.0 else s_nm / total
    return pd.DataFrame(
        {
            "station_index": np.arange(len(points), dtype=int),
            "x_nm": points[:, 0],
            "y_nm": points[:, 1],
            "s_nm": s_nm,
            "s_fraction": s_fraction,
        }
    )


def _load_predicted_medoids(path: Path) -> list[MedoidTrajectory]:
    if not path.exists():
        raise FileNotFoundError(f"missing predicted medoids: {path}")
    frame = pd.read_parquet(path)
    required = {"cluster_id", "medoid_track_id", "station_index", "x_nm", "y_nm"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"predicted medoids are missing columns: {sorted(missing)}")
    medoids: list[MedoidTrajectory] = []
    for cluster_id, group in frame.groupby("cluster_id", sort=True):
        ordered = group.sort_values("station_index", kind="stable")
        track_ids = ordered["medoid_track_id"].astype(str).unique().tolist()
        medoids.append(
            MedoidTrajectory(
                medoid_id=f"C{int(cluster_id)}",
                points=ordered[["x_nm", "y_nm"]].to_numpy(dtype=float),
                cluster_id=int(cluster_id),
                medoid_track_id=track_ids[0] if track_ids else None,
            )
        )
    return medoids


def _normalize_ground_truth_windows(windows: pd.DataFrame) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame(columns=WINDOW_COLUMNS)
    normalized = windows.copy()
    for column in WINDOW_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = "" if column in {"gt_medoid_id", "window_id", "class_name", "notes"} else 0
    normalized = normalized[WINDOW_COLUMNS]
    normalized["gt_medoid_id"] = normalized["gt_medoid_id"].astype(str)
    normalized["window_id"] = normalized["window_id"].astype(str)
    normalized["class_name"] = normalized["class_name"].astype(str)
    normalized["start_station_index"] = normalized["start_station_index"].astype(int)
    normalized["end_station_index"] = normalized["end_station_index"].astype(int)
    for column in ["start_s_fraction", "end_s_fraction", "start_s_nm", "end_s_nm"]:
        normalized[column] = normalized[column].astype(float)
    normalized["notes"] = normalized["notes"].fillna("").astype(str)
    return normalized


def _read_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_id_from_state(state: dict[str, Any]) -> str | None:
    config = state.get("config")
    if isinstance(config, dict) and config.get("dataset_id"):
        return str(config["dataset_id"])
    return None


def _dataset_id_from_path(run_dir: Path) -> str | None:
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent.name
    return None


def _path_from_state_or_default(state: dict[str, Any], key: str, default: Path) -> Path:
    value = state.get(key)
    return Path(value) if value else default


def _read_parquet_or_empty(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def _string_list(data: np.lib.npyio.NpzFile, key: str, length: int, *, prefix: str) -> list[str]:
    if key not in data:
        return [f"{prefix}{index:03d}" for index in range(length)]
    return [str(item) for item in data[key].tolist()]


def _optional_string_list(data: np.lib.npyio.NpzFile, key: str, length: int) -> list[str | None]:
    if key not in data:
        return [None] * length
    values = [str(item) for item in data[key].tolist()]
    return [value or None for value in values]


def _optional_int_list(data: np.lib.npyio.NpzFile, key: str, length: int) -> list[int | None]:
    if key not in data:
        return [None] * length
    values = [int(item) for item in data[key].tolist()]
    return [value if value >= 0 else None for value in values]
