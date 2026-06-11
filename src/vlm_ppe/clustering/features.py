from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    track_ids: list[str]
    raw: np.ndarray
    standardized: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    n_resample: int


def build_shape_features(resampled: pd.DataFrame) -> FeatureSet:
    track_ids: list[str] = []
    vectors: list[np.ndarray] = []
    station_counts: set[int] = set()
    for flight_id, group in resampled.groupby("flight_id", sort=False):
        ordered = group.sort_values("station_index", kind="stable")
        points = ordered[["x_nm", "y_nm"]].to_numpy(dtype=float)
        station_counts.add(len(points))
        track_ids.append(str(flight_id))
        vectors.append(points.reshape(-1))

    if not vectors:
        raise ValueError("cannot build features from empty resampled tracks")
    if len(station_counts) != 1:
        raise ValueError("all resampled tracks must have the same station count")

    raw = np.vstack(vectors)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    standardized = (raw - mean) / scale
    return FeatureSet(
        track_ids=track_ids,
        raw=raw,
        standardized=standardized,
        mean=mean,
        scale=scale,
        n_resample=station_counts.pop(),
    )


def write_features(feature_set: FeatureSet, path: str | Path, metadata_path: str | Path) -> tuple[str, str]:
    feature_path = Path(path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        track_ids=np.asarray(feature_set.track_ids, dtype=object),
        raw=feature_set.raw,
        standardized=feature_set.standardized,
        mean=feature_set.mean,
        scale=feature_set.scale,
        n_resample=np.asarray([feature_set.n_resample], dtype=int),
    )
    metadata = pd.DataFrame({"flight_id": feature_set.track_ids, "feature_row": range(len(feature_set.track_ids))})
    metadata_output = Path(metadata_path)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(metadata_output, index=False)
    return feature_path.as_posix(), metadata_output.as_posix()


def load_features(path: str | Path) -> FeatureSet:
    data = np.load(Path(path), allow_pickle=True)
    return FeatureSet(
        track_ids=[str(item) for item in data["track_ids"].tolist()],
        raw=np.asarray(data["raw"], dtype=float),
        standardized=np.asarray(data["standardized"], dtype=float),
        mean=np.asarray(data["mean"], dtype=float),
        scale=np.asarray(data["scale"], dtype=float),
        n_resample=int(np.asarray(data["n_resample"], dtype=int)[0]),
    )
