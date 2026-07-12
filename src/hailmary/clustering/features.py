"""Deterministic 2-D trajectory shape features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hailmary.geometry.polyline import readonly_float64


def _track_tensor(
    tracks: Mapping[str, object] | object,
    *,
    track_ids: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(tracks, Mapping):
        normalized_tracks = {str(key): value for key, value in tracks.items()}
        if len(normalized_tracks) != len(tracks):
            raise ValueError("track mapping keys are not unique after string normalization")
        ids = (
            tuple(sorted(normalized_tracks))
            if track_ids is None
            else tuple(str(item) for item in track_ids)
        )
        if len(set(ids)) != len(ids):
            raise ValueError("track_ids must be unique")
        missing = [item for item in ids if item not in normalized_tracks]
        if missing:
            raise ValueError(f"tracks are missing IDs: {missing}")
        arrays = [readonly_float64(normalized_tracks[item], name=f"track {item!r}", ndim=2) for item in ids]
        if not arrays:
            raise ValueError("cannot build features from no tracks")
        tensor = readonly_float64(np.stack(arrays, axis=0), name="track tensor", ndim=3)
    else:
        if track_ids is None:
            raise ValueError("track_ids are required when tracks is an array")
        ids = tuple(str(item) for item in track_ids)
        tensor = readonly_float64(tracks, name="track tensor", ndim=3)
    if tensor.shape[0] != len(ids):
        raise ValueError("track_ids length does not match track tensor")
    if tensor.shape[2:] != (2,):
        raise ValueError("each resampled track must have shape (station, 2)")
    if tensor.shape[1] < 2:
        raise ValueError("each resampled track must have at least two stations")
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("track_ids must be unique and nonempty")
    return ids, tensor


@dataclass(frozen=True)
class ShapeFeatureTransform:
    """Persisted training-cohort standardization transform."""

    mean: np.ndarray
    scale: np.ndarray
    station_count: int
    coordinate_count: int = 2

    def __post_init__(self) -> None:
        mean = readonly_float64(self.mean, name="feature mean", ndim=1)
        scale = readonly_float64(self.scale, name="feature scale", ndim=1)
        if self.station_count < 2:
            raise ValueError("station_count must be at least two")
        if self.coordinate_count != 2:
            raise ValueError("version-1 shape features require two coordinates")
        expected = self.station_count * self.coordinate_count
        if len(mean) != expected or len(scale) != expected:
            raise ValueError("normalization vectors do not match the declared feature shape")
        if np.any(scale <= 0.0):
            raise ValueError("feature scales must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @property
    def feature_count(self) -> int:
        return len(self.mean)

    def transform(self, raw: object) -> np.ndarray:
        array = readonly_float64(raw, name="raw features", ndim=2)
        if array.shape[1:] != (self.feature_count,):
            raise ValueError("raw feature width does not match the training transform")
        return readonly_float64((array - self.mean) / self.scale, name="standardized features", ndim=2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_count": self.station_count,
            "coordinate_count": self.coordinate_count,
            "feature_order": "flattened_xy_upstream_to_threshold",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShapeFeatureTransform":
        return cls(
            mean=payload["mean"],
            scale=payload["scale"],
            station_count=int(payload["station_count"]),
            coordinate_count=int(payload.get("coordinate_count", 2)),
        )


@dataclass(frozen=True)
class ShapeFeatureSet:
    track_ids: tuple[str, ...]
    tracks_m: np.ndarray
    raw: np.ndarray
    standardized: np.ndarray
    transform: ShapeFeatureTransform

    def __post_init__(self) -> None:
        ids, tracks = _track_tensor(self.tracks_m, track_ids=self.track_ids)
        raw = readonly_float64(self.raw, name="raw features", ndim=2)
        standardized = readonly_float64(self.standardized, name="standardized features", ndim=2)
        expected = (len(ids), self.transform.feature_count)
        if raw.shape != expected or standardized.shape != expected:
            raise ValueError("feature arrays have an invalid shape")
        if tracks.shape[1] != self.transform.station_count:
            raise ValueError("track station count does not match transform")
        object.__setattr__(self, "track_ids", ids)
        object.__setattr__(self, "tracks_m", tracks)
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "standardized", standardized)

    # Compatibility names used in the existing PPE feature tests.
    @property
    def mean(self) -> np.ndarray:
        return self.transform.mean

    @property
    def scale(self) -> np.ndarray:
        return self.transform.scale

    @property
    def n_resample(self) -> int:
        return self.transform.station_count


def fit_shape_features(
    tracks: Mapping[str, object] | object,
    *,
    track_ids: Sequence[str] | None = None,
    zero_scale_tolerance: float = 1.0e-12,
) -> ShapeFeatureSet:
    """Fit the version-1 flattened-geometry transform on a training cohort."""

    if not np.isfinite(zero_scale_tolerance) or zero_scale_tolerance < 0.0:
        raise ValueError("zero_scale_tolerance must be finite and nonnegative")
    ids, tensor = _track_tensor(tracks, track_ids=track_ids)
    raw = readonly_float64(tensor.reshape(len(ids), -1), name="raw features", ndim=2)
    mean = raw.mean(axis=0, dtype=np.float64)
    scale = raw.std(axis=0, dtype=np.float64)
    scale = np.where(scale <= float(zero_scale_tolerance), 1.0, scale)
    transform = ShapeFeatureTransform(mean=mean, scale=scale, station_count=tensor.shape[1])
    return ShapeFeatureSet(
        track_ids=ids,
        tracks_m=tensor,
        raw=raw,
        standardized=transform.transform(raw),
        transform=transform,
    )


def transform_shape_features(
    tracks: Mapping[str, object] | object,
    transform: ShapeFeatureTransform,
    *,
    track_ids: Sequence[str] | None = None,
) -> ShapeFeatureSet:
    """Transform held-out tracks without refitting training statistics."""

    ids, tensor = _track_tensor(tracks, track_ids=track_ids)
    if tensor.shape[1] != transform.station_count:
        raise ValueError("held-out station count does not match training transform")
    raw = readonly_float64(tensor.reshape(len(ids), -1), name="raw features", ndim=2)
    return ShapeFeatureSet(
        track_ids=ids,
        tracks_m=tensor,
        raw=raw,
        standardized=transform.transform(raw),
        transform=transform,
    )


build_shape_features = fit_shape_features
FeatureSet = ShapeFeatureSet
