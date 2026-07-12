"""True observed-track medoids using the declared station-space metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hailmary.geometry.polyline import readonly_float64


def mean_station_distance_m(track_a_m: object, track_b_m: object) -> float:
    left = readonly_float64(track_a_m, name="track_a_m", ndim=2)
    right = readonly_float64(track_b_m, name="track_b_m", ndim=2)
    if left.shape != right.shape or left.shape[1:] != (2,):
        raise ValueError("tracks must have the same (station, 2) shape")
    return float(np.linalg.norm(left - right, axis=1).mean())


def pairwise_station_distances_m(tracks_m: object) -> np.ndarray:
    tracks = readonly_float64(tracks_m, name="tracks_m", ndim=3)
    if tracks.shape[2:] != (2,) or tracks.shape[0] == 0 or tracks.shape[1] < 2:
        raise ValueError("tracks_m must have shape (track, station>=2, 2)")
    difference = tracks[:, np.newaxis, :, :] - tracks[np.newaxis, :, :, :]
    distances = np.linalg.norm(difference, axis=3).mean(axis=2)
    return readonly_float64(distances, name="pairwise distances", ndim=2)


@dataclass(frozen=True)
class MedoidRecord:
    cluster_id: int
    medoid_flight_id: str
    member_flight_ids: tuple[str, ...]
    points_m: np.ndarray
    mean_distance_m: float
    max_distance_m: float
    acceptance_radius_m: float
    pairwise_mean_distance_m: float

    def __post_init__(self) -> None:
        if self.cluster_id < 0:
            raise ValueError("cluster_id must be nonnegative")
        members = tuple(sorted(str(item) for item in self.member_flight_ids))
        if not members or len(set(members)) != len(members):
            raise ValueError("member_flight_ids must be nonempty and unique")
        if self.medoid_flight_id not in members:
            raise ValueError("medoid must be an observed cluster member")
        points = readonly_float64(self.points_m, name="medoid points_m", ndim=2)
        if points.shape[1:] != (2,) or len(points) < 2:
            raise ValueError("medoid points_m must have shape (station>=2, 2)")
        metrics = (
            self.mean_distance_m,
            self.max_distance_m,
            self.acceptance_radius_m,
            self.pairwise_mean_distance_m,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in metrics):
            raise ValueError("medoid distance summaries must be finite and nonnegative")
        object.__setattr__(self, "medoid_flight_id", str(self.medoid_flight_id))
        object.__setattr__(self, "member_flight_ids", members)
        object.__setattr__(self, "points_m", points)

    @property
    def member_count(self) -> int:
        return len(self.member_flight_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "medoid_flight_id": self.medoid_flight_id,
            "member_flight_ids": list(self.member_flight_ids),
            "member_count": self.member_count,
            "points_m": self.points_m.tolist(),
            "mean_distance_m": self.mean_distance_m,
            "max_distance_m": self.max_distance_m,
            "acceptance_radius_m": self.acceptance_radius_m,
            "pairwise_mean_distance_m": self.pairwise_mean_distance_m,
            "metric": "mean_euclidean_station_distance_2d_v1",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MedoidRecord":
        return cls(
            cluster_id=int(payload["cluster_id"]),
            medoid_flight_id=str(payload["medoid_flight_id"]),
            member_flight_ids=tuple(str(item) for item in payload["member_flight_ids"]),
            points_m=payload["points_m"],
            mean_distance_m=float(payload["mean_distance_m"]),
            max_distance_m=float(payload["max_distance_m"]),
            acceptance_radius_m=float(payload["acceptance_radius_m"]),
            pairwise_mean_distance_m=float(payload["pairwise_mean_distance_m"]),
        )


def select_medoid(
    track_ids: Sequence[str],
    tracks_m: object,
    *,
    cluster_id: int = 0,
    acceptance_quantile: float = 0.95,
    acceptance_multiplier: float = 1.0,
) -> MedoidRecord:
    """Select the member minimizing total distance, tying by ``flight_id``."""

    if not 0.0 <= acceptance_quantile <= 1.0:
        raise ValueError("acceptance_quantile must be within [0, 1]")
    if not np.isfinite(acceptance_multiplier) or acceptance_multiplier <= 0.0:
        raise ValueError("acceptance_multiplier must be finite and positive")
    ids = tuple(str(item) for item in track_ids)
    tracks = readonly_float64(tracks_m, name="tracks_m", ndim=3)
    if len(ids) != len(tracks) or not ids:
        raise ValueError("track_ids must match a nonempty track tensor")
    if len(set(ids)) != len(ids):
        raise ValueError("track_ids must be unique")
    order = np.asarray(sorted(range(len(ids)), key=lambda index: ids[index]), dtype=int)
    ordered_ids = tuple(ids[index] for index in order)
    ordered_tracks = tracks[order]
    pairwise = pairwise_station_distances_m(ordered_tracks)
    distance_sums = pairwise.sum(axis=1)
    medoid_index = int(np.argmin(distance_sums))
    member_distances = pairwise[medoid_index]
    try:
        acceptance_radius = float(  # pyright: ignore[reportCallIssue]
            np.quantile(member_distances, acceptance_quantile, method="higher")
        )
    except TypeError:  # NumPy < 1.22 compatibility
        acceptance_radius = float(  # pyright: ignore[reportCallIssue]
            np.quantile(  # pyright: ignore[reportCallIssue]
                member_distances,
                acceptance_quantile,
                interpolation="higher",
            )
        )
    return MedoidRecord(
        cluster_id=int(cluster_id),
        medoid_flight_id=ordered_ids[medoid_index],
        member_flight_ids=ordered_ids,
        points_m=ordered_tracks[medoid_index],
        mean_distance_m=float(member_distances.mean()),
        max_distance_m=float(member_distances.max()),
        acceptance_radius_m=acceptance_radius * float(acceptance_multiplier),
        pairwise_mean_distance_m=float(pairwise.mean()),
    )


def compute_cluster_medoids(
    track_ids: Sequence[str],
    tracks_m: object,
    labels: object,
    *,
    acceptance_quantile: float = 0.95,
    acceptance_multiplier: float = 1.0,
) -> tuple[MedoidRecord, ...]:
    ids = tuple(str(item) for item in track_ids)
    tracks = readonly_float64(tracks_m, name="tracks_m", ndim=3)
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or len(label_array) != len(ids) or len(tracks) != len(ids):
        raise ValueError("track IDs, tracks, and labels must have matching lengths")
    medoids: list[MedoidRecord] = []
    for cluster_id in sorted(int(item) for item in np.unique(label_array) if item >= 0):
        indices = np.flatnonzero(label_array == cluster_id)
        medoids.append(
            select_medoid(
                [ids[index] for index in indices],
                tracks[indices],
                cluster_id=cluster_id,
                acceptance_quantile=acceptance_quantile,
                acceptance_multiplier=acceptance_multiplier,
            )
        )
    if not medoids:
        raise ValueError("cannot compute medoids when every track is noise")
    return tuple(medoids)


compute_medoids = compute_cluster_medoids
