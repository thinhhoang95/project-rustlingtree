"""Final every-flight assignment, including HDBSCAN noise provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from hailmary.geometry.polyline import readonly_float64

from .medoid import MedoidRecord, mean_station_distance_m


@dataclass(frozen=True, order=True)
class FlightAssignment:
    flight_id: str
    cluster_id: int
    membership_probability: float
    medoid_distance_m: float
    was_hdbscan_noise: bool
    out_of_distribution: bool
    included_in_template_training: bool

    def __post_init__(self) -> None:
        if not self.flight_id.strip():
            raise ValueError("flight_id must be nonempty")
        if self.cluster_id < 0:
            raise ValueError("final cluster_id must be nonnegative")
        if not np.isfinite(self.membership_probability) or not 0.0 <= self.membership_probability <= 1.0:
            raise ValueError("membership_probability must be within [0, 1]")
        if not np.isfinite(self.medoid_distance_m) or self.medoid_distance_m < 0.0:
            raise ValueError("medoid_distance_m must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "flight_id": self.flight_id,
            "cluster_id": self.cluster_id,
            "membership_probability": self.membership_probability,
            "medoid_distance_m": self.medoid_distance_m,
            "was_hdbscan_noise": self.was_hdbscan_noise,
            "out_of_distribution": self.out_of_distribution,
            "included_in_template_training": self.included_in_template_training,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FlightAssignment":
        return cls(
            flight_id=str(payload["flight_id"]),
            cluster_id=int(payload["cluster_id"]),
            membership_probability=float(payload["membership_probability"]),
            medoid_distance_m=float(payload["medoid_distance_m"]),
            was_hdbscan_noise=bool(payload["was_hdbscan_noise"]),
            out_of_distribution=bool(payload["out_of_distribution"]),
            included_in_template_training=bool(payload["included_in_template_training"]),
        )


def assign_all_flights(
    track_ids: Sequence[str],
    tracks_m: object,
    preliminary_labels: object,
    probabilities: object,
    medoids: Sequence[MedoidRecord],
) -> tuple[FlightAssignment, ...]:
    """Replace every ``-1`` label with the nearest medoid deterministically."""

    ids = tuple(str(item) for item in track_ids)
    tracks = readonly_float64(tracks_m, name="tracks_m", ndim=3)
    labels = np.asarray(preliminary_labels, dtype=np.int64)
    membership = readonly_float64(probabilities, name="probabilities", ndim=1)
    if len(ids) != len(tracks) or labels.shape != (len(ids),) or membership.shape != (len(ids),):
        raise ValueError("track IDs, tracks, labels, and probabilities must have matching lengths")
    if len(set(ids)) != len(ids):
        raise ValueError("track_ids must be unique")
    ordered_medoids = tuple(sorted(medoids, key=lambda item: item.cluster_id))
    medoid_by_cluster = {item.cluster_id: item for item in ordered_medoids}
    if not ordered_medoids or len(medoid_by_cluster) != len(ordered_medoids):
        raise ValueError("medoids must contain unique cluster IDs")

    assignments: list[FlightAssignment] = []
    for flight_id, track, label, probability in zip(ids, tracks, labels, membership, strict=True):
        was_noise = int(label) < 0
        if was_noise:
            choices = [
                (mean_station_distance_m(track, medoid.points_m), medoid.cluster_id, medoid)
                for medoid in ordered_medoids
            ]
            distance, cluster_id, medoid = min(choices, key=lambda item: (item[0], item[1]))
        else:
            cluster_id = int(label)
            try:
                medoid = medoid_by_cluster[cluster_id]
            except KeyError as exc:
                raise ValueError(f"label {cluster_id} has no cluster medoid") from exc
            distance = mean_station_distance_m(track, medoid.points_m)
        out_of_distribution = distance > medoid.acceptance_radius_m + 1.0e-9
        assignments.append(
            FlightAssignment(
                flight_id=flight_id,
                cluster_id=cluster_id,
                membership_probability=float(probability),
                medoid_distance_m=float(distance),
                was_hdbscan_noise=was_noise,
                out_of_distribution=out_of_distribution,
                included_in_template_training=(not was_noise and not out_of_distribution),
            )
        )
    return tuple(sorted(assignments, key=lambda item: item.flight_id))


assign_noise_to_medoids = assign_all_flights


def assign_new_flights_to_nearest_medoid(
    track_ids: Sequence[str],
    tracks_m: object,
    medoids: Sequence[MedoidRecord],
) -> tuple[FlightAssignment, ...]:
    """Portable online assignment when no version-bound HDBSCAN model is retained.

    New flights use the declared unstandardized station-distance metric and
    retain zero membership probability/noise provenance rather than implying
    that density membership was predicted.
    """

    ids = tuple(str(item) for item in track_ids)
    return assign_all_flights(
        ids,
        tracks_m,
        preliminary_labels=np.full(len(ids), -1, dtype=np.int64),
        probabilities=np.zeros(len(ids), dtype=np.float64),
        medoids=medoids,
    )


def _canonical_prediction_labels(
    raw_training_labels: np.ndarray,
    canonical_training_labels: np.ndarray,
) -> dict[int, int]:
    """Map estimator-native cluster IDs to canonical artifact cluster IDs."""

    if raw_training_labels.shape != canonical_training_labels.shape:
        raise ValueError("prediction-label rows do not match the clustering artifact")
    mapping: dict[int, int] = {}
    for raw_label in sorted(int(item) for item in np.unique(raw_training_labels) if item >= 0):
        canonical = {
            int(item)
            for item in canonical_training_labels[raw_training_labels == raw_label]
            if int(item) >= 0
        }
        if len(canonical) != 1:
            raise ValueError("selected estimator cannot be mapped to canonical cluster IDs")
        mapping[raw_label] = canonical.pop()
    return mapping


def assign_new_flights_with_membership(
    track_ids: Sequence[str],
    tracks_m: object,
    library: object,
) -> tuple[FlightAssignment, ...]:
    """Predict density membership when the portable training state is present.

    The canonical artifact stores the standardized training matrix instead of
    pickling HDBSCAN internals.  This function deterministically reconstructs
    the selected estimator, applies the persisted feature transform, and only
    falls back to the declared nearest-medoid rule for predicted noise or for
    legacy artifacts that do not carry prediction data.
    """

    medoids = tuple(getattr(library, "medoids"))
    training = getattr(library, "prediction_training", None)
    if training is None:
        return assign_new_flights_to_nearest_medoid(track_ids, tracks_m, medoids)

    from hailmary.clustering.features import transform_shape_features

    ids = tuple(str(item) for item in track_ids)
    feature_set = transform_shape_features(
        tracks_m,
        getattr(library, "feature_transform"),
        track_ids=ids,
    )
    selection = getattr(library, "clustering")
    parameters = dict(getattr(selection, "selected_parameters"))
    training_features = np.asarray(training.standardized_features, dtype=np.float64)
    canonical_training = np.asarray(selection.labels, dtype=np.int64)

    try:
        if str(selection.algorithm) == "hdbscan":
            import hdbscan  # pyright: ignore[reportMissingImports]

            model = hdbscan.HDBSCAN(
                min_cluster_size=int(parameters["min_cluster_size"]),
                min_samples=None
                if parameters.get("min_samples") is None
                else int(parameters["min_samples"]),
                cluster_selection_method=str(parameters["cluster_selection_method"]),
                metric="euclidean",
                prediction_data=True,
                core_dist_n_jobs=1,
                approx_min_span_tree=False,
                gen_min_span_tree=False,
            ).fit(np.array(training_features, dtype=np.float64, order="C", copy=True))
            raw_training = np.asarray(model.labels_, dtype=np.int64)
            raw_labels, probabilities = hdbscan.approximate_predict(
                model,
                np.array(feature_set.standardized, dtype=np.float64, order="C", copy=True),
            )
        elif str(selection.algorithm) == "kmeans" and int(parameters.get("k", 1)) > 1:
            from sklearn.cluster import KMeans  # pyright: ignore[reportMissingImports]

            model = KMeans(
                n_clusters=int(parameters["k"]),
                n_init=int(parameters.get("n_init", 50)),
                random_state=int(parameters.get("random_state", 17)),
                algorithm="lloyd",
            ).fit(training_features)
            raw_training = np.asarray(model.labels_, dtype=np.int64)
            raw_labels = np.asarray(model.predict(feature_set.standardized), dtype=np.int64)
            probabilities = np.ones(len(ids), dtype=np.float64)
        elif str(selection.algorithm) == "kmeans":
            raw_training = np.zeros(len(training_features), dtype=np.int64)
            raw_labels = np.zeros(len(ids), dtype=np.int64)
            probabilities = np.ones(len(ids), dtype=np.float64)
        else:
            return assign_new_flights_to_nearest_medoid(ids, feature_set.tracks_m, medoids)

        label_mapping = _canonical_prediction_labels(raw_training, canonical_training)
        predicted_labels = np.asarray(
            [label_mapping.get(int(label), -1) for label in raw_labels],
            dtype=np.int64,
        )
        predicted_probabilities = np.asarray(probabilities, dtype=np.float64)
    except (ImportError, KeyError, TypeError, ValueError):
        # Portable artifacts remain usable when their original estimator is
        # unavailable; the fallback is explicit through noise/zero-probability
        # provenance on every returned assignment.
        return assign_new_flights_to_nearest_medoid(ids, feature_set.tracks_m, medoids)

    return assign_all_flights(
        ids,
        feature_set.tracks_m,
        preliminary_labels=predicted_labels,
        probabilities=predicted_probabilities,
        medoids=medoids,
    )


def assign_new_flights(
    track_ids: Sequence[str],
    tracks_m: object,
    medoids_or_library: object,
) -> tuple[FlightAssignment, ...]:
    """Assign held-out flights using membership prediction when available."""

    if hasattr(medoids_or_library, "feature_transform") and hasattr(
        medoids_or_library,
        "clustering",
    ):
        return assign_new_flights_with_membership(
            track_ids,
            tracks_m,
            medoids_or_library,
        )
    return assign_new_flights_to_nearest_medoid(
        track_ids,
        tracks_m,
        medoids_or_library,  # type: ignore[arg-type]
    )


__all__ = [
    "FlightAssignment",
    "assign_all_flights",
    "assign_new_flights",
    "assign_new_flights_to_nearest_medoid",
    "assign_new_flights_with_membership",
    "assign_noise_to_medoids",
]
