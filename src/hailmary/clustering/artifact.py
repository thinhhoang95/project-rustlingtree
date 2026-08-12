"""Canonical, content-addressed clustering artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .assignment import FlightAssignment, assign_all_flights
from .features import ShapeFeatureTransform, fit_shape_features
from .hdbscan_runner import (
    ClusteringSelection,
    HDBSCANSelectionConfig,
    _coerce_config,
    run_hdbscan_sweep,
)
from .medoid import MedoidRecord, compute_cluster_medoids

CLUSTER_LIBRARY_SCHEMA_VERSION = "hailmary.cluster-library.v1"


@dataclass(frozen=True)
class PredictionTrainingData:
    """Portable inputs used to reconstruct the selected online predictor.

    HDBSCAN's prediction tree is a version-bound Python object and therefore is
    not a suitable canonical artifact payload.  Persisting the standardized
    training matrix and its stable row identities lets the exact selected
    estimator be reconstructed deterministically while keeping the JSON
    artifact portable across processes.
    """

    track_ids: tuple[str, ...]
    standardized_features: np.ndarray

    def __post_init__(self) -> None:
        identifiers = tuple(str(item) for item in self.track_ids)
        matrix = np.array(
            self.standardized_features,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("prediction training IDs must be nonempty and unique")
        if matrix.ndim != 2 or matrix.shape[0] != len(identifiers) or matrix.shape[1] == 0:
            raise ValueError("prediction training features must have shape (track, feature)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("prediction training features must be finite")
        matrix.setflags(write=False)
        object.__setattr__(self, "track_ids", identifiers)
        object.__setattr__(self, "standardized_features", matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_ids": list(self.track_ids),
            "standardized_features": self.standardized_features.tolist(),
            "reconstruction": "refit_selected_estimator_v1",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionTrainingData":
        return cls(
            track_ids=tuple(str(item) for item in payload["track_ids"]),
            standardized_features=payload["standardized_features"],
        )


def canonical_json_dumps(payload: object) -> str:
    """Serialize JSON data canonically and reject NaN/Infinity."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_content_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


def _canonical_mapping(value: Mapping[str, Any] | None, *, name: str) -> str:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    encoded = canonical_json_dumps(dict(value))
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return encoded


@dataclass(frozen=True, init=False)
class ClusterLibrary:
    """One runway-partitioned, deterministic medoid library."""

    schema_version: str
    dataset_id: str
    airport: str
    runway: str
    feature_transform: ShapeFeatureTransform
    clustering: ClusteringSelection
    medoids: tuple[MedoidRecord, ...]
    assignments: tuple[FlightAssignment, ...]
    prediction_training: PredictionTrainingData | None
    artifact_content_hash: str
    _projection_json: str
    _selection_config_json: str
    _metadata_json: str

    def __init__(
        self,
        *,
        dataset_id: str,
        airport: str,
        runway: str,
        projection: Mapping[str, Any],
        feature_transform: ShapeFeatureTransform,
        clustering: ClusteringSelection,
        medoids: tuple[MedoidRecord, ...],
        assignments: tuple[FlightAssignment, ...],
        prediction_training: PredictionTrainingData | None = None,
        selection_config: object,
        metadata: Mapping[str, Any] | None = None,
        schema_version: str = CLUSTER_LIBRARY_SCHEMA_VERSION,
        artifact_content_hash: str | None = None,
    ) -> None:
        scalar_fields = {
            "schema_version": str(schema_version).strip(),
            "dataset_id": str(dataset_id).strip(),
            "airport": str(airport).strip().upper(),
            "runway": str(runway).strip().upper(),
        }
        if any(not value for value in scalar_fields.values()):
            raise ValueError("schema_version, dataset_id, airport, and runway must be nonempty")
        if not scalar_fields["runway"].startswith("RW"):
            scalar_fields["runway"] = f"RW{scalar_fields['runway']}"
        medoid_tuple = tuple(sorted(medoids, key=lambda item: item.cluster_id))
        assignment_tuple = tuple(sorted(assignments, key=lambda item: item.flight_id))
        if not medoid_tuple or len({item.cluster_id for item in medoid_tuple}) != len(medoid_tuple):
            raise ValueError("cluster library must contain unique medoids")
        if not assignment_tuple or len({item.flight_id for item in assignment_tuple}) != len(assignment_tuple):
            raise ValueError("cluster library must assign every unique flight")
        cluster_ids = {item.cluster_id for item in medoid_tuple}
        if any(item.cluster_id not in cluster_ids for item in assignment_tuple):
            raise ValueError("an assignment references a cluster without a medoid")
        if len(clustering.labels) != len(assignment_tuple):
            raise ValueError("clustering labels and final assignments must have equal lengths")
        if any(len(item.points_m) != feature_transform.station_count for item in medoid_tuple):
            raise ValueError("medoid station counts must match the feature transform")
        if prediction_training is not None:
            if set(prediction_training.track_ids) != {
                item.flight_id for item in assignment_tuple
            }:
                raise ValueError(
                    "prediction training rows must match the assigned flight IDs"
                )
            if prediction_training.standardized_features.shape != (
                len(assignment_tuple),
                feature_transform.feature_count,
            ):
                raise ValueError(
                    "prediction training features do not match the cluster feature transform"
                )

        projection_json = _canonical_mapping(projection, name="projection")
        if isinstance(selection_config, HDBSCANSelectionConfig):
            selection_config_payload = selection_config.to_dict()
        elif is_dataclass(selection_config) and not isinstance(selection_config, type):
            selection_config_payload = asdict(selection_config)
        elif isinstance(selection_config, Mapping):
            selection_config_payload = dict(selection_config)
        else:
            raise ValueError("selection_config must be a clustering config or mapping")
        selection_config_json = _canonical_mapping(selection_config_payload, name="selection_config")
        metadata_json = _canonical_mapping(metadata, name="metadata")
        for name, value in scalar_fields.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "feature_transform", feature_transform)
        object.__setattr__(self, "clustering", clustering)
        object.__setattr__(self, "medoids", medoid_tuple)
        object.__setattr__(self, "assignments", assignment_tuple)
        object.__setattr__(self, "prediction_training", prediction_training)
        object.__setattr__(self, "_projection_json", projection_json)
        object.__setattr__(self, "_selection_config_json", selection_config_json)
        object.__setattr__(self, "_metadata_json", metadata_json)
        computed_hash = canonical_content_hash(self.to_dict(include_hash=False))
        if artifact_content_hash is not None and artifact_content_hash != computed_hash:
            raise ValueError("cluster artifact content hash does not match its canonical payload")
        object.__setattr__(self, "artifact_content_hash", computed_hash)

    @property
    def projection(self) -> dict[str, Any]:
        return json.loads(self._projection_json)

    @property
    def selection_config(self) -> dict[str, Any]:
        return json.loads(self._selection_config_json)

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self._metadata_json)

    @property
    def resample_station_count(self) -> int:
        return self.feature_transform.station_count

    @property
    def hdbscan_outlier_assignments(self) -> tuple[FlightAssignment, ...]:
        """Assignments whose original density-clustering label was ``-1``."""

        return tuple(item for item in self.assignments if item.was_hdbscan_noise)

    @property
    def corpus_assignments(self) -> tuple[FlightAssignment, ...]:
        """Observed-flight assignments eligible for corpus construction.

        Cluster artifacts retain nearest-medoid fallback assignments for
        prediction provenance.  Corpus construction has a stricter contract:
        an HDBSCAN noise label is a rejected observation, not a route member.
        """

        return tuple(item for item in self.assignments if not item.was_hdbscan_noise)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "partition": {"airport": self.airport, "runway": self.runway},
            "projection": self.projection,
            "resample_station_count": self.resample_station_count,
            "feature_normalization": self.feature_transform.to_dict(),
            "clustering_config": self.selection_config,
            "clustering_result": self.clustering.to_dict(),
            "clusters": [item.to_dict() for item in self.medoids],
            "assignments": [item.to_dict() for item in self.assignments],
            "metadata": self.metadata,
        }
        if self.prediction_training is not None:
            payload["prediction_training"] = self.prediction_training.to_dict()
        if include_hash:
            payload["artifact_content_hash"] = self.artifact_content_hash
        return payload

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClusterLibrary":
        partition = dict(payload["partition"])
        return cls(
            schema_version=str(payload["schema_version"]),
            dataset_id=str(payload["dataset_id"]),
            airport=str(partition["airport"]),
            runway=str(partition["runway"]),
            projection=dict(payload["projection"]),
            feature_transform=ShapeFeatureTransform.from_dict(dict(payload["feature_normalization"])),
            clustering=ClusteringSelection.from_dict(dict(payload["clustering_result"])),
            medoids=tuple(MedoidRecord.from_dict(item) for item in payload["clusters"]),
            assignments=tuple(FlightAssignment.from_dict(item) for item in payload["assignments"]),
            prediction_training=None
            if payload.get("prediction_training") is None
            else PredictionTrainingData.from_dict(dict(payload["prediction_training"])),
            selection_config=dict(payload["clustering_config"]),
            metadata=dict(payload.get("metadata", {})),
            artifact_content_hash=None
            if payload.get("artifact_content_hash") is None
            else str(payload["artifact_content_hash"]),
        )

    @classmethod
    def from_json(cls, value: str) -> "ClusterLibrary":
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise ValueError("cluster library JSON must contain an object")
        return cls.from_dict(payload)

    @classmethod
    def read(cls, path: str | Path) -> "ClusterLibrary":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def build_cluster_library(
    resampled_tracks_m: Mapping[str, object],
    *,
    dataset_id: str,
    airport: str,
    runway: str,
    projection: Mapping[str, Any],
    config: object | None = None,
    acceptance_quantile: float | None = None,
    acceptance_multiplier: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ClusterLibrary:
    """Build the complete deterministic cluster artifact for one partition."""

    original_config = HDBSCANSelectionConfig() if config is None else config
    runner_config = _coerce_config(original_config)
    features = fit_shape_features(resampled_tracks_m)
    configured_station_count = getattr(original_config, "n_resample", None)
    if configured_station_count is not None and int(configured_station_count) != features.n_resample:
        raise ValueError("resampled track station count does not match ClusteringConfig.n_resample")
    clustering = run_hdbscan_sweep(
        features.standardized,
        runner_config,
        tracks_m=features.tracks_m,
    )
    resolved_quantile = float(
        getattr(original_config, "acceptance_quantile", 0.95)
        if acceptance_quantile is None
        else acceptance_quantile
    )
    resolved_multiplier = float(
        getattr(original_config, "acceptance_multiplier", 1.0)
        if acceptance_multiplier is None
        else acceptance_multiplier
    )
    preliminary_medoids = compute_cluster_medoids(
        features.track_ids,
        features.tracks_m,
        clustering.labels,
        acceptance_quantile=resolved_quantile,
        acceptance_multiplier=resolved_multiplier,
    )
    label_mapping = {
        medoid.cluster_id: canonical_id
        for canonical_id, medoid in enumerate(
            sorted(preliminary_medoids, key=lambda item: item.medoid_flight_id)
        )
    }
    canonical_labels = np.asarray(
        [label_mapping.get(int(label), -1) for label in clustering.labels],
        dtype=np.int64,
    )
    canonical_diagnostics = tuple(
        replace(item, cluster_id=label_mapping[item.cluster_id])
        for item in clustering.cluster_diagnostics
    )
    candidates = tuple(
        (
            replace(
                candidate,
                cluster_diagnostics=tuple(
                    replace(item, cluster_id=label_mapping[item.cluster_id])
                    for item in candidate.cluster_diagnostics
                ),
            )
            if candidate.algorithm == clustering.algorithm
            and candidate.parameters == clustering.selected_parameters
            else candidate
        )
        for candidate in clustering.candidates
    )
    clustering = replace(
        clustering,
        labels=canonical_labels,
        cluster_diagnostics=tuple(
            sorted(canonical_diagnostics, key=lambda item: item.cluster_id)
        ),
        candidates=candidates,
    )
    medoids = tuple(
        sorted(
            (
                replace(medoid, cluster_id=label_mapping[medoid.cluster_id])
                for medoid in preliminary_medoids
            ),
            key=lambda item: item.cluster_id,
        )
    )
    assignments = assign_all_flights(
        features.track_ids,
        features.tracks_m,
        clustering.labels,
        clustering.probabilities,
        medoids,
    )
    artifact_metadata = {
        "station_order": "upstream_to_threshold",
        "medoid_metric": "mean_euclidean_station_distance_2d_v1",
        "every_flight_assigned": True,
        **({} if metadata is None else dict(metadata)),
    }
    return ClusterLibrary(
        dataset_id=dataset_id,
        airport=airport,
        runway=runway,
        projection=projection,
        feature_transform=features.transform,
        clustering=clustering,
        medoids=medoids,
        assignments=assignments,
        prediction_training=PredictionTrainingData(
            track_ids=features.track_ids,
            standardized_features=features.standardized,
        ),
        selection_config=original_config,
        metadata=artifact_metadata,
    )


load_cluster_library = ClusterLibrary.read
