"""Deterministic HDBSCAN parameter selection with a KMeans fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata
from itertools import product
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import KMeans  # pyright: ignore[reportMissingImports]
from sklearn.metrics import silhouette_score  # pyright: ignore[reportMissingImports]

from hailmary.geometry.polyline import readonly_float64


@dataclass(frozen=True)
class HDBSCANSelectionConfig:
    min_cluster_sizes: tuple[int, ...] = (4, 8, 12, 16, 24)
    min_samples_values: tuple[int | None, ...] = (None, 3, 5, 8)
    cluster_selection_methods: tuple[str, ...] = ("eom", "leaf")
    silhouette_weight: float = 0.35
    persistence_weight: float = 0.25
    coverage_weight: float = 0.25
    fragmentation_weight: float = 0.15
    max_noise_fraction: float = 0.60
    max_fragmentation: float = 0.25
    minimum_cluster_count: int = 2
    kmeans_k_values: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    kmeans_n_init: int = 50
    random_state: int = 17

    def __post_init__(self) -> None:
        sizes = tuple(sorted(set(int(item) for item in self.min_cluster_sizes)))
        if not sizes or any(item < 2 for item in sizes):
            raise ValueError("min_cluster_sizes must contain integers at least 2")
        samples = tuple(self.min_samples_values)
        if any(item is not None and (isinstance(item, bool) or int(item) < 1) for item in samples):
            raise ValueError("min_samples_values must contain None or positive integers")
        methods = tuple(sorted(set(str(item) for item in self.cluster_selection_methods)))
        if not methods or any(item not in {"eom", "leaf"} for item in methods):
            raise ValueError("cluster_selection_methods must contain eom and/or leaf")
        weights = (
            self.silhouette_weight,
            self.persistence_weight,
            self.coverage_weight,
            self.fragmentation_weight,
        )
        if not all(np.isfinite(item) and item >= 0.0 for item in weights) or sum(weights) <= 0.0:
            raise ValueError("score weights must be finite, nonnegative, and not all zero")
        if not 0.0 <= self.max_noise_fraction <= 1.0:
            raise ValueError("max_noise_fraction must be within [0, 1]")
        if not 0.0 <= self.max_fragmentation <= 1.0:
            raise ValueError("max_fragmentation must be within [0, 1]")
        if self.minimum_cluster_count < 1:
            raise ValueError("minimum_cluster_count must be positive")
        k_values = tuple(sorted(set(int(item) for item in self.kmeans_k_values)))
        if any(item < 2 for item in k_values):
            raise ValueError("kmeans_k_values must contain integers at least 2")
        if self.kmeans_n_init < 1:
            raise ValueError("kmeans_n_init must be positive")
        if isinstance(self.random_state, bool) or int(self.random_state) != self.random_state:
            raise ValueError("random_state must be an integer")
        object.__setattr__(self, "min_cluster_sizes", sizes)
        object.__setattr__(self, "min_samples_values", samples)
        object.__setattr__(self, "cluster_selection_methods", methods)
        object.__setattr__(self, "kmeans_k_values", k_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_cluster_sizes": list(self.min_cluster_sizes),
            "min_samples_values": list(self.min_samples_values),
            "cluster_selection_methods": list(self.cluster_selection_methods),
            "score_weights": {
                "silhouette": self.silhouette_weight,
                "persistence": self.persistence_weight,
                "coverage": self.coverage_weight,
                "fragmentation_penalty": self.fragmentation_weight,
            },
            "rejection_thresholds": {
                "max_noise_fraction": self.max_noise_fraction,
                "max_fragmentation": self.max_fragmentation,
                "minimum_cluster_count": self.minimum_cluster_count,
            },
            "kmeans_fallback": {
                "k_values": list(self.kmeans_k_values),
                "n_init": self.kmeans_n_init,
                "random_state": self.random_state,
            },
            "tie_break": "lexicographic_serialized_parameters",
        }

    @classmethod
    def from_clustering_config(cls, config: object) -> "HDBSCANSelectionConfig":
        """Adapt the package-level :class:`hailmary.config.ClusteringConfig`."""

        score = getattr(config, "score")
        k_min = int(getattr(config, "kmeans_k_min"))
        k_max = int(getattr(config, "kmeans_k_max"))
        return cls(
            min_cluster_sizes=tuple(getattr(config, "min_cluster_sizes")),
            min_samples_values=tuple(getattr(config, "min_samples")),
            cluster_selection_methods=tuple(getattr(config, "selection_methods")),
            silhouette_weight=float(getattr(score, "silhouette_weight")),
            persistence_weight=float(getattr(score, "persistence_weight")),
            coverage_weight=float(getattr(score, "coverage_weight")),
            fragmentation_weight=float(getattr(score, "fragmentation_weight")),
            max_noise_fraction=float(getattr(score, "max_noise_fraction")),
            max_fragmentation=float(getattr(score, "max_cluster_fraction")),
            minimum_cluster_count=int(getattr(score, "min_clusters")),
            kmeans_k_values=tuple(range(k_min, k_max + 1)),
            kmeans_n_init=int(getattr(config, "kmeans_n_init")),
            random_state=int(getattr(config, "random_state")),
        )


def _coerce_config(config: object | None) -> HDBSCANSelectionConfig:
    if config is None:
        return HDBSCANSelectionConfig()
    if isinstance(config, HDBSCANSelectionConfig):
        return config
    required = ("min_cluster_sizes", "min_samples", "selection_methods", "score")
    if all(hasattr(config, name) for name in required):
        return HDBSCANSelectionConfig.from_clustering_config(config)
    raise TypeError("config must be HDBSCANSelectionConfig or hailmary.config.ClusteringConfig")


@dataclass(frozen=True)
class CandidateMetrics:
    algorithm: str
    parameters: tuple[tuple[str, Any], ...]
    cluster_count: int
    clustered_fraction: float
    noise_fraction: float
    silhouette: float | None
    mean_persistence: float | None
    fragmentation: float
    composite_score: float | None
    accepted: bool
    rejection_reason: str | None = None

    @property
    def parameter_dict(self) -> dict[str, Any]:
        return dict(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "parameters": self.parameter_dict,
            "cluster_count": self.cluster_count,
            "clustered_fraction": self.clustered_fraction,
            "noise_fraction": self.noise_fraction,
            "silhouette": self.silhouette,
            "mean_persistence": self.mean_persistence,
            "fragmentation": self.fragmentation,
            "composite_score": self.composite_score,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateMetrics":
        return cls(
            algorithm=str(payload["algorithm"]),
            parameters=tuple(sorted(dict(payload["parameters"]).items())),
            cluster_count=int(payload["cluster_count"]),
            clustered_fraction=float(payload["clustered_fraction"]),
            noise_fraction=float(payload["noise_fraction"]),
            silhouette=None if payload.get("silhouette") is None else float(payload["silhouette"]),
            mean_persistence=None
            if payload.get("mean_persistence") is None
            else float(payload["mean_persistence"]),
            fragmentation=float(payload["fragmentation"]),
            composite_score=None
            if payload.get("composite_score") is None
            else float(payload["composite_score"]),
            accepted=bool(payload["accepted"]),
            rejection_reason=None if payload.get("rejection_reason") is None else str(payload["rejection_reason"]),
        )


@dataclass(frozen=True)
class ClusteringSelection:
    algorithm: str
    labels: np.ndarray
    probabilities: np.ndarray
    selected_parameters: tuple[tuple[str, Any], ...]
    candidates: tuple[CandidateMetrics, ...]
    library_versions: tuple[tuple[str, str], ...]
    used_fallback: bool

    def __post_init__(self) -> None:
        labels = np.array(self.labels, dtype=np.int64, order="C", copy=True)
        probabilities = readonly_float64(self.probabilities, name="membership probabilities", ndim=1)
        if labels.ndim != 1 or len(labels) != len(probabilities):
            raise ValueError("labels and probabilities must be equal-length vectors")
        if np.any(labels < -1):
            raise ValueError("cluster labels must be -1 or nonnegative")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("membership probabilities must be within [0, 1]")
        labels.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "selected_parameters", tuple(sorted(self.selected_parameters)))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "library_versions", tuple(sorted(self.library_versions)))

    @property
    def parameter_dict(self) -> dict[str, Any]:
        return dict(self.selected_parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "used_fallback": self.used_fallback,
            "selected_parameters": self.parameter_dict,
            "labels": self.labels.tolist(),
            "probabilities": self.probabilities.tolist(),
            "candidates": [item.to_dict() for item in self.candidates],
            "library_versions": dict(self.library_versions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClusteringSelection":
        return cls(
            algorithm=str(payload["algorithm"]),
            labels=payload["labels"],
            probabilities=payload["probabilities"],
            selected_parameters=tuple(sorted(dict(payload["selected_parameters"]).items())),
            candidates=tuple(CandidateMetrics.from_dict(item) for item in payload.get("candidates", [])),
            library_versions=tuple(sorted((str(k), str(v)) for k, v in dict(payload.get("library_versions", {})).items())),
            used_fallback=bool(payload.get("used_fallback", False)),
        )


def _canonical_labels(labels: np.ndarray, features: np.ndarray) -> np.ndarray:
    result = np.asarray(labels, dtype=np.int64).copy()
    cluster_keys: list[tuple[tuple[float, ...], int]] = []
    for old_label in sorted(int(item) for item in np.unique(result) if item >= 0):
        centroid = features[result == old_label].mean(axis=0)
        cluster_keys.append((tuple(float(item) for item in centroid), old_label))
    mapping = {old: new for new, (_, old) in enumerate(sorted(cluster_keys))}
    for old, new in mapping.items():
        result[labels == old] = new
    return result


def _safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
    mask = labels >= 0
    clustered_labels = labels[mask]
    cluster_count = len(np.unique(clustered_labels))
    if cluster_count < 2 or int(mask.sum()) <= cluster_count:
        return None
    return float(silhouette_score(features[mask], clustered_labels, metric="euclidean"))


def _parameters_key(parameters: Mapping[str, Any]) -> str:
    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _versions() -> tuple[tuple[str, str], ...]:
    versions: list[tuple[str, str]] = []
    for package in ("numpy", "scikit-learn", "hdbscan"):
        try:
            versions.append((package, metadata.version(package)))
        except metadata.PackageNotFoundError:
            versions.append((package, "unavailable"))
    return tuple(versions)


def _run_kmeans_fallback(features: np.ndarray, config: HDBSCANSelectionConfig, candidates: list[CandidateMetrics]) -> ClusteringSelection:
    sample_count = len(features)
    unique_count = len(np.unique(features, axis=0))
    if sample_count == 1 or unique_count == 1:
        labels = np.zeros(1, dtype=np.int64)
        if sample_count > 1:
            labels = np.zeros(sample_count, dtype=np.int64)
        selected_k = 1
    else:
        valid_k = [item for item in config.kmeans_k_values if 2 <= item < sample_count and item <= unique_count]
        if not valid_k:
            # With only two samples there is no defined silhouette comparison;
            # one deterministic cluster is preferable to two singleton templates.
            labels = np.zeros(sample_count, dtype=np.int64)
            selected_k = 1
        else:
            runs: list[tuple[float, int, np.ndarray]] = []
            for k in valid_k:
                model = KMeans(
                    n_clusters=k,
                    n_init=config.kmeans_n_init,
                    random_state=config.random_state,
                    algorithm="lloyd",
                )
                run_labels = _canonical_labels(model.fit_predict(features), features)
                silhouette = _safe_silhouette(features, run_labels)
                score = -1.0 if silhouette is None else silhouette
                actual_cluster_count = len(np.unique(run_labels))
                candidates.append(
                    CandidateMetrics(
                        algorithm="kmeans",
                        parameters=(("k", k), ("n_init", config.kmeans_n_init), ("random_state", config.random_state)),
                        cluster_count=actual_cluster_count,
                        clustered_fraction=1.0,
                        noise_fraction=0.0,
                        silhouette=silhouette,
                        mean_persistence=None,
                        fragmentation=actual_cluster_count / sample_count,
                        composite_score=score,
                        accepted=True,
                    )
                )
                runs.append((score, k, run_labels))
            _, selected_k, labels = sorted(runs, key=lambda item: (-item[0], item[1]))[0]
    return ClusteringSelection(
        algorithm="kmeans",
        labels=labels,
        probabilities=np.ones(sample_count, dtype=np.float64),
        selected_parameters=(("k", selected_k), ("n_init", config.kmeans_n_init), ("random_state", config.random_state)),
        candidates=tuple(candidates),
        library_versions=_versions(),
        used_fallback=True,
    )


def run_hdbscan_sweep(
    standardized_features: object,
    config: object | None = None,
) -> ClusteringSelection:
    """Fit the fixed scored sweep and select with deterministic tie-breaking."""

    features = readonly_float64(standardized_features, name="standardized_features", ndim=2)
    if len(features) == 0 or features.shape[1] == 0:
        raise ValueError("standardized_features must be a nonempty matrix")
    config = _coerce_config(config)
    candidates: list[CandidateMetrics] = []
    accepted_runs: list[tuple[CandidateMetrics, np.ndarray, np.ndarray]] = []
    try:
        import hdbscan  # pyright: ignore[reportMissingImports]
    except ImportError:
        return _run_kmeans_fallback(features, config, candidates)

    parameter_grid = sorted(
        product(config.min_cluster_sizes, config.min_samples_values, config.cluster_selection_methods),
        key=lambda item: _parameters_key(
            {"cluster_selection_method": item[2], "min_cluster_size": item[0], "min_samples": item[1]}
        ),
    )
    for min_cluster_size, min_samples, method in parameter_grid:
        if min_cluster_size > len(features):
            continue
        parameters = {
            "cluster_selection_method": method,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
        }
        try:
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=method,
                metric="euclidean",
                prediction_data=True,
                core_dist_n_jobs=1,
                approx_min_span_tree=False,
                gen_min_span_tree=False,
            )
            # Some hdbscan backends request a writable memoryview for small
            # cohorts.  The public Hailmary feature arrays remain immutable;
            # give the estimator its own branch-local working copy.
            raw_labels = np.asarray(
                model.fit_predict(np.array(features, dtype=np.float64, order="C", copy=True)),
                dtype=np.int64,
            )
            labels = _canonical_labels(raw_labels, features)
            probabilities = np.asarray(model.probabilities_, dtype=np.float64)
            cluster_count = len({int(item) for item in labels if item >= 0})
            clustered_count = int(np.count_nonzero(labels >= 0))
            clustered_fraction = clustered_count / len(labels)
            noise_fraction = 1.0 - clustered_fraction
            silhouette = _safe_silhouette(features, labels)
            persistence_values = np.asarray(getattr(model, "cluster_persistence_", []), dtype=np.float64)
            persistence = float(persistence_values.mean()) if len(persistence_values) else 0.0
            fragmentation = cluster_count / max(clustered_count, 1)
            score = (
                config.silhouette_weight * (0.0 if silhouette is None else silhouette)
                + config.persistence_weight * persistence
                + config.coverage_weight * clustered_fraction
                - config.fragmentation_weight * fragmentation
            )
            reasons: list[str] = []
            if cluster_count < config.minimum_cluster_count:
                reasons.append("too_few_clusters")
            if noise_fraction > config.max_noise_fraction:
                reasons.append("excessive_noise")
            if fragmentation > config.max_fragmentation:
                reasons.append("excessive_fragmentation")
            accepted = not reasons
            metric = CandidateMetrics(
                algorithm="hdbscan",
                parameters=tuple(sorted(parameters.items())),
                cluster_count=cluster_count,
                clustered_fraction=clustered_fraction,
                noise_fraction=noise_fraction,
                silhouette=silhouette,
                mean_persistence=persistence,
                fragmentation=fragmentation,
                composite_score=score,
                accepted=accepted,
                rejection_reason=None if accepted else ",".join(reasons),
            )
            candidates.append(metric)
            if accepted:
                accepted_runs.append((metric, labels, probabilities))
        except Exception as exc:  # invalid parameter/cohort combinations are recorded, not fatal
            candidates.append(
                CandidateMetrics(
                    algorithm="hdbscan",
                    parameters=tuple(sorted(parameters.items())),
                    cluster_count=0,
                    clustered_fraction=0.0,
                    noise_fraction=1.0,
                    silhouette=None,
                    mean_persistence=None,
                    fragmentation=0.0,
                    composite_score=None,
                    accepted=False,
                    rejection_reason=f"fit_failed:{type(exc).__name__}",
                )
            )

    if not accepted_runs:
        return _run_kmeans_fallback(features, config, candidates)
    selected_metric, selected_labels, selected_probabilities = sorted(
        accepted_runs,
        key=lambda item: (
            -float(item[0].composite_score if item[0].composite_score is not None else -np.inf),
            _parameters_key(item[0].parameter_dict),
        ),
    )[0]
    return ClusteringSelection(
        algorithm="hdbscan",
        labels=selected_labels,
        probabilities=selected_probabilities,
        selected_parameters=selected_metric.parameters,
        candidates=tuple(candidates),
        library_versions=_versions(),
        used_fallback=False,
    )


run_candidate_hdbscan = run_hdbscan_sweep
