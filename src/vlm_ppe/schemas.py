from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PPEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = "2026-04-01"
    operation: str = "arrival"
    runway: str | None = None
    manifest_path: Path = Path("data_manifest.json")
    output_root: Path = Path("data/artifacts/ppe/2026-04-01")
    n_resample: int = Field(default=100, ge=2)
    k_min: int = Field(default=1, ge=1)
    k_max: int = Field(default=8, ge=1)
    kmeans_n_init: int = Field(default=50, ge=1)
    kmeans_random_state: int = 17
    max_retries: int = Field(default=2, ge=0)
    max_k_expansion: int = Field(default=12, ge=1)
    vlm_model: str = "gemini-3.5-flash"
    min_track_points: int = Field(default=2, ge=2)

    @field_validator("operation")
    @classmethod
    def normalize_operation(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"arrival", "departure"}:
            raise ValueError("operation must be arrival or departure")
        return normalized

    @field_validator("k_max")
    @classmethod
    def validate_k_max(cls, value: int, info) -> int:
        k_min = info.data.get("k_min", 1)
        if value < k_min:
            raise ValueError("k_max must be greater than or equal to k_min")
        return value


class CoordinateSystem(BaseModel):
    type: Literal["local_azimuthal_equidistant"] = "local_azimuthal_equidistant"
    unit: Literal["NM"] = "NM"
    origin_lat: float
    origin_lon: float
    proj4: str


class KMetric(BaseModel):
    k: int
    inertia: float
    silhouette: float | None = None
    cluster_count_min: int
    cluster_count_max: int
    cluster_count_mean: float


class EvidenceImage(BaseModel):
    kind: str
    path: str
    caption: str


class ClusterReview(BaseModel):
    chosen_k: int
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: list[str] = Field(default_factory=list)
    rejected_alternatives: list[str] = Field(default_factory=list)
    clusters_to_recheck: list[int] = Field(default_factory=list)
    retry_requested: bool = False
    requested_k_max: int | None = Field(default=None, ge=1)
    suggested_action: Literal["accept", "retry", "human_review"] = "accept"

    @field_validator("rationale", "rejected_alternatives")
    @classmethod
    def clean_text_list(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class ClusterMedoid(BaseModel):
    cluster_id: int
    medoid_track_id: str
    n_tracks: int
    mean_distance_nm: float
    max_distance_nm: float
    template_points: list[tuple[float, float]]


class PPEState(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    run_dir: str
    config: dict
    retry_count: int = 0
    k_max_current: int
    chosen_k_override: int | None = None
    coordinate_system: dict | None = None
    tracks_path: str | None = None
    track_index_path: str | None = None
    resampled_tracks_path: str | None = None
    features_path: str | None = None
    feature_metadata_path: str | None = None
    k_metrics_path: str | None = None
    clustering_dir: str | None = None
    evidence_images: list[dict] = Field(default_factory=list)
    vlm_reviews: list[dict] = Field(default_factory=list)
    chosen_k: int | None = None
    cluster_assignments_path: str | None = None
    medoids_path: str | None = None
    medoid_summary_path: str | None = None
    medoid_report_path: str | None = None
    status: str = "initialized"
    errors: list[dict] = Field(default_factory=list)


class GraphEvent(BaseModel):
    node: str
    status: Literal["started", "completed", "failed"]
    message: str | None = None
    payload: dict = Field(default_factory=dict)
