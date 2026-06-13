from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    subcluster_review_enabled: bool = True
    subcluster_min_tracks: int = Field(default=4, ge=2)
    subcluster_max_polygons: int = Field(default=4, ge=1)
    subcluster_max_reviews: int = Field(default=64, ge=1)
    vlm_model: str = "openai/gpt-5.5"
    vlm_reasoning_effort: Literal["low", "medium", "high"] = "medium"
    window_review_max_attempts: int = Field(default=3, ge=1, le=10)
    window_review_max_patterns: int = Field(default=8, ge=1, le=20)
    min_track_points: int = Field(default=2, ge=2)
    track_filter_center_lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    track_filter_center_lon: float | None = Field(default=None, ge=-180.0, le=180.0)
    track_filter_radius_nm: float | None = Field(default=None, gt=0.0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_to_console: bool = True

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_subcluster_config(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "subcluster_k_max" in payload:
            if "subcluster_max_polygons" in payload:
                raise ValueError("use only one of subcluster_k_max or subcluster_max_polygons")
            payload["subcluster_max_polygons"] = payload.pop("subcluster_k_max")
        return payload

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

    @model_validator(mode="after")
    def validate_track_filter(self) -> Self:
        has_center_lat = self.track_filter_center_lat is not None
        has_center_lon = self.track_filter_center_lon is not None
        if has_center_lat != has_center_lon:
            raise ValueError("track_filter_center_lat and track_filter_center_lon must be provided together")
        if self.track_filter_radius_nm is not None and not (has_center_lat and has_center_lon):
            raise ValueError("track_filter_center_lat/lon are required when track_filter_radius_nm is set")
        return self


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

    @field_validator("rationale", "rejected_alternatives", mode="before")
    @classmethod
    def coerce_text_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("rationale", "rejected_alternatives")
    @classmethod
    def clean_text_list(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class SubclusterPolygon(BaseModel):
    subcluster_id: int = Field(ge=1)
    label: str = ""
    polygon: list[tuple[float, float]] = Field(min_length=3)
    rationale: str = ""

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        return value.strip()

    @field_validator("rationale")
    @classmethod
    def clean_rationale(cls, value: str) -> str:
        return value.strip()

    @field_validator("polygon", mode="before")
    @classmethod
    def coerce_polygon_payload(cls, value: object) -> object:
        if isinstance(value, dict):
            for key in ("polygon", "vertices", "points", "coordinates"):
                if key in value:
                    return value[key]
        return value


class SubclusterReview(BaseModel):
    subcluster_count: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: list[str] = Field(default_factory=list)
    subclusters: list[SubclusterPolygon] = Field(default_factory=list)
    uncaptured_tracks_policy: Literal["keep_as_residual", "human_review"] = "keep_as_residual"
    suggested_action: Literal["accept", "human_review"] = "accept"

    @model_validator(mode="before")
    @classmethod
    def coerce_legacy_subcluster_payload(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        payload = dict(value)
        if "subcluster_count" not in payload:
            for key in ("n_subclusters", "num_subclusters", "N"):
                if key in payload:
                    payload["subcluster_count"] = payload[key]
                    break

        direct_subclusters: list[dict[str, object]] = []
        for key in list(payload.keys()):
            normalized = str(key).strip().lower()
            if not normalized.startswith("subcluster "):
                continue
            polygon = payload.pop(key)
            suffix = normalized.removeprefix("subcluster ").strip()
            subcluster_id = int(suffix) if suffix.isdigit() else len(direct_subclusters) + 1
            direct_subclusters.append(
                {
                    "subcluster_id": subcluster_id,
                    "label": str(key),
                    "polygon": polygon,
                }
            )
        if direct_subclusters and "subclusters" not in payload:
            payload["subclusters"] = direct_subclusters
        if "subcluster_count" not in payload and "subclusters" in payload and isinstance(payload["subclusters"], list):
            payload["subcluster_count"] = max(1, len(payload["subclusters"]))
        return payload

    @field_validator("rationale", mode="before")
    @classmethod
    def coerce_rationale(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("rationale")
    @classmethod
    def clean_rationale_list(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def validate_subcluster_count(self) -> Self:
        if self.subcluster_count == 1 and self.subclusters:
            raise ValueError("subclusters must be empty when subcluster_count is 1")
        if self.subcluster_count > 1 and not self.subclusters:
            raise ValueError("subclusters are required when subcluster_count is greater than 1")
        if self.subclusters and len(self.subclusters) > self.subcluster_count:
            raise ValueError("subclusters cannot contain more entries than subcluster_count")
        seen: set[int] = set()
        for item in self.subclusters:
            if item.subcluster_id in seen:
                raise ValueError(f"duplicate subcluster_id={item.subcluster_id}")
            seen.add(item.subcluster_id)
        return self


class ClusterMedoid(BaseModel):
    cluster_id: int
    medoid_track_id: str
    n_tracks: int
    mean_distance_nm: float
    max_distance_nm: float
    template_points: list[tuple[float, float]]


InterventionClass = Literal["no_stretch", "dogleg", "trombone", "PMS", "other"]


class InterventionWindow(BaseModel):
    cluster_id: int
    window_id: str
    class_name: InterventionClass
    confidence: float = Field(ge=0.0, le=1.0)
    visual_reason: str = ""
    start_station_index: int = Field(ge=0)
    end_station_index: int = Field(ge=0)
    start_s_fraction: float = Field(ge=0.0, le=1.0)
    end_s_fraction: float = Field(ge=0.0, le=1.0)
    start_s_nm: float = Field(ge=0.0)
    end_s_nm: float = Field(ge=0.0)
    length_nm: float = Field(ge=0.0)
    peak_residual_energy_nm2: float = Field(ge=0.0)
    peak_heading_dispersion: float = Field(ge=0.0, le=1.0)
    track_ids: list[str] = Field(default_factory=list)

    @field_validator("visual_reason")
    @classmethod
    def clean_visual_reason(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_window_bounds(self) -> Self:
        if self.end_station_index < self.start_station_index:
            raise ValueError("end_station_index must be greater than or equal to start_station_index")
        if self.end_s_fraction < self.start_s_fraction:
            raise ValueError("end_s_fraction must be greater than or equal to start_s_fraction")
        if self.end_s_nm < self.start_s_nm:
            raise ValueError("end_s_nm must be greater than or equal to start_s_nm")
        return self


class WindowProposal(BaseModel):
    window_id: str
    start_station_index: int = Field(ge=0)
    end_station_index: int = Field(ge=0)
    class_name: InterventionClass
    confidence: float = Field(ge=0.0, le=1.0)
    visual_reason: str = ""

    @field_validator("visual_reason")
    @classmethod
    def clean_visual_reason(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_proposal_bounds(self) -> Self:
        if self.end_station_index < self.start_station_index:
            raise ValueError("end_station_index must be greater than or equal to start_station_index")
        return self


class WindowReview(BaseModel):
    cluster_id: int
    windows: list[WindowProposal] = Field(default_factory=list)
    outlier_notes: list[str] = Field(default_factory=list)
    all_patterns_identified: bool = False
    suggested_action: Literal["accept", "revise", "human_review"] = "accept"

    @field_validator("outlier_notes", mode="before")
    @classmethod
    def coerce_outlier_notes(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("outlier_notes")
    @classmethod
    def clean_outlier_notes(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class WindowPatternCountReview(BaseModel):
    cluster_id: int
    pattern_count: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: list[str] = Field(default_factory=list)
    outlier_notes: list[str] = Field(default_factory=list)
    suggested_action: Literal["accept", "human_review"] = "accept"

    @field_validator("rationale", "outlier_notes", mode="before")
    @classmethod
    def coerce_text_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("rationale", "outlier_notes")
    @classmethod
    def clean_text_list(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class PPEState(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    run_dir: str
    audit_log_path: str | None = None
    graph_events_path: str | None = None
    vlm_interactions_path: str | None = None
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
    subcluster_tree_path: str | None = None
    subcluster_reviews: list[dict] = Field(default_factory=list)
    subcluster_review_paths: list[str] = Field(default_factory=list)
    medoids_path: str | None = None
    medoid_summary_path: str | None = None
    medoid_report_path: str | None = None
    residual_profiles_path: str | None = None
    intervention_windows_path: str | None = None
    intervention_windows: list[dict] = Field(default_factory=list)
    window_evidence_images: list[dict] = Field(default_factory=list)
    window_pattern_count_reviews: list[dict] = Field(default_factory=list)
    window_pattern_count_review_paths: list[str] = Field(default_factory=list)
    window_reviews: list[dict] = Field(default_factory=list)
    window_review_paths: list[str] = Field(default_factory=list)
    status: str = "initialized"
    errors: list[dict] = Field(default_factory=list)


class GraphEvent(BaseModel):
    timestamp_utc: str | None = None
    node: str
    status: Literal["started", "completed", "failed"]
    message: str | None = None
    payload: dict = Field(default_factory=dict)
