"""Frozen configuration for the Hailmary milestone.

Defaults make the design document's previously open choices explicit and are
serialized into artifacts, so experiments never depend on hidden constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ConfigurationError

M_PER_NM = 1_852.0
MPS_PER_KNOT = 0.514444


def _positive(values: tuple[float, ...], *, name: str) -> None:
    if not values or any(value <= 0.0 for value in values):
        raise ConfigurationError(f"{name} must contain positive values")


@dataclass(frozen=True)
class HDBSCANScoreConfig:
    silhouette_weight: float = 0.35
    persistence_weight: float = 0.25
    coverage_weight: float = 0.25
    fragmentation_weight: float = 0.15
    max_noise_fraction: float = 0.60
    max_cluster_fraction: float = 0.25
    min_clusters: int = 2

    def __post_init__(self) -> None:
        weights = (
            self.silhouette_weight,
            self.persistence_weight,
            self.coverage_weight,
            self.fragmentation_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ConfigurationError("HDBSCAN score weights must be non-negative with a positive sum")
        if not 0.0 <= self.max_noise_fraction < 1.0:
            raise ConfigurationError("max_noise_fraction must be in [0, 1)")
        if not 0.0 < self.max_cluster_fraction <= 1.0:
            raise ConfigurationError("max_cluster_fraction must be in (0, 1]")
        if self.min_clusters < 1:
            raise ConfigurationError("min_clusters must be positive")


@dataclass(frozen=True)
class ClusteringConfig:
    n_resample: int = 128
    terminal_radius_nm: float = 50.0
    threshold_capture_radius_nm: float = 2.0
    min_cluster_sizes: tuple[int, ...] = (4, 8, 12, 16, 24)
    min_samples: tuple[int | None, ...] = (None, 3, 5, 8)
    selection_methods: tuple[str, ...] = ("eom", "leaf")
    score: HDBSCANScoreConfig = field(default_factory=HDBSCANScoreConfig)
    kmeans_k_min: int = 2
    kmeans_k_max: int = 8
    kmeans_n_init: int = 50
    random_state: int = 17
    acceptance_quantile: float = 0.99
    acceptance_multiplier: float = 1.25

    def __post_init__(self) -> None:
        if self.n_resample < 2:
            raise ConfigurationError("n_resample must be at least 2")
        if self.terminal_radius_nm <= 0.0 or self.threshold_capture_radius_nm <= 0.0:
            raise ConfigurationError("terminal/capture radii must be positive")
        if not self.min_cluster_sizes or any(value < 2 for value in self.min_cluster_sizes):
            raise ConfigurationError("min_cluster_sizes must contain values >= 2")
        if any(value is not None and value < 1 for value in self.min_samples):
            raise ConfigurationError("min_samples must contain positive integers or None")
        if not self.selection_methods or any(value not in {"eom", "leaf"} for value in self.selection_methods):
            raise ConfigurationError("selection_methods must contain eom and/or leaf")
        if self.kmeans_k_min < 1 or self.kmeans_k_max < self.kmeans_k_min:
            raise ConfigurationError("invalid KMeans fallback range")
        if not 0.5 <= self.acceptance_quantile < 1.0:
            raise ConfigurationError("acceptance_quantile must be in [0.5, 1)")
        if self.acceptance_multiplier <= 0.0:
            raise ConfigurationError("acceptance_multiplier must be positive")


@dataclass(frozen=True)
class TemplateConfig:
    schema_version: str = "hailmary.template.v1"
    speed_action_count: int = 16
    path_stretch_count: int = 8
    commitment_gate_nm: float = 4.0
    speed_reduction_kts: tuple[float, ...] = (10.0, 15.0, 20.0)
    speed_band_names: tuple[str, ...] = ("light", "medium", "heavy")
    min_effective_reduction_kts: float = 2.0
    max_speed_actions: int = 2
    max_path_stretches: int = 1
    max_historical_clamp_kts: float = 5.0
    max_clamped_fraction: float = 0.05
    max_timing_error_fraction: float = 0.10
    max_timing_error_s: float = 30.0
    aircraft_typecode: str = "A320"
    payload_kg: float = 12_000.0
    zero_wind: bool = True

    def __post_init__(self) -> None:
        if self.speed_action_count != 16 or self.path_stretch_count != 8:
            raise ConfigurationError("version 1 requires exactly 16 speed and 8 path-stretch locations")
        if self.commitment_gate_nm <= 0.0:
            raise ConfigurationError("commitment_gate_nm must be positive")
        if len(self.speed_reduction_kts) != len(self.speed_band_names):
            raise ConfigurationError("speed reduction values and names must have equal length")
        _positive(self.speed_reduction_kts, name="speed_reduction_kts")
        if tuple(sorted(self.speed_reduction_kts)) != self.speed_reduction_kts:
            raise ConfigurationError("speed reductions must be ordered light to heavy")
        if self.max_speed_actions < 1 or self.max_path_stretches < 1:
            raise ConfigurationError("intervention limits must be positive")
        if not 0.0 <= self.max_clamped_fraction <= 1.0:
            raise ConfigurationError("max_clamped_fraction must be in [0, 1]")


@dataclass(frozen=True)
class StretchConfig:
    variant_names: tuple[str, ...] = ("short", "medium", "long")
    rejoin_span_nm: tuple[float, ...] = (10.0, 14.0, 18.0)
    added_distance_nm: tuple[float, ...] = (2.0, 5.0, 9.0)
    added_distance_tolerance_nm: float = 0.25
    candidate_azimuth_count: int = 24
    max_turn_deg: float = 110.0
    minimum_medoid_clearance_nm: float = 0.0
    selector: str = "semi_local_outcome"
    geometry_only_ablation_selector: str = "geometry_clearance"
    new_conflict_penalty: float = 2.0
    conflict_duration_scale_s: float = 60.0

    def __post_init__(self) -> None:
        if not (len(self.variant_names) == len(self.rejoin_span_nm) == len(self.added_distance_nm) == 3):
            raise ConfigurationError("stretch configuration requires short, medium, and long triples")
        _positive(self.rejoin_span_nm, name="rejoin_span_nm")
        _positive(self.added_distance_nm, name="added_distance_nm")
        if self.added_distance_tolerance_nm <= 0.0:
            raise ConfigurationError("added_distance_tolerance_nm must be positive")
        if self.candidate_azimuth_count < 4:
            raise ConfigurationError("candidate_azimuth_count must be at least 4")
        if not 0.0 < self.max_turn_deg < 180.0:
            raise ConfigurationError("max_turn_deg must be between 0 and 180")
        if self.new_conflict_penalty < 0.0 or self.conflict_duration_scale_s <= 0.0:
            raise ConfigurationError(
                "stretch conflict penalty must be nonnegative and duration scale positive"
            )


@dataclass(frozen=True)
class ScenarioConfig:
    separation_s: float = 90.0
    pressure_window_s: float = 600.0
    feature_correlation_limit: float = 0.30
    exogenous_coupling: str = "materialized_events"
    intervention_ordering: str = "any_chronological_order"
    registered_factor_pairs: tuple[tuple[str, str], ...] = (
        ("error_magnitude", "time_to_final"),
        ("error_magnitude", "pressure"),
        ("commitment", "pressure"),
        ("commitment", "error_magnitude"),
    )

    def __post_init__(self) -> None:
        if self.separation_s <= 0.0 or self.pressure_window_s <= 0.0:
            raise ConfigurationError("separation and pressure window must be positive")
        if not 0.0 <= self.feature_correlation_limit < 1.0:
            raise ConfigurationError("feature_correlation_limit must be in [0, 1)")
        if self.exogenous_coupling != "materialized_events":
            raise ConfigurationError("version 1 requires materialized exogenous events")
        if self.intervention_ordering != "any_chronological_order":
            raise ConfigurationError("version 1 allows interventions in any chronological order")


@dataclass(frozen=True)
class FeatureConfig:
    schema_version: str = "hailmary.features.leader_follower.v1"
    ratio_capacity_floor_s: float = 1.0
    ratio_clip_max: float = 10.0
    commitment_time_scale_s: float = 1_200.0
    time_weight: float = 1.0 / 3.0
    freedom_weight: float = 1.0 / 3.0
    gate_weight: float = 1.0 / 3.0
    station_freedom_weight: float = 0.5
    budget_freedom_weight: float = 0.5


@dataclass(frozen=True)
class OutcomeConfig:
    trailer_count: int = 3
    pair_weight: float = 1.0
    propagation_weight: float = 1.0
    intervention_weight: float = 0.3
    throughput_weight: float = 0.1
    speed_magnitude_normalizer_kts: float = 20.0
    stretch_magnitude_normalizer_nm: float = 9.0
    throughput_gap_normalizer_s: float = 90.0

    def __post_init__(self) -> None:
        if self.trailer_count < 0:
            raise ConfigurationError("trailer_count cannot be negative")
        if any(
            value < 0.0
            for value in (
                self.pair_weight,
                self.propagation_weight,
                self.intervention_weight,
                self.throughput_weight,
            )
        ):
            raise ConfigurationError("outcome weights cannot be negative")
        _positive(
            (
                self.speed_magnitude_normalizer_kts,
                self.stretch_magnitude_normalizer_nm,
                self.throughput_gap_normalizer_s,
            ),
            name="outcome normalizers",
        )


@dataclass(frozen=True)
class HailmaryConfig:
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    stretch: StretchConfig = field(default_factory=StretchConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
