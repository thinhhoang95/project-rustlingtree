"""Frozen configuration for the Hailmary milestone.

Defaults make the design document's previously open choices explicit and are
serialized into artifacts, so experiments never depend on hidden constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .errors import ConfigurationError

M_PER_NM = 1_852.0
MPS_PER_KNOT = 0.514444


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _positive(values: tuple[float, ...], *, name: str) -> None:
    if not values or any(not _finite_number(value) or value <= 0.0 for value in values):
        raise ConfigurationError(f"{name} must contain positive values")


@dataclass(frozen=True)
class HDBSCANScoreConfig:
    """Global runway-independent HDBSCAN candidate policy.

    Coverage receives more weight than mean persistence because a small but
    repeatable approach stream must not be discarded merely to make a larger
    parent cluster more persistent. Physical dispersion and the robust entry-
    bearing limit guard against the opposite failure: merging distinct route
    families into one broad cluster. These defaults were validated together
    over every reconstructable KDFW runway; there is intentionally no runway-
    specific parameter table.
    """

    silhouette_weight: float = 0.35
    persistence_weight: float = 0.15
    coverage_weight: float = 0.35
    fragmentation_weight: float = 0.15
    dispersion_weight: float = 0.10
    max_noise_fraction: float = 0.60
    max_cluster_fraction: float = 0.25
    min_clusters: int = 2
    max_entry_bearing_span_deg: float = 60.0

    def __post_init__(self) -> None:
        weights = (
            self.silhouette_weight,
            self.persistence_weight,
            self.coverage_weight,
            self.fragmentation_weight,
            self.dispersion_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ConfigurationError(
                "HDBSCAN score weights must be non-negative with a positive sum"
            )
        if not 0.0 <= self.max_noise_fraction < 1.0:
            raise ConfigurationError("max_noise_fraction must be in [0, 1)")
        if not 0.0 < self.max_cluster_fraction <= 1.0:
            raise ConfigurationError("max_cluster_fraction must be in (0, 1]")
        if self.min_clusters < 1:
            raise ConfigurationError("min_clusters must be positive")
        if not 0.0 < self.max_entry_bearing_span_deg <= 360.0:
            raise ConfigurationError("max_entry_bearing_span_deg must be in (0, 360]")


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
        if not self.min_cluster_sizes or any(
            value < 2 for value in self.min_cluster_sizes
        ):
            raise ConfigurationError("min_cluster_sizes must contain values >= 2")
        if any(value is not None and value < 1 for value in self.min_samples):
            raise ConfigurationError(
                "min_samples must contain positive integers or None"
            )
        if not self.selection_methods or any(
            value not in {"eom", "leaf"} for value in self.selection_methods
        ):
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
        if self.schema_version != "hailmary.template.v1":
            raise ConfigurationError("unsupported template schema version")
        if (
            isinstance(self.speed_action_count, bool)
            or not isinstance(self.speed_action_count, int)
            or isinstance(self.path_stretch_count, bool)
            or not isinstance(self.path_stretch_count, int)
            or self.speed_action_count != 16
            or self.path_stretch_count != 8
        ):
            raise ConfigurationError(
                "version 1 requires exactly 16 speed and 8 path-stretch locations"
            )
        if (
            not _finite_number(self.commitment_gate_nm)
            or self.commitment_gate_nm <= 0.0
        ):
            raise ConfigurationError("commitment_gate_nm must be positive")
        if self.speed_band_names != ("light", "medium", "heavy"):
            raise ConfigurationError(
                "version 1 requires speed bands light, medium, and heavy"
            )
        if len(self.speed_reduction_kts) != len(self.speed_band_names):
            raise ConfigurationError(
                "speed reduction values and names must have equal length"
            )
        _positive(self.speed_reduction_kts, name="speed_reduction_kts")
        if tuple(sorted(self.speed_reduction_kts)) != self.speed_reduction_kts:
            raise ConfigurationError("speed reductions must be ordered light to heavy")
        if (
            not _finite_number(self.min_effective_reduction_kts)
            or self.min_effective_reduction_kts <= 0.0
        ):
            raise ConfigurationError("min_effective_reduction_kts must be positive")

        if (
            isinstance(self.max_speed_actions, bool)
            or not isinstance(self.max_speed_actions, int)
            or isinstance(self.max_path_stretches, bool)
            or not isinstance(self.max_path_stretches, int)
            or self.max_speed_actions < 1
            or self.max_path_stretches < 1
        ):
            raise ConfigurationError("intervention limits must be positive")
        if (
            not _finite_number(self.max_clamped_fraction)
            or not 0.0 <= self.max_clamped_fraction <= 1.0
        ):
            raise ConfigurationError("max_clamped_fraction must be in [0, 1]")
        if (
            not _finite_number(self.max_historical_clamp_kts)
            or self.max_historical_clamp_kts < 0.0
        ):
            raise ConfigurationError(
                "max_historical_clamp_kts must be finite and non-negative"
            )
        if (
            not _finite_number(self.max_timing_error_fraction)
            or not 0.0 <= self.max_timing_error_fraction <= 1.0
        ):
            raise ConfigurationError("max_timing_error_fraction must be in [0, 1]")
        if not _finite_number(self.max_timing_error_s) or self.max_timing_error_s < 0.0:
            raise ConfigurationError(
                "max_timing_error_s must be finite and non-negative"
            )
        if self.aircraft_typecode != "A320":
            raise ConfigurationError("version 1 requires aircraft_typecode A320")
        if not _finite_number(self.payload_kg) or self.payload_kg < 0.0:
            raise ConfigurationError("payload_kg must be finite and non-negative")
        if self.zero_wind is not True:
            raise ConfigurationError("version 1 requires zero_wind=true")


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
        if self.variant_names != ("short", "medium", "long"):
            raise ConfigurationError(
                "stretch variant names must be short, medium, and long"
            )
        if self.selector != "semi_local_outcome":
            raise ConfigurationError("version 1 requires semi_local_outcome selector")
        if self.geometry_only_ablation_selector != "geometry_clearance":
            raise ConfigurationError("version 1 requires geometry_clearance ablation")
        if not (
            len(self.variant_names)
            == len(self.rejoin_span_nm)
            == len(self.added_distance_nm)
            == 3
        ):
            raise ConfigurationError(
                "stretch configuration requires short, medium, and long triples"
            )
        _positive(self.rejoin_span_nm, name="rejoin_span_nm")
        _positive(self.added_distance_nm, name="added_distance_nm")
        if (
            not _finite_number(self.added_distance_tolerance_nm)
            or self.added_distance_tolerance_nm <= 0.0
        ):
            raise ConfigurationError("added_distance_tolerance_nm must be positive")
        if (
            isinstance(self.candidate_azimuth_count, bool)
            or not isinstance(self.candidate_azimuth_count, int)
            or self.candidate_azimuth_count < 4
        ):
            raise ConfigurationError("candidate_azimuth_count must be at least 4")
        if not _finite_number(self.max_turn_deg) or not 0.0 < self.max_turn_deg < 180.0:
            raise ConfigurationError("max_turn_deg must be between 0 and 180")
        if (
            not _finite_number(self.minimum_medoid_clearance_nm)
            or self.minimum_medoid_clearance_nm < 0.0
        ):
            raise ConfigurationError("minimum_medoid_clearance_nm must be non-negative")
        if (
            not _finite_number(self.new_conflict_penalty)
            or not _finite_number(self.conflict_duration_scale_s)
            or self.new_conflict_penalty < 0.0
            or self.conflict_duration_scale_s <= 0.0
        ):
            raise ConfigurationError(
                "stretch conflict penalty must be nonnegative and duration scale positive"
            )


@dataclass(frozen=True)
class ScenarioConfig:
    separation_s: float = 90.0
    pressure_window_s: float = 600.0
    exogenous_coupling: str = "materialized_events"
    intervention_ordering: str = "any_chronological_order"

    def __post_init__(self) -> None:
        if (
            not _finite_number(self.separation_s)
            or not _finite_number(self.pressure_window_s)
            or self.separation_s <= 0.0
            or self.pressure_window_s <= 0.0
        ):
            raise ConfigurationError("separation and pressure window must be positive")
        if self.exogenous_coupling != "materialized_events":
            raise ConfigurationError("version 1 requires materialized exogenous events")
        if self.intervention_ordering != "any_chronological_order":
            raise ConfigurationError(
                "version 1 allows interventions in any chronological order"
            )


@dataclass(frozen=True)
class FeatureConfig:
    schema_version: str = "hailmary.features.leader_follower.v3"
    ratio_capacity_floor_s: float = 1.0
    ratio_clip_max: float = 10.0
    commitment_time_scale_s: float = 1_200.0
    time_weight: float = 1.0 / 3.0
    freedom_weight: float = 1.0 / 3.0
    gate_weight: float = 1.0 / 3.0
    station_freedom_weight: float = 0.5
    budget_freedom_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.schema_version != "hailmary.features.leader_follower.v3":
            raise ConfigurationError(
                "unsupported leader-follower feature schema version"
            )
        for name, value in (
            ("ratio_capacity_floor_s", self.ratio_capacity_floor_s),
            ("ratio_clip_max", self.ratio_clip_max),
            ("commitment_time_scale_s", self.commitment_time_scale_s),
        ):
            if not _finite_number(value) or value <= 0.0:
                raise ConfigurationError(f"{name} must be finite and positive")
        weights = {
            "time_weight": self.time_weight,
            "freedom_weight": self.freedom_weight,
            "gate_weight": self.gate_weight,
            "station_freedom_weight": self.station_freedom_weight,
            "budget_freedom_weight": self.budget_freedom_weight,
        }
        for name, value in weights.items():
            if not _finite_number(value) or value < 0.0:
                raise ConfigurationError(f"{name} must be finite and non-negative")
        if self.time_weight + self.freedom_weight + self.gate_weight <= 0.0:
            raise ConfigurationError(
                "commitment feature weights require a positive sum"
            )
        if self.station_freedom_weight + self.budget_freedom_weight <= 0.0:
            raise ConfigurationError("freedom feature weights require a positive sum")


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
        if (
            isinstance(self.trailer_count, bool)
            or not isinstance(self.trailer_count, int)
            or self.trailer_count < 0
        ):
            raise ConfigurationError("trailer_count must be a non-negative integer")
        if any(
            not _finite_number(value) or value < 0.0
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
class LearningConfig:
    """Reproducible Phase-0 defaults for causal rule learning.

    These values intentionally live beside the other experiment configuration
    instead of being hidden in the future trainer implementation. Later
    phases may tune them, but every run starts from one complete, serializable
    set of choices.
    """

    population_limit: int = 1_000
    ga_interval: int = 50
    ga_min_experience: int = 20
    certification_interval: int = 500
    action_min_noop_samples: int = 30
    veto_min_rival_samples: int = 60
    action_lcb_z: float = 1.96
    veto_lcb_z: float = 1.96
    contender_min_samples: int = 2
    contender_lcb_z: float = 1.96
    base_exploration_rate: float = 0.05
    exploration_coverage_floor: int = 200
    covering_width_min_fraction: float = 0.05
    covering_width_max_fraction: float = 0.25
    mutation_probability: float = 0.04
    crossover_probability: float = 0.80
    variance_floor: float = 1.0e-6
    offspring_evidence_discount: float = 0.50
    young_rule_protection_epochs: int = 100
    random_seed: int = 17

    def __post_init__(self) -> None:
        positive_integers = {
            "population_limit": self.population_limit,
            "ga_interval": self.ga_interval,
            "ga_min_experience": self.ga_min_experience,
            "certification_interval": self.certification_interval,
            "action_min_noop_samples": self.action_min_noop_samples,
            "veto_min_rival_samples": self.veto_min_rival_samples,
            "contender_min_samples": self.contender_min_samples,
            "exploration_coverage_floor": self.exploration_coverage_floor,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"{name} must be positive")
        if self.contender_min_samples < 2:
            raise ConfigurationError("contender_min_samples must be at least two")
        if self.ga_min_experience < 2:
            raise ConfigurationError("ga_min_experience must be at least two")
        if (
            isinstance(self.young_rule_protection_epochs, bool)
            or not isinstance(self.young_rule_protection_epochs, int)
            or self.young_rule_protection_epochs < 0
        ):
            raise ConfigurationError("young_rule_protection_epochs cannot be negative")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ConfigurationError("random_seed cannot be negative")

        positive_floats = {
            "action_lcb_z": self.action_lcb_z,
            "veto_lcb_z": self.veto_lcb_z,
            "contender_lcb_z": self.contender_lcb_z,
            "variance_floor": self.variance_floor,
        }
        for name, value in positive_floats.items():
            if not _finite_number(value) or value <= 0.0:
                raise ConfigurationError(f"{name} must be positive")

        probabilities = {
            "base_exploration_rate": self.base_exploration_rate,
            "mutation_probability": self.mutation_probability,
            "crossover_probability": self.crossover_probability,
        }
        for name, value in probabilities.items():
            if not _finite_number(value) or not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} must be in [0, 1]")
        if (
            not _finite_number(self.offspring_evidence_discount)
            or not 0.0 < self.offspring_evidence_discount <= 1.0
        ):
            raise ConfigurationError("offspring_evidence_discount must be in (0, 1]")
        if (
            not _finite_number(self.covering_width_min_fraction)
            or not 0.0 < self.covering_width_min_fraction <= 1.0
        ):
            raise ConfigurationError("covering_width_min_fraction must be in (0, 1]")
        if (
            not _finite_number(self.covering_width_max_fraction)
            or not self.covering_width_min_fraction
            <= self.covering_width_max_fraction
            <= 1.0
        ):
            raise ConfigurationError(
                "covering width fractions must be ordered and no greater than one"
            )


@dataclass(frozen=True)
class HailmaryConfig:
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    stretch: StretchConfig = field(default_factory=StretchConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
