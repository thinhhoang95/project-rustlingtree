"""Derive an accurate role-bound vector from canonical query primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from hailmary.config import FeatureConfig, M_PER_NM, MPS_PER_KNOT, ScenarioConfig, TemplateConfig
from hailmary.evaluation.spacing import compute_spacing
from hailmary.features.anchors import LeaderFollowerAnchor
from hailmary.features.minimum_time import ReachabilityMap
from hailmary.features.schema import FeatureSchema, FeatureVector, leader_follower_feature_schema


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass(frozen=True)
class CommitmentComponents:
    time_component: float
    remaining_station_fraction: float
    remaining_intervention_budget_fraction: float
    freedom_remaining: float
    freedom_component: float
    gate_component: float
    commitment_fraction: float


def commitment_components(
    *,
    time_to_threshold_s: float,
    remaining_station_fraction: float,
    remaining_intervention_budget_fraction: float,
    intercept_or_final_gate_flag: float,
    config: FeatureConfig | None = None,
) -> CommitmentComponents:
    cfg = FeatureConfig() if config is None else config
    if cfg.commitment_time_scale_s <= 0.0:
        raise ValueError("commitment_time_scale_s must be positive")
    station_fraction = _clip01(remaining_station_fraction)
    budget_fraction = _clip01(remaining_intervention_budget_fraction)
    gate_component = _clip01(intercept_or_final_gate_flag)
    time_component = _clip01(1.0 - float(time_to_threshold_s) / cfg.commitment_time_scale_s)
    freedom_remaining = float(
        cfg.station_freedom_weight * station_fraction
        + cfg.budget_freedom_weight * budget_fraction
    )
    freedom_component = _clip01(1.0 - freedom_remaining)
    commitment = float(
        cfg.time_weight * time_component
        + cfg.freedom_weight * freedom_component
        + cfg.gate_weight * gate_component
    )
    return CommitmentComponents(
        time_component=time_component,
        remaining_station_fraction=station_fraction,
        remaining_intervention_budget_fraction=budget_fraction,
        freedom_remaining=freedom_remaining,
        freedom_component=freedom_component,
        gate_component=gate_component,
        commitment_fraction=_clip01(commitment),
    )


def pressure_in_half_open_window(
    eta_values_s: Iterable[float],
    *,
    sim_time_s: float,
    window_s: float,
    required_interval_s: float,
) -> tuple[int, float, float]:
    """Return ``(count, capacity_slots, ratio)`` for ``[t, t + window)``."""

    now = float(sim_time_s)
    window = float(window_s)
    required = float(required_interval_s)
    if not all(np.isfinite(value) for value in (now, window, required)):
        raise ValueError("pressure inputs must be finite")
    if window <= 0.0 or required <= 0.0:
        raise ValueError("pressure window and required interval must be positive")
    end = now + window
    count = 0
    for raw_eta in eta_values_s:
        eta = float(raw_eta)
        if not np.isfinite(eta):
            raise ValueError("pressure ETA values must be finite")
        if now <= eta < end:
            count += 1
    capacity = window / required
    return count, float(capacity), float(count / capacity)


@dataclass(frozen=True)
class LeaderFollowerFeatureInputs:
    """Canonical primitive values required for one bound edge."""

    sim_time_s: float
    leader_eta_s: float
    follower_eta_s: float
    required_interval_s: float
    follower_distance_to_resource_m: float
    leader_distance_to_resource_m: float
    follower_cas_kts: float
    follower_cas_lower_kts: float
    reachability: ReachabilityMap
    intercept_or_final_gate_flag: float
    remaining_action_station_count: int
    total_action_station_count: int
    remaining_intervention_budget: int
    total_intervention_budget: int
    live_nominal_etas_s: tuple[float, ...]
    trailing_spacing_margins_s: tuple[float, ...] = ()
    airport: str = "UNKNOWN"
    runway: str = "UNKNOWN"
    segment: str = "UNKNOWN"
    leader_cluster: str = "unassigned"
    follower_cluster: str = "unassigned"
    effective_remaining_action_station_fraction: float | None = None
    effective_remaining_intervention_budget_fraction: float | None = None

    def __post_init__(self) -> None:
        scalar_values = (
            self.sim_time_s,
            self.leader_eta_s,
            self.follower_eta_s,
            self.required_interval_s,
            self.follower_distance_to_resource_m,
            self.leader_distance_to_resource_m,
            self.follower_cas_kts,
            self.follower_cas_lower_kts,
            self.intercept_or_final_gate_flag,
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("feature input scalars must be finite")
        if self.required_interval_s <= 0.0:
            raise ValueError("required_interval_s must be positive")
        if self.follower_distance_to_resource_m < 0.0 or self.leader_distance_to_resource_m < 0.0:
            raise ValueError("resource distances cannot be negative")
        if self.follower_cas_kts < 0.0 or self.follower_cas_lower_kts < 0.0:
            raise ValueError("CAS values cannot be negative")
        if self.total_action_station_count <= 0 or self.total_intervention_budget <= 0:
            raise ValueError("action-station and intervention totals must be positive")
        if not 0 <= self.remaining_action_station_count <= self.total_action_station_count:
            raise ValueError("remaining action-station count is out of range")
        if not 0 <= self.remaining_intervention_budget <= self.total_intervention_budget:
            raise ValueError("remaining intervention budget is out of range")
        categories = (
            self.airport,
            self.runway,
            self.segment,
            self.leader_cluster,
            self.follower_cluster,
        )
        if any(not str(value).strip() for value in categories):
            raise ValueError("categorical feature scopes must be non-empty")
        if not all(np.isfinite(value) for value in self.live_nominal_etas_s):
            raise ValueError("live nominal ETAs must be finite")
        if not all(np.isfinite(value) for value in self.trailing_spacing_margins_s):
            raise ValueError("trailing spacing margins must be finite")
        for name in (
            "effective_remaining_action_station_fraction",
            "effective_remaining_intervention_budget_fraction",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1] when supplied")


def _capacity_ratio(
    required_delay_s: float,
    capacity_s: float,
    *,
    config: FeatureConfig,
) -> tuple[float, float, float | None]:
    undefined = 1.0 if capacity_s <= 0.0 else 0.0
    denominator = max(float(capacity_s), float(config.ratio_capacity_floor_s))
    raw_ratio = float(required_delay_s / denominator)
    clipped = float(np.clip(raw_ratio, 0.0, config.ratio_clip_max))
    diagnostic_raw = None if capacity_s <= 0.0 else float(required_delay_s / capacity_s)
    return clipped, undefined, diagnostic_raw


def derive_leader_follower_state_vector(
    inputs: LeaderFollowerFeatureInputs,
    *,
    feature_config: FeatureConfig | None = None,
    scenario_config: ScenarioConfig | None = None,
    schema: FeatureSchema | None = None,
) -> FeatureVector:
    feature_cfg = FeatureConfig() if feature_config is None else feature_config
    scenario_cfg = ScenarioConfig() if scenario_config is None else scenario_config
    active_schema = (
        leader_follower_feature_schema(feature_cfg.schema_version)
        if schema is None
        else schema
    )

    spacing = compute_spacing(
        leader_eta_s=inputs.leader_eta_s,
        follower_eta_s=inputs.follower_eta_s,
        required_interval_s=inputs.required_interval_s,
    )
    speed_capacity = inputs.reachability.speed_capacity_s
    path_capacity = inputs.reachability.path_capacity_s
    speed_ratio, speed_mask, speed_ratio_raw = _capacity_ratio(
        spacing.required_delay_s,
        speed_capacity,
        config=feature_cfg,
    )
    path_ratio, path_mask, path_ratio_raw = _capacity_ratio(
        spacing.required_delay_s,
        path_capacity,
        config=feature_cfg,
    )

    remaining_station_fraction = (
        inputs.remaining_action_station_count / inputs.total_action_station_count
        if inputs.effective_remaining_action_station_fraction is None
        else inputs.effective_remaining_action_station_fraction
    )
    remaining_budget_fraction = (
        inputs.remaining_intervention_budget / inputs.total_intervention_budget
        if inputs.effective_remaining_intervention_budget_fraction is None
        else inputs.effective_remaining_intervention_budget_fraction
    )
    follower_time = float(inputs.follower_eta_s - inputs.sim_time_s)
    leader_time = float(inputs.leader_eta_s - inputs.sim_time_s)
    commitment = commitment_components(
        time_to_threshold_s=follower_time,
        remaining_station_fraction=remaining_station_fraction,
        remaining_intervention_budget_fraction=remaining_budget_fraction,
        intercept_or_final_gate_flag=inputs.intercept_or_final_gate_flag,
        config=feature_cfg,
    )
    pressure_count, pressure_capacity, pressure_ratio = pressure_in_half_open_window(
        inputs.live_nominal_etas_s,
        sim_time_s=inputs.sim_time_s,
        window_s=scenario_cfg.pressure_window_s,
        required_interval_s=inputs.required_interval_s,
    )

    if inputs.trailing_spacing_margins_s:
        trailing_margin = float(min(inputs.trailing_spacing_margins_s))
        trailing_mask = 0.0
    else:
        trailing_margin = 0.0
        trailing_mask = 1.0

    named = {
        "spacing_deviation_s": spacing.spacing_deviation_s,
        "abs_spacing_deviation_s": abs(spacing.spacing_deviation_s),
        "required_delay_s": spacing.required_delay_s,
        "predicted_interval_s": spacing.predicted_interval_s,
        "required_interval_s": spacing.required_interval_s,
        "follower_time_to_resource_s": follower_time,
        "leader_time_to_resource_s": leader_time,
        "follower_distance_to_resource_m": float(inputs.follower_distance_to_resource_m),
        "leader_distance_to_resource_m": float(inputs.leader_distance_to_resource_m),
        "follower_cas_kts": float(inputs.follower_cas_kts),
        "follower_cas_lower_kts": float(inputs.follower_cas_lower_kts),
        "follower_cas_margin_kts": float(inputs.follower_cas_kts - inputs.follower_cas_lower_kts),
        "speed_capacity_s": speed_capacity,
        "path_capacity_s": path_capacity,
        "required_delay_over_speed_capacity": speed_ratio,
        "required_delay_over_path_capacity": path_ratio,
        "commitment_fraction": commitment.commitment_fraction,
        "intercept_or_final_gate_flag": commitment.gate_component,
        "remaining_action_station_fraction": commitment.remaining_station_fraction,
        "local_flow_count": float(len(inputs.live_nominal_etas_s)),
        "pressure_ratio": pressure_ratio,
        "trailing_min_spacing_margin_s": trailing_margin,
        "speed_capacity_undefined_mask": speed_mask,
        "path_capacity_undefined_mask": path_mask,
        "trailing_spacing_undefined_mask": trailing_mask,
    }
    diagnostics = {
        "raw": {
            "speed_capacity_s": speed_capacity,
            "path_capacity_s": path_capacity,
            "required_delay_over_speed_capacity": speed_ratio_raw,
            "required_delay_over_path_capacity": path_ratio_raw,
        },
        "pressure": {
            "count": pressure_count,
            "capacity_slots": pressure_capacity,
            "window_s": scenario_cfg.pressure_window_s,
            "window_start_s": float(inputs.sim_time_s),
            "window_end_exclusive_s": float(inputs.sim_time_s + scenario_cfg.pressure_window_s),
        },
        "commitment": {
            "time_component": commitment.time_component,
            "remaining_station_fraction": commitment.remaining_station_fraction,
            "remaining_intervention_budget_fraction": commitment.remaining_intervention_budget_fraction,
            "freedom_remaining": commitment.freedom_remaining,
            "freedom_component": commitment.freedom_component,
            "gate_component": commitment.gate_component,
            "weights": {
                "time": feature_cfg.time_weight,
                "freedom": feature_cfg.freedom_weight,
                "gate": feature_cfg.gate_weight,
            },
        },
        "reachability": inputs.reachability,
    }
    return active_schema.encode(
        named,
        categories={
            "airport": inputs.airport,
            "runway": inputs.runway,
            "segment": inputs.segment,
            "leader_cluster": inputs.leader_cluster,
            "follower_cluster": inputs.follower_cluster,
        },
        diagnostics=diagnostics,
    )


@runtime_checkable
class LeaderFollowerStateQuery(Protocol):
    """Narrow canonical query layer used by the state-vector adapter."""

    def simulation_time_s(self, state: Any) -> float: ...

    def nominal_eta_s(self, state: Any, flight_id: str, resource_id: str) -> float: ...

    def distance_to_resource_m(self, state: Any, flight_id: str, resource_id: str) -> float: ...

    def cas_kts(self, state: Any, flight_id: str) -> float: ...

    def cas_lower_kts(self, state: Any, flight_id: str) -> float: ...

    def reachability_map(self, state: Any, flight_id: str, resource_id: str) -> ReachabilityMap: ...

    def intercept_or_final_gate_flag(self, state: Any, flight_id: str) -> float: ...

    def action_station_counts(self, state: Any, flight_id: str) -> tuple[int, int]: ...

    def intervention_budget(self, state: Any, flight_id: str) -> tuple[int, int]: ...

    def live_flight_ids(self, state: Any, resource_id: str) -> Sequence[str]: ...

    def trailing_spacing_margins_s(
        self,
        state: Any,
        anchor: LeaderFollowerAnchor,
    ) -> Iterable[float]: ...

    def categorical_scope(
        self, state: Any, anchor: LeaderFollowerAnchor
    ) -> Mapping[str, str]: ...


def state_vector_from_query(
    state: Any,
    anchor: LeaderFollowerAnchor,
    *,
    query: LeaderFollowerStateQuery,
    required_interval_s: float = 90.0,
    feature_config: FeatureConfig | None = None,
    scenario_config: ScenarioConfig | None = None,
    schema: FeatureSchema | None = None,
) -> FeatureVector:
    remaining_stations, total_stations = query.action_station_counts(state, anchor.follower_id)
    remaining_budget, total_budget = query.intervention_budget(state, anchor.follower_id)
    live_ids = tuple(str(item) for item in query.live_flight_ids(state, anchor.resource_id))
    live_etas = tuple(
        float(query.nominal_eta_s(state, flight_id, anchor.resource_id))
        for flight_id in live_ids
    )
    scope = dict(query.categorical_scope(state, anchor))
    inputs = LeaderFollowerFeatureInputs(
        sim_time_s=query.simulation_time_s(state),
        leader_eta_s=query.nominal_eta_s(state, anchor.leader_id, anchor.resource_id),
        follower_eta_s=query.nominal_eta_s(state, anchor.follower_id, anchor.resource_id),
        required_interval_s=required_interval_s,
        follower_distance_to_resource_m=query.distance_to_resource_m(
            state, anchor.follower_id, anchor.resource_id
        ),
        leader_distance_to_resource_m=query.distance_to_resource_m(
            state, anchor.leader_id, anchor.resource_id
        ),
        follower_cas_kts=query.cas_kts(state, anchor.follower_id),
        follower_cas_lower_kts=query.cas_lower_kts(state, anchor.follower_id),
        reachability=query.reachability_map(state, anchor.follower_id, anchor.resource_id),
        intercept_or_final_gate_flag=query.intercept_or_final_gate_flag(state, anchor.follower_id),
        remaining_action_station_count=remaining_stations,
        total_action_station_count=total_stations,
        remaining_intervention_budget=remaining_budget,
        total_intervention_budget=total_budget,
        live_nominal_etas_s=live_etas,
        trailing_spacing_margins_s=tuple(query.trailing_spacing_margins_s(state, anchor)),
        airport=scope["airport"],
        runway=scope["runway"],
        segment=scope["segment"],
        leader_cluster=scope["leader_cluster"],
        follower_cluster=scope["follower_cluster"],
    )
    return derive_leader_follower_state_vector(
        inputs,
        feature_config=feature_config,
        scenario_config=scenario_config,
        schema=schema,
    )


def simulator_state_vector(
    simulator: Any,
    anchor: LeaderFollowerAnchor,
    *,
    action_candidates: Sequence[Any] = (),
    action_applier: Callable[[Any, Any], Any] | None = None,
    feature_config: FeatureConfig | None = None,
    scenario_config: ScenarioConfig | None = None,
    template_config: TemplateConfig | None = None,
    schema: FeatureSchema | None = None,
    reachability_cache_namespace: str | None = "default",
) -> FeatureVector:
    """Derive one finite vector from the landed Simulator contracts."""

    from hailmary.errors import StaleActionError
    from hailmary.features.anchors import active_resource_predictions
    from hailmary.features.minimum_time import simulator_reachability_map
    from hailmary.simulator.interpolation import MonotoneTrajectory

    state = simulator.state
    if (
        anchor.state_version != str(state.dynamic_content_hash)
        or anchor.epoch != state.decision_epoch_index
    ):
        raise StaleActionError("leader/follower anchor was derived from a stale simulator epoch")
    feature_cfg = _scenario_feature_config(simulator, feature_config)
    template_cfg = _scenario_template_config(simulator, template_config)
    resource = state.definition.resource(anchor.resource_id)
    required_interval = float(resource.required_interval_s)
    predictions = active_resource_predictions(simulator, resource_id=anchor.resource_id)
    by_flight = {item.flight_id: item for item in predictions}
    if anchor.leader_id not in by_flight or anchor.follower_id not in by_flight:
        raise StaleActionError("anchor flights are no longer active at the bound resource")
    leader = by_flight[anchor.leader_id]
    follower = by_flight[anchor.follower_id]

    follower_dynamic = state.flight(anchor.follower_id)
    follower_variant = state.definition.variant(follower_dynamic.current_variant_id)
    follower_trajectory = MonotoneTrajectory.from_variant(follower_variant)
    lower_cas_mps = follower_trajectory.value_at_elapsed(
        ("lower_cas_mps",),
        float(follower.sample.elapsed_time_s),
        required=True,
    )
    if follower.sample.cas_mps is None or lower_cas_mps is None:
        raise ValueError("canonical follower variant must expose CAS and its lower envelope")

    (
        remaining_stations,
        total_stations,
        remaining_budget,
        total_budget,
        effective_station_fraction,
        effective_budget_fraction,
    ) = _simulator_intervention_freedom(
        state,
        anchor.follower_id,
        template_config=template_cfg,
    )

    from hailmary.features.anchors import build_current_segment_anchors

    ordered = build_current_segment_anchors(simulator).flow_for_segment(
        anchor.segment_id
    ).ordered_flight_ids
    follower_index = ordered.index(anchor.follower_id)
    trailing = tuple(
        by_flight[flight_id] for flight_id in ordered[follower_index + 1 : follower_index + 4]
    )
    trailing_margins: list[float] = []
    previous_eta = follower.eta_s
    for trailer in trailing:
        trailing_margins.append(float(trailer.eta_s - previous_eta - required_interval))
        previous_eta = trailer.eta_s

    def cluster_label(flight_id: str) -> str:
        definition_flight = state.definition.flight(flight_id)
        if definition_flight.cluster_id:
            return str(definition_flight.cluster_id)
        dynamic = state.flight(flight_id)
        variant = state.definition.variant(dynamic.current_variant_id)
        return str(getattr(variant, "cluster_id", "unassigned") or "unassigned")

    resource_metadata = resource.metadata_dict
    reachability = simulator_reachability_map(
        simulator,
        flight_id=anchor.follower_id,
        resource_id=anchor.resource_id,
        action_candidates=action_candidates,
        action_applier=action_applier,
        cache_namespace=reachability_cache_namespace,
    )
    live_etas = tuple(by_flight[flight_id].eta_s for flight_id in ordered)
    follower_distance = max(0.0, float(follower.sample.s_m - follower.resource_s_m))
    leader_distance = max(0.0, float(leader.sample.s_m - leader.resource_s_m))
    inputs = LeaderFollowerFeatureInputs(
        sim_time_s=float(state.sim_time_s),
        leader_eta_s=leader.eta_s,
        follower_eta_s=follower.eta_s,
        required_interval_s=required_interval,
        follower_distance_to_resource_m=follower_distance,
        leader_distance_to_resource_m=leader_distance,
        follower_cas_kts=float(follower.sample.cas_mps / MPS_PER_KNOT),
        follower_cas_lower_kts=float(lower_cas_mps / MPS_PER_KNOT),
        reachability=reachability,
        intercept_or_final_gate_flag=max(
            float(follower_distance <= template_cfg.commitment_gate_nm * M_PER_NM),
            float(follower_dynamic.intercept_gate_active),
        ),
        remaining_action_station_count=remaining_stations,
        total_action_station_count=total_stations,
        remaining_intervention_budget=remaining_budget,
        total_intervention_budget=total_budget,
        live_nominal_etas_s=live_etas,
        trailing_spacing_margins_s=tuple(trailing_margins),
        airport=str(resource_metadata.get("airport", "UNKNOWN")),
        runway=str(resource_metadata.get("runway", "UNKNOWN")),
        segment=anchor.segment_id or str(resource_metadata.get("segment_id", "UNKNOWN")),
        leader_cluster=cluster_label(anchor.leader_id),
        follower_cluster=cluster_label(anchor.follower_id),
        effective_remaining_action_station_fraction=effective_station_fraction,
        effective_remaining_intervention_budget_fraction=effective_budget_fraction,
    )
    return derive_leader_follower_state_vector(
        inputs,
        feature_config=feature_cfg,
        scenario_config=scenario_config,
        schema=schema,
    )


def _simulator_intervention_freedom(
    state: Any,
    flight_id: str,
    *,
    template_config: TemplateConfig,
) -> tuple[int, int, int, int, float, float]:
    dynamic = state.flight(flight_id)
    definition = state.definition.flight(flight_id)
    crossed = set(dynamic.crossed_action_station_keys)
    speed_budget_available = dynamic.speed_action_count < template_config.max_speed_actions
    stretch_budget_available = dynamic.path_stretch_count < template_config.max_path_stretches
    remaining_stations = sum(
        1
        for station in definition.action_stations
        if (station.station_type, station.station_index) not in crossed
        and (
            (station.station_type == "speed" and speed_budget_available)
            or (station.station_type == "path_stretch" and stretch_budget_available)
        )
    )
    total_stations = max(1, len(definition.action_stations))
    total_budget = template_config.max_speed_actions + template_config.max_path_stretches
    remaining_budget = max(0, template_config.max_speed_actions - dynamic.speed_action_count) + max(
        0,
        template_config.max_path_stretches - dynamic.path_stretch_count,
    )
    station_fraction = float(remaining_stations / total_stations)
    budget_fraction = float(remaining_budget / total_budget)
    if dynamic.action_station_fraction_cap is not None:
        station_fraction = min(station_fraction, dynamic.action_station_fraction_cap)
    if dynamic.intervention_budget_fraction_cap is not None:
        budget_fraction = min(budget_fraction, dynamic.intervention_budget_fraction_cap)
    return (
        remaining_stations,
        total_stations,
        remaining_budget,
        total_budget,
        station_fraction,
        budget_fraction,
    )


def _scenario_feature_config(
    simulator: Any,
    explicit: FeatureConfig | None,
) -> FeatureConfig:
    if explicit is not None:
        return explicit
    metadata = simulator.state.definition.metadata_dict
    payload = metadata.get("feature_config")
    if isinstance(payload, dict):
        return FeatureConfig(**payload)
    return FeatureConfig()


def _scenario_template_config(
    simulator: Any,
    explicit: TemplateConfig | None,
) -> TemplateConfig:
    if explicit is not None:
        return explicit
    metadata = simulator.state.definition.metadata_dict
    payload = metadata.get("template_config")
    if isinstance(payload, dict):
        return TemplateConfig(**payload)
    return TemplateConfig()


def simulator_flight_commitment_components(
    simulator: Any,
    flight_id: str,
    *,
    resource_id: str,
    feature_config: FeatureConfig | None = None,
    template_config: TemplateConfig | None = None,
) -> CommitmentComponents:
    """Compute the canonical commitment primitive for one active flight."""

    from hailmary.features.anchors import active_resource_predictions

    feature_cfg = _scenario_feature_config(simulator, feature_config)
    template_cfg = _scenario_template_config(simulator, template_config)
    resolved_resource = str(resource_id)
    simulator.state.definition.resource(resolved_resource)
    predictions = active_resource_predictions(simulator, resource_id=resolved_resource)
    try:
        prediction = next(item for item in predictions if item.flight_id == flight_id)
    except StopIteration as exc:
        raise ValueError(f"flight {flight_id!r} is not active at resource {resolved_resource!r}") from exc
    state = simulator.state
    dynamic = state.flight(flight_id)
    (
        _remaining_stations,
        _total_stations,
        _remaining_budget,
        _total_budget,
        station_fraction,
        budget_fraction,
    ) = _simulator_intervention_freedom(
        state,
        flight_id,
        template_config=template_cfg,
    )
    distance = max(0.0, float(prediction.sample.s_m - prediction.resource_s_m))
    gate = max(
        float(distance <= template_cfg.commitment_gate_nm * M_PER_NM),
        float(dynamic.intercept_gate_active),
    )
    return commitment_components(
        time_to_threshold_s=float(prediction.eta_s - state.sim_time_s),
        remaining_station_fraction=station_fraction,
        remaining_intervention_budget_fraction=budget_fraction,
        intercept_or_final_gate_flag=gate,
        config=feature_cfg,
    )


__all__ = [
    "CommitmentComponents",
    "LeaderFollowerFeatureInputs",
    "LeaderFollowerStateQuery",
    "commitment_components",
    "derive_leader_follower_state_vector",
    "pressure_in_half_open_window",
    "simulator_flight_commitment_components",
    "simulator_state_vector",
    "state_vector_from_query",
]
