from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from itertools import product
import math
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from hailmary.config import FeatureConfig, M_PER_NM, ScenarioConfig, TemplateConfig
from hailmary.errors import CorrelationGateError
from hailmary.ids import stable_id

from hailmary.scenario.models import (
    ActionStationDefinition,
    FlightDefinition,
    MaterializedExogenousEvent,
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
    freeze_variant,
    freeze_weather,
)


@dataclass(frozen=True, slots=True)
class FlightGenerationSpec:
    flight_id: str
    baseline_variant_id: str
    observed_release_time_s: float
    cluster_id: str = ""
    callsign: str = ""
    icao24: str = ""
    runway: str = ""
    release_offset_s: float | None = None
    action_stations: tuple[ActionStationDefinition, ...] = ()
    resource_crossings: tuple[ResourceCrossingDefinition, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.flight_id or not self.baseline_variant_id:
            raise ValueError("flight and baseline-variant identities must be non-empty")
        if not math.isfinite(self.observed_release_time_s):
            raise ValueError("observed_release_time_s must be finite")
        if self.release_offset_s is not None and not math.isfinite(self.release_offset_s):
            raise ValueError("release_offset_s must be finite when supplied")

    @classmethod
    def from_template(
        cls,
        *,
        flight_id: str,
        observed_release_time_s: float,
        template: object,
        callsign: str = "",
        icao24: str = "",
        runway: str = "",
        release_offset_s: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FlightGenerationSpec":
        """Bind a flight to a duck-typed ``ClusterTemplate``.

        Template action stations use ``entry_order``/``kind`` while scenario
        events use ``station_index``/``station_type``.  Keeping the translation
        here prevents the engine from importing offline template types.
        """

        baseline = getattr(template, "baseline_variant", None)
        variant_id = getattr(baseline, "variant_id", "")
        cluster_id = str(getattr(template, "cluster_id", ""))
        if baseline is None or not variant_id or not cluster_id:
            raise TypeError("template must expose cluster_id and baseline_variant.variant_id")
        raw_stations = (
            *tuple(getattr(template, "speed_action_stations", ())),
            *tuple(getattr(template, "path_stretch_stations", ())),
        )
        stations = tuple(
            ActionStationDefinition(
                station_index=int(getattr(station, "entry_order")),
                s_m=float(getattr(station, "s_m")),
                station_type=str(getattr(station, "kind")),
            )
            for station in raw_stations
        )
        return cls(
            flight_id=flight_id,
            baseline_variant_id=str(variant_id),
            observed_release_time_s=float(observed_release_time_s),
            cluster_id=cluster_id,
            callsign=callsign,
            icao24=icao24,
            runway=runway or str(getattr(template, "runway_id", "")),
            release_offset_s=release_offset_s,
            action_stations=stations,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class ScenarioGenerator:
    """Deterministically materialize a scenario from release specifications.

    Named streams are derived from ``master_seed`` and a stable stream name,
    making generated releases independent of input iteration order.  Runtime
    branches consume no generator randomness for exogenous events: those events
    are materialized into the returned immutable definition.
    """

    master_seed: int

    def named_rng(self, stream_name: str) -> np.random.Generator:
        if not stream_name:
            raise ValueError("stream_name must be non-empty")
        digest = hashlib.sha256(f"{self.master_seed}:{stream_name}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:16], byteorder="big", signed=False)
        return np.random.default_rng(seed)

    def generate(
        self,
        *,
        scenario_id: str,
        flight_specs: Iterable[FlightGenerationSpec],
        variants: Iterable[object],
        resources: Iterable[ResourceDefinition],
        exogenous_events: Iterable[MaterializedExogenousEvent] = (),
        release_jitter_s: float = 0.0,
        weather: object | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ScenarioDefinition:
        jitter = float(release_jitter_s)
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError("release_jitter_s must be finite and non-negative")

        flights: list[FlightDefinition] = []
        for spec in sorted(flight_specs, key=lambda item: item.flight_id):
            if spec.release_offset_s is not None:
                release_offset_s = float(spec.release_offset_s)
            elif jitter > 0.0:
                release_offset_s = float(
                    self.named_rng(f"release:{spec.flight_id}").uniform(-jitter, jitter)
                )
            else:
                release_offset_s = 0.0
            flights.append(
                FlightDefinition(
                    flight_id=spec.flight_id,
                    release_time_s=float(spec.observed_release_time_s + release_offset_s),
                    baseline_variant_id=spec.baseline_variant_id,
                    cluster_id=spec.cluster_id,
                    callsign=spec.callsign,
                    icao24=spec.icao24,
                    runway=spec.runway,
                    observed_release_time_s=float(spec.observed_release_time_s),
                    release_offset_s=release_offset_s,
                    action_stations=spec.action_stations,
                    resource_crossings=spec.resource_crossings,
                    metadata=spec.metadata,
                )
            )

        return ScenarioDefinition(
            scenario_id=scenario_id,
            seed=int(self.master_seed),
            flights=tuple(flights),
            resources=tuple(sorted(resources, key=lambda item: item.resource_id)),
            variants=tuple(variants),
            exogenous_events=tuple(
                sorted(exogenous_events, key=lambda item: (item.time_s, item.event_id))
            ),
            weather=weather,
            metadata={} if metadata is None else metadata,
        )

    def generate_factorial(
        self,
        *,
        scenario_id_prefix: str,
        flight_specs: Iterable[FlightGenerationSpec],
        variants: Iterable[object],
        resources: Iterable[ResourceDefinition],
        factor_levels: Mapping[str, tuple[float, float]],
        replicates: int = 1,
        offset_model: Callable[[FlightGenerationSpec, "FactorialCondition", np.random.Generator], float]
        | None = None,
        config: ScenarioConfig | None = None,
        feature_config: FeatureConfig | None = None,
        template_config: TemplateConfig | None = None,
        exogenous_events: Iterable[MaterializedExogenousEvent] = (),
        weather: object | None = None,
    ) -> "FactorialScenarioBatch":
        """Build a balanced, audited full-factorial scenario batch.

        Every condition is realized in runtime-visible data. ``pressure``
        compresses or expands the observed schedule, ``time_to_final`` places a
        materialized disturbance relative to the target threshold ETA,
        ``error_magnitude`` becomes a flight-time shift when that event fires,
        and ``commitment`` is solved into operational-freedom caps plus an
        optional intercept gate. The resulting canonical
        ``commitment_fraction`` is measured from a provisional simulator state
        before the correlation gate. All requested factor values are carried by
        that event, so extensions do not silently become metadata-only labels.

        The optional offset model supplies an additional per-flight release
        shift. Correlations are gated on values recovered from the realized
        schedule and events, not on the requested design matrix.
        """

        cfg = config or ScenarioConfig()
        feature_cfg = feature_config or FeatureConfig()
        template_cfg = template_config or TemplateConfig()
        conditions = build_factorial_conditions(
            factor_levels,
            replicates=replicates,
            master_seed=self.master_seed,
        )
        _validate_design_factor_levels(factor_levels)
        base_specs = tuple(sorted(flight_specs, key=lambda item: item.flight_id))
        if not base_specs:
            raise ValueError("factorial generation requires at least one flight")
        immutable_variants = tuple(freeze_variant(variant) for variant in variants)
        immutable_resources = tuple(resources)
        immutable_weather = freeze_weather(weather)
        base_exogenous = tuple(exogenous_events)
        variant_by_id = {
            _variant_identity(variant): variant for variant in immutable_variants
        }
        missing_variants = sorted(
            {
                spec.baseline_variant_id
                for spec in base_specs
                if spec.baseline_variant_id not in variant_by_id
            }
        )
        if missing_variants:
            raise ValueError(f"factorial flight specs reference unknown variants {missing_variants!r}")

        realizations = tuple(
            _realize_factorial_condition(
                condition=condition,
                base_specs=base_specs,
                variant_by_id=variant_by_id,
                rng=self.named_rng(f"factorial:{condition.condition_id}"),
                offset_model=offset_model,
                feature_config=feature_cfg,
                template_config=template_cfg,
            )
            for condition in conditions
        )
        realizations = tuple(
            _measure_factorial_commitment(
                realization,
                generator=self,
                scenario_id_prefix=scenario_id_prefix,
                variants=immutable_variants,
                resources=immutable_resources,
                base_exogenous=base_exogenous,
                weather=immutable_weather,
                feature_config=feature_cfg,
                template_config=template_cfg,
            )
            for realization in realizations
        )
        realized_conditions = tuple(
            FactorialCondition(values=item.realized_values, replicate=item.condition.replicate)
            for item in realizations
        )
        audit = audit_factor_correlations(
            realized_conditions,
            registered_pairs=cfg.registered_factor_pairs,
            threshold=cfg.feature_correlation_limit,
            ignore_unregistered=False,
        )
        audit.assert_passed()

        generated: list[FactorialScenario] = []
        for realization in realizations:
            condition = realization.condition
            definition = self.generate(
                scenario_id=f"{scenario_id_prefix}-{condition.condition_id}",
                flight_specs=realization.flight_specs,
                variants=immutable_variants,
                resources=immutable_resources,
                exogenous_events=(*base_exogenous, realization.disturbance),
                weather=immutable_weather,
                metadata={
                    "mode": "factorial_training",
                    "factorial_condition_id": condition.condition_id,
                    "factorial_values": condition.values_dict,
                    "factorial_realized_values": realization.realized_values_dict,
                    "factorial_feature_config": _feature_config_payload(feature_cfg),
                    "factorial_template_config": _template_config_payload(template_cfg),
                    "correlation_audit": audit.to_dict(),
                },
            )
            generated.append(
                FactorialScenario(
                    condition=condition,
                    definition=definition,
                    realized_values=realization.realized_values,
                )
            )
        return FactorialScenarioBatch(scenarios=tuple(generated), correlation_audit=audit)


@dataclass(frozen=True, slots=True)
class _FactorialRealization:
    condition: "FactorialCondition"
    flight_specs: tuple[FlightGenerationSpec, ...]
    disturbance: MaterializedExogenousEvent
    realized_values: tuple[tuple[str, float], ...]

    @property
    def realized_values_dict(self) -> dict[str, float]:
        return dict(self.realized_values)


def _measure_factorial_commitment(
    realization: _FactorialRealization,
    *,
    generator: "ScenarioGenerator",
    scenario_id_prefix: str,
    variants: tuple[object, ...],
    resources: tuple[ResourceDefinition, ...],
    base_exogenous: tuple[MaterializedExogenousEvent, ...],
    weather: object | None,
    feature_config: FeatureConfig,
    template_config: TemplateConfig,
) -> _FactorialRealization:
    if "commitment" not in realization.realized_values_dict:
        return realization

    from hailmary.features import simulator_flight_commitment_components
    from hailmary.simulator import Simulator

    definition = generator.generate(
        scenario_id=f"{scenario_id_prefix}-{realization.condition.condition_id}",
        flight_specs=realization.flight_specs,
        variants=variants,
        resources=resources,
        exogenous_events=(*base_exogenous, realization.disturbance),
        weather=weather,
        metadata={"mode": "factorial_commitment_audit"},
    )
    simulator = Simulator(definition)
    while True:
        batch = simulator.advance_next()
        if batch is None:
            raise ValueError("factorial commitment disturbance was never reached")
        if any(event.event_id == realization.disturbance.event_id for event in batch.events):
            break

    payload = realization.disturbance.payload_dict
    target_flight_id = str(payload["target_flight_id"])
    resource_id = str(payload["commitment_resource_id"])
    components = simulator_flight_commitment_components(
        simulator,
        target_flight_id,
        resource_id=resource_id,
        feature_config=feature_config,
        template_config=template_config,
    )
    requested = realization.condition.value("commitment")
    computed = float(components.commitment_fraction)
    if abs(computed - requested) > 1e-8:
        raise ValueError(
            f"commitment target {requested:.6g} is not physically realizable for "
            f"{target_flight_id!r}; canonical commitment_fraction is {computed:.6g}"
        )

    realized_values = dict(realization.realized_values)
    realized_values["commitment"] = computed
    ordered_values = tuple(sorted(realized_values.items()))
    payload["state_updates"]["factorial_values"] = dict(ordered_values)
    payload["state_updates"]["commitment"] = computed
    payload["canonical_commitment"] = {
        "time_component": components.time_component,
        "remaining_station_fraction": components.remaining_station_fraction,
        "remaining_intervention_budget_fraction": (
            components.remaining_intervention_budget_fraction
        ),
        "freedom_remaining": components.freedom_remaining,
        "freedom_component": components.freedom_component,
        "gate_component": components.gate_component,
        "commitment_fraction": components.commitment_fraction,
    }
    disturbance = MaterializedExogenousEvent(
        event_id=realization.disturbance.event_id,
        time_s=realization.disturbance.time_s,
        stream_name=realization.disturbance.stream_name,
        payload=payload,
    )
    specs = tuple(
        replace(
            spec,
            metadata={
                **dict(spec.metadata),
                "factorial_realized_values": dict(ordered_values),
            },
        )
        for spec in realization.flight_specs
    )
    return _FactorialRealization(
        condition=realization.condition,
        flight_specs=specs,
        disturbance=disturbance,
        realized_values=ordered_values,
    )


def _realize_factorial_condition(
    *,
    condition: "FactorialCondition",
    base_specs: tuple[FlightGenerationSpec, ...],
    variant_by_id: Mapping[str, object],
    rng: np.random.Generator,
    offset_model: Callable[[FlightGenerationSpec, "FactorialCondition", np.random.Generator], float]
    | None,
    feature_config: FeatureConfig,
    template_config: TemplateConfig,
) -> _FactorialRealization:
    values = condition.values_dict
    _validate_design_factor_values(values)
    requested_pressure = float(values.get("pressure", 1.0))
    observed_anchor = min(spec.observed_release_time_s for spec in base_specs)

    preliminary: list[tuple[FlightGenerationSpec, float, float]] = []
    for spec in base_specs:
        pressure_release = observed_anchor + (
            spec.observed_release_time_s - observed_anchor
        ) / requested_pressure
        if offset_model is None:
            additional_offset = float(values.get("release_offset_s", 0.0))
        else:
            additional_offset = float(offset_model(spec, condition, rng))
        if not np.isfinite(additional_offset):
            raise ValueError("factorial release offset model returned a non-finite value")
        realized_release = pressure_release + additional_offset
        preliminary.append(
            (
                spec,
                float(realized_release - spec.observed_release_time_s),
                additional_offset,
            )
        )

    observed_span = float(
        max(spec.observed_release_time_s for spec in base_specs)
        - min(spec.observed_release_time_s for spec in base_specs)
    )
    realized_times = [spec.observed_release_time_s + offset for spec, offset, _ in preliminary]
    realized_span = float(max(realized_times) - min(realized_times))
    realized_pressure = requested_pressure
    if observed_span > 1e-9 and realized_span > 1e-9:
        realized_pressure = observed_span / realized_span

    realized = dict(values)
    if "pressure" in realized:
        realized["pressure"] = float(realized_pressure)
    if "release_offset_s" in realized:
        realized["release_offset_s"] = float(
            np.mean([additional for _, _, additional in preliminary])
        )

    target_spec, target_offset, _ = preliminary[-1]
    target_release_s = target_spec.observed_release_time_s + target_offset
    target_variant = variant_by_id[target_spec.baseline_variant_id]
    duration_s = _variant_duration_s(target_variant)
    nominal_threshold_eta_s = target_release_s + duration_s
    error_shift_s = float(realized.get("error_magnitude", 0.0))
    post_disturbance_threshold_eta_s = nominal_threshold_eta_s + error_shift_s
    if "time_to_final" in realized:
        time_to_final_s = float(realized["time_to_final"])
    elif "commitment" in realized:
        time_to_final_s = float((1.0 - realized["commitment"]) * duration_s)
    else:
        time_to_final_s = float(duration_s)
    if "commitment" in realized and time_to_final_s > duration_s + 1e-9:
        raise ValueError(
            "canonical commitment realization requires time_to_final no greater than trajectory duration"
        )
    event_time_s = float(post_disturbance_threshold_eta_s - time_to_final_s)
    if "time_to_final" in realized:
        # Recover from the post-disturbance ETA visible to the feature vector.
        realized["time_to_final"] = float(
            post_disturbance_threshold_eta_s - event_time_s
        )

    commitment_controls: dict[str, Any] | None = None
    commitment_resource_id: str | None = None
    if "commitment" in realized:
        remaining_distance_m = _variant_station_at_time_to_final(
            target_variant,
            time_to_final_s=time_to_final_s,
        )
        physical_gate = float(
            remaining_distance_m <= template_config.commitment_gate_nm * M_PER_NM
        )
        commitment_controls = _solve_commitment_controls(
            target=float(realized["commitment"]),
            time_to_final_s=time_to_final_s,
            physical_gate=physical_gate,
            feature_config=feature_config,
        )
        commitment_controls["target_flight_id"] = target_spec.flight_id
        commitment_resource_id = _variant_threshold_resource_id(target_spec, target_variant)

    realized_values = tuple(sorted((str(name), float(value)) for name, value in realized.items()))
    state_updates: dict[str, Any] = {
        "factorial_condition_id": condition.condition_id,
        "factorial_values": dict(realized_values),
        **dict(realized_values),
    }
    payload: dict[str, Any] = {
        "effect_kind": "factorial_spacing_disturbance",
        "target_flight_id": target_spec.flight_id,
        "nominal_threshold_eta_s": nominal_threshold_eta_s,
        "post_disturbance_threshold_eta_s": post_disturbance_threshold_eta_s,
        "nominal_time_to_final_s": time_to_final_s,
        "state_updates": state_updates,
    }
    if error_shift_s > 0.0:
        payload["flight_time_shift_s"] = error_shift_s
    if commitment_controls is not None:
        payload["commitment_controls"] = commitment_controls
        payload["commitment_resource_id"] = commitment_resource_id
    disturbance = MaterializedExogenousEvent(
        event_id=stable_id(
            "event",
            {
                "condition_id": condition.condition_id,
                "target_flight_id": target_spec.flight_id,
                "time_s": event_time_s,
                "effect_kind": payload["effect_kind"],
            },
            length=28,
        ),
        time_s=event_time_s,
        stream_name="factorial_disturbance",
        payload=payload,
    )

    conditioned_specs = tuple(
        FlightGenerationSpec(
            flight_id=spec.flight_id,
            baseline_variant_id=spec.baseline_variant_id,
            observed_release_time_s=spec.observed_release_time_s,
            cluster_id=spec.cluster_id,
            callsign=spec.callsign,
            icao24=spec.icao24,
            runway=spec.runway,
            release_offset_s=offset,
            action_stations=spec.action_stations,
            resource_crossings=spec.resource_crossings,
            metadata={
                **dict(spec.metadata),
                "factorial_condition_id": condition.condition_id,
                "factorial_values": condition.values_dict,
                "factorial_realized_values": dict(realized_values),
            },
        )
        for spec, offset, _ in preliminary
    )
    return _FactorialRealization(
        condition=condition,
        flight_specs=conditioned_specs,
        disturbance=disturbance,
        realized_values=realized_values,
    )


def _variant_identity(variant: object) -> str:
    if isinstance(variant, Mapping):
        value = variant.get("variant_id", variant.get("content_hash", variant.get("id")))
    else:
        value = getattr(
            variant,
            "variant_id",
            getattr(variant, "content_hash", getattr(variant, "id", None)),
        )
    if value is None or not str(value):
        raise ValueError("factorial trajectory variants must expose a stable identity")
    return str(value)


def _variant_duration_s(variant: object) -> float:
    direct = getattr(variant, "duration_s", None)
    if direct is not None:
        duration = float(direct)
    else:
        names = ("relative_elapsed_time_s", "elapsed_time_s", "t_s", "time_s")
        raw: Any | None = None
        for name in names:
            if isinstance(variant, Mapping) and name in variant:
                raw = variant[name]
                break
            if hasattr(variant, name):
                raw = getattr(variant, name)
                break
        if raw is None:
            raise ValueError("factorial trajectory variant lacks an elapsed-time profile")
        profile = np.asarray(raw, dtype=np.float64)
        if profile.ndim != 1 or len(profile) < 2 or not np.isfinite(profile).all():
            raise ValueError("factorial trajectory elapsed-time profile must be finite and one-dimensional")
        duration = float(np.max(profile) - np.min(profile))
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("factorial trajectory duration must be finite and positive")
    return duration


def _variant_array(variant: object, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if isinstance(variant, Mapping) and name in variant:
            return np.asarray(variant[name], dtype=np.float64)
        if hasattr(variant, name):
            return np.asarray(getattr(variant, name), dtype=np.float64)
    raise ValueError(f"factorial trajectory variant lacks one of fields {names!r}")


def _variant_station_at_time_to_final(variant: object, *, time_to_final_s: float) -> float:
    raw_time = _variant_array(
        variant,
        ("relative_elapsed_time_s", "elapsed_time_s", "t_s", "time_s"),
    )
    raw_s = _variant_array(variant, ("s_m",))
    if raw_time.ndim != 1 or raw_s.ndim != 1 or len(raw_time) != len(raw_s) or len(raw_s) < 2:
        raise ValueError("factorial trajectory station/time grids are incompatible")
    if not np.isfinite(raw_time).all() or not np.isfinite(raw_s).all():
        raise ValueError("factorial trajectory station/time grids must be finite")
    if np.all(np.diff(raw_time) > 0.0):
        ordered_time = raw_time
        ordered_s = raw_s
    elif np.all(np.diff(raw_time) < 0.0):
        ordered_time = raw_time[::-1]
        ordered_s = raw_s[::-1]
    else:
        raise ValueError("factorial trajectory time grid must be strictly monotone")
    elapsed = float(ordered_time[-1] - ordered_time[0] - time_to_final_s)
    elapsed = float(np.clip(elapsed, 0.0, ordered_time[-1] - ordered_time[0]))
    normalized_time = ordered_time - ordered_time[0]
    return float(np.interp(elapsed, normalized_time, ordered_s))


def _variant_threshold_resource_id(
    spec: FlightGenerationSpec,
    variant: object,
) -> str:
    for crossing in spec.resource_crossings:
        if abs(float(crossing.s_m)) <= 1e-6:
            return crossing.resource_id
    direct = getattr(variant, "threshold_resource_id", None)
    if direct:
        return str(direct)
    for crossing in getattr(variant, "resource_crossings", ()):
        if abs(float(getattr(crossing, "s_m"))) <= 1e-6:
            resource_id = str(getattr(crossing, "resource_id", ""))
            if resource_id:
                return resource_id
    raise ValueError("commitment realization requires a threshold resource crossing")


def _solve_commitment_controls(
    *,
    target: float,
    time_to_final_s: float,
    physical_gate: float,
    feature_config: FeatureConfig,
) -> dict[str, float]:
    time_component = float(
        np.clip(1.0 - time_to_final_s / feature_config.commitment_time_scale_s, 0.0, 1.0)
    )
    freedom_weight = float(feature_config.freedom_weight)
    fraction_weight = float(
        feature_config.station_freedom_weight + feature_config.budget_freedom_weight
    )
    if freedom_weight <= 0.0 or fraction_weight <= 0.0:
        raise ValueError("commitment realization requires positive freedom weights")
    candidate_gates = (physical_gate,) if physical_gate >= 1.0 else (0.0, 1.0)
    for gate in candidate_gates:
        freedom_component = (
            target
            - feature_config.time_weight * time_component
            - feature_config.gate_weight * gate
        ) / freedom_weight
        if not -1e-10 <= freedom_component <= 1.0 + 1e-10:
            continue
        freedom_component = float(np.clip(freedom_component, 0.0, 1.0))
        cap = float((1.0 - freedom_component) / fraction_weight)
        if not -1e-10 <= cap <= 1.0 + 1e-10:
            continue
        cap = float(np.clip(cap, 0.0, 1.0))
        computed = float(
            feature_config.time_weight * time_component
            + feature_config.freedom_weight * (1.0 - fraction_weight * cap)
            + feature_config.gate_weight * gate
        )
        if abs(computed - target) <= 1e-9:
            return {
                "action_station_fraction_cap": cap,
                "intervention_budget_fraction_cap": cap,
                "intercept_gate_flag": gate,
                "time_component": time_component,
                "computed_commitment_fraction": computed,
            }
    raise ValueError(
        f"commitment target {target:.6g} is infeasible at time_to_final={time_to_final_s:.6g}s "
        "under the canonical commitment formula"
    )


def _feature_config_payload(config: FeatureConfig) -> dict[str, float | str]:
    return {
        "schema_version": config.schema_version,
        "commitment_time_scale_s": config.commitment_time_scale_s,
        "time_weight": config.time_weight,
        "freedom_weight": config.freedom_weight,
        "gate_weight": config.gate_weight,
        "station_freedom_weight": config.station_freedom_weight,
        "budget_freedom_weight": config.budget_freedom_weight,
    }


def _template_config_payload(config: TemplateConfig) -> dict[str, float | int | str]:
    return {
        "schema_version": config.schema_version,
        "commitment_gate_nm": config.commitment_gate_nm,
        "max_speed_actions": config.max_speed_actions,
        "max_path_stretches": config.max_path_stretches,
    }


def _validate_design_factor_levels(factor_levels: Mapping[str, tuple[float, float]]) -> None:
    for combination in product(*(factor_levels[name] for name in sorted(factor_levels))):
        _validate_design_factor_values(
            dict(zip(sorted(factor_levels), map(float, combination), strict=True))
        )


def _validate_design_factor_values(values: Mapping[str, float]) -> None:
    if "error_magnitude" in values and float(values["error_magnitude"]) < 0.0:
        raise ValueError("error_magnitude must be non-negative")
    if "time_to_final" in values and float(values["time_to_final"]) < 0.0:
        raise ValueError("time_to_final must be non-negative")
    if "pressure" in values and float(values["pressure"]) <= 0.0:
        raise ValueError("pressure must be positive")
    if "commitment" in values and not 0.0 <= float(values["commitment"]) <= 1.0:
        raise ValueError("commitment must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class FactorialCondition:
    values: tuple[tuple[str, float], ...]
    replicate: int
    condition_id: str = ""

    def __post_init__(self) -> None:
        normalized = tuple(sorted((str(name), float(value)) for name, value in self.values))
        if not normalized or len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("factorial conditions require unique named values")
        if any(not np.isfinite(value) for _, value in normalized):
            raise ValueError("factorial condition values must be finite")
        if self.replicate < 0:
            raise ValueError("factorial replicate cannot be negative")
        object.__setattr__(self, "values", normalized)
        computed = stable_id(
            "condition",
            {"values": normalized, "replicate": self.replicate},
            length=24,
        )
        if self.condition_id and self.condition_id != computed:
            raise ValueError("condition_id does not match factorial values")
        object.__setattr__(self, "condition_id", computed)

    @property
    def values_dict(self) -> dict[str, float]:
        return dict(self.values)

    def value(self, name: str, default: float | None = None) -> float:
        values = self.values_dict
        if name in values:
            return values[name]
        if default is None:
            raise KeyError(name)
        return float(default)


@dataclass(frozen=True, slots=True)
class CorrelationRecord:
    first_feature: str
    second_feature: str
    correlation: float
    sample_count: int
    passed: bool


@dataclass(frozen=True, slots=True)
class CorrelationAudit:
    threshold: float
    records: tuple[CorrelationRecord, ...]

    @property
    def passed(self) -> bool:
        return all(record.passed for record in self.records)

    def assert_passed(self) -> None:
        failed = [
            f"{item.first_feature}/{item.second_feature}={item.correlation:.3f}"
            for item in self.records
            if not item.passed
        ]
        if failed:
            raise CorrelationGateError(
                f"factorial correlation gate {self.threshold:.3f} exceeded: {', '.join(failed)}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "passed": self.passed,
            "records": [
                {
                    "first_feature": item.first_feature,
                    "second_feature": item.second_feature,
                    "correlation": item.correlation,
                    "sample_count": item.sample_count,
                    "passed": item.passed,
                }
                for item in self.records
            ],
        }


@dataclass(frozen=True, slots=True)
class FactorialScenario:
    condition: FactorialCondition
    definition: ScenarioDefinition
    realized_values: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        values = self.condition.values if not self.realized_values else self.realized_values
        normalized = tuple(sorted((str(name), float(value)) for name, value in values))
        if len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("realized factorial values must have unique names")
        if any(not np.isfinite(value) for _, value in normalized):
            raise ValueError("realized factorial values must be finite")
        object.__setattr__(self, "realized_values", normalized)

    @property
    def realized_values_dict(self) -> dict[str, float]:
        return dict(self.realized_values)


@dataclass(frozen=True, slots=True)
class FactorialScenarioBatch:
    scenarios: tuple[FactorialScenario, ...]
    correlation_audit: CorrelationAudit


def build_factorial_conditions(
    factor_levels: Mapping[str, tuple[float, float]],
    *,
    replicates: int,
    master_seed: int,
) -> tuple[FactorialCondition, ...]:
    del master_seed  # retained in the API/provenance; the balanced grid itself has no random draws.
    if replicates < 1:
        raise ValueError("replicates must be positive")
    names = tuple(sorted(str(name) for name in factor_levels))
    if not names:
        raise ValueError("at least one factorial feature is required")
    levels: list[tuple[float, float]] = []
    for name in names:
        raw = factor_levels[name]
        if len(raw) != 2:
            raise ValueError(f"factor {name!r} must have exactly low/high levels")
        low, high = map(float, raw)
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            raise ValueError(f"factor {name!r} levels must be finite and distinct")
        levels.append((low, high))
    return tuple(
        FactorialCondition(values=tuple(zip(names, combination, strict=True)), replicate=replicate)
        for replicate in range(replicates)
        for combination in product(*levels)
    )


def audit_factor_correlations(
    conditions: Iterable[FactorialCondition],
    *,
    registered_pairs: Iterable[tuple[str, str]],
    threshold: float = 0.30,
    ignore_unregistered: bool = False,
) -> CorrelationAudit:
    rows = tuple(condition.values_dict for condition in conditions)
    if not rows:
        raise ValueError("cannot audit an empty factorial design")
    records: list[CorrelationRecord] = []
    for first, second in registered_pairs:
        if first not in rows[0] or second not in rows[0]:
            if ignore_unregistered:
                continue
            raise ValueError(f"registered correlation pair {first!r}/{second!r} is absent")
        left = np.asarray([row[first] for row in rows], dtype=float)
        right = np.asarray([row[second] for row in rows], dtype=float)
        if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
        records.append(
            CorrelationRecord(
                first_feature=first,
                second_feature=second,
                correlation=correlation,
                sample_count=len(rows),
                passed=abs(correlation) <= threshold + 1e-12,
            )
        )
    return CorrelationAudit(threshold=float(threshold), records=tuple(records))


def build_scenario_definition(
    *,
    scenario_id: str,
    seed: int,
    flights: Iterable[FlightDefinition],
    resources: Iterable[ResourceDefinition],
    variants: Iterable[object],
    exogenous_events: Iterable[MaterializedExogenousEvent] = (),
    weather: object | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ScenarioDefinition:
    """Build a definition when release times have already been materialized."""

    return ScenarioDefinition(
        scenario_id=scenario_id,
        seed=int(seed),
        flights=tuple(sorted(flights, key=lambda item: item.flight_id)),
        resources=tuple(sorted(resources, key=lambda item: item.resource_id)),
        variants=tuple(variants),
        exogenous_events=tuple(sorted(exogenous_events, key=lambda item: (item.time_s, item.event_id))),
        weather=weather,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "CorrelationAudit",
    "CorrelationRecord",
    "FactorialCondition",
    "FactorialScenario",
    "FactorialScenarioBatch",
    "FlightGenerationSpec",
    "ScenarioGenerator",
    "audit_factor_correlations",
    "build_factorial_conditions",
    "build_scenario_definition",
]
