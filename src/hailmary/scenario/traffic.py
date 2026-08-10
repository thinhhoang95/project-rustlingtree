"""ADS-B-centred demand windows and deterministic global traffic scaling."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hailmary.data.adsb import RawADSBTrack, reconstruct_terminal_entry
from hailmary.data.catalog import CatalogArrival
from hailmary.geometry.frame import LocalFrame
from hailmary.ids import canonical_data, content_hash, stable_id
from hailmary.scenario.generator import FlightGenerationSpec, ScenarioGenerator
from hailmary.scenario.models import (
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
)
from hailmary.templates import ClusterTemplate, TrajectoryVariant
from hailmary.templates.speed import (
    cas_to_tas,
    monotone_command_profile,
    tas_to_cas,
)


def _runway(value: str) -> str:
    token = str(value).strip().upper()
    if not token:
        raise ValueError("runway must be non-empty")
    return token if token.startswith("RW") else f"RW{token}"


@dataclass(frozen=True, slots=True, order=True)
class ArrivalClusterKey:
    airport: str
    runway: str
    cluster: str

    def __post_init__(self) -> None:
        if not str(self.airport).strip() or not str(self.cluster).strip():
            raise ValueError("arrival-cluster identities must be non-empty")
        object.__setattr__(self, "airport", str(self.airport).strip().upper())
        object.__setattr__(self, "runway", _runway(self.runway))
        object.__setattr__(self, "cluster", str(self.cluster).strip())

    @property
    def qualified_id(self) -> str:
        return f"{self.airport}:{self.runway}:{self.cluster}"

    @classmethod
    def parse(cls, value: str) -> "ArrivalClusterKey":
        parts = str(value).split(":", 2)
        if len(parts) != 3:
            raise ValueError("qualified cluster ID must be AIRPORT:RUNWAY:CLUSTER")
        return cls(*parts)


@dataclass(frozen=True, slots=True)
class DemandWindowConfig:
    width_s: int = 3_600
    stride_s: int = 1_200

    def __post_init__(self) -> None:
        for name in ("width_s", "stride_s"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True, order=True)
class DemandWindow:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValueError("demand-window boundaries must be finite")
        if self.end_s <= self.start_s:
            raise ValueError("demand-window end must follow start")

    @property
    def window_id(self) -> str:
        return stable_id(
            "window",
            {"start_s": self.start_s, "end_s": self.end_s},
            length=20,
        )

    def contains(self, time_s: float) -> bool:
        return self.start_s <= float(time_s) < self.end_s


def iter_demand_windows(
    start_s: float,
    stop_s: float,
    *,
    config: DemandWindowConfig | None = None,
) -> tuple[DemandWindow, ...]:
    """Create half-open windows whose starts are in ``[start_s, stop_s)``."""

    cfg = DemandWindowConfig() if config is None else config
    start = float(start_s)
    stop = float(stop_s)
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError("window start/stop must be finite and increasing")
    count = int(math.ceil((stop - start) / cfg.stride_s))
    return tuple(
        DemandWindow(start + index * cfg.stride_s, start + index * cfg.stride_s + cfg.width_s)
        for index in range(count)
        if start + index * cfg.stride_s < stop
    )


@dataclass(frozen=True, slots=True)
class TrafficScaleConfig:
    global_scale: float = 1.0
    replicate: int = 0
    master_seed: int = 17
    intensity_bandwidth_s: float = 1_200.0
    sparse_cluster_min_count: int = 4

    def __post_init__(self) -> None:
        if not math.isfinite(self.global_scale) or self.global_scale < 0.0:
            raise ValueError("global_scale must be finite and non-negative")
        if type(self.replicate) is not int or self.replicate < 0:
            raise ValueError("replicate must be a non-negative integer")
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise ValueError("master_seed must be a non-negative integer")
        if not math.isfinite(self.intensity_bandwidth_s) or self.intensity_bandwidth_s <= 0.0:
            raise ValueError("intensity_bandwidth_s must be finite and positive")
        if type(self.sparse_cluster_min_count) is not int or self.sparse_cluster_min_count < 1:
            raise ValueError("sparse_cluster_min_count must be positive")

    def target_count(self, observed_count: int) -> int:
        if type(observed_count) is not int or observed_count < 0:
            raise ValueError("observed_count must be a non-negative integer")
        scaled = Decimal(str(self.global_scale)) * Decimal(observed_count)
        return int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True, slots=True, order=True)
class ClusterCount:
    key: ArrivalClusterKey
    count: int

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ValueError("cluster count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ObservedArrival:
    dataset_id: str
    key: ArrivalClusterKey
    flight_id: str
    terminal_entry_time_s: float
    terminal_entry_ground_speed_mps: float
    terminal_entry_altitude_m: float
    baseline_variant_id: str
    callsign: str = ""
    icao24: str = ""
    source_day: str = ""
    synthetic: bool = False
    donor_flight_id: str | None = None
    intensity_scope: str = "observed"

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.flight_id or not self.baseline_variant_id:
            raise ValueError("arrival identities must be non-empty")
        values = (
            self.terminal_entry_time_s,
            self.terminal_entry_ground_speed_mps,
            self.terminal_entry_altitude_m,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("terminal-entry profile must be finite")
        if self.terminal_entry_ground_speed_mps <= 0.0:
            raise ValueError("terminal-entry ground speed must be positive")
        if self.synthetic and not self.donor_flight_id:
            raise ValueError("synthetic arrivals require donor_flight_id provenance")

    @property
    def cluster_id(self) -> str:
        return self.key.qualified_id


@dataclass(frozen=True, slots=True)
class TrafficScenario:
    window: DemandWindow
    scale: float
    replicate: int
    definition: ScenarioDefinition
    observed_cluster_counts: tuple[ClusterCount, ...]
    target_cluster_counts: tuple[ClusterCount, ...]
    rejection_counts: tuple[tuple[str, int], ...] = ()
    schema_version: str = "hailmary.traffic_scenario.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "hailmary.traffic_scenario.v2":
            raise ValueError("unsupported traffic-scenario schema")
        if self.definition.schema_version != "hailmary.scenario.v2":
            raise ValueError("traffic scenarios require scenario schema v2")
        observed = {item.key: item.count for item in self.observed_cluster_counts}
        target = {item.key: item.count for item in self.target_cluster_counts}
        if set(observed) != set(target):
            raise ValueError("observed and target count tables must have identical cluster keys")
        realized = Counter(
            ArrivalClusterKey.parse(flight.cluster_id) for flight in self.definition.flights
        )
        if realized != Counter(target):
            raise ValueError("scenario flights do not realize the target cluster counts")

    @property
    def observed_runway_counts(self) -> Mapping[tuple[str, str], int]:
        result: Counter[tuple[str, str]] = Counter()
        for item in self.observed_cluster_counts:
            result[(item.key.airport, item.key.runway)] += item.count
        return MappingProxyType(dict(sorted(result.items())))

    @property
    def target_runway_counts(self) -> Mapping[tuple[str, str], int]:
        result: Counter[tuple[str, str]] = Counter()
        for item in self.target_cluster_counts:
            result[(item.key.airport, item.key.runway)] += item.count
        return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True, slots=True)
class TrafficScenarioBatch:
    dataset_id: str
    scale_config: TrafficScaleConfig
    scenarios: tuple[TrafficScenario, ...]
    source_partition: str = ""
    audit: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "hailmary.traffic_scenario_batch.v2"
    batch_content_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != "hailmary.traffic_scenario_batch.v2":
            raise ValueError("unsupported traffic-scenario-batch schema")
        if not str(self.dataset_id).strip():
            raise ValueError("traffic scenario batch dataset_id must be non-empty")
        scenarios = tuple(sorted(self.scenarios, key=lambda item: item.window.start_s))
        if any(not math.isclose(item.scale, self.scale_config.global_scale) for item in scenarios):
            raise ValueError("every traffic scenario must use the batch global scale")
        if any(item.replicate != self.scale_config.replicate for item in scenarios):
            raise ValueError("every traffic scenario must use the batch replicate")
        if any(
            item.definition.metadata_dict.get("dataset_id") != self.dataset_id
            for item in scenarios
        ):
            raise ValueError("every traffic scenario must use the batch dataset")
        starts = [item.window.start_s for item in scenarios]
        if len(starts) != len(set(starts)):
            raise ValueError("traffic scenario windows must have unique starts")
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "audit", MappingProxyType(dict(sorted(self.audit.items()))))
        payload = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "scale_config": canonical_data(self.scale_config),
            "scenario_hashes": tuple(item.definition.definition_hash for item in scenarios),
            "source_partition": self.source_partition,
            "audit": canonical_data(self.audit),
        }
        computed = content_hash(payload, namespace="hailmary.traffic_batch.v2")
        if self.batch_content_hash and self.batch_content_hash != computed:
            raise ValueError("traffic batch content hash does not match content")
        object.__setattr__(self, "batch_content_hash", computed)


def _named_rng(seed: int, stream_name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{stream_name}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def materialize_arrival_variant(
    template: ClusterTemplate,
    arrival: ObservedArrival,
    *,
    route_traversals: Sequence[Any] = (),
) -> TrajectoryVariant:
    """Bind one observed terminal-entry state to its compiled medoid template.

    Lateral geometry remains exactly the medoid path. Altitude and speed start
    at the arrival's observed terminal-entry state and blend into the medoid
    profiles downstream, with speed constrained by the template CAS envelope.
    This is the production materialization used by ``TrafficScenarioBuilder``.
    """

    baseline = template.baseline_variant
    normalized_station = baseline.s_m / float(baseline.s_m[-1])
    upstream_weight = normalized_station**4
    altitude = np.maximum(
        0.0,
        baseline.altitude_m
        + (
            arrival.terminal_entry_altitude_m
            - float(baseline.altitude_m[-1])
        )
        * upstream_weight,
    )
    entry_cas = float(
        tas_to_cas(
            np.asarray([arrival.terminal_entry_ground_speed_mps]),
            np.asarray([arrival.terminal_entry_altitude_m]),
        )[0]
    )
    desired_command = baseline.reference_command_cas_mps + (
        entry_cas - float(baseline.reference_command_cas_mps[-1])
    ) * upstream_weight
    candidate_command = monotone_command_profile(
        baseline.s_m,
        desired_command,
    )
    lower = baseline.lower_cas_mps
    upper = baseline.upper_cas_mps
    # Project the donor-adjusted command onto the compiled pointwise envelope
    # while retaining the required downstream-nonincreasing order.  The
    # suffix cap prevents a locally high command from making a later bound
    # impossible; the compiled baseline proves the feasible set is non-empty.
    suffix_upper = np.minimum.accumulate(upper[::-1])[::-1]
    command = np.empty_like(candidate_command)
    previous = -float("inf")
    for index, proposed in enumerate(candidate_command):
        allowed_lower = max(float(lower[index]), previous)
        allowed_upper = float(suffix_upper[index])
        if allowed_lower > allowed_upper + 1.0e-9:
            command = baseline.command_cas_mps
            break
        command[index] = np.clip(proposed, allowed_lower, allowed_upper)
        previous = float(command[index])
    tas = cas_to_tas(command, altitude)
    ground = tas + (
        arrival.terminal_entry_ground_speed_mps - float(tas[-1])
    ) * upstream_weight
    if np.any(ground <= 0.0):
        raise ValueError(
            f"arrival {arrival.flight_id!r} cannot produce a positive ground-speed profile"
        )
    threshold_id = baseline.threshold_resource_id
    if threshold_id is None:
        raise ValueError("compiled traffic templates require a threshold resource")
    resource_stations = {
        item.resource_id: item.s_m
        for item in baseline.resource_crossings
        if item.resource_id != threshold_id
    }
    for traversal in route_traversals:
        resource_stations[traversal.entry_resource_id] = traversal.entry_s_m
        resource_stations[traversal.exit_resource_id] = traversal.exit_s_m
    diagnostics = replace(
        baseline.diagnostics,
        message="materialized medoid geometry with observed terminal-entry state",
        speed_source="joint_observed_terminal_entry_profile",
        details=(
            *baseline.diagnostics.details,
            ("terminal_entry_donor_flight_id", arrival.donor_flight_id or arrival.flight_id),
            ("terminal_entry_ground_speed_mps", arrival.terminal_entry_ground_speed_mps),
            ("terminal_entry_altitude_m", arrival.terminal_entry_altitude_m),
        ),
    )
    return TrajectoryVariant.from_kinematic_profile(
        template_id=baseline.template_id,
        cluster_id=baseline.cluster_id,
        s_m=baseline.s_m,
        lat_deg=baseline.lat_deg,
        lon_deg=baseline.lon_deg,
        east_m=baseline.east_m,
        north_m=baseline.north_m,
        altitude_m=altitude,
        cas_mps=command,
        tas_mps=tas,
        ground_speed_mps=ground,
        command_cas_mps=command,
        reference_command_cas_mps=command,
        lower_cas_mps=lower,
        upper_cas_mps=upper,
        threshold_resource_id=threshold_id,
        resource_stations_m=tuple(sorted(resource_stations.items())),
        diagnostics=diagnostics,
        action_provenance=baseline.action_provenance,
    )


class TrafficScenarioBuilder:
    """Materialize independently scaled snapshots from one observed corpus."""

    def __init__(
        self,
        arrivals: Iterable[ObservedArrival],
        *,
        templates_by_cluster: Mapping[str, ClusterTemplate],
        route_graph: Any | None = None,
        training_arrivals: Iterable[ObservedArrival] | None = None,
        rejection_counts: Mapping[str, int] | None = None,
    ) -> None:
        ordered = tuple(
            sorted(arrivals, key=lambda item: (item.terminal_entry_time_s, item.flight_id))
        )
        if not ordered:
            raise ValueError("traffic builder requires at least one observed arrival")
        if len({item.flight_id for item in ordered}) != len(ordered):
            raise ValueError("observed arrival flight IDs must be unique")
        datasets = {item.dataset_id for item in ordered}
        if len(datasets) != 1:
            raise ValueError("traffic builder cannot mix datasets")
        templates = dict(templates_by_cluster)
        missing = sorted({item.cluster_id for item in ordered} - set(templates))
        if missing:
            raise ValueError(f"traffic clusters lack compiled templates: {missing}")
        if any(not isinstance(item, ClusterTemplate) for item in templates.values()):
            raise TypeError("templates_by_cluster must contain ClusterTemplate values")
        training = tuple(
            sorted(
                ordered if training_arrivals is None else training_arrivals,
                key=lambda item: (item.terminal_entry_time_s, item.flight_id),
            )
        )
        if not training:
            raise ValueError("traffic builder requires a non-empty training intensity corpus")
        if len({item.flight_id for item in training}) != len(training):
            raise ValueError("training arrival flight IDs must be unique")
        if {item.dataset_id for item in training} != datasets:
            raise ValueError("training arrivals must use the observed-arrival dataset")
        if any(item.synthetic for item in training):
            raise ValueError("training intensity corpus cannot contain synthetic arrivals")
        dataset_id = next(iter(datasets))
        for qualified_id, template in templates.items():
            expected = ArrivalClusterKey(
                template.airport_id,
                template.runway_id,
                template.cluster_id,
            ).qualified_id
            if qualified_id != expected:
                raise ValueError(
                    "template mapping keys must be qualified airport/runway/cluster IDs"
                )
            if template.dataset_id != dataset_id:
                raise ValueError("templates and traffic corpus must use the same dataset")
        mismatched_variants = sorted(
            item.flight_id
            for item in ordered
            if item.baseline_variant_id
            != templates[item.cluster_id].baseline_variant.variant_id
        )
        if mismatched_variants:
            raise ValueError(
                "arrival baseline variants do not match their compiled templates: "
                f"{mismatched_variants}"
            )
        if route_graph is not None:
            if route_graph.dataset_id != dataset_id:
                raise ValueError("route graph and traffic corpus must use the same dataset")
            graph_clusters = {
                item.qualified_cluster_id for item in route_graph.traversals
            }
            missing_graph = sorted({item.cluster_id for item in ordered} - graph_clusters)
            if missing_graph:
                raise ValueError(
                    f"traffic clusters lack route-graph traversals: {missing_graph}"
                )
        self.arrivals = ordered
        self.training_arrivals = training
        self.templates_by_cluster = templates
        self.route_graph = route_graph
        self.rejection_counts = dict(sorted((rejection_counts or {}).items()))
        self.dataset_id = dataset_id
        self.cluster_keys = tuple(sorted({item.key for item in ordered}))

    def _stream(self, window: DemandWindow, key: ArrivalClusterKey, cfg: TrafficScaleConfig, purpose: str) -> str:
        return ":".join(
            (
                self.dataset_id,
                window.window_id,
                key.qualified_id,
                format(cfg.global_scale, ".17g"),
                str(cfg.replicate),
                purpose,
            )
        )

    def _intensity_pool(
        self, key: ArrivalClusterKey, cfg: TrafficScaleConfig
    ) -> tuple[tuple[ObservedArrival, ...], str]:
        cluster = tuple(item for item in self.training_arrivals if item.key == key)
        if len(cluster) >= cfg.sparse_cluster_min_count:
            return cluster, "cluster"
        runway = tuple(
            item
            for item in self.training_arrivals
            if item.key.airport == key.airport and item.key.runway == key.runway
        )
        if len(runway) >= cfg.sparse_cluster_min_count:
            return runway, "runway"
        airport = tuple(item for item in self.training_arrivals if item.key.airport == key.airport)
        if airport:
            return airport, "airport"
        raise ValueError(f"no training intensity fallback exists for {key.qualified_id}")

    def _sample_time(
        self,
        *,
        window: DemandWindow,
        pool: Sequence[ObservedArrival],
        rng: np.random.Generator,
        bandwidth_s: float,
    ) -> float:
        day_s = 86_400.0
        for _ in range(256):
            center = pool[int(rng.integers(0, len(pool)))].terminal_entry_time_s % day_s
            candidate_tod = (center + float(rng.normal(0.0, bandwidth_s))) % day_s
            day_index = math.floor(window.start_s / day_s)
            candidates = (
                (day_index - 1) * day_s + candidate_tod,
                day_index * day_s + candidate_tod,
                (day_index + 1) * day_s + candidate_tod,
            )
            valid = [value for value in candidates if window.contains(value)]
            if valid:
                return float(min(valid))
        # Conditioning must still produce the exact count when a sparse
        # empirical kernel places negligible mass inside this particular window.
        return float(rng.uniform(window.start_s, window.end_s))

    def _materialize_cluster(
        self,
        baseline: tuple[ObservedArrival, ...],
        *,
        key: ArrivalClusterKey,
        target: int,
        window: DemandWindow,
        config: TrafficScaleConfig,
    ) -> tuple[ObservedArrival, ...]:
        observed = len(baseline)
        if target == observed:
            return baseline
        rng = _named_rng(config.master_seed, self._stream(window, key, config, "count"))
        if target < observed:
            indices = rng.choice(observed, size=target, replace=False)
            return tuple(sorted((baseline[int(index)] for index in indices), key=lambda item: (item.terminal_entry_time_s, item.flight_id)))

        intensity_pool, scope = self._intensity_pool(key, config)
        donor_pool = tuple(item for item in self.training_arrivals if item.key == key)
        if not donor_pool:
            donor_pool = baseline
        if not donor_pool:
            raise ValueError("a zero baseline cluster cannot receive scaled additions")
        additions: list[ObservedArrival] = []
        for addition_index in range(target - observed):
            addition_rng = _named_rng(
                config.master_seed,
                self._stream(window, key, config, f"addition:{addition_index}"),
            )
            donor = donor_pool[int(addition_rng.integers(0, len(donor_pool)))]
            time_s = self._sample_time(
                window=window,
                pool=intensity_pool,
                rng=addition_rng,
                bandwidth_s=config.intensity_bandwidth_s,
            )
            flight_id = stable_id(
                "synthetic_flight",
                {
                    "dataset_id": self.dataset_id,
                    "window_id": window.window_id,
                    "cluster_id": key.qualified_id,
                    "scale": config.global_scale,
                    "replicate": config.replicate,
                    "addition_index": addition_index,
                },
                length=28,
            )
            additions.append(
                ObservedArrival(
                    dataset_id=self.dataset_id,
                    key=key,
                    flight_id=flight_id,
                    terminal_entry_time_s=time_s,
                    terminal_entry_ground_speed_mps=donor.terminal_entry_ground_speed_mps,
                    terminal_entry_altitude_m=donor.terminal_entry_altitude_m,
                    baseline_variant_id=donor.baseline_variant_id,
                    callsign=f"SYN{addition_index:04d}",
                    source_day=donor.source_day,
                    synthetic=True,
                    donor_flight_id=donor.flight_id,
                    intensity_scope=scope,
                )
            )
        return tuple(sorted((*baseline, *additions), key=lambda item: (item.terminal_entry_time_s, item.flight_id)))

    def build_scenario(
        self,
        window: DemandWindow,
        *,
        scale_config: TrafficScaleConfig | None = None,
    ) -> TrafficScenario:
        cfg = TrafficScaleConfig() if scale_config is None else scale_config
        baseline_by_key = {
            key: tuple(item for item in self.arrivals if item.key == key and window.contains(item.terminal_entry_time_s))
            for key in self.cluster_keys
        }
        observed_counts = tuple(ClusterCount(key, len(baseline_by_key[key])) for key in self.cluster_keys)
        targets = {key: cfg.target_count(len(baseline_by_key[key])) for key in self.cluster_keys}
        target_counts = tuple(ClusterCount(key, targets[key]) for key in self.cluster_keys)
        materialized = tuple(
            arrival
            for key in self.cluster_keys
            for arrival in self._materialize_cluster(
                baseline_by_key[key], key=key, target=targets[key], window=window, config=cfg
            )
        )

        runway_resources: dict[str, ResourceDefinition] = {}
        specs: list[FlightGenerationSpec] = []
        variants: dict[str, object] = {}
        for arrival in materialized:
            threshold_id = f"{arrival.key.airport}:{arrival.key.runway}:threshold"
            runway_resources[threshold_id] = ResourceDefinition(
                threshold_id,
                kind="runway_threshold",
                metadata={"airport": arrival.key.airport, "runway": arrival.key.runway},
            )
            template = self.templates_by_cluster[arrival.cluster_id]
            route_traversals = (
                ()
                if self.route_graph is None
                else self.route_graph.traversals_for(arrival.cluster_id)
            )
            variant = materialize_arrival_variant(
                template,
                arrival,
                route_traversals=route_traversals,
            )
            variants[variant.variant_id] = variant
            spec = FlightGenerationSpec.from_template(
                flight_id=arrival.flight_id,
                observed_release_time_s=arrival.terminal_entry_time_s,
                template=template,
                callsign=arrival.callsign,
                icao24=arrival.icao24,
                runway=arrival.key.runway,
                metadata={
                    "dataset_id": arrival.dataset_id,
                    "airport": arrival.key.airport,
                    "runway": arrival.key.runway,
                    "source_day": arrival.source_day,
                    "synthetic": arrival.synthetic,
                    "donor_flight_id": arrival.donor_flight_id,
                    "intensity_scope": arrival.intensity_scope,
                    "terminal_entry_ground_speed_mps": arrival.terminal_entry_ground_speed_mps,
                    "terminal_entry_altitude_m": arrival.terminal_entry_altitude_m,
                },
            )
            spec = replace(
                spec,
                baseline_variant_id=variant.variant_id,
                cluster_id=arrival.cluster_id,
                resource_crossings=(ResourceCrossingDefinition(threshold_id, 0.0),),
            )
            if self.route_graph is not None:
                from hailmary.topology import attach_route_graph

                spec = attach_route_graph(spec, self.route_graph)
            specs.append(spec)

        segment_resources = () if self.route_graph is None else self.route_graph.resources()
        resources = {
            item.resource_id: item for item in (*runway_resources.values(), *segment_resources)
        }
        definition = ScenarioGenerator(cfg.master_seed).generate(
            scenario_id=stable_id(
                "traffic_scenario",
                {
                    "dataset_id": self.dataset_id,
                    "window": canonical_data(window),
                    "scale": cfg.global_scale,
                    "replicate": cfg.replicate,
                },
                length=28,
            ),
            flight_specs=specs,
            variants=variants.values(),
            resources=resources.values(),
            metadata={
                "schema_version": "hailmary.traffic_provenance.v2",
                "dataset_id": self.dataset_id,
                "demand_window": canonical_data(window),
                "global_scale": cfg.global_scale,
                "replicate": cfg.replicate,
                "observed_cluster_counts": canonical_data(observed_counts),
                "target_cluster_counts": canonical_data(target_counts),
                "route_graph_hash": None if self.route_graph is None else self.route_graph.artifact_content_hash,
                "rejection_counts": self.rejection_counts,
            },
        )
        # ScenarioGenerator is generic; traffic generation owns the v2 contract.
        definition = ScenarioDefinition(
            scenario_id=definition.scenario_id,
            seed=definition.seed,
            flights=definition.flights,
            resources=definition.resources,
            variants=definition.variants,
            exogenous_events=definition.exogenous_events,
            decision_trigger_kinds=definition.decision_trigger_kinds,
            weather=definition.weather,
            metadata=definition.metadata,
            schema_version="hailmary.scenario.v2",
        )
        return TrafficScenario(
            window=window,
            scale=cfg.global_scale,
            replicate=cfg.replicate,
            definition=definition,
            observed_cluster_counts=observed_counts,
            target_cluster_counts=target_counts,
            rejection_counts=tuple(self.rejection_counts.items()),
        )

    def build_batch(
        self,
        windows: Iterable[DemandWindow],
        *,
        scale_config: TrafficScaleConfig | None = None,
        include_empty: bool = False,
        source_partition: str = "",
    ) -> TrafficScenarioBatch:
        cfg = TrafficScaleConfig() if scale_config is None else scale_config
        scenarios = tuple(
            scenario
            for scenario in (self.build_scenario(window, scale_config=cfg) for window in windows)
            if include_empty or scenario.definition.flights
        )
        return TrafficScenarioBatch(
            dataset_id=self.dataset_id,
            scale_config=cfg,
            scenarios=scenarios,
            source_partition=source_partition,
            audit={
                "window_count": len(scenarios),
                "independent_overlapping_snapshots": True,
                "runway_totals_derived_from_cluster_targets": True,
                "rejection_counts": self.rejection_counts,
            },
        )


@dataclass(frozen=True, slots=True)
class TerminalEntryCorpus:
    dataset_id: str
    arrivals: tuple[ObservedArrival, ...]
    rejection_counts: tuple[tuple[str, int], ...]
    airport: str = ""
    schema_version: str = "hailmary.terminal_entry_corpus.v1"
    corpus_content_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != "hailmary.terminal_entry_corpus.v1":
            raise ValueError("unsupported terminal-entry corpus schema")
        if not str(self.dataset_id).strip():
            raise ValueError("terminal-entry corpus dataset_id must be non-empty")
        arrivals = tuple(
            sorted(
                self.arrivals,
                key=lambda item: (item.terminal_entry_time_s, item.flight_id),
            )
        )
        if any(item.dataset_id != self.dataset_id for item in arrivals):
            raise ValueError("terminal-entry corpus cannot mix datasets")
        if any(item.synthetic for item in arrivals):
            raise ValueError("terminal-entry corpus must contain observed arrivals only")
        if len({item.flight_id for item in arrivals}) != len(arrivals):
            raise ValueError("terminal-entry corpus flight IDs must be unique")
        rejections = tuple(
            sorted((str(key), int(value)) for key, value in self.rejection_counts)
        )
        if any(not key or value < 0 for key, value in rejections):
            raise ValueError("terminal-entry rejection counts must be named and non-negative")
        if len({key for key, _ in rejections}) != len(rejections):
            raise ValueError("terminal-entry rejection reasons must be unique")
        object.__setattr__(self, "arrivals", arrivals)
        object.__setattr__(self, "rejection_counts", rejections)
        object.__setattr__(self, "airport", str(self.airport).strip().upper())
        payload = self._content_payload()
        computed = content_hash(
            payload,
            namespace="hailmary.terminal_entry_corpus.v1",
        )
        if self.corpus_content_hash and self.corpus_content_hash != computed:
            raise ValueError("terminal-entry corpus content hash does not match content")
        object.__setattr__(self, "corpus_content_hash", computed)

    def _content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "airport": self.airport,
            "arrivals": [
                {
                    "airport": item.key.airport,
                    "runway": item.key.runway,
                    "cluster_id": item.key.cluster,
                    "flight_id": item.flight_id,
                    "terminal_entry_time_s": item.terminal_entry_time_s,
                    "terminal_entry_ground_speed_mps": (
                        item.terminal_entry_ground_speed_mps
                    ),
                    "terminal_entry_altitude_m": item.terminal_entry_altitude_m,
                    "baseline_variant_id": item.baseline_variant_id,
                    "callsign": item.callsign,
                    "icao24": item.icao24,
                    "source_day": item.source_day,
                }
                for item in self.arrivals
            ],
            "rejection_counts": dict(self.rejection_counts),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_payload(),
            "corpus_content_hash": self.corpus_content_hash,
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TerminalEntryCorpus":
        dataset_id = str(payload["dataset_id"])
        return cls(
            dataset_id=dataset_id,
            airport=str(payload.get("airport", "")),
            arrivals=tuple(
                ObservedArrival(
                    dataset_id=dataset_id,
                    key=ArrivalClusterKey(
                        str(item["airport"]),
                        str(item["runway"]),
                        str(item["cluster_id"]),
                    ),
                    flight_id=str(item["flight_id"]),
                    terminal_entry_time_s=float(item["terminal_entry_time_s"]),
                    terminal_entry_ground_speed_mps=float(
                        item["terminal_entry_ground_speed_mps"]
                    ),
                    terminal_entry_altitude_m=float(
                        item["terminal_entry_altitude_m"]
                    ),
                    baseline_variant_id=str(item["baseline_variant_id"]),
                    callsign=str(item.get("callsign", "")),
                    icao24=str(item.get("icao24", "")),
                    source_day=str(item.get("source_day", "")),
                )
                for item in payload["arrivals"]
            ),
            rejection_counts=tuple(
                (str(key), int(value))
                for key, value in dict(payload.get("rejection_counts", {})).items()
            ),
            schema_version=str(payload.get("schema_version", "")),
            corpus_content_hash=str(payload.get("corpus_content_hash", "")),
        )

    @classmethod
    def read(cls, path: str | Path) -> "TerminalEntryCorpus":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("terminal-entry corpus JSON must contain an object")
        return cls.from_dict(payload)


def build_terminal_entry_corpus(
    catalog: Iterable[CatalogArrival],
    tracks: Iterable[RawADSBTrack],
    *,
    dataset_id: str,
    airport: str,
    cluster_by_flight: Mapping[str, str] | None = None,
    variant_by_cluster: Mapping[str, str] | None = None,
    radius_nm: float = 50.0,
) -> TerminalEntryCorpus:
    """Join the arrival catalogue to raw ADS-B and reject unreconstructable entries.

    Missing cluster assignments receive a deterministic singleton cluster.  A
    caller can then compile that flight itself as the singleton medoid rather
    than silently dropping sparse runway traffic.
    """

    raw_tracks = tuple(tracks)
    track_by_id = {item.flight_id: item for item in raw_tracks}
    if len(track_by_id) != len(raw_tracks):
        raise ValueError("raw ADS-B track flight IDs must be unique")
    assignments = {} if cluster_by_flight is None else dict(cluster_by_flight)
    variants = {} if variant_by_cluster is None else dict(variant_by_cluster)
    rejections: Counter[str] = Counter()
    result: list[ObservedArrival] = []
    for arrival in sorted(catalog, key=lambda item: (item.event_time_s, item.flight_id)):
        track = track_by_id.get(arrival.flight_id)
        if track is None:
            rejections["missing_raw_track"] += 1
            continue
        entry = reconstruct_terminal_entry(
            track,
            LocalFrame(arrival.threshold_lat_deg, arrival.threshold_lon_deg),
            radius_nm=radius_nm,
        )
        if entry is None:
            rejections["terminal_entry_not_reconstructable"] += 1
            continue
        index = entry.segment_index
        dt = float(track.time_s[index + 1] - track.time_s[index])
        frame = LocalFrame(arrival.threshold_lat_deg, arrival.threshold_lon_deg)
        segment = frame.project_points(
            track.lat_deg[index : index + 2], track.lon_deg[index : index + 2]
        )
        ground_speed = float(np.linalg.norm(segment[1] - segment[0]) / dt)
        raw_cluster = assignments.get(arrival.flight_id)
        cluster = (
            str(raw_cluster)
            if raw_cluster is not None
            else stable_id("singleton", {"flight_id": arrival.flight_id}, length=16)
        )
        key = ArrivalClusterKey(airport, arrival.runway, cluster)
        variant_id = variants.get(key.qualified_id, variants.get(cluster, ""))
        if not variant_id:
            # The explicit sentinel is useful to the offline build command; the
            # hot TrafficScenarioBuilder still refuses to materialize it.
            variant_id = f"pending_medoid:{key.qualified_id}"
        result.append(
            ObservedArrival(
                dataset_id=dataset_id,
                key=key,
                flight_id=arrival.flight_id,
                terminal_entry_time_s=entry.time_s,
                terminal_entry_ground_speed_mps=ground_speed,
                terminal_entry_altitude_m=entry.geoaltitude_m,
                baseline_variant_id=variant_id,
                callsign=arrival.callsign,
                icao24=arrival.icao24,
                source_day=dataset_id,
            )
        )
    return TerminalEntryCorpus(
        dataset_id=dataset_id,
        arrivals=tuple(result),
        rejection_counts=tuple(sorted(rejections.items())),
        airport=airport,
    )


__all__ = [
    "ArrivalClusterKey",
    "ClusterCount",
    "DemandWindow",
    "DemandWindowConfig",
    "ObservedArrival",
    "TerminalEntryCorpus",
    "TrafficScaleConfig",
    "TrafficScenario",
    "TrafficScenarioBatch",
    "TrafficScenarioBuilder",
    "build_terminal_entry_corpus",
    "iter_demand_windows",
    "materialize_arrival_variant",
]
