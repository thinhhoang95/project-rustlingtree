"""Local reachability and action-capacity maps.

Capacities intentionally consider only actions feasible at the *current*
station-crossing epoch. They do not look ahead to future stations or compose
remaining interventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from hailmary.ids import content_hash


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ReachabilityMap:
    eta_nominal_s: float
    eta_earliest_s: float
    eta_latest_speed_s: float
    eta_latest_path_s: float

    def __post_init__(self) -> None:
        for name in (
            "eta_nominal_s",
            "eta_earliest_s",
            "eta_latest_speed_s",
            "eta_latest_path_s",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        tolerance = 1e-9
        if self.eta_earliest_s > self.eta_nominal_s + tolerance:
            raise ValueError("eta_earliest_s cannot be later than the nominal ETA")
        if self.eta_latest_speed_s < self.eta_nominal_s - tolerance:
            raise ValueError("a slowdown-only latest speed ETA cannot precede nominal ETA")
        if self.eta_latest_path_s < self.eta_nominal_s - tolerance:
            raise ValueError("a path-stretch latest ETA cannot precede nominal ETA")

    @property
    def speed_capacity_s(self) -> float:
        return float(max(0.0, self.eta_latest_speed_s - self.eta_nominal_s))

    @property
    def path_capacity_s(self) -> float:
        return float(max(0.0, self.eta_latest_path_s - self.eta_nominal_s))


def _latest_local_eta(nominal_s: float, candidates: Iterable[float], *, lever: str) -> float:
    values = [nominal_s]
    for index, candidate in enumerate(candidates):
        value = _finite(candidate, name=f"{lever}_action_etas_s[{index}]")
        if value < nominal_s - 1e-9:
            raise ValueError(f"{lever} action ETA cannot precede slowdown-only nominal ETA")
        values.append(value)
    return float(max(values))


def compute_reachability_map(
    *,
    eta_nominal_s: float,
    eta_earliest_s: float,
    speed_action_etas_s: Iterable[float] = (),
    path_action_etas_s: Iterable[float] = (),
) -> ReachabilityMap:
    """Build a map from explicit current-epoch feasible action ETAs."""

    nominal = _finite(eta_nominal_s, name="eta_nominal_s")
    earliest = _finite(eta_earliest_s, name="eta_earliest_s")
    return ReachabilityMap(
        eta_nominal_s=nominal,
        eta_earliest_s=earliest,
        eta_latest_speed_s=_latest_local_eta(nominal, speed_action_etas_s, lever="speed"),
        eta_latest_path_s=_latest_local_eta(nominal, path_action_etas_s, lever="path"),
    )


@runtime_checkable
class LocalReachabilityQuery(Protocol):
    """Duck-typed query contract for a state/variant implementation."""

    def nominal_eta_s(self, state: Any, flight_id: str, resource_id: str) -> float: ...

    def earliest_eta_s(self, state: Any, flight_id: str, resource_id: str) -> float: ...

    def feasible_action_etas_s(
        self,
        state: Any,
        flight_id: str,
        resource_id: str,
        lever: str,
    ) -> Iterable[float]: ...


def reachability_from_query(
    state: Any,
    *,
    flight_id: str,
    resource_id: str,
    query: LocalReachabilityQuery,
) -> ReachabilityMap:
    return compute_reachability_map(
        eta_nominal_s=query.nominal_eta_s(state, flight_id, resource_id),
        eta_earliest_s=query.earliest_eta_s(state, flight_id, resource_id),
        speed_action_etas_s=query.feasible_action_etas_s(state, flight_id, resource_id, "speed"),
        path_action_etas_s=query.feasible_action_etas_s(state, flight_id, resource_id, "path_stretch"),
    )


def earliest_resource_eta_s(simulator: Any, flight_id: str, resource_id: str) -> float:
    """Numerically integrate the current variant's maximum feasible profile."""

    from hailmary.features.anchors import resource_eta_s, resource_station_m
    from hailmary.templates.speed import cas_to_tas

    state = simulator.state
    nominal = resource_eta_s(simulator, flight_id, resource_id)
    dynamic = state.flight(flight_id)
    variant = state.definition.variant(dynamic.current_variant_id)
    upper = getattr(variant, "upper_cas_mps", None)
    altitude = getattr(variant, "altitude_m", None)
    stations = getattr(variant, "s_m", None)
    if upper is None or altitude is None or stations is None:
        return nominal

    sample = simulator.sample_flight(flight_id)
    current_station = float(sample.s_m)
    target_station = resource_station_m(simulator, flight_id, resource_id)
    if current_station <= target_station + 1e-9:
        return nominal

    raw_s = np.asarray(stations, dtype=float)
    raw_upper = np.asarray(upper, dtype=float)
    raw_altitude = np.asarray(altitude, dtype=float)
    if not (raw_s.ndim == raw_upper.ndim == raw_altitude.ndim == 1):
        raise ValueError("variant station/envelope arrays must be one-dimensional")
    if not (len(raw_s) == len(raw_upper) == len(raw_altitude)):
        raise ValueError("variant station/envelope arrays must have equal length")
    order = np.argsort(raw_s, kind="stable")
    sorted_s = raw_s[order]
    sorted_upper = raw_upper[order]
    sorted_altitude = raw_altitude[order]
    interior = sorted_s[(sorted_s > target_station) & (sorted_s < current_station)]
    integration_s = np.unique(np.concatenate(([target_station], interior, [current_station])))
    upper_cas = np.interp(integration_s, sorted_s, sorted_upper)
    altitude_m = np.interp(integration_s, sorted_s, sorted_altitude)
    maximum_ground_speed = np.asarray(cas_to_tas(upper_cas, altitude_m), dtype=float)
    if np.any(~np.isfinite(maximum_ground_speed)) or np.any(maximum_ground_speed <= 0.0):
        raise ValueError("maximum feasible speed profile must be finite and positive")
    ds = np.diff(integration_s)
    mean_speed = 0.5 * (maximum_ground_speed[:-1] + maximum_ground_speed[1:])
    travel_time_s = float(np.sum(ds / mean_speed))
    return float(min(nominal, state.sim_time_s + travel_time_s))


def simulator_reachability_map(
    simulator: Any,
    *,
    flight_id: str,
    resource_id: str,
    action_candidates: Sequence[Any] = (),
    action_applier: Callable[[Any, Any], Any] | None = None,
    cache_namespace: str | None = "default",
) -> ReachabilityMap:
    """Evaluate only currently enumerated actions on isolated simulator forks."""

    from hailmary.errors import InfeasibleActionError
    from hailmary.features.anchors import resource_eta_s
    from hailmary.rollout.paired import (
        dynamic_content_fingerprint,
        rebind_action_to_branch,
    )

    candidate_ids = tuple(
        str(getattr(action, "action_id", ""))
        or content_hash(action, namespace="reachability-action")
        for action in action_candidates
    )
    cache_key = (
        str(getattr(simulator.state, "dynamic_content_hash", "")),
        str(flight_id),
        str(resource_id),
        candidate_ids,
        cache_namespace,
    )
    cache: dict[tuple[Any, ...], ReachabilityMap] | None = None
    if cache_namespace is not None:
        cache = getattr(simulator, "_hailmary_reachability_cache", None)
        if cache is None:
            cache = {}
            setattr(simulator, "_hailmary_reachability_cache", cache)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    nominal = resource_eta_s(simulator, flight_id, resource_id)
    earliest = earliest_resource_eta_s(simulator, flight_id, resource_id)
    speed_etas: list[float] = []
    path_etas: list[float] = []
    parent_hash = dynamic_content_fingerprint(simulator)

    for index, action in enumerate(action_candidates):
        if str(getattr(action, "bound_flight_id", flight_id)) != flight_id:
            continue
        if not bool(getattr(action, "feasible", True)):
            continue
        lever = getattr(action, "lever", "")
        lever_value = str(getattr(lever, "value", lever)).lower()
        if lever_value not in {"speed", "path_stretch"}:
            continue
        branch = simulator.fork(label=f"reachability:{lever_value}:{index}")
        rebound = rebind_action_to_branch(action, branch)
        try:
            if action_applier is None:
                branch.apply(rebound)
            else:
                action_applier(branch, rebound)
        except InfeasibleActionError:
            continue
        eta = resource_eta_s(branch, flight_id, resource_id)
        if lever_value == "speed":
            speed_etas.append(eta)
        else:
            path_etas.append(eta)

    if dynamic_content_fingerprint(simulator) != parent_hash:
        raise RuntimeError("reachability action evaluation mutated its parent simulator")
    result = compute_reachability_map(
        eta_nominal_s=nominal,
        eta_earliest_s=earliest,
        speed_action_etas_s=speed_etas,
        path_action_etas_s=path_etas,
    )
    if cache is not None:
        if len(cache) >= 512:
            cache.pop(next(iter(cache)))
        cache[cache_key] = result
    return result


__all__ = [
    "LocalReachabilityQuery",
    "ReachabilityMap",
    "compute_reachability_map",
    "earliest_resource_eta_s",
    "reachability_from_query",
    "simulator_reachability_map",
]
