"""Bound-pair plus next-three-trailers semi-local outcome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from hailmary.config import OutcomeConfig
from hailmary.evaluation.spacing import spacing_ratio_score

if TYPE_CHECKING:
    from hailmary.features.anchors import LeaderFollowerAnchor


@dataclass(frozen=True)
class OutcomeCohort:
    """Role IDs frozen at the parent epoch for fair paired comparisons."""

    anchor_id: str
    resource_id: str
    leader_id: str
    follower_id: str
    trailer_ids: tuple[str, ...]
    required_interval_s: float = 90.0

    def __post_init__(self) -> None:
        identifiers = (self.leader_id, self.follower_id, *self.trailer_ids)
        if (
            not self.anchor_id
            or not self.resource_id
            or any(not item for item in identifiers)
        ):
            raise ValueError("outcome cohort IDs cannot be empty")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("outcome cohort flights must be distinct")
        if not np.isfinite(self.required_interval_s) or self.required_interval_s <= 0.0:
            raise ValueError("required_interval_s must be finite and positive")

    @property
    def ordered_flight_ids(self) -> tuple[str, ...]:
        return (self.leader_id, self.follower_id, *self.trailer_ids)

    @property
    def effective_trailer_count(self) -> int:
        return len(self.trailer_ids)

    @property
    def evaluated_edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            zip(self.ordered_flight_ids, self.ordered_flight_ids[1:], strict=False)
        )


def freeze_outcome_cohort(
    anchor: LeaderFollowerAnchor,
    ordered_flight_ids: Sequence[str],
    *,
    trailer_count: int = 3,
    required_interval_s: float = 90.0,
) -> OutcomeCohort:
    """Freeze the original pair and at most ``k`` following trailers."""

    if trailer_count < 0:
        raise ValueError("trailer_count cannot be negative")
    ordered = tuple(str(item) for item in ordered_flight_ids)
    try:
        leader_index = ordered.index(anchor.leader_id)
        follower_index = ordered.index(anchor.follower_id)
    except ValueError as exc:
        raise ValueError("anchor flights are not both present in the flow") from exc
    if follower_index != leader_index + 1:
        raise ValueError("leader/follower anchor must be adjacent in the parent flow")
    trailers = ordered[follower_index + 1 : follower_index + 1 + trailer_count]
    return OutcomeCohort(
        anchor_id=anchor.anchor_id,
        resource_id=anchor.resource_id,
        leader_id=anchor.leader_id,
        follower_id=anchor.follower_id,
        trailer_ids=trailers,
        required_interval_s=float(required_interval_s),
    )


def rollout_horizon_s(
    crossing_times_s: Mapping[str, float],
    cohort: OutcomeCohort,
    *,
    nominal_slot_s: float | None = None,
) -> float:
    """One nominal slot after trailer 3, the last trailer, or the follower."""

    slot = (
        cohort.required_interval_s if nominal_slot_s is None else float(nominal_slot_s)
    )
    if not np.isfinite(slot) or slot <= 0.0:
        raise ValueError("nominal_slot_s must be finite and positive")
    terminal_id = cohort.trailer_ids[-1] if cohort.trailer_ids else cohort.follower_id
    try:
        crossing = float(crossing_times_s[terminal_id])
    except KeyError as exc:
        raise ValueError(
            f"missing crossing time for horizon flight {terminal_id!r}"
        ) from exc
    if not np.isfinite(crossing):
        raise ValueError("horizon crossing time must be finite")
    return float(crossing + slot)


@dataclass(frozen=True)
class InterventionSummary:
    """Branch-local non-no-op counts and realized magnitudes."""

    action_count: int = 0
    speed_action_count: int = 0
    total_speed_reduction_kts: float = 0.0
    stretch_action_count: int = 0
    total_stretch_added_distance_nm: float = 0.0

    def __post_init__(self) -> None:
        counts = (self.action_count, self.speed_action_count, self.stretch_action_count)
        if any(type(value) is not int for value in counts):
            raise TypeError("intervention counts must be exact non-bool integers")
        if any(value < 0 for value in counts):
            raise ValueError("intervention counts cannot be negative")
        if self.speed_action_count + self.stretch_action_count > self.action_count:
            raise ValueError("lever counts cannot exceed total action_count")
        magnitudes = (
            self.total_speed_reduction_kts,
            self.total_stretch_added_distance_nm,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in magnitudes):
            raise ValueError("intervention magnitudes must be finite and non-negative")

    def difference(self, baseline: "InterventionSummary") -> "InterventionSummary":
        """Return interventions added after a frozen cumulative baseline.

        Both summaries describe cumulative trajectory lineage. A final branch
        cannot contain fewer interventions, or less realized magnitude, than
        the parent from which it was forked. Rejecting such a delta keeps a
        mismatched outcome plan from silently reducing rollout parsimony.
        """

        if not isinstance(baseline, InterventionSummary):
            raise TypeError("baseline must be an InterventionSummary")
        values: dict[str, int | float] = {
            "action_count": self.action_count - baseline.action_count,
            "speed_action_count": self.speed_action_count - baseline.speed_action_count,
            "total_speed_reduction_kts": (
                self.total_speed_reduction_kts - baseline.total_speed_reduction_kts
            ),
            "stretch_action_count": self.stretch_action_count
            - baseline.stretch_action_count,
            "total_stretch_added_distance_nm": (
                self.total_stretch_added_distance_nm
                - baseline.total_stretch_added_distance_nm
            ),
        }
        negative_fields = tuple(name for name, value in values.items() if value < 0)
        if negative_fields:
            raise ValueError(
                "intervention summary has negative rollout deltas for "
                + ", ".join(negative_fields)
            )
        return InterventionSummary(
            action_count=int(values["action_count"]),
            speed_action_count=int(values["speed_action_count"]),
            total_speed_reduction_kts=float(values["total_speed_reduction_kts"]),
            stretch_action_count=int(values["stretch_action_count"]),
            total_stretch_added_distance_nm=float(
                values["total_stretch_added_distance_nm"]
            ),
        )

    def normalized_cost(
        self, config: OutcomeConfig
    ) -> tuple[float, Mapping[str, float]]:
        # Version 1 has at most two speed actions and one stretch: three total.
        count_component = float(np.clip(self.action_count / 3.0, 0.0, 1.0))
        speed_component = (
            float(
                np.clip(
                    (self.total_speed_reduction_kts / self.speed_action_count)
                    / config.speed_magnitude_normalizer_kts,
                    0.0,
                    1.0,
                )
            )
            if self.speed_action_count
            else 0.0
        )
        stretch_component = (
            float(
                np.clip(
                    (self.total_stretch_added_distance_nm / self.stretch_action_count)
                    / config.stretch_magnitude_normalizer_nm,
                    0.0,
                    1.0,
                )
            )
            if self.stretch_action_count
            else 0.0
        )
        normalized = float(
            (count_component + speed_component + stretch_component) / 3.0
        )
        return normalized, {
            "count_component": count_component,
            "speed_magnitude_component": speed_component,
            "stretch_magnitude_component": stretch_component,
        }


@dataclass(frozen=True)
class EdgeOutcome:
    leader_id: str
    follower_id: str
    predicted_interval_s: float
    score: float
    dynamically_feasible: bool


@dataclass(frozen=True)
class SemiLocalOutcome:
    score: float
    pair_score: float
    propagation_score: float
    intervention_penalty: float
    throughput_score: float
    pair_term: float
    propagation_term: float
    intervention_term: float
    throughput_term: float
    edge_outcomes: tuple[EdgeOutcome, ...]
    effective_trailer_count: int
    horizon_s: float
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class SimulatorOutcomePlan:
    """Parent-frozen cohort, baseline crossings, and common rollout horizon."""

    cohort: OutcomeCohort
    horizon_s: float
    baseline_crossing_times_s: tuple[tuple[str, float], ...]
    root_dynamic_content_hash: str
    baseline_intervention_summary: InterventionSummary = InterventionSummary()
    root_time_s: float = 0.0

    def __post_init__(self) -> None:
        if (
            type(self.root_dynamic_content_hash) is not str
            or not self.root_dynamic_content_hash
            or self.root_dynamic_content_hash != self.root_dynamic_content_hash.strip()
        ):
            raise ValueError(
                "outcome-plan root_dynamic_content_hash must be a non-empty exact string"
            )
        if not np.isfinite(self.horizon_s):
            raise ValueError("outcome-plan horizon must be finite")
        if not np.isfinite(self.root_time_s):
            raise ValueError("outcome-plan root time must be finite")
        if self.horizon_s < self.root_time_s - 1.0e-9:
            raise ValueError("outcome-plan horizon cannot precede its root time")
        if (
            tuple(flight_id for flight_id, _ in self.baseline_crossing_times_s)
            != self.cohort.ordered_flight_ids
        ):
            raise ValueError(
                "baseline crossing order must match the frozen outcome cohort"
            )
        if not isinstance(self.baseline_intervention_summary, InterventionSummary):
            raise TypeError(
                "baseline_intervention_summary must be an InterventionSummary"
            )

    @property
    def baseline_crossing_times(self) -> dict[str, float]:
        return dict(self.baseline_crossing_times_s)


def _throughput_score(
    intervals_s: Sequence[float], *, required_s: float, normalizer_s: float
) -> float:
    if not intervals_s:
        return 0.0
    inefficient_gaps = [
        max(0.0, float(interval) - required_s) for interval in intervals_s
    ]
    mean_excess = float(np.mean(inefficient_gaps))
    return float(-np.clip(mean_excess / normalizer_s, 0.0, 1.0))


def score_semi_local_outcome(
    cohort: OutcomeCohort,
    *,
    crossing_times_s: Mapping[str, float],
    feasible_by_flight: Mapping[str, bool] | None = None,
    intervention: InterventionSummary | None = None,
    config: OutcomeConfig | None = None,
    horizon_s: float | None = None,
) -> SemiLocalOutcome:
    """Score the frozen bound pair and propagation through up to three trailers."""

    cfg = OutcomeConfig() if config is None else config
    feasibility = {} if feasible_by_flight is None else feasible_by_flight
    times: dict[str, float] = {}
    for flight_id in cohort.ordered_flight_ids:
        try:
            crossing = float(crossing_times_s[flight_id])
        except KeyError as exc:
            raise ValueError(f"missing crossing time for {flight_id!r}") from exc
        if not np.isfinite(crossing):
            raise ValueError(f"non-finite crossing time for {flight_id!r}")
        times[flight_id] = crossing

    edges: list[EdgeOutcome] = []
    for leader_id, follower_id in cohort.evaluated_edges:
        interval = float(times[follower_id] - times[leader_id])
        dynamically_feasible = bool(
            feasibility.get(leader_id, True) and feasibility.get(follower_id, True)
        )
        edges.append(
            EdgeOutcome(
                leader_id=leader_id,
                follower_id=follower_id,
                predicted_interval_s=interval,
                score=spacing_ratio_score(
                    interval,
                    cohort.required_interval_s,
                    dynamically_feasible=dynamically_feasible,
                ),
                dynamically_feasible=dynamically_feasible,
            )
        )

    pair_score = float(edges[0].score)
    propagation_scores = [edge.score for edge in edges[1:]]
    propagation_score = (
        float(np.mean(propagation_scores)) if propagation_scores else 0.0
    )
    summary = InterventionSummary() if intervention is None else intervention
    intervention_penalty, intervention_components = summary.normalized_cost(cfg)
    throughput_score = _throughput_score(
        [edge.predicted_interval_s for edge in edges],
        required_s=cohort.required_interval_s,
        normalizer_s=cfg.throughput_gap_normalizer_s,
    )

    pair_term = float(cfg.pair_weight * pair_score)
    propagation_term = float(cfg.propagation_weight * propagation_score)
    intervention_term = float(-cfg.intervention_weight * intervention_penalty)
    throughput_term = float(cfg.throughput_weight * throughput_score)
    score = float(pair_term + propagation_term + intervention_term + throughput_term)
    effective_horizon = (
        rollout_horizon_s(times, cohort) if horizon_s is None else float(horizon_s)
    )
    if not np.isfinite(effective_horizon):
        raise ValueError("horizon_s must be finite")
    return SemiLocalOutcome(
        score=score,
        pair_score=pair_score,
        propagation_score=propagation_score,
        intervention_penalty=intervention_penalty,
        throughput_score=throughput_score,
        pair_term=pair_term,
        propagation_term=propagation_term,
        intervention_term=intervention_term,
        throughput_term=throughput_term,
        edge_outcomes=tuple(edges),
        effective_trailer_count=cohort.effective_trailer_count,
        horizon_s=effective_horizon,
        diagnostics={
            "weights": {
                "pair": cfg.pair_weight,
                "propagation": cfg.propagation_weight,
                "intervention": cfg.intervention_weight,
                "throughput": cfg.throughput_weight,
            },
            "intervention_components": intervention_components,
        },
    )


@runtime_checkable
class OutcomeQuery(Protocol):
    """Canonical branch query needed by the scorer adapter."""

    def crossing_time_s(
        self, state: Any, flight_id: str, resource_id: str
    ) -> float: ...

    def completed_feasibly(self, state: Any, flight_id: str) -> bool: ...

    def intervention_summary(self, state: Any) -> InterventionSummary: ...


def score_outcome_from_query(
    state: Any,
    cohort: OutcomeCohort,
    *,
    query: OutcomeQuery,
    config: OutcomeConfig | None = None,
    horizon_s: float | None = None,
) -> SemiLocalOutcome:
    times = {
        flight_id: query.crossing_time_s(state, flight_id, cohort.resource_id)
        for flight_id in cohort.ordered_flight_ids
    }
    feasible = {
        flight_id: bool(query.completed_feasibly(state, flight_id))
        for flight_id in cohort.ordered_flight_ids
    }
    return score_semi_local_outcome(
        cohort,
        crossing_times_s=times,
        feasible_by_flight=feasible,
        intervention=query.intervention_summary(state),
        config=config,
        horizon_s=horizon_s,
    )


def simulator_outcome_plan(
    simulator: Any,
    anchor: LeaderFollowerAnchor,
    *,
    config: OutcomeConfig | None = None,
) -> SimulatorOutcomePlan:
    """Freeze the currently available next-three-trailer rollout target."""

    from hailmary.errors import StaleActionError
    from hailmary.features.anchors import (
        build_current_segment_anchors,
        resource_eta_s,
    )

    cfg = OutcomeConfig() if config is None else config
    anchors = build_current_segment_anchors(simulator)
    if anchor.anchor_id not in {item.anchor_id for item in anchors.leader_follower}:
        raise StaleActionError(
            "outcome anchor is stale for the current threshold ordering"
        )
    resource = simulator.state.definition.resource(anchor.resource_id)
    flow = anchors.flow_for_segment(anchor.segment_id)
    cohort = freeze_outcome_cohort(
        anchor,
        flow.ordered_flight_ids,
        trailer_count=cfg.trailer_count,
        required_interval_s=float(resource.required_interval_s),
    )
    crossing_pairs = tuple(
        (
            flight_id,
            resource_eta_s(simulator, flight_id, anchor.resource_id),
        )
        for flight_id in cohort.ordered_flight_ids
    )
    horizon = rollout_horizon_s(dict(crossing_pairs), cohort)
    return SimulatorOutcomePlan(
        cohort=cohort,
        horizon_s=horizon,
        baseline_crossing_times_s=crossing_pairs,
        root_dynamic_content_hash=str(simulator.dynamic_content_hash),
        baseline_intervention_summary=simulator_intervention_summary(simulator),
        root_time_s=float(simulator.state.sim_time_s),
    )


def simulator_intervention_summary(simulator: Any) -> InterventionSummary:
    """Recover composed intervention counts/magnitudes from variant lineage."""

    from hailmary.config import M_PER_NM, MPS_PER_KNOT

    definition = simulator.state.definition
    action_count = 0
    speed_count = 0
    speed_reduction_kts = 0.0
    stretch_count = 0
    stretch_distance_nm = 0.0
    for dynamic in simulator.state.flights:
        variant = definition.variant(dynamic.current_variant_id)
        visited: set[str] = set()
        while True:
            variant_id = str(
                getattr(variant, "variant_id", getattr(variant, "content_hash", ""))
            )
            if variant_id in visited:
                raise ValueError("trajectory action provenance contains a cycle")
            visited.add(variant_id)
            provenance = getattr(variant, "action_provenance", None)
            if provenance is None:
                break
            lever = str(getattr(provenance, "lever", "baseline"))
            if lever == "speed":
                action_count += 1
                speed_count += 1
                speed_reduction_kts += (
                    float(getattr(provenance, "speed_reduction_mps", 0.0))
                    / MPS_PER_KNOT
                )
            elif lever == "path_stretch":
                action_count += 1
                stretch_count += 1
                stretch_distance_nm += (
                    float(getattr(provenance, "added_distance_m", 0.0)) / M_PER_NM
                )
            parent_variant_id = getattr(provenance, "parent_variant_id", None)
            if not parent_variant_id:
                break
            variant = definition.variant(str(parent_variant_id))
    return InterventionSummary(
        action_count=action_count,
        speed_action_count=speed_count,
        total_speed_reduction_kts=speed_reduction_kts,
        stretch_action_count=stretch_count,
        total_stretch_added_distance_nm=stretch_distance_nm,
    )


def score_simulator_outcome(
    simulator: Any,
    cohort: OutcomeCohort | SimulatorOutcomePlan | None = None,
    *,
    outcome_plan: SimulatorOutcomePlan | None = None,
    baseline_intervention_summary: InterventionSummary | None = None,
    config: OutcomeConfig | None = None,
    horizon_s: float | None = None,
) -> SemiLocalOutcome:
    """Score a realized branch through its canonical current variants.

    Passing an outcome plan (either positionally or by keyword) subtracts its
    root intervention baseline before applying the parsimony term. Passing an
    OutcomeCohort retains the original cumulative-scoring behavior unless an
    explicit baseline intervention summary is supplied.
    """

    from hailmary.features.anchors import resource_eta_s

    positional_plan = cohort if isinstance(cohort, SimulatorOutcomePlan) else None
    if positional_plan is not None:
        if outcome_plan is not None and outcome_plan != positional_plan:
            raise ValueError("conflicting positional and keyword outcome plans")
        outcome_plan = positional_plan
        cohort = None
    if outcome_plan is not None:
        from hailmary.rollout.paired import dynamic_content_fingerprint

        if cohort is not None and cohort != outcome_plan.cohort:
            raise ValueError("outcome cohort conflicts with the frozen outcome plan")
        fork_origin_hash = getattr(
            simulator,
            "fork_origin_dynamic_content_hash",
            None,
        )
        scored_root_hash = fork_origin_hash or dynamic_content_fingerprint(simulator)
        if (
            type(scored_root_hash) is not str
            or scored_root_hash != outcome_plan.root_dynamic_content_hash
        ):
            raise ValueError(
                "outcome plan root dynamic-content hash does not match the scored branch root"
            )
        resolved_cohort = outcome_plan.cohort
        if (
            baseline_intervention_summary is not None
            and baseline_intervention_summary
            != outcome_plan.baseline_intervention_summary
        ):
            raise ValueError(
                "intervention baseline conflicts with the frozen outcome plan"
            )
        baseline_intervention_summary = outcome_plan.baseline_intervention_summary
        if horizon_s is not None and not np.isclose(
            float(horizon_s),
            outcome_plan.horizon_s,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("outcome horizon conflicts with the frozen outcome plan")
        horizon_s = outcome_plan.horizon_s
    else:
        if cohort is None:
            raise TypeError("provide an outcome cohort or outcome plan")
        resolved_cohort = cohort

    crossing_times = {
        flight_id: resource_eta_s(simulator, flight_id, resolved_cohort.resource_id)
        for flight_id in resolved_cohort.ordered_flight_ids
    }
    feasibility: dict[str, bool] = {}
    for flight_id in resolved_cohort.ordered_flight_ids:
        dynamic = simulator.state.flight(flight_id)
        variant = simulator.state.definition.variant(dynamic.current_variant_id)
        diagnostics = getattr(variant, "diagnostics", None)
        feasibility[flight_id] = bool(getattr(diagnostics, "feasible", True))
    intervention = simulator_intervention_summary(simulator)
    if baseline_intervention_summary is not None:
        intervention = intervention.difference(baseline_intervention_summary)
    return score_semi_local_outcome(
        resolved_cohort,
        crossing_times_s=crossing_times,
        feasible_by_flight=feasibility,
        intervention=intervention,
        config=config,
        horizon_s=horizon_s,
    )


__all__ = [
    "EdgeOutcome",
    "InterventionSummary",
    "OutcomeCohort",
    "OutcomeQuery",
    "SemiLocalOutcome",
    "SimulatorOutcomePlan",
    "freeze_outcome_cohort",
    "rollout_horizon_s",
    "score_outcome_from_query",
    "score_semi_local_outcome",
    "score_simulator_outcome",
    "simulator_intervention_summary",
    "simulator_outcome_plan",
]
