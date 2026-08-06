"""Action enumeration and application at an exact decision-event batch."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from hailmary.actions.models import (
    ActionCandidate,
    ActionIdentity,
    ActionLever,
    ActionRealization,
)
from hailmary.actions.vocabulary import ActionVocabulary, action_vocabulary
from hailmary.actions.speed import realize_speed_variant
from hailmary.actions.stretch import (
    PathStretchRealizer,
    StretchOutcomeEvaluation,
)
from hailmary.config import M_PER_NM, MPS_PER_KNOT, TemplateConfig
from hailmary.errors import InfeasibleActionError, StaleActionError
from hailmary.templates.models import TrajectoryVariant


def _future_conflicts(
    simulator: Any,
    *,
    start_time_s: float,
    horizon_s: float,
) -> tuple[object, ...]:
    """Evaluate branch conflicts intersecting the common inner horizon."""

    from hailmary.evaluation.conflict import TimedTrajectory, detect_conflicts

    state = getattr(simulator, "state")
    trajectories: list[TimedTrajectory] = []
    for dynamic in state.flights:
        variant = state.definition.variant(dynamic.current_variant_id)
        trajectories.append(
            TimedTrajectory.from_variant(
                variant,
                flight_id=dynamic.flight_id,
                release_time_s=dynamic.trajectory_clock_origin_s,
            )
        )
    interval_start = float(start_time_s)
    interval_end = float(horizon_s)
    if not np.isfinite(interval_start) or not np.isfinite(interval_end):
        raise ValueError("conflict evaluation interval must be finite")
    if interval_end < interval_start:
        raise ValueError("conflict evaluation interval is reversed")
    return tuple(
        record
        for record in detect_conflicts(trajectories)
        if record.end_time_s >= interval_start - 1.0e-9
        and record.start_time_s <= interval_end + 1.0e-9
    )


def _conflict_duration_by_pair(
    records: tuple[object, ...],
    *,
    start_time_s: float,
    horizon_s: float,
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for record in records:
        pair = tuple(getattr(record, "ordered_pair"))
        start = max(float(getattr(record, "start_time_s")), float(start_time_s))
        end = min(float(getattr(record, "end_time_s")), float(horizon_s))
        result[pair] = result.get(pair, 0.0) + max(0.0, end - start)
    return result


def _run_no_later_action_conflict_rollout(
    simulator: Any,
    *,
    start_time_s: float,
    horizon_s: float,
) -> dict[tuple[str, str], float]:
    """Advance one child while integrating conflicts on each event interval.

    Trajectory-origin shifts are event effects.  Reconstructing one trajectory
    after the child reaches its horizon would apply the last origin to the
    entire past.  Sampling every pre-event interval keeps those effects
    chronological and also makes the no-later-action rule explicit: event
    batches advance directly, with no policy callback.
    """

    interval_start = float(start_time_s)
    interval_end = float(horizon_s)
    if abs(float(simulator.state.sim_time_s) - interval_start) > 1e-8:
        raise ValueError(
            "inner conflict rollout must start at the frozen decision time"
        )
    totals: dict[tuple[str, str], float] = {}
    while float(simulator.state.sim_time_s) < interval_end - 1e-9:
        cursor = float(simulator.state.sim_time_s)
        next_event_time = simulator.next_event_time_s
        boundary = (
            interval_end
            if next_event_time is None
            else min(interval_end, float(next_event_time))
        )
        if boundary > cursor + 1e-12:
            interval_records = _future_conflicts(
                simulator,
                start_time_s=cursor,
                horizon_s=boundary,
            )
            for pair, duration_s in _conflict_duration_by_pair(
                interval_records,
                start_time_s=cursor,
                horizon_s=boundary,
            ).items():
                totals[pair] = totals.get(pair, 0.0) + duration_s
        if next_event_time is None or float(next_event_time) > interval_end + 1e-9:
            simulator.run_until(interval_end)
            break
        batch = simulator.advance_next()
        if batch is None:
            simulator.run_until(interval_end)
            break
    if abs(float(simulator.state.sim_time_s) - interval_end) > 1e-8:
        raise RuntimeError("inner conflict rollout did not reach its frozen horizon")
    return totals


def _variant_station_mapping_m(
    variant: TrajectoryVariant,
) -> tuple[tuple[float, float], ...]:
    metadata = dict(variant.action_provenance.realization_metadata)
    raw_mapping = metadata.get("parent_to_variant_station_mapping_m")
    if raw_mapping is None:
        raise InfeasibleActionError(
            "path-stretch variant is missing its physical station mapping"
        )
    try:
        return tuple((float(item[0]), float(item[1])) for item in raw_mapping)
    except (TypeError, ValueError, IndexError) as exc:
        raise InfeasibleActionError(
            "path-stretch variant has an invalid station mapping"
        ) from exc


def _automatic_simap_validator(current: TrajectoryVariant) -> object | None:
    """Reconstruct only the recorded version-1 A320 replay boundary.

    Synthetic and custom-aircraft variants deliberately stay kinematic unless
    the caller supplies an explicit validator.  This avoids silently changing
    scientific assumptions while ensuring actions derived from a replay-backed
    A320 template are themselves compiled/validated through the same public
    SIMAP boundary.
    """

    details = dict(current.diagnostics.details)
    if details.get("simap_replay_supported") is not True:
        return None
    if details.get("version1_default_aircraft_assumption") is not True:
        return None
    if str(details.get("aircraft_typecode", "")).upper() != "A320":
        return None
    from hailmary.adapters.simap import SIMAPAdapter

    payload_kg = float(details.get("payload_kg", 12_000.0))
    engine_value = details.get("engine_name")
    engine_name = None if engine_value in {None, ""} else str(engine_value)
    adapter = SIMAPAdapter(payload_kg=payload_kg)
    if (
        engine_name is not None
        and adapter.resolved_aircraft_config.engine_name != engine_name
    ):
        return None
    return adapter


def _default_stretch_outcome_evaluator(
    simulator: Any,
    action: ActionCandidate,
    realizer: PathStretchRealizer,
) -> Callable[[TrajectoryVariant], StretchOutcomeEvaluation]:
    """Build the frozen, no-later-action three-candidate rollout oracle."""

    from hailmary.evaluation.outcome import (
        score_simulator_outcome,
        simulator_outcome_plan,
    )
    from hailmary.features.anchors import build_current_segment_anchors

    parent_hash = str(getattr(simulator, "dynamic_content_hash"))
    state = getattr(simulator, "state")
    decision_time_s = float(state.sim_time_s)
    anchors = build_current_segment_anchors(simulator)
    try:
        anchor = next(
            item
            for item in anchors.leader_follower
            if item.anchor_id == action.anchor_id
        )
    except StopIteration as exc:
        raise StaleActionError(
            "path-stretch action anchor is stale for the current flow"
        ) from exc
    if anchor.follower_id != action.bound_flight_id:
        raise InfeasibleActionError(
            "path-stretch oracle requires the bound leader-follower aircraft"
        )
    plan = simulator_outcome_plan(simulator, anchor)
    baseline_branch = simulator.fork(label="stretch-inner:baseline")
    if str(baseline_branch.dynamic_content_hash) != parent_hash:
        raise RuntimeError("path-stretch baseline child did not match its parent")
    baseline_duration = _run_no_later_action_conflict_rollout(
        baseline_branch,
        start_time_s=decision_time_s,
        horizon_s=plan.horizon_s,
    )
    if str(getattr(simulator, "dynamic_content_hash")) != parent_hash:
        raise RuntimeError("path-stretch baseline rollout mutated its parent")

    def evaluate(variant: TrajectoryVariant) -> StretchOutcomeEvaluation:
        branch = simulator.fork(label=f"stretch-inner:{variant.action_provenance.band}")
        if str(branch.dynamic_content_hash) != parent_hash:
            raise RuntimeError("path-stretch candidate child did not match its parent")
        branch.install_variant(variant)
        branch.replace_flight_variant(
            action.bound_flight_id,
            variant.variant_id,
            action_lever=ActionLever.PATH_STRETCH.value,
            splice_s_m=action.s_m,
            station_mapping_m=_variant_station_mapping_m(variant),
            expected_version=branch.state.version,
        )
        candidate_duration = _run_no_later_action_conflict_rollout(
            branch,
            start_time_s=decision_time_s,
            horizon_s=plan.horizon_s,
        )
        outcome = score_simulator_outcome(
            branch,
            outcome_plan=plan,
        )
        new_pairs = tuple(sorted(set(candidate_duration).difference(baseline_duration)))
        excess_duration_s = sum(
            max(0.0, duration - baseline_duration.get(pair, 0.0))
            for pair, duration in candidate_duration.items()
        )
        conflict_penalty = (
            realizer.config.new_conflict_penalty * len(new_pairs)
            + excess_duration_s / realizer.config.conflict_duration_scale_s
        )
        if str(getattr(simulator, "dynamic_content_hash")) != parent_hash:
            raise RuntimeError("path-stretch inner rollout mutated its parent")
        return StretchOutcomeEvaluation(
            score=float(outcome.score - conflict_penalty),
            outcome_score=float(outcome.score),
            new_conflict_count=len(new_pairs),
            conflict_penalty=float(conflict_penalty),
            diagnostics=(
                ("new_conflict_pairs", new_pairs),
                ("excess_conflict_duration_s", float(excess_duration_s)),
                (
                    "baseline_conflict_pair_durations_s",
                    tuple(sorted(baseline_duration.items())),
                ),
                (
                    "candidate_conflict_pair_durations_s",
                    tuple(sorted(candidate_duration.items())),
                ),
                ("pair_score", float(outcome.pair_score)),
                ("propagation_score", float(outcome.propagation_score)),
                ("intervention_penalty", float(outcome.intervention_penalty)),
                ("throughput_score", float(outcome.throughput_score)),
                ("horizon_s", float(plan.horizon_s)),
                ("later_interventions_suppressed", True),
            ),
        )

    return evaluate


@dataclass(frozen=True, init=False)
class ActionCatalog:
    config: TemplateConfig

    def __init__(self, config: TemplateConfig | None = None) -> None:
        object.__setattr__(
            self, "config", TemplateConfig() if config is None else config
        )

    @property
    def vocabulary(self) -> ActionVocabulary:
        return action_vocabulary(self.config)

    def enumerate_for_batch(
        self,
        simulator: object,
        batch: object,
        *,
        anchor_id: str,
        bound_flight_id: str,
        resource_id: str,
        segment_id: str,
    ) -> tuple[ActionCandidate, ...]:
        state = getattr(simulator, "state")
        decision = getattr(batch, "decision_epoch", None)
        if decision is None:
            return ()
        if (
            getattr(decision, "state_id") != state.state_id
            or getattr(decision, "state_version") != state.version
        ):
            raise StaleActionError(
                "event batch no longer describes the current simulator state"
            )
        dynamic = state.flight(bound_flight_id)
        if str(getattr(dynamic.lifecycle, "value", dynamic.lifecycle)) != "active":
            return ()
        definition_flight = state.definition.flight(bound_flight_id)
        try:
            traversal = next(
                item
                for item in definition_flight.segment_traversals
                if item.segment_id == segment_id
            )
        except StopIteration as exc:
            raise InfeasibleActionError(
                "action flight is not committed to the bound segment"
            ) from exc
        if traversal.exit_resource_id != resource_id:
            raise InfeasibleActionError(
                "action resource is not the bound segment exit gate"
            )
        station_events = [
            event
            for event in getattr(batch, "events")
            if str(getattr(event.kind, "value", event.kind)) == "ACTION_STATION_CROSSED"
            and event.flight_id == bound_flight_id
        ]
        if not station_events:
            return ()
        current_variant = state.definition.variant(dynamic.current_variant_id)
        station_events.sort(
            key=lambda event: (
                0 if event.payload_dict.get("station_type") == "speed" else 1,
                event.station_index,
            )
        )
        vocabulary = self.vocabulary
        no_op_identity = vocabulary.identities[0]
        speed_identities = vocabulary.identities[1:-1]
        stretch_identity = vocabulary.identities[-1]
        first = station_events[0]
        candidates: list[ActionCandidate] = [
            self._candidate(
                state,
                anchor_id=anchor_id,
                flight_id=bound_flight_id,
                resource_id=resource_id,
                segment_id=segment_id,
                lever=no_op_identity.lever,
                band=no_op_identity.band,
                station_index=first.station_index,
                s_m=float(first.payload_dict["s_m"]),
            )
        ]
        for event in station_events:
            station_type = str(event.payload_dict.get("station_type", "speed"))
            station_s = float(event.payload_dict["s_m"])
            if (
                station_type == "speed"
                and dynamic.speed_action_count < self.config.max_speed_actions
            ):
                for identity, reduction in zip(
                    speed_identities,
                    vocabulary.speed_reductions_kts,
                    strict=True,
                ):
                    if isinstance(current_variant, TrajectoryVariant):
                        reference = float(
                            np.interp(
                                station_s,
                                current_variant.s_m,
                                current_variant.reference_command_cas_mps,
                            )
                        )
                        current_command = float(
                            np.interp(
                                station_s,
                                current_variant.s_m,
                                current_variant.command_cas_mps,
                            )
                        )
                        lower = float(
                            np.interp(
                                station_s,
                                current_variant.s_m,
                                current_variant.lower_cas_mps,
                            )
                        )
                        requested = reference - reduction * MPS_PER_KNOT
                        realized = max(requested, lower)
                        effective_kts = (current_command - realized) / MPS_PER_KNOT
                        if (
                            effective_kts
                            < self.config.min_effective_reduction_kts - 1e-9
                        ):
                            continue
                    candidates.append(
                        self._candidate(
                            state,
                            anchor_id=anchor_id,
                            flight_id=bound_flight_id,
                            resource_id=resource_id,
                            segment_id=segment_id,
                            lever=identity.lever,
                            band=identity.band,
                            station_index=event.station_index,
                            s_m=station_s,
                            metadata=(("reduction_kts", reduction),),
                        )
                    )
            if (
                station_type == "path_stretch"
                and dynamic.path_stretch_count < self.config.max_path_stretches
                and station_s > traversal.entry_s_m + 1.0e-6
            ):
                candidates.append(
                    self._candidate(
                        state,
                        anchor_id=anchor_id,
                        flight_id=bound_flight_id,
                        resource_id=resource_id,
                        segment_id=segment_id,
                        lever=stretch_identity.lever,
                        band=stretch_identity.band,
                        station_index=event.station_index,
                        s_m=station_s,
                        metadata=(("segment_entry_s_m", traversal.entry_s_m),),
                    )
                )
        return tuple(candidates)

    @staticmethod
    def _candidate(
        state: object,
        *,
        anchor_id: str,
        flight_id: str,
        resource_id: str,
        segment_id: str,
        lever: ActionLever,
        band: str,
        station_index: int,
        s_m: float,
        metadata: tuple[tuple[str, object], ...] = (),
    ) -> ActionCandidate:
        return ActionCandidate(
            anchor_id=anchor_id,
            bound_flight_id=flight_id,
            resource_id=resource_id,
            segment_id=segment_id,
            lever=lever,
            band=band,
            state_id=getattr(state, "state_id"),
            state_version=int(getattr(state, "version")),
            epoch_index=int(getattr(state, "decision_epoch_index")),
            station_index=int(station_index),
            s_m=float(s_m),
            dynamic_content_hash=str(getattr(state, "dynamic_content_hash", "")),
            realization_metadata=metadata,
        )


def _install_variant(simulator: object, variant: TrajectoryVariant) -> None:
    method = getattr(simulator, "install_variant", None)
    if callable(method):
        method(variant)
        return
    # Compatibility with the initial engine contract: replace the immutable
    # definition on this branch, sharing every existing array by identity.
    from hailmary.scenario.models import ScenarioDefinition
    from hailmary.simulator.state import evolve_state

    state = getattr(simulator, "state")
    definition = state.definition
    if not isinstance(definition, ScenarioDefinition):
        raise TypeError(
            "simulator definition does not support branch-local variant installation"
        )
    if variant.variant_id in definition.variant_ids:
        return
    new_definition = replace(definition, variants=(*definition.variants, variant))
    setattr(
        simulator,
        "state",
        evolve_state(
            state,
            transition=f"install-variant:{variant.variant_id}",
            definition=new_definition,
        ),
    )


def _increment_action_counter(
    simulator: object, flight_id: str, lever: ActionLever
) -> None:
    from hailmary.simulator.state import evolve_state

    state = getattr(simulator, "state")
    dynamic = state.flight(flight_id)
    updated = replace(
        dynamic,
        speed_action_count=dynamic.speed_action_count
        + (1 if lever is ActionLever.SPEED else 0),
        path_stretch_count=dynamic.path_stretch_count
        + (1 if lever is ActionLever.PATH_STRETCH else 0),
    )
    setattr(
        simulator,
        "state",
        evolve_state(
            state,
            transition=f"increment-action-counter:{flight_id}:{lever.value}",
            flights=tuple(
                updated if item.flight_id == flight_id else item
                for item in state.flights
            ),
        ),
    )


def apply_action(
    simulator: object,
    action: ActionCandidate,
    *,
    config: TemplateConfig | None = None,
    stretch_realizer: PathStretchRealizer | None = None,
    stretch_outcome_evaluator: Callable[[TrajectoryVariant], float] | None = None,
    stretch_selector: str | None = None,
    variant_validator: object | None = None,
) -> ActionRealization:
    """Apply one epoch-bound action and return its audited realization."""

    settings = config or TemplateConfig()
    identity = ActionIdentity(action.lever, action.band)
    if identity not in action_vocabulary(settings).identities:
        raise InfeasibleActionError(
            f"unsupported action identity {identity.lever.value}/{identity.band}"
        )
    action.assert_applicable(simulator)
    state = getattr(simulator, "state")
    dynamic = state.flight(action.bound_flight_id)
    definition_flight = state.definition.flight(action.bound_flight_id)
    try:
        traversal = next(
            item
            for item in definition_flight.segment_traversals
            if item.segment_id == action.segment_id
        )
    except StopIteration as exc:
        raise InfeasibleActionError(
            "action flight is no longer committed to the bound segment"
        ) from exc
    if traversal.exit_resource_id != action.resource_id:
        raise InfeasibleActionError("action is bound to the wrong segment resource")
    current = state.definition.variant(dynamic.current_variant_id)
    if not isinstance(current, TrajectoryVariant):
        raise TypeError("Hailmary actions require TrajectoryVariant inputs")
    automatic_validator = _automatic_simap_validator(current)
    resolved_validator = (
        variant_validator if variant_validator is not None else automatic_validator
    )
    validator_name = (
        "kinematic_only"
        if resolved_validator is None
        else f"{type(resolved_validator).__module__}.{type(resolved_validator).__qualname__}"
    )

    if action.lever is ActionLever.NO_OP:
        record_action = getattr(simulator, "record_action", None)
        if not callable(record_action):
            raise TypeError(
                "simulator must implement record_action for audited no-op actions"
            )
        record_action(
            {
                "action_id": action.action_id,
                "anchor_id": action.anchor_id,
                "flight_id": action.bound_flight_id,
                "resource_id": action.resource_id,
                "segment_id": action.segment_id,
                "lever": action.lever.value,
                "band": action.band,
                "time_s": state.sim_time_s,
            },
            expected_version=state.version,
        )
        return ActionRealization(
            action=action,
            variant_id=None,
            realized_delay_s=0.0,
            intervention_magnitude=0.0,
        )

    if action.lever is ActionLever.SPEED:
        station_mapping_m = None
        if dynamic.speed_action_count >= settings.max_speed_actions:
            raise InfeasibleActionError("aircraft has exhausted its two speed actions")
        reductions = dict(
            zip(settings.speed_band_names, settings.speed_reduction_kts, strict=True)
        )
        if action.band not in reductions:
            raise InfeasibleActionError(f"unknown speed band {action.band!r}")
        variant = realize_speed_variant(
            current,
            anchor_s_m=action.s_m,
            band=action.band,
            reduction_kts=reductions[action.band],
            min_effective_reduction_kts=settings.min_effective_reduction_kts,
            validator=resolved_validator,  # type: ignore[arg-type]
        )
        magnitude = float(reductions[action.band])
        audit: tuple[tuple[str, object], ...] = (
            ("reduction_kts", magnitude),
            ("variant_validator", validator_name),
        )
    else:
        if dynamic.path_stretch_count >= settings.max_path_stretches:
            raise InfeasibleActionError("aircraft has exhausted its one path stretch")
        if stretch_realizer is None:
            raise InfeasibleActionError(
                "path-stretch action requires a configured realization layer"
            )
        stretch_validator = (
            variant_validator
            if variant_validator is not None
            else stretch_realizer.validator or automatic_validator
        )
        if stretch_validator is not stretch_realizer.validator:
            stretch_realizer = replace(
                stretch_realizer,
                validator=stretch_validator,  # type: ignore[arg-type]
            )
        selected_stretch_selector = stretch_selector or stretch_realizer.config.selector
        resolved_stretch_evaluator = stretch_outcome_evaluator
        if (
            selected_stretch_selector == stretch_realizer.config.selector
            and resolved_stretch_evaluator is None
        ):
            resolved_stretch_evaluator = _default_stretch_outcome_evaluator(
                simulator,
                action,
                stretch_realizer,
            )
        realized = stretch_realizer.realize(
            current,
            anchor_s_m=action.s_m,
            minimum_rejoin_station_m=traversal.entry_s_m,
            outcome_evaluator=resolved_stretch_evaluator,
            selector=selected_stretch_selector,
        )
        variant = realized.variant
        station_mapping_m = realized.station_mapping_m
        magnitude = float(variant.action_provenance.added_distance_m / M_PER_NM)
        audit = (
            ("selector", realized.selector),
            (
                "variant_validator",
                "kinematic_only"
                if stretch_realizer.validator is None
                else (
                    f"{type(stretch_realizer.validator).__module__}."
                    f"{type(stretch_realizer.validator).__qualname__}"
                ),
            ),
            ("chosen_variant", realized.chosen_name),
            ("candidate_scores", realized.candidate_scores),
            (
                "candidate_failures",
                tuple(
                    (item.name, item.failure)
                    for item in realized.candidates
                    if item.failure is not None
                ),
            ),
            (
                "candidate_diagnostics",
                tuple(
                    (item.name, item.selection_diagnostics)
                    for item in realized.candidates
                ),
            ),
        )

    delay = max(0.0, variant.duration_s - current.duration_s)
    _install_variant(simulator, variant)
    installed_state = getattr(simulator, "state")
    replace_method = getattr(simulator, "replace_flight_variant")
    normalized_lever = ActionLever(action.lever)
    try:
        replace_method(
            action.bound_flight_id,
            variant.variant_id,
            action_id=action.action_id,
            action_lever=normalized_lever.value,
            splice_s_m=action.s_m,
            station_mapping_m=station_mapping_m,
            expected_version=installed_state.version,
        )
    except TypeError as exc:
        if not any(
            keyword in str(exc)
            for keyword in ("action_lever", "splice_s_m", "station_mapping_m")
        ):
            raise
        replace_method(
            action.bound_flight_id,
            variant.variant_id,
            action_id=action.action_id,
            expected_version=installed_state.version,
        )
        _increment_action_counter(simulator, action.bound_flight_id, normalized_lever)
    return ActionRealization(
        action=action,
        variant_id=variant.variant_id,
        realized_delay_s=delay,
        intervention_magnitude=magnitude,
        audit=audit,
    )
