"""Library-clearance doglegs and deterministic three-variant realization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from hailmary._arrays import readonly_float64
from hailmary.actions.splice import preserve_compiled_live_prefix
from hailmary.config import M_PER_NM, StretchConfig, TemplateConfig
from hailmary.errors import InfeasibleActionError
from hailmary.geometry.dogleg import DoglegGeometry, construct_runway_away_dogleg
from hailmary.geometry.frame import LocalFrame
from hailmary.geometry.polyline import cumulative_lengths_m
from hailmary.templates.models import (
    ActionProvenance,
    TrajectoryVariant,
    VariantDiagnostics,
)


class VariantValidator(Protocol):
    def validate(self, variant: TrajectoryVariant) -> VariantDiagnostics: ...


@dataclass(frozen=True)
class StretchCandidateResult:
    name: str
    variant: TrajectoryVariant | None
    geometry: DoglegGeometry | None
    score: float | None = None
    failure: str | None = None
    selection_diagnostics: tuple[tuple[str, Any], ...] = ()
    station_mapping_m: tuple[tuple[float, float], ...] = ()

    @property
    def feasible(self) -> bool:
        return (
            self.variant is not None
            and self.geometry is not None
            and self.failure is None
        )


@dataclass(frozen=True)
class StretchRealization:
    chosen_name: str
    variant: TrajectoryVariant
    candidates: tuple[StretchCandidateResult, ...]
    selector: str

    @property
    def candidate_scores(self) -> tuple[tuple[str, float | None], ...]:
        return tuple((candidate.name, candidate.score) for candidate in self.candidates)

    @property
    def station_mapping_m(self) -> tuple[tuple[float, float], ...]:
        """Monotone parent-to-child station mapping for the chosen dogleg."""

        chosen = next(
            candidate
            for candidate in self.candidates
            if candidate.name == self.chosen_name
        )
        if not chosen.station_mapping_m:
            raise RuntimeError("chosen path-stretch candidate has no station mapping")
        return chosen.station_mapping_m


@dataclass(frozen=True)
class StretchOutcomeEvaluation:
    """Serializable result from one identical-child inner rollout."""

    score: float
    outcome_score: float
    new_conflict_count: int = 0
    conflict_penalty: float = 0.0
    diagnostics: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        values = (self.score, self.outcome_score, self.conflict_penalty)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("stretch outcome values must be finite")
        if self.new_conflict_count < 0 or self.conflict_penalty < 0.0:
            raise ValueError("stretch conflict counts and penalties cannot be negative")

    @property
    def artifact_diagnostics(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("outcome_score", self.outcome_score),
            ("new_conflict_count", self.new_conflict_count),
            ("conflict_penalty", self.conflict_penalty),
            *self.diagnostics,
        )


def _finite_artifact_number(value: float) -> float | None:
    """Represent unbounded geometric clearance without serializing infinity."""

    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _variant_with_exact_station(
    current: TrajectoryVariant,
    station_s_m: float,
) -> TrajectoryVariant:
    """Return an equivalent profile whose grid contains ``station_s_m``."""

    station = float(station_s_m)
    if not 0.0 < station < current.path_length_m:
        raise InfeasibleActionError(
            "path-stretch anchor is outside the controllable trajectory"
        )
    nearest = int(np.argmin(np.abs(current.s_m - station)))
    if abs(float(current.s_m[nearest]) - station) <= 1e-7:
        return current
    insertion_index = int(np.searchsorted(current.s_m, station))
    stations = np.insert(current.s_m, insertion_index, station)

    def inserted(values: np.ndarray) -> np.ndarray:
        return np.insert(
            values, insertion_index, np.interp(station, current.s_m, values)
        )

    threshold_id = current.threshold_resource_id or "runway_threshold"
    resources = tuple(
        (crossing.resource_id, crossing.s_m)
        for crossing in current.resource_crossings
        if crossing.resource_id != threshold_id
    )
    return TrajectoryVariant.from_kinematic_profile(
        template_id=current.template_id,
        cluster_id=current.cluster_id,
        s_m=stations,
        lat_deg=inserted(current.lat_deg),
        lon_deg=inserted(current.lon_deg),
        east_m=inserted(current.east_m),
        north_m=inserted(current.north_m),
        altitude_m=inserted(current.altitude_m),
        cas_mps=inserted(current.cas_mps),
        tas_mps=inserted(current.tas_mps),
        ground_speed_mps=inserted(current.ground_speed_mps),
        command_cas_mps=inserted(current.command_cas_mps),
        reference_command_cas_mps=inserted(current.reference_command_cas_mps),
        lower_cas_mps=inserted(current.lower_cas_mps),
        upper_cas_mps=inserted(current.upper_cas_mps),
        threshold_resource_id=threshold_id,
        resource_stations_m=resources,
        diagnostics=current.diagnostics,
        action_provenance=current.action_provenance,
    )


def _compile_geometry_variant(
    current: TrajectoryVariant,
    geometry: DoglegGeometry,
    *,
    band: str,
    validator: VariantValidator | None,
    lineage_parent: TrajectoryVariant | None = None,
) -> tuple[TrajectoryVariant, tuple[tuple[float, float], ...]]:
    parent = current if lineage_parent is None else lineage_parent
    base_points = np.column_stack((current.east_m, current.north_m))
    rejoin = geometry.rejoin_index
    action = geometry.action_index
    corridor_points = geometry.points_m
    corridor_parent_s = geometry.parent_stations_m
    if corridor_parent_s is None:
        corridor_parent_s = np.asarray(
            (
                current.s_m[rejoin],
                0.5 * (current.s_m[rejoin] + current.s_m[action]),
                current.s_m[action],
            ),
            dtype=np.float64,
        )
    controls = np.vstack(
        (
            base_points[:rejoin],
            corridor_points,
            base_points[action + 1 :],
        )
    )
    control_base_s = np.concatenate(
        (
            current.s_m[:rejoin],
            corridor_parent_s,
            current.s_m[action + 1 :],
        )
    )
    control_distance = cumulative_lengths_m(controls)
    # Preserve every control vertex in the executable grid.  In particular,
    # the action point must be represented exactly so a live branch can splice
    # onto the dogleg without moving the aircraft at the action instant.
    action_control_index = rejoin + len(corridor_points) - 1
    action_child_s = float(control_distance[action_control_index])
    # Extra resolution is only needed on the new downstream dogleg.  Avoid
    # subdividing the already-flown upstream segments, which would perturb the
    # trapezoidal clock approximation before the live splice.
    uniform_s = np.linspace(
        0.0,
        action_child_s,
        max(len(current.s_m), len(controls), 128),
    )
    new_s = np.unique(np.concatenate((uniform_s, control_distance)))
    new_points = np.column_stack(
        (
            np.interp(new_s, control_distance, controls[:, 0]),
            np.interp(new_s, control_distance, controls[:, 1]),
        )
    )
    mapped_base_s = np.interp(new_s, control_distance, control_base_s)
    station_mapping = tuple(
        (float(parent_s), float(child_s))
        for parent_s, child_s in zip(control_base_s, control_distance, strict=True)
    )

    def mapped(values: np.ndarray) -> np.ndarray:
        return np.interp(mapped_base_s, current.s_m, values)

    altitude = mapped(current.altitude_m)
    command = mapped(current.command_cas_mps)
    reference = mapped(current.reference_command_cas_mps)
    lower = mapped(current.lower_cas_mps)
    upper = mapped(current.upper_cas_mps)
    command = np.clip(command, lower, upper)
    tas = mapped(current.tas_mps)
    ground = mapped(current.ground_speed_mps)
    frame = LocalFrame(float(current.lat_deg[0]), float(current.lon_deg[0]))
    lat, lon = frame.unproject(new_points[:, 0], new_points[:, 1])

    resource_stations: list[tuple[str, float]] = []
    threshold_id = current.threshold_resource_id or "runway_threshold"
    for crossing in current.resource_crossings:
        if abs(crossing.s_m) <= 1e-8:
            continue
        new_station = float(np.interp(crossing.s_m, control_base_s, control_distance))
        resource_stations.append((crossing.resource_id, new_station))

    realized_added = float(new_s[-1] - current.s_m[-1])
    diagnostics = VariantDiagnostics(
        feasible=True,
        message=f"compiled {band} runway-away path stretch",
        speed_source=current.diagnostics.speed_source,
        wind_model=current.diagnostics.wind_model,
        cas_derivation=current.diagnostics.cas_derivation,
        observed_duration_s=current.diagnostics.observed_duration_s,
        details=(
            ("geometry_family", "raised_cosine_lateral_lane_change_v1"),
            ("target_added_distance_m", geometry.target_added_distance_m),
            ("realized_added_distance_m", realized_added),
            (
                "minimum_other_medoid_clearance_m",
                _finite_artifact_number(geometry.medoid_clearance_m),
            ),
            ("runway_away_displacement_m", geometry.runway_away_displacement_m),
            ("azimuth_rad", geometry.azimuth_rad),
        ),
    )
    variant = TrajectoryVariant.from_kinematic_profile(
        template_id=current.template_id,
        cluster_id=current.cluster_id,
        s_m=new_s,
        lat_deg=lat,
        lon_deg=lon,
        east_m=new_points[:, 0],
        north_m=new_points[:, 1],
        altitude_m=altitude,
        cas_mps=command,
        tas_mps=tas,
        ground_speed_mps=ground,
        command_cas_mps=command,
        reference_command_cas_mps=reference,
        lower_cas_mps=lower,
        upper_cas_mps=upper,
        threshold_resource_id=threshold_id,
        resource_stations_m=tuple(resource_stations),
        diagnostics=diagnostics,
        action_provenance=ActionProvenance(
            lever="path_stretch",
            band=band,
            parent_variant_id=parent.variant_id,
            anchor_station_index=action,
            added_distance_m=realized_added,
            realization_metadata=(
                ("geometry_family", "raised_cosine_lateral_lane_change_v1"),
                (
                    "medoid_clearance_m",
                    _finite_artifact_number(geometry.medoid_clearance_m),
                ),
                ("runway_away_displacement_m", geometry.runway_away_displacement_m),
                ("parent_to_variant_station_mapping_m", station_mapping),
            ),
        ),
    )
    if validator is not None:
        compile_variant = getattr(validator, "compile", None)
        if callable(compile_variant):
            compiled = compile_variant(variant)
            if not isinstance(compiled, TrajectoryVariant):
                raise TypeError("variant compiler must return TrajectoryVariant")
            child_anchor_s_m = float(
                np.interp(
                    current.s_m[action],
                    np.asarray(station_mapping)[:, 0],
                    np.asarray(station_mapping)[:, 1],
                )
            )
            variant = preserve_compiled_live_prefix(
                current,
                compiled,
                parent_anchor_s_m=float(current.s_m[action]),
                child_anchor_s_m=child_anchor_s_m,
                station_mapping_m=station_mapping,
                envelope_tolerance_mps=float(
                    getattr(validator, "physical_envelope_tolerance_mps", 0.5)
                ),
            )
            validated = variant.diagnostics
        else:
            validated = validator.validate(variant)
            variant = replace(variant, diagnostics=validated, variant_id="")
        if not validated.feasible:
            raise InfeasibleActionError(validated.message)
    if variant.duration_s + 1e-8 < parent.duration_s:
        raise InfeasibleActionError("path stretch unexpectedly reduces traversal time")
    return variant, station_mapping


@dataclass(frozen=True)
class PathStretchRealizer:
    other_medoid_polylines_m: tuple[np.ndarray, ...] = ()
    boundary_polylines_m: tuple[np.ndarray, ...] = ()
    config: StretchConfig = StretchConfig()
    template_config: TemplateConfig = TemplateConfig()
    validator: VariantValidator | None = None

    def __post_init__(self) -> None:
        medoids = tuple(
            readonly_float64(
                item,
                name=f"other_medoid_polylines_m[{index}]",
                ndim=2,
            )
            for index, item in enumerate(self.other_medoid_polylines_m)
        )
        boundaries = tuple(
            readonly_float64(
                item,
                name=f"boundary_polylines_m[{index}]",
                ndim=2,
            )
            for index, item in enumerate(self.boundary_polylines_m)
        )
        if any(item.shape[1] != 2 for item in (*medoids, *boundaries)):
            raise ValueError("path-stretch geometry polylines must have shape (n, 2)")
        object.__setattr__(self, "other_medoid_polylines_m", medoids)
        object.__setattr__(self, "boundary_polylines_m", boundaries)

    def candidates(
        self,
        current: TrajectoryVariant,
        *,
        anchor_s_m: float,
        minimum_rejoin_station_m: float | None = None,
    ) -> tuple[StretchCandidateResult, ...]:
        geometry_source = _variant_with_exact_station(current, float(anchor_s_m))
        action_index = int(np.searchsorted(geometry_source.s_m, float(anchor_s_m)))
        base_points = np.column_stack((geometry_source.east_m, geometry_source.north_m))
        gate_m = self.template_config.commitment_gate_nm * M_PER_NM
        if minimum_rejoin_station_m is not None:
            requested_rejoin = float(minimum_rejoin_station_m)
            if not np.isfinite(requested_rejoin) or requested_rejoin < 0.0:
                raise ValueError("minimum_rejoin_station_m must be finite and non-negative")
            gate_m = max(gate_m, requested_rejoin)
        results: list[StretchCandidateResult] = []
        for name, configured_span_nm, added_nm in zip(
            self.config.variant_names,
            self.config.rejoin_span_nm,
            self.config.added_distance_nm,
            strict=True,
        ):
            available_span_m = float(geometry_source.s_m[action_index] - gate_m)
            rejoin_span_m = min(configured_span_nm * M_PER_NM, available_span_m)
            try:
                geometry = construct_runway_away_dogleg(
                    base_points,
                    geometry_source.s_m,
                    action_index=action_index,
                    rejoin_span_m=rejoin_span_m,
                    target_added_distance_m=added_nm * M_PER_NM,
                    other_medoid_polylines_m=self.other_medoid_polylines_m,
                    boundary_polylines_m=self.boundary_polylines_m,
                    candidate_azimuth_count=self.config.candidate_azimuth_count,
                    max_turn_deg=self.config.max_turn_deg,
                    added_distance_tolerance_m=self.config.added_distance_tolerance_nm
                    * M_PER_NM,
                    minimum_medoid_clearance_m=self.config.minimum_medoid_clearance_nm
                    * M_PER_NM,
                    minimum_rejoin_station_m=gate_m,
                )
                variant, station_mapping = _compile_geometry_variant(
                    geometry_source,
                    geometry,
                    band=name,
                    validator=self.validator,
                    lineage_parent=current,
                )
                error = abs(
                    (variant.path_length_m - current.path_length_m)
                    - added_nm * M_PER_NM
                )
                if error > self.config.added_distance_tolerance_nm * M_PER_NM + 1e-6:
                    raise InfeasibleActionError(
                        "compiled dogleg misses its added-distance tolerance"
                    )
                results.append(
                    StretchCandidateResult(
                        name=name,
                        variant=variant,
                        geometry=geometry,
                        station_mapping_m=station_mapping,
                    )
                )
            except (InfeasibleActionError, ValueError) as exc:
                results.append(
                    StretchCandidateResult(
                        name=name, variant=None, geometry=None, failure=str(exc)
                    )
                )
        return tuple(results)

    def realize(
        self,
        current: TrajectoryVariant,
        *,
        anchor_s_m: float,
        minimum_rejoin_station_m: float | None = None,
        outcome_evaluator: Callable[
            [TrajectoryVariant],
            float | StretchOutcomeEvaluation,
        ]
        | None = None,
        selector: str | None = None,
    ) -> StretchRealization:
        selection = selector or self.config.selector
        raw_candidates = self.candidates(
            current,
            anchor_s_m=anchor_s_m,
            minimum_rejoin_station_m=minimum_rejoin_station_m,
        )
        scored: list[StretchCandidateResult] = []
        for candidate in raw_candidates:
            if not candidate.feasible:
                scored.append(candidate)
                continue
            assert candidate.variant is not None and candidate.geometry is not None
            if selection == self.config.geometry_only_ablation_selector:
                clearance = candidate.geometry.medoid_clearance_m
                score = 1.0e30 if np.isinf(clearance) else float(clearance)
            elif selection == self.config.selector:
                if outcome_evaluator is None:
                    raise InfeasibleActionError(
                        "the default path-stretch selector requires a full semi-local outcome evaluator"
                    )
                evaluation = outcome_evaluator(candidate.variant)
                if isinstance(evaluation, StretchOutcomeEvaluation):
                    score = float(evaluation.score)
                    selection_diagnostics = evaluation.artifact_diagnostics
                else:
                    score = float(evaluation)
                    selection_diagnostics = ()
                if not np.isfinite(score):
                    raise InfeasibleActionError(
                        "path-stretch outcome evaluator returned a non-finite score"
                    )
                candidate = replace(
                    candidate,
                    selection_diagnostics=selection_diagnostics,
                )
            else:
                raise ValueError(f"unknown path-stretch selector {selection!r}")
            scored.append(replace(candidate, score=score))
        feasible = [
            candidate
            for candidate in scored
            if candidate.feasible and candidate.score is not None
        ]
        if not feasible:
            failures = "; ".join(f"{item.name}: {item.failure}" for item in scored)
            raise InfeasibleActionError(
                f"all path-stretch variants are infeasible ({failures})"
            )
        rank = {name: index for index, name in enumerate(self.config.variant_names)}
        chosen = max(
            feasible,
            key=lambda item: (
                float(item.score) if item.score is not None else -np.inf,
                -rank[item.name],
            ),
        )
        assert chosen.variant is not None
        audit = tuple(
            (
                candidate.name,
                {
                    "feasible": candidate.feasible,
                    "score": candidate.score,
                    "failure": candidate.failure,
                    "diagnostics": candidate.selection_diagnostics,
                },
            )
            for candidate in scored
        )
        provenance = replace(
            chosen.variant.action_provenance,
            realization_metadata=(
                *chosen.variant.action_provenance.realization_metadata,
                ("selector", selection),
                ("candidate_scores", audit),
                ("chosen_variant", chosen.name),
            ),
        )
        logged_variant = replace(
            chosen.variant, action_provenance=provenance, variant_id=""
        )
        scored = [
            replace(item, variant=logged_variant) if item.name == chosen.name else item
            for item in scored
        ]
        return StretchRealization(
            chosen_name=chosen.name,
            variant=logged_variant,
            candidates=tuple(scored),
            selector=selection,
        )


def medoid_polylines(
    variants: Iterable[TrajectoryVariant], *, exclude_cluster_id: str
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.column_stack((variant.east_m, variant.north_m))
        for variant in variants
        if variant.cluster_id != exclude_cluster_id
    )
