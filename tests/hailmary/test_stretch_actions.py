from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hailmary.actions.stretch import PathStretchRealizer
from hailmary.adapters import SIMAPAdapter
from hailmary.config import M_PER_NM, StretchConfig
from hailmary.geometry.dogleg import construct_runway_away_dogleg
from hailmary.templates.models import VariantDiagnostics

from .test_templates import _straight_variant
from .test_adapters import _ConstantPerformanceBackend, _aircraft_config


def test_dogleg_meets_added_distance_and_chooses_runway_away_free_side() -> None:
    variant = _straight_variant()
    base = np.column_stack((variant.east_m, variant.north_m))
    north_side_medoid = np.column_stack(
        (
            np.linspace(50_000.0, 100_000.0, 40),
            np.full(40, 2_000.0),
        )
    )
    action_index = int(np.argmin(np.abs(variant.s_m - 80_000.0)))

    geometry = construct_runway_away_dogleg(
        base,
        variant.s_m,
        action_index=action_index,
        rejoin_span_m=10.0 * M_PER_NM,
        target_added_distance_m=2.0 * M_PER_NM,
        other_medoid_polylines_m=(north_side_medoid,),
        max_turn_deg=120.0,
        added_distance_tolerance_m=1e-4,
    )

    midpoint = 0.5 * (geometry.rejoin_point_m + geometry.action_point_m)
    assert geometry.realized_added_distance_m == pytest.approx(2.0 * M_PER_NM, abs=1e-4)
    assert np.linalg.norm(geometry.apex_point_m) > np.linalg.norm(midpoint)
    assert geometry.runway_away_displacement_m > 0.0
    assert geometry.apex_point_m[1] < 0.0  # away from the northern competing medoid
    assert not geometry.points_m.flags.writeable


def test_realizer_compiles_short_medium_long_to_target_distance() -> None:
    baseline = _straight_variant()
    settings = StretchConfig(max_turn_deg=120.0)
    realizer = PathStretchRealizer(config=settings)

    candidates = realizer.candidates(baseline, anchor_s_m=90_000.0)

    assert [candidate.name for candidate in candidates] == ["short", "medium", "long"]
    assert all(candidate.feasible for candidate in candidates)
    for candidate, target_nm in zip(candidates, settings.added_distance_nm, strict=True):
        assert candidate.variant is not None
        assert candidate.geometry is not None
        realized = candidate.variant.path_length_m - baseline.path_length_m
        assert realized == pytest.approx(target_nm * M_PER_NM, abs=settings.added_distance_tolerance_nm * M_PER_NM)
        assert candidate.geometry.runway_away_displacement_m > 0.0
        assert candidate.variant.duration_s >= baseline.duration_s


def test_smooth_doglegs_preserve_splices_and_never_fold_back() -> None:
    baseline = _straight_variant()
    base_tangent = np.asarray([1.0, 0.0])
    candidates = PathStretchRealizer().candidates(baseline, anchor_s_m=90_000.0)

    assert all(candidate.feasible for candidate in candidates)
    for candidate in candidates:
        assert candidate.geometry is not None
        assert candidate.station_mapping_m
        geometry = candidate.geometry
        points = geometry.points_m
        parent_s = geometry.parent_stations_m
        assert parent_s is not None
        assert np.array_equal(points[0], geometry.rejoin_point_m)
        assert np.array_equal(points[-1], geometry.action_point_m)
        assert np.all(np.diff(parent_s) > 0.0)

        segment = np.diff(points, axis=0)
        unit_tangent = segment / np.linalg.norm(segment, axis=1)[:, np.newaxis]
        # The analytic raised-cosine slope is zero at each endpoint; the
        # executable chord approximation remains within 1.5 degrees.
        assert np.dot(unit_tangent[0], base_tangent) > np.cos(np.deg2rad(1.5))
        assert np.dot(unit_tangent[-1], base_tangent) > np.cos(np.deg2rad(1.5))
        assert np.min(unit_tangent @ base_tangent) >= 0.45 - 1e-10

        mapping = np.asarray(candidate.station_mapping_m)
        assert np.all(np.diff(mapping[:, 0]) > 0.0)
        assert np.all(np.diff(mapping[:, 1]) > 0.0)
        action_parent_s = float(baseline.s_m[np.searchsorted(baseline.s_m, 90_000.0)])
        assert np.min(np.abs(mapping[:, 0] - action_parent_s)) <= 1e-8


def test_smooth_short_medium_long_compile_through_public_simap_replay() -> None:
    adapter = SIMAPAdapter(
        aircraft_config=_aircraft_config(),
        performance_backend=_ConstantPerformanceBackend(),
    )
    candidates = PathStretchRealizer(validator=adapter).candidates(
        _straight_variant(),
        anchor_s_m=90_000.0,
    )

    assert all(candidate.feasible for candidate in candidates)
    for candidate in candidates:
        assert candidate.variant is not None
        details = dict(candidate.variant.diagnostics.details)
        assert candidate.variant.diagnostics.feasible
        assert candidate.variant.diagnostics.message == "public SIMAP coupled replay validation passed"
        assert details["geometry_family"] == "raised_cosine_lateral_lane_change_v1"
        assert details["raw_max_bank_ratio"] <= adapter.maximum_raw_bank_ratio
        assert details["replay_final_threshold_error_m"] <= 0.10 * M_PER_NM

    first_variant = candidates[0].variant
    assert first_variant is not None
    calibrated = dict(first_variant.diagnostics.details)
    assert calibrated["simap_replay_pass_count"] == 2
    assert calibrated["simap_first_pass_threshold_error_m"] > calibrated[
        "replay_final_threshold_error_m"
    ]


def test_equal_outcome_scores_tie_break_short_then_medium_then_long() -> None:
    baseline = _straight_variant()
    realizer = PathStretchRealizer(config=StretchConfig(max_turn_deg=120.0))

    result = realizer.realize(
        baseline,
        anchor_s_m=90_000.0,
        outcome_evaluator=lambda _variant: 1.0,
    )

    assert result.chosen_name == "short"
    assert result.candidate_scores == (("short", 1.0), ("medium", 1.0), ("long", 1.0))


class _RejectShortValidator:
    def validate(self, variant):
        if variant.action_provenance.band == "short":
            return VariantDiagnostics(feasible=False, message="synthetic short failure")
        return replace(variant.diagnostics, feasible=True)


def test_infeasible_short_cannot_win_and_tie_falls_through_to_medium() -> None:
    baseline = _straight_variant()
    realizer = PathStretchRealizer(
        config=StretchConfig(max_turn_deg=120.0),
        validator=_RejectShortValidator(),
    )

    result = realizer.realize(
        baseline,
        anchor_s_m=90_000.0,
        outcome_evaluator=lambda _variant: 5.0,
    )

    assert result.chosen_name == "medium"
    short = next(item for item in result.candidates if item.name == "short")
    assert not short.feasible
    assert short.failure == "synthetic short failure"
