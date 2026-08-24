from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hailmary.actions.stretch import PathStretchRealizer
from hailmary.adapters import SIMAPAdapter
from hailmary.config import M_PER_NM, StretchConfig
from hailmary.geometry.dogleg import construct_runway_away_dogleg
from hailmary.geometry.frame import LocalFrame
from hailmary.simulator import MonotoneTrajectory
from hailmary.templates.models import TrajectoryVariant, VariantDiagnostics

from .test_templates import _straight_variant
from .test_adapters import _ConstantPerformanceBackend, _aircraft_config


def test_realizer_detaches_and_freezes_mutable_geometry_inputs() -> None:
    medoid = np.asarray(((0.0, 1.0), (2.0, 3.0)), dtype=np.float64)
    boundary = np.asarray(((4.0, 5.0), (6.0, 7.0)), dtype=np.float64)
    expected_medoid = medoid.copy()
    expected_boundary = boundary.copy()

    realizer = PathStretchRealizer(
        other_medoid_polylines_m=(medoid,),
        boundary_polylines_m=(boundary,),
    )

    medoid[0, 0] = 100.0
    boundary[0, 0] = 200.0
    stored_medoid = realizer.other_medoid_polylines_m[0]
    stored_boundary = realizer.boundary_polylines_m[0]
    assert np.array_equal(stored_medoid, expected_medoid)
    assert np.array_equal(stored_boundary, expected_boundary)
    assert not np.shares_memory(stored_medoid, medoid)
    assert not np.shares_memory(stored_boundary, boundary)
    assert not stored_medoid.flags.writeable
    assert not stored_boundary.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        stored_medoid[0, 0] = -1.0
    with pytest.raises(ValueError, match="read-only"):
        stored_boundary[0, 0] = -1.0


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
    for candidate, target_nm in zip(
        candidates, settings.added_distance_nm, strict=True
    ):
        assert candidate.variant is not None
        assert candidate.geometry is not None
        realized = candidate.variant.path_length_m - baseline.path_length_m
        assert realized == pytest.approx(
            target_nm * M_PER_NM, abs=settings.added_distance_tolerance_nm * M_PER_NM
        )
        assert candidate.geometry.runway_away_displacement_m > 0.0
        assert candidate.variant.duration_s >= baseline.duration_s


def test_kinematic_dogleg_preserves_clock_and_geometry_before_live_splice() -> None:
    source = _straight_variant()
    stations = source.s_m
    east = stations + 5_000.0
    north = np.full_like(stations, 3_000.0)
    lat, lon = LocalFrame(32.8, -97.1).unproject(east, north)
    speed = 90.0 + 30.0 * stations / float(stations[-1])
    baseline = TrajectoryVariant.from_kinematic_profile(
        template_id=source.template_id,
        cluster_id=source.cluster_id,
        s_m=stations,
        lat_deg=lat,
        lon_deg=lon,
        east_m=east,
        north_m=north,
        altitude_m=source.altitude_m,
        cas_mps=speed,
        tas_mps=speed,
        ground_speed_mps=speed,
        command_cas_mps=speed,
        reference_command_cas_mps=speed,
        lower_cas_mps=np.full_like(speed, 70.0),
        upper_cas_mps=np.full_like(speed, 130.0),
        threshold_resource_id="RWY",
        resource_stations_m=(("MERGE", 50_000.0),),
    )
    anchor_s_m = 90_000.0
    candidate = next(
        item
        for item in PathStretchRealizer(
            config=StretchConfig(max_turn_deg=120.0)
        ).candidates(baseline, anchor_s_m=anchor_s_m)
        if item.feasible
    )
    assert candidate.variant is not None
    mapping = np.asarray(candidate.station_mapping_m, dtype=np.float64)
    child_anchor_s_m = float(
        np.interp(anchor_s_m, mapping[:, 0], mapping[:, 1])
    )
    parent_elapsed_s = MonotoneTrajectory.from_variant(
        baseline
    ).elapsed_at_station(anchor_s_m)
    child_elapsed_s = MonotoneTrajectory.from_variant(
        candidate.variant
    ).elapsed_at_station(child_anchor_s_m)
    upstream_parent_s_m = 95_000.0
    upstream_child_s_m = float(
        np.interp(upstream_parent_s_m, mapping[:, 0], mapping[:, 1])
    )

    assert child_elapsed_s == pytest.approx(parent_elapsed_s, abs=1.0e-9)
    for name in ("lat_deg", "lon_deg", "east_m", "north_m"):
        parent_value = float(
            np.interp(upstream_parent_s_m, baseline.s_m, getattr(baseline, name))
        )
        child_value = float(
            np.interp(
                upstream_child_s_m,
                candidate.variant.s_m,
                getattr(candidate.variant, name),
            )
        )
        assert child_value == pytest.approx(parent_value, abs=1.0e-10)

    # Historical continuity is a curve-in-time requirement, not just an exact
    # match at the action station.  Preserve every original parent knot so a
    # differently sampled dogleg grid cannot change interpolation between the
    # already-flown samples.
    parent_prefix = baseline.s_m >= anchor_s_m
    mapped_prefix_s_m = np.interp(
        baseline.s_m[parent_prefix], mapping[:, 0], mapping[:, 1]
    )
    for name, tolerance in (
        ("lat_deg", 1.0e-12),
        ("lon_deg", 1.0e-12),
        ("east_m", 1.0e-8),
        ("north_m", 1.0e-8),
        ("altitude_m", 1.0e-8),
        ("cas_mps", 1.0e-10),
        ("tas_mps", 1.0e-10),
        ("ground_speed_mps", 1.0e-10),
    ):
        np.testing.assert_allclose(
            np.interp(
                mapped_prefix_s_m,
                candidate.variant.s_m,
                getattr(candidate.variant, name),
            ),
            getattr(baseline, name)[parent_prefix],
            rtol=0.0,
            atol=tolerance,
        )
    np.testing.assert_allclose(
        np.interp(
            mapped_prefix_s_m,
            candidate.variant.s_m,
            candidate.variant.elapsed_time_s,
        ),
        baseline.elapsed_time_s[parent_prefix],
        rtol=0.0,
        atol=1.0e-10,
    )
    details = dict(candidate.variant.diagnostics.details)
    assert details["live_splice_prefix_source"] == "parent_physical_profile"
    assert details["live_splice_parent_prefix_knots"] == int(
        np.count_nonzero(parent_prefix)
    )


def test_near_knot_live_anchor_does_not_create_a_sub_ulp_time_segment() -> None:
    baseline = _straight_variant()
    represented_anchor_s_m = float(baseline.s_m[-6])
    recovered_event_station_s_m = represented_anchor_s_m - 3.7e-7
    candidates = PathStretchRealizer(
        config=StretchConfig(max_turn_deg=120.0)
    ).candidates(baseline, anchor_s_m=recovered_event_station_s_m)

    feasible = [item.variant for item in candidates if item.variant is not None]
    assert feasible
    contemporary_epoch_s = 1_775_035_134.1680675
    for variant in feasible:
        ordered_elapsed_s = variant.elapsed_time_s[::-1]
        absolute_time_s = contemporary_epoch_s + (
            ordered_elapsed_s - float(ordered_elapsed_s[0])
        )
        assert np.all(np.diff(absolute_time_s) > 0.0)


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
        assert (
            candidate.variant.diagnostics.message
            == "public SIMAP coupled replay validation passed"
        )
        assert details["geometry_family"] == "raised_cosine_lateral_lane_change_v1"
        assert details["raw_max_bank_ratio"] <= adapter.maximum_raw_bank_ratio
        assert details["replay_final_threshold_error_m"] <= 0.10 * M_PER_NM

    first_variant = candidates[0].variant
    assert first_variant is not None
    calibrated = dict(first_variant.diagnostics.details)
    assert calibrated["simap_replay_pass_count"] == 2
    assert (
        calibrated["simap_first_pass_threshold_error_m"]
        > calibrated["replay_final_threshold_error_m"]
    )


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
