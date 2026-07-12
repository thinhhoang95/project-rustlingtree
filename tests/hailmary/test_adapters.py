from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from hailmary.adapters import (
    HailmaryScheduleView,
    SIMAPAdapter,
    get_cached_a320_context,
    get_cached_a320_envelope,
    planned_a320_cas_envelope,
    simplify_reference_path,
)
from hailmary.geometry import LocalFrame
from hailmary.config import M_PER_NM
from hailmary.scenario import FlightDefinition, ResourceDefinition, ScenarioDefinition
from hailmary.simulator import Simulator
from hailmary.templates import MedoidTrack, TemplateCompiler
from hailmary.templates.models import TrajectoryVariant
from simap.config import AircraftConfig, ModeConfig

from .test_templates import _straight_variant


def _mode(name: str) -> ModeConfig:
    return ModeConfig(
        name=name,  # type: ignore[arg-type]
        tau_v_s=10.0,
        vs_min_mps=-10.0,
        vs_max_mps=2.0,
        cd0=0.02,
        k=0.05,
        phi_comfort_max_rad=0.4,
        phi_procedure_max_rad=0.4,
        tau_phi_s=2.0,
        p_max_rps=0.1,
        vs_1g_ref_cas_mps=60.0,
        cas_min_mps=60.0,
        cas_max_mps=200.0,
    )


def _aircraft_config() -> AircraftConfig:
    return AircraftConfig(
        typecode="TEST",
        engine_name="TEST-ENGINE",
        mass_kg=60_000.0,
        reference_mass_kg=60_000.0,
        wing_area_m2=120.0,
        vmo_kts=300.0,
        mmo=0.78,
        clean=_mode("clean"),
        approach=_mode("approach"),
        final=_mode("final"),
    )


class _ConstantPerformanceBackend:
    def drag_newtons(self, **_kwargs) -> float:
        return 10_000.0

    def idle_thrust_newtons(self, **_kwargs) -> float:
        return 0.0

    def thrust_bounds_newtons(self, **_kwargs) -> tuple[float, float]:
        return 0.0, 1_000_000.0


def _variant(*, command_mps: float = 90.0) -> TrajectoryVariant:
    total_m = np.deg2rad(0.2) * 6_371_000.0
    s_m = np.asarray([0.0, total_m / 2.0, total_m])
    return TrajectoryVariant.from_kinematic_profile(
        template_id="TEMPLATE",
        cluster_id="0",
        s_m=s_m,
        lat_deg=np.asarray([0.0, 0.0, 0.0]),
        lon_deg=np.asarray([0.0, 0.1, 0.2]),
        east_m=s_m,
        north_m=np.zeros(3),
        altitude_m=np.asarray([100.0, 1_000.0, 2_000.0]),
        cas_mps=np.full(3, command_mps),
        tas_mps=np.full(3, command_mps),
        ground_speed_mps=np.full(3, 100.0),
        command_cas_mps=np.full(3, command_mps),
        reference_command_cas_mps=np.full(3, command_mps),
        lower_cas_mps=np.full(3, 50.0),
        upper_cas_mps=np.full(3, 220.0),
        threshold_resource_id="RWY",
    )


def test_cached_a320_context_reuses_resolved_openap_objects() -> None:
    first = get_cached_a320_context()
    second = get_cached_a320_context()

    assert first is second
    assert first.aircraft_config.typecode == "A320"
    assert first.payload_kg == 12_000.0

    first_envelope = get_cached_a320_envelope([0.0, 10_000.0], altitude_m=[100.0, 2_000.0])
    second_envelope = get_cached_a320_envelope([0.0, 10_000.0], altitude_m=[100.0, 2_000.0])
    assert first_envelope is second_envelope
    assert SIMAPAdapter().version1_default_aircraft_assumption
    assert not SIMAPAdapter(
        aircraft_config=_aircraft_config(),
        performance_backend=_ConstantPerformanceBackend(),
    ).version1_default_aircraft_assumption


def test_planned_envelope_applies_10k_cap_and_freezes_arrays() -> None:
    envelope = planned_a320_cas_envelope(
        [0.0, 40_000.0],
        altitude_m=[1_000.0, 4_000.0],
        aircraft_config=_aircraft_config(),
    )

    assert envelope.upper_cas_mps[0] == pytest.approx(250.0 * 0.514444)
    assert envelope.upper_cas_mps[1] == pytest.approx(300.0 * 0.514444)
    assert not envelope.s_m.flags.writeable
    assert not envelope.lower_cas_mps.flags.writeable
    assert not envelope.upper_cas_mps.flags.writeable


def test_reference_path_simplification_preserves_endpoints_and_freezes_simap_arrays() -> None:
    lat = np.zeros(5)
    lon = np.asarray([0.2, 0.15, 0.1, 0.05, 0.0])

    geometry = simplify_reference_path(
        lat,
        lon,
        threshold_lat_deg=0.0,
        threshold_lon_deg=0.0,
        lateral_tolerance_m=10.0,
    )

    np.testing.assert_allclose(geometry.control_lon_deg[[0, -1]], [0.2, 0.0])
    assert len(geometry.control_lon_deg) == 2
    assert geometry.reference_path.s_m[0] == pytest.approx(geometry.total_length_m)
    assert geometry.reference_path.s_m[-1] == pytest.approx(0.0)
    assert not geometry.control_lat_deg.flags.writeable
    assert not geometry.reference_path.s_m.flags.writeable
    assert not geometry.reference_path.curvature_inv_m.flags.writeable


def test_reference_path_curvature_is_wrap_safe_on_westbound_dogleg() -> None:
    points = np.asarray(
        [
            [100_000.0, 0.0],
            [90_000.0, -10_000.0],
            [80_000.0, 0.0],
            [0.0, 0.0],
        ]
    )
    lat, lon = LocalFrame(0.0, 0.0).unproject(points[:, 0], points[:, 1])

    geometry = simplify_reference_path(
        lat,
        lon,
        threshold_lat_deg=float(lat[-1]),
        threshold_lon_deg=float(lon[-1]),
        lateral_tolerance_m=0.0,
    )
    path = geometry.reference_path
    wrap_safe_curvature = np.gradient(
        np.unwrap(path.track_rad),
        path.s_from_start_m,
        edge_order=1,
    )

    np.testing.assert_allclose(path.curvature_inv_m, wrap_safe_curvature)
    assert abs(float(path.curvature_inv_m[-1])) < 1.0e-10


def test_simap_adapter_returns_diagnostics_without_mutating_variant() -> None:
    variant = _variant(command_mps=90.0)
    original_id = variant.variant_id
    adapter = SIMAPAdapter(aircraft_config=_aircraft_config())

    accepted = adapter.validate(variant)
    rejected = adapter.validate(_variant(command_mps=180.0))

    assert accepted.feasible
    assert "validation passed" in accepted.message
    assert not rejected.feasible
    assert "upper envelope" in rejected.message
    assert variant.variant_id == original_id


def test_simap_public_replay_compiles_physical_profiles_and_timing() -> None:
    variant = _straight_variant(
        length_m=50_000.0,
        speed_mps=100.0,
        station_count=101,
        lower_cas_mps=70.0,
    )
    adapter = SIMAPAdapter(
        aircraft_config=_aircraft_config(),
        performance_backend=_ConstantPerformanceBackend(),
        path_simplification_tolerance_m=10.0,
    )

    compiled = adapter.compile(variant)
    details = dict(compiled.diagnostics.details)

    assert compiled.diagnostics.feasible
    assert compiled.diagnostics.message == "public SIMAP coupled replay validation passed"
    assert details["simap_replay_supported"] is True
    assert details["simap_optimizer_used"] is False
    assert details["version1_default_aircraft_assumption"] is False
    assert details["performance_backend_source"] == "explicit_performance_backend"
    assert str(details["performance_backend_fingerprint"]).startswith(
        "simap-performance-backend_"
    )
    assert details["simap_replay_contract"] == (
        "bounded_inverse_commands_then_public_time_domain_replay"
    )
    assert compiled.duration_s == pytest.approx(compiled.diagnostics.compiled_duration_s)
    assert compiled.duration_s == pytest.approx(details["simap_replay_duration_s"])
    assert compiled.variant_id != variant.variant_id
    assert compiled.resource("RWY").elapsed_time_s == pytest.approx(compiled.duration_s)
    assert not compiled.elapsed_time_s.flags.writeable
    compiled_envelope = adapter.envelope(compiled.s_m, compiled.altitude_m)
    np.testing.assert_allclose(
        compiled.lower_cas_mps,
        compiled_envelope.lower_cas_mps,
    )
    np.testing.assert_allclose(
        compiled.upper_cas_mps,
        compiled_envelope.upper_cas_mps,
    )


def test_template_compiler_uses_simap_replay_as_executable_timing_source() -> None:
    length_m = 50.0 * M_PER_NM
    speed_mps = 100.0
    sample_count = 41
    track = MedoidTrack(
        flight_id="SIMAP_MEDOID",
        time_s=np.linspace(0.0, length_m / speed_mps, sample_count),
        lat_deg=np.linspace(length_m / 111_319.49079327357, 0.0, sample_count),
        lon_deg=np.zeros(sample_count),
        altitude_m=np.zeros(sample_count),
    )
    config = _aircraft_config()
    adapter = SIMAPAdapter(
        aircraft_config=config,
        performance_backend=_ConstantPerformanceBackend(),
        path_simplification_tolerance_m=10.0,
    )

    template = TemplateCompiler(
        station_count=256,
        aircraft_config=config,
        validator=adapter,
    ).compile(
        track,
        cluster_id="SIMAP_CLUSTER",
        member_count=4,
        dataset_id="SIMAP_TEST",
        airport_id="TEST",
        runway_id="RWY",
        threshold_resource_id="RWY",
    )

    variant = template.baseline_variant
    assert variant.diagnostics.feasible
    assert variant.diagnostics.compiled_duration_s == pytest.approx(variant.duration_s)
    assert dict(variant.diagnostics.details)["simap_replay_supported"] is True
    assert variant.duration_s != pytest.approx(length_m / speed_mps, abs=1.0e-3)


def test_simap_replay_rejects_sustained_bank_beyond_aircraft_limit() -> None:
    first_leg = np.column_stack((np.linspace(0.0, 10_000.0, 51), np.zeros(51)))
    second_leg = np.column_stack(
        (np.full(50, 10_000.0), np.linspace(200.0, 10_000.0, 50))
    )
    points = np.vstack((first_leg, second_leg))
    stations = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    frame = LocalFrame(0.0, 0.0)
    lat, lon = frame.unproject(points[:, 0], points[:, 1])
    speed = np.full(len(stations), 100.0)
    variant = TrajectoryVariant.from_kinematic_profile(
        template_id="TURN_TEMPLATE",
        cluster_id="TURN_CLUSTER",
        s_m=stations,
        lat_deg=lat,
        lon_deg=lon,
        east_m=points[:, 0],
        north_m=points[:, 1],
        altitude_m=np.zeros(len(stations)),
        cas_mps=speed,
        tas_mps=speed,
        ground_speed_mps=speed,
        command_cas_mps=speed,
        reference_command_cas_mps=speed,
        lower_cas_mps=np.full(len(stations), 70.0),
        upper_cas_mps=np.full(len(stations), 120.0),
        threshold_resource_id="RWY",
    )
    base = _aircraft_config()

    def low_bank(mode: ModeConfig) -> ModeConfig:
        return replace(
            mode,
            phi_comfort_max_rad=0.05,
            phi_procedure_max_rad=0.05,
        )

    constrained = replace(
        base,
        clean=low_bank(base.clean),
        approach=low_bank(base.approach),
        final=low_bank(base.final),
    )
    diagnostics = SIMAPAdapter(
        aircraft_config=constrained,
        performance_backend=_ConstantPerformanceBackend(),
        path_simplification_tolerance_m=10.0,
    ).validate(variant)

    assert not diagnostics.feasible
    assert "sustained bank" in diagnostics.message


def test_sustained_bank_measure_is_invariant_to_variant_station_density() -> None:
    def right_angle_variant(samples_per_leg: int) -> TrajectoryVariant:
        first_leg = np.column_stack(
            (
                np.linspace(0.0, 10_000.0, samples_per_leg),
                np.zeros(samples_per_leg),
            )
        )
        second_leg = np.column_stack(
            (
                np.full(samples_per_leg - 1, 10_000.0),
                np.linspace(
                    10_000.0 / (samples_per_leg - 1),
                    10_000.0,
                    samples_per_leg - 1,
                ),
            )
        )
        points = np.vstack((first_leg, second_leg))
        stations = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        )
        lat, lon = LocalFrame(0.0, 0.0).unproject(points[:, 0], points[:, 1])
        speed = np.full(len(stations), 100.0)
        return TrajectoryVariant.from_kinematic_profile(
            template_id="TURN_DENSITY_TEMPLATE",
            cluster_id="TURN_DENSITY_CLUSTER",
            s_m=stations,
            lat_deg=lat,
            lon_deg=lon,
            east_m=points[:, 0],
            north_m=points[:, 1],
            altitude_m=np.zeros(len(stations)),
            cas_mps=speed,
            tas_mps=speed,
            ground_speed_mps=speed,
            command_cas_mps=speed,
            reference_command_cas_mps=speed,
            lower_cas_mps=np.full(len(stations), 70.0),
            upper_cas_mps=np.full(len(stations), 120.0),
            threshold_resource_id="RWY",
        )

    base = _aircraft_config()

    def low_bank(mode: ModeConfig) -> ModeConfig:
        return replace(
            mode,
            phi_comfort_max_rad=0.05,
            phi_procedure_max_rad=0.05,
        )

    adapter = SIMAPAdapter(
        aircraft_config=replace(
            base,
            clean=low_bank(base.clean),
            approach=low_bank(base.approach),
            final=low_bank(base.final),
        ),
        performance_backend=_ConstantPerformanceBackend(),
        path_simplification_tolerance_m=10.0,
    )

    sparse = adapter.validate(right_angle_variant(21))
    dense = adapter.validate(right_angle_variant(101))
    sparse_details = dict(sparse.details)
    dense_details = dict(dense.details)

    assert not sparse.feasible
    assert not dense.feasible
    assert sparse_details["bank_demand_sample_count"] != dense_details[
        "bank_demand_sample_count"
    ]
    assert sparse_details["maximum_contiguous_overbank_distance_m"] == pytest.approx(
        dense_details["maximum_contiguous_overbank_distance_m"],
        abs=1.0,
    )
    assert sparse_details["maximum_contiguous_overbank_duration_s"] == pytest.approx(
        dense_details["maximum_contiguous_overbank_duration_s"],
        abs=0.02,
    )


def test_schedule_view_uses_canonical_time_order_and_returns_fresh_payloads() -> None:
    variant = _variant()
    definition = ScenarioDefinition(
        scenario_id="ADAPTER",
        seed=3,
        flights=(
            FlightDefinition(
                flight_id="F1",
                release_time_s=100.0,
                baseline_variant_id=variant.variant_id,
                cluster_id="0",
                callsign="CALL1",
                icao24="abc",
                runway="RW35C",
            ),
        ),
        resources=(ResourceDefinition("RWY"),),
        variants=(variant,),
    )
    view = HailmaryScheduleView(definition)

    first = view.arrival_schedule()
    first[0]["points"][0][1] = 999.0
    second = view.arrival_schedule()

    assert second[0]["points"][0][1] == pytest.approx(0.0)
    assert [point[0] for point in second[0]["points"]] == sorted(
        point[0] for point in second[0]["points"]
    )
    assert second[0]["time_at_first_fix"] == pytest.approx(100.0)
    assert second[0]["time_at_last_event"] == pytest.approx(100.0 + variant.duration_s)
    assert second[0]["cas_profile"]["source"] == "hailmary_canonical_variant"

    simulator = Simulator(definition)
    simulator.run()
    assert HailmaryScheduleView(simulator, include_completed=False).arrival_schedule() == []


def test_schedule_adapter_has_no_mutable_scenario_manager_dependency() -> None:
    import hailmary.adapters.scenario_manager as module

    source = inspect.getsource(module)
    assert "from mcp_tools" not in source
    assert "import mcp_tools" not in source
