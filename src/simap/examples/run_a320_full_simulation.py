from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np

from simap import (
    ConstantWeather,
    ConstraintEnvelope,
    EffectivePolarBackend,
    LateralGuidanceConfig,
    LongitudinalPlanRequest,
    OptimizerConfig,
    ReferencePath,
    ScalarProfile,
    SimulationRequest,
    ThresholdBoundary,
    UpstreamBoundary,
    build_default_aircraft_config,
    build_speed_schedule_from_wrap,
    extract_aircraft_data,
    load_openap,
    plan_longitudinal_descent,
    plot_constraint_envelope,
    plot_longitudinal_plan,
    simulate_plan,
    suggest_approach_mass_kg,
)


def build_demo_inputs() -> tuple[LongitudinalPlanRequest, ReferencePath]:
    openap = load_openap("A320")
    aircraft_data = extract_aircraft_data(openap)
    mass_kg = suggest_approach_mass_kg(aircraft_data, payload_kg=12_000.0)
    cfg, openap = build_default_aircraft_config("A320", mass_kg=mass_kg, openap_objects=openap)
    perf = EffectivePolarBackend(cfg=cfg, openap=openap)
    weather = ConstantWeather()

    reference_path = ReferencePath.from_geographic(
        lat_deg=np.asarray([48.7600, 48.5600, 48.3538], dtype=float),
        lon_deg=np.asarray([11.0500, 11.3000, 11.7861], dtype=float),
    )

    threshold = ThresholdBoundary(
        h_m=450.0,
        cas_mps=float(openap.wrap.landing_speed()["default"]),
        gamma_rad=-np.deg2rad(3.0),
    )
    upstream = UpstreamBoundary(
        h_m=3_000.0,
        cas_window_mps=(
            float(openap.wrap.finalapp_vcas()["default"]),
            float(openap.wrap.descent_const_vcas()["default"]),
        ),
    )

    speed_schedule = build_speed_schedule_from_wrap(openap.wrap)
    max_s_m = max(reference_path.total_length_m, 60_000.0)
    envelope = ConstraintEnvelope.from_profiles(
        altitude_lower=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([threshold.h_m, upstream.h_m - 75.0], dtype=float),
        ),
        altitude_upper=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([threshold.h_m + 25.0, upstream.h_m + 150.0], dtype=float),
        ),
        cas_lower=ScalarProfile(
            s_m=speed_schedule.s_m,
            y=np.maximum(speed_schedule.y - 8.0 * 0.514444, threshold.cas_mps),
        ),
        cas_upper=ScalarProfile(
            s_m=speed_schedule.s_m,
            y=speed_schedule.y,
        ),
        gamma_lower=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([-np.deg2rad(4.5), -np.deg2rad(0.5)], dtype=float),
        ),
        gamma_upper=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([-np.deg2rad(2.0), np.deg2rad(0.5)], dtype=float),
        ),
    )
    request = LongitudinalPlanRequest(
        cfg=cfg,
        perf=perf,
        threshold=threshold,
        upstream=upstream,
        constraints=envelope,
        weather=weather,
        optimizer=OptimizerConfig(num_nodes=31, maxiter=300),
        reference_track_rad=reference_path.track_angle_rad(reference_path.total_length_m),
    )
    return request, reference_path


def plot_simulation_overview(reference_path: ReferencePath, simulation) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))

    axes[0, 0].plot(reference_path.east_m, reference_path.north_m, label="Reference path", linewidth=2.0)
    axes[0, 0].plot(simulation.east_m, simulation.north_m, label="Simulated track", linewidth=1.8)
    axes[0, 0].set_title("Ground Track")
    axes[0, 0].set_xlabel("East [m]")
    axes[0, 0].set_ylabel("North [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(simulation.t_s, simulation.cross_track_m, linewidth=1.8)
    axes[0, 1].set_title("Cross-Track Error")
    axes[0, 1].set_xlabel("Elapsed Time [s]")
    axes[0, 1].set_ylabel("Cross-track [m]")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(simulation.t_s, np.rad2deg(simulation.phi_rad), label="Bank", linewidth=1.8)
    axes[1, 0].plot(simulation.t_s, np.rad2deg(simulation.phi_req_rad), label="Bank command", linewidth=1.4)
    axes[1, 0].set_title("Bank Response")
    axes[1, 0].set_xlabel("Elapsed Time [s]")
    axes[1, 0].set_ylabel("Bank [deg]")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(simulation.t_s, np.rad2deg(simulation.track_error_rad), linewidth=1.8)
    axes[1, 1].set_title("Track Error")
    axes[1, 1].set_xlabel("Elapsed Time [s]")
    axes[1, 1].set_ylabel("Track error [deg]")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("A320 Coupled Descent Simulation")
    fig.tight_layout()


def main() -> None:
    request, reference_path = build_demo_inputs()
    plan = plan_longitudinal_descent(request)
    simulation = simulate_plan(
        SimulationRequest(
            cfg=request.cfg,
            plan=plan,
            reference_path=reference_path,
            weather=request.weather,
            guidance=LateralGuidanceConfig(
                lookahead_m=700.0,
                cross_track_gain=3.0,
                track_error_gain=4.0,
            ),
            dt_s=0.5,
            threshold_tolerance_m=0.0,
        )
    )

    plot_constraint_envelope(request.constraints, show=False)
    plot_longitudinal_plan(plan, envelope=request.constraints, show=False)
    plot_simulation_overview(reference_path, simulation)

    print(plan.to_pandas().head())
    print(plan.to_pandas().tail())
    print(simulation.to_pandas().head())
    print(simulation.to_pandas().tail())
    print(
        {
            "plan_success": plan.solver_success,
            "simulation_success": simulation.success,
            "tod_m": plan.tod_m,
            "max_cross_track_m": simulation.max_abs_cross_track_m,
            "max_track_error_deg": float(np.rad2deg(simulation.max_abs_track_error_rad)),
            "final_threshold_error_m": simulation.final_threshold_error_m,
        }
    )
    plt.show()


if __name__ == "__main__":
    main()
