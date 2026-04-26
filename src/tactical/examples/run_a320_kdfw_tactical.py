from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np

from simap import LateralGuidanceConfig, OptimizerConfig, plot_constraint_envelope, plot_longitudinal_plan
from simap.units import m_to_ft, mps_to_kts
from tactical import TacticalCommand, TacticalCondition, solve_tactical_command


LATERAL_PATH = "JUSST SWTCH THEMM TUSLE SEEVR BRDJE NUSSS YAHBT ZINGG RW17C"


def plot_tactical_overview(bundle) -> None:
    request = bundle.request
    plan = bundle.plan
    simulation = bundle.simulation
    if plan is None or simulation is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    axes[0, 0].plot(request.reference_path.lon_deg, request.reference_path.lat_deg, label="Reference path", linewidth=2.0)
    axes[0, 0].plot(simulation.lon_deg, simulation.lat_deg, label="Simulated track", linewidth=1.5)
    waypoint_lats = [waypoint.lat_deg for waypoint in bundle.path.waypoints]
    waypoint_lons = [waypoint.lon_deg for waypoint in bundle.path.waypoints]
    axes[0, 0].scatter(waypoint_lons, waypoint_lats, s=16.0, color="black", zorder=3)
    axes[0, 0].set_title("KDFW Tactical Route")
    axes[0, 0].set_xlabel("Longitude [deg]")
    axes[0, 0].set_ylabel("Latitude [deg]")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(plan.s_m / 1_000.0, m_to_ft(plan.h_m), label="Plan", linewidth=1.8)
    axes[0, 1].plot(simulation.s_m / 1_000.0, m_to_ft(simulation.h_m), label="Simulation", linewidth=1.3)
    axes[0, 1].invert_xaxis()
    axes[0, 1].set_title("Altitude")
    axes[0, 1].set_xlabel("Distance to RW17C [km]")
    axes[0, 1].set_ylabel("Altitude [ft]")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(plan.s_m / 1_000.0, mps_to_kts(plan.v_cas_mps), label="Plan", linewidth=1.8)
    axes[1, 0].plot(simulation.s_m / 1_000.0, mps_to_kts(simulation.v_cas_mps), label="Simulation", linewidth=1.3)
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_title("Calibrated Airspeed")
    axes[1, 0].set_xlabel("Distance to RW17C [km]")
    axes[1, 0].set_ylabel("CAS [kt]")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(simulation.t_s, simulation.cross_track_m, linewidth=1.5)
    axes[1, 1].set_title("Cross-Track Error")
    axes[1, 1].set_xlabel("Elapsed time [s]")
    axes[1, 1].set_ylabel("Cross-track [m]")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("A320 Tactical Command: JUSST to KDFW RW17C")
    fig.tight_layout()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    command = TacticalCommand(
        lateral_path=LATERAL_PATH,
        upstream=TacticalCondition(
            fix_identifier="JUSST",
            cas_kts=290.0,
            altitude_ft=26_000.0,
        ),
        altitude_constraints=(),
        runway_altitude_ft=620.0,
    )

    bundle = solve_tactical_command(
        command,
        fixes_csv=repo_root / "data/kdfw_procs/airport_related_fixes.csv",
        optimizer=OptimizerConfig(num_nodes=31, maxiter=350),
        guidance=LateralGuidanceConfig(
            lookahead_m=2_500.0,
            cross_track_gain=1.0,
            track_error_gain=2.0,
        ),
        dt_s=0.5,
    )
    plan = bundle.plan
    simulation = bundle.simulation
    assert plan is not None
    assert simulation is not None

    plot_constraint_envelope(bundle.request.constraints, show=False)
    plot_longitudinal_plan(plan, envelope=bundle.request.constraints, show=False)
    plot_tactical_overview(bundle)

    print(plan.to_pandas().head())
    print(plan.to_pandas().tail())
    print(simulation.to_pandas().head())
    print(simulation.to_pandas().tail())
    print(
        {
            "route": bundle.path.identifiers,
            "plan_success": plan.solver_success,
            "plan_message": plan.solver_message,
            "simulation_success": simulation.success,
            "simulation_message": simulation.message,
            "tod_m": plan.tod_m,
            "path_length_m": bundle.request.reference_path.total_length_m,
            "start_cas_kts": float(mps_to_kts(plan.v_cas_mps[-1])),
            "start_altitude_ft": float(m_to_ft(plan.h_m[-1])),
            "max_cross_track_m": simulation.max_abs_cross_track_m,
            "max_track_error_deg": float(np.rad2deg(simulation.max_abs_track_error_rad)),
            "final_threshold_error_m": simulation.final_threshold_error_m,
        }
    )
    plt.show()


if __name__ == "__main__":
    main()
