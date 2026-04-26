from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np

from simap import LateralGuidanceConfig, OptimizerConfig
from simap.units import m_to_ft, mps_to_kts
from tactical import TacticalCommand, TacticalCondition, solve_tactical_command


LATERAL_PATH = "JUSST SWTCH THEMM TUSLE SEEVR BRDJE NUSSS YAHBT ZINGG RW17C"


def _shade_pre_tod_extension(ax, *, raw_tod_m: float, tactical_start_m: float) -> None:
    if tactical_start_m <= raw_tod_m:
        return
    ax.axvspan(raw_tod_m, tactical_start_m, color="0.5", alpha=0.08, label="Pre-TOD profile extension")
    ax.axvline(raw_tod_m, color="0.25", linestyle="--", linewidth=1.0, label="Raw TOD")


def _plot_envelope_band(ax, s_m, lower, upper, *, color: str, scale: float = 1.0) -> None:
    ax.fill_between(s_m, lower * scale, upper * scale, color=color, alpha=0.14, label="Enforced envelope")


def _closest_simulation_idx_to_raw_tod(simulation, raw_tod_m: float) -> int:
    return int(np.argmin(np.abs(simulation.s_m - raw_tod_m)))


def _raw_tod_replay_window(simulation, raw_tod_m: float, *, window_s: float = 20.0) -> tuple[np.ndarray, float]:
    tod_idx = _closest_simulation_idx_to_raw_tod(simulation, raw_tod_m)
    tod_t_s = float(simulation.t_s[tod_idx])
    mask = np.abs(simulation.t_s - tod_t_s) <= window_s
    return mask, tod_t_s


def _plot_longitudinal_response(
    ax,
    *,
    raw_s_m,
    raw_y,
    simulation_s_m,
    replay_y,
    raw_label: str,
    raw_color: str,
    replay_color: str,
    x_scale: float = 1.0,
    scale: float = 1.0,
    raw_linewidth: float = 1.1,
    replay_linewidth: float = 2.0,
) -> None:
    ax.plot(raw_s_m * x_scale, raw_y * scale, color=raw_color, linewidth=raw_linewidth, label=raw_label)
    ax.plot(
        simulation_s_m * x_scale,
        replay_y * scale,
        color=replay_color,
        linewidth=replay_linewidth,
        label="Integrated response",
    )


def plot_tactical_longitudinal(bundle) -> None:
    request = bundle.request
    raw_plan = bundle.raw_plan
    plan = bundle.plan
    simulation = bundle.simulation
    if raw_plan is None or plan is None or simulation is None:
        return

    raw_s_m = raw_plan.s_m
    raw_tod_m = float(raw_plan.s_m[-1])
    tactical_start_m = float(plan.s_m[-1])
    h_lower_m, h_upper_m = request.constraints.h_bounds_many(raw_s_m)
    cas_lower_mps, cas_upper_mps = request.constraints.cas_bounds_many(raw_s_m)
    gamma_lower_rad, gamma_upper_rad = request.constraints.gamma_bounds_many(raw_s_m)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0), sharex=True)

    _plot_longitudinal_response(
        axes[0, 0],
        raw_s_m=raw_plan.s_m,
        raw_y=raw_plan.h_m,
        simulation_s_m=simulation.s_m,
        replay_y=simulation.h_m,
        raw_label="Raw descent reference",
        raw_color="#1b7f1b",
        replay_color="#2ca02c",
    )
    if bundle.command.altitude_constraints:
        _plot_envelope_band(axes[0, 0], raw_s_m, h_lower_m, h_upper_m, color="#2ca02c")
    axes[0, 0].scatter([raw_tod_m], [raw_plan.h_m[-1]], s=24.0, color="#1b7f1b", zorder=3, label="TOD boundary")
    axes[0, 0].set_title("Altitude")
    axes[0, 0].set_ylabel("h [m]")

    _plot_longitudinal_response(
        axes[0, 1],
        raw_s_m=raw_plan.s_m,
        raw_y=raw_plan.v_cas_mps,
        simulation_s_m=simulation.s_m,
        replay_y=simulation.v_cas_mps,
        raw_label="Raw descent reference",
        raw_color="#c95f00",
        replay_color="#ff7f0e",
    )
    _plot_envelope_band(axes[0, 1], raw_s_m, cas_lower_mps, cas_upper_mps, color="#ff7f0e")
    axes[0, 1].set_title("CAS")
    axes[0, 1].set_ylabel("v_cas [m/s]")

    _plot_longitudinal_response(
        axes[1, 0],
        raw_s_m=raw_plan.s_m,
        raw_y=np.rad2deg(raw_plan.gamma_rad),
        simulation_s_m=simulation.s_m,
        replay_y=np.rad2deg(simulation.gamma_rad),
        raw_label="Raw descent reference",
        raw_color="#6f49a8",
        replay_color="#9467bd",
    )
    if gamma_lower_rad is not None and gamma_upper_rad is not None:
        _plot_envelope_band(
            axes[1, 0],
            raw_s_m,
            np.rad2deg(gamma_lower_rad),
            np.rad2deg(gamma_upper_rad),
            color="#9467bd",
        )
    axes[1, 0].set_title("Flight-Path Angle")
    axes[1, 0].set_xlabel("Distance From Threshold [m]")
    axes[1, 0].set_ylabel("gamma [deg]")

    _plot_longitudinal_response(
        axes[1, 1],
        raw_s_m=raw_plan.s_m,
        raw_y=raw_plan.thrust_n,
        simulation_s_m=simulation.s_m,
        replay_y=simulation.thrust_n,
        raw_label="Raw descent reference",
        raw_color="#6f403b",
        replay_color="#8c564b",
    )
    axes[1, 1].set_title("Thrust")
    axes[1, 1].set_xlabel("Distance From Threshold [m]")
    axes[1, 1].set_ylabel("T [N]")

    for ax in axes.flat:
        _shade_pre_tod_extension(ax, raw_tod_m=raw_tod_m, tactical_start_m=tactical_start_m)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("A320 Tactical Longitudinal Response")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))


def _plot_replay_series(ax, *, t_rel_s, y, color: str, title: str, ylabel: str) -> None:
    ax.plot(t_rel_s, y, color=color, linewidth=2.0)
    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0, label="Raw TOD")
    ax.set_title(title)
    ax.set_xlabel("Time Relative to Raw TOD [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_tod_replay_window(bundle, *, window_s: float = 20.0) -> None:
    raw_plan = bundle.raw_plan
    simulation = bundle.simulation
    if raw_plan is None or simulation is None:
        return

    raw_tod_m = float(raw_plan.s_m[-1])
    mask, tod_t_s = _raw_tod_replay_window(simulation, raw_tod_m, window_s=window_s)
    t_rel_s = simulation.t_s[mask] - tod_t_s

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    _plot_replay_series(
        axes[0, 0],
        t_rel_s=t_rel_s,
        y=m_to_ft(simulation.h_m[mask]),
        color="#1f77b4",
        title="Altitude Response Near Raw TOD",
        ylabel="Altitude [ft]",
    )
    _plot_replay_series(
        axes[0, 1],
        t_rel_s=t_rel_s,
        y=mps_to_kts(simulation.v_cas_mps[mask]),
        color="#ff7f0e",
        title="CAS Response Near Raw TOD",
        ylabel="CAS [kt]",
    )

    _plot_replay_series(
        axes[1, 0],
        t_rel_s=t_rel_s,
        y=np.rad2deg(simulation.gamma_rad[mask]),
        color="#9467bd",
        title="Flight-Path Angle Command Near Raw TOD",
        ylabel="gamma [deg]",
    )
    _plot_replay_series(
        axes[1, 1],
        t_rel_s=t_rel_s,
        y=simulation.thrust_n[mask],
        color="#8c564b",
        title="Thrust Command Near Raw TOD",
        ylabel="T [N]",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("A320 Tactical Integrated Response Around Raw TOD")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))


def print_tod_diagnostics(bundle, *, plan_window: int = 4, simulation_window: int = 24) -> None:
    raw_plan = bundle.raw_plan
    plan = bundle.plan
    simulation = bundle.simulation
    if raw_plan is None or plan is None:
        return

    raw_tod_m = float(raw_plan.s_m[-1])
    extension_start_idx = len(raw_plan.s_m)
    print("\nPlan nodes around raw TOD")
    print("idx      s_m       t_s       h_m   cas_mps   cas_kt  gamma_deg   thrust_n")
    for idx in range(max(0, extension_start_idx - plan_window), min(len(plan.s_m), extension_start_idx + plan_window + 1)):
        marker = "  <- last raw" if idx == extension_start_idx - 1 else "  <- first extension" if idx == extension_start_idx else ""
        print(
            f"{idx:3d} "
            f"{plan.s_m[idx]:10.1f} "
            f"{plan.t_s[idx]:8.2f} "
            f"{plan.h_m[idx]:8.1f} "
            f"{plan.v_cas_mps[idx]:8.3f} "
            f"{mps_to_kts(plan.v_cas_mps[idx]):8.2f} "
            f"{np.rad2deg(plan.gamma_rad[idx]):10.3f} "
            f"{plan.thrust_n[idx]:10.1f}"
            f"{marker}"
        )

    if simulation is None:
        return

    closest_idx = _closest_simulation_idx_to_raw_tod(simulation, raw_tod_m)
    print("\nSimulation time steps around raw TOD")
    print("idx       t_s        s_m       h_m   cas_mps   cas_kt  gamma_deg   thrust_n  ds_to_tod")
    for idx in range(max(0, closest_idx - simulation_window), min(len(simulation.s_m), closest_idx + simulation_window + 1)):
        print(
            f"{idx:5d} "
            f"{simulation.t_s[idx]:8.2f} "
            f"{simulation.s_m[idx]:10.1f} "
            f"{simulation.h_m[idx]:8.1f} "
            f"{simulation.v_cas_mps[idx]:8.3f} "
            f"{mps_to_kts(simulation.v_cas_mps[idx]):8.2f} "
            f"{np.rad2deg(simulation.gamma_rad[idx]):10.3f} "
            f"{simulation.thrust_n[idx]:10.1f} "
            f"{simulation.s_m[idx] - raw_tod_m:10.1f}"
        )


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
    raw_plan = bundle.raw_plan
    simulation = bundle.simulation
    assert plan is not None
    assert raw_plan is not None
    assert simulation is not None
    extension_start_idx = len(raw_plan.s_m)
    first_extension_cas_mps = float(plan.v_cas_mps[extension_start_idx]) if extension_start_idx < len(plan.s_m) else float("nan")

    plot_tactical_longitudinal(bundle)
    plot_tod_replay_window(bundle)

    print(plan.to_pandas().head())
    print(plan.to_pandas().tail())
    print(simulation.to_pandas().head())
    print(simulation.to_pandas().tail())
    print(
        {
            "route": bundle.path.identifiers,
            "plan_success": plan.solver_success,
            "plan_message": plan.solver_message,
            "raw_tod_m": plan.tod_m,
            "tactical_start_s_m": float(plan.s_m[-1]),
            "raw_tod_cas_mps": float(raw_plan.v_cas_mps[-1]),
            "raw_tod_cas_kts": float(mps_to_kts(raw_plan.v_cas_mps[-1])),
            "first_extension_cas_mps": first_extension_cas_mps,
            "first_extension_cas_kts": float(mps_to_kts(first_extension_cas_mps)),
            "splice_cas_jump_mps": first_extension_cas_mps - float(raw_plan.v_cas_mps[-1]),
            "splice_cas_jump_kts": float(mps_to_kts(first_extension_cas_mps - float(raw_plan.v_cas_mps[-1]))),
            "simulation_success": simulation.success,
            "simulation_message": simulation.message,
            "path_length_m": bundle.request.reference_path.total_length_m,
            "start_cas_kts": float(mps_to_kts(plan.v_cas_mps[-1])),
            "start_altitude_ft": float(m_to_ft(plan.h_m[-1])),
            "max_cross_track_m": simulation.max_abs_cross_track_m,
            "max_track_error_deg": float(np.rad2deg(simulation.max_abs_track_error_rad)),
            "final_threshold_error_m": simulation.final_threshold_error_m,
        }
    )
    print_tod_diagnostics(bundle)
    plt.show()


if __name__ == "__main__":
    main()
