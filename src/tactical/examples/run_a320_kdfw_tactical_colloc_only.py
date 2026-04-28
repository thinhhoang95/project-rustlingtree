from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from simap import LateralGuidanceConfig, OptimizerConfig
from simap.units import m_to_ft, mps_to_kts
from tactical.diagnostics import _evaluation_bundle, render_tod_neighborhood
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


def _constraint_blocks(request, plan, inequality_residuals: np.ndarray) -> list[tuple[str, np.ndarray]]:
    blocks: list[tuple[str, np.ndarray]] = []
    num_nodes = len(plan.s_m)
    offset = 0

    def take(label: str, width: int) -> None:
        nonlocal offset
        blocks.append((label, np.asarray(inequality_residuals[offset : offset + width], dtype=float)))
        offset += width

    take("altitude lower", num_nodes)
    take("altitude upper", num_nodes)
    take("CAS lower", num_nodes)
    take("CAS upper", num_nodes)
    take("thrust lower", num_nodes)
    take("thrust upper", num_nodes)
    take("bank lower", num_nodes)
    take("bank upper", num_nodes)
    take("roll lower", num_nodes)
    take("roll upper", num_nodes)
    take("along-track speed", num_nodes)
    if request.optimizer.idle_thrust_margin_fraction is not None:
        take("thrust to idle", num_nodes)
        take("idle band upper", num_nodes)

    gamma_lower, gamma_upper = request.constraints.gamma_bounds_many(plan.s_m)
    if gamma_lower is not None:
        take("gamma lower", num_nodes)
    if gamma_upper is not None:
        take("gamma upper", num_nodes)
    if request.constraints.cl_max is not None:
        take("CL upper", num_nodes)
    if request.optimizer.enforce_monotonic_descent:
        take("monotonic descent", num_nodes - 1)
    if request.optimizer.gamma_gradient_limit_deg_per_km is not None:
        take("gamma gradient upper", num_nodes - 1)
        take("gamma gradient lower", num_nodes - 1)
    if request.optimizer.gamma_curvature_limit_deg_per_km2 is not None:
        take("gamma curvature upper", num_nodes - 2)
        take("gamma curvature lower", num_nodes - 2)
    take("upstream CAS lower", 1)
    take("upstream CAS upper", 1)
    return blocks


def _render_solver_failure_summary(console: Console, *, request, raw_plan) -> None:
    _, _, equality_residuals, inequality_residuals, _, _ = _evaluation_bundle(request, raw_plan)
    num_nodes = len(raw_plan.s_m)
    dynamic = equality_residuals[: 6 * (num_nodes - 1)].reshape(num_nodes - 1, 6)
    segment_mid_s = 0.5 * (raw_plan.s_m[:-1] + raw_plan.s_m[1:])
    state_names = ("h [m]", "v_tas [m/s]", "t [s]", "cross-track [m]", "heading err [rad]", "bank [rad]")

    summary = Table(title="Solver failure summary", box=box.SIMPLE_HEAVY, expand=False)
    summary.add_column("field")
    summary.add_column("value", justify="right")
    summary.add_column("notes")
    summary.add_row("solver success", f"{raw_plan.solver_success}", "raw NLP result")
    summary.add_row("status", f"{raw_plan.solver_status}", "SciPy trust-constr status")
    summary.add_row("message", raw_plan.solver_message, "")
    summary.add_row("objective", f"{float(raw_plan.objective_value):+.6f}", "")
    summary.add_row("constraint slack", f"{float(raw_plan.constraint_slack):.3e}", "")
    summary.add_row("collocation max", f"{float(raw_plan.collocation_residual_max):.3e}", "max equality residual")
    summary.add_row("replay max", f"{float(raw_plan.replay_residual_max):.3e}", "post-solve IVP replay")
    summary.add_row("TOD", f"{float(raw_plan.tod_m):,.1f} m", "")

    worst_state_idx = int(np.argmax(np.max(np.abs(dynamic), axis=0)))
    worst_state_residuals = np.abs(dynamic[:, worst_state_idx])
    worst_state_seg_idx = int(np.argmax(worst_state_residuals))
    summary.add_row(
        "worst collocation",
        f"{state_names[worst_state_idx]} @ {float(segment_mid_s[worst_state_seg_idx]):,.1f} m",
        f"{float(worst_state_residuals[worst_state_seg_idx]):.3e}",
    )

    console.rule("[bold red]Solver failure diagnostics[/bold red]")
    console.print(summary)

    table = Table(title="Constraint envelope margins", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("family")
    table.add_column("min margin", justify="right")
    table.add_column("worst s_m [m]", justify="right")
    table.add_column("status")
    for label, values in _constraint_blocks(request, raw_plan, inequality_residuals):
        values = np.asarray(values, dtype=float)
        worst_idx = int(np.argmin(values))
        min_margin = float(np.min(values))
        worst_s_m = float(raw_plan.s_m[min(worst_idx, len(raw_plan.s_m) - 1)])
        status = "feasible" if min_margin >= 0.0 else "violated"
        table.add_row(label, f"{min_margin:+.3e}", f"{worst_s_m:,.1f}", status)
    console.print(table)


def _plot_longitudinal_response(
    ax,
    *,
    raw_s_m,
    raw_y,
    simulation_s_m=None,
    replay_y=None,
    raw_label: str,
    raw_color: str,
    replay_color: str,
    x_scale: float = 1.0,
    scale: float = 1.0,
    raw_linewidth: float = 1.1,
    replay_linewidth: float = 2.0,
) -> None:
    ax.plot(raw_s_m * x_scale, raw_y * scale, color=raw_color, linewidth=raw_linewidth, label=raw_label)
    if simulation_s_m is not None and replay_y is not None:
        ax.plot(
            simulation_s_m * x_scale,
            replay_y * scale,
            color=replay_color,
            linewidth=replay_linewidth,
            label="Integrated response",
        )


def plot_tactical_longitudinal(bundle, *, not_converged: bool = False) -> None:
    request = bundle.request
    raw_plan = bundle.raw_plan
    plan = bundle.plan
    simulation = bundle.simulation
    if raw_plan is None:
        return

    raw_s_m = raw_plan.s_m
    raw_tod_m = float(raw_plan.s_m[-1])
    tactical_start_m = float(plan.s_m[-1]) if plan is not None else raw_tod_m
    h_lower_m, h_upper_m = request.constraints.h_bounds_many(raw_s_m)
    cas_lower_mps, cas_upper_mps = request.constraints.cas_bounds_many(raw_s_m)
    gamma_lower_rad, gamma_upper_rad = request.constraints.gamma_bounds_many(raw_s_m)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0), sharex=True)

    _plot_longitudinal_response(
        axes[0, 0],
        raw_s_m=raw_plan.s_m,
        raw_y=raw_plan.h_m,
        simulation_s_m=None if simulation is None else simulation.s_m,
        replay_y=None if simulation is None else simulation.h_m,
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
        simulation_s_m=None if simulation is None else simulation.s_m,
        replay_y=None if simulation is None else simulation.v_cas_mps,
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
        simulation_s_m=None if simulation is None else simulation.s_m,
        replay_y=None if simulation is None else np.rad2deg(simulation.gamma_rad),
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
        simulation_s_m=None if simulation is None else simulation.s_m,
        replay_y=None if simulation is None else simulation.thrust_n,
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
    title = "A320 Tactical Longitudinal Response"
    if not_converged:
        title += " [Not Converged]"
    fig.suptitle(title)
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
    title = "A320 Tactical Integrated Response Around Raw TOD"
    if bundle.plan is None or bundle.simulation is None:
        title += " [Not Converged]"
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    console = Console()
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

    console.rule("[bold cyan]Building coupled tactical descent profile[/bold cyan]")
    bundle = solve_tactical_command(
        command,
        fixes_csv=repo_root / "data/kdfw_procs/airport_related_fixes.csv",
        optimizer=OptimizerConfig(
            num_nodes=45,
            maxiter=800,
            idle_thrust_margin_fraction=0.03,
            gamma_smoothness_weight=5.0,
            enforce_monotonic_descent=True,
            gamma_gradient_limit_deg_per_km=0.35,
            gamma_curvature_limit_deg_per_km2=0.12,
            verbose=1,
        ),
        guidance=LateralGuidanceConfig(
            lookahead_m=2_500.0,
            cross_track_gain=1.0,
            track_error_gain=2.0,
        ),
        dt_s=0.5,
        prefer_smooth_idle=False,
        console=console,
    )
    plan = bundle.plan
    raw_plan = bundle.raw_plan
    simulation = bundle.simulation
    assert raw_plan is not None

    not_converged = plan is None or simulation is None
    if not_converged:
        _render_solver_failure_summary(console, request=bundle.request, raw_plan=raw_plan)
        print(raw_plan.to_pandas().head())
        print(raw_plan.to_pandas().tail())
    plot_tactical_longitudinal(bundle, not_converged=not_converged)
    if simulation is not None:
        plot_tod_replay_window(bundle)
        render_tod_neighborhood(console, bundle)

    if plan is not None and simulation is not None:
        extension_start_idx = len(raw_plan.s_m)
        first_extension_cas_mps = float(plan.v_cas_mps[extension_start_idx]) if extension_start_idx < len(plan.s_m) else float("nan")

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
    plt.show()


if __name__ == "__main__":
    main()
