from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openap import aero
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from simap import CoupledDescentPlanRequest, CoupledDescentPlanResult, planned_cas_bounds_mps
from simap.coupled_descent_planner import (
    _PlannerScale,
    _SolverProfilingState,
    _TrajectoryEvaluationCache,
    _equality_constraints,
    _inequality_constraints,
    _pack,
)
from simap.openap_adapter import openap_dT
from simap.units import m_to_ft, mps_to_kts

from .models import TacticalCommand, TacticalCondition, TacticalPlanBundle


@dataclass(frozen=True)
class ObjectiveBreakdown:
    total: float
    optimizer_objective: float
    tod_reward: float
    thrust_penalty: float
    gamma_smoothness: float
    cross_track_penalty: float
    heading_error_penalty: float
    bank_penalty: float
    roll_rate_penalty: float
    slack_penalty: float
    thrust_delta_rms_n: float
    thrust_delta_peak_n: float
    gamma_step_max_deg: float
    max_abs_cross_track_m: float
    max_abs_heading_error_deg: float
    max_abs_bank_deg: float
    max_abs_roll_rate_rps: float
    ds_m: float
    thrust_scale_n: float
    lateral_scale_m: float
    heading_scale_deg: float
    bank_scale_deg: float
    roll_scale_rps: float


def _fmt_ft(value_m: float) -> str:
    return f"{m_to_ft(float(value_m)):,.1f}"


def _fmt_kt(value_mps: float) -> str:
    return f"{mps_to_kts(float(value_mps)):.1f}"


def _fmt_signed_ft(value_m: float) -> str:
    return f"{m_to_ft(float(value_m)):+.1f}"


def _fmt_signed_kt(value_mps: float) -> str:
    return f"{mps_to_kts(float(value_mps)):+.1f}"


def _fmt_deg(value_rad: float) -> str:
    return f"{np.rad2deg(float(value_rad)):+.2f}"


def _fmt_signed_deg(value_rad: float) -> str:
    return f"{np.rad2deg(float(value_rad)):+.2f}"


def _fmt_signed_m(value_m: float) -> str:
    return f"{float(value_m):+.3f}"


def _fmt_signed_s(value_s: float) -> str:
    return f"{float(value_s):+.3f}"


def _fmt_speed_window(lower_mps: float, upper_mps: float) -> str:
    lower_kt = mps_to_kts(float(lower_mps))
    if np.isinf(upper_mps):
        return f"{lower_kt:.1f} .. ∞"
    upper_kt = mps_to_kts(float(upper_mps))
    if np.isclose(lower_kt, upper_kt, rtol=0.0, atol=1e-9):
        return f"{lower_kt:.1f}"
    return f"{lower_kt:.1f} .. {upper_kt:.1f}"


def _fmt_margin(value: float) -> str:
    return f"{value:+.3e}"


def _threshold_v_tas_mps(request: CoupledDescentPlanRequest) -> float:
    threshold_delta_isa = request.weather.delta_isa_K(0.0, request.threshold.h_m, 0.0)
    return float(aero.cas2tas(request.threshold.cas_mps, request.threshold.h_m, dT=openap_dT(threshold_delta_isa)))


def _pack_plan(request: CoupledDescentPlanRequest, plan: CoupledDescentPlanResult) -> np.ndarray:
    return _pack(
        h_m=plan.h_m,
        v_tas_mps=plan.v_tas_mps,
        t_s=plan.t_s,
        cross_track_m=plan.cross_track_m,
        heading_error_rad=plan.heading_error_rad,
        phi_rad=plan.phi_rad,
        gamma_rad=plan.gamma_rad,
        thrust_n=plan.thrust_n,
        roll_rate_rps=plan.roll_rate_rps,
        tod_m=float(plan.tod_m),
        constraint_slack=float(plan.constraint_slack),
    )


def _evaluation_bundle(
    request: CoupledDescentPlanRequest,
    plan: CoupledDescentPlanResult,
) -> tuple[
    object,
    _PlannerScale,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    threshold_v_tas = _threshold_v_tas_mps(request)
    scale = _PlannerScale.from_request(request)
    profiling = _SolverProfilingState()
    evaluation_cache = _TrajectoryEvaluationCache(request=request, profiling=profiling)
    z = _pack_plan(request, plan)
    evaluation = evaluation_cache.evaluate(z)
    equality = _equality_constraints(
        z,
        request=request,
        threshold_v_tas=threshold_v_tas,
        evaluation_cache=evaluation_cache,
    )
    inequality = _inequality_constraints(
        z,
        request=request,
        scale=scale,
        evaluation_cache=evaluation_cache,
    )
    return evaluation, scale, equality, inequality, z, threshold_v_tas


def _objective_breakdown(
    request: CoupledDescentPlanRequest,
    evaluation,
    scale: _PlannerScale,
) -> ObjectiveBreakdown:
    ds_m = max(float(evaluation.tod_m) / max(request.optimizer.num_nodes - 1, 1), 1.0)
    thrust_scale = max(1.0, float(scale.thrust_n))
    thrust_delta = (evaluation.thrust_n - evaluation.idle_thrust_n) / thrust_scale
    gamma_gradient = np.diff(evaluation.gamma_rad) / max(ds_m, 1.0)
    lateral_scale = max(100.0, float(np.max(np.abs(evaluation.cross_track_m))), 1.0)
    heading_scale_rad = np.deg2rad(5.0)
    bank_scale_rad = np.deg2rad(20.0)
    roll_scale = max(1e-3, max(mode.p_max_rps for mode in (request.cfg.clean, request.cfg.approach, request.cfg.final)))

    tod_reward = -request.optimizer.tod_reward_weight * (float(evaluation.tod_m) / 1_000.0)
    thrust_penalty = (
        request.optimizer.thrust_penalty_weight
        * (float(evaluation.tod_m) / 1_000.0)
        * float(np.mean(thrust_delta**2))
    )
    gamma_smoothness = (
        request.optimizer.gamma_smoothness_weight
        * float(np.mean((gamma_gradient / np.deg2rad(1.0) * 1_000.0) ** 2))
    )
    cross_track_penalty = request.optimizer.cross_track_penalty_weight * float(
        np.mean((evaluation.cross_track_m / lateral_scale) ** 2)
    )
    heading_error_penalty = request.optimizer.heading_error_penalty_weight * float(
        np.mean((evaluation.heading_error_rad / heading_scale_rad) ** 2)
    )
    bank_penalty = request.optimizer.bank_penalty_weight * float(np.mean((evaluation.phi_rad / bank_scale_rad) ** 2))
    roll_rate_penalty = request.optimizer.roll_rate_penalty_weight * float(
        np.mean((evaluation.roll_rate_rps / roll_scale) ** 2)
    )
    slack_penalty = request.optimizer.slack_penalty_weight * float(evaluation.constraint_slack**2)
    total = (
        tod_reward
        + thrust_penalty
        + gamma_smoothness
        + cross_track_penalty
        + heading_error_penalty
        + bank_penalty
        + roll_rate_penalty
        + slack_penalty
    )

    return ObjectiveBreakdown(
        total=float(total),
        optimizer_objective=float(total),
        tod_reward=float(tod_reward),
        thrust_penalty=float(thrust_penalty),
        gamma_smoothness=float(gamma_smoothness),
        cross_track_penalty=float(cross_track_penalty),
        heading_error_penalty=float(heading_error_penalty),
        bank_penalty=float(bank_penalty),
        roll_rate_penalty=float(roll_rate_penalty),
        slack_penalty=float(slack_penalty),
        thrust_delta_rms_n=float(np.sqrt(np.mean((evaluation.thrust_n - evaluation.idle_thrust_n) ** 2))),
        thrust_delta_peak_n=float(np.max(np.abs(evaluation.thrust_n - evaluation.idle_thrust_n))),
        gamma_step_max_deg=float(np.max(np.abs(np.rad2deg(np.diff(evaluation.gamma_rad)))))
        if len(evaluation.gamma_rad) > 1
        else 0.0,
        max_abs_cross_track_m=float(np.max(np.abs(evaluation.cross_track_m))),
        max_abs_heading_error_deg=float(np.max(np.abs(np.rad2deg(evaluation.heading_error_rad)))),
        max_abs_bank_deg=float(np.max(np.abs(np.rad2deg(evaluation.phi_rad)))),
        max_abs_roll_rate_rps=float(np.max(np.abs(evaluation.roll_rate_rps))),
        ds_m=float(ds_m),
        thrust_scale_n=float(thrust_scale),
        lateral_scale_m=float(lateral_scale),
        heading_scale_deg=5.0,
        bank_scale_deg=20.0,
        roll_scale_rps=float(roll_scale),
    )


def _boundary_snapshot_table(plan: CoupledDescentPlanResult) -> Table:
    table = Table(title="Boundary snapshot", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("point")
    table.add_column("s_m [m]", justify="right")
    table.add_column("h [ft]", justify="right")
    table.add_column("CAS [kt]", justify="right")
    table.add_column("gamma [deg]", justify="right")
    table.add_column("cross-track [m]", justify="right")
    table.add_column("heading err [deg]", justify="right")
    table.add_column("bank [deg]", justify="right")
    table.add_column("thrust [N]", justify="right")
    table.add_row(
        "threshold",
        "0.0",
        _fmt_ft(plan.h_m[0]),
        _fmt_kt(plan.v_cas_mps[0]),
        _fmt_deg(plan.gamma_rad[0]),
        f"{float(plan.cross_track_m[0]):+.3f}",
        _fmt_deg(plan.heading_error_rad[0]),
        _fmt_deg(plan.phi_rad[0]),
        f"{float(plan.thrust_n[0]):,.1f}",
    )
    table.add_row(
        "raw TOD / upstream splice",
        f"{float(plan.tod_m):,.1f}",
        _fmt_ft(plan.h_m[-1]),
        _fmt_kt(plan.v_cas_mps[-1]),
        _fmt_deg(plan.gamma_rad[-1]),
        f"{float(plan.cross_track_m[-1]):+.3f}",
        _fmt_deg(plan.heading_error_rad[-1]),
        _fmt_deg(plan.phi_rad[-1]),
        f"{float(plan.thrust_n[-1]):,.1f}",
    )
    return table


def _render_objective_table(console: Console, request: CoupledDescentPlanRequest, breakdown: ObjectiveBreakdown, plan: CoupledDescentPlanResult) -> None:
    table = Table(title="Objective breakdown", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("term")
    table.add_column("value", justify="right")
    table.add_column("notes")
    table.add_row("TOD reward", f"{breakdown.tod_reward:+.6f}", "larger TOD is rewarded")
    table.add_row("thrust penalty", f"{breakdown.thrust_penalty:+.6f}", "mean squared idle-thrust delta")
    table.add_row("gamma smoothness", f"{breakdown.gamma_smoothness:+.6f}", "mean squared gamma gradient")
    table.add_row("cross-track penalty", f"{breakdown.cross_track_penalty:+.6f}", "normalized by lateral scale")
    table.add_row("heading error penalty", f"{breakdown.heading_error_penalty:+.6f}", "normalized by 5 deg")
    table.add_row("bank penalty", f"{breakdown.bank_penalty:+.6f}", "normalized by 20 deg")
    table.add_row("roll rate penalty", f"{breakdown.roll_rate_penalty:+.6f}", "normalized by mode roll limit")
    table.add_row("slack penalty", f"{breakdown.slack_penalty:+.6f}", "constraint slack squared")
    table.add_row(
        "recomputed total",
        f"{breakdown.total:+.6f}",
        f"optimizer objective={float(plan.objective_value):+.6f}, Δ={breakdown.total - float(plan.objective_value):+.3e}",
    )
    console.print(table)


def _render_support_metrics_table(console: Console, breakdown: ObjectiveBreakdown) -> None:
    table = Table(title="Signal magnitudes and normalization", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("notes")
    table.add_row("thrust delta RMS [N]", f"{breakdown.thrust_delta_rms_n:,.1f}", "raw thrust minus idle thrust")
    table.add_row("thrust delta peak [N]", f"{breakdown.thrust_delta_peak_n:,.1f}", "largest absolute thrust delta")
    table.add_row("gamma step max [deg]", f"{breakdown.gamma_step_max_deg:.3f}", "largest adjacent-node gamma change")
    table.add_row("max |cross-track| [m]", f"{breakdown.max_abs_cross_track_m:,.3f}", "used in the lateral scale clamp")
    table.add_row("max |heading error| [deg]", f"{breakdown.max_abs_heading_error_deg:.3f}", "")
    table.add_row("max |bank| [deg]", f"{breakdown.max_abs_bank_deg:.3f}", "")
    table.add_row("max |roll rate| [rps]", f"{breakdown.max_abs_roll_rate_rps:.3f}", "")
    table.add_row("ds_m [m]", f"{breakdown.ds_m:,.3f}", "TOD span divided by collocation segments")
    table.add_row("thrust scale [N]", f"{breakdown.thrust_scale_n:,.1f}", "from thrust bounds")
    table.add_row("lateral scale [m]", f"{breakdown.lateral_scale_m:,.1f}", "max(100 m, peak cross-track)")
    table.add_row("heading scale [deg]", f"{breakdown.heading_scale_deg:.1f}", "fixed objective normalization")
    table.add_row("bank scale [deg]", f"{breakdown.bank_scale_deg:.1f}", "fixed objective normalization")
    table.add_row("roll scale [rps]", f"{breakdown.roll_scale_rps:.3f}", "largest mode roll-rate limit")
    console.print(table)


def _render_equality_residual_table(console: Console, request: CoupledDescentPlanRequest, plan: CoupledDescentPlanResult, equality_residuals: np.ndarray) -> None:
    num_nodes = request.optimizer.num_nodes
    dynamic = equality_residuals[: 6 * (num_nodes - 1)].reshape(num_nodes - 1, 6)
    segment_mid_s = 0.5 * (plan.s_m[:-1] + plan.s_m[1:])
    table = Table(title="Collocation residuals", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("state")
    table.add_column("max |residual|", justify="right")
    table.add_column("worst s_m [m]", justify="right")
    state_names = ("h [m]", "v_tas [m/s]", "t [s]", "cross-track [m]", "heading err [rad]", "bank [rad]")
    for idx, name in enumerate(state_names):
        abs_residual = np.abs(dynamic[:, idx])
        worst_idx = int(np.argmax(abs_residual))
        table.add_row(name, f"{float(np.max(abs_residual)):.3e}", f"{float(segment_mid_s[worst_idx]):,.1f}")
    console.print(table)


def _render_inequality_residual_table(
    console: Console,
    request: CoupledDescentPlanRequest,
    plan: CoupledDescentPlanResult,
    inequality_residuals: np.ndarray,
) -> None:
    blocks: list[tuple[str, np.ndarray]] = []
    num_nodes = len(plan.s_m)
    offset = 0

    def take(label: str, width: int) -> None:
        nonlocal offset
        blocks.append((label, inequality_residuals[offset : offset + width]))
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

    table = Table(title="Constraint envelope margins", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("family")
    table.add_column("min margin", justify="right")
    table.add_column("worst s_m [m]", justify="right")
    table.add_column("status")

    for label, values in blocks:
        values = np.asarray(values, dtype=float)
        worst_idx = int(np.argmin(values))
        min_margin = float(np.min(values))
        worst_s_m = float(plan.s_m[min(worst_idx, len(plan.s_m) - 1)])
        status = "feasible" if min_margin >= 0.0 else "violated"
        table.add_row(label, _fmt_margin(min_margin), f"{worst_s_m:,.1f}", status)
    console.print(table)


def _render_endpoint_defects_table(
    console: Console,
    request: CoupledDescentPlanRequest,
    plan: CoupledDescentPlanResult,
) -> None:
    table = Table(title="Endpoint defects", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("endpoint")
    table.add_column("quantity")
    table.add_column("actual", justify="right")
    table.add_column("target", justify="right")
    table.add_column("residual", justify="right")

    rows = [
        (
            "threshold",
            "altitude [ft]",
            _fmt_ft(plan.h_m[0]),
            _fmt_ft(request.threshold.h_m),
            _fmt_signed_ft(plan.h_m[0] - request.threshold.h_m),
        ),
        (
            "threshold",
            "CAS [kt]",
            _fmt_kt(plan.v_cas_mps[0]),
            _fmt_kt(request.threshold.cas_mps),
            _fmt_signed_kt(plan.v_cas_mps[0] - request.threshold.cas_mps),
        ),
        (
            "threshold",
            "gamma [deg]",
            _fmt_deg(plan.gamma_rad[0]),
            _fmt_deg(request.threshold.gamma_rad),
            _fmt_signed_deg(plan.gamma_rad[0] - request.threshold.gamma_rad),
        ),
        (
            "threshold",
            "cross-track [m]",
            f"{float(plan.cross_track_m[0]):+.3f}",
            f"{request.threshold_lateral.cross_track_m:+.3f}",
            _fmt_signed_m(plan.cross_track_m[0] - request.threshold_lateral.cross_track_m),
        ),
        (
            "threshold",
            "heading err [deg]",
            _fmt_deg(plan.heading_error_rad[0]),
            _fmt_deg(request.threshold_lateral.heading_error_rad),
            _fmt_signed_deg(plan.heading_error_rad[0] - request.threshold_lateral.heading_error_rad),
        ),
        (
            "threshold",
            "bank [deg]",
            _fmt_deg(plan.phi_rad[0]),
            _fmt_deg(request.threshold_lateral.bank_rad),
            _fmt_signed_deg(plan.phi_rad[0] - request.threshold_lateral.bank_rad),
        ),
        (
            "threshold",
            "time [s]",
            f"{float(plan.t_s[0]):.3f}",
            "0.000",
            _fmt_signed_s(plan.t_s[0]),
        ),
        (
            "upstream",
            "altitude [ft]",
            _fmt_ft(plan.h_m[-1]),
            _fmt_ft(request.upstream.h_m),
            _fmt_signed_ft(plan.h_m[-1] - request.upstream.h_m),
        ),
        (
            "upstream",
            "gamma [deg]",
            _fmt_deg(plan.gamma_rad[-1]),
            _fmt_deg(request.upstream.gamma_rad),
            _fmt_signed_deg(plan.gamma_rad[-1] - request.upstream.gamma_rad),
        ),
    ]

    if request.upstream.cas_target_mps is not None:
        rows.append(
            (
                "upstream",
                "CAS [kt]",
                _fmt_kt(plan.v_cas_mps[-1]),
                _fmt_kt(request.upstream.cas_target_mps),
                _fmt_signed_kt(plan.v_cas_mps[-1] - request.upstream.cas_target_mps),
            )
        )

    rows.extend(
        [
            (
                "upstream",
                "cross-track [m]",
                f"{float(plan.cross_track_m[-1]):+.3f}",
                f"{request.upstream_lateral.cross_track_m:+.3f}",
                _fmt_signed_m(plan.cross_track_m[-1] - request.upstream_lateral.cross_track_m),
            ),
            (
                "upstream",
                "heading err [deg]",
                _fmt_deg(plan.heading_error_rad[-1]),
                _fmt_deg(request.upstream_lateral.heading_error_rad),
                _fmt_signed_deg(plan.heading_error_rad[-1] - request.upstream_lateral.heading_error_rad),
            ),
            (
                "upstream",
                "bank [deg]",
                _fmt_deg(plan.phi_rad[-1]),
                _fmt_deg(request.upstream_lateral.bank_rad),
                _fmt_signed_deg(plan.phi_rad[-1] - request.upstream_lateral.bank_rad),
            ),
        ]
    )

    for row in rows:
        table.add_row(*row)
    console.print(table)


def render_tactical_setup(console: Console, *, bundle: TacticalPlanBundle) -> None:
    command = bundle.command
    request = bundle.request
    path = bundle.path
    path_length_m = float(request.reference_path.total_length_m)
    max_tod_m = float(min(request.constraints.s_m[-1], path_length_m))
    initial_tod_guess = request.initial_tod_guess_m
    if initial_tod_guess is None:
        initial_tod_guess = request.optimizer.initial_tod_guess_m

    console.rule("[bold cyan]Tactical NLP setup[/bold cyan]")
    console.print(
        Panel.fit(
            f"[bold]Route[/bold]: {path.identifiers[0]} -> {path.identifiers[-1]} ({len(path.waypoints)} waypoints)\n"
            f"[bold]Reference length[/bold]: {path_length_m:,.1f} m\n"
            f"[bold]TOD search ceiling[/bold]: {max_tod_m:,.1f} m\n"
            f"[bold]Collocation points[/bold]: {request.optimizer.num_nodes}\n"
            f"[bold]Max iterations[/bold]: {request.optimizer.maxiter}\n"
            f"[bold]Initial TOD guess[/bold]: "
            f"{'auto' if initial_tod_guess is None else f'{float(initial_tod_guess):,.1f} m'}\n"
            f"[bold]Minimum along-track speed[/bold]: {request.optimizer.min_alongtrack_speed_mps:.2f} m/s",
            title="Solve configuration",
            border_style="cyan",
        )
    )

    boundary_table = Table(title="Boundary conditions", box=box.SIMPLE_HEAVY, expand=False)
    boundary_table.add_column("point")
    boundary_table.add_column("s_m [m]", justify="right")
    boundary_table.add_column("h [ft]", justify="right")
    boundary_table.add_column("CAS [kt]", justify="right")
    boundary_table.add_column("gamma [deg]", justify="right")
    boundary_table.add_column("cross-track [m]", justify="right")
    boundary_table.add_column("heading err [deg]", justify="right")
    boundary_table.add_column("bank [deg]", justify="right")
    boundary_table.add_column("notes")
    boundary_table.add_row(
        "threshold",
        "0.0",
        f"{_fmt_ft(request.threshold.h_m)}",
        f"{_fmt_kt(request.threshold.cas_mps)}",
        _fmt_deg(request.threshold.gamma_rad),
        f"{request.threshold_lateral.cross_track_m:+.3f}",
        _fmt_deg(request.threshold_lateral.heading_error_rad),
        _fmt_deg(request.threshold_lateral.bank_rad),
        "runway threshold",
    )
    cas_target = request.upstream.cas_target_mps
    if cas_target is None:
        cas_note = f"window {_fmt_speed_window(request.upstream.cas_lower_mps, request.upstream.cas_upper_mps)}"
    else:
        cas_note = f"target {_fmt_kt(cas_target)} kt"
    boundary_table.add_row(
        "upstream / tactical start target",
        f"{path_length_m:,.1f}",
        f"{_fmt_ft(request.upstream.h_m)}",
        cas_note,
        _fmt_deg(request.upstream.gamma_rad),
        f"{request.upstream_lateral.cross_track_m:+.3f}",
        _fmt_deg(request.upstream_lateral.heading_error_rad),
        _fmt_deg(request.upstream_lateral.bank_rad),
        "endpoint the NLP is trying to meet",
    )
    console.print(boundary_table)

    solver_table = Table(title="Solver settings", box=box.SIMPLE_HEAVY, expand=False)
    solver_table.add_column("setting")
    solver_table.add_column("value", justify="right")
    solver_table.add_column("notes")
    solver_table.add_row("constraint tolerance", f"{request.optimizer.constraint_tolerance:.1e}", "")
    solver_table.add_row("initial slack", f"{request.optimizer.initial_slack:.3f}", "")
    solver_table.add_row("verbose", f"{request.optimizer.verbose:d}", "passed to trust-constr")
    console.print(solver_table)

    objective_table = Table(title="Objective weights", box=box.SIMPLE_HEAVY, expand=False)
    objective_table.add_column("term")
    objective_table.add_column("weight", justify="right")
    objective_table.add_row("TOD reward", f"{request.optimizer.tod_reward_weight:.3f}")
    objective_table.add_row("thrust penalty", f"{request.optimizer.thrust_penalty_weight:.3f}")
    objective_table.add_row("gamma smoothness", f"{request.optimizer.gamma_smoothness_weight:.3f}")
    objective_table.add_row("cross-track penalty", f"{request.optimizer.cross_track_penalty_weight:.3f}")
    objective_table.add_row("heading error penalty", f"{request.optimizer.heading_error_penalty_weight:.3f}")
    objective_table.add_row("bank penalty", f"{request.optimizer.bank_penalty_weight:.3f}")
    objective_table.add_row("roll rate penalty", f"{request.optimizer.roll_rate_penalty_weight:.3f}")
    objective_table.add_row("slack penalty", f"{request.optimizer.slack_penalty_weight:.3e}")
    console.print(objective_table)

    altitude_table = Table(title="Altitude envelope", box=box.SIMPLE_HEAVY, expand=False)
    altitude_table.add_column("s_m [m]", justify="right")
    altitude_table.add_column("lower [ft]", justify="right")
    altitude_table.add_column("upper [ft]", justify="right")
    for s_m, lower_m, upper_m in zip(request.constraints.s_m, request.constraints.h_lower_m, request.constraints.h_upper_m, strict=True):
        altitude_table.add_row(f"{float(s_m):,.1f}", _fmt_ft(lower_m), _fmt_ft(upper_m))
    console.print(altitude_table)

    route_lower = request.constraints.cas_lower_mps
    route_upper = request.constraints.cas_upper_mps
    mode_bounds = np.asarray([planned_cas_bounds_mps(request.cfg, float(s_m)) for s_m in request.constraints.s_m], dtype=float)
    mode_lower = mode_bounds[:, 0]
    mode_upper = mode_bounds[:, 1]
    effective_lower = np.maximum(route_lower, mode_lower)
    effective_upper = np.minimum(route_upper, mode_upper)
    cas_table = Table(title="CAS envelope and mode limits", box=box.SIMPLE_HEAVY, expand=False)
    cas_table.add_column("s_m [m]", justify="right")
    cas_table.add_column("route lower [kt]", justify="right")
    cas_table.add_column("route upper [kt]", justify="right")
    cas_table.add_column("mode lower [kt]", justify="right")
    cas_table.add_column("mode upper [kt]", justify="right")
    cas_table.add_column("effective lower [kt]", justify="right")
    cas_table.add_column("effective upper [kt]", justify="right")
    for s_m, r_lower, r_upper, m_lower, m_upper, e_lower, e_upper in zip(
        request.constraints.s_m,
        route_lower,
        route_upper,
        mode_lower,
        mode_upper,
        effective_lower,
        effective_upper,
        strict=True,
    ):
        cas_table.add_row(
            f"{float(s_m):,.1f}",
            _fmt_kt(r_lower),
            _fmt_kt(r_upper),
            _fmt_kt(m_lower),
            _fmt_kt(m_upper),
            _fmt_kt(e_lower),
            _fmt_kt(e_upper),
        )
    console.print(cas_table)

    gamma_lower, gamma_upper = request.constraints.gamma_bounds_many(request.constraints.s_m)
    if gamma_lower is not None and gamma_upper is not None:
        gamma_table = Table(title="Flight-path angle envelope", box=box.SIMPLE_HEAVY, expand=False)
        gamma_table.add_column("s_m [m]", justify="right")
        gamma_table.add_column("lower [deg]", justify="right")
        gamma_table.add_column("upper [deg]", justify="right")
        for s_m, lower_rad, upper_rad in zip(request.constraints.s_m, gamma_lower, gamma_upper, strict=True):
            gamma_table.add_row(f"{float(s_m):,.1f}", _fmt_deg(lower_rad), _fmt_deg(upper_rad))
        console.print(gamma_table)

    if command.altitude_constraints:
        constraint_table = Table(title="Waypoint altitude constraints", box=box.SIMPLE_HEAVY, expand=False)
        constraint_table.add_column("fix")
        constraint_table.add_column("lower [ft]", justify="right")
        constraint_table.add_column("upper [ft]", justify="right")
        for constraint in command.altitude_constraints:
            constraint_table.add_row(
                constraint.fix_identifier.upper(),
                "—" if constraint.lower_ft is None else f"{constraint.lower_ft:,.1f}",
                "—" if constraint.upper_ft is None else f"{constraint.upper_ft:,.1f}",
            )
        console.print(constraint_table)
    else:
        console.print(
            Panel.fit(
                "No waypoint-specific altitude constraints were supplied.",
                title="Waypoint altitude constraints",
                border_style="dim",
            )
        )


def render_coupled_solution(
    console: Console,
    *,
    request: CoupledDescentPlanRequest,
    plan: CoupledDescentPlanResult,
) -> None:
    evaluation, scale, equality_residuals, inequality_residuals, _, _ = _evaluation_bundle(request, plan)
    breakdown = _objective_breakdown(request, evaluation, scale)

    if plan.solver_success and float(plan.collocation_residual_max) <= 1e-2 and float(plan.replay_residual_max) <= 25.0:
        border_style = "green"
    elif plan.solver_success:
        border_style = "yellow"
    else:
        border_style = "red"

    console.rule("[bold cyan]Coupled NLP result[/bold cyan]")
    console.print(
        Panel.fit(
            f"[bold]Solver success[/bold]: {plan.solver_success}\n"
            f"[bold]Status[/bold]: {plan.solver_status}\n"
            f"[bold]Message[/bold]: {plan.solver_message}\n"
            f"[bold]Objective[/bold]: {float(plan.objective_value):+.6f} "
            f"(recomputed {breakdown.total:+.6f}, Δ {breakdown.total - float(plan.objective_value):+.3e})\n"
            f"[bold]TOD[/bold]: {float(plan.tod_m):,.1f} m\n"
            f"[bold]Extension remaining[/bold]: {max(0.0, float(request.reference_path.total_length_m - plan.tod_m)):,.1f} m\n"
            f"[bold]Collocation residual max[/bold]: {float(plan.collocation_residual_max):.3e}\n"
            f"[bold]Replay residual max[/bold]: {float(plan.replay_residual_max):.3e}\n"
            f"[bold]Constraint slack[/bold]: {float(plan.constraint_slack):.3e}\n"
            f"[bold]Solve wall time[/bold]: {plan.solve_profile.total_wall_time_s:.3f} s "
            f"(objective calls={plan.solve_profile.objective_calls}, "
            f"equality calls={plan.solve_profile.equality_calls}, "
            f"inequality calls={plan.solve_profile.inequality_calls}, "
            f"trajectory evals={plan.solve_profile.trajectory_evaluations}, "
            f"cache hits={plan.solve_profile.trajectory_cache_hits})",
            title="Coupled descent diagnostics",
            border_style=border_style,
        )
    )

    console.print(_boundary_snapshot_table(plan))
    _render_endpoint_defects_table(console, request, plan)
    _render_objective_table(console, request, breakdown, plan)
    _render_support_metrics_table(console, breakdown)
    _render_equality_residual_table(console, request, plan, equality_residuals)
    _render_inequality_residual_table(console, request, plan, inequality_residuals)


def render_tactical_extension_summary(
    console: Console,
    *,
    request: CoupledDescentPlanRequest,
    raw_plan: CoupledDescentPlanResult,
    extended_plan: CoupledDescentPlanResult,
    start_condition: TacticalCondition,
    start_s_m: float,
    num_extension_nodes: int,
) -> None:
    extension_distance_m = float(start_s_m - raw_plan.tod_m)
    console.rule("[bold cyan]Tactical extension[/bold cyan]")
    console.print(
        Panel.fit(
            "The extension segment is not re-optimized. It carries the raw TOD state forward to the tactical start "
            "point by holding altitude and CAS fixed, zeroing the lateral and attitude offsets, integrating time "
            "from distance using TAS, and recomputing thrust from level-flight drag at each extension node.\n"
            f"[bold]Extension distance[/bold]: {extension_distance_m:,.1f} m\n"
            f"[bold]Extension nodes[/bold]: {num_extension_nodes}\n"
            f"[bold]Requested tactical start[/bold]: h={start_condition.altitude_ft:,.1f} ft, "
            f"CAS={start_condition.cas_kts:.1f} kt, gamma={start_condition.gamma_deg:+.2f} deg",
            title="What the extension does",
            border_style="cyan",
        )
    )

    table = Table(title="TOD versus tactical start", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("point")
    table.add_column("s_m [m]", justify="right")
    table.add_column("t_s [s]", justify="right")
    table.add_column("h [ft]", justify="right")
    table.add_column("CAS [kt]", justify="right")
    table.add_column("TAS [kt]", justify="right")
    table.add_column("gamma [deg]", justify="right")
    table.add_column("cross-track [m]", justify="right")
    table.add_column("heading err [deg]", justify="right")
    table.add_column("thrust [N]", justify="right")
    table.add_row(
        "raw TOD boundary",
        f"{float(raw_plan.s_m[-1]):,.1f}",
        f"{float(raw_plan.t_s[-1]):,.2f}",
        _fmt_ft(raw_plan.h_m[-1]),
        _fmt_kt(raw_plan.v_cas_mps[-1]),
        _fmt_kt(raw_plan.v_tas_mps[-1]),
        _fmt_deg(raw_plan.gamma_rad[-1]),
        f"{float(raw_plan.cross_track_m[-1]):+.3f}",
        _fmt_deg(raw_plan.heading_error_rad[-1]),
        f"{float(raw_plan.thrust_n[-1]):,.1f}",
    )
    table.add_row(
        "tactical start",
        f"{float(extended_plan.s_m[-1]):,.1f}",
        f"{float(extended_plan.t_s[-1]):,.2f}",
        _fmt_ft(extended_plan.h_m[-1]),
        _fmt_kt(extended_plan.v_cas_mps[-1]),
        _fmt_kt(extended_plan.v_tas_mps[-1]),
        _fmt_deg(extended_plan.gamma_rad[-1]),
        f"{float(extended_plan.cross_track_m[-1]):+.3f}",
        _fmt_deg(extended_plan.heading_error_rad[-1]),
        f"{float(extended_plan.thrust_n[-1]):,.1f}",
    )
    console.print(table)


def render_tod_neighborhood(
    console: Console,
    bundle: TacticalPlanBundle,
    *,
    plan_window: int = 4,
    simulation_window: int = 24,
) -> None:
    raw_plan = bundle.raw_plan
    plan = bundle.plan
    simulation = bundle.simulation
    if raw_plan is None or plan is None:
        return

    raw_tod_m = float(raw_plan.s_m[-1])
    tactical_start_m = float(plan.s_m[-1])
    extension_start_idx = len(raw_plan.s_m)

    console.rule("[bold cyan]Raw TOD neighborhood[/bold cyan]")
    console.print(
        Panel.fit(
            f"[bold]Raw TOD[/bold]: {raw_tod_m:,.1f} m\n"
            f"[bold]Tactical start[/bold]: {tactical_start_m:,.1f} m\n"
            f"[bold]Extension span[/bold]: {max(0.0, tactical_start_m - raw_tod_m):,.1f} m",
            title="Window around TOD",
            border_style="cyan",
        )
    )

    table = Table(title="Plan nodes around raw TOD", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("idx", justify="right")
    table.add_column("s_m [m]", justify="right")
    table.add_column("t_s [s]", justify="right")
    table.add_column("h [ft]", justify="right")
    table.add_column("CAS [kt]", justify="right")
    table.add_column("gamma [deg]", justify="right")
    table.add_column("thrust [N]", justify="right")
    table.add_column("note")
    start_idx = max(0, extension_start_idx - plan_window)
    end_idx = min(len(plan.s_m), extension_start_idx + plan_window + 1)
    for idx in range(start_idx, end_idx):
        note = ""
        if idx == extension_start_idx - 1:
            note = "last raw"
        elif idx == extension_start_idx:
            note = "first extension"
        elif idx == len(plan.s_m) - 1:
            note = "tactical start"
        table.add_row(
            f"{idx}",
            f"{float(plan.s_m[idx]):,.1f}",
            f"{float(plan.t_s[idx]):,.2f}",
            _fmt_ft(plan.h_m[idx]),
            _fmt_kt(plan.v_cas_mps[idx]),
            _fmt_deg(plan.gamma_rad[idx]),
            f"{float(plan.thrust_n[idx]):,.1f}",
            note,
        )
    console.print(table)

    if simulation is None:
        return

    closest_idx = int(np.argmin(np.abs(simulation.s_m - raw_tod_m)))
    tod_t_s = float(simulation.t_s[closest_idx])
    t_rel_s = simulation.t_s - tod_t_s
    mask = np.abs(t_rel_s) <= simulation_window

    sim_table = Table(title="Simulation time steps around raw TOD", box=box.SIMPLE_HEAVY, expand=False)
    sim_table.add_column("idx", justify="right")
    sim_table.add_column("t_rel [s]", justify="right")
    sim_table.add_column("s_m [m]", justify="right")
    sim_table.add_column("h [ft]", justify="right")
    sim_table.add_column("CAS [kt]", justify="right")
    sim_table.add_column("gamma [deg]", justify="right")
    sim_table.add_column("thrust [N]", justify="right")
    sim_table.add_column("ds to TOD [m]", justify="right")
    for idx in np.flatnonzero(mask):
        sim_table.add_row(
            f"{idx}",
            f"{float(t_rel_s[idx]):+.2f}",
            f"{float(simulation.s_m[idx]):,.1f}",
            _fmt_ft(simulation.h_m[idx]),
            _fmt_kt(simulation.v_cas_mps[idx]),
            _fmt_deg(simulation.gamma_rad[idx]),
            f"{float(simulation.thrust_n[idx]):,.1f}",
            f"{float(simulation.s_m[idx] - raw_tod_m):+.1f}",
        )
    console.print(sim_table)
