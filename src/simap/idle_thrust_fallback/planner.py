from __future__ import annotations

from time import perf_counter
from typing import Any, cast

import numpy as np
from openap import aero
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import bank_limit_rad, mode_for_s
from ..nlp_colloc.coupled import (
    CoupledDescentPlanRequest,
    CoupledDescentPlanResult,
    CoupledDescentSolveProfile,
    _SolverProfilingState,
    _TrajectoryEvaluationCache,
    _cas_from_tas,
    _equality_constraints,
    _integrate_time_profile,
    _pack,
    _planned_cas_bounds_many,
    _replay_solution,
)
from ..openap_adapter import openap_dT
from ..weather import ConstantWeather, alongtrack_wind_mps

__all__ = ["plan_idle_thrust_fallback"]


def _render_tod_search(
    console: Console,
    *,
    sample_tod_m: np.ndarray,
    sample_errors_mps: np.ndarray,
    target_cas_mps: float,
    target_label: str,
    lower_tod_m: float,
    max_tod_m: float,
    selected_tod_m: float,
    selected_error_mps: float,
    bracket: tuple[float, float] | None,
) -> None:
    console.rule("[bold cyan]Step 4.1  Choose top-of-descent distance[/bold cyan]")
    console.print(
        Panel.fit(
            f"[bold]Target CAS[/bold]: {target_cas_mps:.3f} m/s ({target_label})\n"
            f"[bold]Sample span[/bold]: {lower_tod_m:,.1f} m -> {max_tod_m:,.1f} m",
            title="TOD search setup",
            border_style="cyan",
        )
    )
    table = Table(title="Candidate TOD samples", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("idx", justify="right")
    table.add_column("TOD_m [m]", justify="right")
    table.add_column("final CAS error [m/s]", justify="right")
    table.add_column("note")
    exact_match = bracket is not None and np.isclose(bracket[0], bracket[1], rtol=0.0, atol=1e-9)
    for idx, (tod_m, error_mps) in enumerate(zip(sample_tod_m, sample_errors_mps, strict=True)):
        note = ""
        if bracket is not None:
            if np.isclose(tod_m, bracket[0], rtol=0.0, atol=1e-9) and np.isclose(tod_m, bracket[1], rtol=0.0, atol=1e-9):
                note = "[green]exact match[/green]"
            elif np.isclose(tod_m, bracket[0], rtol=0.0, atol=1e-9):
                note = "[cyan]bracket left[/cyan]"
            elif np.isclose(tod_m, bracket[1], rtol=0.0, atol=1e-9):
                note = "[cyan]bracket right[/cyan]"
        if np.isclose(tod_m, selected_tod_m, rtol=0.0, atol=1e-9):
            if exact_match:
                note = note or "[green]selected root candidate[/green]"
            elif bracket is None:
                note = note or "[yellow]closest sampled candidate[/yellow]"
        table.add_row(f"{idx}", f"{tod_m:,.1f}", f"{error_mps:+.4f}", note)
    console.print(table)

    if bracket is None:
        border_style = "yellow"
        title = "Closest sampled candidate"
        body = (
            f"[bold]Selected sample[/bold]: TOD = {selected_tod_m:,.1f} m\n"
            f"[bold]Final CAS error[/bold]: {selected_error_mps:+.4f} m/s"
        )
    elif exact_match:
        border_style = "green"
        title = "Exact candidate match"
        body = (
            f"[bold]Exact sample[/bold]: TOD = {selected_tod_m:,.1f} m\n"
            f"[bold]Final CAS error[/bold]: {selected_error_mps:+.4f} m/s"
        )
    else:
        border_style = "green"
        title = "Root finding result"
        body = (
            f"[bold]Bracket[/bold]: [{bracket[0]:,.1f}, {bracket[1]:,.1f}] m\n"
            f"[bold]Selected TOD[/bold]: {selected_tod_m:,.1f} m\n"
            f"[bold]Final CAS error[/bold]: {selected_error_mps:+.4f} m/s"
        )
    console.print(Panel.fit(body, title=title, border_style=border_style))


def _render_speed_envelope_enforcement(
    console: Console,
    *,
    s_m: np.ndarray,
    v_cas_before_mps: np.ndarray,
    cas_lower_env_mps: np.ndarray,
    mode_lower_mps: np.ndarray,
    v_cas_after_mps: np.ndarray,
) -> None:
    effective_lower_mps = np.maximum(cas_lower_env_mps, mode_lower_mps)
    clamped_mask = v_cas_before_mps < effective_lower_mps - 1e-9
    console.rule("[bold cyan]Step 4.4  Enforce CAS envelope[/bold cyan]")
    if not np.any(clamped_mask):
        console.print(
            Panel.fit(
                "No CAS samples were raised to the effective lower envelope.",
                title="CAS envelope enforcement",
                border_style="green",
            )
        )
        return

    indices = np.flatnonzero(clamped_mask)
    console.print(
        Panel.fit(
            f"[bold]Envelope active[/bold]: {indices.size} / {s_m.size} nodes\n"
            f"[bold]s-range[/bold]: {s_m[indices[0]]:,.1f} m -> {s_m[indices[-1]]:,.1f} m\n"
            f"[bold]Effective lower[/bold] = max(route constraint, mode-specific bound)",
            title="CAS envelope enforcement",
            border_style="yellow",
        )
    )
    table = Table(title="Raised CAS nodes", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("idx", justify="right")
    table.add_column("s_m [m]", justify="right")
    table.add_column("before [m/s]", justify="right")
    table.add_column("effective lower [m/s]", justify="right")
    table.add_column("after [m/s]", justify="right")
    table.add_column("driver")
    for idx in indices:
        driver = "route" if cas_lower_env_mps[idx] >= mode_lower_mps[idx] else "mode"
        table.add_row(
            f"{idx}",
            f"{s_m[idx]:,.1f}",
            f"{v_cas_before_mps[idx]:.3f}",
            f"{effective_lower_mps[idx]:.3f}",
            f"{v_cas_after_mps[idx]:.3f}",
            driver,
        )
    console.print(table)


def _render_thrust_reconstruction(
    console: Console,
    *,
    s_m: np.ndarray,
    modes: tuple[Any, ...],
    thrust_guess_n: np.ndarray,
    thrust_n: np.ndarray,
    thrust_lower_n: np.ndarray,
    thrust_upper_n: np.ndarray,
) -> None:
    changed_mask = ~np.isclose(thrust_guess_n, thrust_n, rtol=0.0, atol=1e-9)
    console.rule("[bold cyan]Step 4.5  Reconstruct thrust[/bold cyan]")
    if not np.any(changed_mask):
        console.print(
            Panel.fit(
                "The feed-forward thrust guess stayed inside the allowed bounds at every node.",
                title="Thrust reconstruction",
                border_style="green",
            )
        )
        return

    indices = np.flatnonzero(changed_mask)
    console.print(
        Panel.fit(
            f"[bold]Thrust adjusted[/bold] on {indices.size} / {s_m.size} nodes\n"
            f"[bold]s-range[/bold]: {s_m[indices[0]]:,.1f} m -> {s_m[indices[-1]]:,.1f} m",
            title="Thrust reconstruction",
            border_style="yellow",
        )
    )
    table = Table(title="Adjusted thrust nodes", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("idx", justify="right")
    table.add_column("s_m [m]", justify="right")
    table.add_column("mode")
    table.add_column("old thrust [N]", justify="right")
    table.add_column("new thrust [N]", justify="right")
    table.add_column("bounds [N]", justify="right")
    table.add_column("reason")
    for idx in indices:
        mode_label = getattr(modes[idx], "name", str(modes[idx]))
        if np.isclose(thrust_n[idx], thrust_lower_n[idx], rtol=0.0, atol=1e-9):
            reason = "raised to lower bound"
        elif np.isclose(thrust_n[idx], thrust_upper_n[idx], rtol=0.0, atol=1e-9):
            reason = "capped at upper bound"
        else:
            reason = "adjusted"
        table.add_row(
            f"{idx}",
            f"{s_m[idx]:,.1f}",
            mode_label,
            f"{thrust_guess_n[idx]:,.1f}",
            f"{thrust_n[idx]:,.1f}",
            f"{thrust_lower_n[idx]:,.1f} .. {thrust_upper_n[idx]:,.1f}",
            reason,
        )
    console.print(table)


def _render_reassembly_summary(
    console: Console,
    *,
    collocation_residual_max: float,
    replay_h_error_m: float,
    replay_v_error_mps: float,
    replay_t_error_s: float,
    replay_residual_max: float,
    solve_profile: CoupledDescentSolveProfile,
) -> None:
    console.rule("[bold cyan]Step 4.6  Reassemble outputs[/bold cyan]")
    table = Table(title="Explicit recomputations", box=box.SIMPLE_HEAVY, expand=False)
    table.add_column("group")
    table.add_column("recomputed quantities")
    table.add_row("speed profile", "v_cas_mps, v_tas_mps, t_s after the envelope clamp")
    table.add_row("geometry", "track angle, NE position, latitude/longitude")
    table.add_row("limits", "mode-by-node, phi_max_rad")
    table.add_row(
        "ground motion",
        "ground velocity, ground speed, along-track speed, cross-track speed, track error",
    )
    table.add_row("validation", "packed state, collocation residual max, replay residuals, solve_profile snapshot")
    console.print(table)

    validation_style = "green" if collocation_residual_max <= 1e-2 and replay_residual_max <= 25.0 else "yellow"
    console.print(
        Panel.fit(
            f"[bold]collocation_residual_max[/bold]: {collocation_residual_max:.3e}\n"
            f"[bold]replay_h_error_m[/bold]: {replay_h_error_m:.3f} m\n"
            f"[bold]replay_v_error_mps[/bold]: {replay_v_error_mps:.3f} m/s\n"
            f"[bold]replay_t_error_s[/bold]: {replay_t_error_s:.3f} s\n"
            f"[bold]replay_residual_max[/bold]: {replay_residual_max:.3e}\n"
            f"[bold]solve_profile[/bold]: wall={solve_profile.total_wall_time_s:.3f} s, "
            f"objective_calls={solve_profile.objective_calls}, equality_calls={solve_profile.equality_calls}, "
            f"trajectory_evals={solve_profile.trajectory_evaluations}, cache_hits={solve_profile.trajectory_cache_hits}",
            title="Validation snapshot",
            border_style=validation_style,
        )
    )


def plan_idle_thrust_fallback(
    request: CoupledDescentPlanRequest,
    *,
    num_nodes: int | None = None,
    console: Console | None = None,
) -> CoupledDescentPlanResult:
    """Construct an idle-thrust fallback descent profile."""

    solve_started_at = perf_counter()
    profile_nodes = int(num_nodes if num_nodes is not None else request.optimizer.num_nodes)
    if profile_nodes < 4:
        raise ValueError("num_nodes must be at least 4")

    max_tod_m = float(min(request.constraints.s_m[-1], request.reference_path.total_length_m))
    threshold_delta_isa = request.weather.delta_isa_K(0.0, request.threshold.h_m, 0.0)
    threshold_v_tas = float(
        aero.cas2tas(request.threshold.cas_mps, request.threshold.h_m, dT=openap_dT(threshold_delta_isa))
    )
    target_is_midpoint = request.upstream.cas_target_mps is None
    target_cas_mps = request.upstream.cas_target_mps
    if target_cas_mps is None:
        target_cas_mps = 0.5 * (request.upstream.cas_lower_mps + request.upstream.cas_upper_mps)

    def final_cas_error(tod_m: float) -> float:
        profile = _integrate_idle_thrust_fallback_profile(
            request=request,
            tod_m=float(tod_m),
            threshold_v_tas=threshold_v_tas,
            num_nodes=max(profile_nodes, 41),
        )
        return float(profile["v_cas_mps"][-1] - target_cas_mps)

    altitude_span_m = max(request.upstream.h_m - request.threshold.h_m, 1.0)
    lower_tod_m = float(
        np.clip(altitude_span_m / max(np.tan(np.deg2rad(8.0)), 1e-3), 1_000.0, max_tod_m)
    )
    sample_tod = np.unique(np.linspace(lower_tod_m, max_tod_m, 9, dtype=float))
    sample_errors = np.asarray([final_cas_error(float(tod_m)) for tod_m in sample_tod], dtype=float)
    bracket: tuple[float, float] | None = None
    exact_sample_error_mps: float | None = None
    for left_idx, right_idx in zip(range(len(sample_tod) - 1), range(1, len(sample_tod)), strict=True):
        left_error = float(sample_errors[left_idx])
        right_error = float(sample_errors[right_idx])
        if left_error == 0.0:
            bracket = (float(sample_tod[left_idx]), float(sample_tod[left_idx]))
            exact_sample_error_mps = left_error
            break
        if left_error * right_error <= 0.0:
            bracket = (float(sample_tod[left_idx]), float(sample_tod[right_idx]))
            break

    selected_sample_idx = int(np.argmin(np.abs(sample_errors)))
    if bracket is None:
        tod_m = float(sample_tod[selected_sample_idx])
        selected_error_mps = float(sample_errors[selected_sample_idx])
    elif np.isclose(bracket[0], bracket[1], rtol=0.0, atol=1e-9):
        tod_m = bracket[0]
        selected_error_mps = float(exact_sample_error_mps if exact_sample_error_mps is not None else sample_errors[selected_sample_idx])
    else:
        tod_m = cast(
            float,
            brentq(
                final_cas_error,
                bracket[0],
                bracket[1],
                xtol=25.0,
                rtol=np.float64(1e-6),
                maxiter=30,
                full_output=False,
            ),
        )
        selected_error_mps = float(final_cas_error(tod_m))

    if console is not None:
        target_label = "window midpoint" if target_is_midpoint else "fixed upstream target"
        _render_tod_search(
            console,
            sample_tod_m=sample_tod,
            sample_errors_mps=sample_errors,
            target_cas_mps=target_cas_mps,
            target_label=target_label,
            lower_tod_m=lower_tod_m,
            max_tod_m=max_tod_m,
            selected_tod_m=tod_m,
            selected_error_mps=selected_error_mps,
            bracket=bracket,
        )

    profile = _integrate_idle_thrust_fallback_profile(
        request=request,
        tod_m=tod_m,
        threshold_v_tas=threshold_v_tas,
        num_nodes=profile_nodes,
        console=console,
        emit_logs=console is not None,
    )
    s_m = profile["s_m"]
    h_m = profile["h_m"]
    v_tas_mps = profile["v_tas_mps"]
    t_s = profile["t_s"]
    gamma_rad = profile["gamma_rad"]
    thrust_n = profile["thrust_n"]
    v_cas_mps = profile["v_cas_mps"]

    zeros = np.zeros_like(s_m)
    track_rad = request.reference_path.track_angle_rad_many(s_m)
    position_ne = request.reference_path.position_ne_many(s_m)
    latlon_arr = request.reference_path.latlon_from_ne_many(position_ne[:, 0], position_ne[:, 1])
    mode_by_node = tuple(mode_for_s(request.cfg, float(s_val)) for s_val in s_m)
    mode = tuple(mode.name for mode in mode_by_node)
    phi_max_rad = np.asarray(
        [
            bank_limit_rad(request.cfg, mode_cfg, float(cas))
            for mode_cfg, cas in zip(mode_by_node, v_cas_mps, strict=True)
        ],
        dtype=float,
    )
    tangent = request.reference_path.tangent_hat_many(s_m)
    normal = request.reference_path.normal_hat_many(s_m)
    ground_velocity_ne_mps = np.empty((profile_nodes, 2), dtype=float)
    for idx, (s_val, h_val, t_val, v_tas, psi_val) in enumerate(
        zip(s_m, h_m, t_s, v_tas_mps, track_rad, strict=True)
    ):
        wind_east, wind_north = request.weather.wind_ne_mps(float(s_val), float(h_val), float(t_val))
        ground_velocity_ne_mps[idx, 0] = float(v_tas) * float(np.cos(psi_val)) + float(wind_east)
        ground_velocity_ne_mps[idx, 1] = float(v_tas) * float(np.sin(psi_val)) + float(wind_north)
    ground_speed_mps = np.hypot(ground_velocity_ne_mps[:, 0], ground_velocity_ne_mps[:, 1])
    alongtrack_speed_mps = np.einsum("ij,ij->i", ground_velocity_ne_mps, tangent)
    crosstrack_speed_mps = np.einsum("ij,ij->i", ground_velocity_ne_mps, normal)
    ground_track_rad = np.arctan2(ground_velocity_ne_mps[:, 1], ground_velocity_ne_mps[:, 0])
    track_error_rad = np.arctan2(np.sin(ground_track_rad - track_rad), np.cos(ground_track_rad - track_rad))

    z = _pack(
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        t_s=t_s,
        cross_track_m=zeros,
        heading_error_rad=zeros,
        phi_rad=zeros,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        roll_rate_rps=zeros,
        tod_m=tod_m,
        constraint_slack=0.0,
    )
    evaluation_cache = _TrajectoryEvaluationCache(request=request, profiling=_SolverProfilingState())
    collocation_residual_max = float(
        np.max(
            np.abs(
                _equality_constraints(
                    z,
                    request=request,
                    threshold_v_tas=threshold_v_tas,
                    evaluation_cache=evaluation_cache,
                )
            )
        )
    )
    replay = _replay_solution(
        request=request,
        s_m=s_m,
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        t_s=t_s,
        cross_track_m=zeros,
        heading_error_rad=zeros,
        phi_rad=zeros,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        roll_rate_rps=zeros,
    )
    solve_profile = CoupledDescentSolveProfile(
        total_wall_time_s=perf_counter() - solve_started_at,
        postprocess_wall_time_s=0.0,
        objective_calls=0,
        objective_time_s=0.0,
        equality_calls=1,
        equality_time_s=0.0,
        inequality_calls=0,
        inequality_time_s=0.0,
        trajectory_evaluations=evaluation_cache.profiling.trajectory_evaluations,
        trajectory_eval_time_s=evaluation_cache.profiling.trajectory_eval_time_s,
        trajectory_cache_hits=evaluation_cache.profiling.trajectory_cache_hits,
    )
    if console is not None:
        _render_reassembly_summary(
            console,
            collocation_residual_max=collocation_residual_max,
            replay_h_error_m=float(replay["h_error_m"]),
            replay_v_error_mps=float(replay["v_error_mps"]),
            replay_t_error_s=float(replay["t_error_s"]),
            replay_residual_max=float(replay["max_error"]),
            solve_profile=solve_profile,
        )
    return CoupledDescentPlanResult(
        s_m=s_m,
        h_m=h_m,
        v_tas_mps=v_tas_mps,
        v_cas_mps=v_cas_mps,
        t_s=t_s,
        east_m=position_ne[:, 0],
        north_m=position_ne[:, 1],
        lat_deg=latlon_arr[:, 0],
        lon_deg=latlon_arr[:, 1],
        cross_track_m=zeros,
        heading_error_rad=zeros,
        psi_rad=track_rad,
        phi_rad=zeros,
        roll_rate_rps=zeros,
        ground_speed_mps=ground_speed_mps,
        alongtrack_speed_mps=alongtrack_speed_mps,
        crosstrack_speed_mps=crosstrack_speed_mps,
        track_error_rad=track_error_rad,
        phi_max_rad=phi_max_rad,
        gamma_rad=gamma_rad,
        thrust_n=thrust_n,
        mode=mode,
        solver_success=True,
        solver_status=2,
        solver_message="constructed idle-thrust fallback profile",
        objective_value=0.0,
        tod_m=float(tod_m),
        collocation_residual_max=collocation_residual_max,
        replay_h_error_m=float(replay["h_error_m"]),
        replay_v_error_mps=float(replay["v_error_mps"]),
        replay_t_error_s=float(replay["t_error_s"]),
        replay_residual_max=float(replay["max_error"]),
        constraint_slack=0.0,
        solve_profile=solve_profile,
    )


def _cubic_gamma_at_s(
    *,
    s_m: float,
    s_tod_m: float,
    h0_m: float,
    h1_m: float,
    gamma0_rad: float,
    gamma1_rad: float,
) -> float:
    x = float(np.clip(float(s_m) / max(float(s_tod_m), 1.0), 0.0, 1.0))
    slope0 = -float(np.tan(gamma0_rad))
    slope1 = -float(np.tan(gamma1_rad))
    dh00 = (6.0 * x**2 - 6.0 * x) / max(float(s_tod_m), 1.0)
    dh10 = 3.0 * x**2 - 4.0 * x + 1.0
    dh01 = (-6.0 * x**2 + 6.0 * x) / max(float(s_tod_m), 1.0)
    dh11 = 3.0 * x**2 - 2.0 * x
    dhds = dh00 * h0_m + dh10 * slope0 + dh01 * h1_m + dh11 * slope1
    return float(-np.arctan(dhds))


def _integrate_idle_thrust_fallback_profile(
    *,
    request: CoupledDescentPlanRequest,
    tod_m: float,
    threshold_v_tas: float,
    num_nodes: int,
    console: Console | None = None,
    emit_logs: bool = False,
) -> dict[str, np.ndarray]:
    s_m = np.linspace(0.0, float(tod_m), int(num_nodes), dtype=float)

    if emit_logs and console is not None:
        console.rule("[bold cyan]Step 4.2  Build fallback altitude and FPA shape[/bold cyan]")
        console.print(
            Panel.fit(
                f"[bold]Domain[/bold]: s = [0.0, {tod_m:,.1f}] m\n"
                f"[bold]Start[/bold] (runway threshold): h = {request.threshold.h_m:.1f} m, "
                f"gamma = {np.rad2deg(request.threshold.gamma_rad):.3f} deg\n"
                f"[bold]End[/bold] (upstream splice point): h = {request.upstream.h_m:.1f} m, "
                f"gamma = {np.rad2deg(request.upstream.gamma_rad):.3f} deg\n"
                "[dim]This is the exact interval integrated in Step 4.3. The later tactical-start extension is appended outside this solve.[/dim]",
                title="Profile endpoints",
                border_style="cyan",
            )
        )
        console.print(
            f"[bold cyan]Integrating dynamics...[/bold cyan] from [bold]{0.0:.1f} m[/bold] "
            f"(runway threshold) to [bold]{tod_m:,.1f} m[/bold] (upstream splice point)"
        )

    def rhs(s_val: float, state: np.ndarray) -> np.ndarray:
        h_val = float(state[0])
        v_val = max(1.0, float(state[1]))
        t_val = float(state[2])
        gamma_val = _cubic_gamma_at_s(
            s_m=float(s_val),
            s_tod_m=float(tod_m),
            h0_m=request.threshold.h_m,
            h1_m=request.upstream.h_m,
            gamma0_rad=request.threshold.gamma_rad,
            gamma1_rad=request.upstream.gamma_rad,
        )
        delta_isa_K = request.weather.delta_isa_K(float(s_val), h_val, t_val)
        mode = mode_for_s(request.cfg, float(s_val))
        thrust_n = request.perf.idle_thrust_newtons(v_tas_mps=v_val, h_m=h_val, delta_isa_K=delta_isa_K)
        drag_n = request.perf.drag_newtons(
            mode=mode,
            mass_kg=request.cfg.mass_kg,
            wing_area_m2=request.cfg.wing_area_m2,
            v_tas_mps=v_val,
            h_m=h_val,
            gamma_rad=gamma_val,
            bank_rad=0.0,
            delta_isa_K=delta_isa_K,
        )
        cos_gamma = float(np.clip(np.cos(gamma_val), 0.05, None))
        track_rad = request.reference_path.track_angle_rad(float(s_val))
        alongtrack_speed = max(
            request.optimizer.min_alongtrack_speed_mps,
            v_val + alongtrack_wind_mps(request.weather, track_rad, float(s_val), h_val, t_val),
        )
        dhds = -float(np.tan(gamma_val))
        dvds = -(((float(thrust_n) - drag_n) / request.cfg.mass_kg) - aero.g0 * float(np.sin(gamma_val))) / (
            v_val * cos_gamma
        )
        dtds = 1.0 / alongtrack_speed
        return np.asarray([dhds, dvds, dtds], dtype=float)

    solution = solve_ivp(
        rhs,
        (0.0, float(tod_m)),
        y0=np.asarray([request.threshold.h_m, threshold_v_tas, 0.0], dtype=float),
        t_eval=s_m,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    if not solution.success or solution.y.shape[1] != len(s_m):
        raise RuntimeError(f"idle-thrust fallback integration failed: {solution.message}")

    h_m = np.asarray(solution.y[0], dtype=float)
    v_tas_mps = np.asarray(solution.y[1], dtype=float)
    t_s = np.asarray(solution.y[2], dtype=float)
    gamma_rad = np.asarray(
        [
            _cubic_gamma_at_s(
                s_m=float(s_val),
                s_tod_m=float(tod_m),
                h0_m=request.threshold.h_m,
                h1_m=request.upstream.h_m,
                gamma0_rad=request.threshold.gamma_rad,
                gamma1_rad=request.upstream.gamma_rad,
            )
            for s_val in s_m
        ],
        dtype=float,
    )
    v_cas_before_mps = np.array(_cas_from_tas(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps, t_s=t_s), dtype=float, copy=True)
    cas_lower_env_mps, _ = request.constraints.cas_bounds_many(s_m)
    mode_lower_mps, _ = _planned_cas_bounds_many(request.cfg, s_m)
    cas_lower_eff_mps = np.maximum(cas_lower_env_mps, mode_lower_mps)
    v_cas_mps = np.maximum(v_cas_before_mps, cas_lower_eff_mps)
    if emit_logs and console is not None:
        _render_speed_envelope_enforcement(
            console,
            s_m=s_m,
            v_cas_before_mps=v_cas_before_mps,
            cas_lower_env_mps=cas_lower_env_mps,
            mode_lower_mps=mode_lower_mps,
            v_cas_after_mps=v_cas_mps,
        )
    if isinstance(request.weather, ConstantWeather):
        v_tas_mps = np.asarray(
            aero.cas2tas(v_cas_mps, h_m, dT=openap_dT(request.weather.delta_isa_offset_K)),
            dtype=float,
        )
    else:
        v_tas_mps = np.asarray(
            [
                float(
                    aero.cas2tas(
                        float(cas),
                        float(h_val),
                        dT=openap_dT(request.weather.delta_isa_K(float(s_val), float(h_val), float(t_val))),
                    )
                )
                for s_val, h_val, cas, t_val in zip(s_m, h_m, v_cas_mps, t_s, strict=True)
            ],
            dtype=float,
        )
    t_s = _integrate_time_profile(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps)
    modes = tuple(mode_for_s(request.cfg, float(s_val)) for s_val in s_m)
    thrust_guess_n = np.empty_like(s_m)
    thrust_lower_n = np.empty_like(s_m)
    thrust_upper_n = np.empty_like(s_m)
    thrust_n = np.empty_like(s_m)
    v_slope = np.gradient(v_tas_mps, s_m, edge_order=2 if len(s_m) > 2 else 1)
    for idx, (s_val, h_val, v_tas, gamma_val, dvds, t_val, mode) in enumerate(
        zip(s_m, h_m, v_tas_mps, gamma_rad, v_slope, t_s, modes, strict=True)
    ):
        delta_isa_K = request.weather.delta_isa_K(float(s_val), float(h_val), float(t_val))
        drag_n = request.perf.drag_newtons(
            mode=mode,
            mass_kg=request.cfg.mass_kg,
            wing_area_m2=request.cfg.wing_area_m2,
            v_tas_mps=float(v_tas),
            h_m=float(h_val),
            gamma_rad=float(gamma_val),
            bank_rad=0.0,
            delta_isa_K=delta_isa_K,
        )
        thrust_lower, thrust_upper = request.perf.thrust_bounds_newtons(
            mode=mode,
            v_tas_mps=float(v_tas),
            h_m=float(h_val),
            delta_isa_K=delta_isa_K,
        )
        thrust_guess = drag_n + request.cfg.mass_kg * (
            aero.g0 * np.sin(gamma_val) - float(v_tas) * np.cos(gamma_val) * float(dvds)
        )
        thrust_guess_n[idx] = float(thrust_guess)
        thrust_lower_n[idx] = float(thrust_lower)
        thrust_upper_n[idx] = float(thrust_upper)
        thrust_n[idx] = float(np.clip(thrust_guess, thrust_lower, thrust_upper))
    if emit_logs and console is not None:
        _render_thrust_reconstruction(
            console,
            s_m=s_m,
            modes=modes,
            thrust_guess_n=thrust_guess_n,
            thrust_n=thrust_n,
            thrust_lower_n=thrust_lower_n,
            thrust_upper_n=thrust_upper_n,
        )
    return {
        "s_m": s_m,
        "h_m": h_m,
        "v_tas_mps": v_tas_mps,
        "t_s": t_s,
        "gamma_rad": gamma_rad,
        "thrust_n": thrust_n,
        "v_cas_mps": _cas_from_tas(request, s_m=s_m, h_m=h_m, v_tas_mps=v_tas_mps, t_s=t_s),
    }
