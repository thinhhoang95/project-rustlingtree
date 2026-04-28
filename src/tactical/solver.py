from __future__ import annotations

from pathlib import Path

from rich.console import Console

from simap import (
    LateralGuidanceConfig,
    OptimizerConfig,
    SimulationRequest,
    plan_coupled_descent,
    plan_full_route_longitudinal_descent,
    plan_idle_thrust_fallback,
    simulate_plan,
)

from .diagnostics import render_tactical_setup
from .builder import build_tactical_plan_request
from .models import TacticalCommand, TacticalPlanBundle
from .plan_extension import extend_plan_to_tactical_start


def _plan_is_usable(plan, *, residual_tolerance: float = 2e-2, replay_tolerance: float = 500.0) -> bool:
    return (
        bool(plan.solver_success)
        and float(plan.constraint_slack) <= residual_tolerance
        and float(plan.collocation_residual_max) <= residual_tolerance
        and float(plan.replay_residual_max) <= replay_tolerance
    )


def solve_tactical_command(
    command: TacticalCommand,
    *,
    fixes_csv: str | Path,
    aircraft_type: str = "A320",
    payload_kg: float = 12_000.0,
    optimizer: OptimizerConfig | None = None,
    simulate: bool = True,
    guidance: LateralGuidanceConfig | None = None,
    dt_s: float = 0.5,
    prefer_idle_thrust_fallback: bool = False,
    console: Console | None = None,
) -> TacticalPlanBundle:
    bundle = build_tactical_plan_request(
        command,
        fixes_csv=fixes_csv,
        aircraft_type=aircraft_type,
        payload_kg=payload_kg,
        optimizer=optimizer,
    )
    request = bundle.request
    if console is not None:
        render_tactical_setup(console, bundle=bundle)
    if prefer_idle_thrust_fallback:
        if console is not None:
            console.print("[cyan]Prefer idle-thrust fallback requested; building fallback profile directly.[/cyan]")
        raw_plan = plan_idle_thrust_fallback(request, console=console)
    elif request.optimizer.idle_thrust_margin_fraction is not None:
        if console is not None:
            console.print("[cyan]Solving free-TOD idle-thrust fallback profile.[/cyan]")
        raw_plan = plan_coupled_descent(request)
    else:
        if console is not None:
            console.print("[cyan]Solving full-route longitudinal FMS profile.[/cyan]")
        raw_plan = plan_full_route_longitudinal_descent(request)
        if _plan_is_usable(raw_plan) and console is not None:
            console.print(
                "[green]Full-route profile accepted.[/green]\n"
                f"[dim]slack={float(raw_plan.constraint_slack):.3e}, "
                f"collocation={float(raw_plan.collocation_residual_max):.3e}, "
                f"replay={float(raw_plan.replay_residual_max):.3e}[/dim]"
            )
    if not _plan_is_usable(raw_plan):
        if console is not None:
            console.print(
                "[yellow]Raw tactical profile failed usability checks; returning diagnostics without extension or simulation.[/yellow]\n"
                f"[dim]solver_success={raw_plan.solver_success}, "
                f"slack={float(raw_plan.constraint_slack):.3e}, "
                f"collocation={float(raw_plan.collocation_residual_max):.3e}, "
                f"replay={float(raw_plan.replay_residual_max):.3e}[/dim]"
            )
        return TacticalPlanBundle(
            command=bundle.command,
            path=bundle.path,
            request=request,
            raw_plan=raw_plan,
            plan=None,
            simulation=None,
        )
    plan = raw_plan
    if request.optimizer.idle_thrust_margin_fraction is not None:
        plan = extend_plan_to_tactical_start(
            request=request,
            plan=raw_plan,
            start_condition=command.upstream,
            start_s_m=float(request.reference_path.total_length_m),
            console=console,
        )
    simulation = None
    if simulate:
        simulation = simulate_plan(
            SimulationRequest(
                cfg=request.cfg,
                perf=request.perf,
                plan=plan,
                reference_path=request.reference_path,
                weather=request.weather,
                guidance=guidance if guidance is not None else LateralGuidanceConfig(),
                dt_s=dt_s,
                threshold_tolerance_m=0.0,
            )
        )
    return TacticalPlanBundle(
        command=bundle.command,
        path=bundle.path,
        request=request,
        raw_plan=raw_plan,
        plan=plan,
        simulation=simulation,
    )
