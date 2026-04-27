from __future__ import annotations

from pathlib import Path

from rich.console import Console

from simap import (
    LateralGuidanceConfig,
    OptimizerConfig,
    SimulationRequest,
    plan_full_route_longitudinal_descent,
    plan_smooth_idle_descent,
    simulate_plan,
)

from .diagnostics import render_tactical_setup
from .builder import build_tactical_plan_request
from .models import TacticalCommand, TacticalPlanBundle


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
    prefer_smooth_idle: bool = False,
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
    if prefer_smooth_idle:
        if console is not None:
            console.print("[cyan]Prefer smooth idle requested; building smooth idle profile directly.[/cyan]")
        raw_plan = plan_smooth_idle_descent(request, console=console)
    else:
        if console is not None:
            console.print("[cyan]Solving full-route longitudinal FMS profile.[/cyan]")
        raw_plan = plan_full_route_longitudinal_descent(request)
        if not _plan_is_usable(raw_plan):
            if console is not None:
                console.print("[yellow]Full-route profile failed usability checks; continuing with NLP result.[/yellow]")
        elif console is not None:
            console.print(
                "[green]Full-route profile accepted.[/green]\n"
                f"[dim]slack={float(raw_plan.constraint_slack):.3e}, "
                f"collocation={float(raw_plan.collocation_residual_max):.3e}, "
                f"replay={float(raw_plan.replay_residual_max):.3e}[/dim]"
            )
    plan = raw_plan
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
