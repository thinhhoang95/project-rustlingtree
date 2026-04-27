from __future__ import annotations

from pathlib import Path

from simap import (
    LateralGuidanceConfig,
    OptimizerConfig,
    SimulationRequest,
    plan_coupled_descent,
    plan_smooth_idle_descent,
    simulate_plan,
)

from .builder import build_tactical_plan_request
from .models import TacticalCommand, TacticalPlanBundle
from .plan_extension import extend_plan_to_tactical_start


def _plan_is_usable(plan, *, residual_tolerance: float = 1e-2, replay_tolerance: float = 25.0) -> bool:
    return (
        bool(plan.solver_success)
        and float(plan.constraint_slack) <= residual_tolerance
        and float(plan.collocation_residual_max) <= replay_tolerance
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
) -> TacticalPlanBundle:
    bundle = build_tactical_plan_request(
        command,
        fixes_csv=fixes_csv,
        aircraft_type=aircraft_type,
        payload_kg=payload_kg,
        optimizer=optimizer,
    )
    request = bundle.request
    if prefer_smooth_idle:
        raw_plan = plan_smooth_idle_descent(request)
    else:
        raw_plan = plan_coupled_descent(request)
        if not _plan_is_usable(raw_plan):
            raw_plan = plan_smooth_idle_descent(request)
    plan = extend_plan_to_tactical_start(
        request=request,
        plan=raw_plan,
        start_condition=command.upstream,
        start_s_m=request.reference_path.total_length_m,
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
