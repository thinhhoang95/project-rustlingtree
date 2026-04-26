from __future__ import annotations

from pathlib import Path

from simap import LateralGuidanceConfig, OptimizerConfig, SimulationRequest, plan_coupled_descent, simulate_plan

from .builder import build_tactical_plan_request
from .models import TacticalCommand, TacticalPlanBundle
from .plan_extension import extend_plan_to_tactical_start


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
) -> TacticalPlanBundle:
    bundle = build_tactical_plan_request(
        command,
        fixes_csv=fixes_csv,
        aircraft_type=aircraft_type,
        payload_kg=payload_kg,
        optimizer=optimizer,
    )
    request = bundle.request
    raw_plan = plan_coupled_descent(request)
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
        plan=plan,
        simulation=simulation,
    )
