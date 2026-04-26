from __future__ import annotations

from pathlib import Path

import numpy as np

from simap import (
    ConstantWeather,
    CoupledDescentPlanRequest,
    EffectivePolarBackend,
    OptimizerConfig,
    ThresholdBoundary,
    UpstreamBoundary,
    build_default_aircraft_config,
    extract_aircraft_data,
    load_openap,
    suggest_approach_mass_kg,
)
from simap.units import ft_to_m, kts_to_mps

from .constraints import build_tactical_constraint_envelope
from .models import TacticalCommand, TacticalPlanBundle
from .navdata import load_fix_catalog
from .path import build_reference_path, resolve_lateral_path, waypoint_s_by_identifier


def build_tactical_plan_request(
    command: TacticalCommand,
    *,
    fixes_csv: str | Path,
    aircraft_type: str = "A320",
    payload_kg: float = 12_000.0,
    optimizer: OptimizerConfig | None = None,
) -> TacticalPlanBundle:
    fix_catalog = load_fix_catalog(fixes_csv)
    resolved_path = resolve_lateral_path(command.lateral_path, fix_catalog)
    if resolved_path.waypoints[0].identifier != command.upstream.fix_identifier.upper():
        raise ValueError("upstream condition must be attached to the first lateral-path waypoint")

    reference_path = build_reference_path(resolved_path)
    waypoint_s_m = waypoint_s_by_identifier(resolved_path, reference_path)

    openap = load_openap(aircraft_type)
    aircraft_data = extract_aircraft_data(openap)
    mass_kg = suggest_approach_mass_kg(aircraft_data, payload_kg=payload_kg)
    cfg, openap = build_default_aircraft_config(aircraft_type, mass_kg=mass_kg, openap_objects=openap)
    perf = EffectivePolarBackend(cfg=cfg, openap=openap)

    runway = resolved_path.waypoints[-1]
    threshold_altitude_ft = command.runway_altitude_ft
    if threshold_altitude_ft is None:
        threshold_altitude_ft = runway.elevation_ft
    if threshold_altitude_ft is None:
        raise ValueError("runway altitude is required when the runway fix has no elevation")

    threshold = ThresholdBoundary(
        h_m=ft_to_m(threshold_altitude_ft),
        cas_mps=float(openap.wrap.landing_speed()["default"]),
        gamma_rad=np.deg2rad(command.runway_gamma_deg),
    )
    upstream = UpstreamBoundary(
        h_m=ft_to_m(command.upstream.altitude_ft),
        cas_window_mps=(kts_to_mps(command.upstream.cas_kts), kts_to_mps(command.upstream.cas_kts)),
        gamma_rad=np.deg2rad(command.upstream.gamma_deg),
    )
    envelope = build_tactical_constraint_envelope(
        total_path_length_m=reference_path.total_length_m,
        threshold_altitude_m=threshold.h_m,
        upstream_altitude_m=upstream.h_m,
        threshold_cas_mps=threshold.cas_mps,
        upstream_cas_kts=command.upstream.cas_kts,
        openap_wrap=openap.wrap,
        altitude_constraints=command.altitude_constraints,
        constraint_s_m=waypoint_s_m,
    )

    request = CoupledDescentPlanRequest(
        cfg=cfg,
        perf=perf,
        threshold=threshold,
        upstream=upstream,
        constraints=envelope,
        reference_path=reference_path,
        weather=ConstantWeather(),
        optimizer=optimizer if optimizer is not None else OptimizerConfig(num_nodes=41, maxiter=400),
    )
    return TacticalPlanBundle(command=command, path=resolved_path, request=request)
