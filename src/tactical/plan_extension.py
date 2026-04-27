from __future__ import annotations

from dataclasses import replace

import numpy as np
from openap import aero

from simap import CoupledDescentPlanResult, CoupledDescentPlanRequest, mode_for_s
from simap.openap_adapter import openap_dT
from simap.units import kts_to_mps, mps_to_kts

from .models import TacticalCondition


def extend_plan_to_tactical_start(
    *,
    request: CoupledDescentPlanRequest,
    plan: CoupledDescentPlanResult,
    start_condition: TacticalCondition,
    start_s_m: float,
    num_extension_nodes: int = 12,
    max_cas_jump_kts: float = 5.0,
) -> CoupledDescentPlanResult:
    if start_s_m <= float(plan.s_m[-1]) + 1e-6:
        return plan

    start_cas_mps = kts_to_mps(start_condition.cas_kts)
    cas_jump_kts = abs(mps_to_kts(float(plan.v_cas_mps[-1]) - start_cas_mps))
    if cas_jump_kts > max_cas_jump_kts:
        raise ValueError(
            "cannot extend tactical plan with a discontinuous CAS splice: "
            f"raw TOD CAS is {mps_to_kts(float(plan.v_cas_mps[-1])):.1f} kt, "
            f"tactical start CAS is {start_condition.cas_kts:.1f} kt "
            f"(jump {cas_jump_kts:.1f} kt > {max_cas_jump_kts:.1f} kt)"
        )

    extension_s = np.linspace(float(plan.s_m[-1]), float(start_s_m), num_extension_nodes + 1, dtype=float)[1:]
    h_extension = np.full_like(extension_s, float(request.upstream.h_m))
    cas_extension = np.full_like(extension_s, start_cas_mps)
    v_tas_extension = np.asarray(
        [
            aero.cas2tas(
                float(cas_mps),
                float(h_m),
                dT=openap_dT(request.weather.delta_isa_K(float(s_m), float(h_m), 0.0)),
            )
            for s_m, h_m, cas_mps in zip(extension_s, h_extension, cas_extension, strict=True)
        ],
        dtype=float,
    )
    ds_extension = np.diff(np.concatenate(([float(plan.s_m[-1])], extension_s)))
    dt_extension = ds_extension / np.maximum(v_tas_extension, 1.0)
    t_extension = float(plan.t_s[-1]) + np.cumsum(dt_extension)
    position = np.asarray([request.reference_path.position_ne(float(s_m)) for s_m in extension_s], dtype=float)
    latlon = np.asarray([request.reference_path.latlon(float(s_m)) for s_m in extension_s], dtype=float)
    psi_extension = np.asarray([request.reference_path.track_angle_rad(float(s_m)) for s_m in extension_s], dtype=float)
    zeros = np.zeros_like(extension_s)
    phi_max = np.full_like(extension_s, float(plan.phi_max_rad[-1]))
    modes = tuple(mode_for_s(request.cfg, float(s_m)) for s_m in extension_s)
    thrust_extension = np.asarray(
        [
            request.perf.drag_newtons(
                mode=mode,
                mass_kg=request.cfg.mass_kg,
                wing_area_m2=request.cfg.wing_area_m2,
                v_tas_mps=float(v_tas),
                h_m=float(h_m),
                gamma_rad=0.0,
                bank_rad=0.0,
                delta_isa_K=request.weather.delta_isa_K(float(s_m), float(h_m), float(t_s)),
            )
            for mode, s_m, h_m, v_tas, t_s in zip(
                modes,
                extension_s,
                h_extension,
                v_tas_extension,
                t_extension,
                strict=True,
            )
        ],
        dtype=float,
    )
    mode = tuple(mode.name for mode in modes)

    return replace(
        plan,
        s_m=np.concatenate([plan.s_m, extension_s]),
        h_m=np.concatenate([plan.h_m, h_extension]),
        v_tas_mps=np.concatenate([plan.v_tas_mps, v_tas_extension]),
        v_cas_mps=np.concatenate([plan.v_cas_mps, cas_extension]),
        t_s=np.concatenate([plan.t_s, t_extension]),
        east_m=np.concatenate([plan.east_m, position[:, 0]]),
        north_m=np.concatenate([plan.north_m, position[:, 1]]),
        lat_deg=np.concatenate([plan.lat_deg, latlon[:, 0]]),
        lon_deg=np.concatenate([plan.lon_deg, latlon[:, 1]]),
        cross_track_m=np.concatenate([plan.cross_track_m, zeros]),
        heading_error_rad=np.concatenate([plan.heading_error_rad, zeros]),
        psi_rad=np.concatenate([plan.psi_rad, psi_extension]),
        phi_rad=np.concatenate([plan.phi_rad, zeros]),
        roll_rate_rps=np.concatenate([plan.roll_rate_rps, zeros]),
        ground_speed_mps=np.concatenate([plan.ground_speed_mps, v_tas_extension]),
        alongtrack_speed_mps=np.concatenate([plan.alongtrack_speed_mps, v_tas_extension]),
        crosstrack_speed_mps=np.concatenate([plan.crosstrack_speed_mps, zeros]),
        track_error_rad=np.concatenate([plan.track_error_rad, zeros]),
        phi_max_rad=np.concatenate([plan.phi_max_rad, phi_max]),
        gamma_rad=np.concatenate([plan.gamma_rad, zeros]),
        thrust_n=np.concatenate([plan.thrust_n, thrust_extension]),
        mode=tuple(plan.mode) + mode,
    )
