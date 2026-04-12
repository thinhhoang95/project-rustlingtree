from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openap import aero

from .config import AircraftConfig, ModeConfig
from .openap_adapter import (
    OpenAPAircraftData,
    OpenAPObjects,
    extract_aircraft_data,
    load_openap,
    wrap_default,
)
from .units import fpm_to_mps, ft_to_m, mps_to_kts


def _default_vs_1g_ref_cas_mps(*, target_cas_mps: float, factor: float) -> float:
    return max(1.0, target_cas_mps / factor)


@dataclass(frozen=True)
class PolarFitResult:
    cd0: float
    k: float


def suggest_approach_mass_kg(aircraft_data: OpenAPAircraftData, payload_kg: float) -> float:
    return min(aircraft_data.mlw_kg, aircraft_data.oew_kg + payload_kg)


def _fit_effective_polar(
    drag_samples_newtons: np.ndarray,
    tas_mps: np.ndarray,
    alt_m: np.ndarray,
    vs_mps: np.ndarray,
    mass_kg: float,
    wing_area_m2: float,
) -> PolarFitResult:
    rows: list[list[float]] = []
    targets: list[float] = []

    for drag_newtons, v_tas_mps, h_m, v_speed_mps in zip(
        drag_samples_newtons,
        tas_mps,
        alt_m,
        vs_mps,
        strict=True,
    ):
        _, rho, _ = aero.atmos(float(h_m), dT=0.0)
        dynamic_pressure = 0.5 * float(rho) * float(v_tas_mps) ** 2
        sin_gamma = float(np.clip(float(v_speed_mps) / max(float(v_tas_mps), 1.0), -0.95, 0.95))
        lift_newtons = mass_kg * aero.g0 * float(np.cos(np.arcsin(sin_gamma)))
        cl = lift_newtons / max(dynamic_pressure * wing_area_m2, 1e-6)
        rows.append([1.0, cl**2])
        targets.append(float(drag_newtons) / max(dynamic_pressure * wing_area_m2, 1e-6))

    design = np.asarray(rows, dtype=float)
    observations = np.asarray(targets, dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(design, observations, rcond=None)
    return PolarFitResult(cd0=float(coeffs[0]), k=float(coeffs[1]))


def fit_mode_polar_from_openap(
    openap: OpenAPObjects,
    aircraft_data: OpenAPAircraftData,
    mass_kg: float,
    *,
    flap_angle_deg: float,
    landing_gear: bool,
    tas_kts_grid: np.ndarray,
    alt_ft_grid: np.ndarray,
    vs_fpm: float,
) -> PolarFitResult:
    samples_drag: list[float] = []
    samples_tas_mps: list[float] = []
    samples_alt_m: list[float] = []
    samples_vs_mps: list[float] = []

    for tas_kts in tas_kts_grid:
        for alt_ft in alt_ft_grid:
            drag_newtons = float(
                openap.drag.nonclean(
                    mass=mass_kg,
                    tas=float(tas_kts),
                    alt=float(alt_ft),
                    flap_angle=float(flap_angle_deg),
                    vs=float(vs_fpm),
                    landing_gear=landing_gear,
                )
            )
            samples_drag.append(drag_newtons)
            samples_tas_mps.append(float(tas_kts) * aero.kts)
            samples_alt_m.append(ft_to_m(float(alt_ft)))
            samples_vs_mps.append(fpm_to_mps(float(vs_fpm)))

    return _fit_effective_polar(
        drag_samples_newtons=np.asarray(samples_drag, dtype=float),
        tas_mps=np.asarray(samples_tas_mps, dtype=float),
        alt_m=np.asarray(samples_alt_m, dtype=float),
        vs_mps=np.asarray(samples_vs_mps, dtype=float),
        mass_kg=mass_kg,
        wing_area_m2=aircraft_data.wing_area_m2,
    )


def build_default_aircraft_config(
    typecode: str,
    mass_kg: float,
    *,
    engine_name: str | None = None,
    openap_objects: OpenAPObjects | None = None,
    approach_gate_m: float = 35_000.0,
    final_gate_m: float = 12_000.0,
    k_h_sinv: float = 0.03,
    a_acc_max_mps2: float = 0.8,
) -> tuple[AircraftConfig, OpenAPObjects]:
    openap = openap_objects or load_openap(typecode, engine_name=engine_name)
    aircraft_data = extract_aircraft_data(openap)
    wrap = openap.wrap

    final_vcas_kts = mps_to_kts(wrap_default(wrap, "finalapp_vcas"))
    landing_vcas_kts = mps_to_kts(wrap_default(wrap, "landing_speed"))

    approach_fit = fit_mode_polar_from_openap(
        openap,
        aircraft_data,
        mass_kg,
        flap_angle_deg=15.0,
        landing_gear=False,
        tas_kts_grid=np.linspace(max(140.0, final_vcas_kts + 10.0), 190.0, 6),
        alt_ft_grid=np.linspace(1_500.0, 6_000.0, 6),
        vs_fpm=-700.0,
    )
    final_fit = fit_mode_polar_from_openap(
        openap,
        aircraft_data,
        mass_kg,
        flap_angle_deg=30.0,
        landing_gear=True,
        tas_kts_grid=np.linspace(max(120.0, landing_vcas_kts), max(145.0, final_vcas_kts + 5.0), 6),
        alt_ft_grid=np.linspace(0.0, 3_000.0, 6),
        vs_fpm=-650.0,
    )

    clean = ModeConfig(
        name="clean",
        tau_v_s=18.0,
        vs_min_mps=float(openap.wrap.descent_vs_concas()["minimum"]),
        vs_max_mps=3.0,
        cd0=aircraft_data.clean_cd0,
        k=aircraft_data.clean_k,
        phi_comfort_max_rad=np.deg2rad(25.0),
        phi_procedure_max_rad=np.deg2rad(25.0),
        tau_phi_s=2.0,
        p_max_rps=np.deg2rad(6.0),
        vs_1g_ref_cas_mps=_default_vs_1g_ref_cas_mps(
            target_cas_mps=wrap_default(wrap, "descent_const_vcas"),
            factor=1.8,
        ),
        cas_max_mps=aircraft_data.vmo_kts * aero.kts,
    )
    approach = ModeConfig(
        name="approach",
        tau_v_s=22.0,
        vs_min_mps=float(openap.wrap.descent_vs_post_concas()["minimum"]),
        vs_max_mps=2.0,
        cd0=approach_fit.cd0,
        k=approach_fit.k,
        phi_comfort_max_rad=np.deg2rad(22.0),
        phi_procedure_max_rad=np.deg2rad(20.0),
        tau_phi_s=2.5,
        p_max_rps=np.deg2rad(5.0),
        vs_1g_ref_cas_mps=_default_vs_1g_ref_cas_mps(
            target_cas_mps=wrap_default(wrap, "finalapp_vcas"),
            factor=1.3,
        ),
        cas_min_mps=wrap_default(wrap, "finalapp_vcas"),
        cas_max_mps=wrap_default(wrap, "descent_const_vcas"),
    )
    final = ModeConfig(
        name="final",
        tau_v_s=25.0,
        vs_min_mps=float(openap.wrap.finalapp_vs()["minimum"]),
        vs_max_mps=1.0,
        cd0=final_fit.cd0,
        k=final_fit.k,
        phi_comfort_max_rad=np.deg2rad(18.0),
        phi_procedure_max_rad=np.deg2rad(15.0),
        tau_phi_s=3.0,
        p_max_rps=np.deg2rad(4.0),
        vs_1g_ref_cas_mps=_default_vs_1g_ref_cas_mps(
            target_cas_mps=wrap_default(wrap, "landing_speed"),
            factor=1.23,
        ),
        cas_min_mps=wrap_default(wrap, "landing_speed"),
        cas_max_mps=wrap_default(wrap, "finalapp_vcas"),
    )

    cfg = AircraftConfig(
        typecode=typecode,
        engine_name=aircraft_data.engine_name,
        mass_kg=mass_kg,
        reference_mass_kg=aircraft_data.mlw_kg,
        wing_area_m2=aircraft_data.wing_area_m2,
        vmo_kts=aircraft_data.vmo_kts,
        mmo=aircraft_data.mmo,
        clean=clean,
        approach=approach,
        final=final,
        k_h_sinv=k_h_sinv,
        a_acc_max_mps2=a_acc_max_mps2,
        final_gate_m=final_gate_m,
        approach_gate_m=approach_gate_m,
    )
    return cfg, openap
