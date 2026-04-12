from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ModeName = Literal["clean", "approach", "final"]


@dataclass(frozen=True)
class ModeConfig:
    name: ModeName
    tau_v_s: float
    vs_min_mps: float
    vs_max_mps: float
    cd0: float
    k: float
    phi_comfort_max_rad: float
    phi_procedure_max_rad: float
    tau_phi_s: float
    p_max_rps: float
    vs_1g_ref_cas_mps: float
    cas_min_mps: float | None = None
    cas_max_mps: float | None = None


@dataclass(frozen=True)
class AircraftConfig:
    typecode: str
    engine_name: str
    mass_kg: float
    reference_mass_kg: float
    wing_area_m2: float
    vmo_kts: float
    mmo: float
    clean: ModeConfig
    approach: ModeConfig
    final: ModeConfig
    k_h_sinv: float = 0.03
    a_acc_max_mps2: float = 0.8
    bank_stall_margin_factor: float = 1.3
    final_gate_m: float = 12_000.0
    approach_gate_m: float = 35_000.0


def mode_for_s(cfg: AircraftConfig, s_m: float) -> ModeConfig:
    if s_m <= cfg.final_gate_m:
        return cfg.final
    if s_m <= cfg.approach_gate_m:
        return cfg.approach
    return cfg.clean


def bank_limit_stall_rad(cfg: AircraftConfig, mode: ModeConfig, v_cas_mps: float) -> float:
    if v_cas_mps <= 0.0:
        return 0.0

    stall_cas_mps = mode.vs_1g_ref_cas_mps * np.sqrt(cfg.mass_kg / cfg.reference_mass_kg)
    margin_ratio = cfg.bank_stall_margin_factor * stall_cas_mps / v_cas_mps
    cos_phi = float(np.clip(margin_ratio**2, 0.0, 1.0))
    return float(np.arccos(cos_phi))


def bank_limit_rad(cfg: AircraftConfig, mode: ModeConfig, v_cas_mps: float) -> float:
    return float(
        min(
            mode.phi_comfort_max_rad,
            mode.phi_procedure_max_rad,
            bank_limit_stall_rad(cfg, mode, v_cas_mps),
        )
    )
