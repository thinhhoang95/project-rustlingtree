from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModeName = Literal["clean", "approach", "final"]


@dataclass(frozen=True)
class ModeConfig:
    name: ModeName
    tau_v_s: float
    vs_min_mps: float
    vs_max_mps: float
    cd0: float
    k: float
    cas_min_mps: float | None = None
    cas_max_mps: float | None = None


@dataclass(frozen=True)
class AircraftConfig:
    typecode: str
    engine_name: str
    mass_kg: float
    wing_area_m2: float
    vmo_kts: float
    mmo: float
    clean: ModeConfig
    approach: ModeConfig
    final: ModeConfig
    k_h_sinv: float = 0.03
    a_acc_max_mps2: float = 0.8
    final_gate_m: float = 12_000.0
    approach_gate_m: float = 35_000.0


def mode_for_x(cfg: AircraftConfig, x_m: float) -> ModeConfig:
    if x_m <= cfg.final_gate_m:
        return cfg.final
    if x_m <= cfg.approach_gate_m:
        return cfg.approach
    return cfg.clean
