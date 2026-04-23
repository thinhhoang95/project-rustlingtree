from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from openap import aero

from .config import AircraftConfig, ModeConfig
from .openap_adapter import OpenAPObjects
from .units import m_to_ft, mps_to_kts


class PerformanceBackend(Protocol):
    def drag_newtons(
        self,
        *,
        mode: ModeConfig,
        mass_kg: float,
        wing_area_m2: float,
        v_tas_mps: float,
        h_m: float,
        gamma_rad: float = 0.0,
        bank_rad: float = 0.0,
        delta_isa_K: float = 0.0,
    ) -> float: ...

    def idle_thrust_newtons(
        self,
        *,
        v_tas_mps: float,
        h_m: float,
        delta_isa_K: float = 0.0,
    ) -> float: ...

    def thrust_bounds_newtons(
        self,
        *,
        mode: ModeConfig,
        v_tas_mps: float,
        h_m: float,
        delta_isa_K: float = 0.0,
    ) -> tuple[float, float]: ...


@dataclass
class EffectivePolarBackend:
    cfg: AircraftConfig
    openap: OpenAPObjects

    def drag_newtons(
        self,
        *,
        mode: ModeConfig,
        mass_kg: float,
        wing_area_m2: float,
        v_tas_mps: float,
        h_m: float,
        gamma_rad: float = 0.0,
        bank_rad: float = 0.0,
        delta_isa_K: float = 0.0,
    ) -> float:
        v_tas_mps = max(1.0, float(v_tas_mps))
        _, rho, _ = aero.atmos(h_m, dT=delta_isa_K)
        dynamic_pressure = 0.5 * float(rho) * v_tas_mps**2
        cos_bank = float(np.clip(np.cos(bank_rad), 0.1, 1.0))
        lift_newtons = mass_kg * aero.g0 * float(np.cos(gamma_rad)) / cos_bank
        cl = lift_newtons / max(dynamic_pressure * wing_area_m2, 1e-6)
        cd = mode.cd0 + mode.k * cl**2
        return float(dynamic_pressure * wing_area_m2 * cd)

    def idle_thrust_newtons(
        self,
        *,
        v_tas_mps: float,
        h_m: float,
        delta_isa_K: float = 0.0,
    ) -> float:
        del delta_isa_K
        return float(
            self.openap.thrust.descent_idle(
                tas=mps_to_kts(v_tas_mps),
                alt=m_to_ft(h_m),
            )
        )

    def thrust_bounds_newtons(
        self,
        *,
        mode: ModeConfig,
        v_tas_mps: float,
        h_m: float,
        delta_isa_K: float = 0.0,
    ) -> tuple[float, float]:
        idle = self.idle_thrust_newtons(v_tas_mps=v_tas_mps, h_m=h_m, delta_isa_K=delta_isa_K)
        max_thrust = float(self.openap.engine.get("max_thrust", np.nan))
        if not np.isfinite(max_thrust) or max_thrust <= 0.0:
            return float(idle), float(max(idle + 1.0, 2.0 * idle))

        _, rho, _ = aero.atmos(h_m, dT=delta_isa_K)
        rho_ratio = float(np.clip(rho / aero.rho0, 0.05, 1.25))
        tas_kts = mps_to_kts(v_tas_mps)
        speed_relief = max(0.35, 1.0 - 0.0008 * max(tas_kts - 120.0, 0.0))
        phase_scale = {
            "clean": 1.0,
            "approach": 0.92,
            "final": 0.85,
        }[mode.name]
        upper = max_thrust * rho_ratio**0.75 * speed_relief * phase_scale
        return float(idle), float(max(upper, idle + 1.0))
