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
        mode: ModeConfig,
        mass_kg: float,
        wing_area_m2: float,
        v_tas_mps: float,
        h_m: float,
        vs_mps: float,
        delta_isa_K: float = 0.0,
    ) -> float: ...

    def idle_thrust_newtons(
        self,
        v_tas_mps: float,
        h_m: float,
        delta_isa_K: float = 0.0,
    ) -> float: ...


@dataclass
class EffectivePolarBackend:
    cfg: AircraftConfig
    openap: OpenAPObjects

    def drag_newtons(
        self,
        mode: ModeConfig,
        mass_kg: float,
        wing_area_m2: float,
        v_tas_mps: float,
        h_m: float,
        vs_mps: float,
        delta_isa_K: float = 0.0,
    ) -> float:
        v_tas_mps = max(1.0, float(v_tas_mps))
        _, rho, _ = aero.atmos(h_m, dT=delta_isa_K)
        dynamic_pressure = 0.5 * float(rho) * v_tas_mps**2
        sin_gamma = float(np.clip(vs_mps / v_tas_mps, -0.95, 0.95))
        lift_newtons = mass_kg * aero.g0 * float(np.cos(np.arcsin(sin_gamma)))
        cl = lift_newtons / max(dynamic_pressure * wing_area_m2, 1e-6)
        cd = mode.cd0 + mode.k * cl**2
        return dynamic_pressure * wing_area_m2 * cd

    def idle_thrust_newtons(
        self,
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
