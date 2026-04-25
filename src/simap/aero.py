from __future__ import annotations

import math

try:  # pragma: no cover - exercised when OpenAP is installed
    from openap.aero import *  # type: ignore[F403]
except ImportError:  # pragma: no cover - fallback used in environments without OpenAP
    g0 = 9.80665
    rho0 = 1.225
    ft = 0.3048
    kts = 0.514444
    fpm = 0.00508

    _T0 = 288.15
    _P0 = 101325.0
    _LAPSE = 0.0065
    _R = 287.05287

    def atmos(h: float, dT: float = 0.0) -> tuple[float, float, float]:
        altitude_m = max(0.0, float(h))
        temp_k = _T0 - _LAPSE * altitude_m + float(dT)
        pressure_pa = _P0 * (max(temp_k - float(dT), 1.0) / _T0) ** (g0 / (_R * _LAPSE))
        density = pressure_pa / (_R * max(temp_k, 1.0))
        return pressure_pa, density, temp_k

    def cas2tas(cas: float, h: float, dT: float = 0.0) -> float:
        _, rho, _ = atmos(h, dT=dT)
        return float(cas) * math.sqrt(rho0 / max(rho, 1e-9))

    def tas2cas(tas: float, h: float, dT: float = 0.0) -> float:
        _, rho, _ = atmos(h, dT=dT)
        return float(tas) * math.sqrt(max(rho, 1e-9) / rho0)
