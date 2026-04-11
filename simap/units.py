from __future__ import annotations

from openap import aero


def m_to_ft(x_m: float) -> float:
    return x_m / aero.ft


def ft_to_m(x_ft: float) -> float:
    return x_ft * aero.ft


def mps_to_kts(x_mps: float) -> float:
    return x_mps / aero.kts


def kts_to_mps(x_kts: float) -> float:
    return x_kts * aero.kts


def mps_to_fpm(x_mps: float) -> float:
    return x_mps / aero.fpm


def fpm_to_mps(x_fpm: float) -> float:
    return x_fpm * aero.fpm


def km_to_m(x_km: float) -> float:
    return x_km * 1_000.0
