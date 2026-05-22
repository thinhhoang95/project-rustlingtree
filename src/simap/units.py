from __future__ import annotations

from typing import overload

import numpy as np
from openap import aero


@overload
def m_to_ft(x_m: float) -> float: ...


@overload
def m_to_ft(x_m: np.ndarray) -> np.ndarray: ...


def m_to_ft(x_m: float | np.ndarray) -> float | np.ndarray:
    return x_m / aero.ft


@overload
def ft_to_m(x_ft: float) -> float: ...


@overload
def ft_to_m(x_ft: np.ndarray) -> np.ndarray: ...


def ft_to_m(x_ft: float | np.ndarray) -> float | np.ndarray:
    return x_ft * aero.ft


@overload
def mps_to_kts(x_mps: float) -> float: ...


@overload
def mps_to_kts(x_mps: np.ndarray) -> np.ndarray: ...


def mps_to_kts(x_mps: float | np.ndarray) -> float | np.ndarray:
    return x_mps / aero.kts


@overload
def kts_to_mps(x_kts: float) -> float: ...


@overload
def kts_to_mps(x_kts: np.ndarray) -> np.ndarray: ...


def kts_to_mps(x_kts: float | np.ndarray) -> float | np.ndarray:
    return x_kts * aero.kts


@overload
def mps_to_fpm(x_mps: float) -> float: ...


@overload
def mps_to_fpm(x_mps: np.ndarray) -> np.ndarray: ...


def mps_to_fpm(x_mps: float | np.ndarray) -> float | np.ndarray:
    return x_mps / aero.fpm


@overload
def fpm_to_mps(x_fpm: float) -> float: ...


@overload
def fpm_to_mps(x_fpm: np.ndarray) -> np.ndarray: ...


def fpm_to_mps(x_fpm: float | np.ndarray) -> float | np.ndarray:
    return x_fpm * aero.fpm


def km_to_m(x_km: float) -> float:
    return x_km * 1_000.0
