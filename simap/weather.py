from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class WeatherProvider(Protocol):
    def wind_ne_mps(self, s_m: float, h_m: float, t_s: float) -> tuple[float, float]: ...

    def delta_isa_K(self, s_m: float, h_m: float, t_s: float) -> float: ...


def alongtrack_wind_mps(
    weather: WeatherProvider,
    track_angle_rad: float,
    s_m: float,
    h_m: float,
    t_s: float,
) -> float:
    wind_east_mps, wind_north_mps = weather.wind_ne_mps(s_m, h_m, t_s)
    return float(
        wind_east_mps * np.cos(track_angle_rad) + wind_north_mps * np.sin(track_angle_rad)
    )


@dataclass(frozen=True)
class ConstantWeather:
    wind_east_mps: float = 0.0
    wind_north_mps: float = 0.0
    delta_isa_offset_K: float = 0.0

    def wind_ne_mps(self, s_m: float, h_m: float, t_s: float) -> tuple[float, float]:
        del s_m, h_m, t_s
        return self.wind_east_mps, self.wind_north_mps

    def delta_isa_K(self, s_m: float, h_m: float, t_s: float) -> float:
        del s_m, h_m, t_s
        return self.delta_isa_offset_K
