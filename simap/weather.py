from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class WeatherProvider(Protocol):
    def alongtrack_mps(self, x_m: float, h_m: float, t_s: float) -> float: ...

    def delta_isa_K(self, x_m: float, h_m: float, t_s: float) -> float: ...


@dataclass(frozen=True)
class ConstantWeather:
    alongtrack_wind_mps: float = 0.0
    delta_isa_offset_K: float = 0.0

    def alongtrack_mps(self, x_m: float, h_m: float, t_s: float) -> float:
        return self.alongtrack_wind_mps

    def delta_isa_K(self, x_m: float, h_m: float, t_s: float) -> float:
        return self.delta_isa_offset_K
