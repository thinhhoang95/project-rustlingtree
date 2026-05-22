from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdvisoryFlight:
    flight_number: str
    icao24: str
    flight_id: str
    runway: str


@dataclass(frozen=True)
class BaseAdvisory:
    flight_number: str
    icao24: str
    flight_id: str
    runway: str
    miles_to_gain_nmi: float
    minutes_to_gain: float
    baseline_success: bool
    baseline_message: str
    what_if_success: bool
    what_if_message: str
    baseline_time_s: float
    what_if_time_s: float


@dataclass(frozen=True)
class FeasibilityAdvisory(BaseAdvisory):
    miles_to_gain_m: float
    search_converged: bool


@dataclass(frozen=True)
class VectoringAdvisory(BaseAdvisory):
    requested_extension_nmi: float
    requested_extension_m: float
    required_feasibility_miles_nmi: float
    required_feasibility_m: float
    feasibility_search_converged: bool


@dataclass(frozen=True)
class SpeedControlAdvisory(BaseAdvisory):
    s_m: float
    cas_kts: float
    equivalent_vectoring_miles_nmi: float
    equivalent_vectoring_m: float
