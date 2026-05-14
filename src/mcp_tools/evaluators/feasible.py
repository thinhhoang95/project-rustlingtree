from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from mcp_tools.scenario_manager.manager import ScenarioManager

_METERS_PER_NM = 1_852.0


@dataclass(frozen=True)
class FeasibleFlight:
    flight_number: str
    icao24: str
    flight_id: str
    runway: str
    missing_distance_nmi: float
    missing_distance_m: float
    simulation_message: str


@dataclass(frozen=True)
class FeasibleEvaluator:
    manager: ScenarioManager

    def evaluate(self) -> list[FeasibleFlight]:
        infeasible_flights: list[FeasibleFlight] = []
        for arrival in self.manager.arrival_schedule():
            simulation = self._simulation_metadata(arrival)
            success = self._simulation_success(arrival, simulation)
            missing_distance_m = self._missing_distance_m(arrival, simulation)
            if success:
                continue

            infeasible_flights.append(
                FeasibleFlight(
                    flight_number=str(arrival.get("callsign", "")),
                    icao24=str(arrival.get("icao24", "")),
                    flight_id=str(arrival.get("flight_id", "")),
                    runway=str(arrival.get("runway", "")),
                    missing_distance_nmi=float(missing_distance_m / _METERS_PER_NM),
                    missing_distance_m=missing_distance_m,
                    simulation_message=str(simulation.get("message", "")),
                )
            )

        return sorted(
            infeasible_flights,
            key=lambda item: (-item.missing_distance_m, item.flight_number, item.icao24),
        )

    @classmethod
    def _simulation_metadata(cls, arrival: dict[str, Any]) -> dict[str, Any]:
        simulation = arrival.get("simulation")
        if not isinstance(simulation, dict):
            raise ValueError(f"{cls._flight_label(arrival)} has missing or malformed simulation metadata")
        return simulation

    @classmethod
    def _simulation_success(cls, arrival: dict[str, Any], simulation: dict[str, Any]) -> bool:
        success = simulation.get("success")
        if not isinstance(success, bool):
            raise ValueError(f"{cls._flight_label(arrival)} has missing or malformed simulation.success")
        return success

    @classmethod
    def _missing_distance_m(cls, arrival: dict[str, Any], simulation: dict[str, Any]) -> float:
        value = simulation.get("final_threshold_error_m")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                f"{cls._flight_label(arrival)} has missing or malformed simulation.final_threshold_error_m"
            )

        distance_m = float(value)
        if not math.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError(
                f"{cls._flight_label(arrival)} has missing or malformed simulation.final_threshold_error_m"
            )
        return distance_m

    @staticmethod
    def _flight_label(arrival: dict[str, Any]) -> str:
        flight_id = str(arrival.get("flight_id", ""))
        callsign = str(arrival.get("callsign", ""))
        if flight_id and callsign:
            return f"flight_id={flight_id} callsign={callsign}"
        if flight_id:
            return f"flight_id={flight_id}"
        if callsign:
            return f"callsign={callsign}"
        return "arrival"
