from __future__ import annotations

import numpy as np
from openap import aero

from simap.backends import EffectivePolarBackend
from simap.calibration import build_default_aircraft_config, suggest_approach_mass_kg
from simap.dynamics import State
from simap.openap_adapter import extract_aircraft_data, load_openap
from simap.profiles import Centerline, FeasibilityConfig, build_simple_glidepath, build_speed_schedule_from_wrap
from simap.simulator import ApproachSimulator, Scenario
from simap.weather import ConstantWeather


def build_centerline_example(intercept_distance_m: float) -> Centerline:
    return Centerline(
        x_m=np.asarray([0.0, intercept_distance_m], dtype=float),
        lat_deg=np.asarray([48.3538, 48.5000], dtype=float),
        lon_deg=np.asarray([11.7861, 11.5000], dtype=float),
    )


def main() -> None:
    openap = load_openap("A320")
    aircraft_data = extract_aircraft_data(openap)
    mass_kg = suggest_approach_mass_kg(aircraft_data, payload_kg=12_000.0)
    cfg, openap = build_default_aircraft_config("A320", mass_kg=mass_kg, openap_objects=openap)
    perf = EffectivePolarBackend(cfg=cfg, openap=openap)

    intercept_distance_m = 55_000.0
    threshold_elevation_m = 450.0
    intercept_altitude_m = 3_500.0

    scenario = Scenario(
        altitude_profile=build_simple_glidepath(
            threshold_elevation_m=threshold_elevation_m,
            intercept_distance_m=intercept_distance_m,
            intercept_altitude_m=intercept_altitude_m,
            glide_deg=3.0,
        ),
        raw_speed_schedule_cas=build_speed_schedule_from_wrap(openap.wrap),
        centerline=build_centerline_example(intercept_distance_m),
        weather=ConstantWeather(alongtrack_wind_mps=-10.0, delta_isa_offset_K=0.0),
        feasibility=FeasibilityConfig(planning_tailwind_mps=5.0, distance_step_m=250.0),
    )
    simulator = ApproachSimulator(cfg=cfg, perf=perf, scenario=scenario)

    v0_cas_mps = simulator.feasible_speed_schedule_cas.value(intercept_distance_m)
    v0_tas_mps = float(aero.cas2tas(v0_cas_mps, intercept_altitude_m, dT=0.0))
    initial = State(
        t_s=0.0,
        x_m=intercept_distance_m,
        h_m=intercept_altitude_m,
        v_tas_mps=v0_tas_mps,
    )
    trajectory = simulator.run(initial, dt_s=1.0)
    df = trajectory.to_pandas()
    print(df.head())
    print(df.tail())
