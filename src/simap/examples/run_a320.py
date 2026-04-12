from __future__ import annotations

import numpy as np
from openap import aero

from simap import (
    ApproachSimulator,
    ConstantWeather,
    FeasibilityConfig,
    ReferencePath,
    Scenario,
    State,
    build_default_aircraft_config,
    build_simple_glidepath,
    build_speed_schedule_from_wrap,
    extract_aircraft_data,
    load_openap,
    suggest_approach_mass_kg,
)
from simap.backends import EffectivePolarBackend


def build_reference_path_example() -> ReferencePath:
    return ReferencePath.from_geographic(
        lat_deg=np.asarray([48.7600, 48.5600, 48.3538], dtype=float),
        lon_deg=np.asarray([11.0500, 11.3000, 11.7861], dtype=float),
    )


def main() -> None:
    openap = load_openap("A320")
    aircraft_data = extract_aircraft_data(openap)
    mass_kg = suggest_approach_mass_kg(aircraft_data, payload_kg=12_000.0)
    cfg, openap = build_default_aircraft_config("A320", mass_kg=mass_kg, openap_objects=openap)
    perf = EffectivePolarBackend(cfg=cfg, openap=openap)

    reference_path = build_reference_path_example()
    intercept_distance_m = reference_path.total_length_m
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
        reference_path=reference_path,
        weather=ConstantWeather(wind_east_mps=-8.0, wind_north_mps=2.0, delta_isa_offset_K=0.0),
        feasibility=FeasibilityConfig(planning_tailwind_mps=5.0, distance_step_m=250.0),
    )
    simulator = ApproachSimulator(cfg=cfg, perf=perf, scenario=scenario)

    v0_cas_mps = simulator.feasible_speed_schedule_cas.value(intercept_distance_m)
    v0_tas_mps = float(aero.cas2tas(v0_cas_mps, intercept_altitude_m, dT=0.0))
    initial = State.on_reference_path(
        t_s=0.0,
        s_m=intercept_distance_m,
        h_m=intercept_altitude_m,
        v_tas_mps=v0_tas_mps,
        reference_path=reference_path,
    )
    trajectory = simulator.run(initial, dt_s=1.0)
    df = trajectory.to_pandas()
    print(df.head())
    print(df.tail())
