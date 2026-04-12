from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap.backends import EffectivePolarBackend
from simap.calibration import build_default_aircraft_config, suggest_approach_mass_kg
from simap.longitudinal_profiles import (
    FeasibilityConfig,
    build_simple_glidepath,
    build_speed_schedule_from_wrap,
)
from simap.longitudinal_simulator import LongitudinalScenario
from simap.openap_adapter import extract_aircraft_data, load_openap
from simap.path_geometry import ReferencePath
from simap.weather import ConstantWeather


@lru_cache(maxsize=1)
def a320_fixture() -> dict[str, object]:
    openap = load_openap("A320")
    aircraft_data = extract_aircraft_data(openap)
    mass_kg = suggest_approach_mass_kg(aircraft_data, payload_kg=12_000.0)
    cfg, openap = build_default_aircraft_config("A320", mass_kg=mass_kg, openap_objects=openap)
    perf = EffectivePolarBackend(cfg=cfg, openap=openap)

    reference_path = ReferencePath.from_geographic(
        lat_deg=np.asarray([48.7600, 48.5600, 48.3538], dtype=float),
        lon_deg=np.asarray([11.0500, 11.3000, 11.7861], dtype=float),
    )
    intercept_distance_m = reference_path.total_length_m
    intercept_altitude_m = 3_500.0
    longitudinal_scenario = LongitudinalScenario(
        altitude_profile=build_simple_glidepath(
            threshold_elevation_m=450.0,
            intercept_distance_m=intercept_distance_m,
            intercept_altitude_m=intercept_altitude_m,
            glide_deg=3.0,
        ),
        raw_speed_schedule_cas=build_speed_schedule_from_wrap(openap.wrap),
        weather=ConstantWeather(),
        feasibility=FeasibilityConfig(planning_tailwind_mps=5.0, distance_step_m=250.0),
    )
    return {
        "cfg": cfg,
        "openap": openap,
        "perf": perf,
        "longitudinal_scenario": longitudinal_scenario,
        "reference_path": reference_path,
        "intercept_distance_m": intercept_distance_m,
        "intercept_altitude_m": intercept_altitude_m,
    }
