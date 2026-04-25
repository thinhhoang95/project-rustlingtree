from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from typing import TypedDict

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap.backends import EffectivePolarBackend
from simap.config import AircraftConfig
from simap.calibration import build_default_aircraft_config, suggest_approach_mass_kg
from simap.longitudinal_planner import (
    LongitudinalPlanRequest,
    OptimizerConfig,
    ThresholdBoundary,
    UpstreamBoundary,
)
from simap.longitudinal_profiles import ConstraintEnvelope, ScalarProfile, build_speed_schedule_from_wrap
from simap.openap_adapter import extract_aircraft_data, load_openap
from simap.path_geometry import ReferencePath
from simap.openap_adapter import OpenAPObjects
from simap.weather import ConstantWeather

HAS_OPENAP = importlib.util.find_spec("openap") is not None


class A320Fixture(TypedDict):
    cfg: AircraftConfig
    openap: OpenAPObjects
    perf: EffectivePolarBackend
    reference_path: ReferencePath
    threshold: ThresholdBoundary
    upstream: UpstreamBoundary
    constraint_envelope: ConstraintEnvelope
    planner_request: LongitudinalPlanRequest


@lru_cache(maxsize=1)
def a320_fixture() -> A320Fixture:
    if not HAS_OPENAP:
        raise RuntimeError("openap is not installed")

    openap = load_openap("A320")
    aircraft_data = extract_aircraft_data(openap)
    mass_kg = suggest_approach_mass_kg(aircraft_data, payload_kg=12_000.0)
    cfg, openap = build_default_aircraft_config("A320", mass_kg=mass_kg, openap_objects=openap)
    perf = EffectivePolarBackend(cfg=cfg, openap=openap)

    reference_path = ReferencePath.from_geographic(
        lat_deg=np.asarray([48.7600, 48.5600, 48.3538], dtype=float),
        lon_deg=np.asarray([11.0500, 11.3000, 11.7861], dtype=float),
    )
    max_s_m = max(reference_path.total_length_m, 60_000.0)
    threshold = ThresholdBoundary(
        h_m=450.0,
        cas_mps=float(openap.wrap.landing_speed()["default"]),
        gamma_rad=-np.deg2rad(3.0),
    )
    upstream = UpstreamBoundary(
        h_m=3_500.0,
        cas_window_mps=(
            float(openap.wrap.finalapp_vcas()["default"]),
            float(openap.wrap.descent_const_vcas()["default"]),
        ),
    )
    speed_schedule = build_speed_schedule_from_wrap(openap.wrap)
    envelope = ConstraintEnvelope.from_profiles(
        altitude_lower=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([threshold.h_m, upstream.h_m - 50.0], dtype=float),
        ),
        altitude_upper=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([threshold.h_m + 25.0, upstream.h_m + 125.0], dtype=float),
        ),
        cas_lower=ScalarProfile(
            s_m=speed_schedule.s_m,
            y=np.maximum(speed_schedule.y - 10.0 * 0.514444, threshold.cas_mps),
        ),
        cas_upper=ScalarProfile(
            s_m=speed_schedule.s_m,
            y=speed_schedule.y,
        ),
        gamma_lower=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([-np.deg2rad(4.0), -np.deg2rad(0.5)], dtype=float),
        ),
        gamma_upper=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([-np.deg2rad(2.0), np.deg2rad(0.5)], dtype=float),
        ),
    )
    request = LongitudinalPlanRequest(
        cfg=cfg,
        perf=perf,
        threshold=threshold,
        upstream=upstream,
        constraints=envelope,
        reference_path=reference_path,
        weather=ConstantWeather(),
        optimizer=OptimizerConfig(num_nodes=31, maxiter=250),
    )
    return {
        "cfg": cfg,
        "openap": openap,
        "perf": perf,
        "reference_path": reference_path,
        "threshold": threshold,
        "upstream": upstream,
        "constraint_envelope": envelope,
        "planner_request": request,
    }
