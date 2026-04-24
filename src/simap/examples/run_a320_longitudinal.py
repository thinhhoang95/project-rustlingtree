from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap import (
    ConstraintEnvelope,
    EffectivePolarBackend,
    LongitudinalPlanRequest,
    OptimizerConfig,
    ScalarProfile,
    ThresholdBoundary,
    UpstreamBoundary,
    build_default_aircraft_config,
    build_speed_schedule_from_wrap,
    extract_aircraft_data,
    load_openap,
    plan_longitudinal_descent,
    plot_constraint_envelope,
    plot_longitudinal_plan,
    suggest_approach_mass_kg,
)


def main() -> None:
    openap = load_openap("A320")
    aircraft_data = extract_aircraft_data(openap)
    mass_kg = suggest_approach_mass_kg(aircraft_data, payload_kg=12_000.0)
    cfg, openap = build_default_aircraft_config("A320", mass_kg=mass_kg, openap_objects=openap)
    perf = EffectivePolarBackend(cfg=cfg, openap=openap)

    # Boundary condition at the runway threshold
    threshold = ThresholdBoundary( 
        h_m=450.0,
        cas_mps=float(openap.wrap.landing_speed()["default"]),
        gamma_rad=-np.deg2rad(3.0),
    )

    # Boundary condition at upstream, before the top of descent. The top of descent will be determined by the optimizer.
    upstream = UpstreamBoundary(
        h_m=3_000.0,
        cas_window_mps=(
            float(openap.wrap.finalapp_vcas()["default"]), # 140 knots
            float(openap.wrap.descent_const_vcas()["default"]), # 290 knots
        ),
    )

    # Speed schedule is built from speed constraints of configuration, where configuration changes happen at specific distances from the runway.
    # s_m = array([0, 8km, 30km, 60km]) y = array([135 knots (touchdown), 140 knots (final), 280 knots (descent), 280 knots (descent)])
    speed_schedule = build_speed_schedule_from_wrap(openap.wrap)

    max_s_m = 60_000.0
    envelope = ConstraintEnvelope.from_profiles(
        altitude_lower=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([threshold.h_m, upstream.h_m - 75.0], dtype=float),
        ),
        altitude_upper=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([threshold.h_m + 25.0, upstream.h_m + 150.0], dtype=float),
        ),
        cas_lower=ScalarProfile(
            s_m=speed_schedule.s_m,
            y=np.maximum(speed_schedule.y - 8.0 * 0.514444, threshold.cas_mps),
        ),
        cas_upper=ScalarProfile(
            s_m=speed_schedule.s_m,
            y=speed_schedule.y,
        ),
        gamma_lower=ScalarProfile(
            s_m=np.asarray([0.0, max_s_m], dtype=float),
            y=np.asarray([-np.deg2rad(4.5), -np.deg2rad(0.5)], dtype=float),
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
        optimizer=OptimizerConfig(num_nodes=31, maxiter=300),
    )

    plot_constraint_envelope(envelope)

    plan = plan_longitudinal_descent(request)
    plot_longitudinal_plan(plan, envelope=envelope)
    print(plan.to_pandas().head())
    print(plan.to_pandas().tail())
    print(
        {
            "success": plan.solver_success,
            "tod_m": plan.tod_m,
            "collocation_residual_max": plan.collocation_residual_max,
            "replay_residual_max": plan.replay_residual_max,
        }
    )


if __name__ == "__main__":
    main()
