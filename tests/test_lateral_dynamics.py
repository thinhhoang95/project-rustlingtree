from __future__ import annotations

import os
import unittest

from openap import aero

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap.config import mode_for_s
from simap.lateral_dynamics import LateralGuidanceConfig, compute_lateral_command
from simap.simulator import State
from simap.weather import ConstantWeather
from tests.helpers import a320_fixture


class LateralDynamicsTests(unittest.TestCase):
    def test_bank_command_scales_with_alongtrack_wind_for_curved_ground_path(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        reference_path = fixture["reference_path"]
        curvature_index = int(abs(reference_path.curvature_inv_m).argmax())
        s_m = float(reference_path.s_m[curvature_index])
        h_m = fixture["intercept_altitude_m"]
        v_tas_mps = float(aero.cas2tas(140.0, h_m, dT=0.0))
        state = State.on_reference_path(
            t_s=0.0,
            s_m=s_m,
            h_m=h_m,
            v_tas_mps=v_tas_mps,
            reference_path=reference_path,
        )
        tangent_hat = reference_path.tangent_hat(s_m)
        mode = mode_for_s(cfg, s_m)

        calm = compute_lateral_command(
            s_m=s_m,
            east_m=state.east_m,
            north_m=state.north_m,
            h_m=h_m,
            t_s=0.0,
            psi_rad=state.psi_rad,
            v_tas_mps=v_tas_mps,
            cfg=cfg,
            mode=mode,
            reference_path=reference_path,
            weather=ConstantWeather(),
            guidance=LateralGuidanceConfig(),
        )
        tailwind = compute_lateral_command(
            s_m=s_m,
            east_m=state.east_m,
            north_m=state.north_m,
            h_m=h_m,
            t_s=0.0,
            psi_rad=state.psi_rad,
            v_tas_mps=v_tas_mps,
            cfg=cfg,
            mode=mode,
            reference_path=reference_path,
            weather=ConstantWeather(
                wind_east_mps=15.0 * float(tangent_hat[0]),
                wind_north_mps=15.0 * float(tangent_hat[1]),
            ),
            guidance=LateralGuidanceConfig(),
        )
        headwind = compute_lateral_command(
            s_m=s_m,
            east_m=state.east_m,
            north_m=state.north_m,
            h_m=h_m,
            t_s=0.0,
            psi_rad=state.psi_rad,
            v_tas_mps=v_tas_mps,
            cfg=cfg,
            mode=mode,
            reference_path=reference_path,
            weather=ConstantWeather(
                wind_east_mps=-15.0 * float(tangent_hat[0]),
                wind_north_mps=-15.0 * float(tangent_hat[1]),
            ),
            guidance=LateralGuidanceConfig(),
        )

        self.assertGreater(abs(tailwind.phi_req_rad), abs(calm.phi_req_rad))
        self.assertGreater(abs(calm.phi_req_rad), abs(headwind.phi_req_rad))


if __name__ == "__main__":
    unittest.main()
