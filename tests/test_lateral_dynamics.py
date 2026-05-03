from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from simap import aero
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
        h_m = fixture["upstream"].h_m
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

    def test_bank_command_projects_to_closest_path_point_when_far_off_route(self) -> None:
        fixture = a320_fixture()
        cfg = fixture["cfg"]
        reference_path = fixture["reference_path"]
        h_m = fixture["upstream"].h_m
        v_tas_mps = float(aero.cas2tas(140.0, h_m, dT=0.0))

        anchor_idx = len(reference_path.s_m) // 2
        anchor_s_m = float(reference_path.s_m[anchor_idx])
        anchor_east_m, anchor_north_m = reference_path.position_ne(anchor_s_m)
        normal_hat = reference_path.normal_hat(anchor_s_m)
        east_m = float(anchor_east_m + 2_500.0 * float(normal_hat[0]))
        north_m = float(anchor_north_m + 2_500.0 * float(normal_hat[1]))

        far_s_m = float(reference_path.total_length_m)
        mode = mode_for_s(cfg, far_s_m)
        command = compute_lateral_command(
            s_m=far_s_m,
            east_m=east_m,
            north_m=north_m,
            h_m=h_m,
            t_s=0.0,
            psi_rad=float(reference_path.track_angle_rad(anchor_s_m)),
            v_tas_mps=v_tas_mps,
            cfg=cfg,
            mode=mode,
            reference_path=reference_path,
            weather=ConstantWeather(),
            guidance=LateralGuidanceConfig(),
        )

        projected_s_m = reference_path.project_s_m(east_m, north_m)
        nearest_ref_east_m, nearest_ref_north_m = reference_path.position_ne(projected_s_m)
        nearest_normal_hat = reference_path.normal_hat(projected_s_m)
        expected_cross_track_m = float(
            (east_m - nearest_ref_east_m) * float(nearest_normal_hat[0])
            + (north_m - nearest_ref_north_m) * float(nearest_normal_hat[1])
        )

        far_ref_east_m, far_ref_north_m = reference_path.position_ne(far_s_m)
        far_normal_hat = reference_path.normal_hat(far_s_m)
        far_cross_track_m = float(
            (east_m - far_ref_east_m) * float(far_normal_hat[0])
            + (north_m - far_ref_north_m) * float(far_normal_hat[1])
        )

        self.assertAlmostEqual(command.cross_track_m, expected_cross_track_m, delta=1e-6)
        self.assertGreater(abs(command.cross_track_m - far_cross_track_m), 1_000.0)


if __name__ == "__main__":
    unittest.main()
