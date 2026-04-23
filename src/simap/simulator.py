from __future__ import annotations

from dataclasses import dataclass

from .lateral_dynamics import wrap_angle_rad
from .path_geometry import ReferencePath


@dataclass(frozen=True)
class State:
    t_s: float
    s_m: float
    h_m: float
    v_tas_mps: float
    east_m: float
    north_m: float
    psi_rad: float
    phi_rad: float = 0.0

    @classmethod
    def on_reference_path(
        cls,
        *,
        t_s: float,
        s_m: float,
        h_m: float,
        v_tas_mps: float,
        reference_path: ReferencePath,
        bank_rad: float = 0.0,
        heading_offset_rad: float = 0.0,
        cross_track_m: float = 0.0,
    ) -> "State":
        east_m, north_m = reference_path.position_ne(s_m)
        normal_hat = reference_path.normal_hat(s_m)
        psi_rad = wrap_angle_rad(reference_path.track_angle_rad(s_m) + heading_offset_rad)
        return cls(
            t_s=t_s,
            s_m=s_m,
            h_m=h_m,
            v_tas_mps=v_tas_mps,
            east_m=east_m + cross_track_m * normal_hat[0],
            north_m=north_m + cross_track_m * normal_hat[1],
            psi_rad=psi_rad,
            phi_rad=bank_rad,
        )
