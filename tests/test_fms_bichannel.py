from __future__ import annotations

import numpy as np
from dataclasses import replace

from simap.fms import FMSRequest
from simap.fms_bichannel import FMSBiChannelRequest, FMSBiChannelState, simulate_fms_bichannel
from tests.test_simulator import build_test_request


def _request() -> FMSRequest:
    coupled = build_test_request()
    request = FMSRequest.from_coupled_request(
        coupled,
        start_s_m=20_000.0,
        dt_s=1.0,
        max_time_s=800.0,
    )
    return replace(request, target_h_m=request.start_h_m - 100.0)


def test_bichannel_matches_longitudinal_grid_and_tracks_straight_path() -> None:
    result = simulate_fms_bichannel(FMSBiChannelRequest(base_request=_request()))

    assert result.success
    assert len(result) == len(result.longitudinal)
    assert np.allclose(result.t_s, result.longitudinal.t_s)
    assert np.all(np.isfinite(result.lat_deg))
    assert np.all(np.isfinite(result.lon_deg))
    assert result.max_abs_cross_track_m < 1e-6
    assert np.all(np.abs(result.phi_req_rad) <= result.phi_max_rad + 1e-9)


def test_bichannel_recovers_from_initial_cross_track_offset() -> None:
    request = _request()
    initial_state = FMSBiChannelState.on_reference_path(
        t_s=0.0,
        s_m=request.start_s_m,
        h_m=request.start_h_m,
        v_tas_mps=request.start_cas_mps,
        reference_path=request.reference_path,
        cross_track_m=150.0,
    )

    result = simulate_fms_bichannel(
        FMSBiChannelRequest(
            base_request=request,
            initial_state=initial_state,
        )
    )

    assert result.success
    assert abs(result.cross_track_m[0]) > 100.0
    assert abs(result.cross_track_m[-1]) < abs(result.cross_track_m[0])
