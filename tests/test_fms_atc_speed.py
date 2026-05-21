from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simap.config import mode_for_s
from simap.fms import (
    ATCSpeedSegment,
    FMSRequest,
    FMSSpeedTargets,
    HoldAwareFMSRequest,
    HoldInstruction,
    simulate_fms_descent,
    simulate_hold_aware_fms_descent,
)
from simap.fms.helpers import _managed_target_cas_mps
from simap.fms_bichannel import FMSBiChannelRequest, simulate_fms_bichannel
from simap.units import kts_to_mps
from tests.test_simulator import build_test_request


def _speed_targets() -> FMSSpeedTargets:
    return FMSSpeedTargets(
        clean_cas_mps=105.0,
        approach_cas_mps=90.0,
        final_cas_mps=70.0,
        below_altitude_limit_h_m=None,
        below_altitude_limit_cas_mps=None,
    )


def _request(*, start_s_m: float = 50_000.0) -> FMSRequest:
    return FMSRequest.from_coupled_request(
        build_test_request(),
        speed_targets=_speed_targets(),
        start_s_m=start_s_m,
        dt_s=1.0,
        max_time_s=1_000.0,
    )


def _target_at(request: FMSRequest, s_m: float) -> float:
    return _managed_target_cas_mps(
        request=request,
        mode=mode_for_s(request.cfg, s_m),
        s_m=s_m,
        h_m=request.start_h_m,
    )


def test_empty_atc_speed_segments_preserve_fms_response() -> None:
    request = _request(start_s_m=20_000.0)
    explicit_empty = replace(request, atc_speed_segments=())

    baseline = simulate_fms_descent(request)
    explicit = simulate_fms_descent(explicit_empty)

    assert np.allclose(explicit.t_s, baseline.t_s)
    assert np.allclose(explicit.s_m, baseline.s_m)
    assert np.allclose(explicit.v_cas_mps, baseline.v_cas_mps)
    assert np.allclose(explicit.target_cas_mps, baseline.target_cas_mps)


def test_atc_speed_segment_persists_until_base_profile_is_lower() -> None:
    request = replace(
        _request(),
        atc_speed_segments=((45_000.0, 95.0),),
    )

    assert request.atc_speed_segments == (ATCSpeedSegment(s_from_m=45_000.0, cas_mps=95.0),)
    assert _target_at(request, 46_000.0) == pytest.approx(105.0)
    assert _target_at(request, 44_000.0) == pytest.approx(95.0)
    assert _target_at(request, 36_000.0) == pytest.approx(95.0)
    assert _target_at(request, request.cfg.approach_gate_m) == pytest.approx(90.0)
    assert _target_at(request, 34_000.0) == pytest.approx(90.0)


def test_lower_atc_segment_reduces_target_without_later_increase() -> None:
    request = replace(
        _request(),
        atc_speed_segments=(
            ATCSpeedSegment(s_from_m=48_000.0, cas_mps=100.0),
            ATCSpeedSegment(s_from_m=44_000.0, cas_mps=95.0),
        ),
    )

    targets = [_target_at(request, s_m) for s_m in (47_000.0, 43_000.0, 36_000.0, 34_000.0)]

    assert targets == pytest.approx([100.0, 95.0, 95.0, 90.0])
    assert all(next_target <= target for target, next_target in zip(targets, targets[1:]))


def test_atc_speed_segment_faster_than_base_profile_raises_early() -> None:
    with pytest.raises(ValueError, match="not lower than base profile"):
        replace(
            _request(),
            atc_speed_segments=(ATCSpeedSegment(s_from_m=45_000.0, cas_mps=110.0),),
        )


def test_atc_speed_segment_below_mode_lower_bound_raises_early() -> None:
    with pytest.raises(ValueError, match="below planned lower CAS bound"):
        replace(
            _request(),
            atc_speed_segments=(ATCSpeedSegment(s_from_m=45_000.0, cas_mps=60.0),),
        )


def test_bichannel_uses_longitudinal_atc_speed_profile() -> None:
    request = _request(start_s_m=20_000.0)
    base_request = replace(
        request,
        target_h_m=request.start_h_m - 100.0,
        atc_speed_segments=(ATCSpeedSegment(s_from_m=19_950.0, cas_mps=85.0),),
    )

    result = simulate_fms_bichannel(FMSBiChannelRequest(base_request=base_request))

    assert result.success
    assert len(result) == len(result.longitudinal)
    assert np.any(result.s_m <= 19_950.0)
    assert np.min(result.longitudinal.target_cas_mps[result.s_m <= 19_950.0]) <= 85.0 + 1e-9
    assert result.max_abs_cross_track_m < 10.0


def test_hold_aware_fms_keeps_hold_speed_outside_atc_managed_profile() -> None:
    hold_speed_kts = 185.0
    request = replace(
        _request(),
        atc_speed_segments=(ATCSpeedSegment(s_from_m=50_000.0, cas_mps=80.0),),
    )
    result = simulate_hold_aware_fms_descent(
        HoldAwareFMSRequest(
            base_request=request,
            holds=(HoldInstruction(holding_altitude_ft=9_500.0, holding_time_s=5.0, holding_speed_kts=hold_speed_kts),),
        )
    )

    hold_mask = np.asarray([phase.startswith("hold") for phase in result.phase], dtype=bool)
    managed_mask = np.asarray([phase == "managed_descent" for phase in result.phase], dtype=bool)

    assert np.any(managed_mask)
    assert np.any(np.isclose(result.target_cas_mps[managed_mask], 80.0))
    assert np.any(hold_mask)
    assert np.allclose(result.target_cas_mps[hold_mask], kts_to_mps(hold_speed_kts))
