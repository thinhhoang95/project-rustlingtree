from __future__ import annotations

import pytest
from openap import aero

from mcp_tools.utils import GroundDistanceEstimationTool
from simap.openap_adapter import openap_dT
from simap.units import ft_to_m, kts_to_mps, mps_to_kts


def test_ground_distance_estimator_returns_required_descent_distance() -> None:
    tool = GroundDistanceEstimationTool(dt_s=1.0)

    estimate = tool.estimate(altitude_ft=26_000.0, tas_kts=430.0)

    assert estimate.success
    assert estimate.required_ground_distance_m > 0.0
    assert 95.0 < estimate.required_ground_distance_nmi < 125.0
    assert estimate.required_ground_distance_nmi == pytest.approx(estimate.tod_distance_nmi)
    assert estimate.final_altitude_ft == pytest.approx(620.0)


def test_ground_distance_estimator_converts_initial_tas_to_cas() -> None:
    tool = GroundDistanceEstimationTool(dt_s=1.0)

    estimate = tool.estimate(altitude_ft=26_000.0, tas_kts=430.0)
    expected_cas_kts = mps_to_kts(
        float(aero.tas2cas(kts_to_mps(430.0), ft_to_m(26_000.0), dT=openap_dT(0.0)))
    )

    assert estimate.initial_cas_kts == pytest.approx(expected_cas_kts)


def test_ground_distance_estimator_call_returns_nmi() -> None:
    tool = GroundDistanceEstimationTool(dt_s=1.0)

    distance_nmi = tool(altitude_ft=26_000.0, tas_kts=430.0)

    assert 95.0 < distance_nmi < 125.0


@pytest.mark.parametrize(
    ("altitude_ft", "tas_kts", "match"),
    [
        (620.0, 430.0, "altitude_ft must be greater than runway_altitude_ft"),
        (26_000.0, 0.0, "tas_kts must be positive"),
        (float("nan"), 430.0, "altitude_ft must be finite"),
    ],
)
def test_ground_distance_estimator_rejects_invalid_inputs(
    altitude_ft: float,
    tas_kts: float,
    match: str,
) -> None:
    tool = GroundDistanceEstimationTool(dt_s=1.0)

    with pytest.raises(ValueError, match=match):
        tool.estimate(altitude_ft=altitude_ft, tas_kts=tas_kts)
