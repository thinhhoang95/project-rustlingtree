from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from mcp_tools.scenario_manager.manager import ScenarioManager
from mcp_tools.scenario_manager.speed_intervention import (
    SpeedInterventionAdvisoryRequest,
    SpeedInterventionSaveRequest,
    SpeedInterventionSimulationRequest,
)
from simap.units import kts_to_mps
from tests.test_scenario_manager import write_fixture_resources


@dataclass(frozen=True)
class _FakeFmsRequest:
    start_s_m: float = 60_000.0
    atc_speed_segments: tuple = ()
    reference_path: object | None = None


def _fake_bichannel_result(*, elapsed_s: float, cas_kts: float) -> SimpleNamespace:
    return SimpleNamespace(
        t_s=np.asarray([0.0, elapsed_s], dtype=float),
        lat_deg=np.asarray([32.0, 32.9], dtype=float),
        lon_deg=np.asarray([-97.0, -97.0], dtype=float),
        h_m=np.asarray([1_000.0, 200.0], dtype=float),
        v_cas_mps=np.asarray([kts_to_mps(cas_kts), kts_to_mps(cas_kts - 10.0)], dtype=float),
        ground_speed_mps=np.asarray([120.0, 105.0], dtype=float),
        longitudinal=SimpleNamespace(level_distance_m=12_000.0, level_time_s=100.0),
        success=True,
        message="ok",
        max_abs_cross_track_m=10.0,
        max_abs_track_error_rad=0.01,
        final_threshold_error_m=2.0,
    )


def _patch_simap(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    import mcp_tools.scenario_manager.speed_intervention as speed_intervention

    captured_segments: list[tuple] = []

    def fake_build_request(**_kwargs):
        return _FakeFmsRequest(), SimpleNamespace()

    def fake_plan(request, **_kwargs):
        segments = tuple(request.base_request.atc_speed_segments)
        captured_segments.append(segments)
        if segments:
            return _fake_bichannel_result(elapsed_s=180.0, cas_kts=180.0)
        return _fake_bichannel_result(elapsed_s=120.0, cas_kts=210.0)

    monkeypatch.setattr(speed_intervention, "_build_request", fake_build_request)
    monkeypatch.setattr(speed_intervention, "plan_fms_bichannel", fake_plan)
    return captured_segments


def test_speed_intervention_simulate_uses_atc_speed_segments(tmp_path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    captured_segments = _patch_simap(monkeypatch)

    response = manager.simulate_speed_intervention(
        SpeedInterventionSimulationRequest(
            flight_id="ARR1",
            advisories=[
                SpeedInterventionAdvisoryRequest(
                    s_m=42_000.0,
                    cas_kts=180.0,
                    lat=32.91,
                    lon=-97.04,
                )
            ],
        )
    )

    assert len(captured_segments) == 2
    assert captured_segments[0] == ()
    assert len(captured_segments[1]) == 1
    segment = captured_segments[1][0]
    assert segment.s_from_m == 42_000.0
    assert segment.cas_mps == pytest.approx(kts_to_mps(180.0))
    assert response["trajectory"]["route_type"] == "speed-intervention"
    assert response["trajectory"]["speed_intervention"]["advisory_count"] == 1
    assert response["advisories"][0]["station_nm_to_runway"] == pytest.approx(42_000.0 / 1_852.0)
    assert response["metrics"]["old_elapsed_min"] == pytest.approx(2.0)
    assert response["metrics"]["new_elapsed_min"] == pytest.approx(3.0)
    assert response["metrics"]["delta_elapsed_min"] == pytest.approx(1.0)
    assert response["metrics"]["equivalent_distance_nm"] == pytest.approx(60.0 * 120.0 / 1_852.0)
    assert response["baseline_simulation"]["message"] == "ok"


def test_speed_intervention_save_replaces_active_intervention_and_serves_arrival(tmp_path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    _patch_simap(monkeypatch)
    manager.diff = [
        {
            "id": "old-path-stretch",
            "flight_id": "ARR1",
            "type": "path-stretch",
            "overrides": {},
        }
    ]
    response = manager.simulate_speed_intervention(
        SpeedInterventionSimulationRequest(
            flight_id="ARR1",
            advisories=[SpeedInterventionAdvisoryRequest(s_m=42_000.0, cas_kts=180.0)],
        )
    )

    saved = manager.save_speed_intervention(
        "ARR1",
        SpeedInterventionSaveRequest(draft_id=response["draft_id"]),
    )

    assert len(manager.intervention_diff()) == 1
    assert saved["diff"]["type"] == "speed-intervention"
    arrival = manager.arrival_schedule()[0]
    assert arrival["route_type"] == "speed-intervention"
    assert arrival["speed_intervention"]["advisory_count"] == 1
    assert arrival["cas_profile"]["columns"] == ["time", "cas_kts"]
    assert arrival["simulation"]["message"] == "ok"


def test_speed_intervention_validation_rejects_bad_inputs(tmp_path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    _patch_simap(monkeypatch)

    with pytest.raises(ValueError, match="at least one speed-intervention advisory"):
        manager.simulate_speed_intervention(
            SpeedInterventionSimulationRequest(flight_id="ARR1", advisories=[])
        )

    with pytest.raises(ValueError, match="must lie within the arrival reference path"):
        manager.simulate_speed_intervention(
            SpeedInterventionSimulationRequest(
                flight_id="ARR1",
                advisories=[SpeedInterventionAdvisoryRequest(s_m=70_000.0, cas_kts=180.0)],
            )
        )

    with pytest.raises(ValueError, match="cas_kts must be positive"):
        manager.simulate_speed_intervention(
            SpeedInterventionSimulationRequest(
                flight_id="ARR1",
                advisories=[SpeedInterventionAdvisoryRequest(s_m=42_000.0, cas_kts=0.0)],
            )
        )

    with pytest.raises(ValueError, match="unknown speed-intervention draft"):
        manager.save_speed_intervention("ARR1", SpeedInterventionSaveRequest(draft_id="missing"))
