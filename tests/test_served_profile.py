from __future__ import annotations

from pathlib import Path

import pytest

from mcp_tools.scenario_manager.manager import ScenarioManager
from mcp_tools.scenario_manager.served_profile import build_served_fms_context
from simap.units import mps_to_kts
from tests.test_scenario_manager import write_fixture_resources


def test_served_context_uses_persisted_cas_profile_at_first_fix(tmp_path: Path) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    arrival = manager.arrival_schedule()[0]

    context = build_served_fms_context(arrival, manager.config.fixes_path)

    assert context.seed.cas_mps is not None
    assert mps_to_kts(context.fms_request.start_cas_mps) == pytest.approx(190.0)


def test_served_context_rejects_missing_cas_profile_by_default(tmp_path: Path) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    arrival = manager.arrival_schedule()[0]
    arrival.pop("cas_profile")

    with pytest.raises(ValueError, match="cas_profile.points\\[0\\].cas_kts"):
        build_served_fms_context(arrival, manager.config.fixes_path)


def test_served_context_falls_back_to_compressed_points_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcp_tools.scenario_manager.served_profile as served_profile

    monkeypatch.setattr(served_profile, "FALLBACK_SPEED_USING_TWO_TRAJECTORY_POINTS", True)
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    arrival = manager.arrival_schedule()[0]
    arrival.pop("cas_profile")

    context = build_served_fms_context(arrival, manager.config.fixes_path)

    assert context.seed.cas_mps is None
    assert mps_to_kts(context.fms_request.start_cas_mps) != pytest.approx(190.0)
    assert context.seed.ground_speed_mps > 1.0


def test_served_context_replays_active_speed_intervention_advisories(tmp_path: Path) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    arrival = manager.arrival_schedule()[0]
    arrival["speed_intervention"] = {
        "advisories": [
            {
                "s_m": 60_000.0,
                "station_nm_to_runway": 60_000.0 / 1_852.0,
                "cas_kts": 160.0,
                "lat": 32.05,
                "lon": -97.05,
            }
        ],
        "advisory_count": 1,
    }

    context = build_served_fms_context(arrival, manager.config.fixes_path)
    without_replay = build_served_fms_context(
        arrival,
        manager.config.fixes_path,
        include_speed_advisories=False,
    )

    assert len(context.speed_advisories) == 1
    assert len(context.fms_request.atc_speed_segments) == 1
    assert context.fms_request.atc_speed_segments[0].s_from_m == pytest.approx(60_000.0)
    assert without_replay.fms_request.atc_speed_segments == ()
