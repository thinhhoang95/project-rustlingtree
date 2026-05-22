from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from mcp_tools.advisors import FeasibilityAdvisor, ProfilePlanner, SpeedControlAdvisor, VectoringAdvisor
from mcp_tools.advisors.models import AdvisoryFlight
from mcp_tools.advisors.profile import METERS_PER_NM, extend_reference_path
from simap.fms import ATCSpeedSegment
from simap.path_geometry import ReferencePath
from simap.units import kts_to_mps


def _write_fixes(path: Path) -> Path:
    fixes_path = path / "fixes.csv"
    fixes_path.write_text(
        "identifier,latitude_deg,longitude_deg,fix_type,elevation_ft\n"
        "FIXA,32.0,-97.0,fix,\n"
        "FINAL35C,32.8,-97.0,fix,\n"
        "RW35C,32.9,-97.0,runway,620\n",
        encoding="utf-8",
    )
    return fixes_path


def _arrival(
    *,
    flight_id: str = "ARR1",
    cas_profile: dict[str, object] | None = None,
    threshold_nmi: float = 0.0,
) -> dict[str, object]:
    arrival: dict[str, object] = {
        "flight_id": flight_id,
        "callsign": f"CALL{flight_id}",
        "icao24": f"icao{flight_id}",
        "runway": "RW35C",
        "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
        "points": [
            [100, 32.0, -97.0, 1_500.0, 3],
            [110, 32.01, -97.0, 1_450.0, 3],
        ],
        "base_route": {
            "lateral_path": ["FIXA", "FINAL35C", "RW35C"],
            "upstream_identifier": "FIXA",
        },
        "threshold_nmi": threshold_nmi,
    }
    if cas_profile is not None:
        arrival["cas_profile"] = cas_profile
    return arrival


def _cas_profile(cas_kts: float = 190.0) -> dict[str, object]:
    return {
        "columns": ["time", "cas_kts"],
        "units": {"cas_kts": "kt"},
        "points": [[100, cas_kts], [102, cas_kts - 1.0]],
    }


class FakeManager:
    def __init__(self, arrivals: list[dict[str, object]], fixes_path: Path | None = None) -> None:
        self._arrivals = arrivals
        self.config = SimpleNamespace(fixes_path=fixes_path or Path("/tmp/fixes.csv"))

    def arrival_schedule(self) -> list[dict[str, object]]:
        return [dict(arrival) for arrival in self._arrivals]


class FakePlanner:
    tod_tolerance_m = 1.0

    def build(self, arrival: dict[str, object], _fixes_path: Path) -> SimpleNamespace:
        identity = AdvisoryFlight(
            flight_number=str(arrival["callsign"]),
            icao24=str(arrival["icao24"]),
            flight_id=str(arrival["flight_id"]),
            runway=str(arrival["runway"]),
        )
        return SimpleNamespace(
            identity=identity,
            request=SimpleNamespace(start_s_m=100_000.0),
            threshold_m=float(arrival.get("threshold_nmi", 0.0)) * METERS_PER_NM,
        )

    def plan(
        self,
        profile: SimpleNamespace,
        *,
        extra_distance_m: float = 0.0,
        atc_speed_segments: tuple[ATCSpeedSegment, ...] = (),
    ) -> SimpleNamespace:
        if atc_speed_segments:
            segment = atc_speed_segments[0]
            if segment.cas_mps >= kts_to_mps(240.0):
                raise ValueError("not lower than base profile")
            return SimpleNamespace(
                success=True,
                message="speed profile complete",
                total_time_s=1_120.0,
                pre_tod_ground_speed_mps=100.0,
            )
        success = extra_distance_m >= profile.threshold_m
        return SimpleNamespace(
            success=success,
            message="ok" if success else "infeasible",
            total_time_s=1_000.0 + extra_distance_m / 100.0,
            pre_tod_ground_speed_mps=100.0,
        )


def test_profile_reconstruction_uses_served_route_and_cas_profile(tmp_path: Path) -> None:
    planner = ProfilePlanner(fms_dt_s=2.0)
    profile = planner.build(_arrival(cas_profile=_cas_profile(190.0)), _write_fixes(tmp_path))

    assert profile.identity.flight_id == "ARR1"
    assert profile.request.start_s_m == pytest.approx(profile.request.reference_path.total_length_m)
    assert profile.request.start_cas_mps == pytest.approx(kts_to_mps(190.0))
    assert profile.request.start_h_m == pytest.approx(1_500.0)


def test_profile_reconstruction_replays_active_speed_advisories(tmp_path: Path) -> None:
    arrival = _arrival(cas_profile=_cas_profile(190.0))
    arrival["speed_intervention"] = {
        "advisories": [
            {
                "s_m": 60_000.0,
                "station_nm_to_runway": 60_000.0 / METERS_PER_NM,
                "cas_kts": 160.0,
                "lat": 32.1,
                "lon": -97.0,
            }
        ],
        "advisory_count": 1,
    }

    profile = ProfilePlanner(fms_dt_s=2.0).build(arrival, _write_fixes(tmp_path))

    assert len(profile.request.atc_speed_segments) == 1
    segment = profile.request.atc_speed_segments[0]
    assert isinstance(segment, ATCSpeedSegment)
    assert segment.s_from_m == pytest.approx(60_000.0)


def test_profile_reconstruction_rejects_missing_cas_profile_by_default(tmp_path: Path) -> None:
    planner = ProfilePlanner(fms_dt_s=2.0)

    with pytest.raises(ValueError, match="cas_profile.points\\[0\\].cas_kts"):
        planner.build(_arrival(cas_profile=None), _write_fixes(tmp_path))


def test_extend_reference_path_adds_prefix_distance_and_keeps_threshold() -> None:
    path = ReferencePath.from_geographic(
        lat_deg=[32.0, 32.5, 32.9],
        lon_deg=[-97.0, -97.0, -97.0],
    )

    extended = extend_reference_path(path, METERS_PER_NM)

    assert extended.total_length_m == pytest.approx(path.total_length_m + METERS_PER_NM)
    assert extended.s_m[-1] == pytest.approx(0.0)
    assert extended.latlon(0.0) == pytest.approx(path.latlon(0.0))


def test_feasibility_advisor_searches_required_extension_and_sorts() -> None:
    manager = FakeManager(
        [
            _arrival(flight_id="A", threshold_nmi=2.0),
            _arrival(flight_id="B", threshold_nmi=4.0),
        ]
    )
    advisor = FeasibilityAdvisor(manager, planner=FakePlanner(), initial_extension_nmi=1.0, max_extension_nmi=8.0)

    result = advisor.evaluate()
    filtered = advisor.evaluate(flight_id="A")

    assert [item.flight_id for item in result] == ["B", "A"]
    assert result[0].miles_to_gain_nmi == pytest.approx(4.0, abs=1.0 / METERS_PER_NM)
    assert result[0].what_if_success
    assert result[0].search_converged
    assert [item.flight_id for item in filtered] == ["A"]


def test_feasibility_advisor_returns_zero_for_feasible_baseline() -> None:
    manager = FakeManager([_arrival(flight_id="A", threshold_nmi=0.0)])

    result = FeasibilityAdvisor(manager, planner=FakePlanner()).evaluate()

    assert result[0].miles_to_gain_nmi == 0.0
    assert result[0].minutes_to_gain == 0.0
    assert result[0].baseline_success


def test_vectoring_advisor_reports_requested_and_required_distance() -> None:
    manager = FakeManager([_arrival(flight_id="A", threshold_nmi=4.0)])

    result = VectoringAdvisor(manager, planner=FakePlanner()).advise(flight_id="A", extra_distance_nmi=2.0)

    assert result.miles_to_gain_nmi == 2.0
    assert result.requested_extension_m == pytest.approx(2.0 * METERS_PER_NM)
    assert result.required_feasibility_miles_nmi == pytest.approx(4.0, abs=1.0 / METERS_PER_NM)
    assert not result.what_if_success


def test_vectoring_advisor_zero_extension_has_zero_time_delta() -> None:
    manager = FakeManager([_arrival(flight_id="A", threshold_nmi=0.0)])

    result = VectoringAdvisor(manager, planner=FakePlanner()).advise(flight_id="A", extra_distance_nmi=0.0)

    assert result.miles_to_gain_nmi == 0.0
    assert result.minutes_to_gain == 0.0


def test_speed_control_advisor_returns_actual_and_equivalent_miles() -> None:
    manager = FakeManager([_arrival(flight_id="A", threshold_nmi=0.0)])

    result = SpeedControlAdvisor(manager, planner=FakePlanner()).advise(
        flight_id="A",
        s_m=50_000.0,
        cas_kts=180.0,
    )

    assert result.miles_to_gain_nmi == 0.0
    assert result.minutes_to_gain == pytest.approx(2.0)
    assert result.equivalent_vectoring_miles_nmi == pytest.approx(120.0 * 100.0 / METERS_PER_NM)


def test_speed_control_advisor_rejects_invalid_station() -> None:
    manager = FakeManager([_arrival(flight_id="A", threshold_nmi=0.0)])

    with pytest.raises(ValueError, match="s_m must lie within"):
        SpeedControlAdvisor(manager, planner=FakePlanner()).advise(
            flight_id="A",
            s_m=150_000.0,
            cas_kts=180.0,
        )


def test_speed_control_advisor_rejects_faster_or_equal_speed() -> None:
    manager = FakeManager([_arrival(flight_id="A", threshold_nmi=0.0)])

    with pytest.raises(ValueError, match="not lower than base profile"):
        SpeedControlAdvisor(manager, planner=FakePlanner()).advise(
            flight_id="A",
            s_m=50_000.0,
            cas_kts=240.0,
        )
