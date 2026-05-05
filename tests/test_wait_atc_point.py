from __future__ import annotations

import pytest

from mcp_tools.scenario_manager import wait_atc_point
from simap.nlp_colloc.tactical.models import PathWaypoint


def _waypoint(identifier: str, lat_deg: float, lon_deg: float, source: str = "fix") -> PathWaypoint:
    return PathWaypoint(identifier=identifier, lat_deg=lat_deg, lon_deg=lon_deg, source=source)


def test_parse_runway_final_course_accepts_rw_prefix_and_wraps_36() -> None:
    assert wait_atc_point.parse_runway_final_course_deg("RW35C") == 350.0
    assert wait_atc_point.parse_runway_final_course_deg("36L") == 0.0


@pytest.mark.parametrize("runway", ["", "RW00", "RW37", "ABC"])
def test_parse_runway_final_course_rejects_invalid_designators(runway: str) -> None:
    with pytest.raises(ValueError):
        wait_atc_point.parse_runway_final_course_deg(runway)


def test_detect_wait_atc_point_chooses_downstream_fix_of_final_downwind_segment() -> None:
    catalog = {
        "RW36L": _waypoint("RW36L", 0.0, 0.0, source="runway"),
        "ENTRY": _waypoint("ENTRY", 3.0, 0.0),
        "CAND1": _waypoint("CAND1", 2.0, 0.0),
        "CAND2": _waypoint("CAND2", 1.0, 0.0),
        "BASE": _waypoint("BASE", 1.0, 1.0),
    }

    point = wait_atc_point.detect_wait_atc_point(
        ["ENTRY", "CAND1", "CAND2", "BASE", "RW36L"],
        catalog,
        runway="36L",
    )

    assert point is not None
    assert point["source"] == "fix"
    assert point["identifier"] == "CAND2"
    assert point["lateral_path_token"] == "CAND2"
    assert point["route_index"] == 2
    assert point["final_course_deg"] == 0.0
    assert point["downwind_course_deg"] == 180.0
    assert wait_atc_point._course_delta_deg(point["matched_course_deg"], 180.0) <= 45.0


def test_detect_wait_atc_point_uses_last_downwind_leg_before_touchdown() -> None:
    catalog = {
        "RW36L": _waypoint("RW36L", 0.0, 0.0, source="runway"),
        "ENTRY": _waypoint("ENTRY", 5.0, 0.0),
        "EARLY": _waypoint("EARLY", 4.0, 0.0),
        "TURN": _waypoint("TURN", 4.0, 1.0),
        "DW1": _waypoint("DW1", 2.0, 1.0),
        "WAIT": _waypoint("WAIT", 1.0, 1.0),
        "BASE": _waypoint("BASE", 1.0, 0.0),
    }

    point = wait_atc_point.detect_wait_atc_point(
        ["ENTRY", "EARLY", "TURN", "DW1", "WAIT", "BASE", "RW36L"],
        catalog,
        runway="36L",
    )

    assert point is not None
    assert point["identifier"] == "WAIT"
    assert point["route_index"] == 4


def test_detect_wait_atc_point_returns_none_without_downwind_segment() -> None:
    catalog = {
        "RW36L": _waypoint("RW36L", 0.0, 0.0, source="runway"),
        "EAST1": _waypoint("EAST1", 1.0, 0.0),
        "EAST2": _waypoint("EAST2", 1.0, 1.0),
    }

    point = wait_atc_point.detect_wait_atc_point(["EAST1", "EAST2", "RW36L"], catalog, runway="36L")

    assert point is None


def test_detect_wait_atc_point_respects_configurable_heading_tolerance() -> None:
    catalog = {
        "RW36L": _waypoint("RW36L", 0.0, 0.0, source="runway"),
        "ANGLED": _waypoint("ANGLED", 1.0, 0.0),
        "WAIT": _waypoint("WAIT", 0.0, 1.0),
        "BASE": _waypoint("BASE", 0.0, 0.5),
    }

    strict_point = wait_atc_point.detect_wait_atc_point(
        ["ANGLED", "WAIT", "BASE", "RW36L"],
        catalog,
        runway="36L",
        heading_tolerance_deg=30.0,
    )
    loose_point = wait_atc_point.detect_wait_atc_point(
        ["ANGLED", "WAIT", "BASE", "RW36L"],
        catalog,
        runway="36L",
        heading_tolerance_deg=50.0,
    )

    assert strict_point is None
    assert loose_point is not None
    assert loose_point["identifier"] == "WAIT"
