from __future__ import annotations

import pytest

from mcp_tools.scenario_manager import wait_atc_point
from simap.nlp_colloc.tactical.models import PathWaypoint


NM_TO_M = wait_atc_point.NM_TO_M


def _waypoint(identifier: str, lat_deg: float, lon_deg: float, source: str = "fix") -> PathWaypoint:
    return PathWaypoint(identifier=identifier, lat_deg=lat_deg, lon_deg=lon_deg, source=source)


def _route_waypoint(identifier: str, distance_nm: float, bearing_deg: float) -> PathWaypoint:
    lat_deg, lon_deg = wait_atc_point._destination_latlon(0.0, 0.0, bearing_deg, distance_nm * NM_TO_M)
    return _waypoint(identifier, lat_deg, lon_deg)


def test_parse_runway_final_course_accepts_rw_prefix_and_wraps_36() -> None:
    assert wait_atc_point.parse_runway_final_course_deg("RW35C") == 350.0
    assert wait_atc_point.parse_runway_final_course_deg("36L") == 0.0


@pytest.mark.parametrize("runway", ["", "RW00", "RW37", "ABC"])
def test_parse_runway_final_course_rejects_invalid_designators(runway: str) -> None:
    with pytest.raises(ValueError):
        wait_atc_point.parse_runway_final_course_deg(runway)


def test_detect_wait_atc_point_chooses_last_downwind_fix_in_annulus() -> None:
    cand1 = _route_waypoint("CAND1", 42.0, 170.0)
    cand2 = _route_waypoint("CAND2", 45.0, 170.0)
    catalog = {
        "RW35C": _waypoint("RW35C", 0.01, 0.0, source="runway"),
        "RW17C": _waypoint("RW17C", -0.01, 0.0, source="runway"),
        "ENTRY": _route_waypoint("ENTRY", 39.0, 170.0),
        "CAND1": cand1,
        "CAND2": cand2,
    }

    point = wait_atc_point.detect_wait_atc_point(
        ["ENTRY", "CAND1", "CAND2", "RW35C"],
        catalog,
        runway="35C",
    )

    assert point["source"] == "fix"
    assert point["identifier"] == "CAND2"
    assert point["lateral_path_token"] == "CAND2"
    assert point["route_index"] == 2
    assert point["final_course_deg"] == 350.0
    assert point["downwind_course_deg"] == 170.0
    assert 40.0 <= point["distance_nm"] <= 50.0
    assert wait_atc_point._course_delta_deg(point["matched_course_deg"], 170.0) <= 45.0


def test_detect_wait_atc_point_filters_out_annulus_misses_and_returns_ghost() -> None:
    catalog = {
        "RW35C": _waypoint("RW35C", 0.0, 0.0, source="runway"),
        "NEAR": _route_waypoint("NEAR", 30.0, 170.0),
        "FAR": _route_waypoint("FAR", 55.0, 170.0),
    }

    point = wait_atc_point.detect_wait_atc_point(["NEAR", "FAR", "RW35C"], catalog, runway="35C")

    assert point["source"] == "ghost"
    assert point["identifier"] == wait_atc_point.GHOST_IDENTIFIER
    assert point["route_index"] is None
    assert point["matched_course_deg"] is None
    assert point["distance_nm"] == 40.0
    assert point["lateral_path_token"] == f"{point['lat']:.6f},{point['lon']:.6f}"


def test_detect_wait_atc_point_requires_downwind_route_heading() -> None:
    catalog = {
        "RW35C": _waypoint("RW35C", 0.0, 0.0, source="runway"),
        "EAST1": _route_waypoint("EAST1", 45.0, 90.0),
        "EAST2": _route_waypoint("EAST2", 42.0, 90.0),
    }

    point = wait_atc_point.detect_wait_atc_point(["EAST1", "EAST2", "RW35C"], catalog, runway="35C")

    assert point["source"] == "ghost"
