from __future__ import annotations

import pytest

from mcp_tools.scenario_manager import wait_atc_point
from simap.nlp_colloc.tactical.models import PathWaypoint


def _waypoint(identifier: str, lat_deg: float, lon_deg: float, source: str = "fix") -> PathWaypoint:
    return PathWaypoint(identifier=identifier, lat_deg=lat_deg, lon_deg=lon_deg, source=source)


def test_detect_wait_atc_point_chooses_earliest_fix_inside_default_ring() -> None:
    catalog = {
        "RW36L": _waypoint("RW36L", 0.0, 0.0, source="runway"),
        "BEFORE": _waypoint("BEFORE", 0.75, 0.0),
        "FIRST": _waypoint("FIRST", 0.65, 0.0),
        "SECOND": _waypoint("SECOND", 0.60, 0.0),
    }

    point = wait_atc_point.detect_wait_atc_point(
        ["BEFORE", "FIRST", "SECOND", "RW36L"],
        catalog,
        runway="36L",
    )

    assert point is not None
    assert point["source"] == "fix"
    assert point["identifier"] == "FIRST"
    assert point["lateral_path_token"] == "FIRST"
    assert point["route_index"] == 1
    assert point["ring_inner_nm"] == 35.0
    assert point["ring_outer_nm"] == 40.0
    assert 35.0 <= point["distance_nm"] <= 40.0


def test_detect_wait_atc_point_returns_none_when_no_fix_is_inside_ring() -> None:
    catalog = {
        "RW36L": _waypoint("RW36L", 0.0, 0.0, source="runway"),
        "TOO_FAR": _waypoint("TOO_FAR", 0.75, 0.0),
        "TOO_CLOSE": _waypoint("TOO_CLOSE", 0.50, 0.0),
    }

    point = wait_atc_point.detect_wait_atc_point(["TOO_FAR", "TOO_CLOSE", "RW36L"], catalog, runway="36L")

    assert point is None


def test_detect_wait_atc_point_skips_coordinate_waypoints_in_ring() -> None:
    catalog = {
        "RW36L": _waypoint("RW36L", 0.0, 0.0, source="runway"),
        "WAIT": _waypoint("WAIT", 0.62, 0.0),
    }

    point = wait_atc_point.detect_wait_atc_point([(0.60, 0.0), "WAIT", "RW36L"], catalog, runway="36L")

    assert point is not None
    assert point["identifier"] == "WAIT"
    assert point["route_index"] == 1


def test_detect_wait_atc_point_can_use_custom_ring_bounds() -> None:
    catalog = {
        "RW36L": _waypoint("RW36L", 0.0, 0.0, source="runway"),
        "NEAR": _waypoint("NEAR", 0.50, 0.0),
        "DEFAULT": _waypoint("DEFAULT", 0.60, 0.0),
    }

    point = wait_atc_point.detect_wait_atc_point(
        ["NEAR", "DEFAULT", "RW36L"],
        catalog,
        runway="36L",
        ring_inner_nm=29.0,
        ring_outer_nm=31.0,
    )

    assert point is not None
    assert point["identifier"] == "NEAR"
    assert point["ring_inner_nm"] == 29.0
    assert point["ring_outer_nm"] == 31.0


def test_detect_wait_atc_point_rejects_inverted_ring_bounds() -> None:
    catalog = {
        "RW36L": _waypoint("RW36L", 0.0, 0.0, source="runway"),
        "WAIT": _waypoint("WAIT", 0.60, 0.0),
    }

    with pytest.raises(ValueError, match="ring_inner_nm"):
        wait_atc_point.detect_wait_atc_point(
            ["WAIT", "RW36L"],
            catalog,
            runway="36L",
            ring_inner_nm=40.0,
            ring_outer_nm=35.0,
        )
