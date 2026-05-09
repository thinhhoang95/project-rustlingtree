from __future__ import annotations

import pytest

from mcp_tools.scenario_manager import wait_atc_point
from simap.nlp_colloc.tactical.models import PathWaypoint


def _waypoint(identifier: str, lat_deg: float, lon_deg: float, source: str = "fix") -> PathWaypoint:
    return PathWaypoint(identifier=identifier, lat_deg=lat_deg, lon_deg=lon_deg, source=source)


def _catalog() -> dict[str, PathWaypoint]:
    return {
        "RW35C": _waypoint("RW35C", 32.87887777777778, -97.02617222222221, source="runway"),
        "FAR_NE": _waypoint("FAR_NE", 33.75, -96.10),
        "BRDJE": _waypoint("BRDJE", 33.32655277777778, -96.42285555555556),
        "COVIE": _waypoint("COVIE", 33.275, -96.49514722222223),
        "FAR_NW": _waypoint("FAR_NW", 33.75, -98.20),
        "VKTRY": _waypoint("VKTRY", 33.3839, -97.64171111111112),
        "GREGS": _waypoint("GREGS", 33.450608333333335, -97.46605555555556),
        "FAR_SW": _waypoint("FAR_SW", 31.90, -98.00),
        "HODAX": _waypoint("HODAX", 32.97505833333334, -97.15059166666667),
        "BEMMR": _waypoint("BEMMR", 33.00171944444445, -97.27038611111111),
        "FAR_SE": _waypoint("FAR_SE", 32.20, -96.20),
        "BORDD": _waypoint("BORDD", 33.013780555555556, -96.92275277777779),
        "PITHY": _waypoint("PITHY", 33.07370277777778, -96.76625833333334),
        "OUTSIDE": _waypoint("OUTSIDE", 32.4, -97.7),
    }


@pytest.mark.parametrize(
    ("route", "cluster", "identifier"),
    [
        (["FAR_NE", "BRDJE", "COVIE", "RW35C"], "NE", "COVIE"),
        (["FAR_NW", "VKTRY", "GREGS", "RW35C"], "NW", "GREGS"),
        (["FAR_SW", "HODAX", "BEMMR", "RW35C"], "SW", "BEMMR"),
        (["FAR_SE", "BORDD", "PITHY", "RW35C"], "SE", "PITHY"),
    ],
)
def test_detect_wait_atc_point_classifies_by_gate_and_chooses_last_fix_inside_capture(
    route: list[str],
    cluster: str,
    identifier: str,
) -> None:
    point = wait_atc_point.detect_wait_atc_point(route, _catalog(), runway="35C")

    assert point is not None
    assert point["source"] == "fix"
    assert point["identifier"] == identifier
    assert point["lateral_path_token"] == identifier
    assert point["selection_method"] == "cluster_capture_polygon"
    assert point["arrival_cluster"] == cluster
    assert point["gate_cluster"] == cluster
    assert point["gate_radius_nm"] == 50.0
    assert point["ring_inner_nm"] == 50.0
    assert point["ring_outer_nm"] == 50.0
    assert point["capture_margin_nm"] == 3.0
    assert point["gate_classification_fallback"] is False


def test_detect_wait_atc_point_skips_coordinates_and_runways_as_selected_point() -> None:
    point = wait_atc_point.detect_wait_atc_point(
        ["FAR_SE", (33.02, -96.80), "PITHY", "RW35C"],
        _catalog(),
        runway="35C",
    )

    assert point is not None
    assert point["identifier"] == "PITHY"
    assert point["route_index"] == 2


def test_detect_wait_atc_point_accepts_boundary_fix_inside_capture_polygon() -> None:
    catalog = _catalog()
    catalog["NE_BOUNDARY"] = _waypoint("NE_BOUNDARY", 33.395008, -96.733883)

    point = wait_atc_point.detect_wait_atc_point(
        ["FAR_NE", "NE_BOUNDARY", "RW35C"],
        catalog,
        runway="35C",
    )

    assert point is not None
    assert point["identifier"] == "NE_BOUNDARY"
    assert point["arrival_cluster"] == "NE"


def test_detect_wait_atc_point_returns_none_and_reports_diagnostic_when_no_fix_is_inside_capture() -> None:
    diagnostics: list[str] = []

    point = wait_atc_point.detect_wait_atc_point(
        ["FAR_NW", "OUTSIDE", "RW35C"],
        _catalog(),
        runway="35C",
        diagnostics=diagnostics,
        trace_label="ARR1/CALL",
    )

    assert point is None
    assert diagnostics == [
        "ARR1/CALL: gate classified NW, but no route fix was inside the NW capture polygon: "
        "FAR_NW > OUTSIDE > RW35C"
    ]


def test_detect_wait_atc_point_rejects_inverted_ring_bounds() -> None:
    with pytest.raises(ValueError, match="ring_inner_nm"):
        wait_atc_point.detect_wait_atc_point(
            ["FAR_NE", "BRDJE", "RW35C"],
            _catalog(),
            runway="35C",
            ring_inner_nm=51.0,
            ring_outer_nm=50.0,
        )


def test_detect_wait_atc_point_rejects_non_positive_gate_radius() -> None:
    with pytest.raises(ValueError, match="gate_radius_nm"):
        wait_atc_point.detect_wait_atc_point(
            ["FAR_NE", "BRDJE", "RW35C"],
            _catalog(),
            runway="35C",
            gate_radius_nm=0.0,
        )
