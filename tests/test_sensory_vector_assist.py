from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_tools.sensory.models import VectorAssistRequest
from mcp_tools.sensory.operational_space import build_mask, mask_for_cluster
from mcp_tools.sensory.vector_assist import (
    build_dogleg_route,
    has_tight_turn,
    project_candidate_to_route,
    vector_assist,
)
from simap.nlp_colloc.tactical.models import PathWaypoint
from simap.nlp_colloc.tactical.navdata import load_fix_catalog


def _waypoint(identifier: str, lat: float, lon: float, source: str = "fix") -> PathWaypoint:
    return PathWaypoint(identifier=identifier, lat_deg=lat, lon_deg=lon, source=source)


def _catalog() -> dict[str, PathWaypoint]:
    return {
        "TTT": _waypoint("TTT", 0.0, 0.0),
        "WLLTR": _waypoint("WLLTR", 1.0, -1.0),
        "PRX": _waypoint("PRX", 1.0, 1.0),
        "BGTOE": _waypoint("BGTOE", -1.0, -1.0),
        "WAITT": _waypoint("WAITT", -1.0, 1.0),
        "MID": _waypoint("MID", 0.35, 0.0),
        "GOOD": _waypoint("GOOD", 0.48, 0.12),
        "FINAL": _waypoint("FINAL", 0.75, 0.0),
        "RW01": _waypoint("RW01", 1.0, 0.0, source="runway"),
    }


def _write_catalog(path: Path) -> Path:
    fixes_path = path / "fixes.csv"
    rows = [
        "identifier,latitude_deg,longitude_deg,fix_type,elevation_ft",
        "TTT,0.0,0.0,fix,",
        "WLLTR,1.0,-1.0,fix,",
        "PRX,1.0,1.0,fix,",
        "BGTOE,-1.0,-1.0,fix,",
        "WAITT,-1.0,1.0,fix,",
        "MID,0.35,0.0,fix,",
        "GOOD,0.48,0.12,fix,",
        "FINAL,0.75,0.0,fix,",
        "RW01,1.0,0.0,runway,",
    ]
    fixes_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return fixes_path


def _arrival(path_stretch: dict[str, Any] | None = None) -> dict[str, Any]:
    arrival = {
        "flight_id": "ARR1",
        "callsign": "CALL1",
        "icao24": "icao1",
        "runway": "RW01",
        "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
        "points": [[0, 0.0, 0.0, 1_000.0, 3], [100, 1.0, 0.0, 200.0, 3]],
        "cas_profile": {"columns": ["time", "cas_kts"], "points": [[0, 180.0], [2, 178.0]]},
        "first_time": 0,
        "last_time": 100,
        "base_route": {
            "lateral_path": ["TTT", "MID", "FINAL", "RW01"],
            "fix_sequence": "TTT>MID>FINAL>RW01",
        },
        "atc_wait_point": {"arrival_cluster": "NE", "identifier": "MID"},
    }
    if path_stretch is not None:
        arrival["path_stretch"] = path_stretch
    return arrival


class _Manager:
    def __init__(self, fixes_path: Path, arrivals: list[dict[str, Any]]) -> None:
        self.config = SimpleNamespace(fixes_path=fixes_path)
        self._arrivals = arrivals
        self.diff: list[dict[str, Any]] = []
        self.path_stretch_drafts: dict[str, Any] = {}

    def arrival_schedule(self) -> list[dict[str, Any]]:
        return [dict(arrival) for arrival in self._arrivals]


def test_operational_masks_use_named_boundary_fixes() -> None:
    real_catalog = load_fix_catalog("data/kdfw_procs/airport_related_fixes.csv")

    north = build_mask("north", real_catalog)
    south = build_mask("south", real_catalog)

    assert north.boundary_fix_identifiers == ("TTT", "WLLTR", "PRX")
    assert south.boundary_fix_identifiers == ("TTT", "BGTOE", "WAITT")
    assert mask_for_cluster("NE", real_catalog).name == "north"
    assert mask_for_cluster("NW", real_catalog).name == "north"
    assert mask_for_cluster("SE", real_catalog).name == "south"
    assert mask_for_cluster("SW", real_catalog).name == "south"
    assert north.contains(real_catalog["TTT"].lat_deg, real_catalog["TTT"].lon_deg)
    assert south.contains(real_catalog["WAITT"].lat_deg, real_catalog["WAITT"].lon_deg)


def test_route_projection_and_dogleg_construction() -> None:
    route = ["A", "B", "C"]
    points = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]

    segment_index = project_candidate_to_route(points, 0.0, 1.2)

    assert segment_index == 1
    assert build_dogleg_route(route, "X", projected_segment_index=1, variant="sandwiched_dogleg") == [
        "A",
        "B",
        "X",
        "C",
    ]
    assert build_dogleg_route(route, "X", projected_segment_index=1, variant="replaced_dogleg") == [
        "A",
        "X",
        "C",
    ]
    with pytest.raises(ValueError, match="first route segment"):
        build_dogleg_route(route, "X", projected_segment_index=0, variant="replaced_dogleg")


def test_tight_turn_detector_rejects_u_turn_geometry() -> None:
    assert has_tight_turn([(0.0, 0.0), (0.0, 1.0), (0.0, 0.1)])
    assert not has_tight_turn([(0.0, 0.0), (0.0, 1.0), (0.4, 1.5)])


def test_vector_assist_scores_candidates_and_stays_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _Manager(_write_catalog(tmp_path), [_arrival()])

    def fake_simulate(_manager: Any, _arrival_payload: dict[str, Any], _route: list[Any], **kwargs: Any) -> Any:
        metadata = kwargs["vector_assist"]
        if metadata.get("fix_identifier") == "GOOD":
            actual_s = 52.0
        elif metadata["candidate_kind"] == "free":
            actual_s = 55.0
        else:
            actual_s = 5.0
        return SimpleNamespace(
            metrics={
                "old_distance_nm": 1.0,
                "new_distance_nm": 2.0,
                "delta_distance_nm": 1.0,
                "old_elapsed_min": 1.0,
                "new_elapsed_min": 1.0 + actual_s / 60.0,
                "delta_elapsed_min": actual_s / 60.0,
            },
            simulation={"success": True, "message": "ok"},
        )

    monkeypatch.setattr("mcp_tools.sensory.vector_assist.simulate_path_stretch_route", fake_simulate)

    response = vector_assist(
        manager,
        VectorAssistRequest(
            flight_id="ARR1",
            target_time_gain_s=50.0,
            grid_spacing_nm=30.0,
            max_exact_candidates=12,
        ),
    )

    assert response.best_identified_candidate is not None
    assert response.best_free_candidate is not None
    assert response.best_identified_candidate.fix_identifier == "GOOD"
    assert response.recommendation is not None
    assert response.recommendation.candidate_kind == "identified"
    assert response.recommendation.path_stretch_request["vector_assist"]["target_time_gain_s"] == 50.0
    assert manager.diff == []
    assert manager.path_stretch_drafts == {}


def test_vector_assist_respects_attempt_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {
        "vector_assist": {
            "attempts": [
                {"variant": "sandwiched_dogleg"},
                {"variant": "replaced_dogleg"},
            ]
        }
    }
    manager = _Manager(_write_catalog(tmp_path), [_arrival(path_stretch=attempts)])

    with pytest.raises(ValueError, match="maximum of 2"):
        vector_assist(manager, VectorAssistRequest(flight_id="ARR1", target_time_gain_s=50.0, grid_spacing_nm=30.0))

    one_replaced = {"vector_assist": {"attempts": [{"variant": "replaced_dogleg"}]}}
    manager = _Manager(_write_catalog(tmp_path), [_arrival(path_stretch=one_replaced)])
    monkeypatch.setattr(
        "mcp_tools.sensory.vector_assist.simulate_path_stretch_route",
        lambda *_args, **_kwargs: SimpleNamespace(
            metrics={"delta_elapsed_min": 1.0},
            simulation={"success": True, "message": "ok"},
        ),
    )

    response = vector_assist(manager, VectorAssistRequest(flight_id="ARR1", target_time_gain_s=50.0, grid_spacing_nm=30.0))

    assert response.attempt_status.replaced_dogleg_used
    assert response.recommendation is None or response.recommendation.variant == "sandwiched_dogleg"
    assert response.rejected_counts["replaced_dogleg_unavailable"] > 0
