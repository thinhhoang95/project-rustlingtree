from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from mcp_tools.scenario_manager.manager import ScenarioManager
from mcp_tools.scenario_manager.path_stretching import (
    PathStretchHandleRequest,
    PathStretchRouteTokenRequest,
    PathStretchSaveRequest,
    PathStretchSimulationRequest,
)
from tests.test_scenario_manager import write_fixture_resources


@dataclass(frozen=True)
class _FakeFmsRequest:
    start_s_m: float = 60_000.0
    atc_speed_segments: tuple = ()
    reference_path: object | None = None


def _fake_bichannel_result() -> SimpleNamespace:
    return SimpleNamespace(
        t_s=np.asarray([0.0, 60.0, 120.0], dtype=float),
        lat_deg=np.asarray([32.0, 32.18, 32.9], dtype=float),
        lon_deg=np.asarray([-97.0, -97.16, -97.0], dtype=float),
        h_m=np.asarray([1_000.0, 700.0, 200.0], dtype=float),
        v_cas_mps=np.asarray([100.0, 95.0, 90.0], dtype=float),
        success=True,
        message="ok",
        max_abs_cross_track_m=12.0,
        max_abs_track_error_rad=0.02,
        final_threshold_error_m=3.0,
    )


def _patch_simap(monkeypatch: pytest.MonkeyPatch) -> list[list[str | tuple[float, float]]]:
    import mcp_tools.scenario_manager.path_stretching as path_stretching
    import mcp_tools.scenario_manager.served_profile as served_profile

    captured_routes: list[list[str | tuple[float, float]]] = []

    def fake_build_request(**kwargs):
        captured_routes.append(list(kwargs["route"]))
        return _FakeFmsRequest(), SimpleNamespace()

    monkeypatch.setattr(served_profile, "_build_request", fake_build_request)
    monkeypatch.setattr(path_stretching, "plan_fms_bichannel", lambda *_args, **_kwargs: _fake_bichannel_result())
    return captured_routes


def test_path_stretch_simulate_uses_fix_and_coordinate_route_tokens(tmp_path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    captured_routes = _patch_simap(monkeypatch)

    response = manager.simulate_path_stretch(
        PathStretchSimulationRequest(
            flight_id="ARR1",
            handles=[
                PathStretchHandleRequest(
                    insert_after_index=1,
                    token_type="fix",
                    fix_identifier="FIXA",
                    lat=32.0,
                    lon=-97.0,
                ),
                PathStretchHandleRequest(
                    insert_after_index=1,
                    token_type="coordinate",
                    lat=32.2,
                    lon=-97.2,
                ),
            ],
        )
    )

    assert captured_routes == [["FIXA", "FIXB", "FIXA", (32.2, -97.2), "FINAL35C", "RW35C"]]
    assert response["new_route_tokens"] == [
        "FIXA",
        "FIXB",
        "FIXA",
        "32.20000,-97.20000",
        "FINAL35C",
        "RW35C",
    ]
    assert response["trajectory"]["base_route"]["lateral_path"] == [
        "FIXA",
        "FIXB",
        "FIXA",
        [32.2, -97.2],
        "FINAL35C",
        "RW35C",
    ]
    assert response["trajectory"]["route_type"] == "path-stretch"
    assert math.isclose(response["metrics"]["new_elapsed_min"], 2.0)


def test_path_stretch_simulate_accepts_edited_route_points(tmp_path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    captured_routes = _patch_simap(monkeypatch)

    response = manager.simulate_path_stretch(
        PathStretchSimulationRequest(
            flight_id="ARR1",
            route=[
                PathStretchRouteTokenRequest(
                    token_type="fix",
                    fix_identifier="FIXA",
                    lat=32.0,
                    lon=-97.0,
                ),
                PathStretchRouteTokenRequest(
                    token_type="coordinate",
                    lat=32.25,
                    lon=-97.25,
                ),
                PathStretchRouteTokenRequest(
                    token_type="fix",
                    fix_identifier="RW35C",
                    lat=33.0,
                    lon=-97.0,
                ),
            ],
        )
    )

    assert captured_routes == [["FIXA", (32.25, -97.25), "RW35C"]]
    assert response["new_route_tokens"] == ["FIXA", "32.25000,-97.25000", "RW35C"]
    assert response["trajectory"]["base_route"]["lateral_path"] == [
        "FIXA",
        [32.25, -97.25],
        "RW35C",
    ]
    assert response["trajectory"]["path_stretch"]["route_point_count"] == 3


def test_path_stretch_save_replaces_active_diff_and_served_arrival(tmp_path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    _patch_simap(monkeypatch)

    first = manager.simulate_path_stretch(
        PathStretchSimulationRequest(
            flight_id="ARR1",
            handles=[
                PathStretchHandleRequest(
                    insert_after_index=1,
                    token_type="coordinate",
                    lat=32.2,
                    lon=-97.2,
                )
            ],
        )
    )
    manager.save_path_stretch("ARR1", PathStretchSaveRequest(draft_id=first["draft_id"]))

    second = manager.simulate_path_stretch(
        PathStretchSimulationRequest(
            flight_id="ARR1",
            handles=[
                PathStretchHandleRequest(
                    insert_after_index=1,
                    token_type="coordinate",
                    lat=32.3,
                    lon=-97.3,
                )
            ],
        )
    )
    saved = manager.save_path_stretch("ARR1", PathStretchSaveRequest(draft_id=second["draft_id"]))

    assert len(manager.intervention_diff()) == 1
    assert saved["diff"]["id"] == second["draft_id"]
    arrival = manager.arrival_schedule()[0]
    assert arrival["route_type"] == "path-stretch"
    assert [32.3, -97.3] in arrival["base_route"]["lateral_path"]
    assert arrival["simulation"]["message"] == "ok"


def test_path_stretch_simulate_preserves_active_speed_intervention_context(tmp_path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    _patch_simap(monkeypatch)
    speed_advisory = {
        "s_m": 42_000.0,
        "station_nm_to_runway": 42_000.0 / 1_852.0,
        "cas_kts": 180.0,
        "lat": 32.1,
        "lon": -97.1,
    }
    base_route = dict(manager.arrival_schedule()[0]["base_route"])
    base_route["speed_advisories"] = [speed_advisory]
    manager.diff = [
        {
            "id": "active-speed",
            "flight_id": "ARR1",
            "type": "speed-intervention",
            "overrides": {
                "route_type": "speed-intervention",
                "base_route": base_route,
                "speed_intervention": {
                    "advisories": [speed_advisory],
                    "advisory_count": 1,
                },
            },
        }
    ]

    response = manager.simulate_path_stretch(
        PathStretchSimulationRequest(
            flight_id="ARR1",
            handles=[
                PathStretchHandleRequest(
                    insert_after_index=1,
                    token_type="coordinate",
                    lat=32.2,
                    lon=-97.2,
                )
            ],
        )
    )

    assert response["trajectory"]["path_stretch"]["handle_count"] == 1
    assert response["trajectory"]["speed_intervention"]["advisory_count"] == 1
    assert response["trajectory"]["base_route"]["speed_advisories"] == [speed_advisory]


def test_path_stretch_records_vector_assist_attempt_metadata(tmp_path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    _patch_simap(monkeypatch)

    response = manager.simulate_path_stretch(
        PathStretchSimulationRequest(
            flight_id="ARR1",
            route=[
                PathStretchRouteTokenRequest(token_type="fix", fix_identifier="FIXA", lat=32.0, lon=-97.0),
                PathStretchRouteTokenRequest(token_type="coordinate", lat=32.25, lon=-97.25),
                PathStretchRouteTokenRequest(token_type="fix", fix_identifier="RW35C", lat=33.0, lon=-97.0),
            ],
            vector_assist={
                "variant": "sandwiched_dogleg",
                "candidate_kind": "free",
                "target_time_gain_s": 60.0,
                "projected_segment_index": 1,
                "lat": 32.25,
                "lon": -97.25,
            },
        )
    )

    vector_assist = response["trajectory"]["path_stretch"]["vector_assist"]
    assert vector_assist["attempt_count"] == 1
    assert vector_assist["attempts"][0]["variant"] == "sandwiched_dogleg"
    assert vector_assist["attempts"][0]["target_time_gain_s"] == 60.0


def test_path_stretch_vector_assist_limits_apply_only_to_tagged_edits(tmp_path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    _patch_simap(monkeypatch)
    active_path_stretch = {
        "vector_assist": {
            "attempts": [
                {"variant": "sandwiched_dogleg"},
                {"variant": "replaced_dogleg"},
            ]
        }
    }
    manager.diff = [
        {
            "id": "active-vector",
            "flight_id": "ARR1",
            "type": "path-stretch",
            "overrides": {"path_stretch": active_path_stretch},
        }
    ]

    generic = manager.simulate_path_stretch(
        PathStretchSimulationRequest(
            flight_id="ARR1",
            handles=[PathStretchHandleRequest(insert_after_index=1, token_type="coordinate", lat=32.2, lon=-97.2)],
        )
    )
    assert generic["trajectory"]["path_stretch"]["handle_count"] == 1

    with pytest.raises(ValueError, match="maximum of 2"):
        manager.simulate_path_stretch(
            PathStretchSimulationRequest(
                flight_id="ARR1",
                handles=[
                    PathStretchHandleRequest(insert_after_index=1, token_type="coordinate", lat=32.3, lon=-97.3)
                ],
                vector_assist={
                    "variant": "sandwiched_dogleg",
                    "candidate_kind": "free",
                    "target_time_gain_s": 60.0,
                    "projected_segment_index": 1,
                    "lat": 32.3,
                    "lon": -97.3,
                },
            )
        )


def test_path_stretch_validation_rejects_bad_inputs(tmp_path, monkeypatch) -> None:
    manager = ScenarioManager(write_fixture_resources(tmp_path))
    _patch_simap(monkeypatch)

    with pytest.raises(ValueError, match="unknown arrival"):
        manager.simulate_path_stretch(
            PathStretchSimulationRequest(
                flight_id="DEP1",
                handles=[
                    PathStretchHandleRequest(
                        insert_after_index=0,
                        token_type="coordinate",
                        lat=32.0,
                        lon=-97.0,
                    )
                ],
            )
        )

    with pytest.raises(ValueError, match="unknown path-stretch fix"):
        manager.simulate_path_stretch(
            PathStretchSimulationRequest(
                flight_id="ARR1",
                handles=[
                    PathStretchHandleRequest(
                        insert_after_index=0,
                        token_type="fix",
                        fix_identifier="NOFIX",
                        lat=32.0,
                        lon=-97.0,
                    )
                ],
            )
        )

    with pytest.raises(ValueError, match="unknown path-stretch draft"):
        manager.save_path_stretch("ARR1", PathStretchSaveRequest(draft_id="missing"))
