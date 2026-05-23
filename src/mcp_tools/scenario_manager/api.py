from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from mcp_tools.advisors import AmanAdvisor, FeasibilityAdvisor, SpeedControlAdvisor, VectoringAdvisor
from mcp_tools.evaluators import ConflictEvaluator, FeasibleEvaluator, RunwayOverlapEvaluator
from mcp_tools.scenario_manager.manager import ScenarioManager
from mcp_tools.scenario_manager.models import (
    AmanAdvisoryItem,
    ArrivalScheduleItem,
    ConflictEvaluationItem,
    DepartureScheduleItem,
    FeasibilityAdvisoryItem,
    FeasibilityEvaluationItem,
    HealthResponse,
    RunwayOverlapEvaluationItem,
    ScenarioResourceConfig,
    SpeedControlAdvisoryItem,
    VectoringAdvisoryItem,
)
from mcp_tools.scenario_manager.path_stretching import (
    PathStretchSaveRequest,
    PathStretchSimulationRequest,
)
from mcp_tools.scenario_manager.speed_intervention import (
    SpeedInterventionSaveRequest,
    SpeedInterventionSimulationRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.scenario_manager = ScenarioManager(ScenarioResourceConfig.default())
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Project Rustlingtree Scenario Manager", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> dict[str, object]:
        manager: ScenarioManager = request.app.state.scenario_manager
        return manager.health()

    @app.get("/departures", response_model=list[DepartureScheduleItem])
    def departures(request: Request) -> list[dict[str, object]]:
        manager: ScenarioManager = request.app.state.scenario_manager
        return manager.departure_schedule()

    @app.get("/arrivals", response_model=list[ArrivalScheduleItem])
    def arrivals(request: Request) -> list[dict[str, object]]:
        manager: ScenarioManager = request.app.state.scenario_manager
        return manager.arrival_schedule()

    @app.get("/tools/evals/feasibility", response_model=list[FeasibilityEvaluationItem])
    def feasibility(request: Request) -> list[dict[str, object]]:
        manager: ScenarioManager = request.app.state.scenario_manager
        return [asdict(item) for item in FeasibleEvaluator(manager).evaluate()]

    @app.get("/tools/evals/conflicts", response_model=list[ConflictEvaluationItem])
    def conflicts(request: Request) -> list[dict[str, object]]:
        manager: ScenarioManager = request.app.state.scenario_manager
        return [asdict(item) for item in ConflictEvaluator(manager).evaluate()]

    @app.get("/tools/evals/runway-overlaps", response_model=list[RunwayOverlapEvaluationItem])
    def runway_overlaps(request: Request) -> list[dict[str, object]]:
        manager: ScenarioManager = request.app.state.scenario_manager
        return [asdict(item) for item in RunwayOverlapEvaluator(manager).evaluate()]

    @app.get("/tools/advisors/feasibility", response_model=list[FeasibilityAdvisoryItem])
    def advisory_feasibility(request: Request, flight_id: str | None = None) -> list[dict[str, object]]:
        """Answer: how much extra upstream distance is needed for vertical feasibility?"""
        manager: ScenarioManager = request.app.state.scenario_manager
        return [asdict(item) for item in FeasibilityAdvisor(manager).evaluate(flight_id=flight_id)]

    @app.get("/tools/advisors/aman", response_model=list[AmanAdvisoryItem])
    def advisory_aman(request: Request) -> list[dict[str, object]]:
        """Answer: which arrivals need delay to clear runway-use overlaps?"""
        manager: ScenarioManager = request.app.state.scenario_manager
        return [asdict(item) for item in AmanAdvisor(manager).evaluate()]

    @app.get("/tools/advisors/vectoring", response_model=VectoringAdvisoryItem)
    def advisory_vectoring(
        request: Request,
        flight_id: str,
        extra_distance_nmi: float,
    ) -> dict[str, object]:
        """Answer: what happens if this arrival gets N extra nautical miles?"""
        manager: ScenarioManager = request.app.state.scenario_manager
        return asdict(
            VectoringAdvisor(manager).advise(
                flight_id=flight_id,
                extra_distance_nmi=extra_distance_nmi,
            )
        )

    @app.get("/tools/advisors/speed-control", response_model=SpeedControlAdvisoryItem)
    def advisory_speed_control(
        request: Request,
        flight_id: str,
        s_m: float,
        cas_kts: float,
    ) -> dict[str, object]:
        """Answer: what happens if this arrival accepts one lower-CAS instruction?"""
        manager: ScenarioManager = request.app.state.scenario_manager
        return asdict(
            SpeedControlAdvisor(manager).advise(
                flight_id=flight_id,
                s_m=s_m,
                cas_kts=cas_kts,
            )
        )

    @app.get("/diff", response_model=list[dict[str, object]])
    def diff(request: Request) -> list[dict[str, object]]:
        """Answer: which trajectory edit diffs are currently active?"""
        manager: ScenarioManager = request.app.state.scenario_manager
        return manager.intervention_diff()

    @app.post("/tools/path-stretch/simulate", response_model=dict[str, object])
    def path_stretch_simulate(
        request: Request,
        body: PathStretchSimulationRequest,
    ) -> dict[str, object]:
        """Answer: what trajectory results from editing this arrival's lateral route?"""
        manager: ScenarioManager = request.app.state.scenario_manager
        try:
            return manager.simulate_path_stretch(body)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/diff/path-stretch/{flight_id}", response_model=dict[str, object])
    def path_stretch_save(
        request: Request,
        flight_id: str,
        body: PathStretchSaveRequest,
    ) -> dict[str, object]:
        """Answer: can this path-stretch draft become the active arrival trajectory?"""
        manager: ScenarioManager = request.app.state.scenario_manager
        try:
            return manager.save_path_stretch(flight_id, body)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/tools/speed-intervention/simulate", response_model=dict[str, object])
    def speed_intervention_simulate(
        request: Request,
        body: SpeedInterventionSimulationRequest,
    ) -> dict[str, object]:
        """Answer: what trajectory results from adding these speed advisories?"""
        manager: ScenarioManager = request.app.state.scenario_manager
        try:
            return manager.simulate_speed_intervention(body)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/diff/speed-intervention/{flight_id}", response_model=dict[str, object])
    def speed_intervention_save(
        request: Request,
        flight_id: str,
        body: SpeedInterventionSaveRequest,
    ) -> dict[str, object]:
        """Answer: can this speed-intervention draft become the active trajectory?"""
        manager: ScenarioManager = request.app.state.scenario_manager
        try:
            return manager.save_speed_intervention(flight_id, body)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
