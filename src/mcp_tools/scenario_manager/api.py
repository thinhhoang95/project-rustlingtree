from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request

from mcp_tools.scenario_manager.manager import ScenarioManager
from mcp_tools.scenario_manager.models import (
    ArrivalScheduleItem,
    DepartureScheduleItem,
    HealthResponse,
    ScenarioResourceConfig,
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

    @app.get("/diff", response_model=list[dict[str, object]])
    def diff(request: Request) -> list[dict[str, object]]:
        manager: ScenarioManager = request.app.state.scenario_manager
        return manager.intervention_diff()

    return app


app = create_app()
