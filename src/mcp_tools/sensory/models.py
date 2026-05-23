from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class VectorAssistRequest(BaseModel):
    flight_id: str
    target_time_gain_s: float
    grid_spacing_nm: float = 1.0
    identified_threshold_s: float = 10.0
    include_map: bool = False
    max_exact_candidates: int = 40


class VectorAssistMapCell(BaseModel):
    lat: float
    lon: float
    estimated_time_gain_s: float


class VectorAssistAttemptStatus(BaseModel):
    previous_attempt_count: int
    remaining_attempts: int
    replaced_dogleg_used: bool
    replaced_dogleg_available: bool


class VectorAssistCandidate(BaseModel):
    candidate_kind: Literal["identified", "free"]
    variant: Literal["sandwiched_dogleg", "replaced_dogleg"]
    fix_identifier: str | None = None
    lat: float
    lon: float
    projected_segment_index: int
    f_a: str
    f_b: str
    target_time_gain_s: float
    actual_time_gain_s: float
    error_s: float
    estimated_time_gain_s: float
    estimated_error_s: float
    simulation_success: bool
    simulation_message: str
    metrics: dict[str, float]
    path_stretch_request: dict[str, Any]


class VectorAssistResponse(BaseModel):
    flight_number: str
    icao24: str
    flight_id: str
    runway: str
    arrival_cluster: str
    operational_mask: str
    target_time_gain_s: float
    attempt_status: VectorAssistAttemptStatus
    best_identified_candidate: VectorAssistCandidate | None = None
    best_free_candidate: VectorAssistCandidate | None = None
    recommendation: VectorAssistCandidate | None = None
    rejected_counts: dict[str, int] = Field(default_factory=dict)
    evaluated_candidate_count: int
    map_cells: list[VectorAssistMapCell] | None = None
