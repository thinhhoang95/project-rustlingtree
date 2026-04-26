from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PathWaypoint:
    identifier: str
    lat_deg: float
    lon_deg: float
    source: str = "fix"
    elevation_ft: float | None = None


@dataclass(frozen=True)
class AltitudeConstraint:
    fix_identifier: str
    lower_ft: float | None = None
    upper_ft: float | None = None

    def __post_init__(self) -> None:
        if self.lower_ft is None and self.upper_ft is None:
            raise ValueError("at least one altitude bound is required")
        if self.lower_ft is not None and self.upper_ft is not None and self.lower_ft > self.upper_ft:
            raise ValueError("altitude lower bound must not exceed upper bound")


@dataclass(frozen=True)
class TacticalCondition:
    fix_identifier: str
    cas_kts: float
    altitude_ft: float
    gamma_deg: float = 0.0


@dataclass(frozen=True)
class TacticalCommand:
    lateral_path: str | list[str | tuple[float, float]]
    upstream: TacticalCondition
    altitude_constraints: tuple[AltitudeConstraint, ...] = field(default_factory=tuple)
    runway_gamma_deg: float = -3.0
    runway_altitude_ft: float | None = None


@dataclass(frozen=True)
class ResolvedTacticalPath:
    waypoints: tuple[PathWaypoint, ...]

    @property
    def identifiers(self) -> tuple[str, ...]:
        return tuple(waypoint.identifier for waypoint in self.waypoints)

    @property
    def lat_deg(self) -> list[float]:
        return [waypoint.lat_deg for waypoint in self.waypoints]

    @property
    def lon_deg(self) -> list[float]:
        return [waypoint.lon_deg for waypoint in self.waypoints]


@dataclass(frozen=True)
class TacticalPlanBundle:
    command: TacticalCommand
    path: ResolvedTacticalPath
    request: object
    plan: object | None = None
    simulation: object | None = None
