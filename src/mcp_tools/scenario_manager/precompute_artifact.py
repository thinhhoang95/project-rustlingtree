from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from openap import aero

from mcp_tools.scenario_manager.models import project_root
from scenario.trajectory_compressor.algorithms import compress_breakpoints
from scenario.trajectory_compressor.compressor import ALTITUDE_BREAKPOINT_MASK, LATERAL_BREAKPOINT_MASK
from scenario.trajectory_compressor.io import load_raw_adsb, split_tracks_by_gap, write_jsonl
from simap.fms import FMSRequest
from simap.fms_bichannel import FMSBiChannelRequest, FMSBiChannelState, plan_fms_bichannel
from simap.lateral_dynamics import LateralGuidanceConfig, wrap_angle_rad
from simap.nlp_colloc.tactical import TacticalCommand, TacticalCondition
from simap.nlp_colloc.tactical.builder import build_tactical_plan_request
from simap.nlp_colloc.tactical.models import PathWaypoint
from simap.nlp_colloc.tactical.navdata import load_fix_catalog
from simap.openap_adapter import openap_dT
from simap.path_geometry import EARTH_RADIUS_M
from simap.units import m_to_ft, mps_to_kts

from mcp_tools.scenario_manager.wait_atc_point import detect_wait_atc_point

DEFAULT_EVENTS_PATH = Path("data/adsb/catalogs/2026-04-01_landings_and_departures.csv")
DEFAULT_FIX_SEQUENCES_PATH = Path("data/adsb/catalogs/2026-04-01_fix_sequences.csv")
DEFAULT_RAW_ADSB_DIR = Path("data/adsb/raw")
DEFAULT_FIXES_CSV = Path("data/kdfw_procs/airport_related_fixes.csv")
DEFAULT_OUTPUT_DIR = Path("data/artifacts")
OUTPUT_FLIGHTS_FILENAME = "simap_arrival_flights.jsonl"
DEFAULT_LATERAL_TOLERANCE_M = 100.0
DEFAULT_ALTITUDE_TOLERANCE_M = 50.0
DEFAULT_SPLIT_GAP_SECONDS = 25 * 60
DEFAULT_FINAL_FIX_DISTANCE_NM = 7.0
DEFAULT_FINAL_FIX_CROSS_TRACK_TOLERANCE_NM = 0.15
METERS_PER_NM = 1_852.0
_RUNWAY_RE = re.compile(r"^RW?(?P<number>\d{1,2})(?P<suffix>[LCR]?)$")


@dataclass(frozen=True)
class SeedState:
    time_s: int
    lat_deg: float
    lon_deg: float
    geoaltitude_m: float
    heading_deg: float | None
    ground_speed_mps: float


@dataclass(frozen=True)
class ArtifactResult:
    flight_id: str
    callsign: str
    icao24: str
    first_time: int | None
    last_time: int | None
    raw_point_count: int
    compressed_point_count: int
    lateral_breakpoint_count: int
    altitude_breakpoint_count: int
    status: str
    reason: str | None = None
    wait_atc_point_found: bool = False
    wait_atc_cluster: str | None = None
    simulation_success: bool | None = None
    simulation_message: str | None = None

    @property
    def compression_ratio(self) -> float:
        if self.raw_point_count == 0:
            return 0.0
        return self.compressed_point_count / self.raw_point_count


@dataclass(frozen=True)
class ArtifactTask:
    row: dict[str, Any]
    raw_flight: pd.DataFrame | None
    fixes_csv: Path
    lateral_tolerance_m: float
    altitude_tolerance_m: float
    final_fix_distance_nm: float
    final_fix_cross_track_tolerance_nm: float


@dataclass(frozen=True)
class FinalFixSelection:
    waypoint: PathWaypoint
    distance_nm: float
    along_track_nm: float
    cross_track_nm: float
    runway_true_heading_deg: float


@dataclass(frozen=True)
class BaseRoute:
    lateral_path: list[str | tuple[float, float]]
    upstream_identifier: str
    runway_identifier: str
    final_fix: FinalFixSelection
    atc_point: dict[str, Any]
    target_final_fix_distance_nm: float
    final_fix_cross_track_tolerance_nm: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "base-route",
            "selection_method": "atc_direct_to_runway_aligned_final_fix",
            "lateral_path": [
                list(token) if isinstance(token, tuple) else token
                for token in self.lateral_path
            ],
            "upstream_identifier": self.upstream_identifier,
            "runway": self.runway_identifier,
            "atc_point": self.atc_point,
            "final_fix": {
                "identifier": self.final_fix.waypoint.identifier,
                "lat": float(self.final_fix.waypoint.lat_deg),
                "lon": float(self.final_fix.waypoint.lon_deg),
                "distance_nm": float(self.final_fix.distance_nm),
                "along_track_nm": float(self.final_fix.along_track_nm),
                "cross_track_nm": float(self.final_fix.cross_track_nm),
                "target_distance_nm": float(self.target_final_fix_distance_nm),
                "cross_track_tolerance_nm": float(self.final_fix_cross_track_tolerance_nm),
            },
            "runway_true_heading_deg": float(self.final_fix.runway_true_heading_deg),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute simulated scenario-manager arrival artifacts.")
    parser.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--fix-sequences-path", type=Path, default=DEFAULT_FIX_SEQUENCES_PATH)
    parser.add_argument("--raw-adsb-dir", type=Path, default=DEFAULT_RAW_ADSB_DIR)
    parser.add_argument("--fixes-csv", type=Path, default=DEFAULT_FIXES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lateral-tolerance-m", type=float, default=DEFAULT_LATERAL_TOLERANCE_M)
    parser.add_argument("--altitude-tolerance-m", type=float, default=DEFAULT_ALTITUDE_TOLERANCE_M)
    parser.add_argument("--final-fix-distance-nm", type=float, default=DEFAULT_FINAL_FIX_DISTANCE_NM)
    parser.add_argument(
        "--final-fix-cross-track-tolerance-nm",
        type=float,
        default=DEFAULT_FINAL_FIX_CROSS_TRACK_TOLERANCE_NM,
    )
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    parser.add_argument("--processes", type=int, default=cpu_count())
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _abs_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _arrival_rows(events_path: Path, fix_sequences_path: Path) -> pd.DataFrame:
    events = pd.read_csv(events_path)
    sequences = pd.read_csv(fix_sequences_path)
    events["flight_id"] = events["flight_id"].astype(str).str.strip()
    sequences["flight_id"] = sequences["flight_id"].astype(str).str.strip()
    arrivals = events.loc[events["operation"].astype(str).str.lower().eq("arrival")].copy()
    return arrivals.merge(
        sequences.loc[:, ["flight_id", "first_time", "last_time", "fix_sequence", "fix_count"]],
        on="flight_id",
        how="left",
    )


def _route_tokens(fix_sequence: object, runway: object) -> list[str]:
    tokens = [token.strip().upper() for token in str(fix_sequence).split(">") if token.strip()]
    tokens = [token for token in tokens if token != "NAN"]
    runway_token = _normalize_runway_identifier(runway)
    if runway_token not in tokens:
        tokens.append(runway_token)
    return tokens


def _normalize_runway_identifier(runway: object) -> str:
    runway_text = str(runway).strip().upper()
    if not runway_text or runway_text == "NAN":
        raise ValueError("missing runway")
    if not runway_text.startswith("RW"):
        runway_text = f"RW{runway_text}"
    match = _RUNWAY_RE.match(runway_text)
    if match is None:
        raise ValueError(f"invalid runway identifier: {runway!r}")
    number = int(match.group("number"))
    if number < 1 or number > 36:
        raise ValueError(f"invalid runway number: {runway!r}")
    return f"RW{number:02d}{match.group('suffix')}"


def _reciprocal_runway_identifier(runway_identifier: str) -> str:
    match = _RUNWAY_RE.match(runway_identifier)
    if match is None:
        raise ValueError(f"invalid runway identifier: {runway_identifier!r}")
    number = int(match.group("number"))
    suffix = match.group("suffix")
    reciprocal_number = ((number + 18 - 1) % 36) + 1
    reciprocal_suffix = {"L": "R", "R": "L"}.get(suffix, suffix)
    return f"RW{reciprocal_number:02d}{reciprocal_suffix}"


def _bearing_deg(
    *,
    origin: PathWaypoint,
    destination: PathWaypoint,
) -> float:
    east_m, north_m = _local_ne_from_origin(
        origin_lat_deg=origin.lat_deg,
        origin_lon_deg=origin.lon_deg,
        lat_deg=destination.lat_deg,
        lon_deg=destination.lon_deg,
    )
    if abs(east_m) < 1e-9 and abs(north_m) < 1e-9:
        raise ValueError("cannot compute bearing between coincident points")
    return float((math.degrees(math.atan2(east_m, north_m)) + 360.0) % 360.0)


def _runway_true_heading_deg(
    runway_identifier: str,
    fix_catalog: dict[str, PathWaypoint],
) -> float:
    runway = fix_catalog.get(runway_identifier)
    if runway is None:
        raise KeyError(f"unknown runway fix: {runway_identifier}")
    reciprocal = fix_catalog.get(_reciprocal_runway_identifier(runway_identifier))
    if reciprocal is not None:
        return _bearing_deg(origin=runway, destination=reciprocal)

    match = _RUNWAY_RE.match(runway_identifier)
    if match is None:
        raise ValueError(f"invalid runway identifier: {runway_identifier!r}")
    number = int(match.group("number"))
    return float(360.0 if number == 36 else number * 10.0)


def _local_ne_from_origin(
    *,
    origin_lat_deg: float,
    origin_lon_deg: float,
    lat_deg: float,
    lon_deg: float,
) -> tuple[float, float]:
    lat0_rad = math.radians(float(origin_lat_deg))
    east_m = EARTH_RADIUS_M * math.cos(lat0_rad) * math.radians(float(lon_deg) - float(origin_lon_deg))
    north_m = EARTH_RADIUS_M * math.radians(float(lat_deg) - float(origin_lat_deg))
    return float(east_m), float(north_m)


def _select_final_fix(
    *,
    runway_identifier: str,
    fix_catalog: dict[str, PathWaypoint],
    target_distance_nm: float,
    cross_track_tolerance_nm: float,
) -> FinalFixSelection:
    if target_distance_nm <= 0.0:
        raise ValueError("final-fix target distance must be positive")
    if cross_track_tolerance_nm <= 0.0:
        raise ValueError("final-fix cross-track tolerance must be positive")

    runway = fix_catalog.get(runway_identifier)
    if runway is None:
        raise KeyError(f"unknown runway fix: {runway_identifier}")
    runway_heading_deg = _runway_true_heading_deg(runway_identifier, fix_catalog)
    outbound_heading_rad = math.radians((runway_heading_deg + 180.0) % 360.0)
    outbound_east = math.sin(outbound_heading_rad)
    outbound_north = math.cos(outbound_heading_rad)

    candidates: list[tuple[float, float, float, FinalFixSelection]] = []
    for waypoint in fix_catalog.values():
        identifier = waypoint.identifier.upper()
        if identifier.startswith("RW"):
            continue
        east_m, north_m = _local_ne_from_origin(
            origin_lat_deg=runway.lat_deg,
            origin_lon_deg=runway.lon_deg,
            lat_deg=waypoint.lat_deg,
            lon_deg=waypoint.lon_deg,
        )
        along_track_m = east_m * outbound_east + north_m * outbound_north
        if along_track_m <= 0.0:
            continue
        cross_track_m = east_m * outbound_north - north_m * outbound_east
        distance_m = math.hypot(east_m, north_m)
        distance_nm = distance_m / METERS_PER_NM
        along_track_nm = along_track_m / METERS_PER_NM
        cross_track_nm = cross_track_m / METERS_PER_NM
        if abs(cross_track_nm) > cross_track_tolerance_nm:
            continue

        selection = FinalFixSelection(
            waypoint=waypoint,
            distance_nm=float(distance_nm),
            along_track_nm=float(along_track_nm),
            cross_track_nm=float(cross_track_nm),
            runway_true_heading_deg=runway_heading_deg,
        )
        candidates.append(
            (
                abs(along_track_nm - target_distance_nm),
                abs(cross_track_nm),
                distance_nm,
                selection,
            )
        )

    if not candidates:
        raise ValueError(
            f"no final fix for {runway_identifier} within "
            f"{cross_track_tolerance_nm:.2f} NM of the extended runway centerline"
        )
    return min(candidates, key=lambda item: item[:3])[3]


def _wait_atc_route_token(
    wait_atc_point: dict[str, Any],
    fix_catalog: dict[str, PathWaypoint],
) -> str | tuple[float, float]:
    token = str(wait_atc_point.get("lateral_path_token") or wait_atc_point.get("identifier") or "").strip().upper()
    if token and token in fix_catalog:
        return token
    return float(wait_atc_point["lat"]), float(wait_atc_point["lon"])


def _dedupe_consecutive_route_tokens(
    route: list[str | tuple[float, float]],
) -> list[str | tuple[float, float]]:
    deduped: list[str | tuple[float, float]] = []
    for token in route:
        if deduped and deduped[-1] == token:
            continue
        deduped.append(token)
    return deduped


def _upstream_identifier_for_route(route: list[str | tuple[float, float]]) -> str:
    first = route[0]
    if isinstance(first, tuple):
        return "COORD01"
    return str(first).upper()


def _build_base_route(
    *,
    row: pd.Series,
    wait_atc_point: dict[str, Any],
    fix_catalog: dict[str, PathWaypoint],
    final_fix_distance_nm: float,
    final_fix_cross_track_tolerance_nm: float,
) -> BaseRoute:
    runway_identifier = _normalize_runway_identifier(row["runway"])
    final_fix = _select_final_fix(
        runway_identifier=runway_identifier,
        fix_catalog=fix_catalog,
        target_distance_nm=final_fix_distance_nm,
        cross_track_tolerance_nm=final_fix_cross_track_tolerance_nm,
    )
    atc_token = _wait_atc_route_token(wait_atc_point, fix_catalog)
    lateral_path = _dedupe_consecutive_route_tokens(
        [atc_token, final_fix.waypoint.identifier, runway_identifier]
    )
    if len(lateral_path) < 2:
        raise ValueError("base route must contain at least two unique waypoints")
    return BaseRoute(
        lateral_path=lateral_path,
        upstream_identifier=_upstream_identifier_for_route(lateral_path),
        runway_identifier=runway_identifier,
        final_fix=final_fix,
        atc_point=wait_atc_point,
        target_final_fix_distance_nm=final_fix_distance_nm,
        final_fix_cross_track_tolerance_nm=final_fix_cross_track_tolerance_nm,
    )


def _flight_raw_tracks(raw_adsb_dir: Path, *, processes: int, split_gap_seconds: int) -> dict[str, pd.DataFrame]:
    tracks = split_tracks_by_gap(load_raw_adsb(raw_adsb_dir, processes), split_gap_seconds)
    return {str(flight_id): flight.copy() for flight_id, flight in tracks.groupby("flight_id", sort=False)}


def _seed_for_flight_at_fix(
    flight: pd.DataFrame, *, fix_lat_deg: float, fix_lon_deg: float
) -> SeedState | None:
    valid = flight.loc[
        flight["time"].notna()
        & flight["lat"].notna()
        & flight["lon"].notna()
        & flight["geoaltitude"].notna()
    ].reset_index(drop=True)
    if len(valid) < 2:
        return None

    distances_m = np.asarray(
        [
            _latlon_distance_m(
                float(lat_deg),
                float(lon_deg),
                fix_lat_deg,
                fix_lon_deg,
            )
            for lat_deg, lon_deg in valid[["lat", "lon"]].itertuples(
                index=False, name=None
            )
        ],
        dtype=float,
    )
    position = int(np.argmin(distances_m))
    row = valid.iloc[position]
    if position + 1 < len(valid):
        neighbor = valid.iloc[position + 1]
    else:
        neighbor = valid.iloc[position - 1]
    dt_s = abs(float(neighbor["time"]) - float(row["time"]))
    if dt_s <= 0.0:
        return None
    distance_m = _latlon_distance_m(
        float(row["lat"]),
        float(row["lon"]),
        float(neighbor["lat"]),
        float(neighbor["lon"]),
    )
    ground_speed_mps = distance_m / dt_s
    if not np.isfinite(ground_speed_mps) or ground_speed_mps <= 1.0:
        return None
    heading = float(row["heading"]) if pd.notna(row["heading"]) else None
    return SeedState(
        time_s=int(row["time"]),
        lat_deg=float(row["lat"]),
        lon_deg=float(row["lon"]),
        geoaltitude_m=float(row["geoaltitude"]),
        heading_deg=heading,
        ground_speed_mps=float(ground_speed_mps),
    )


def _latlon_distance_m(lat_a_deg: float, lon_a_deg: float, lat_b_deg: float, lon_b_deg: float) -> float:
    lat0_rad = np.deg2rad(0.5 * (lat_a_deg + lat_b_deg))
    dx = EARTH_RADIUS_M * np.cos(lat0_rad) * np.deg2rad(lon_b_deg - lon_a_deg)
    dy = EARTH_RADIUS_M * np.deg2rad(lat_b_deg - lat_a_deg)
    return float(np.hypot(dx, dy))


def _heading_deg_to_psi_rad(heading_deg: float) -> float:
    return wrap_angle_rad(np.deg2rad(90.0 - heading_deg))


def _build_request(
    *,
    route: list[str | tuple[float, float]],
    upstream_identifier: str,
    seed: SeedState,
    fixes_csv: Path,
) -> tuple[FMSRequest, FMSBiChannelState]:
    if len(route) < 2:
        raise ValueError("route must contain at least one upstream waypoint and a runway")
    h_m = max(float(seed.geoaltitude_m), 1.0)
    cas_mps = float(
        aero.tas2cas(
            max(seed.ground_speed_mps, 1.0),
            h_m,
            dT=openap_dT(0.0),
        )
    )
    command = TacticalCommand(
        lateral_path=route,
        upstream=TacticalCondition(
            fix_identifier=upstream_identifier,
            cas_kts=max(80.0, mps_to_kts(cas_mps)),
            altitude_ft=m_to_ft(h_m),
        ),
        altitude_constraints=(),
    )
    bundle = build_tactical_plan_request(command, fixes_csv=fixes_csv)
    start_s_m = bundle.request.reference_path.total_length_m
    fms_request = FMSRequest.from_coupled_request(bundle.request, start_s_m=start_s_m)
    east_m, north_m = bundle.request.reference_path.position_ne(fms_request.start_s_m)
    psi_rad = (
        _heading_deg_to_psi_rad(seed.heading_deg)
        if seed.heading_deg is not None
        else bundle.request.reference_path.track_angle_rad(fms_request.start_s_m)
    )
    initial_state = FMSBiChannelState(
        t_s=0.0,
        s_m=fms_request.start_s_m,
        h_m=fms_request.start_h_m,
        v_tas_mps=float(
            aero.cas2tas(
                fms_request.start_cas_mps, fms_request.start_h_m, dT=openap_dT(0.0)
            )
        ),
        east_m=east_m,
        north_m=north_m,
        psi_rad=psi_rad,
        phi_rad=0.0,
    )
    return fms_request, initial_state


def _payload_from_result(
    *,
    row: pd.Series,
    seed_time_s: int,
    wait_atc_point: dict[str, Any] | None,
    base_route: BaseRoute,
    result,
    lateral_tolerance_m: float,
    altitude_tolerance_m: float,
) -> tuple[ArtifactResult, dict[str, Any]]:
    times = np.rint(seed_time_s + result.t_s).astype(np.int64)
    latitudes = np.asarray(result.lat_deg, dtype=float)
    longitudes = np.asarray(result.lon_deg, dtype=float)
    geoaltitudes = np.asarray(result.h_m, dtype=float)
    breakpoints = compress_breakpoints(
        times=times,
        latitudes=latitudes,
        longitudes=longitudes,
        geoaltitudes=geoaltitudes,
        lateral_tolerance_m=lateral_tolerance_m,
        altitude_tolerance_m=altitude_tolerance_m,
    )
    lateral_indices = set(int(index) for index in breakpoints.lateral_indices)
    altitude_indices = set(int(index) for index in breakpoints.altitude_indices)
    points: list[list[int | float]] = []
    for index in breakpoints.minimal_indices:
        int_index = int(index)
        mask = 0
        if int_index in lateral_indices:
            mask |= LATERAL_BREAKPOINT_MASK
        if int_index in altitude_indices:
            mask |= ALTITUDE_BREAKPOINT_MASK
        points.append(
            [
                int(times[int_index]),
                float(latitudes[int_index]),
                float(longitudes[int_index]),
                float(geoaltitudes[int_index]),
                mask,
            ]
        )

    payload = {
        "flight_id": str(row["flight_id"]),
        "callsign": str(row["callsign"]),
        "icao24": str(row["icao24"]),
        "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
        "breakpoint_mask_bits": {
            "lateral": LATERAL_BREAKPOINT_MASK,
            "altitude": ALTITUDE_BREAKPOINT_MASK,
        },
        "points": points,
        "lateral_breakpoint_times": [int(times[index]) for index in breakpoints.lateral_indices],
        "altitude_breakpoint_times": [int(times[index]) for index in breakpoints.altitude_indices],
        "wait_atc_point": wait_atc_point,
        "base_route": base_route.to_payload(),
        "first_time": int(times[0]),
        "last_time": int(times[-1]),
        "raw_point_count": len(times),
        "compressed_point_count": len(points),
        "lateral_tolerance_m": lateral_tolerance_m,
        "altitude_tolerance_m": altitude_tolerance_m,
        "simulation": {
            "success": bool(result.success),
            "message": result.message,
            "max_abs_cross_track_m": float(result.max_abs_cross_track_m),
            "max_abs_track_error_rad": float(result.max_abs_track_error_rad),
            "final_threshold_error_m": float(result.final_threshold_error_m),
        },
    }
    artifact = ArtifactResult(
        flight_id=str(row["flight_id"]),
        callsign=str(row["callsign"]),
        icao24=str(row["icao24"]),
        first_time=int(times[0]),
        last_time=int(times[-1]),
        raw_point_count=len(times),
        compressed_point_count=len(points),
        lateral_breakpoint_count=len(breakpoints.lateral_indices),
        altitude_breakpoint_count=len(breakpoints.altitude_indices),
        status="generated",
        wait_atc_point_found=wait_atc_point is not None,
        wait_atc_cluster=str(wait_atc_point["arrival_cluster"]) if wait_atc_point is not None else None,
        simulation_success=bool(result.success),
        simulation_message=str(result.message),
    )
    return artifact, payload


def _skip_result(row: pd.Series, reason: str) -> ArtifactResult:
    return ArtifactResult(
        flight_id=str(row["flight_id"]),
        callsign=str(row["callsign"]),
        icao24=str(row["icao24"]),
        first_time=None,
        last_time=None,
        raw_point_count=0,
        compressed_point_count=0,
        lateral_breakpoint_count=0,
        altitude_breakpoint_count=0,
        status="skipped",
        reason=reason,
    )


def _reason_counts(results: list[ArtifactResult]) -> dict[str, int]:
    counts = Counter(result.reason or "unknown" for result in results if result.status == "skipped")
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _simulation_message_counts(results: list[ArtifactResult]) -> dict[str, int]:
    counts = Counter(
        result.simulation_message or "unknown"
        for result in results
        if result.status == "generated" and result.simulation_success is False
    )
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _print_precompute_summary(
    *,
    console: Console,
    manifest: dict[str, Any],
    diagnostics: list[str],
) -> None:
    console.print(
        "[bold]Artifact summary[/bold] "
        f"generated={manifest['generated_count']} "
        f"skipped_arrivals={manifest['skipped_arrival_count']} "
        f"skipped_departures={manifest['skipped_departure_count']} "
        f"sim_success={manifest['simulation_success_count']} "
        f"sim_failed={manifest['simulation_failure_count']}"
    )

    skip_counts = manifest.get("skipped_arrival_reason_counts", {})
    if skip_counts:
        console.print("[bold yellow]Skipped arrivals by reason[/bold yellow]")
        for reason, count in skip_counts.items():
            console.print(f"  {count:>4}  {reason}")

    simulation_failure_counts = manifest.get("simulation_failure_message_counts", {})
    if simulation_failure_counts:
        console.print("[bold yellow]Generated artifacts with failed SIMAP simulation[/bold yellow]")
        for message, count in simulation_failure_counts.items():
            console.print(f"  {count:>4}  {message}")

    if diagnostics:
        console.print(f"[yellow]wait_atc_point diagnostics:[/yellow] {len(diagnostics)} messages")


def _process_arrival_task(task: ArtifactTask) -> tuple[ArtifactResult, dict[str, Any] | None, list[str]]:
    row = pd.Series(task.row)
    diagnostics: list[str] = []
    try:
        if task.raw_flight is None:
            return _skip_result(row, "missing raw ADS-B flight"), None, diagnostics
        route = _route_tokens(row["fix_sequence"], row["runway"])
        if len(route) < 2:
            return _skip_result(row, "missing route fixes before runway"), None, diagnostics
        fix_catalog = load_fix_catalog(task.fixes_csv)
        wait_atc_point = detect_wait_atc_point(
            route,
            fix_catalog,
            runway=str(row["runway"]),
            diagnostics=diagnostics,
            trace_label=f'{row["flight_id"]}/{row["callsign"]}',
        )
        if wait_atc_point is None:
            return _skip_result(row, "missing ATC decision point"), None, diagnostics
        base_route = _build_base_route(
            row=row,
            wait_atc_point=wait_atc_point,
            fix_catalog=fix_catalog,
            final_fix_distance_nm=task.final_fix_distance_nm,
            final_fix_cross_track_tolerance_nm=task.final_fix_cross_track_tolerance_nm,
        )
        seed = _seed_for_flight_at_fix(
            task.raw_flight,
            fix_lat_deg=float(wait_atc_point["lat"]),
            fix_lon_deg=float(wait_atc_point["lon"]),
        )
        if seed is None:
            return _skip_result(row, "missing raw seed near ATC decision point"), None, diagnostics
        fms_request, initial_state = _build_request(
            route=base_route.lateral_path,
            upstream_identifier=base_route.upstream_identifier,
            seed=seed,
            fixes_csv=task.fixes_csv,
        )
        result = plan_fms_bichannel(
            FMSBiChannelRequest(
                base_request=fms_request,
                guidance=LateralGuidanceConfig(),
                initial_state=initial_state,
            )
        )
        artifact, payload = _payload_from_result(
            row=row,
            seed_time_s=seed.time_s,
            wait_atc_point=wait_atc_point,
            base_route=base_route,
            result=result,
            lateral_tolerance_m=task.lateral_tolerance_m,
            altitude_tolerance_m=task.altitude_tolerance_m,
        )
        return artifact, payload, diagnostics
    except Exception as exc:
        return _skip_result(row, f"{type(exc).__name__}: {exc}"), None, diagnostics


def _run_artifact_tasks(tasks: list[ArtifactTask], processes: int):
    if not tasks:
        return
    worker_count = max(1, min(int(processes), len(tasks)))
    if worker_count <= 1:
        for task in tasks:
            yield _process_arrival_task(task)
        return
    with Pool(processes=worker_count) as pool:
        yield from pool.imap_unordered(_process_arrival_task, tasks, chunksize=1)


def precompute_artifacts(
    *,
    events_path: Path,
    fix_sequences_path: Path,
    raw_adsb_dir: Path,
    fixes_csv: Path,
    output_dir: Path,
    lateral_tolerance_m: float = DEFAULT_LATERAL_TOLERANCE_M,
    altitude_tolerance_m: float = DEFAULT_ALTITUDE_TOLERANCE_M,
    final_fix_distance_nm: float = DEFAULT_FINAL_FIX_DISTANCE_NM,
    final_fix_cross_track_tolerance_nm: float = DEFAULT_FINAL_FIX_CROSS_TRACK_TOLERANCE_NM,
    split_gap_seconds: int = DEFAULT_SPLIT_GAP_SECONDS,
    processes: int = 1,
    limit: int | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    arrivals = _arrival_rows(events_path, fix_sequences_path)
    if limit is not None:
        arrivals = arrivals.head(limit).copy()
    if console is not None:
        console.print(
            "[bold]Precomputing base-route arrivals[/bold] "
            f"arrivals={len(arrivals)} "
            f"final_fix_target={final_fix_distance_nm:.1f}NM "
            f"centerline_tolerance={final_fix_cross_track_tolerance_nm:.2f}NM "
            f"processes={processes}"
        )
        console.print(
            "[dim]Inputs:[/dim] "
            f"events={events_path.as_posix()} "
            f"fix_sequences={fix_sequences_path.as_posix()} "
            f"raw_adsb={raw_adsb_dir.as_posix()} "
            f"fixes={fixes_csv.as_posix()}"
        )
    if console is not None:
        with console.status("[bold]Loading raw ADS-B tracks...[/bold]"):
            raw_by_flight = _flight_raw_tracks(raw_adsb_dir, processes=processes, split_gap_seconds=split_gap_seconds)
    else:
        raw_by_flight = _flight_raw_tracks(raw_adsb_dir, processes=processes, split_gap_seconds=split_gap_seconds)
    if console is not None:
        console.print(f"[bold]Loaded raw ADS-B tracks[/bold] flights={len(raw_by_flight)}")

    tasks: list[ArtifactTask] = []
    for _idx, row in arrivals.iterrows():
        flight_id = str(row["flight_id"])
        tasks.append(
            ArtifactTask(
                row=dict(row),
                raw_flight=raw_by_flight.get(flight_id),
                fixes_csv=fixes_csv,
                lateral_tolerance_m=lateral_tolerance_m,
                altitude_tolerance_m=altitude_tolerance_m,
                final_fix_distance_nm=final_fix_distance_nm,
                final_fix_cross_track_tolerance_nm=final_fix_cross_track_tolerance_nm,
            )
        )
    if console is not None:
        missing_raw_count = sum(1 for task in tasks if task.raw_flight is None)
        if missing_raw_count:
            console.print(f"[yellow]Arrivals missing raw ADS-B before simulation:[/yellow] {missing_raw_count}")
    task_results: list[tuple[ArtifactResult, dict[str, Any] | None, list[str]]] = []
    if console is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("[green]ok {task.fields[generated]}[/green]"),
            TextColumn("[red]skipped {task.fields[skipped]}[/red]"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        with progress:
            progress_task = progress.add_task(
                "[cyan]Computing artifact flights[/cyan]",
                total=len(tasks),
                generated=0,
                skipped=0,
            )
            for task_result in _run_artifact_tasks(tasks, processes=processes) or ():
                task_results.append(task_result)
                result, _payload, _diagnostics = task_result
                progress.update(
                    progress_task,
                    generated=sum(1 for item, _payload, _messages in task_results if item.status == "generated"),
                    skipped=sum(1 for item, _payload, _messages in task_results if item.status == "skipped"),
                )
                progress.advance(progress_task)
    else:
        task_results = list(_run_artifact_tasks(tasks, processes=processes) or ())
    results = [result for result, _payload, _diagnostics in task_results]
    payloads = [payload for _result, payload, _diagnostics in task_results if payload is not None]
    diagnostics = [message for _result, _payload, messages in task_results for message in messages]
    if console is not None and diagnostics:
        console.print("[bold yellow]Sample wait_atc_point diagnostics[/bold yellow]")
        for message in diagnostics[:100]:
            console.print(f"[yellow]wait_atc_point:[/yellow] {message}")
        if len(diagnostics) > 100:
            console.print(
                f"[yellow]wait_atc_point:[/yellow] {len(diagnostics) - 100} additional diagnostics suppressed"
            )

    if console is not None:
        with console.status("[bold]Writing artifact outputs...[/bold]"):
            output_dir.mkdir(parents=True, exist_ok=True)
            flights_path = output_dir / OUTPUT_FLIGHTS_FILENAME
            write_jsonl(flights_path, sorted(payloads, key=lambda item: (int(item["first_time"]), str(item["flight_id"]))))
            manifest = _manifest(
                results=results,
                events_path=events_path,
                fix_sequences_path=fix_sequences_path,
                raw_adsb_dir=raw_adsb_dir,
                fixes_csv=fixes_csv,
                flights_path=flights_path,
                lateral_tolerance_m=lateral_tolerance_m,
                altitude_tolerance_m=altitude_tolerance_m,
                final_fix_distance_nm=final_fix_distance_nm,
                final_fix_cross_track_tolerance_nm=final_fix_cross_track_tolerance_nm,
                processes=processes,
                skipped_departure_count=_departure_count(events_path),
            )
            manifest_path = output_dir / "manifest.json"
            with manifest_path.open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
        _print_precompute_summary(console=console, manifest=manifest, diagnostics=diagnostics)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        flights_path = output_dir / OUTPUT_FLIGHTS_FILENAME
        write_jsonl(flights_path, sorted(payloads, key=lambda item: (int(item["first_time"]), str(item["flight_id"]))))
        manifest = _manifest(
            results=results,
            events_path=events_path,
            fix_sequences_path=fix_sequences_path,
            raw_adsb_dir=raw_adsb_dir,
            fixes_csv=fixes_csv,
            flights_path=flights_path,
            lateral_tolerance_m=lateral_tolerance_m,
            altitude_tolerance_m=altitude_tolerance_m,
            final_fix_distance_nm=final_fix_distance_nm,
            final_fix_cross_track_tolerance_nm=final_fix_cross_track_tolerance_nm,
            processes=processes,
            skipped_departure_count=_departure_count(events_path),
        )
        manifest_path = output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return manifest


def _departure_count(events_path: Path) -> int:
    events = pd.read_csv(events_path)
    return int(events["operation"].astype(str).str.lower().eq("departure").sum())


def _manifest(
    *,
    results: list[ArtifactResult],
    events_path: Path,
    fix_sequences_path: Path,
    raw_adsb_dir: Path,
    fixes_csv: Path,
    flights_path: Path,
    lateral_tolerance_m: float,
    altitude_tolerance_m: float,
    final_fix_distance_nm: float,
    final_fix_cross_track_tolerance_nm: float,
    processes: int,
    skipped_departure_count: int,
) -> dict[str, Any]:
    generated = [result for result in results if result.status == "generated"]
    skipped = [result for result in results if result.status == "skipped"]
    simulation_success = [
        result for result in generated if result.simulation_success is True
    ]
    simulation_failure = [
        result for result in generated if result.simulation_success is False
    ]
    wait_atc_point_count = sum(1 for result in generated if result.wait_atc_point_found)
    wait_atc_cluster_counts = {
        cluster: sum(1 for result in generated if result.wait_atc_cluster == cluster)
        for cluster in ("NE", "NW", "SW", "SE")
    }
    return {
        "created_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_type": "simap_fms_bichannel_base_route_arrivals",
        "events_path": events_path.as_posix(),
        "fix_sequences_path": fix_sequences_path.as_posix(),
        "raw_adsb_dir": raw_adsb_dir.as_posix(),
        "fixes_csv": fixes_csv.as_posix(),
        "flights_path": flights_path.as_posix(),
        "arrival_count": len(results),
        "generated_count": len(generated),
        "skipped_arrival_count": len(skipped),
        "skipped_departure_count": skipped_departure_count,
        "skipped_arrival_reason_counts": _reason_counts(results),
        "simulation_success_count": len(simulation_success),
        "simulation_failure_count": len(simulation_failure),
        "simulation_failure_message_counts": _simulation_message_counts(results),
        "wait_atc_point_count": wait_atc_point_count,
        "wait_atc_point_success_rate": (
            float(wait_atc_point_count / len(generated)) if generated else 0.0
        ),
        "wait_atc_cluster_counts": wait_atc_cluster_counts,
        "raw_point_count": sum(result.raw_point_count for result in generated),
        "compressed_point_count": sum(result.compressed_point_count for result in generated),
        "lateral_tolerance_m": lateral_tolerance_m,
        "altitude_tolerance_m": altitude_tolerance_m,
        "base_route": {
            "type": "base-route",
            "final_fix_target_distance_nm": final_fix_distance_nm,
            "final_fix_cross_track_tolerance_nm": final_fix_cross_track_tolerance_nm,
        },
        "processes": int(processes),
        "breakpoint_mask_bits": {
            "lateral": LATERAL_BREAKPOINT_MASK,
            "altitude": ALTITUDE_BREAKPOINT_MASK,
        },
        "layout": {
            "flights": flights_path.name,
            "point_columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
        },
        "flights": [
            {
                "flight_id": result.flight_id,
                "callsign": result.callsign,
                "icao24": result.icao24,
                "first_time": result.first_time,
                "last_time": result.last_time,
                "raw_point_count": result.raw_point_count,
                "compressed_point_count": result.compressed_point_count,
                "compression_ratio": result.compression_ratio,
                "lateral_breakpoint_count": result.lateral_breakpoint_count,
                "altitude_breakpoint_count": result.altitude_breakpoint_count,
                "status": result.status,
                "reason": result.reason,
                "wait_atc_point_found": result.wait_atc_point_found,
                "wait_atc_cluster": result.wait_atc_cluster,
                "simulation_success": result.simulation_success,
                "simulation_message": result.simulation_message,
            }
            for result in sorted(results, key=lambda item: (item.status, item.first_time or 0, item.flight_id))
        ],
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = project_root()
    console = Console()
    manifest = precompute_artifacts(
        events_path=_abs_path(root, args.events_path),
        fix_sequences_path=_abs_path(root, args.fix_sequences_path),
        raw_adsb_dir=_abs_path(root, args.raw_adsb_dir),
        fixes_csv=_abs_path(root, args.fixes_csv),
        output_dir=_abs_path(root, args.output_dir),
        lateral_tolerance_m=args.lateral_tolerance_m,
        altitude_tolerance_m=args.altitude_tolerance_m,
        final_fix_distance_nm=args.final_fix_distance_nm,
        final_fix_cross_track_tolerance_nm=args.final_fix_cross_track_tolerance_nm,
        split_gap_seconds=args.split_gap_seconds,
        processes=args.processes,
        limit=args.limit,
        console=console,
    )
    console.print(
        json.dumps(
            {
                "generated_count": manifest["generated_count"],
                "skipped_arrival_count": manifest["skipped_arrival_count"],
                "skipped_departure_count": manifest["skipped_departure_count"],
                "simulation_success_count": manifest["simulation_success_count"],
                "simulation_failure_count": manifest["simulation_failure_count"],
                "skipped_arrival_reason_counts": manifest["skipped_arrival_reason_counts"],
                "simulation_failure_message_counts": manifest["simulation_failure_message_counts"],
                "wait_atc_point_count": manifest["wait_atc_point_count"],
                "wait_atc_point_success_rate": manifest["wait_atc_point_success_rate"],
                "flights_path": manifest["flights_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
