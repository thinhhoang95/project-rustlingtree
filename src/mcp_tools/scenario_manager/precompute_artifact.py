from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from simap.openap_adapter import openap_dT
from simap.path_geometry import EARTH_RADIUS_M, ReferencePath
from simap.units import m_to_ft, mps_to_kts


DEFAULT_EVENTS_PATH = Path("data/adsb/catalogs/2026-04-01_landings_and_departures.csv")
DEFAULT_FIX_SEQUENCES_PATH = Path("data/adsb/catalogs/2026-04-01_fix_sequences.csv")
DEFAULT_RAW_ADSB_DIR = Path("data/adsb/raw")
DEFAULT_FIXES_CSV = Path("data/kdfw_procs/airport_related_fixes.csv")
DEFAULT_OUTPUT_DIR = Path("data/artifacts")
DEFAULT_LATERAL_TOLERANCE_M = 100.0
DEFAULT_ALTITUDE_TOLERANCE_M = 50.0
DEFAULT_SPLIT_GAP_SECONDS = 25 * 60


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute simulated scenario-manager arrival artifacts.")
    parser.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--fix-sequences-path", type=Path, default=DEFAULT_FIX_SEQUENCES_PATH)
    parser.add_argument("--raw-adsb-dir", type=Path, default=DEFAULT_RAW_ADSB_DIR)
    parser.add_argument("--fixes-csv", type=Path, default=DEFAULT_FIXES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lateral-tolerance-m", type=float, default=DEFAULT_LATERAL_TOLERANCE_M)
    parser.add_argument("--altitude-tolerance-m", type=float, default=DEFAULT_ALTITUDE_TOLERANCE_M)
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
    runway_token = f"RW{str(runway).strip().upper()}"
    if runway_token not in tokens:
        tokens.append(runway_token)
    return tokens


def _flight_raw_tracks(raw_adsb_dir: Path, *, processes: int, split_gap_seconds: int) -> dict[str, pd.DataFrame]:
    tracks = split_tracks_by_gap(load_raw_adsb(raw_adsb_dir, processes), split_gap_seconds)
    return {str(flight_id): flight.copy() for flight_id, flight in tracks.groupby("flight_id", sort=False)}


def _seed_for_flight(flight: pd.DataFrame, first_time: int) -> SeedState | None:
    matches = flight.index[flight["time"].eq(first_time)].to_list()
    if not matches:
        return None
    row_index = matches[0]
    position = int(flight.index.get_loc(row_index))
    if len(flight) < 2:
        return None
    if position + 1 < len(flight):
        neighbor = flight.iloc[position + 1]
        row = flight.loc[row_index]
    else:
        neighbor = flight.iloc[position - 1]
        row = flight.loc[row_index]
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


def _ne_from_latlon(reference_path: ReferencePath, *, lat_deg: float, lon_deg: float) -> tuple[float, float]:
    lat0_rad = np.deg2rad(reference_path.origin_lat_deg)
    east_m = EARTH_RADIUS_M * np.cos(lat0_rad) * np.deg2rad(lon_deg - reference_path.origin_lon_deg)
    north_m = EARTH_RADIUS_M * np.deg2rad(lat_deg - reference_path.origin_lat_deg)
    return float(east_m), float(north_m)


def _heading_deg_to_psi_rad(heading_deg: float) -> float:
    return wrap_angle_rad(np.deg2rad(90.0 - heading_deg))


def _build_request(row: pd.Series, seed: SeedState, fixes_csv: Path) -> tuple[FMSRequest, FMSBiChannelState]:
    route = _route_tokens(row["fix_sequence"], row["runway"])
    if len(route) < 2:
        raise ValueError("route must contain at least one fix and a runway")
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
            fix_identifier=route[0],
            cas_kts=max(80.0, mps_to_kts(cas_mps)),
            altitude_ft=m_to_ft(h_m),
        ),
        altitude_constraints=(),
    )
    bundle = build_tactical_plan_request(command, fixes_csv=fixes_csv)
    fms_request = FMSRequest.from_coupled_request(bundle.request, start_s_m=bundle.request.reference_path.total_length_m)
    east_m, north_m = _ne_from_latlon(bundle.request.reference_path, lat_deg=seed.lat_deg, lon_deg=seed.lon_deg)
    psi_rad = (
        _heading_deg_to_psi_rad(seed.heading_deg)
        if seed.heading_deg is not None
        else bundle.request.reference_path.track_angle_rad(fms_request.start_s_m)
    )
    initial_state = FMSBiChannelState(
        t_s=0.0,
        s_m=fms_request.start_s_m,
        h_m=fms_request.start_h_m,
        v_tas_mps=fms_request.start_cas_mps,
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


def _process_arrival_task(task: ArtifactTask) -> tuple[ArtifactResult, dict[str, Any] | None]:
    row = pd.Series(task.row)
    try:
        if task.raw_flight is None:
            return _skip_result(row, "missing raw ADS-B flight"), None
        seed = _seed_for_flight(task.raw_flight, int(row["first_time"]))
        if seed is None:
            return _skip_result(row, "missing exact raw seed at first entry fix time"), None
        fms_request, initial_state = _build_request(row, seed, task.fixes_csv)
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
            result=result,
            lateral_tolerance_m=task.lateral_tolerance_m,
            altitude_tolerance_m=task.altitude_tolerance_m,
        )
        return artifact, payload
    except Exception as exc:
        return _skip_result(row, f"{type(exc).__name__}: {exc}"), None


def _run_artifact_tasks(tasks: list[ArtifactTask], processes: int) -> list[tuple[ArtifactResult, dict[str, Any] | None]]:
    if not tasks:
        return []
    worker_count = max(1, min(int(processes), len(tasks)))
    if worker_count <= 1:
        return [_process_arrival_task(task) for task in tasks]
    with Pool(processes=worker_count) as pool:
        return list(pool.imap_unordered(_process_arrival_task, tasks, chunksize=1))


def precompute_artifacts(
    *,
    events_path: Path,
    fix_sequences_path: Path,
    raw_adsb_dir: Path,
    fixes_csv: Path,
    output_dir: Path,
    lateral_tolerance_m: float = DEFAULT_LATERAL_TOLERANCE_M,
    altitude_tolerance_m: float = DEFAULT_ALTITUDE_TOLERANCE_M,
    split_gap_seconds: int = DEFAULT_SPLIT_GAP_SECONDS,
    processes: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    arrivals = _arrival_rows(events_path, fix_sequences_path)
    if limit is not None:
        arrivals = arrivals.head(limit).copy()
    raw_by_flight = _flight_raw_tracks(raw_adsb_dir, processes=processes, split_gap_seconds=split_gap_seconds)

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
            )
        )
    task_results = _run_artifact_tasks(tasks, processes=processes)
    results = [result for result, _payload in task_results]
    payloads = [payload for _result, payload in task_results if payload is not None]

    output_dir.mkdir(parents=True, exist_ok=True)
    flights_path = output_dir / "flights.jsonl"
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
    processes: int,
    skipped_departure_count: int,
) -> dict[str, Any]:
    generated = [result for result in results if result.status == "generated"]
    skipped = [result for result in results if result.status == "skipped"]
    return {
        "created_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_type": "simap_fms_bichannel_arrivals",
        "events_path": events_path.as_posix(),
        "fix_sequences_path": fix_sequences_path.as_posix(),
        "raw_adsb_dir": raw_adsb_dir.as_posix(),
        "fixes_csv": fixes_csv.as_posix(),
        "flights_path": flights_path.as_posix(),
        "arrival_count": len(results),
        "generated_count": len(generated),
        "skipped_arrival_count": len(skipped),
        "skipped_departure_count": skipped_departure_count,
        "raw_point_count": sum(result.raw_point_count for result in generated),
        "compressed_point_count": sum(result.compressed_point_count for result in generated),
        "lateral_tolerance_m": lateral_tolerance_m,
        "altitude_tolerance_m": altitude_tolerance_m,
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
            }
            for result in sorted(results, key=lambda item: (item.status, item.first_time or 0, item.flight_id))
        ],
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = project_root()
    manifest = precompute_artifacts(
        events_path=_abs_path(root, args.events_path),
        fix_sequences_path=_abs_path(root, args.fix_sequences_path),
        raw_adsb_dir=_abs_path(root, args.raw_adsb_dir),
        fixes_csv=_abs_path(root, args.fixes_csv),
        output_dir=_abs_path(root, args.output_dir),
        lateral_tolerance_m=args.lateral_tolerance_m,
        altitude_tolerance_m=args.altitude_tolerance_m,
        split_gap_seconds=args.split_gap_seconds,
        processes=args.processes,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "generated_count": manifest["generated_count"],
                "skipped_arrival_count": manifest["skipped_arrival_count"],
                "skipped_departure_count": manifest["skipped_departure_count"],
                "flights_path": manifest["flights_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
