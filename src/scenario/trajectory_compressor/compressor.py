from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import numpy as np

from scenario.trajectory_compressor.algorithms import compress_breakpoints
from scenario.trajectory_compressor.io import (
    CompressedFlightResult,
    FlightCompressionTask,
)

LATERAL_BREAKPOINT_MASK = 1
ALTITUDE_BREAKPOINT_MASK = 2


def compress_flight_task(
    args: tuple[FlightCompressionTask, float, float],
) -> tuple[CompressedFlightResult, dict[str, object]]:
    task, lateral_tolerance_m, altitude_tolerance_m = args
    records = task.records

    if not records:
        payload: dict[str, object] = {
            "flight_id": task.flight_id,
            "callsign": task.callsign,
            "icao24": task.icao24,
            "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
            "points": [],
            "lateral_breakpoint_times": [],
            "altitude_breakpoint_times": [],
            "raw_point_count": 0,
            "compressed_point_count": 0,
            "lateral_tolerance_m": lateral_tolerance_m,
            "altitude_tolerance_m": altitude_tolerance_m,
        }
        result = CompressedFlightResult(task.flight_id, task.callsign, task.icao24, None, None, 0, 0, 0, 0)
        return result, payload

    data = np.array(records, dtype=float)
    times = data[:, 0].astype(np.int64)
    latitudes = data[:, 1]
    longitudes = data[:, 2]
    geoaltitudes = data[:, 3]

    breakpoints = compress_breakpoints(
        times=times,
        latitudes=latitudes,
        longitudes=longitudes,
        geoaltitudes=geoaltitudes,
        lateral_tolerance_m=lateral_tolerance_m,
        altitude_tolerance_m=altitude_tolerance_m,
    )
    lateral_index_set = set(int(index) for index in breakpoints.lateral_indices)
    altitude_index_set = set(int(index) for index in breakpoints.altitude_indices)

    points: list[list[float | int]] = []
    for index in breakpoints.minimal_indices:
        int_index = int(index)
        mask = 0
        if int_index in lateral_index_set:
            mask |= LATERAL_BREAKPOINT_MASK
        if int_index in altitude_index_set:
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
        "flight_id": task.flight_id,
        "callsign": task.callsign,
        "icao24": task.icao24,
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
        "raw_point_count": len(records),
        "compressed_point_count": len(points),
        "lateral_tolerance_m": lateral_tolerance_m,
        "altitude_tolerance_m": altitude_tolerance_m,
    }
    result = CompressedFlightResult(
        flight_id=task.flight_id,
        callsign=task.callsign,
        icao24=task.icao24,
        first_time=int(times[0]),
        last_time=int(times[-1]),
        raw_point_count=len(records),
        compressed_point_count=len(points),
        lateral_breakpoint_count=len(breakpoints.lateral_indices),
        altitude_breakpoint_count=len(breakpoints.altitude_indices),
    )
    return result, payload


def compress_flights(
    tasks: list[FlightCompressionTask],
    output_dir: Path,
    lateral_tolerance_m: float,
    altitude_tolerance_m: float,
    processes: int,
) -> Iterator[tuple[CompressedFlightResult, dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_args = [(task, lateral_tolerance_m, altitude_tolerance_m) for task in tasks]

    if processes <= 1 or len(worker_args) <= 1:
        for item in worker_args:
            yield compress_flight_task(item)
        return

    worker_count = max(1, min(processes, len(worker_args)))
    with Pool(processes=worker_count) as pool:
        yield from pool.imap_unordered(compress_flight_task, worker_args)
