from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hllrd.fit import (
    HLLRDEvent,
    HLLRDFitResult,
    _event_row_extension,
    _event_row_lag,
    _row_event_basis,
    refit_registered_coefficients,
)
from hllrd.geometry import LocalProjection, normal_deviation_m, resample_polyline_by_fraction
from hllrd.matrix import MatrixArtifact


@dataclass(frozen=True)
class EventTraceMatchRow:
    event: int
    flight_id: str
    flight_row: int
    active: bool
    lag_offset: int
    extension_offset: int
    station_start: int
    station_end: int
    station_count: int
    center_trace_rmse_m: float
    event_trace_rmse_m: float
    model_trace_rmse_m: float
    lifted_model_trace_rmse_m: float
    event_trace_p95_m: float
    model_trace_p95_m: float
    lifted_model_trace_p95_m: float
    center_normal_rmse_m: float
    event_normal_rmse_m: float
    model_normal_rmse_m: float


@dataclass(frozen=True)
class EventTraceMatchResult:
    event: int
    rows: tuple[EventTraceMatchRow, ...]
    summary: dict[str, Any]

    def row_dicts(self) -> list[dict[str, Any]]:
        return [asdict(row) for row in self.rows]


def evaluate_event_trace_match(
    matrix: MatrixArtifact,
    model: HLLRDFitResult,
    *,
    event_index: int,
    raw_tracks: pd.DataFrame | None = None,
    active_only: bool = True,
    trace_lift: str = "none",
) -> EventTraceMatchResult:
    if not (0 <= int(event_index) < len(model.events)):
        raise ValueError("event_index is out of range")
    if matrix.X.shape != model.reconstruction.shape:
        raise ValueError("matrix and model shapes do not match")

    event = model.events[int(event_index)]
    center_xy = center_path_xy(matrix, model)
    event_normal = event_reconstruction_matrix(event)
    event_xy = center_xy[None, :, :] + event_normal[:, :, None] * matrix.normals_xy[None, :, :]
    model_xy = center_xy[None, :, :] + model.reconstruction[:, :, None] * matrix.normals_xy[None, :, :]
    raw_polylines_by_flight = _raw_track_polylines(raw_tracks, matrix) if raw_tracks is not None else None
    actual_xy_by_flight = (
        closest_raw_track_points_to_matrix_stations(raw_tracks, matrix)
        if raw_tracks is not None
        else _matrix_sampled_paths(matrix)
    )
    tangent_reconstruction = _trace_lift_tangent_reconstruction(
        matrix,
        model,
        center_xy=center_xy,
        actual_xy_by_flight=actual_xy_by_flight,
        mode=trace_lift,
    )
    lifted_model_xy = (
        model_xy
        if tangent_reconstruction is None
        else model_xy + tangent_reconstruction[:, :, None] * _tangents_from_normals(matrix.normals_xy)[None, :, :]
    )

    rows: list[EventTraceMatchRow] = []
    for row_index, flight_id in enumerate(matrix.flight_ids):
        active = bool(event.active_mask[row_index])
        if active_only and not active:
            continue
        actual_xy = actual_xy_by_flight.get(str(flight_id))
        if actual_xy is None:
            continue
        lag = _event_row_lag(event, row_index)
        extension = _event_row_extension(event, row_index)
        station_start = max(0, event.start + lag)
        station_end = min(matrix.X.shape[1], event.end + lag + extension)
        if station_end <= station_start:
            continue
        station_slice = slice(station_start, station_end)
        actual_segment = actual_xy[station_slice]
        center_segment = center_xy[station_slice]
        event_segment = event_xy[row_index, station_slice]
        model_segment = model_xy[row_index, station_slice]
        lifted_model_segment = lifted_model_xy[row_index, station_slice]
        if raw_polylines_by_flight is None:
            center_trace_rmse = _xy_rmse(actual_segment, center_segment)
            event_trace_rmse = _xy_rmse(actual_segment, event_segment)
            model_trace_rmse = _xy_rmse(actual_segment, model_segment)
            lifted_model_trace_rmse = _xy_rmse(actual_segment, lifted_model_segment)
            event_trace_p95 = _xy_percentile_error(actual_segment, event_segment, 95.0)
            model_trace_p95 = _xy_percentile_error(actual_segment, model_segment, 95.0)
            lifted_model_trace_p95 = _xy_percentile_error(actual_segment, lifted_model_segment, 95.0)
        else:
            raw_polyline = raw_polylines_by_flight.get(str(flight_id))
            if raw_polyline is None:
                continue
            raw_window = _raw_polyline_window(raw_polyline, actual_xy, station_start, station_end)
            center_distances = _point_to_polyline_distances(center_segment, raw_window)
            event_distances = _point_to_polyline_distances(event_segment, raw_window)
            model_distances = _point_to_polyline_distances(model_segment, raw_window)
            lifted_model_distances = _point_to_polyline_distances(lifted_model_segment, raw_window)
            center_trace_rmse = _rms(center_distances)
            event_trace_rmse = _rms(event_distances)
            model_trace_rmse = _rms(model_distances)
            lifted_model_trace_rmse = _rms(lifted_model_distances)
            event_trace_p95 = _percentile(event_distances, 95.0)
            model_trace_p95 = _percentile(model_distances, 95.0)
            lifted_model_trace_p95 = _percentile(lifted_model_distances, 95.0)
        actual_normal = normal_deviation_m(
            actual_xy,
            center_xy,
            matrix.normals_xy,
        )
        actual_normal_segment = actual_normal[station_slice]
        rows.append(
            EventTraceMatchRow(
                event=int(event_index),
                flight_id=str(flight_id),
                flight_row=int(row_index),
                active=active,
                lag_offset=int(lag),
                extension_offset=int(extension),
                station_start=int(station_start),
                station_end=int(station_end),
                station_count=int(station_end - station_start),
                center_trace_rmse_m=center_trace_rmse,
                event_trace_rmse_m=event_trace_rmse,
                model_trace_rmse_m=model_trace_rmse,
                lifted_model_trace_rmse_m=lifted_model_trace_rmse,
                event_trace_p95_m=event_trace_p95,
                model_trace_p95_m=model_trace_p95,
                lifted_model_trace_p95_m=lifted_model_trace_p95,
                center_normal_rmse_m=_rms(actual_normal_segment),
                event_normal_rmse_m=_rms(actual_normal_segment - event_normal[row_index, station_slice]),
                model_normal_rmse_m=_rms(actual_normal_segment - model.reconstruction[row_index, station_slice]),
            )
        )

    return EventTraceMatchResult(
        event=int(event_index),
        rows=tuple(rows),
        summary=_summarize_rows(int(event_index), rows),
    )


def plot_event_trace_overlay(
    path: str | Path,
    matrix: MatrixArtifact,
    model: HLLRDFitResult,
    *,
    event_index: int,
    raw_tracks: pd.DataFrame | None = None,
    max_flights: int = 60,
    trace_lift: str = "none",
) -> None:
    if not (0 <= int(event_index) < len(model.events)):
        raise ValueError("event_index is out of range")
    event = model.events[int(event_index)]
    center_xy = center_path_xy(matrix, model)
    event_normal = event_reconstruction_matrix(event)
    event_xy = center_xy[None, :, :] + event_normal[:, :, None] * matrix.normals_xy[None, :, :]
    model_xy = center_xy[None, :, :] + model.reconstruction[:, :, None] * matrix.normals_xy[None, :, :]
    actual_xy_by_flight = (
        closest_raw_track_points_to_matrix_stations(raw_tracks, matrix)
        if raw_tracks is not None
        else _matrix_sampled_paths(matrix)
    )
    tangent_reconstruction = _trace_lift_tangent_reconstruction(
        matrix,
        model,
        center_xy=center_xy,
        actual_xy_by_flight=actual_xy_by_flight,
        mode=trace_lift,
    )
    lifted_model_xy = (
        None
        if tangent_reconstruction is None
        else model_xy + tangent_reconstruction[:, :, None] * _tangents_from_normals(matrix.normals_xy)[None, :, :]
    )
    trace_paths = (
        _raw_track_polylines(raw_tracks, matrix)
        if raw_tracks is not None
        else _matrix_sampled_paths(matrix)
    )
    active_rows = [index for index, active in enumerate(event.active_mask) if active]
    if max_flights > 0 and len(active_rows) > int(max_flights):
        selected = np.linspace(0, len(active_rows) - 1, int(max_flights), dtype=int)
        active_rows = [active_rows[index] for index in selected]

    projection = LocalProjection(matrix.origin_lat_deg, matrix.origin_lon_deg)
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    center_lat, center_lon = projection.unproject(center_xy[:, 0], center_xy[:, 1])
    ax.plot(center_lon, center_lat, color="#202020", linewidth=1.8, alpha=0.9, label="center")
    for plot_index, row_index in enumerate(active_rows):
        flight_id = str(matrix.flight_ids[row_index])
        raw_xy = trace_paths.get(flight_id)
        if raw_xy is not None:
            raw_lat, raw_lon = projection.unproject(raw_xy[:, 0], raw_xy[:, 1])
            ax.plot(
                raw_lon,
                raw_lat,
                color="#777777",
                linewidth=0.6,
                alpha=0.28,
                label="ADS-B trace" if plot_index == 0 else None,
            )
        lag = _event_row_lag(event, row_index)
        extension = _event_row_extension(event, row_index)
        station_start = max(0, event.start + lag)
        station_end = min(matrix.X.shape[1], event.end + lag + extension)
        if station_end <= station_start:
            continue
        station_slice = slice(station_start, station_end)
        event_lat, event_lon = projection.unproject(
            event_xy[row_index, station_slice, 0],
            event_xy[row_index, station_slice, 1],
        )
        model_lat, model_lon = projection.unproject(
            model_xy[row_index, station_slice, 0],
            model_xy[row_index, station_slice, 1],
        )
        ax.plot(
            event_lon,
            event_lat,
            color="#b2182b",
            linewidth=1.1,
            alpha=0.42,
            label="event reconstruction" if plot_index == 0 else None,
        )
        ax.plot(
            model_lon,
            model_lat,
            color="#2166ac",
            linewidth=0.9,
            alpha=0.35,
            label="model reconstruction" if plot_index == 0 else None,
        )
        if lifted_model_xy is not None:
            lifted_lat, lifted_lon = projection.unproject(
                lifted_model_xy[row_index, station_slice, 0],
                lifted_model_xy[row_index, station_slice, 1],
            )
            ax.plot(
                lifted_lon,
                lifted_lat,
                color="#1b9e77",
                linewidth=1.0,
                alpha=0.42,
                label="lifted model reconstruction" if plot_index == 0 else None,
            )
    ax.set_title(f"HLLRD event {int(event_index)} trace overlay")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if center_lat.size:
        ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.mean(center_lat)))))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def center_path_xy(matrix: MatrixArtifact, model: HLLRDFitResult) -> np.ndarray:
    if model.column_center.shape[0] != matrix.reference_xy_m.shape[0]:
        raise ValueError("model column center length does not match matrix station count")
    return matrix.reference_xy_m + model.column_center[:, None] * matrix.normals_xy


def event_reconstruction_matrix(event: HLLRDEvent) -> np.ndarray:
    coefficients = np.asarray(event.coefficients, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[1] != 2:
        raise ValueError("event coefficients must have shape N x 2")
    reconstruction = np.zeros((coefficients.shape[0], event.basis.shape[0]), dtype=float)
    for row_index in range(coefficients.shape[0]):
        row_basis = _row_event_basis(
            event,
            event.basis.shape[0],
            lag=_event_row_lag(event, row_index),
            extension=_event_row_extension(event, row_index),
        )
        reconstruction[row_index, :] = coefficients[row_index] @ row_basis.T
    return reconstruction


def resample_raw_tracks_to_matrix_stations(
    tracks: pd.DataFrame,
    matrix: MatrixArtifact,
) -> dict[str, np.ndarray]:
    required = {"flight_id", "time", "lat", "lon"}
    missing = sorted(required - set(tracks.columns))
    if missing:
        raise ValueError(f"tracks missing required columns: {missing}")
    projection = LocalProjection(matrix.origin_lat_deg, matrix.origin_lon_deg)
    clean = tracks.loc[
        tracks["flight_id"].notna()
        & tracks["time"].notna()
        & tracks["lat"].notna()
        & tracks["lon"].notna()
    ].copy()
    clean["flight_id"] = clean["flight_id"].astype(str)
    flight_set = set(matrix.flight_ids)
    clean = clean.loc[clean["flight_id"].isin(flight_set)]

    samples: dict[str, np.ndarray] = {}
    for flight_id, flight in clean.groupby("flight_id", sort=False):
        ordered = flight.sort_values("time", kind="stable").drop_duplicates("time", keep="last")
        lat = ordered["lat"].to_numpy(dtype=float)
        lon = ordered["lon"].to_numpy(dtype=float)
        finite = np.isfinite(lat) & np.isfinite(lon)
        if int(np.count_nonzero(finite)) < 2:
            continue
        x_m, y_m = projection.project(lat[finite], lon[finite])
        try:
            sample_x, sample_y = resample_polyline_by_fraction(x_m, y_m, matrix.stations)
        except ValueError:
            continue
        samples[str(flight_id)] = np.column_stack((sample_x, sample_y))
    return samples


def closest_raw_track_points_to_matrix_stations(
    tracks: pd.DataFrame,
    matrix: MatrixArtifact,
) -> dict[str, np.ndarray]:
    polylines = _raw_track_polylines(tracks, matrix)
    return {
        flight_id: _closest_polyline_points_at_stations(polyline, matrix.reference_xy_m)
        for flight_id, polyline in polylines.items()
    }


def _raw_track_polylines(
    tracks: pd.DataFrame,
    matrix: MatrixArtifact,
) -> dict[str, np.ndarray]:
    projection = LocalProjection(matrix.origin_lat_deg, matrix.origin_lon_deg)
    clean = tracks.loc[
        tracks["flight_id"].notna()
        & tracks["time"].notna()
        & tracks["lat"].notna()
        & tracks["lon"].notna()
    ].copy()
    clean["flight_id"] = clean["flight_id"].astype(str)
    flight_set = set(matrix.flight_ids)
    clean = clean.loc[clean["flight_id"].isin(flight_set)]

    polylines: dict[str, np.ndarray] = {}
    for flight_id, flight in clean.groupby("flight_id", sort=False):
        ordered = flight.sort_values("time", kind="stable").drop_duplicates("time", keep="last")
        lat = ordered["lat"].to_numpy(dtype=float)
        lon = ordered["lon"].to_numpy(dtype=float)
        finite = np.isfinite(lat) & np.isfinite(lon)
        if int(np.count_nonzero(finite)) < 2:
            continue
        x_m, y_m = projection.project(lat[finite], lon[finite])
        polylines[str(flight_id)] = np.column_stack((x_m, y_m))
    return polylines


def _closest_polyline_points_at_stations(
    polyline_xy_m: np.ndarray,
    reference_xy_m: np.ndarray,
) -> np.ndarray:
    polyline = np.asarray(polyline_xy_m, dtype=float)
    reference = np.asarray(reference_xy_m, dtype=float)
    if polyline.ndim != 2 or polyline.shape[1] != 2:
        raise ValueError("polyline_xy_m must have shape N x 2")
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("reference_xy_m must have shape M x 2")
    if polyline.shape[0] < 2:
        raise ValueError("polyline_xy_m must contain at least two points")

    start = polyline[:-1]
    segment = polyline[1:] - start
    segment_length2 = np.sum(segment * segment, axis=1)
    valid = segment_length2 > 1.0e-12
    if not np.any(valid):
        raise ValueError("polyline_xy_m must contain at least one nonzero segment")
    start = start[valid]
    segment = segment[valid]
    segment_length2 = segment_length2[valid]

    samples = np.zeros_like(reference)
    for station_index, station_xy in enumerate(reference):
        delta = station_xy - start
        fraction = np.sum(delta * segment, axis=1) / segment_length2
        fraction = np.clip(fraction, 0.0, 1.0)
        closest = start + fraction[:, None] * segment
        distances2 = np.sum((closest - station_xy) ** 2, axis=1)
        samples[station_index, :] = closest[int(np.argmin(distances2))]
    return samples


def _matrix_sampled_paths(matrix: MatrixArtifact) -> dict[str, np.ndarray]:
    return {
        str(flight_id): matrix.reference_xy_m + matrix.X[row_index, :, None] * matrix.normals_xy
        for row_index, flight_id in enumerate(matrix.flight_ids)
    }


def _trace_lift_tangent_reconstruction(
    matrix: MatrixArtifact,
    model: HLLRDFitResult,
    *,
    center_xy: np.ndarray,
    actual_xy_by_flight: dict[str, np.ndarray],
    mode: str,
) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode == "stored":
        if model.trace_tangent_reconstruction is None:
            raise ValueError("model does not contain a stored trace tangent lift")
        return np.asarray(model.trace_tangent_reconstruction, dtype=float)
    if mode != "registered-dictionary":
        raise ValueError("trace_lift must be 'none', 'stored', or 'registered-dictionary'")
    tangent_residual = trace_tangent_residual_from_samples(
        matrix,
        center_xy=center_xy,
        actual_xy_by_flight=actual_xy_by_flight,
    )
    tangent_center = np.median(tangent_residual, axis=0)
    tangent_centered = tangent_residual - tangent_center[None, :]
    _coefficients, tangent_reconstruction_centered = refit_registered_coefficients(
        tangent_centered,
        model.events,
        ridge=model.config.ridge,
    )
    return tangent_reconstruction_centered + tangent_center[None, :]


def trace_tangent_residual_from_samples(
    matrix: MatrixArtifact,
    *,
    center_xy: np.ndarray,
    actual_xy_by_flight: dict[str, np.ndarray],
) -> np.ndarray:
    tangents = _tangents_from_normals(matrix.normals_xy)
    tangent_residual = np.zeros_like(matrix.X, dtype=float)
    for row_index, flight_id in enumerate(matrix.flight_ids):
        actual_xy = actual_xy_by_flight.get(str(flight_id))
        if actual_xy is None:
            continue
        tangent_residual[row_index, :] = np.sum((actual_xy - center_xy) * tangents, axis=1)
    return tangent_residual


def _tangents_from_normals(normals_xy: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals_xy, dtype=float)
    return np.column_stack((normals[:, 1], -normals[:, 0]))


def _shifted_basis(basis: np.ndarray, lag: int) -> np.ndarray:
    base = np.asarray(basis, dtype=float)
    shifted = np.zeros_like(base)
    lag_int = int(lag)
    if lag_int == 0:
        return base.copy()
    if abs(lag_int) >= base.shape[0]:
        return shifted
    if lag_int > 0:
        shifted[lag_int:, :] = base[:-lag_int, :]
    else:
        shifted[:lag_int, :] = base[-lag_int:, :]
    return shifted


def _raw_polyline_window(
    raw_polyline: np.ndarray,
    station_xy: np.ndarray,
    station_start: int,
    station_end: int,
) -> np.ndarray:
    polyline = np.asarray(raw_polyline, dtype=float)
    if polyline.shape[0] < 2:
        return polyline
    stations = np.asarray(station_xy, dtype=float)
    distances2 = np.sum((polyline[:, None, :] - stations[None, :, :]) ** 2, axis=2)
    nearest_station = np.argmin(distances2, axis=1)
    left = max(0, int(station_start) - 2)
    right = min(stations.shape[0] - 1, int(station_end) + 1)
    keep = (nearest_station >= left) & (nearest_station <= right)
    keep_indices = np.flatnonzero(keep)
    if keep_indices.size:
        start = max(0, int(keep_indices[0]) - 1)
        end = min(polyline.shape[0], int(keep_indices[-1]) + 2)
        if end - start >= 2:
            return polyline[start:end]
    return polyline


def _point_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    query = np.asarray(points, dtype=float)
    line = np.asarray(polyline, dtype=float)
    if query.size == 0:
        return np.zeros(0, dtype=float)
    if line.shape[0] == 0:
        return np.full(query.shape[0], np.inf, dtype=float)
    if line.shape[0] == 1:
        return np.linalg.norm(query - line[0], axis=1)
    start = line[:-1]
    segment = line[1:] - start
    segment_length2 = np.sum(segment * segment, axis=1)
    valid = segment_length2 > 1.0e-12
    if not np.any(valid):
        return np.linalg.norm(query - line[0], axis=1)
    start = start[valid]
    segment = segment[valid]
    segment_length2 = segment_length2[valid]
    delta = query[:, None, :] - start[None, :, :]
    fraction = np.sum(delta * segment[None, :, :], axis=2) / segment_length2[None, :]
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start[None, :, :] + fraction[:, :, None] * segment[None, :, :]
    distances2 = np.sum((query[:, None, :] - closest) ** 2, axis=2)
    return np.sqrt(np.min(distances2, axis=1))


def _summarize_rows(event_index: int, rows: list[EventTraceMatchRow]) -> dict[str, Any]:
    active_rows = [row for row in rows if row.active]
    center_rmse = _mean_attr(active_rows, "center_trace_rmse_m")
    event_rmse = _mean_attr(active_rows, "event_trace_rmse_m")
    model_rmse = _mean_attr(active_rows, "model_trace_rmse_m")
    lifted_model_rmse = _mean_attr(active_rows, "lifted_model_trace_rmse_m")
    center_normal_rmse = _mean_attr(active_rows, "center_normal_rmse_m")
    event_normal_rmse = _mean_attr(active_rows, "event_normal_rmse_m")
    model_normal_rmse = _mean_attr(active_rows, "model_normal_rmse_m")
    return {
        "event": int(event_index),
        "flight_count": len(rows),
        "active_flight_count": len(active_rows),
        "center_trace_rmse_m": center_rmse,
        "event_trace_rmse_m": event_rmse,
        "model_trace_rmse_m": model_rmse,
        "lifted_model_trace_rmse_m": lifted_model_rmse,
        "event_trace_rmse_reduction_fraction": _reduction_fraction(center_rmse, event_rmse),
        "model_trace_rmse_reduction_fraction": _reduction_fraction(center_rmse, model_rmse),
        "lifted_model_trace_rmse_reduction_fraction": _reduction_fraction(center_rmse, lifted_model_rmse),
        "event_improves_center_fraction": _improvement_fraction(active_rows, "event_trace_rmse_m"),
        "model_improves_center_fraction": _improvement_fraction(active_rows, "model_trace_rmse_m"),
        "lifted_model_improves_center_fraction": _improvement_fraction(active_rows, "lifted_model_trace_rmse_m"),
        "event_trace_p95_m": _mean_attr(active_rows, "event_trace_p95_m"),
        "model_trace_p95_m": _mean_attr(active_rows, "model_trace_p95_m"),
        "lifted_model_trace_p95_m": _mean_attr(active_rows, "lifted_model_trace_p95_m"),
        "center_normal_rmse_m": center_normal_rmse,
        "event_normal_rmse_m": event_normal_rmse,
        "model_normal_rmse_m": model_normal_rmse,
        "event_normal_rmse_reduction_fraction": _reduction_fraction(center_normal_rmse, event_normal_rmse),
        "model_normal_rmse_reduction_fraction": _reduction_fraction(center_normal_rmse, model_normal_rmse),
        "event_normal_improves_center_fraction": _improvement_fraction(active_rows, "event_normal_rmse_m", baseline="center_normal_rmse_m"),
        "model_normal_improves_center_fraction": _improvement_fraction(active_rows, "model_normal_rmse_m", baseline="center_normal_rmse_m"),
    }


def _mean_attr(rows: list[EventTraceMatchRow], attribute: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(getattr(row, attribute)) for row in rows]))


def _reduction_fraction(baseline: float, value: float) -> float:
    if float(baseline) <= 0.0:
        return 0.0
    return float((float(baseline) - float(value)) / float(baseline))


def _improvement_fraction(
    rows: list[EventTraceMatchRow],
    attribute: str,
    *,
    baseline: str = "center_trace_rmse_m",
) -> float:
    if not rows:
        return 0.0
    improved = [float(getattr(row, attribute)) < float(getattr(row, baseline)) for row in rows]
    return float(np.mean(improved))


def _xy_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    distances = np.linalg.norm(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float), axis=1)
    return _rms(distances)


def _xy_percentile_error(actual: np.ndarray, predicted: np.ndarray, percentile: float) -> float:
    distances = np.linalg.norm(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float), axis=1)
    return _percentile(distances, percentile)


def _percentile(values: np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.percentile(array, float(percentile)))


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(array * array)))
