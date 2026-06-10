from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "hllrd").exists():
            return candidate
    raise RuntimeError("Could not find the project root.")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hllrd.elastic_fpca import load_elastic_fpca_result  # noqa: E402
from hllrd.fit import load_fit_result  # noqa: E402
from hllrd.geometry import LocalProjection, cumulative_distance_m  # noqa: E402
from hllrd.matrix import load_matrix_artifact  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "hllrd" / "south-east"
DEFAULT_MATRIX = DEFAULT_OUTPUT_DIR / "matrix_SE_from_merge.npz"
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "model_from_merge_L40_K6.npz"
DEFAULT_ELASTIC_FPCA = DEFAULT_OUTPUT_DIR / "elastic_fpca_from_merge_L40_K6.npz"
NM_PER_M = 1.0 / 1852.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the horizontal FPCA warp for one South-East event flight. "
            "By default this inspects Event 1 and the flight with the largest "
            "absolute horizontal PC1 score."
        )
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="Path to matrix_SE_from_merge.npz.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to model_from_merge_L40_K6.npz.")
    parser.add_argument(
        "--elastic-fpca",
        type=Path,
        default=DEFAULT_ELASTIC_FPCA,
        help="Path to elastic_fpca_from_merge_L40_K6.npz.",
    )
    parser.add_argument("--event", type=int, default=1, help="HLLRD event index to inspect. Defaults to Event 1.")
    parser.add_argument(
        "--component",
        type=int,
        default=1,
        help="One-based horizontal PC component used for scores in the summary. Defaults to PC1.",
    )
    parser.add_argument(
        "--flight",
        type=str,
        default=None,
        help=(
            "Flight ID, row:<matrix row>, local:<active-event row>, or a bare numeric matrix row to inspect. "
            "If omitted, choose the active flight with the largest absolute horizontal PC score."
        ),
    )
    parser.add_argument(
        "--station-step",
        type=int,
        default=1,
        help="Print every Nth event station in the mapping table. Defaults to every station.",
    )
    parser.add_argument(
        "--line-step",
        type=int,
        default=4,
        help="Draw one geographic correspondence line every N event stations.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path for the detailed station mapping table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Figure output path. Defaults to data/hllrd/south-east/hfpca_event<E>_<flight>.png.",
    )
    parser.add_argument("--show", action="store_true", help="Show the Matplotlib window after saving.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save a figure. Useful with --show.",
    )
    return parser


def mean_xy_m(matrix, model) -> np.ndarray:
    return matrix.reference_xy_m + model.column_center[:, None] * matrix.normals_xy


def flight_xy_m(matrix, flight_row: int) -> np.ndarray:
    return matrix.reference_xy_m + matrix.X[flight_row, :, None] * matrix.normals_xy


def resolve_active_flight(event, matrix, component_index: int, selector: str | None) -> tuple[int, int]:
    active_rows = np.asarray(event.active_rows, dtype=int)
    if active_rows.size == 0:
        raise RuntimeError(f"Event {event.event_index} has no active flights.")

    if selector is None:
        scores = np.asarray(event.horizontal_coefficients[:, component_index], dtype=float)
        local_index = int(np.nanargmax(np.abs(scores)))
        return local_index, int(active_rows[local_index])

    if selector.startswith("local:"):
        local_index = int(selector.removeprefix("local:"))
        if 0 <= local_index < active_rows.size:
            return local_index, int(active_rows[local_index])
        raise ValueError(f"Active local row must be in [0, {active_rows.size - 1}].")

    if selector.startswith("row:"):
        row_index = int(selector.removeprefix("row:"))
        matches = np.flatnonzero(active_rows == row_index)
        if matches.size == 0:
            raise ValueError(f"Matrix row {row_index} is not active in Event {event.event_index}.")
        return int(matches[0]), row_index

    if selector in event.active_flight_ids:
        local_index = int(event.active_flight_ids.index(selector))
        return local_index, int(active_rows[local_index])

    if selector in matrix.flight_ids:
        row_index = int(matrix.flight_ids.index(selector))
        matches = np.flatnonzero(active_rows == row_index)
        if matches.size == 0:
            raise ValueError(f"Flight {selector!r} is in the matrix but is not active in Event {event.event_index}.")
        return int(matches[0]), row_index

    try:
        numeric = int(selector)
    except ValueError as exc:
        raise ValueError(
            f"Unknown flight selector {selector!r}; pass a flight ID, row:<matrix row>, or local:<active row>."
        ) from exc

    matches = np.flatnonzero(active_rows == numeric)
    if matches.size:
        return int(matches[0]), int(numeric)
    raise ValueError(
        f"Numeric matrix row {numeric} is not active in Event {event.event_index}. "
        f"Use local:{numeric} if you meant active local row {numeric}."
    )


def monotone_inverse(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    monotone_x = np.maximum.accumulate(np.asarray(x, dtype=float))
    unique_x, unique_indices = np.unique(monotone_x, return_index=True)
    if unique_x.size < 2:
        return np.asarray(x_new, dtype=float).copy()
    return np.interp(x_new, unique_x, np.asarray(y, dtype=float)[unique_indices])


def interpolate_station_path(path_xy_m: np.ndarray, station_coordinate: np.ndarray) -> np.ndarray:
    stations = np.arange(path_xy_m.shape[0], dtype=float)
    clipped = np.clip(np.asarray(station_coordinate, dtype=float), stations[0], stations[-1])
    x = np.interp(clipped, stations, path_xy_m[:, 0])
    y = np.interp(clipped, stations, path_xy_m[:, 1])
    return np.column_stack((x, y))


def build_mapping_rows(
    *,
    event,
    local_index: int,
    matrix,
    model,
    station_distance_nm: np.ndarray,
    station_step: int,
) -> list[dict[str, float | int | str]]:
    start = int(event.start)
    end = int(event.end)
    length = int(event.length)
    local_template_station = np.arange(length, dtype=float)
    global_template_station = local_template_station + start

    gamma = np.asarray(event.warps[:, local_index], dtype=float)
    mapped_local_flight_station = gamma * float(length - 1)
    mapped_global_flight_station = mapped_local_flight_station + start
    mapped_floor = np.floor(mapped_global_flight_station).astype(int)
    mapped_ceil = np.ceil(mapped_global_flight_station).astype(int)

    original_function = np.asarray(event.functions[:, local_index], dtype=float)
    original_at_gamma = np.interp(gamma, event.time, original_function)
    aligned_function = np.asarray(event.aligned_functions[:, local_index], dtype=float)
    fmean = np.asarray(event.fmean, dtype=float)

    raw_offset = np.asarray(matrix.X[int(event.active_rows[local_index]), start:end], dtype=float)
    centered_offset = raw_offset - np.asarray(model.column_center[start:end], dtype=float)

    rows: list[dict[str, float | int | str]] = []
    for j in range(0, length, max(1, int(station_step))):
        rows.append(
            {
                "event_index": int(event.event_index),
                "local_template_station": int(j),
                "template_station": int(global_template_station[j]),
                "template_nm": float(station_distance_nm[int(global_template_station[j])]),
                "gamma_template_to_flight": float(gamma[j]),
                "mapped_flight_station": float(mapped_global_flight_station[j]),
                "mapped_flight_station_floor": int(mapped_floor[j]),
                "mapped_flight_station_ceil": int(mapped_ceil[j]),
                "warp_shift_stations": float(mapped_global_flight_station[j] - global_template_station[j]),
                "raw_normal_offset_m": float(raw_offset[j]),
                "centered_normal_offset_m": float(centered_offset[j]),
                "original_centered_at_gamma_m": float(original_at_gamma[j]),
                "aligned_centered_m": float(aligned_function[j]),
                "template_fmean_m": float(fmean[j]),
                "aligned_minus_fmean_m": float(aligned_function[j] - fmean[j]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise RuntimeError("No mapping rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    *,
    event,
    matrix,
    local_index: int,
    row_index: int,
    component_index: int,
    station_distance_nm: np.ndarray,
    rows: list[dict[str, float | int | str]],
) -> None:
    scores = np.asarray(event.horizontal_coefficients[:, component_index], dtype=float)
    score = float(scores[local_index])
    score_std = float(np.std(scores))
    score_z = float((score - float(np.mean(scores))) / score_std) if score_std > 1.0e-12 else 0.0
    gamma = np.asarray(event.warps[:, local_index], dtype=float)
    local_station = np.arange(event.length, dtype=float)
    mapped_station = gamma * float(event.length - 1)
    displacement = mapped_station - local_station
    slopes = np.diff(gamma) / np.diff(event.time) if event.length > 1 else np.array([], dtype=float)
    active_fraction = 100.0 * float(event.active_count) / float(len(matrix.flight_ids))
    start_nm = float(station_distance_nm[event.start])
    end_nm = float(station_distance_nm[event.end - 1])
    reconstruction = np.interp(gamma, event.time, event.functions[:, local_index])
    aligned_error = float(np.max(np.abs(reconstruction - event.aligned_functions[:, local_index])))

    print()
    print(f"Event {event.event_index}: stations {event.start}:{event.end} ({start_nm:.2f}-{end_nm:.2f} NM)")
    print(f"Active flights: {event.active_count}/{len(matrix.flight_ids)} ({active_fraction:.1f}%)")
    print(f"Flight: {matrix.flight_ids[row_index]} (matrix row {row_index}, active local row {local_index})")
    print(f"Horizontal PC{component_index + 1} score: {score:.6g} (z={score_z:.3f})")
    print(
        "Warp displacement, in event-local station units: "
        f"min={float(np.min(displacement)):.2f}, "
        f"median={float(np.median(displacement)):.2f}, "
        f"max={float(np.max(displacement)):.2f}"
    )
    if slopes.size:
        print(
            "Warp local stretch d gamma / d template_time: "
            f"min={float(np.min(slopes)):.3f}, "
            f"median={float(np.median(slopes)):.3f}, "
            f"max={float(np.max(slopes)):.3f}"
        )
    print(f"Plain gamma resampling vs stored aligned trace, max abs diff: {aligned_error:.6g} m")
    print()
    print(
        "Mapping convention: gamma(template station) gives the fractional station on the selected flight "
        "used to align that flight to the template."
    )
    print()
    print(
        f"{'tmpl':>5} {'tmpl_nm':>8} {'gamma':>8} {'flight_stn':>10} {'shift':>8} "
        f"{'orig@gamma_m':>13} {'aligned_m':>11} {'fmean_m':>10}"
    )
    for row in rows:
        print(
            f"{int(row['template_station']):5d} "
            f"{float(row['template_nm']):8.2f} "
            f"{float(row['gamma_template_to_flight']):8.4f} "
            f"{float(row['mapped_flight_station']):10.2f} "
            f"{float(row['warp_shift_stations']):8.2f} "
            f"{float(row['original_centered_at_gamma_m']):13.2f} "
            f"{float(row['aligned_centered_m']):11.2f} "
            f"{float(row['template_fmean_m']):10.2f}"
        )


def make_figure(
    *,
    event,
    matrix,
    projection: LocalProjection,
    mean_path_xy_m: np.ndarray,
    flight_path_xy_m: np.ndarray,
    local_index: int,
    row_index: int,
    component_index: int,
    station_distance_nm: np.ndarray,
    line_step: int,
    output_path: Path | None,
    show: bool,
) -> None:
    start = int(event.start)
    end = int(event.end)
    length = int(event.length)
    local_template_station = np.arange(length, dtype=float)
    global_template_station = local_template_station + start
    gamma = np.asarray(event.warps[:, local_index], dtype=float)
    mapped_local_flight_station = gamma * float(length - 1)
    mapped_global_flight_station = mapped_local_flight_station + start

    inverse_template_from_flight = monotone_inverse(
        mapped_global_flight_station,
        global_template_station,
        global_template_station,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=True)
    ax_map, ax_gamma, ax_function, ax_inverse = axes.ravel()

    mean_lat, mean_lon = projection.unproject(mean_path_xy_m[:, 0], mean_path_xy_m[:, 1])
    flight_lat, flight_lon = projection.unproject(flight_path_xy_m[:, 0], flight_path_xy_m[:, 1])
    ax_map.plot(mean_lon, mean_lat, color="#303030", linewidth=1.8, alpha=0.7, label="template center")
    ax_map.plot(
        flight_lon,
        flight_lat,
        color="#2166ac",
        linewidth=1.0,
        alpha=0.45,
        label=f"flight {matrix.flight_ids[row_index]}",
    )

    mean_segment = mean_path_xy_m[start:end]
    mapped_flight_segment = interpolate_station_path(flight_path_xy_m, mapped_global_flight_station)
    mean_segment_lat, mean_segment_lon = projection.unproject(mean_segment[:, 0], mean_segment[:, 1])
    mapped_lat, mapped_lon = projection.unproject(mapped_flight_segment[:, 0], mapped_flight_segment[:, 1])
    ax_map.plot(mean_segment_lon, mean_segment_lat, color="black", linewidth=4.0, alpha=0.85, label="event template")
    ax_map.scatter(mean_segment_lon, mean_segment_lat, s=14, color="black", alpha=0.75)
    ax_map.scatter(mapped_lon, mapped_lat, s=15, color="#b2182b", alpha=0.85, label="mapped flight points")

    for j in range(0, length, max(1, int(line_step))):
        ax_map.plot(
            [mean_segment_lon[j], mapped_lon[j]],
            [mean_segment_lat[j], mapped_lat[j]],
            color="#b2182b",
            linewidth=0.75,
            alpha=0.55,
        )
    ax_map.set_title("Geographic station correspondences")
    ax_map.set_xlabel("longitude")
    ax_map.set_ylabel("latitude")
    ax_map.set_aspect(1.0 / np.cos(np.deg2rad(float(np.mean(mean_lat)))))
    ax_map.grid(True, alpha=0.25)
    ax_map.legend(loc="best", fontsize=8)

    ax_gamma.plot(global_template_station, mapped_global_flight_station, color="#b2182b", linewidth=2.0)
    ax_gamma.plot(global_template_station, global_template_station, color="#555555", linestyle="--", linewidth=1.2)
    ax_gamma.set_title("Warp: template station to flight station")
    ax_gamma.set_xlabel("template global station")
    ax_gamma.set_ylabel("mapped flight global station")
    ax_gamma.grid(True, alpha=0.25)

    ax_function.plot(global_template_station, event.functions[:, local_index], color="#2166ac", label="original centered")
    ax_function.plot(
        global_template_station,
        np.interp(gamma, event.time, event.functions[:, local_index]),
        color="#b2182b",
        label="plain original sampled at gamma",
    )
    ax_function.plot(
        global_template_station,
        event.aligned_functions[:, local_index],
        color="#ef8a62",
        linestyle="--",
        label="stored aligned",
    )
    ax_function.plot(global_template_station, event.fmean, color="black", linewidth=1.8, label="template fmean")
    ax_function.set_title("Normal-offset function alignment")
    ax_function.set_xlabel("template global station")
    ax_function.set_ylabel("centered normal offset (m)")
    ax_function.grid(True, alpha=0.25)
    ax_function.legend(loc="best", fontsize=8)

    ax_inverse.plot(global_template_station, inverse_template_from_flight, color="#1b9e77", linewidth=2.0)
    ax_inverse.plot(global_template_station, global_template_station, color="#555555", linestyle="--", linewidth=1.2)
    ax_inverse.set_title("Inverse view: flight station to template station")
    ax_inverse.set_xlabel("flight global station")
    ax_inverse.set_ylabel("mapped template global station")
    ax_inverse.grid(True, alpha=0.25)

    score = float(event.horizontal_coefficients[local_index, component_index])
    start_nm = float(station_distance_nm[event.start])
    end_nm = float(station_distance_nm[event.end - 1])
    fig.suptitle(
        f"South-East horizontal FPCA Event {event.event_index} warp inspection | "
        f"{matrix.flight_ids[row_index]} | PC{component_index + 1} score={score:.4g} | "
        f"{start_nm:.1f}-{end_nm:.1f} NM",
        fontsize=13,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    matrix = load_matrix_artifact(args.matrix)
    model = load_fit_result(args.model)
    elastic = load_elastic_fpca_result(args.elastic_fpca)
    event = elastic.event_by_index(int(args.event))

    if matrix.X.shape != model.residual.shape:
        raise RuntimeError("Matrix and model residual shapes do not match.")
    if event.end > matrix.X.shape[1]:
        raise RuntimeError(f"Event {event.event_index} extends beyond the matrix station count.")

    component_index = int(args.component) - 1
    if component_index < 0 or component_index >= event.horizontal_coefficients.shape[1]:
        raise ValueError(f"--component must be in [1, {event.horizontal_coefficients.shape[1]}].")

    local_index, row_index = resolve_active_flight(event, matrix, component_index, args.flight)
    projection = LocalProjection(matrix.origin_lat_deg, matrix.origin_lon_deg)
    mean_path_xy_m = mean_xy_m(matrix, model)
    selected_flight_xy_m = flight_xy_m(matrix, row_index)
    station_distance_nm = cumulative_distance_m(mean_path_xy_m[:, 0], mean_path_xy_m[:, 1]) * NM_PER_M

    rows = build_mapping_rows(
        event=event,
        local_index=local_index,
        matrix=matrix,
        model=model,
        station_distance_nm=station_distance_nm,
        station_step=max(1, int(args.station_step)),
    )
    print_summary(
        event=event,
        matrix=matrix,
        local_index=local_index,
        row_index=row_index,
        component_index=component_index,
        station_distance_nm=station_distance_nm,
        rows=rows,
    )
    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"Saved CSV: {args.csv}")

    output_path = args.output
    if output_path is None and not args.no_save:
        safe_flight_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in matrix.flight_ids[row_index])
        output_path = DEFAULT_OUTPUT_DIR / f"hfpca_event{event.event_index}_{safe_flight_id}_warp.png"
    make_figure(
        event=event,
        matrix=matrix,
        projection=projection,
        mean_path_xy_m=mean_path_xy_m,
        flight_path_xy_m=selected_flight_xy_m,
        local_index=local_index,
        row_index=row_index,
        component_index=component_index,
        station_distance_nm=station_distance_nm,
        line_step=max(1, int(args.line_step)),
        output_path=None if args.no_save else output_path,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
