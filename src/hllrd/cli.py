from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from hllrd.data import filter_tracks_to_cluster, load_cluster_flights, normalize_cluster, trim_tracks_from_anchor
from hllrd.evaluate import (
    center_path_xy,
    closest_raw_track_points_to_matrix_stations,
    evaluate_event_trace_match,
    plot_event_trace_overlay,
    trace_tangent_residual_from_samples,
)
from hllrd.fit import (
    HLLRDV1Config,
    HLLRDV2Config,
    augment_with_trace_tangent_lift,
    fit_localized_low_rank,
    generate_candidate_summary,
    load_fit_result,
    save_fit_result,
    transform_with_model,
)
from hllrd.matrix import MatrixBuildConfig, load_matrix_artifact, save_matrix_artifact, build_matrix_from_tracks
from hllrd.report import (
    plot_reconstruction_heatmap,
    plot_residual_energy,
    write_event_summary_csv,
    write_model_summary_json,
)
from scenario.trajectory_compressor.io import load_raw_adsb, split_tracks_by_gap


DEFAULT_RAW_ADSB_DIR = Path("data/adsb/raw")
DEFAULT_CLUSTER_ARTIFACTS = Path("data/artifacts/simap_arrival_flights.jsonl")
DEFAULT_SPLIT_GAP_SECONDS = 25 * 60


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HLLRD V1 localized low-rank analysis for clustered ADS-B arrivals.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_matrix_parser = subparsers.add_parser("build-matrix", description="Build an aligned normal-residual matrix.")
    _add_build_matrix_args(build_matrix_parser)
    build_matrix_parser.set_defaults(func=run_build_matrix)

    candidates_parser = subparsers.add_parser("candidates", description="Generate V1 candidate interval diagnostics.")
    _add_candidates_args(candidates_parser)
    candidates_parser.set_defaults(func=run_candidates)

    fit_parser = subparsers.add_parser("fit", description="Fit the V1 localized low-rank model.")
    _add_fit_args(fit_parser)
    fit_parser.set_defaults(func=run_fit)

    transform_parser = subparsers.add_parser("transform", description="Apply a fitted model to another matrix.")
    _add_transform_args(transform_parser)
    transform_parser.set_defaults(func=run_transform)

    report_parser = subparsers.add_parser("report", description="Write model summaries and plots.")
    _add_report_args(report_parser)
    report_parser.set_defaults(func=run_report)

    evaluate_parser = subparsers.add_parser("evaluate-event", description="Evaluate one fitted event against ADS-B traces.")
    _add_evaluate_event_args(evaluate_parser)
    evaluate_parser.set_defaults(func=run_evaluate_event)

    augment_trace_parser = subparsers.add_parser("augment-trace", description="Store a tangential ADS-B trace lift in a fitted model.")
    _add_augment_trace_args(augment_trace_parser)
    augment_trace_parser.set_defaults(func=run_augment_trace)

    run_parser = subparsers.add_parser("run", description="Run build-matrix, candidates, fit, and report.")
    _add_run_args(run_parser)
    run_parser.set_defaults(func=run_pipeline)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def build_matrix_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build an HLLRD aligned normal-residual matrix.")
    _add_build_matrix_args(parser)
    run_build_matrix(parser.parse_args(argv))


def candidates_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate HLLRD candidate diagnostics.")
    _add_candidates_args(parser)
    run_candidates(parser.parse_args(argv))


def fit_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit an HLLRD V1 model.")
    _add_fit_args(parser)
    run_fit(parser.parse_args(argv))


def transform_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply an HLLRD V1 model.")
    _add_transform_args(parser)
    run_transform(parser.parse_args(argv))


def report_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate HLLRD model reports.")
    _add_report_args(parser)
    run_report(parser.parse_args(argv))


def evaluate_event_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate one HLLRD event against ADS-B traces.")
    _add_evaluate_event_args(parser)
    run_evaluate_event(parser.parse_args(argv))


def augment_trace_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Store a tangential ADS-B trace lift in a fitted HLLRD model.")
    _add_augment_trace_args(parser)
    run_augment_trace(parser.parse_args(argv))


def run_build_matrix(args: argparse.Namespace) -> None:
    console = Console()
    cluster = normalize_cluster(args.cluster)
    flights = load_cluster_flights(args.cluster_artifacts, cluster)
    if args.max_flights is not None:
        flights = flights[: args.max_flights]
    if not flights:
        raise SystemExit(f"No flights found for cluster {cluster} in {args.cluster_artifacts}")

    console.print(f"[bold]Loading raw ADS-B[/bold] from {args.raw_adsb_dir}")
    tracks = load_raw_adsb(args.raw_adsb_dir, args.processes)
    tracks = split_tracks_by_gap(tracks, args.split_gap_seconds)
    tracks = filter_tracks_to_cluster(tracks, flights)
    if tracks.empty:
        raise SystemExit(f"No raw ADS-B rows matched {len(flights)} {cluster} cluster flights")

    config = MatrixBuildConfig(
        station_count=args.stations,
        min_points_per_flight=args.min_points,
        center_method=args.center_method,
    )
    artifact = build_matrix_from_tracks(
        tracks,
        config=config,
        cluster=cluster,
        metadata={
            "raw_adsb_dir": args.raw_adsb_dir.as_posix(),
            "cluster_artifacts": args.cluster_artifacts.as_posix(),
            "cluster_flight_count": len(flights),
            "raw_row_count": int(len(tracks)),
        },
    )
    save_matrix_artifact(args.output, artifact)
    _print_matrix_summary(console, artifact, args.output)


def run_candidates(args: argparse.Namespace) -> None:
    matrix = load_matrix_artifact(args.matrix)
    config = _fit_config_from_args(args)
    rows = generate_candidate_summary(matrix.X_centered, config, already_centered=True)
    if args.output.suffix.lower() == ".json":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2, sort_keys=True)
            stream.write("\n")
    else:
        _write_rows_csv(args.output, rows)
    if args.plot is not None:
        plot_residual_energy(args.plot, matrix.X_centered)
    Console().print(f"[green]Wrote[/green] {len(rows):,} candidate rows to {args.output}")


def run_fit(args: argparse.Namespace) -> None:
    matrix = load_matrix_artifact(args.matrix)
    config = _fit_config_from_args(args)
    result = fit_localized_low_rank(
        matrix.X,
        config,
        already_centered=False,
        metadata={
            "matrix": args.matrix.as_posix(),
            "cluster": matrix.cluster,
            "flight_count": len(matrix.flight_ids),
            "config": asdict(config),
        },
    )
    save_fit_result(args.output, result, flight_ids=matrix.flight_ids)
    if args.summary is not None:
        write_model_summary_json(args.summary, result)
    if args.events_csv is not None:
        write_event_summary_csv(args.events_csv, result)
    if args.energy_plot is not None:
        plot_residual_energy(args.energy_plot, matrix.X_centered, result)
    if args.heatmap is not None:
        plot_reconstruction_heatmap(args.heatmap, result)
    _print_fit_summary(Console(), result, args.output)


def run_transform(args: argparse.Namespace) -> None:
    matrix = load_matrix_artifact(args.matrix)
    model = load_fit_result(args.model)
    X = matrix.X_centered if args.already_centered else matrix.X
    transform = transform_with_model(X, model, already_centered=args.already_centered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        coefficients=transform.coefficients,
        reconstruction=transform.reconstruction,
        residual=transform.residual,
        active_counts=transform.active_counts,
        lag_offsets=(
            transform.lag_offsets
            if transform.lag_offsets is not None
            else np.zeros((matrix.X.shape[0], len(model.events)), dtype=int)
        ),
        extension_offsets=(
            transform.extension_offsets
            if transform.extension_offsets is not None
            else np.zeros((matrix.X.shape[0], len(model.events)), dtype=int)
        ),
        explained_fraction=np.asarray(transform.explained_fraction, dtype=float),
        matrix=np.asarray(args.matrix.as_posix(), dtype=str),
        model=np.asarray(args.model.as_posix(), dtype=str),
    )
    if args.summary is not None:
        payload = {
            "explained_fraction": transform.explained_fraction,
            "average_active_events_per_flight": float(np.mean(transform.active_counts)) if transform.active_counts.size else 0.0,
            "matrix": args.matrix.as_posix(),
            "model": args.model.as_posix(),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    Console().print(f"[green]Wrote[/green] transform artifact to {args.output}")


def run_report(args: argparse.Namespace) -> None:
    result = load_fit_result(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_model_summary_json(args.output_dir / "summary.json", result)
    write_event_summary_csv(args.output_dir / "events.csv", result)
    plot_reconstruction_heatmap(args.output_dir / "reconstruction_heatmap.png", result)
    if args.matrix is not None:
        matrix = load_matrix_artifact(args.matrix)
        plot_residual_energy(args.output_dir / "residual_energy.png", matrix.X_centered, result)
    Console().print(f"[green]Wrote[/green] report files to {args.output_dir}")


def run_evaluate_event(args: argparse.Namespace) -> None:
    matrix = load_matrix_artifact(args.matrix)
    model = load_fit_result(args.model)
    raw_tracks = None
    if args.trace_source == "raw":
        raw_tracks = _load_raw_tracks_for_event_evaluation(matrix, args)
    result = evaluate_event_trace_match(
        matrix,
        model,
        event_index=args.event,
        raw_tracks=raw_tracks,
        active_only=not args.include_inactive,
        trace_lift=args.trace_lift,
    )
    _write_rows_csv(args.output, result.row_dicts())
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as stream:
            json.dump(result.summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
    if args.plot is not None:
        plot_event_trace_overlay(
            args.plot,
            matrix,
            model,
            event_index=args.event,
            raw_tracks=raw_tracks,
            max_flights=args.plot_max_flights,
            trace_lift=args.trace_lift,
        )
    message = (
        f"[green]Wrote[/green] {len(result.rows):,} event-trace rows to {args.output} "
        f"(event RMSE {result.summary['event_trace_rmse_m']:.1f} m, "
        f"model RMSE {result.summary['model_trace_rmse_m']:.1f} m"
    )
    if args.trace_lift != "none":
        message += f", lifted model RMSE {result.summary['lifted_model_trace_rmse_m']:.1f} m"
    Console().print(message + ")")


def run_augment_trace(args: argparse.Namespace) -> None:
    matrix = load_matrix_artifact(args.matrix)
    model = load_fit_result(args.model)
    raw_tracks = _load_raw_tracks_for_event_evaluation(matrix, args)
    actual_xy_by_flight = closest_raw_track_points_to_matrix_stations(raw_tracks, matrix)
    tangent_residual = trace_tangent_residual_from_samples(
        matrix,
        center_xy=center_path_xy(matrix, model),
        actual_xy_by_flight=actual_xy_by_flight,
    )
    augmented = augment_with_trace_tangent_lift(
        model,
        tangent_residual,
        center_method=args.trace_center_method,
        ridge=args.trace_ridge,
        extra_residual_config=_trace_extra_residual_config_from_args(args),
    )
    save_fit_result(args.output, augmented, flight_ids=matrix.flight_ids)
    if args.summary is not None:
        write_model_summary_json(args.summary, augmented)
    info = augmented.metadata.get("trace_tangent_lift", {})
    Console().print(
        f"[green]Wrote[/green] trace-augmented model to {args.output} "
        f"(tangent explained {float(info.get('explained_fraction', 0.0)):.3%})"
    )


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"matrix_{normalize_cluster(args.cluster)}.npz"
    candidates_path = output_dir / "candidates.csv"
    model_path = output_dir / "model.npz"
    summary_path = output_dir / "summary.json"
    events_path = output_dir / "events.csv"
    energy_plot = output_dir / "residual_energy.png"
    heatmap = output_dir / "reconstruction_heatmap.png"

    build_args = argparse.Namespace(**vars(args), output=matrix_path)
    run_build_matrix(build_args)

    candidates_args = argparse.Namespace(**vars(args), matrix=matrix_path, output=candidates_path, plot=None)
    run_candidates(candidates_args)

    fit_args = argparse.Namespace(
        **vars(args),
        matrix=matrix_path,
        output=model_path,
        summary=summary_path,
        events_csv=events_path,
        energy_plot=energy_plot,
        heatmap=heatmap,
    )
    run_fit(fit_args)


def _add_build_matrix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-adsb-dir", type=Path, default=DEFAULT_RAW_ADSB_DIR)
    parser.add_argument("--cluster-artifacts", type=Path, default=DEFAULT_CLUSTER_ARTIFACTS)
    parser.add_argument("--cluster", required=True, help="Arrival cluster: NE, NW, SE, or SW.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stations", type=int, default=200)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--center-method", choices=["median", "mean"], default="median")
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    parser.add_argument("--processes", type=int, default=max(cpu_count() - 1, 1))
    parser.add_argument("--max-flights", type=int, default=None)


def _add_fit_tuning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--L-min", dest="L_min", type=int, default=None)
    parser.add_argument("--L-max", dest="L_max", type=int, default=None)
    parser.add_argument("--kappa-peak", type=float, default=2.0)
    parser.add_argument("--min-peak-distance", type=int, default=None)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--activation-scale", type=float, default=1.0)
    parser.add_argument("--n-min", type=int, default=None)
    parser.add_argument("--K-max", dest="K_max", type=int, default=None)
    parser.add_argument("--epsilon-gain", type=float, default=0.001)
    parser.add_argument("--lambda-i", type=float, default=0.0)
    parser.add_argument("--lambda-activation", type=float, default=0.0)
    parser.add_argument("--c-null", type=float, default=4.0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--endpoint-trim-threshold", type=float, default=0.05)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.8)
    parser.add_argument("--keep-next-longer", action="store_true")
    parser.add_argument("--no-peak-backtrack", dest="peak_backtrack_enabled", action="store_false")
    parser.add_argument("--peak-backtrack-rise-fraction", type=float, default=0.05)
    parser.add_argument("--no-local-simplifier", dest="local_simplifier_enabled", action="store_false")
    parser.add_argument("--local-simplifier-gain-sigma", type=float, default=128.0)
    parser.add_argument("--local-simplifier-max-points", type=int, default=4)
    parser.add_argument("--lag-registered", action="store_true", help="Enable V2 per-flight station-lag registration.")
    parser.add_argument("--max-lag-stations", type=int, default=0)
    parser.add_argument("--lag-direction", choices=["both", "nonnegative", "nonpositive"], default="both")
    parser.add_argument("--lag-penalty", type=float, default=0.0)
    parser.add_argument("--max-extend-stations", type=int, default=0)
    parser.add_argument("--extend-direction", choices=["both", "nonnegative", "nonpositive"], default="nonnegative")
    parser.add_argument("--extend-penalty", type=float, default=0.0)
    parser.add_argument("--registration-iterations", type=int, default=5)
    parser.add_argument("--registration-tolerance", type=int, default=0)


def _add_candidates_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, default=None)
    _add_fit_tuning_args(parser)


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--events-csv", type=Path, default=None)
    parser.add_argument("--energy-plot", type=Path, default=None)
    parser.add_argument("--heatmap", type=Path, default=None)
    _add_fit_tuning_args(parser)


def _add_transform_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--already-centered", action="store_true")


def _add_evaluate_event_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--event", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--trace-source", choices=["raw", "matrix"], default="raw")
    parser.add_argument(
        "--trace-lift",
        choices=["none", "stored", "registered-dictionary"],
        default="none",
        help="Optionally reconstruct tangential displacement from raw ADS-B or from a stored model lift.",
    )
    parser.add_argument("--raw-adsb-dir", type=Path, default=None)
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--plot-max-flights", type=int, default=60)


def _add_augment_trace_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--raw-adsb-dir", type=Path, default=None)
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--trace-center-method", choices=["median", "mean", "none"], default="median")
    parser.add_argument("--trace-ridge", type=float, default=None)
    parser.add_argument("--trace-extra-events", action="store_true")
    parser.add_argument("--trace-extra-L-min", dest="trace_extra_L_min", type=int, default=10)
    parser.add_argument("--trace-extra-L-max", dest="trace_extra_L_max", type=int, default=10)
    parser.add_argument("--trace-extra-K-max", dest="trace_extra_K_max", type=int, default=20)


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-adsb-dir", type=Path, default=DEFAULT_RAW_ADSB_DIR)
    parser.add_argument("--cluster-artifacts", type=Path, default=DEFAULT_CLUSTER_ARTIFACTS)
    parser.add_argument("--cluster", required=True, help="Arrival cluster: NE, NW, SE, or SW.")
    parser.add_argument("--stations", type=int, default=200)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--center-method", choices=["median", "mean"], default="median")
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    parser.add_argument("--processes", type=int, default=max(cpu_count() - 1, 1))
    parser.add_argument("--max-flights", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    _add_fit_tuning_args(parser)


def _fit_config_from_args(args: argparse.Namespace) -> HLLRDV1Config:
    base_kwargs = dict(
        L_min=args.L_min,
        L_max=args.L_max,
        kappa_peak=args.kappa_peak,
        min_peak_distance=args.min_peak_distance,
        smoothing_window=args.smoothing_window,
        activation_scale=args.activation_scale,
        n_min=args.n_min,
        K_max=args.K_max,
        epsilon_gain=args.epsilon_gain,
        lambda_i=args.lambda_i,
        lambda_activation=args.lambda_activation,
        c_null=args.c_null,
        ridge=args.ridge,
        endpoint_trim_threshold=args.endpoint_trim_threshold,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        keep_next_longer=args.keep_next_longer,
        peak_backtrack_enabled=args.peak_backtrack_enabled,
        peak_backtrack_rise_fraction=args.peak_backtrack_rise_fraction,
        local_simplifier_enabled=args.local_simplifier_enabled,
        local_simplifier_gain_sigma=args.local_simplifier_gain_sigma,
        local_simplifier_max_points=args.local_simplifier_max_points,
    )
    if bool(args.lag_registered) or int(args.max_lag_stations) > 0 or int(args.max_extend_stations) > 0:
        return HLLRDV2Config(
            **base_kwargs,
            lag_enabled=bool(args.lag_registered) or int(args.max_lag_stations) > 0 or int(args.max_extend_stations) > 0,
            max_lag_stations=args.max_lag_stations,
            lag_direction=args.lag_direction,
            lag_penalty=args.lag_penalty,
            max_extend_stations=args.max_extend_stations,
            extend_direction=args.extend_direction,
            extend_penalty=args.extend_penalty,
            registration_iterations=args.registration_iterations,
            registration_tolerance=args.registration_tolerance,
        )
    return HLLRDV1Config(**base_kwargs)


def _trace_extra_residual_config_from_args(args: argparse.Namespace) -> HLLRDV1Config | None:
    if not bool(args.trace_extra_events):
        return None
    return HLLRDV1Config(
        L_min=args.trace_extra_L_min,
        L_max=args.trace_extra_L_max,
        K_max=args.trace_extra_K_max,
        kappa_peak=0.0,
        n_min=5,
        c_null=0.0,
        epsilon_gain=0.0,
        activation_scale=0.0,
        peak_backtrack_enabled=False,
        local_simplifier_enabled=False,
    )


def _load_raw_tracks_for_event_evaluation(matrix: Any, args: argparse.Namespace) -> Any:
    raw_adsb_dir = args.raw_adsb_dir
    if raw_adsb_dir is None:
        metadata_dir = matrix.metadata.get("raw_adsb_dir")
        if not metadata_dir:
            raise SystemExit("matrix metadata does not include raw_adsb_dir; pass --raw-adsb-dir")
        raw_adsb_dir = Path(str(metadata_dir))
    tracks = load_raw_adsb(raw_adsb_dir, int(args.processes))
    tracks = split_tracks_by_gap(tracks, int(args.split_gap_seconds))
    flight_ids = set(matrix.flight_ids)
    tracks = tracks.loc[tracks["flight_id"].astype(str).isin(flight_ids)].copy()
    if tracks.empty:
        raise SystemExit(f"No raw ADS-B rows matched matrix flight IDs under {raw_adsb_dir}")

    trim_info = matrix.metadata.get("trim", {})
    if trim_info:
        tracks = trim_tracks_from_anchor(
            tracks,
            anchor_lat_deg=float(trim_info.get("refined_anchor_lat_deg", trim_info["coarse_anchor_lat_deg"])),
            anchor_lon_deg=float(trim_info.get("refined_anchor_lon_deg", trim_info["coarse_anchor_lon_deg"])),
            max_anchor_distance_nm=float(trim_info.get("max_anchor_distance_nm", 15.0)),
            min_points_after_anchor=int(matrix.metadata.get("config", {}).get("min_points_per_flight", 3)),
            refine_anchor=False,
        ).tracks
    return tracks


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["peak_index", "start", "end", "length", "score"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_matrix_summary(console: Console, artifact: Any, output: Path) -> None:
    table = Table(title="HLLRD matrix")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Cluster", str(artifact.cluster or ""))
    table.add_row("Flights", f"{len(artifact.flight_ids):,}")
    table.add_row("Stations", f"{artifact.X.shape[1]:,}")
    table.add_row("Skipped flights", f"{len(artifact.skipped_flights):,}")
    table.add_row("sigma_hat", f"{artifact.sigma_hat:.3f}")
    table.add_row("Output", output.as_posix())
    console.print(table)


def _print_fit_summary(console: Console, result: Any, output: Path) -> None:
    model_name = "HLLRD V2 fit" if isinstance(result.config, HLLRDV2Config) else "HLLRD V1 fit"
    table = Table(title=model_name)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Events", f"{len(result.events):,}")
    table.add_row("Explained fraction", f"{result.explained_fraction:.3%}")
    table.add_row("sigma_hat", f"{result.sigma_hat:.3f}")
    table.add_row("Output", output.as_posix())
    console.print(table)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
