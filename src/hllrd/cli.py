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

from hllrd.data import filter_tracks_to_cluster, load_cluster_flights, normalize_cluster
from hllrd.elastic_fpca import ElasticFPCAConfig, fit_elastic_event_fpca, save_elastic_fpca_result
from hllrd.fit import (
    HLLRDV1Config,
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

    elastic_parser = subparsers.add_parser(
        "elastic-fpca",
        description="Fit elastic vertical and horizontal fPCA per event.",
    )
    _add_elastic_fpca_args(elastic_parser)
    elastic_parser.set_defaults(func=run_elastic_fpca)

    run_parser = subparsers.add_parser("run", description="Run build-matrix, candidates, fit, and report.")
    _add_run_args(run_parser)
    run_parser.set_defaults(func=run_pipeline)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(_normalize_std_grid_argv(argv))
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


def elastic_fpca_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit HLLRD event-level elastic fPCA.")
    _add_elastic_fpca_args(parser)
    run_elastic_fpca(parser.parse_args(_normalize_std_grid_argv(argv)))


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
        explained_fraction=np.asarray(transform.explained_fraction, dtype=float),
        matrix=np.asarray(args.matrix.as_posix(), dtype=str),
        model=np.asarray(args.model.as_posix(), dtype=str),
    )
    if args.summary is not None:
        payload = {
            "explained_fraction": transform.explained_fraction,
            "average_active_events_per_flight": (
                float(np.mean(transform.active_counts)) if transform.active_counts.size else 0.0
            ),
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


def run_elastic_fpca(args: argparse.Namespace) -> None:
    matrix = load_matrix_artifact(args.matrix)
    model = load_fit_result(args.model)
    config = ElasticFPCAConfig(
        components=args.components,
        std_grid=args.std_grid,
        min_active_flights=args.min_active_flights,
        parallel=args.parallel,
        cores=args.cores,
    )
    result = fit_elastic_event_fpca(
        matrix,
        model,
        config=config,
        metadata={
            "matrix": args.matrix.as_posix(),
            "model": args.model.as_posix(),
            "cluster": matrix.cluster,
            "flight_count": len(matrix.flight_ids),
            "config": asdict(config),
        },
    )
    save_elastic_fpca_result(args.output, result)
    _print_elastic_summary(Console(), result, args.output)


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
    parser.add_argument(
        "--empirical-null-repeats",
        type=int,
        default=HLLRDV1Config().empirical_null_repeats,
        help="Number of quiet windows per length used to calibrate the empirical null; use 0 for analytic only.",
    )
    parser.add_argument(
        "--empirical-null-quantile",
        type=float,
        default=HLLRDV1Config().empirical_null_quantile,
        help="Quantile of quiet-window active gains used as the empirical null threshold.",
    )
    parser.add_argument("--empirical-null-seed", type=int, default=HLLRDV1Config().empirical_null_seed)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--endpoint-trim-threshold", type=float, default=0.05)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.8)
    parser.add_argument("--keep-next-longer", action="store_true")
    parser.add_argument("--no-peak-backtrack", dest="peak_backtrack_enabled", action="store_false")
    parser.add_argument("--peak-backtrack-rise-fraction", type=float, default=0.05)
    parser.add_argument(
        "--local-simplifier",
        dest="local_simplifier_enabled",
        action="store_true",
        default=HLLRDV1Config().local_simplifier_enabled,
    )
    parser.add_argument("--no-local-simplifier", dest="local_simplifier_enabled", action="store_false")
    parser.add_argument("--local-simplifier-gain-sigma", type=float, default=128.0)
    parser.add_argument("--local-simplifier-max-points", type=int, default=4)
    parser.add_argument(
        "--local-simplifier-max-relative-loss",
        type=float,
        default=HLLRDV1Config().local_simplifier_max_relative_loss,
    )


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


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)


def _add_elastic_fpca_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--std-grid", type=_parse_std_grid, default=ElasticFPCAConfig().std_grid)
    parser.add_argument("--min-active-flights", type=int, default=5)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--cores", type=int, default=1)


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
    return HLLRDV1Config(
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
        empirical_null_repeats=args.empirical_null_repeats,
        empirical_null_quantile=args.empirical_null_quantile,
        empirical_null_seed=args.empirical_null_seed,
        ridge=args.ridge,
        endpoint_trim_threshold=args.endpoint_trim_threshold,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        keep_next_longer=args.keep_next_longer,
        peak_backtrack_enabled=args.peak_backtrack_enabled,
        peak_backtrack_rise_fraction=args.peak_backtrack_rise_fraction,
        local_simplifier_enabled=args.local_simplifier_enabled,
        local_simplifier_gain_sigma=args.local_simplifier_gain_sigma,
        local_simplifier_max_points=args.local_simplifier_max_points,
        local_simplifier_max_relative_loss=args.local_simplifier_max_relative_loss,
    )


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["peak_index", "start", "end", "length", "score"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_std_grid(value: str) -> tuple[float, ...]:
    try:
        grid = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--std-grid must be a comma-separated list of numbers") from exc
    try:
        ElasticFPCAConfig(std_grid=grid).validate()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return grid


def _normalize_std_grid_argv(argv: list[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    normalized: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "--std-grid" and index + 1 < len(values):
            normalized.append(f"--std-grid={values[index + 1]}")
            index += 2
            continue
        normalized.append(value)
        index += 1
    return normalized


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
    table = Table(title="HLLRD V1 fit")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Events", f"{len(result.events):,}")
    table.add_row("Explained fraction", f"{result.explained_fraction:.3%}")
    table.add_row("sigma_hat", f"{result.sigma_hat:.3f}")
    table.add_row("Output", output.as_posix())
    console.print(table)


def _print_elastic_summary(console: Console, result: Any, output: Path) -> None:
    table = Table(title="HLLRD elastic fPCA")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Events", f"{len(result.events):,}")
    table.add_row("Components", f"{result.config.components:,}")
    table.add_row("Std grid", ",".join(f"{value:g}" for value in result.config.std_grid))
    table.add_row("Output", output.as_posix())
    console.print(table)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
