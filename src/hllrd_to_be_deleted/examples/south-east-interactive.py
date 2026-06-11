from __future__ import annotations

import argparse
import sys
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
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

from hllrd_to_be_deleted.fit import load_fit_result  # noqa: E402
from hllrd_to_be_deleted.data import trim_tracks_from_anchor  # noqa: E402
from hllrd_to_be_deleted.elastic_fpca import (  # noqa: E402
    _invert_warp,
    horizontal_component_gamma,
    load_elastic_fpca_result,
    vertical_component_delta,
)
from hllrd_to_be_deleted.geometry import LocalProjection, cumulative_distance_m  # noqa: E402
from hllrd_to_be_deleted.matrix import load_matrix_artifact  # noqa: E402
from scenario.trajectory_compressor.io import load_raw_adsb, split_tracks_by_gap  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "hllrd" / "south-east"
DEFAULT_MATRIX = DEFAULT_OUTPUT_DIR / "matrix_SE_from_merge.npz"
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "model_from_merge_L40_K6.npz"
DEFAULT_ELASTIC_FPCA = DEFAULT_OUTPUT_DIR / "elastic_fpca_from_merge_L40_K6.npz"
DEFAULT_SPLIT_GAP_SECONDS = 25 * 60
NM_PER_M = 1.0 / 1852.0


def interpolate_station_path(path_xy_m: np.ndarray, station_coordinate: np.ndarray) -> np.ndarray:
    stations = np.arange(path_xy_m.shape[0], dtype=float)
    clipped = np.clip(np.asarray(station_coordinate, dtype=float), stations[0], stations[-1])
    x = np.interp(clipped, stations, path_xy_m[:, 0])
    y = np.interp(clipped, stations, path_xy_m[:, 1])
    return np.column_stack((x, y))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive South-East HLLRD event response viewer.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="Path to matrix_SE_from_merge.npz.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to model_from_merge_L40_K6.npz.")
    parser.add_argument(
        "--elastic-fpca",
        type=Path,
        default=DEFAULT_ELASTIC_FPCA,
        help="Path to elastic FPCA event artifact.",
    )
    parser.add_argument(
        "--raw-adsb-dir",
        type=Path,
        default=None,
        help="Raw ADS-B directory for grey background tracks.",
    )
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Raw ADS-B loading worker count. Defaults to 1 for GUI launch reliability.",
    )
    parser.add_argument("--background-source", choices=["raw", "matrix"], default="raw")
    parser.add_argument(
        "--background-alpha",
        type=float,
        default=0.08,
        help="Transparency for all-flight background paths.",
    )
    parser.add_argument(
        "--background-linewidth",
        type=float,
        default=0.35,
        help="Line width for all-flight background paths.",
    )
    return parser


class SouthEastInteractive:
    def __init__(
        self,
        matrix_path: Path,
        model_path: Path,
        elastic_fpca_path: Path,
        *,
        raw_adsb_dir: Path | None,
        split_gap_seconds: int,
        processes: int,
        background_source: str,
        background_alpha: float,
        background_linewidth: float,
    ) -> None:
        self.matrix = load_matrix_artifact(matrix_path)
        self.result = load_fit_result(model_path)
        self.elastic = self._load_elastic_fpca(elastic_fpca_path, matrix_path, model_path)
        if not self.result.events:
            raise RuntimeError(f"Model contains no HLLRD events: {model_path}")
        if self.matrix.X.shape[1] != self.result.residual.shape[1]:
            raise RuntimeError("Matrix station count does not match model station count.")
        if not self.elastic.events:
            raise RuntimeError(f"Elastic FPCA artifact contains no drawable events: {elastic_fpca_path}")
        self._validate_elastic_consistency(elastic_fpca_path)

        self.background_alpha = float(background_alpha)
        self.background_linewidth = float(background_linewidth)
        self.background_source = str(background_source)
        self.projection = LocalProjection(self.matrix.origin_lat_deg, self.matrix.origin_lon_deg)
        self.mean_xy_m = self._mean_xy_m()
        self.station_distance_nm = cumulative_distance_m(self.mean_xy_m[:, 0], self.mean_xy_m[:, 1]) * NM_PER_M
        self.raw_background_tracks = (
            self._load_raw_background_tracks(raw_adsb_dir, split_gap_seconds, processes)
            if self.background_source == "raw"
            else None
        )
        self.score_by_event = {
            int(event.event_index): {
                "vertical": np.zeros(event.component_count, dtype=float),
                "horizontal": np.zeros(event.component_count, dtype=float),
            }
            for event in self.elastic.events
        }
        self.current_event_index = int(self.elastic.events[0].event_index)
        self.current_family = "vertical"
        self.current_component = 0
        self.show_warp_lines = True

        self.fig: Figure
        self.ax: Axes
        self.event_radio: RadioButtons
        self.family_radio: RadioButtons
        self.component_radio: RadioButtons
        self.warp_checkbox: CheckButtons
        self.reset_button: Button
        self.amplitude_slider: Slider
        self.response_line: Line2D
        self.response_segment_line: Line2D
        self.window_line: Line2D
        self.window_endpoints: PathCollection
        self.warp_line_collection: LineCollection
        self.title_text: Text

    def _mean_xy_m(self) -> np.ndarray:
        return self.matrix.reference_xy_m + self.result.column_center[:, None] * self.matrix.normals_xy

    def _load_elastic_fpca(self, elastic_fpca_path: Path, matrix_path: Path, model_path: Path):
        if not elastic_fpca_path.exists():
            command = (
                "hllrd elastic-fpca "
                f"--matrix {matrix_path.as_posix()} "
                f"--model {model_path.as_posix()} "
                f"--output {elastic_fpca_path.as_posix()}"
            )
            raise RuntimeError(f"Elastic FPCA artifact is missing: {elastic_fpca_path}\nGenerate it with:\n{command}")
        return load_elastic_fpca_result(elastic_fpca_path)

    def _validate_elastic_consistency(self, elastic_fpca_path: Path) -> None:
        flight_count, station_count = self.matrix.X.shape
        if self.result.residual.shape != (flight_count, station_count):
            raise RuntimeError("Matrix and model residual shapes do not match.")
        for elastic_event in self.elastic.events:
            event_index = int(elastic_event.event_index)
            if event_index < 0 or event_index >= len(self.result.events):
                raise RuntimeError(
                    f"Elastic FPCA artifact {elastic_fpca_path} references missing model event {event_index}."
                )
            model_event = self.result.events[event_index]
            if int(model_event.start) != int(elastic_event.start) or int(model_event.end) != int(elastic_event.end):
                raise RuntimeError(
                    f"Elastic FPCA event {event_index} interval "
                    f"{elastic_event.start}:{elastic_event.end} does not match model interval "
                    f"{model_event.start}:{model_event.end}."
                )
            active_rows = np.flatnonzero(model_event.active_mask).astype(int)
            if not np.array_equal(active_rows, np.asarray(elastic_event.active_rows, dtype=int)):
                raise RuntimeError(f"Elastic FPCA event {event_index} active rows do not match the model.")
            if elastic_event.end > station_count:
                raise RuntimeError(f"Elastic FPCA event {event_index} extends beyond the matrix station count.")

    def xy_to_latlon(self, xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.projection.unproject(xy_m[:, 0], xy_m[:, 1])

    def flight_xy_m(self, flight_row: int) -> np.ndarray:
        return self.matrix.reference_xy_m + self.matrix.X[flight_row, :, None] * self.matrix.normals_xy

    def _load_raw_background_tracks(
        self,
        raw_adsb_dir: Path | None,
        split_gap_seconds: int,
        processes: int,
    ) -> dict[str, np.ndarray]:
        source_dir = raw_adsb_dir or self._raw_adsb_dir_from_matrix()
        try:
            tracks = load_raw_adsb(source_dir, int(processes))
        except FileNotFoundError:
            if int(processes) <= 1:
                raise
            tracks = load_raw_adsb(source_dir, 1)
        tracks = split_tracks_by_gap(tracks, int(split_gap_seconds))
        flight_ids = set(self.matrix.flight_ids)
        tracks = tracks.loc[tracks["flight_id"].astype(str).isin(flight_ids)].copy()
        if tracks.empty:
            raise RuntimeError(f"No raw ADS-B rows matched matrix flight IDs under {source_dir}")

        trim_info = self.matrix.metadata.get("trim", {})
        if trim_info:
            tracks = trim_tracks_from_anchor(
                tracks,
                anchor_lat_deg=float(trim_info.get("refined_anchor_lat_deg", trim_info["coarse_anchor_lat_deg"])),
                anchor_lon_deg=float(trim_info.get("refined_anchor_lon_deg", trim_info["coarse_anchor_lon_deg"])),
                max_anchor_distance_nm=float(trim_info.get("max_anchor_distance_nm", 15.0)),
                min_points_after_anchor=int(self.matrix.metadata.get("config", {}).get("min_points_per_flight", 3)),
                refine_anchor=False,
            ).tracks

        background: dict[str, np.ndarray] = {}
        for flight_id, flight in tracks.groupby("flight_id", sort=False):
            ordered = flight.sort_values("time", kind="stable")
            lat_lon = ordered.loc[:, ["lat", "lon"]].to_numpy(dtype=float)
            if lat_lon.shape[0] >= 2:
                background[str(flight_id)] = lat_lon
        if not background:
            raise RuntimeError("Raw ADS-B background contains no drawable tracks")
        return background

    def _raw_adsb_dir_from_matrix(self) -> Path:
        raw_adsb_dir = self.matrix.metadata.get("raw_adsb_dir")
        if not raw_adsb_dir:
            raise RuntimeError("matrix metadata does not include raw_adsb_dir; pass --raw-adsb-dir")
        return Path(str(raw_adsb_dir))

    def current_elastic_event(self):
        return self.elastic.event_by_index(self.current_event_index)

    def event_response_xy_m(self) -> np.ndarray:
        event = self.current_elastic_event()
        response = self.aggregate_amplitude_response_xy_m(event)
        segment = slice(event.start, event.end)
        response_segment = response[segment]
        gamma = self.aggregate_horizontal_gamma(event)
        inverse_gamma = _invert_warp(gamma, event.time)
        response[segment] = np.column_stack(
            (
                np.interp(inverse_gamma, event.time, response_segment[:, 0]),
                np.interp(inverse_gamma, event.time, response_segment[:, 1]),
            )
        )
        return response

    def event_mean_response_xy_m(self, event) -> np.ndarray:
        response = self.mean_xy_m.copy()
        segment = slice(event.start, event.end)
        response[segment] = self.mean_xy_m[segment] + event.fmean[:, None] * self.matrix.normals_xy[segment]
        return response

    def aggregate_amplitude_response_xy_m(self, event) -> np.ndarray:
        normal_offset_m = np.zeros(self.mean_xy_m.shape[0], dtype=float)
        normal_offset_m[event.start : event.end] = event.fmean + self.aggregate_vertical_delta(event)
        return self.mean_xy_m + normal_offset_m[:, None] * self.matrix.normals_xy

    def aggregate_vertical_delta(self, event) -> np.ndarray:
        delta = np.zeros(event.length, dtype=float)
        scores = self.event_scores(event)["vertical"]
        for component, score in enumerate(scores[: event.component_count]):
            if not np.isclose(float(score), 0.0):
                delta += vertical_component_delta(event, component, float(score), self.elastic.config.std_grid)
        return delta

    def aggregate_horizontal_gamma(self, event) -> np.ndarray:
        time = np.asarray(event.time, dtype=float)
        gamma = time.copy()
        scores = self.event_scores(event)["horizontal"]
        for component, score in enumerate(scores[: event.component_count]):
            if not np.isclose(float(score), 0.0):
                component_gamma = horizontal_component_gamma(event, component, float(score), self.elastic.config.std_grid)
                gamma += component_gamma - time
        if gamma.size:
            gamma = np.maximum.accumulate(np.clip(gamma, float(time[0]), float(time[-1])))
            gamma[0] = float(time[0])
            gamma[-1] = float(time[-1])
        return gamma

    def event_scores(self, event) -> dict[str, np.ndarray]:
        event_index = int(event.event_index)
        scores = self.score_by_event.get(event_index)
        if scores is None or scores["vertical"].shape != (event.component_count,):
            scores = {
                "vertical": np.zeros(event.component_count, dtype=float),
                "horizontal": np.zeros(event.component_count, dtype=float),
            }
            self.score_by_event[event_index] = scores
        return scores

    def current_score(self) -> float:
        event = self.current_elastic_event()
        return float(self.event_scores(event)[self.current_family][self.current_component])

    def create(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(10.5, 7.2))
        self.fig.subplots_adjust(left=0.25, bottom=0.2, right=0.96, top=0.9)
        self.ax.set_facecolor("#f7f7f7")

        self.plot_static_layers()

        event_ax = self.fig.add_axes((0.025, 0.44, 0.16, 0.42))
        event_labels = [str(event.event_index) for event in self.elastic.events]
        self.event_radio = RadioButtons(event_ax, event_labels, active=0)
        event_ax.set_title("Event ID", fontsize=10)

        family_ax = self.fig.add_axes((0.025, 0.27, 0.16, 0.11))
        self.family_radio = RadioButtons(family_ax, ["vertical", "horizontal"], active=0)
        family_ax.set_title("Family", fontsize=10)

        component_ax = self.fig.add_axes((0.025, 0.08, 0.16, 0.13))
        component_labels = [f"PC{index + 1}" for index in range(self.elastic.max_components)]
        self.component_radio = RadioButtons(component_ax, component_labels, active=0)
        component_ax.set_title("Component", fontsize=10)

        warp_ax = self.fig.add_axes((0.025, 0.215, 0.16, 0.04))
        self.warp_checkbox = CheckButtons(warp_ax, ["warp lines"], [self.show_warp_lines])

        reset_ax = self.fig.add_axes((0.025, 0.025, 0.16, 0.035))
        self.reset_button = Button(reset_ax, "Reset")

        std_grid = np.asarray(self.elastic.config.std_grid, dtype=float)
        amplitude_ax = self.fig.add_axes((0.25, 0.08, 0.65, 0.04))
        self.amplitude_slider = Slider(
            amplitude_ax,
            "Score (std)",
            float(std_grid[0]),
            float(std_grid[-1]),
            valinit=0.0,
            valstep=0.05,
        )

        self.response_line = self.ax.plot(
            [],
            [],
            color="#b2182b",
            linewidth=2.0,
            alpha=0.9,
            label="FPCA response trajectory",
            zorder=5,
        )[0]
        self.response_segment_line = self.ax.plot(
            [],
            [],
            color="#b2182b",
            linewidth=4.0,
            alpha=0.96,
            solid_capstyle="round",
            label="FPCA response event segment",
            zorder=6,
        )[0]
        self.window_line = self.ax.plot(
            [],
            [],
            color="black",
            linewidth=5.0,
            alpha=0.9,
            solid_capstyle="round",
            label="HLLRD center window",
            zorder=4,
        )[0]
        self.window_endpoints = self.ax.scatter([], [], s=24, color="black", edgecolor="white", linewidth=0.5, zorder=7)
        self.warp_line_collection = LineCollection(
            [],
            colors="#ffcc00",
            linewidths=0.8,
            alpha=0.78,
            label="warping station matches",
            zorder=5.5,
        )
        self.ax.add_collection(self.warp_line_collection)
        self.title_text = self.ax.set_title("")

        self.event_radio.on_clicked(self.on_event_selected)
        self.family_radio.on_clicked(self.on_family_selected)
        self.component_radio.on_clicked(self.on_component_selected)
        self.warp_checkbox.on_clicked(self.on_warp_lines_toggled)
        self.reset_button.on_clicked(self.on_reset_clicked)
        self.amplitude_slider.on_changed(self.on_score_changed)
        self.update_event(self.current_event_index)
        self.ax.legend(loc="best", fontsize=8)

    def plot_static_layers(self) -> None:
        mean_lat, mean_lon = self.xy_to_latlon(self.mean_xy_m)
        self.ax.plot(
            mean_lon,
            mean_lat,
            color="#303030",
            linewidth=2.0,
            alpha=0.82,
            label="model center trajectory",
            zorder=3,
        )
        if self.raw_background_tracks is not None:
            for lat_lon in self.raw_background_tracks.values():
                self.ax.plot(
                    lat_lon[:, 1],
                    lat_lon[:, 0],
                    color="#555555",
                    linewidth=self.background_linewidth,
                    alpha=self.background_alpha,
                    zorder=1,
                )
        else:
            for row_index in range(len(self.matrix.flight_ids)):
                xy_m = self.flight_xy_m(row_index)
                lat, lon = self.xy_to_latlon(xy_m)
                self.ax.plot(
                    lon,
                    lat,
                    color="#555555",
                    linewidth=self.background_linewidth,
                    alpha=self.background_alpha,
                    zorder=1,
                )

        trim_info = self.matrix.metadata.get("trim", {})
        refined_lat = trim_info.get("refined_anchor_lat_deg")
        refined_lon = trim_info.get("refined_anchor_lon_deg")
        if refined_lat is not None and refined_lon is not None:
            self.ax.scatter(
                [float(refined_lon)],
                [float(refined_lat)],
                color="#1b9e77",
                s=38,
                label="merge anchor",
                zorder=8,
            )

        lon_pad = max(0.01, 0.04 * float(np.ptp(mean_lon)))
        lat_pad = max(0.01, 0.04 * float(np.ptp(mean_lat)))
        self.ax.set_xlim(float(np.min(mean_lon) - lon_pad), float(np.max(mean_lon) + lon_pad))
        self.ax.set_ylim(float(np.min(mean_lat) - lat_pad), float(np.max(mean_lat) + lat_pad))
        self.ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.mean(mean_lat)))))
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("longitude")
        self.ax.set_ylabel("latitude")

    def on_event_selected(self, label: str | None) -> None:
        if label is None:
            return
        self.update_event(int(label))

    def on_family_selected(self, label: str | None) -> None:
        if label is None:
            return
        self.current_family = str(label)
        self.sync_slider_to_current_score()
        self.update_response()

    def on_component_selected(self, label: str | None) -> None:
        if label is None:
            return
        component = int(str(label).removeprefix("PC")) - 1
        if component >= self.current_elastic_event().component_count:
            self.current_component = 0
            self.component_radio.set_active(0)
            return
        self.current_component = component
        self.sync_slider_to_current_score()
        self.update_response()

    def on_warp_lines_toggled(self, _label: str | None) -> None:
        self.show_warp_lines = bool(self.warp_checkbox.get_status()[0])
        self.update_response()

    def on_score_changed(self, value: float) -> None:
        event = self.current_elastic_event()
        self.event_scores(event)[self.current_family][self.current_component] = float(value)
        self.update_response()

    def on_reset_clicked(self, _event) -> None:
        scores = self.event_scores(self.current_elastic_event())
        scores["vertical"][:] = 0.0
        scores["horizontal"][:] = 0.0
        self.sync_slider_to_current_score()
        self.update_response()

    def sync_slider_to_current_score(self) -> None:
        self.amplitude_slider.eventson = False
        self.amplitude_slider.set_val(self.current_score())
        self.amplitude_slider.eventson = True

    def update_event(self, event_index: int) -> None:
        self.current_event_index = int(event_index)
        event = self.current_elastic_event()
        if self.current_component >= event.component_count:
            self.current_component = 0
            self.component_radio.eventson = False
            self.component_radio.set_active(0)
            self.component_radio.eventson = True
        self.sync_slider_to_current_score()
        self.update_response()

    def update_response(self) -> None:
        event = self.current_elastic_event()
        xy_m = self.event_response_xy_m()
        lat, lon = self.xy_to_latlon(xy_m)
        self.response_line.set_data(lon, lat)

        segment = xy_m[event.start : event.end]
        segment_lat, segment_lon = self.xy_to_latlon(segment)
        self.response_segment_line.set_data(segment_lon, segment_lat)
        self.update_warp_lines(event, xy_m)

        mean_segment = self.mean_xy_m[event.start : event.end]
        mean_segment_lat, mean_segment_lon = self.xy_to_latlon(mean_segment)
        self.window_line.set_data(mean_segment_lon, mean_segment_lat)
        self.window_endpoints.set_offsets(
            np.column_stack(
                (
                    [mean_segment_lon[0], mean_segment_lon[-1]],
                    [mean_segment_lat[0], mean_segment_lat[-1]],
                )
            )
        )

        active_count = int(event.active_count)
        active_fraction = 100.0 * float(active_count) / float(len(self.matrix.flight_ids))
        start_nm = self.station_distance_nm[event.start]
        end_nm = self.station_distance_nm[event.end - 1]
        self.title_text.set_text(
            f"Event {self.current_event_index}: {start_nm:.1f}-{end_nm:.1f} NM from merge | "
            f"active {active_count}/{len(self.matrix.flight_ids)} ({active_fraction:.1f}%) | "
            f"{self.current_family} PC{self.current_component + 1} score={self.amplitude_slider.val:+.2f} std | "
            "response around FPCA mean"
        )
        self.fig.canvas.draw_idle()

    def update_warp_lines(self, event, response_xy_m: np.ndarray) -> None:
        if not self.show_warp_lines:
            self.warp_line_collection.set_segments([])
            return

        length = int(event.length)
        if length < 2:
            self.warp_line_collection.set_segments([])
            return

        gamma = self.aggregate_horizontal_gamma(event)
        mapped_local_response_station = np.asarray(gamma, dtype=float) * float(length - 1)
        mapped_global_response_station = mapped_local_response_station + int(event.start)

        template_segment = self.mean_xy_m[event.start : event.end]
        mapped_response_segment = interpolate_station_path(response_xy_m, mapped_global_response_station)
        template_lat, template_lon = self.xy_to_latlon(template_segment)
        mapped_lat, mapped_lon = self.xy_to_latlon(mapped_response_segment)
        segments = np.stack(
            (
                np.column_stack((template_lon, template_lat)),
                np.column_stack((mapped_lon, mapped_lat)),
            ),
            axis=1,
        )
        self.warp_line_collection.set_segments(segments)

    def show(self) -> None:
        self.create()
        plt.show()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app = SouthEastInteractive(
        args.matrix,
        args.model,
        args.elastic_fpca,
        raw_adsb_dir=args.raw_adsb_dir,
        split_gap_seconds=args.split_gap_seconds,
        processes=args.processes,
        background_source=args.background_source,
        background_alpha=args.background_alpha,
        background_linewidth=args.background_linewidth,
    )
    app.show()


if __name__ == "__main__":
    main()
