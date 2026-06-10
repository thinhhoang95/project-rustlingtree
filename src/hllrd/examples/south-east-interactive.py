from __future__ import annotations

import argparse
import sys
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
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

from hllrd.fit import load_fit_result  # noqa: E402
from hllrd.data import trim_tracks_from_anchor  # noqa: E402
from hllrd.elastic_fpca import elastic_component_delta, load_elastic_fpca_result  # noqa: E402
from hllrd.geometry import LocalProjection, cumulative_distance_m  # noqa: E402
from hllrd.matrix import load_matrix_artifact  # noqa: E402
from scenario.trajectory_compressor.io import load_raw_adsb, split_tracks_by_gap  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "hllrd" / "south-east"
DEFAULT_MATRIX = DEFAULT_OUTPUT_DIR / "matrix_SE_from_merge.npz"
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "model_from_merge_L40_K6.npz"
DEFAULT_ELASTIC_FPCA = DEFAULT_OUTPUT_DIR / "elastic_fpca_from_merge_L40_K6.npz"
DEFAULT_SPLIT_GAP_SECONDS = 25 * 60
NM_PER_M = 1.0 / 1852.0


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
        self.current_event_index = int(self.elastic.events[0].event_index)
        self.current_family = "vertical"
        self.current_component = 0

        self.fig: Figure
        self.ax: Axes
        self.event_radio: RadioButtons
        self.family_radio: RadioButtons
        self.component_radio: RadioButtons
        self.amplitude_slider: Slider
        self.response_line: Line2D
        self.response_segment_line: Line2D
        self.window_line: Line2D
        self.window_endpoints: PathCollection
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
        if self.current_family == "horizontal":
            if self._horizontal_uses_observed_range(event, self.current_component):
                return self.observed_horizontal_response_xy_m(event)
            return self.elastic_normal_offset_response_xy_m(event, "horizontal")
        return self.elastic_normal_offset_response_xy_m(event, "vertical")

    def elastic_normal_offset_response_xy_m(self, event, family: str) -> np.ndarray:
        normal_offset_m = np.zeros(self.mean_xy_m.shape[0], dtype=float)
        delta = elastic_component_delta(
            event,
            family,
            self.current_component,
            float(self.amplitude_slider.val),
            self.elastic.config.std_grid,
        )
        normal_offset_m[event.start : event.end] = delta
        return self.mean_xy_m + normal_offset_m[:, None] * self.matrix.normals_xy

    def _horizontal_uses_observed_range(self, event, component: int) -> bool:
        component_index = int(component)
        if component_index < 0 or component_index >= event.horizontal_coefficients.shape[1]:
            return False
        scores = np.asarray(event.horizontal_coefficients[:, component_index], dtype=float)
        if scores.size < 3 or float(np.std(scores)) <= 1.0e-12:
            return False
        trajectories = np.stack(
            [self.flight_xy_m(int(row)) for row in np.asarray(event.active_rows, dtype=int)],
            axis=0,
        )
        extent = self._event_trombone_extent_m(event, trajectories)
        if float(np.ptp(extent)) <= 1.0e-12:
            return False
        correlation = float(np.corrcoef(scores, extent)[0, 1])
        return bool(np.isfinite(correlation) and abs(correlation) >= 0.5)

    def observed_horizontal_response_xy_m(self, event) -> np.ndarray:
        component = int(self.current_component)
        if component < 0 or component >= event.horizontal_coefficients.shape[1]:
            return self.mean_xy_m.copy()
        scores = np.asarray(event.horizontal_coefficients[:, component], dtype=float)
        if scores.size == 0:
            return self.mean_xy_m.copy()
        score_scale = float(np.std(scores))
        if not np.isfinite(score_scale) or score_scale <= 1.0e-12:
            return self.mean_xy_m.copy()

        score_z = (scores - float(np.mean(scores))) / score_scale
        low_z = float(np.min(score_z))
        high_z = float(np.max(score_z))
        if low_z >= 0.0 and high_z <= 0.0:
            return self.mean_xy_m.copy()

        target_z = self._horizontal_target_score_z(score_z, float(self.amplitude_slider.val))
        if np.isclose(target_z, 0.0):
            return self.mean_xy_m.copy()
        if target_z < 0.0 and low_z < 0.0:
            endpoint_xy_m, endpoint_extent_m = self._score_endpoint_observation(event, score_z, upper=False)
            fraction = float(np.clip(target_z / low_z, 0.0, 1.0))
        elif target_z > 0.0 and high_z > 0.0:
            endpoint_xy_m, endpoint_extent_m = self._score_endpoint_observation(event, score_z, upper=True)
            fraction = float(np.clip(target_z / high_z, 0.0, 1.0))
        else:
            return self.mean_xy_m.copy()
        return self._trombone_extent_response_xy_m(event, endpoint_xy_m, endpoint_extent_m, fraction)

    def _horizontal_target_score_z(self, score_z: np.ndarray, amplitude_std: float) -> float:
        grid = np.asarray(self.elastic.config.std_grid, dtype=float)
        if grid.size < 2:
            return 0.0
        low_ui = float(grid[0])
        high_ui = float(grid[-1])
        amplitude = float(np.clip(amplitude_std, low_ui, high_ui))
        if amplitude < 0.0 and low_ui < 0.0:
            return float(np.min(score_z)) * (amplitude / low_ui)
        if amplitude > 0.0 and high_ui > 0.0:
            return float(np.max(score_z)) * (amplitude / high_ui)
        return 0.0

    def _score_endpoint_observation(
        self,
        event,
        score_z: np.ndarray,
        *,
        upper: bool,
    ) -> tuple[np.ndarray, float]:
        count = int(score_z.size)
        if count == 0:
            return self.mean_xy_m.copy(), self._mean_trombone_extent_m(event)
        side_mask = score_z > 0.0 if upper else score_z < 0.0
        if not np.any(side_mask):
            side_mask = score_z == np.max(score_z) if upper else score_z == np.min(score_z)
        candidate_indices = np.flatnonzero(side_mask)
        if candidate_indices.size == 0:
            return self.mean_xy_m.copy(), self._mean_trombone_extent_m(event)

        trajectories = np.stack(
            [self.flight_xy_m(int(row)) for row in np.asarray(event.active_rows, dtype=int)],
            axis=0,
        )
        extent = self._event_trombone_extent_m(event, trajectories)
        score_extent_correlation = np.corrcoef(score_z, extent)[0, 1] if count > 1 else 0.0
        if not np.isfinite(score_extent_correlation):
            score_extent_correlation = 0.0
        choose_long = upper if score_extent_correlation >= 0.0 else not upper
        candidate_extent = extent[candidate_indices]
        selected_position = int(np.argmax(candidate_extent) if choose_long else np.argmin(candidate_extent))
        selected_local = int(candidate_indices[selected_position])
        return trajectories[selected_local].copy(), float(extent[selected_local])

    def _trombone_extent_response_xy_m(
        self,
        event,
        endpoint_xy_m: np.ndarray,
        endpoint_extent_m: float,
        fraction: float,
    ) -> np.ndarray:
        axis = self._event_trombone_axis(event)
        if axis is None:
            return self.mean_xy_m + fraction * (endpoint_xy_m - self.mean_xy_m)
        segment, _midpoint, outward, mean_extent_m, weights = axis
        if float(np.max(weights)) <= 1.0e-12:
            return self.mean_xy_m + fraction * (endpoint_xy_m - self.mean_xy_m)
        target_extent_m = mean_extent_m + fraction * (float(endpoint_extent_m) - mean_extent_m)
        response = self.mean_xy_m.copy()
        response[segment] = response[segment] + ((target_extent_m - mean_extent_m) * weights)[:, None] * outward
        return response

    def _mean_trombone_extent_m(self, event) -> float:
        axis = self._event_trombone_axis(event)
        return 0.0 if axis is None else float(axis[3])

    def _event_trombone_extent_m(self, event, trajectories_xy_m: np.ndarray) -> np.ndarray:
        trajectories = np.asarray(trajectories_xy_m, dtype=float)
        if trajectories.ndim != 3 or trajectories.shape[1:] != self.mean_xy_m.shape:
            raise ValueError("trajectories_xy_m must have shape flight_count x station_count x 2")
        axis = self._event_trombone_axis(event)
        if axis is None:
            segment = slice(event.start, event.end)
            return np.linalg.norm(trajectories[:, segment, :] - self.mean_xy_m[segment][None, :, :], axis=2).max(axis=1)
        segment, midpoint, outward, _mean_extent_m, _weights = axis
        return np.max((trajectories[:, segment, :] - midpoint) @ outward, axis=1)

    def _event_trombone_axis(self, event) -> tuple[slice, np.ndarray, np.ndarray, float, np.ndarray] | None:
        segment = slice(event.start, event.end)
        start_xy = self.mean_xy_m[event.start]
        end_xy = self.mean_xy_m[event.end - 1]
        chord = end_xy - start_xy
        chord_norm = float(np.linalg.norm(chord))
        if chord_norm <= 1.0e-12:
            return None
        chord_unit = chord / chord_norm
        outward = np.asarray([-chord_unit[1], chord_unit[0]], dtype=float)
        midpoint = 0.5 * (start_xy + end_xy)
        mean_segment = self.mean_xy_m[segment] - midpoint
        if float(np.max(mean_segment @ outward)) < float(np.max(mean_segment @ (-outward))):
            outward = -outward
        mean_coordinates = mean_segment @ outward
        mean_extent_m = float(np.max(mean_coordinates))
        if mean_extent_m <= 1.0e-12:
            weights = np.zeros_like(mean_coordinates)
        else:
            weights = np.clip(mean_coordinates / mean_extent_m, 0.0, 1.0)
        return segment, midpoint, outward, mean_extent_m, weights

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

        std_grid = np.asarray(self.elastic.config.std_grid, dtype=float)
        amplitude_ax = self.fig.add_axes((0.25, 0.08, 0.65, 0.04))
        self.amplitude_slider = Slider(
            amplitude_ax,
            "Scale",
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
            label="deformed trajectory",
            zorder=5,
        )[0]
        self.response_segment_line = self.ax.plot(
            [],
            [],
            color="#b2182b",
            linewidth=4.0,
            alpha=0.96,
            solid_capstyle="round",
            label="active event segment",
            zorder=6,
        )[0]
        self.window_line = self.ax.plot(
            [],
            [],
            color="black",
            linewidth=5.0,
            alpha=0.9,
            solid_capstyle="round",
            label="window on center",
            zorder=4,
        )[0]
        self.window_endpoints = self.ax.scatter([], [], s=24, color="black", edgecolor="white", linewidth=0.5, zorder=7)
        self.title_text = self.ax.set_title("")

        self.event_radio.on_clicked(self.on_event_selected)
        self.family_radio.on_clicked(self.on_family_selected)
        self.component_radio.on_clicked(self.on_component_selected)
        self.amplitude_slider.on_changed(lambda _value: self.update_response())
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
        self.update_response()

    def update_event(self, event_index: int) -> None:
        self.current_event_index = int(event_index)
        if self.current_component >= self.current_elastic_event().component_count:
            self.current_component = 0
            self.component_radio.set_active(0)
            return
        self.amplitude_slider.eventson = False
        self.amplitude_slider.set_val(0.0)
        self.amplitude_slider.eventson = True
        self.update_response()

    def update_response(self) -> None:
        event = self.current_elastic_event()
        xy_m = self.event_response_xy_m()
        lat, lon = self.xy_to_latlon(xy_m)
        self.response_line.set_data(lon, lat)

        segment = xy_m[event.start : event.end]
        segment_lat, segment_lon = self.xy_to_latlon(segment)
        self.response_segment_line.set_data(segment_lon, segment_lat)

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
        observed_range = (
            self.current_family == "horizontal"
            and self._horizontal_uses_observed_range(event, self.current_component)
        )
        scale_label = "observed range" if observed_range else "std"
        self.title_text.set_text(
            f"Event {self.current_event_index}: {start_nm:.1f}-{end_nm:.1f} NM from merge | "
            f"active {active_count}/{len(self.matrix.flight_ids)} ({active_fraction:.1f}%) | "
            f"{self.current_family} PC{self.current_component + 1}={self.amplitude_slider.val:.2f} {scale_label}"
        )
        self.fig.canvas.draw_idle()

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
