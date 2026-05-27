from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from hllrd.geometry import LocalProjection, cumulative_distance_m  # noqa: E402
from hllrd.matrix import load_matrix_artifact  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "hllrd" / "south-east"
DEFAULT_MATRIX = DEFAULT_OUTPUT_DIR / "matrix_SE_from_merge.npz"
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "model_from_merge_L40_K6.npz"
NM_PER_M = 1.0 / 1852.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive South-East HLLRD event response viewer.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="Path to matrix_SE_from_merge.npz.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to model_from_merge_L40_K6.npz.")
    parser.add_argument("--background-alpha", type=float, default=0.08, help="Transparency for all-flight background paths.")
    parser.add_argument("--background-linewidth", type=float, default=0.35, help="Line width for all-flight background paths.")
    return parser


class SouthEastInteractive:
    def __init__(self, matrix_path: Path, model_path: Path, *, background_alpha: float, background_linewidth: float) -> None:
        self.matrix = load_matrix_artifact(matrix_path)
        self.result = load_fit_result(model_path)
        if not self.result.events:
            raise RuntimeError(f"Model contains no HLLRD events: {model_path}")
        if self.matrix.X.shape[1] != self.result.residual.shape[1]:
            raise RuntimeError("Matrix station count does not match model station count.")

        self.background_alpha = float(background_alpha)
        self.background_linewidth = float(background_linewidth)
        self.projection = LocalProjection(self.matrix.origin_lat_deg, self.matrix.origin_lon_deg)
        self.mean_xy_m = self._mean_xy_m()
        self.station_distance_nm = cumulative_distance_m(self.mean_xy_m[:, 0], self.mean_xy_m[:, 1]) * NM_PER_M
        self.current_event_index = 0

        self.fig: plt.Figure
        self.ax: plt.Axes
        self.event_radio: RadioButtons
        self.z1_slider: Slider
        self.z2_slider: Slider
        self.response_line = None
        self.response_segment_line = None
        self.window_line = None
        self.window_endpoints = None
        self.title_text = None

    def _mean_xy_m(self) -> np.ndarray:
        mean_normal_m = np.mean(self.matrix.X, axis=0)
        return self.matrix.reference_xy_m + mean_normal_m[:, None] * self.matrix.normals_xy

    def xy_to_latlon(self, xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.projection.unproject(xy_m[:, 0], xy_m[:, 1])

    def flight_xy_m(self, flight_row: int) -> np.ndarray:
        return self.matrix.reference_xy_m + self.matrix.X[flight_row, :, None] * self.matrix.normals_xy

    def event_response_xy_m(self, z: np.ndarray) -> np.ndarray:
        event = self.result.events[self.current_event_index]
        normal_offset_m = event.basis @ z
        return self.mean_xy_m + normal_offset_m[:, None] * self.matrix.normals_xy

    def active_coefficients(self, event_index: int) -> np.ndarray:
        event = self.result.events[event_index]
        active = np.flatnonzero(event.active_mask)
        coefficients = event.coefficients[active] if active.size else event.coefficients
        if coefficients.size == 0:
            return np.zeros((1, 2), dtype=float)
        return coefficients

    def slider_limits(self, event_index: int, component: int) -> tuple[float, float]:
        values = self.active_coefficients(event_index)[:, component]
        lower = float(min(np.min(values), 0.0))
        upper = float(max(np.max(values), 0.0))
        if np.isclose(lower, upper):
            pad = max(abs(lower) * 0.1, 1.0)
            lower -= pad
            upper += pad
        return lower, upper

    def create(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(10.5, 7.2))
        self.fig.subplots_adjust(left=0.25, bottom=0.22, right=0.96, top=0.9)
        self.ax.set_facecolor("#f7f7f7")

        self.plot_static_layers()

        z1_min, z1_max = self.slider_limits(0, 0)
        z2_min, z2_max = self.slider_limits(0, 1)

        event_ax = self.fig.add_axes((0.025, 0.38, 0.16, 0.48))
        event_labels = [str(index) for index in range(len(self.result.events))]
        self.event_radio = RadioButtons(event_ax, event_labels, active=0)
        event_ax.set_title("Event ID", fontsize=10)

        z1_ax = self.fig.add_axes((0.25, 0.11, 0.65, 0.035))
        z2_ax = self.fig.add_axes((0.25, 0.055, 0.65, 0.035))
        self.z1_slider = Slider(z1_ax, "Z1", z1_min, z1_max, valinit=0.0)
        self.z2_slider = Slider(z2_ax, "Z2", z2_min, z2_max, valinit=0.0)

        self.response_line = self.ax.plot([], [], color="#b2182b", linewidth=2.0, alpha=0.9, label="deformed trajectory", zorder=5)[0]
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
        self.window_line = self.ax.plot([], [], color="black", linewidth=5.0, alpha=0.9, solid_capstyle="round", label="window on mean", zorder=4)[0]
        self.window_endpoints = self.ax.scatter([], [], s=24, color="black", edgecolor="white", linewidth=0.5, zorder=7)
        self.title_text = self.ax.set_title("")

        self.event_radio.on_clicked(self.on_event_selected)
        self.z1_slider.on_changed(lambda _value: self.update_response())
        self.z2_slider.on_changed(lambda _value: self.update_response())
        self.update_event(0)
        self.ax.legend(loc="best", fontsize=8)

    def plot_static_layers(self) -> None:
        mean_lat, mean_lon = self.xy_to_latlon(self.mean_xy_m)
        self.ax.plot(mean_lon, mean_lat, color="#303030", linewidth=2.0, alpha=0.82, label="mean trajectory", zorder=3)
        for row_index in range(len(self.matrix.flight_ids)):
            xy_m = self.flight_xy_m(row_index)
            lat, lon = self.xy_to_latlon(xy_m)
            self.ax.plot(lon, lat, color="#555555", linewidth=self.background_linewidth, alpha=self.background_alpha, zorder=1)

        trim_info = self.matrix.metadata.get("trim", {})
        refined_lat = trim_info.get("refined_anchor_lat_deg")
        refined_lon = trim_info.get("refined_anchor_lon_deg")
        if refined_lat is not None and refined_lon is not None:
            self.ax.scatter([float(refined_lon)], [float(refined_lat)], color="#1b9e77", s=38, label="merge anchor", zorder=8)

        lon_pad = max(0.01, 0.04 * float(np.ptp(mean_lon)))
        lat_pad = max(0.01, 0.04 * float(np.ptp(mean_lat)))
        self.ax.set_xlim(float(np.min(mean_lon) - lon_pad), float(np.max(mean_lon) + lon_pad))
        self.ax.set_ylim(float(np.min(mean_lat) - lat_pad), float(np.max(mean_lat) + lat_pad))
        self.ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.mean(mean_lat)))))
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("longitude")
        self.ax.set_ylabel("latitude")

    def on_event_selected(self, label: str) -> None:
        self.update_event(int(label))

    def update_event(self, event_index: int) -> None:
        self.current_event_index = int(event_index)
        for slider, component in ((self.z1_slider, 0), (self.z2_slider, 1)):
            lower, upper = self.slider_limits(self.current_event_index, component)
            slider.eventson = False
            slider.valmin = lower
            slider.valmax = upper
            slider.ax.set_xlim(lower, upper)
            slider.set_val(0.0)
            slider.eventson = True
        self.update_response()

    def update_response(self) -> None:
        event = self.result.events[self.current_event_index]
        z = np.asarray([self.z1_slider.val, self.z2_slider.val], dtype=float)
        xy_m = self.event_response_xy_m(z)
        lat, lon = self.xy_to_latlon(xy_m)
        self.response_line.set_data(lon, lat)

        segment = xy_m[event.start : event.end]
        segment_lat, segment_lon = self.xy_to_latlon(segment)
        self.response_segment_line.set_data(segment_lon, segment_lat)

        mean_segment = self.mean_xy_m[event.start : event.end]
        mean_segment_lat, mean_segment_lon = self.xy_to_latlon(mean_segment)
        self.window_line.set_data(mean_segment_lon, mean_segment_lat)
        self.window_endpoints.set_offsets(np.column_stack(([mean_segment_lon[0], mean_segment_lon[-1]], [mean_segment_lat[0], mean_segment_lat[-1]])))

        active_count = int(event.active_count)
        active_fraction = 100.0 * float(event.active_fraction)
        start_nm = self.station_distance_nm[event.start]
        end_nm = self.station_distance_nm[event.end - 1]
        self.title_text.set_text(
            f"Event {self.current_event_index}: {start_nm:.1f}-{end_nm:.1f} NM from merge | "
            f"active {active_count}/{len(self.matrix.flight_ids)} ({active_fraction:.1f}%) | "
            f"Z=({z[0]:.1f}, {z[1]:.1f})"
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
        background_alpha=args.background_alpha,
        background_linewidth=args.background_linewidth,
    )
    app.show()


if __name__ == "__main__":
    main()
