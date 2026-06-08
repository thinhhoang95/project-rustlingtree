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
from hllrd.geometry import LocalProjection, cumulative_distance_m  # noqa: E402
from hllrd.matrix import load_matrix_artifact  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "hllrd" / "south-east"
DEFAULT_MATRIX = DEFAULT_OUTPUT_DIR / "matrix_SE_from_merge.npz"
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "model_from_merge_L40_K6.npz"
NM_PER_M = 1.0 / 1852.0


def _simplify_xy_by_point_count(xy_m: np.ndarray, approximation_points: int) -> tuple[np.ndarray, int]:
    points = np.asarray(xy_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("xy_m must have shape N x 2")
    if points.shape[0] <= 2:
        return points.copy(), 0

    target_points = max(0, min(int(approximation_points), points.shape[0] - 2))
    retained = [0, points.shape[0] - 1]
    _error, fitted = _piecewise_linear_xy(points, retained)
    while len(retained) - 2 < target_points:
        best_index: int | None = None
        best_error = np.inf
        best_fitted = fitted
        retained_set = set(retained)
        for index in range(1, points.shape[0] - 1):
            if index in retained_set:
                continue
            candidate_error, candidate_fitted = _piecewise_linear_xy(points, [*retained, index])
            if candidate_error < best_error:
                best_index = index
                best_error = candidate_error
                best_fitted = candidate_fitted
        if best_index is None:
            break
        retained.append(best_index)
        retained.sort()
        fitted = best_fitted
    return fitted, len(retained) - 2


def _piecewise_linear_xy(xy_m: np.ndarray, retained_indices: list[int]) -> tuple[float, np.ndarray]:
    points = np.asarray(xy_m, dtype=float)
    retained = sorted(set(int(index) for index in retained_indices))
    if retained[0] != 0 or retained[-1] != points.shape[0] - 1:
        raise ValueError("retained_indices must include both endpoints")

    fitted = np.empty_like(points, dtype=float)
    x = np.arange(points.shape[0], dtype=float)
    for start, end in zip(retained[:-1], retained[1:], strict=True):
        fraction = (x[start : end + 1] - float(start)) / float(end - start)
        fitted[start : end + 1] = (1.0 - fraction[:, None]) * points[start] + fraction[:, None] * points[end]
    residual = points - fitted
    return float(np.sum(residual * residual)), fitted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive South-East HLLRD event response viewer.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="Path to matrix_SE_from_merge.npz.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to model_from_merge_L40_K6.npz.")
    parser.add_argument("--background-alpha", type=float, default=0.08, help="Transparency for all-flight background paths.")
    parser.add_argument("--background-linewidth", type=float, default=0.35, help="Line width for all-flight background paths.")
    parser.add_argument(
        "--raw-basis-response",
        action="store_true",
        help="Show the raw rank-2 basis response instead of the simplified local response.",
    )
    return parser


class SouthEastInteractive:
    def __init__(
        self,
        matrix_path: Path,
        model_path: Path,
        *,
        background_alpha: float,
        background_linewidth: float,
        raw_basis_response: bool,
    ) -> None:
        self.matrix = load_matrix_artifact(matrix_path)
        self.result = load_fit_result(model_path)
        if not self.result.events:
            raise RuntimeError(f"Model contains no HLLRD events: {model_path}")
        if self.matrix.X.shape[1] != self.result.residual.shape[1]:
            raise RuntimeError("Matrix station count does not match model station count.")

        self.background_alpha = float(background_alpha)
        self.background_linewidth = float(background_linewidth)
        self.raw_basis_response = bool(raw_basis_response)
        self.current_display_approximation_points: int | None = None
        self.projection = LocalProjection(self.matrix.origin_lat_deg, self.matrix.origin_lon_deg)
        self.mean_xy_m = self._mean_xy_m()
        self.station_distance_nm = cumulative_distance_m(self.mean_xy_m[:, 0], self.mean_xy_m[:, 1]) * NM_PER_M
        self.current_event_index = 0

        self.fig: Figure
        self.ax: Axes
        self.event_radio: RadioButtons
        self.z1_slider: Slider
        self.z2_slider: Slider
        self.response_line: Line2D
        self.response_segment_line: Line2D
        self.window_line: Line2D
        self.window_endpoints: PathCollection
        self.title_text: Text

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
        xy_m = self.mean_xy_m + normal_offset_m[:, None] * self.matrix.normals_xy
        if not self.raw_basis_response:
            xy_m, self.current_display_approximation_points = self._simplified_display_xy(
                xy_m,
                normal_offset_m,
                event.start,
                event.end,
                event.simplifier,
            )
        else:
            self.current_display_approximation_points = None
        return xy_m

    def _simplified_display_xy(
        self,
        xy_m: np.ndarray,
        normal_offset_m: np.ndarray,
        start: int,
        end: int,
        simplifier: dict[str, object],
    ) -> tuple[np.ndarray, int | None]:
        if not simplifier.get("enabled", False):
            return xy_m, None
        local_offset = np.asarray(normal_offset_m[start:end], dtype=float)
        local_xy = np.asarray(xy_m[start:end], dtype=float)
        if local_xy.shape[0] <= 2 or np.allclose(local_offset, 0.0):
            return xy_m, 0
        approximation_points = int(round(float(simplifier.get("median_approximation_points", 0.0))))
        approximation_points = max(0, min(approximation_points, int(simplifier.get("max_approximation_points", 4))))
        simplified_xy, used_points = _simplify_xy_by_point_count(local_xy, approximation_points)
        displayed = np.asarray(xy_m, dtype=float).copy()
        displayed[start:end] = simplified_xy
        return displayed, used_points

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

    def on_event_selected(self, label: str | None) -> None:
        if label is None:
            return
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
            f"Z=({z[0]:.1f}, {z[1]:.1f}) | "
            f"{self._display_mode_label()}"
        )
        self.fig.canvas.draw_idle()

    def _display_mode_label(self) -> str:
        if self.raw_basis_response:
            return "raw basis"
        if self.current_display_approximation_points is None:
            return "raw basis"
        return f"approx pts: {self.current_display_approximation_points}"

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
        raw_basis_response=args.raw_basis_response,
    )
    app.show()


if __name__ == "__main__":
    main()
