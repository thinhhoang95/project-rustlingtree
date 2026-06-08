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
from hllrd.geometry import LocalProjection, cumulative_distance_m  # noqa: E402
from hllrd.matrix import load_matrix_artifact  # noqa: E402
from scenario.trajectory_compressor.io import load_raw_adsb, split_tracks_by_gap  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "hllrd" / "south-east"
DEFAULT_MATRIX = DEFAULT_OUTPUT_DIR / "matrix_SE_from_merge.npz"
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "model_from_merge_L40_K6.npz"
DEFAULT_SPLIT_GAP_SECONDS = 25 * 60
NM_PER_M = 1.0 / 1852.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive South-East HLLRD event response viewer.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="Path to matrix_SE_from_merge.npz.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to model_from_merge_L40_K6.npz.")
    parser.add_argument("--raw-adsb-dir", type=Path, default=None, help="Raw ADS-B directory for grey background tracks.")
    parser.add_argument("--split-gap-seconds", type=int, default=DEFAULT_SPLIT_GAP_SECONDS)
    parser.add_argument("--processes", type=int, default=1, help="Raw ADS-B loading worker count. Defaults to 1 for GUI launch reliability.")
    parser.add_argument("--background-source", choices=["raw", "matrix"], default="raw")
    parser.add_argument("--background-alpha", type=float, default=0.08, help="Transparency for all-flight background paths.")
    parser.add_argument("--background-linewidth", type=float, default=0.35, help="Line width for all-flight background paths.")
    parser.add_argument(
        "--lag-range-stations",
        type=int,
        default=12,
        help="Fallback exploratory lag range for events without learned lag offsets.",
    )
    return parser


class SouthEastInteractive:
    def __init__(
        self,
        matrix_path: Path,
        model_path: Path,
        *,
        raw_adsb_dir: Path | None,
        split_gap_seconds: int,
        processes: int,
        background_source: str,
        background_alpha: float,
        background_linewidth: float,
        lag_range_stations: int,
    ) -> None:
        self.matrix = load_matrix_artifact(matrix_path)
        self.result = load_fit_result(model_path)
        if not self.result.events:
            raise RuntimeError(f"Model contains no HLLRD events: {model_path}")
        if self.matrix.X.shape[1] != self.result.residual.shape[1]:
            raise RuntimeError("Matrix station count does not match model station count.")

        self.background_alpha = float(background_alpha)
        self.background_linewidth = float(background_linewidth)
        self.background_source = str(background_source)
        self.lag_range_stations = max(0, int(lag_range_stations))
        self.projection = LocalProjection(self.matrix.origin_lat_deg, self.matrix.origin_lon_deg)
        self.mean_xy_m = self._mean_xy_m()
        self.station_distance_nm = cumulative_distance_m(self.mean_xy_m[:, 0], self.mean_xy_m[:, 1]) * NM_PER_M
        self.raw_background_tracks = (
            self._load_raw_background_tracks(raw_adsb_dir, split_gap_seconds, processes)
            if self.background_source == "raw"
            else None
        )
        self.current_event_index = 0

        self.fig: Figure
        self.ax: Axes
        self.event_radio: RadioButtons
        self.z1_slider: Slider
        self.z2_slider: Slider
        self.shift_slider: Slider
        self.extend_slider: Slider
        self.response_line: Line2D
        self.response_segment_line: Line2D
        self.window_line: Line2D
        self.window_endpoints: PathCollection
        self.title_text: Text

    def _mean_xy_m(self) -> np.ndarray:
        return self.matrix.reference_xy_m + self.result.column_center[:, None] * self.matrix.normals_xy

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

    def event_response_xy_m(self, z: np.ndarray, shift_offset: int, extend_offset: int) -> np.ndarray:
        event = self.result.events[self.current_event_index]
        basis = self.extended_turn_basis(event, extend_offset)
        basis = self.shifted_basis(basis, shift_offset)
        normal_offset_m = basis @ z
        return self.mean_xy_m + normal_offset_m[:, None] * self.matrix.normals_xy

    def extended_turn_basis(self, event: object, extend_offset: int) -> np.ndarray:
        base = np.asarray(event.basis, dtype=float)
        extension = int(extend_offset)
        if extension == 0:
            return base.copy()

        start = int(event.start)
        end = int(event.end)
        local = base[start:end, :]
        if local.shape[0] < 2:
            return base.copy()
        target_length = max(2, min(local.shape[0] + extension, base.shape[0] - start))
        if target_length == local.shape[0]:
            return base.copy()

        if extension > 0:
            local_deformed = self.extend_local_basis_at_peak(local, event, extension)
            local_deformed = local_deformed[:target_length, :]
        else:
            local_deformed = self.resample_local_basis(local, target_length)

        deformed = np.zeros_like(base)
        deformed[start : start + local_deformed.shape[0], :] = local_deformed
        return deformed

    def extend_local_basis_at_peak(self, local_basis: np.ndarray, event: object, extension: int) -> np.ndarray:
        local = np.asarray(local_basis, dtype=float)
        lag = max(0, int(extension))
        pivot = int(np.clip(int(event.peak_index) - int(event.start), 0, local.shape[0] - 1))
        hold = np.repeat(local[pivot : pivot + 1, :], lag, axis=0)
        return np.vstack((local[: pivot + 1, :], hold, local[pivot + 1 :, :]))

    def resample_local_basis(self, local_basis: np.ndarray, target_length: int) -> np.ndarray:
        local = np.asarray(local_basis, dtype=float)
        source = np.linspace(0.0, 1.0, local.shape[0])
        target = np.linspace(0.0, 1.0, int(target_length))
        return np.column_stack([np.interp(target, source, local[:, component]) for component in range(local.shape[1])])

    def shifted_basis(self, basis: np.ndarray, lag_offset: int) -> np.ndarray:
        base = np.asarray(basis, dtype=float)
        shifted = np.zeros_like(base)
        lag = int(lag_offset)
        if lag == 0:
            return base.copy()
        if abs(lag) >= base.shape[0]:
            return shifted
        if lag > 0:
            shifted[lag:, :] = base[:-lag, :]
        else:
            shifted[:lag, :] = base[-lag:, :]
        return shifted

    def deformed_event_window(self, event_index: int, shift_offset: int, extend_offset: int) -> tuple[int, int]:
        event = self.result.events[event_index]
        event_length = event.end - event.start
        extended_length = max(2, min(event_length + int(extend_offset), self.matrix.X.shape[1] - event.start))
        start = max(0, event.start + int(shift_offset))
        end = min(self.matrix.X.shape[1], event.start + extended_length + int(shift_offset))
        if end <= start:
            return event.start, event.end
        return start, end

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

    def event_lag_offsets(self, event_index: int) -> np.ndarray:
        event = self.result.events[event_index]
        if event.lag_offsets is None:
            return np.zeros(0, dtype=int)
        values = np.asarray(event.lag_offsets, dtype=int)
        active = np.asarray(event.active_mask, dtype=bool)
        if active.shape == values.shape and np.any(active):
            values = values[active]
        return values

    def event_extension_offsets(self, event_index: int) -> np.ndarray:
        event = self.result.events[event_index]
        if event.extension_offsets is None:
            return np.zeros(0, dtype=int)
        values = np.asarray(event.extension_offsets, dtype=int)
        active = np.asarray(event.active_mask, dtype=bool)
        if active.shape == values.shape and np.any(active):
            values = values[active]
        return values

    def shift_slider_limits(self, event_index: int) -> tuple[int, int]:
        event = self.result.events[event_index]
        learned_lags = self.event_lag_offsets(event_index)
        if learned_lags.size:
            lower = int(min(np.min(learned_lags), 0))
            upper = int(max(np.max(learned_lags), 0))
        else:
            config_lag = int(getattr(self.result.config, "max_lag_stations", 0) or 0)
            fallback = max(config_lag, self.lag_range_stations)
            lower = -fallback
            upper = fallback
        lower = max(lower, -event.start)
        upper = min(upper, self.matrix.X.shape[1] - event.end)
        if lower == upper:
            fallback = max(1, self.lag_range_stations)
            lower = max(-fallback, -event.start)
            upper = min(fallback, self.matrix.X.shape[1] - event.end)
        return int(lower), int(upper)

    def extend_slider_limits(self, event_index: int) -> tuple[int, int]:
        event = self.result.events[event_index]
        learned_extensions = self.event_extension_offsets(event_index)
        learned_span = int(np.max(np.abs(learned_extensions), initial=0)) if learned_extensions.size else 0
        config_extension = int(getattr(self.result.config, "max_extend_stations", 0) or 0)
        config_lag = int(getattr(self.result.config, "max_lag_stations", 0) or 0)
        fallback = max(learned_span, config_extension, config_lag, self.lag_range_stations)
        lower = -min(max(1, fallback), max(1, event.length - 2))
        upper = min(max(1, fallback), self.matrix.X.shape[1] - event.end)
        return int(lower), int(upper)

    def create(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(10.5, 7.6))
        self.fig.subplots_adjust(left=0.25, bottom=0.27, right=0.96, top=0.9)
        self.ax.set_facecolor("#f7f7f7")

        self.plot_static_layers()

        z1_min, z1_max = self.slider_limits(0, 0)
        z2_min, z2_max = self.slider_limits(0, 1)
        shift_min, shift_max = self.shift_slider_limits(0)
        extend_min, extend_max = self.extend_slider_limits(0)

        event_ax = self.fig.add_axes((0.025, 0.38, 0.16, 0.48))
        event_labels = [str(index) for index in range(len(self.result.events))]
        self.event_radio = RadioButtons(event_ax, event_labels, active=0)
        event_ax.set_title("Event ID", fontsize=10)

        z1_ax = self.fig.add_axes((0.25, 0.18, 0.65, 0.03))
        z2_ax = self.fig.add_axes((0.25, 0.13, 0.65, 0.03))
        shift_ax = self.fig.add_axes((0.25, 0.08, 0.65, 0.03))
        extend_ax = self.fig.add_axes((0.25, 0.03, 0.65, 0.03))
        self.z1_slider = Slider(z1_ax, "Z1", z1_min, z1_max, valinit=0.0)
        self.z2_slider = Slider(z2_ax, "Z2", z2_min, z2_max, valinit=0.0)
        self.shift_slider = Slider(shift_ax, "Shift", shift_min, shift_max, valinit=0, valstep=1)
        self.extend_slider = Slider(extend_ax, "Extend", extend_min, extend_max, valinit=0, valstep=1)

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
        self.window_line = self.ax.plot([], [], color="black", linewidth=5.0, alpha=0.9, solid_capstyle="round", label="window on center", zorder=4)[0]
        self.window_endpoints = self.ax.scatter([], [], s=24, color="black", edgecolor="white", linewidth=0.5, zorder=7)
        self.title_text = self.ax.set_title("")

        self.event_radio.on_clicked(self.on_event_selected)
        self.z1_slider.on_changed(lambda _value: self.update_response())
        self.z2_slider.on_changed(lambda _value: self.update_response())
        self.shift_slider.on_changed(lambda _value: self.update_response())
        self.extend_slider.on_changed(lambda _value: self.update_response())
        self.update_event(0)
        self.ax.legend(loc="best", fontsize=8)

    def plot_static_layers(self) -> None:
        mean_lat, mean_lon = self.xy_to_latlon(self.mean_xy_m)
        self.ax.plot(mean_lon, mean_lat, color="#303030", linewidth=2.0, alpha=0.82, label="model center trajectory", zorder=3)
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
        shift_lower, shift_upper = self.shift_slider_limits(self.current_event_index)
        self.shift_slider.eventson = False
        self.shift_slider.valmin = shift_lower
        self.shift_slider.valmax = shift_upper
        self.shift_slider.valstep = 1
        self.shift_slider.ax.set_xlim(shift_lower, shift_upper)
        self.shift_slider.set_val(0)
        self.shift_slider.eventson = True

        extend_lower, extend_upper = self.extend_slider_limits(self.current_event_index)
        self.extend_slider.eventson = False
        self.extend_slider.valmin = extend_lower
        self.extend_slider.valmax = extend_upper
        self.extend_slider.valstep = 1
        self.extend_slider.ax.set_xlim(extend_lower, extend_upper)
        self.extend_slider.set_val(0)
        self.extend_slider.eventson = True
        self.update_response()

    def update_response(self) -> None:
        event = self.result.events[self.current_event_index]
        z = np.asarray([self.z1_slider.val, self.z2_slider.val], dtype=float)
        shift = int(round(float(self.shift_slider.val)))
        extension = int(round(float(self.extend_slider.val)))
        xy_m = self.event_response_xy_m(z, shift, extension)
        lat, lon = self.xy_to_latlon(xy_m)
        self.response_line.set_data(lon, lat)

        window_start, window_end = self.deformed_event_window(self.current_event_index, shift, extension)
        segment = xy_m[window_start:window_end]
        segment_lat, segment_lon = self.xy_to_latlon(segment)
        self.response_segment_line.set_data(segment_lon, segment_lat)

        mean_segment = self.mean_xy_m[window_start:window_end]
        mean_segment_lat, mean_segment_lon = self.xy_to_latlon(mean_segment)
        self.window_line.set_data(mean_segment_lon, mean_segment_lat)
        self.window_endpoints.set_offsets(np.column_stack(([mean_segment_lon[0], mean_segment_lon[-1]], [mean_segment_lat[0], mean_segment_lat[-1]])))

        active_count = int(event.active_count)
        active_fraction = 100.0 * float(event.active_fraction)
        start_nm = self.station_distance_nm[window_start]
        end_nm = self.station_distance_nm[window_end - 1]
        lag_description = self.lag_description(self.current_event_index)
        self.title_text.set_text(
            f"Event {self.current_event_index}: {start_nm:.1f}-{end_nm:.1f} NM from merge | "
            f"active {active_count}/{len(self.matrix.flight_ids)} ({active_fraction:.1f}%) | "
            f"Z=({z[0]:.1f}, {z[1]:.1f}) | shift={shift:+d} st | extend={extension:+d} st{lag_description}"
        )
        self.fig.canvas.draw_idle()

    def lag_description(self, event_index: int) -> str:
        parts: list[str] = []
        lag_values = self.event_lag_offsets(event_index)
        if lag_values.size:
            parts.append(f"lag {int(np.min(lag_values)):+d}..{int(np.max(lag_values)):+d}")
        extension_values = self.event_extension_offsets(event_index)
        if extension_values.size:
            parts.append(f"extend {int(np.min(extension_values)):+d}..{int(np.max(extension_values)):+d}")
        if not parts:
            return ""
        return " | learned " + ", ".join(parts)

    def show(self) -> None:
        self.create()
        plt.show()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app = SouthEastInteractive(
        args.matrix,
        args.model,
        raw_adsb_dir=args.raw_adsb_dir,
        split_gap_seconds=args.split_gap_seconds,
        processes=args.processes,
        background_source=args.background_source,
        background_alpha=args.background_alpha,
        background_linewidth=args.background_linewidth,
        lag_range_stations=args.lag_range_stations,
    )
    app.show()


if __name__ == "__main__":
    main()
