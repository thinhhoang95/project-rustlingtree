from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hllrd.fit import HLLRDEvent, HLLRDFitResult
from hllrd.matrix import MatrixArtifact


Family = Literal["vertical", "horizontal"]


@dataclass(frozen=True)
class ElasticFPCAConfig:
    components: int = 3
    std_grid: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    min_active_flights: int = 5
    parallel: bool = False
    cores: int = 1

    def validate(self) -> None:
        if self.components < 1:
            raise ValueError("components must be at least 1")
        if self.min_active_flights < 1:
            raise ValueError("min_active_flights must be at least 1")
        if self.cores == 0:
            raise ValueError("cores must be nonzero")
        grid = np.asarray(self.std_grid, dtype=float)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("std_grid must contain at least two values")
        if not np.all(np.isfinite(grid)):
            raise ValueError("std_grid must contain finite values")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("std_grid must be strictly increasing")
        if not np.any(np.isclose(grid, 0.0)):
            raise ValueError("std_grid must include 0.0")


@dataclass(frozen=True)
class ElasticEventFPCA:
    event_index: int
    start: int
    end: int
    active_rows: np.ndarray
    active_flight_ids: tuple[str, ...]
    time: np.ndarray
    functions: np.ndarray
    fmean: np.ndarray
    aligned_functions: np.ndarray
    warps: np.ndarray
    vertical_f_pca: np.ndarray
    vertical_coefficients: np.ndarray
    vertical_latent: np.ndarray
    horizontal_gam_pca: np.ndarray
    horizontal_coefficients: np.ndarray
    horizontal_latent: np.ndarray

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def active_count(self) -> int:
        return len(self.active_flight_ids)

    @property
    def component_count(self) -> int:
        return int(min(self.vertical_f_pca.shape[2], self.horizontal_gam_pca.shape[2]))


@dataclass(frozen=True)
class ElasticFPCAResult:
    events: tuple[ElasticEventFPCA, ...]
    config: ElasticFPCAConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def event_by_index(self, event_index: int) -> ElasticEventFPCA:
        for event in self.events:
            if event.event_index == event_index:
                return event
        raise KeyError(f"elastic FPCA artifact has no event {event_index}")

    @property
    def max_components(self) -> int:
        if not self.events:
            return 0
        return max(event.component_count for event in self.events)


def extract_event_functions(
    event: HLLRDEvent,
    event_index: int,
    flight_ids: tuple[str, ...],
) -> ElasticEventFPCA:
    """Extract active event-local normal-offset functions for fdasrsf."""

    active_rows = np.flatnonzero(event.active_mask)
    reconstruction = event.coefficients @ event.basis.T
    functions = reconstruction[active_rows, event.start : event.end].T
    if functions.ndim != 2:
        raise ValueError("extracted event functions must be a 2D array")
    if not np.all(np.isfinite(functions)):
        raise ValueError(f"event {event_index} contains non-finite local functions")
    active_flight_ids = tuple(flight_ids[int(row)] for row in active_rows)
    time = np.linspace(0.0, 1.0, event.length)
    empty_components = np.zeros((event.length, 0, 0), dtype=float)
    return ElasticEventFPCA(
        event_index=int(event_index),
        start=int(event.start),
        end=int(event.end),
        active_rows=active_rows.astype(int),
        active_flight_ids=active_flight_ids,
        time=time,
        functions=functions,
        fmean=np.zeros(event.length, dtype=float),
        aligned_functions=np.zeros_like(functions),
        warps=np.zeros_like(functions),
        vertical_f_pca=empty_components,
        vertical_coefficients=np.zeros((functions.shape[1], 0), dtype=float),
        vertical_latent=np.zeros(0, dtype=float),
        horizontal_gam_pca=np.zeros((0, event.length, 0), dtype=float),
        horizontal_coefficients=np.zeros((functions.shape[1], 0), dtype=float),
        horizontal_latent=np.zeros(0, dtype=float),
    )


def fit_elastic_event_fpca(
    matrix: MatrixArtifact,
    model: HLLRDFitResult,
    *,
    config: ElasticFPCAConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> ElasticFPCAResult:
    cfg = config or ElasticFPCAConfig()
    cfg.validate()
    if matrix.X.shape[0] != model.residual.shape[0]:
        raise ValueError("matrix flight count does not match model flight count")
    if matrix.X.shape[1] != model.residual.shape[1]:
        raise ValueError("matrix station count does not match model station count")
    expected_stations = np.linspace(0.0, 1.0, matrix.X.shape[1])
    if matrix.stations.shape != expected_stations.shape or not np.allclose(matrix.stations, expected_stations):
        raise ValueError("elastic FPCA requires a uniform constant-speed station grid")

    fs = _import_fdasrsf()
    std_grid = np.asarray(cfg.std_grid, dtype=float)
    events: list[ElasticEventFPCA] = []
    skipped: list[dict[str, Any]] = []
    for event_index, hllrd_event in enumerate(model.events):
        extracted = extract_event_functions(hllrd_event, event_index, matrix.flight_ids)
        if extracted.active_count < cfg.min_active_flights:
            skipped.append(
                {
                    "event": event_index,
                    "reason": "min_active_flights",
                    "active_count": extracted.active_count,
                }
            )
            continue
        component_count = min(cfg.components, extracted.length, extracted.active_count)
        if component_count < 1:
            skipped.append({"event": event_index, "reason": "no_components"})
            continue

        warp = fs.fdawarp(np.ascontiguousarray(extracted.functions), extracted.time)
        warp.srsf_align(
            parallel=bool(cfg.parallel),
            cores=int(cfg.cores),
            verbose=False,
        )
        vpca = fs.fdavpca(warp)
        vpca.calc_fpca(no=component_count, stds=std_grid)
        hpca = fs.fdahpca(warp)
        hpca.calc_fpca(no=component_count, stds=std_grid)

        vertical_f_pca = _coerce_vertical_pca(np.asarray(vpca.f_pca, dtype=float), extracted.length, std_grid.size)
        horizontal_gam_pca = _coerce_horizontal_pca(
            np.asarray(hpca.gam_pca, dtype=float),
            extracted.length,
            std_grid.size,
        )
        events.append(
            ElasticEventFPCA(
                event_index=extracted.event_index,
                start=extracted.start,
                end=extracted.end,
                active_rows=extracted.active_rows,
                active_flight_ids=extracted.active_flight_ids,
                time=extracted.time,
                functions=extracted.functions,
                fmean=np.asarray(warp.fmean, dtype=float),
                aligned_functions=np.asarray(warp.fn, dtype=float),
                warps=np.asarray(warp.gam, dtype=float),
                vertical_f_pca=vertical_f_pca,
                vertical_coefficients=np.asarray(vpca.coef, dtype=float),
                vertical_latent=np.asarray(vpca.latent, dtype=float),
                horizontal_gam_pca=horizontal_gam_pca,
                horizontal_coefficients=np.asarray(hpca.coef, dtype=float),
                horizontal_latent=np.asarray(hpca.latent, dtype=float),
            )
        )

    return ElasticFPCAResult(
        events=tuple(events),
        config=cfg,
        metadata={
            **(metadata or {}),
            "skipped_events": skipped,
        },
    )


def vertical_component_delta(
    event: ElasticEventFPCA,
    component: int,
    amplitude_std: float,
    std_grid: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    component_index = _validate_component(event, component)
    if np.isclose(float(amplitude_std), 0.0):
        return np.zeros(event.length, dtype=float)
    grid = np.asarray(std_grid, dtype=float)
    values = _interp_std_grid(event.vertical_f_pca[:, :, component_index], grid, float(amplitude_std))
    baseline = _interp_std_grid(event.vertical_f_pca[:, :, component_index], grid, 0.0)
    return values - baseline


def horizontal_component_delta(
    event: ElasticEventFPCA,
    component: int,
    amplitude_std: float,
    std_grid: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    component_index = _validate_component(event, component)
    if np.isclose(float(amplitude_std), 0.0):
        return np.zeros(event.length, dtype=float)
    grid = np.asarray(std_grid, dtype=float)
    gammas = event.horizontal_gam_pca[:, :, component_index]
    gamma = _interp_std_grid(gammas.T, grid, float(amplitude_std))
    gamma = np.clip(gamma, 0.0, 1.0)
    if gamma.size:
        gamma[0] = 0.0
        gamma[-1] = 1.0
    inverse_gamma = _invert_warp(gamma, event.time)
    warped = np.interp(inverse_gamma, event.time, event.fmean)
    return warped - event.fmean


def elastic_component_delta(
    event: ElasticEventFPCA,
    family: Family,
    component: int,
    amplitude_std: float,
    std_grid: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    if family == "vertical":
        return vertical_component_delta(event, component, amplitude_std, std_grid)
    if family == "horizontal":
        return horizontal_component_delta(event, component, amplitude_std, std_grid)
    raise ValueError("family must be 'vertical' or 'horizontal'")


def save_elastic_fpca_result(path: Path, result: ElasticFPCAResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_count = len(result.events)
    std_grid = np.asarray(result.config.std_grid, dtype=float)
    max_length = max((event.length for event in result.events), default=0)
    max_active = max((event.active_count for event in result.events), default=0)
    max_components = max((event.component_count for event in result.events), default=0)
    max_flight_id_length = max(
        (len(flight_id) for event in result.events for flight_id in event.active_flight_ids),
        default=1,
    )

    event_indices = np.zeros(event_count, dtype=int)
    intervals = np.zeros((event_count, 2), dtype=int)
    lengths = np.zeros(event_count, dtype=int)
    active_counts = np.zeros(event_count, dtype=int)
    component_counts = np.zeros(event_count, dtype=int)
    active_rows = np.full((event_count, max_active), -1, dtype=int)
    active_flight_ids = np.full((event_count, max_active), "", dtype=f"<U{max_flight_id_length}")
    time = np.full((event_count, max_length), np.nan, dtype=float)
    functions = np.full((event_count, max_length, max_active), np.nan, dtype=float)
    fmean = np.full((event_count, max_length), np.nan, dtype=float)
    aligned = np.full((event_count, max_length, max_active), np.nan, dtype=float)
    warps = np.full((event_count, max_length, max_active), np.nan, dtype=float)
    vertical_f_pca = np.full((event_count, max_length, std_grid.size, max_components), np.nan, dtype=float)
    vertical_coefficients = np.full((event_count, max_active, max_components), np.nan, dtype=float)
    vertical_latent = np.full((event_count, max_components), np.nan, dtype=float)
    horizontal_gam_pca = np.full((event_count, std_grid.size, max_length, max_components), np.nan, dtype=float)
    horizontal_coefficients = np.full((event_count, max_active, max_components), np.nan, dtype=float)
    horizontal_latent = np.full((event_count, max_components), np.nan, dtype=float)

    for row, event in enumerate(result.events):
        _validate_event_artifact_shapes(event)
        length = event.length
        active_count = event.active_count
        components = event.component_count
        event_indices[row] = event.event_index
        intervals[row] = [event.start, event.end]
        lengths[row] = length
        active_counts[row] = active_count
        component_counts[row] = components
        active_rows[row, :active_count] = event.active_rows
        active_flight_ids[row, :active_count] = np.asarray(event.active_flight_ids, dtype=str)
        time[row, :length] = event.time
        functions[row, :length, :active_count] = event.functions
        fmean[row, :length] = event.fmean
        aligned[row, :length, :active_count] = event.aligned_functions
        warps[row, :length, :active_count] = event.warps
        vertical_f_pca[row, :length, :, :components] = event.vertical_f_pca
        vertical_coefficients[row, :active_count, :components] = event.vertical_coefficients
        vertical_latent[row, :components] = event.vertical_latent
        horizontal_gam_pca[row, :, :length, :components] = event.horizontal_gam_pca
        horizontal_coefficients[row, :active_count, :components] = event.horizontal_coefficients
        horizontal_latent[row, :components] = event.horizontal_latent

    np.savez_compressed(
        path,
        event_indices=event_indices,
        intervals=intervals,
        lengths=lengths,
        active_counts=active_counts,
        component_counts=component_counts,
        active_rows=active_rows,
        active_flight_ids=active_flight_ids,
        time=time,
        functions=functions,
        fmean=fmean,
        aligned_functions=aligned,
        warps=warps,
        vertical_f_pca=vertical_f_pca,
        vertical_coefficients=vertical_coefficients,
        vertical_latent=vertical_latent,
        horizontal_gam_pca=horizontal_gam_pca,
        horizontal_coefficients=horizontal_coefficients,
        horizontal_latent=horizontal_latent,
        std_grid=std_grid,
        config=np.asarray(json.dumps(asdict(result.config), sort_keys=True), dtype=str),
        metadata=np.asarray(json.dumps(result.metadata, sort_keys=True), dtype=str),
    )


def load_elastic_fpca_result(path: Path) -> ElasticFPCAResult:
    with np.load(path, allow_pickle=False) as data:
        config_payload = json.loads(str(np.asarray(data["config"]).item()))
        config_payload["std_grid"] = tuple(float(value) for value in config_payload["std_grid"])
        config = ElasticFPCAConfig(**config_payload)
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        events: list[ElasticEventFPCA] = []
        event_indices = np.asarray(data["event_indices"], dtype=int)
        intervals = np.asarray(data["intervals"], dtype=int)
        lengths = np.asarray(data["lengths"], dtype=int)
        active_counts = np.asarray(data["active_counts"], dtype=int)
        component_counts = np.asarray(data["component_counts"], dtype=int)
        for row, event_index in enumerate(event_indices.tolist()):
            length = int(lengths[row])
            active_count = int(active_counts[row])
            components = int(component_counts[row])
            events.append(
                ElasticEventFPCA(
                    event_index=int(event_index),
                    start=int(intervals[row, 0]),
                    end=int(intervals[row, 1]),
                    active_rows=np.asarray(data["active_rows"][row, :active_count], dtype=int),
                    active_flight_ids=tuple(
                        str(item) for item in data["active_flight_ids"][row, :active_count].tolist()
                    ),
                    time=np.asarray(data["time"][row, :length], dtype=float),
                    functions=np.asarray(data["functions"][row, :length, :active_count], dtype=float),
                    fmean=np.asarray(data["fmean"][row, :length], dtype=float),
                    aligned_functions=np.asarray(data["aligned_functions"][row, :length, :active_count], dtype=float),
                    warps=np.asarray(data["warps"][row, :length, :active_count], dtype=float),
                    vertical_f_pca=np.asarray(data["vertical_f_pca"][row, :length, :, :components], dtype=float),
                    vertical_coefficients=np.asarray(
                        data["vertical_coefficients"][row, :active_count, :components],
                        dtype=float,
                    ),
                    vertical_latent=np.asarray(data["vertical_latent"][row, :components], dtype=float),
                    horizontal_gam_pca=np.asarray(
                        data["horizontal_gam_pca"][row, :, :length, :components],
                        dtype=float,
                    ),
                    horizontal_coefficients=np.asarray(
                        data["horizontal_coefficients"][row, :active_count, :components],
                        dtype=float,
                    ),
                    horizontal_latent=np.asarray(data["horizontal_latent"][row, :components], dtype=float),
                )
            )
    return ElasticFPCAResult(events=tuple(events), config=config, metadata=metadata)


def _validate_event_artifact_shapes(event: ElasticEventFPCA) -> None:
    length = event.length
    active_count = event.active_count
    if length < 1:
        raise ValueError(f"event {event.event_index} must have positive length")
    if event.time.shape != (length,):
        raise ValueError(f"event {event.event_index} time shape does not match its interval length")
    for name, values in (
        ("functions", event.functions),
        ("aligned_functions", event.aligned_functions),
        ("warps", event.warps),
    ):
        if values.shape != (length, active_count):
            raise ValueError(f"event {event.event_index} {name} shape must be length x active_count")
    if event.fmean.shape != (length,):
        raise ValueError(f"event {event.event_index} fmean shape does not match its interval length")
    if event.vertical_f_pca.ndim != 3 or event.vertical_f_pca.shape[0] != length:
        raise ValueError(f"event {event.event_index} vertical_f_pca shape must start with interval length")
    if event.horizontal_gam_pca.ndim != 3 or event.horizontal_gam_pca.shape[1] != length:
        raise ValueError(f"event {event.event_index} horizontal_gam_pca shape must include interval length")
    if event.vertical_coefficients.shape[0] != active_count:
        raise ValueError(f"event {event.event_index} vertical coefficient rows must match active_count")
    if event.horizontal_coefficients.shape[0] != active_count:
        raise ValueError(f"event {event.event_index} horizontal coefficient rows must match active_count")


def _import_fdasrsf() -> Any:
    try:
        import fdasrsf as fs
    except ImportError as exc:
        raise RuntimeError("fdasrsf is required for elastic FPCA; install fdasrsf>=2.6.9") from exc
    return fs


def _coerce_vertical_pca(values: np.ndarray, length: int, std_count: int) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError("fdasrsf vertical f_pca must be a 3D array")
    if values.shape[0] == length and values.shape[1] == std_count:
        return values
    if values.shape[0] == std_count and values.shape[1] == length:
        return np.transpose(values, (1, 0, 2))
    raise ValueError(f"unexpected vertical f_pca shape {values.shape}")


def _coerce_horizontal_pca(values: np.ndarray, length: int, std_count: int) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError("fdasrsf horizontal gam_pca must be a 3D array")
    if values.shape[0] == std_count and values.shape[1] == length:
        return values
    if values.shape[0] == length and values.shape[1] == std_count:
        return np.transpose(values, (1, 0, 2))
    raise ValueError(f"unexpected horizontal gam_pca shape {values.shape}")


def _validate_component(event: ElasticEventFPCA, component: int) -> int:
    component_index = int(component)
    if component_index < 0 or component_index >= event.component_count:
        raise ValueError(f"component must be in [0, {event.component_count - 1}]")
    return component_index


def _interp_std_grid(values_by_station: np.ndarray, std_grid: np.ndarray, amplitude_std: float) -> np.ndarray:
    values = np.asarray(values_by_station, dtype=float)
    if values.ndim != 2 or values.shape[1] != std_grid.size:
        raise ValueError("values_by_station must have shape station_count x std_count")
    clipped = float(np.clip(amplitude_std, float(std_grid[0]), float(std_grid[-1])))
    return np.asarray([np.interp(clipped, std_grid, row) for row in values], dtype=float)


def _invert_warp(gamma: np.ndarray, time: np.ndarray) -> np.ndarray:
    warp = np.asarray(gamma, dtype=float)
    grid = np.asarray(time, dtype=float)
    if warp.shape != grid.shape:
        raise ValueError("gamma and time must have the same shape")
    if warp.ndim != 1:
        raise ValueError("gamma and time must be one-dimensional")
    if warp.size == 0:
        return warp.copy()
    monotone_warp = np.maximum.accumulate(np.clip(warp, float(grid[0]), float(grid[-1])))
    monotone_warp[0] = float(grid[0])
    monotone_warp[-1] = float(grid[-1])
    unique_warp, unique_indices = np.unique(monotone_warp, return_index=True)
    if unique_warp.size < 2:
        return grid.copy()
    return np.interp(grid, unique_warp, grid[unique_indices])
