from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from hllrd.candidates import (
    CandidateFit,
    analytic_null_threshold,
    backtrack_peak_rise_start,
    centered_interval,
    find_residual_energy_peaks,
    is_duplicate_interval,
    length_grid,
    local_rank2_candidate,
    smooth_energy,
)
from hllrd.matrix import estimate_noise_sigma, robust_center_columns
from hllrd.simplifier import simplify_local_deviation_block


@dataclass(frozen=True)
class HLLRDV1Config:
    L_min: int | None = None
    L_max: int | None = None
    kappa_peak: float = 2.0
    min_peak_distance: int | None = None
    smoothing_window: int = 5
    activation_scale: float = 1.0
    n_min: int | None = None
    K_max: int | None = None
    epsilon_gain: float = 0.001
    lambda_i: float = 0.0
    lambda_activation: float = 0.0
    c_null: float = 4.0
    empirical_null_repeats: int = 12
    empirical_null_quantile: float = 0.99
    empirical_null_seed: int = 1729
    ridge: float = 1e-6
    endpoint_trim_threshold: float = 0.05
    duplicate_iou_threshold: float = 0.8
    keep_next_longer: bool = False
    center_method: str = "median"
    peak_backtrack_enabled: bool = True
    peak_backtrack_rise_fraction: float = 0.05
    local_simplifier_enabled: bool = False
    local_simplifier_gain_sigma: float = 128.0
    local_simplifier_max_points: int = 4
    local_simplifier_max_relative_loss: float = 0.05


@dataclass(frozen=True)
class HLLRDEvent:
    start: int
    end: int
    basis: np.ndarray
    coefficients: np.ndarray
    active_mask: np.ndarray
    raw_gain: float
    active_gain: float
    score: float
    threshold: float
    peak_index: int
    simplifier: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    @property
    def active_fraction(self) -> float:
        return float(np.mean(self.active_mask)) if self.active_mask.size else 0.0


@dataclass(frozen=True)
class HLLRDFitResult:
    events: tuple[HLLRDEvent, ...]
    dictionary: np.ndarray
    coefficients: np.ndarray
    reconstruction: np.ndarray
    residual: np.ndarray
    explained_fraction: float
    sigma_hat: float
    activation_energy_floor: float
    column_center: np.ndarray
    config: HLLRDV1Config
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HLLRDTransformResult:
    coefficients: np.ndarray
    reconstruction: np.ndarray
    residual: np.ndarray
    explained_fraction: float
    active_counts: np.ndarray


def fit_localized_low_rank(
    X: np.ndarray,
    config: HLLRDV1Config | None = None,
    *,
    already_centered: bool = False,
    metadata: dict[str, Any] | None = None,
) -> HLLRDFitResult:
    cfg = config or HLLRDV1Config()
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2D matrix")
    n, M = matrix.shape
    if n == 0 or M == 0:
        raise ValueError("X must be non-empty")

    if already_centered:
        X_centered = matrix.copy()
        column_center = np.zeros(M, dtype=float)
    else:
        X_centered, column_center = robust_center_columns(matrix, method=cfg.center_method)
    sigma_hat = estimate_noise_sigma(X_centered)
    lengths = length_grid(M, cfg.L_min, cfg.L_max)
    L_min = min(lengths)
    activation_energy_floor = quiet_window_energy_floor(X_centered, L_min)
    min_peak_distance = int(cfg.min_peak_distance or np.ceil(0.5 * L_min))
    n_min = int(cfg.n_min or max(5, np.ceil(0.01 * n)))
    K_max = int(cfg.K_max or min(25, np.ceil(M / L_min)))
    original_energy = float(np.sum(X_centered * X_centered))
    null_thresholds = _null_thresholds_by_length(
        X_centered,
        lengths=lengths,
        sigma_hat=sigma_hat,
        activation_energy_floor=activation_energy_floor,
        n_min=n_min,
        min_peak_distance=min_peak_distance,
        config=cfg,
    )

    R = X_centered.copy()
    selected: list[HLLRDEvent] = []
    for _iteration in range(K_max):
        peaks = find_residual_energy_peaks(
            R,
            kappa_peak=cfg.kappa_peak,
            min_peak_distance=min_peak_distance,
            smoothing_window=cfg.smoothing_window,
        )
        peak_start_indices = _peak_backtrack_starts(R, peaks, config=cfg)
        candidates: list[CandidateFit] = []
        for peak in peaks:
            accepted_index: int | None = None
            for length_index, length in enumerate(lengths):
                candidate = _score_interval(
                    R,
                    peak_index=int(peak),
                    peak_start_index=peak_start_indices.get(int(peak)),
                    length=int(length),
                    sigma_hat=sigma_hat,
                    null_thresholds=null_thresholds,
                    activation_energy_floor=activation_energy_floor,
                    n_min=n_min,
                    config=cfg,
                )
                if candidate.score > 0.0:
                    candidates.append(candidate)
                    accepted_index = length_index
                    break
            if accepted_index is not None and cfg.keep_next_longer and accepted_index + 1 < len(lengths):
                backup = _score_interval(
                    R,
                    peak_index=int(peak),
                    peak_start_index=peak_start_indices.get(int(peak)),
                    length=int(lengths[accepted_index + 1]),
                    sigma_hat=sigma_hat,
                    null_thresholds=null_thresholds,
                    activation_energy_floor=activation_energy_floor,
                    n_min=n_min,
                    config=cfg,
                )
                if backup.score > 0.0:
                    candidates.append(backup)

        candidates = [
            candidate
            for candidate in candidates
            if not any(
                is_duplicate_interval(
                    (candidate.start, candidate.end),
                    (event.start, event.end),
                    iou_threshold=cfg.duplicate_iou_threshold,
                )
                for event in selected
            )
        ]
        if not candidates:
            break
        best = max(candidates, key=lambda item: item.score)
        if best.score <= 0.0:
            break
        trimmed = _trim_candidate(
            best,
            R,
            sigma_hat=sigma_hat,
            null_thresholds=null_thresholds,
            activation_energy_floor=activation_energy_floor,
            n_min=n_min,
            L_min=L_min,
            config=cfg,
        )
        if trimmed.score <= 0.0:
            break

        committed = _simplify_candidate_for_commit(
            trimmed,
            R,
            sigma_hat=sigma_hat,
            activation_energy_floor=activation_energy_floor,
            n_min=n_min,
            config=cfg,
        )
        event = _event_from_candidate(committed)
        selected.append(event)
        R = R - committed.coefficients @ committed.basis.T
        if original_energy <= 0.0 or trimmed.active_gain / original_energy < cfg.epsilon_gain:
            break

    dictionary = build_dictionary(tuple(selected), M)
    if dictionary.size == 0:
        coefficients = np.zeros((n, 0), dtype=float)
        reconstruction = np.zeros_like(X_centered)
        residual = X_centered.copy()
        explained_fraction = 0.0
        refit_events: tuple[HLLRDEvent, ...] = ()
    else:
        coefficients = refit_coefficients(X_centered, dictionary, ridge=cfg.ridge)
        coefficients = apply_activation_threshold(
            coefficients,
            event_lengths=[event.length for event in selected],
            activation_energy_floor=activation_energy_floor,
            activation_scale=cfg.activation_scale,
        )
        reconstruction = coefficients @ dictionary.T
        residual = X_centered - reconstruction
        explained_fraction = _explained_fraction(X_centered, residual)
        refit_events = _events_with_refit_coefficients(tuple(selected), coefficients)

    return HLLRDFitResult(
        events=refit_events,
        dictionary=dictionary,
        coefficients=coefficients,
        reconstruction=reconstruction,
        residual=residual,
        explained_fraction=float(explained_fraction),
        sigma_hat=float(sigma_hat),
        activation_energy_floor=float(activation_energy_floor),
        column_center=column_center,
        config=cfg,
        metadata={
            **(metadata or {}),
            "null_thresholds_by_length": {str(length): float(null_thresholds[int(length)]) for length in lengths},
        },
    )


def transform_with_model(
    X: np.ndarray,
    model: HLLRDFitResult,
    *,
    already_centered: bool = False,
) -> HLLRDTransformResult:
    matrix = np.asarray(X, dtype=float)
    if already_centered:
        X_centered = matrix
    else:
        if model.column_center.shape[0] != matrix.shape[1]:
            raise ValueError("model column center length does not match X")
        X_centered = matrix - model.column_center
    if model.dictionary.size == 0:
        coefficients = np.zeros((matrix.shape[0], 0), dtype=float)
        reconstruction = np.zeros_like(X_centered)
    else:
        coefficients = refit_coefficients(X_centered, model.dictionary, ridge=model.config.ridge)
        coefficients = apply_activation_threshold(
            coefficients,
            event_lengths=[event.length for event in model.events],
            activation_energy_floor=model.activation_energy_floor,
            activation_scale=model.config.activation_scale,
        )
        reconstruction = coefficients @ model.dictionary.T
    residual = X_centered - reconstruction
    active_counts = np.zeros(matrix.shape[0], dtype=int)
    for event_index in range(len(model.events)):
        block = coefficients[:, 2 * event_index : 2 * event_index + 2]
        active_counts += np.linalg.norm(block, axis=1) > 0.0
    return HLLRDTransformResult(
        coefficients=coefficients,
        reconstruction=reconstruction,
        residual=residual,
        explained_fraction=_explained_fraction(X_centered, residual),
        active_counts=active_counts,
    )


def generate_candidate_summary(
    X: np.ndarray,
    config: HLLRDV1Config | None = None,
    *,
    already_centered: bool = True,
) -> list[dict[str, float | int]]:
    cfg = config or HLLRDV1Config()
    matrix = np.asarray(X, dtype=float)
    if not already_centered:
        matrix, _center = robust_center_columns(matrix, method=cfg.center_method)
    n, M = matrix.shape
    sigma_hat = estimate_noise_sigma(matrix)
    lengths = length_grid(M, cfg.L_min, cfg.L_max)
    L_min = min(lengths)
    activation_energy_floor = quiet_window_energy_floor(matrix, L_min)
    n_min = int(cfg.n_min or max(5, np.ceil(0.01 * n)))
    min_peak_distance = int(cfg.min_peak_distance or np.ceil(0.5 * L_min))
    null_thresholds = _null_thresholds_by_length(
        matrix,
        lengths=lengths,
        sigma_hat=sigma_hat,
        activation_energy_floor=activation_energy_floor,
        n_min=n_min,
        min_peak_distance=min_peak_distance,
        config=cfg,
    )
    peaks = find_residual_energy_peaks(
        matrix,
        kappa_peak=cfg.kappa_peak,
        min_peak_distance=min_peak_distance,
        smoothing_window=cfg.smoothing_window,
    )
    peak_start_indices = _peak_backtrack_starts(matrix, peaks, config=cfg)
    rows: list[dict[str, float | int]] = []
    for peak in peaks:
        for length in lengths:
            candidate = _score_interval(
                matrix,
                peak_index=int(peak),
                peak_start_index=peak_start_indices.get(int(peak)),
                length=int(length),
                sigma_hat=sigma_hat,
                null_thresholds=null_thresholds,
                activation_energy_floor=activation_energy_floor,
                n_min=n_min,
                config=cfg,
            )
            rows.append(
                {
                    "peak_index": candidate.peak_index,
                    "start": candidate.start,
                    "end": candidate.end,
                    "length": candidate.length,
                    "active_count": candidate.active_count,
                    "active_fraction": candidate.active_fraction,
                    "raw_gain": candidate.raw_gain,
                    "active_gain": candidate.active_gain,
                    "threshold": candidate.threshold,
                    "score": candidate.score,
                }
            )
    return rows


def build_dictionary(events: tuple[HLLRDEvent, ...], M: int) -> np.ndarray:
    if not events:
        return np.zeros((M, 0), dtype=float)
    return np.column_stack([event.basis[:, component] for event in events for component in range(2)])


def refit_coefficients(X: np.ndarray, dictionary: np.ndarray, *, ridge: float) -> np.ndarray:
    if dictionary.size == 0:
        return np.zeros((X.shape[0], 0), dtype=float)
    gram = dictionary.T @ dictionary
    regularized = gram + float(ridge) * np.eye(gram.shape[0])
    return np.linalg.solve(regularized, dictionary.T @ X.T).T


def quiet_window_energy_floor(X: np.ndarray, min_length: int) -> float:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2D matrix")
    if matrix.shape[1] == 0:
        return 0.0
    width = max(1, min(int(min_length), matrix.shape[1]))
    energy = np.mean(matrix * matrix, axis=0)
    if width == 1:
        return float(np.min(energy))
    kernel = np.ones(width, dtype=float) / float(width)
    rolling = np.convolve(energy, kernel, mode="valid")
    return float(np.min(rolling)) if rolling.size else 0.0


def activation_threshold_for_length(length: int, activation_energy_floor: float, activation_scale: float) -> float:
    return float(activation_scale) * np.sqrt(max(1, int(length)) * max(0.0, float(activation_energy_floor)))


def apply_activation_threshold(
    coefficients: np.ndarray,
    *,
    event_lengths: list[int],
    activation_energy_floor: float,
    activation_scale: float,
) -> np.ndarray:
    activated = coefficients.copy()
    for event_index, length in enumerate(event_lengths):
        block = activated[:, 2 * event_index : 2 * event_index + 2]
        threshold = activation_threshold_for_length(length, activation_energy_floor, activation_scale)
        inactive = np.linalg.norm(block, axis=1) <= threshold
        block[inactive, :] = 0.0
    return activated


def event_summary(result: HLLRDFitResult) -> list[dict[str, Any]]:
    total_energy = float(np.sum((result.reconstruction + result.residual) ** 2))
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(result.events):
        event_reconstruction = event.coefficients @ event.basis.T
        event_energy = float(np.sum(event_reconstruction * event_reconstruction))
        rows.append(
            {
                "event": index,
                "start": event.start,
                "end": event.end,
                "length": event.length,
                "peak_index": event.peak_index,
                "active_count": event.active_count,
                "active_fraction": event.active_fraction,
                "raw_gain": event.raw_gain,
                "active_gain": event.active_gain,
                "score": event.score,
                "explained_fraction": event_energy / total_energy if total_energy > 0.0 else 0.0,
                "local_simplifier_enabled": int(bool(event.simplifier.get("enabled", False))),
                "local_simplifier_active_count": int(event.simplifier.get("active_count", 0)),
                "local_simplifier_min_gain_per_point_m2": float(
                    event.simplifier.get("min_gain_per_point_m2", 0.0)
                ),
                "local_simplifier_mean_points": float(event.simplifier.get("mean_approximation_points", 0.0)),
                "local_simplifier_median_points": float(event.simplifier.get("median_approximation_points", 0.0)),
                "local_simplifier_max_points": int(event.simplifier.get("max_observed_approximation_points", 0)),
                "local_simplifier_initial_error_m2": float(event.simplifier.get("initial_error_m2", 0.0)),
                "local_simplifier_residual_error_m2": float(event.simplifier.get("residual_error_m2", 0.0)),
                "local_simplifier_reduced_error_m2": float(event.simplifier.get("reduced_error_m2", 0.0)),
                "local_simplifier_relative_reconstruction_loss": float(
                    event.simplifier.get("relative_reconstruction_loss", 0.0)
                ),
                "local_simplifier_point_count_histogram": event.simplifier.get("point_count_histogram", {}),
            }
        )
    return rows


def save_fit_result(path: Path, result: HLLRDFitResult, *, flight_ids: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_count = len(result.events)
    basis = np.stack([event.basis for event in result.events], axis=0) if event_count else np.zeros((0, result.residual.shape[1], 2))
    event_coefficients = (
        np.stack([event.coefficients for event in result.events], axis=0)
        if event_count
        else np.zeros((0, result.residual.shape[0], 2))
    )
    active_masks = (
        np.stack([event.active_mask for event in result.events], axis=0)
        if event_count
        else np.zeros((0, result.residual.shape[0]), dtype=bool)
    )
    intervals = np.asarray([[event.start, event.end, event.peak_index] for event in result.events], dtype=int)
    metrics = {
        "explained_fraction": result.explained_fraction,
        "sigma_hat": result.sigma_hat,
        "activation_energy_floor": result.activation_energy_floor,
        "events": event_summary(result),
        "metadata": result.metadata,
    }
    np.savez_compressed(
        path,
        dictionary=result.dictionary,
        coefficients=result.coefficients,
        reconstruction=result.reconstruction,
        residual=result.residual,
        column_center=result.column_center,
        basis=basis,
        event_coefficients=event_coefficients,
        active_masks=active_masks,
        intervals=intervals,
        flight_ids=np.asarray(flight_ids, dtype=str),
        config=np.asarray(json.dumps(asdict(result.config), sort_keys=True), dtype=str),
        metrics=np.asarray(json.dumps(metrics, sort_keys=True), dtype=str),
    )


def load_fit_result(path: Path) -> HLLRDFitResult:
    with np.load(path, allow_pickle=False) as data:
        config_payload = json.loads(str(np.asarray(data["config"]).item()))
        config_payload.pop("tau_z", None)
        config = HLLRDV1Config(**config_payload)
        metrics = json.loads(str(np.asarray(data["metrics"]).item()))
        basis_stack = np.asarray(data["basis"], dtype=float)
        event_coefficients = np.asarray(data["event_coefficients"], dtype=float)
        active_masks = np.asarray(data["active_masks"], dtype=bool)
        intervals = np.asarray(data["intervals"], dtype=int)
        events: list[HLLRDEvent] = []
        metric_rows = metrics.get("events", [])
        for index in range(basis_stack.shape[0]):
            metric = metric_rows[index] if index < len(metric_rows) else {}
            events.append(
                HLLRDEvent(
                    start=int(intervals[index, 0]),
                    end=int(intervals[index, 1]),
                    peak_index=int(intervals[index, 2]),
                    basis=basis_stack[index],
                    coefficients=event_coefficients[index],
                    active_mask=active_masks[index],
                    raw_gain=float(metric.get("raw_gain", 0.0)),
                    active_gain=float(metric.get("active_gain", 0.0)),
                    score=float(metric.get("score", 0.0)),
                    threshold=0.0,
                    simplifier=_simplifier_from_metric(metric),
                )
            )
        return HLLRDFitResult(
            events=tuple(events),
            dictionary=np.asarray(data["dictionary"], dtype=float),
            coefficients=np.asarray(data["coefficients"], dtype=float),
            reconstruction=np.asarray(data["reconstruction"], dtype=float),
            residual=np.asarray(data["residual"], dtype=float),
            explained_fraction=float(metrics.get("explained_fraction", 0.0)),
            sigma_hat=float(metrics.get("sigma_hat", 0.0)),
            activation_energy_floor=float(metrics.get("activation_energy_floor", 0.0)),
            column_center=np.asarray(data["column_center"], dtype=float),
            config=config,
            metadata=dict(metrics.get("metadata", {})),
        )


def _null_thresholds_by_length(
    X_centered: np.ndarray,
    *,
    lengths: tuple[int, ...],
    sigma_hat: float,
    activation_energy_floor: float,
    n_min: int,
    min_peak_distance: int,
    config: HLLRDV1Config,
) -> dict[int, float]:
    matrix = np.asarray(X_centered, dtype=float)
    n, M = matrix.shape
    thresholds = {
        int(length): analytic_null_threshold(int(length), n, sigma_hat, config.c_null)
        for length in lengths
    }
    repeats = max(0, int(config.empirical_null_repeats))
    if repeats == 0 or M < 2 or not lengths:
        return thresholds

    quantile = float(np.clip(config.empirical_null_quantile, 0.0, 1.0))
    station_energy = np.mean(matrix * matrix, axis=0)
    for length in lengths:
        length = int(length)
        width = max(1, min(length, M))
        if width == 1:
            rolling = station_energy.copy()
        else:
            kernel = np.ones(width, dtype=float) / float(width)
            rolling = np.convolve(station_energy, kernel, mode="valid")
        if rolling.size == 0:
            continue
        quiet_starts = np.argsort(rolling, kind="stable")[: min(repeats, rolling.size)]
        gains: list[float] = []
        for start in quiet_starts.tolist():
            end = int(start) + width
            peak = int(start) + width // 2
            activation_threshold = activation_threshold_for_length(
                width,
                activation_energy_floor,
                config.activation_scale,
            )
            candidate = local_rank2_candidate(
                matrix,
                peak_index=peak,
                start=int(start),
                end=end,
                activation_threshold=activation_threshold,
                threshold=0.0,
                n_min=n_min,
            )
            gains.append(float(candidate.active_gain))
        if gains:
            empirical = float(np.quantile(np.asarray(gains, dtype=float), quantile))
            thresholds[length] = max(thresholds[length], empirical)
    return thresholds


def _null_threshold_for_length(
    length: int,
    n: int,
    sigma_hat: float,
    config: HLLRDV1Config,
    null_thresholds: dict[int, float],
) -> float:
    analytic = analytic_null_threshold(length, n, sigma_hat, config.c_null)
    if not null_thresholds:
        return analytic
    if int(length) in null_thresholds:
        return max(analytic, float(null_thresholds[int(length)]))
    keys = sorted(int(key) for key in null_thresholds)
    if not keys:
        return analytic
    anchor = next((key for key in keys if key >= int(length)), keys[-1])
    scaled = float(null_thresholds[anchor]) * (n + int(length)) / max(1, n + anchor)
    return max(analytic, scaled)


def _score_interval(
    R: np.ndarray,
    *,
    peak_index: int,
    peak_start_index: int | None = None,
    length: int,
    sigma_hat: float,
    null_thresholds: dict[int, float],
    activation_energy_floor: float,
    n_min: int,
    config: HLLRDV1Config,
) -> CandidateFit:
    n, M = R.shape
    start, end = centered_interval(peak_index, length, M)
    if config.peak_backtrack_enabled and peak_start_index is not None:
        start = min(start, max(0, min(int(peak_start_index), int(peak_index))))
    return _score_explicit_interval(
        R,
        peak_index=peak_index,
        start=start,
        end=end,
        sigma_hat=sigma_hat,
        null_thresholds=null_thresholds,
        activation_energy_floor=activation_energy_floor,
        n_min=n_min,
        config=config,
    )


def _peak_backtrack_starts(
    R: np.ndarray,
    peaks: np.ndarray,
    *,
    config: HLLRDV1Config,
) -> dict[int, int]:
    if not config.peak_backtrack_enabled or len(peaks) == 0:
        return {}
    residual = np.asarray(R, dtype=float)
    energy = np.mean(residual * residual, axis=0)
    smoothed = smooth_energy(energy, config.smoothing_window)
    baseline = float(np.median(smoothed)) if smoothed.size else 0.0
    return {
        int(peak): backtrack_peak_rise_start(
            smoothed,
            int(peak),
            baseline=baseline,
            rise_fraction=config.peak_backtrack_rise_fraction,
        )
        for peak in peaks
    }


def _score_explicit_interval(
    R: np.ndarray,
    *,
    peak_index: int,
    start: int,
    end: int,
    sigma_hat: float,
    null_thresholds: dict[int, float],
    activation_energy_floor: float,
    n_min: int,
    config: HLLRDV1Config,
) -> CandidateFit:
    n, _M = R.shape
    threshold = _null_threshold_for_length(end - start, n, sigma_hat, config, null_thresholds)
    activation_threshold = activation_threshold_for_length(end - start, activation_energy_floor, config.activation_scale)
    return local_rank2_candidate(
        R,
        peak_index=peak_index,
        start=start,
        end=end,
        activation_threshold=activation_threshold,
        threshold=threshold,
        n_min=n_min,
        lambda_i=config.lambda_i,
        lambda_activation=config.lambda_activation,
    )


def _trim_candidate(
    candidate: CandidateFit,
    R: np.ndarray,
    *,
    sigma_hat: float,
    null_thresholds: dict[int, float],
    activation_energy_floor: float,
    n_min: int,
    L_min: int,
    config: HLLRDV1Config,
) -> CandidateFit:
    start = candidate.start
    end = candidate.end
    if end - start <= L_min:
        return candidate
    local_basis = candidate.basis[start:end, :]
    fitted = candidate.coefficients @ local_basis.T
    station_energy = np.sum(fitted * fitted, axis=0)
    if station_energy.size == 0:
        return candidate
    reference = float(np.median(station_energy))
    if reference <= 0.0:
        return candidate
    edge_threshold = config.endpoint_trim_threshold * reference
    left = 0
    right = station_energy.size
    while right - left > L_min and station_energy[left] < edge_threshold:
        left += 1
    while right - left > L_min and station_energy[right - 1] < edge_threshold:
        right -= 1
    if left == 0 and right == station_energy.size:
        return candidate
    peak = min(max(candidate.peak_index, start + left), start + right - 1)
    return _score_explicit_interval(
        R,
        peak_index=peak,
        start=start + left,
        end=start + right,
        sigma_hat=sigma_hat,
        null_thresholds=null_thresholds,
        activation_energy_floor=activation_energy_floor,
        n_min=n_min,
        config=config,
    )


def _simplify_candidate_for_commit(
    candidate: CandidateFit,
    R: np.ndarray,
    *,
    sigma_hat: float,
    activation_energy_floor: float,
    n_min: int,
    config: HLLRDV1Config,
) -> CandidateFit:
    if not config.local_simplifier_enabled or candidate.score <= 0.0:
        return candidate

    start = candidate.start
    end = candidate.end
    local_basis = candidate.basis[start:end, :]
    local_fit = candidate.coefficients @ local_basis.T
    min_gain = _local_simplifier_min_gain_per_point(config, sigma_hat)
    simplified = simplify_local_deviation_block(
        local_fit,
        active_mask=candidate.active_mask,
        min_gain_per_point_m2=min_gain,
        max_approximation_points=config.local_simplifier_max_points,
    )
    simplified_basis = _rank2_basis_from_local_block(simplified.values)
    activation_threshold = activation_threshold_for_length(end - start, activation_energy_floor, config.activation_scale)
    simplified_candidate = _candidate_from_local_basis(
        R,
        peak_index=candidate.peak_index,
        start=start,
        end=end,
        local_basis=simplified_basis,
        activation_threshold=activation_threshold,
        threshold=candidate.threshold,
        n_min=n_min,
        lambda_i=config.lambda_i,
        lambda_activation=config.lambda_activation,
        simplifier={
            "enabled": True,
            **simplified.diagnostics,
        },
    )
    if simplified_candidate.score <= 0.0:
        return candidate
    relative_loss = _relative_reconstruction_loss(candidate, simplified_candidate)
    if relative_loss > max(0.0, float(config.local_simplifier_max_relative_loss)):
        return candidate
    return replace(
        simplified_candidate,
        simplifier={
            **simplified_candidate.simplifier,
            "relative_reconstruction_loss": relative_loss,
        },
    )


def _event_from_candidate(candidate: CandidateFit) -> HLLRDEvent:
    return HLLRDEvent(
        start=candidate.start,
        end=candidate.end,
        basis=candidate.basis,
        coefficients=candidate.coefficients,
        active_mask=candidate.active_mask,
        raw_gain=candidate.raw_gain,
        active_gain=candidate.active_gain,
        score=candidate.score,
        threshold=candidate.threshold,
        peak_index=candidate.peak_index,
        simplifier=dict(candidate.simplifier),
    )


def _events_with_refit_coefficients(
    events: tuple[HLLRDEvent, ...],
    coefficients: np.ndarray,
) -> tuple[HLLRDEvent, ...]:
    refit: list[HLLRDEvent] = []
    for index, event in enumerate(events):
        block = coefficients[:, 2 * index : 2 * index + 2]
        active_mask = np.linalg.norm(block, axis=1) > 0.0
        active_gain = float(np.sum(block * block))
        refit.append(
            HLLRDEvent(
                start=event.start,
                end=event.end,
                basis=event.basis,
                coefficients=block,
                active_mask=active_mask,
                raw_gain=event.raw_gain,
                active_gain=active_gain,
                score=event.score,
                threshold=event.threshold,
                peak_index=event.peak_index,
                simplifier=dict(event.simplifier),
            )
        )
    return tuple(refit)


def _local_simplifier_min_gain_per_point(config: HLLRDV1Config, sigma_hat: float) -> float:
    gain_sigma = max(0.0, float(config.local_simplifier_gain_sigma))
    sigma = max(0.0, float(sigma_hat))
    return float((gain_sigma * sigma) ** 2)


def _rank2_basis_from_local_block(local: np.ndarray) -> np.ndarray:
    block = np.asarray(local, dtype=float)
    if block.ndim != 2:
        raise ValueError("local must be a 2D matrix")
    basis = np.zeros((block.shape[1], 2), dtype=float)
    if block.size == 0:
        return basis
    _U, _singular_values, Wt = np.linalg.svd(block, full_matrices=False)
    rank = min(2, Wt.shape[0])
    if rank:
        basis[:, :rank] = Wt[:rank, :].T
    return basis


def _relative_reconstruction_loss(raw: CandidateFit, committed: CandidateFit) -> float:
    raw_fit = raw.coefficients @ raw.basis.T
    committed_fit = committed.coefficients @ committed.basis.T
    active_mask = raw.active_mask | committed.active_mask
    if np.any(active_mask):
        raw_fit = raw_fit[active_mask]
        committed_fit = committed_fit[active_mask]
    denominator = float(np.sum(raw_fit * raw_fit))
    if denominator <= 0.0:
        return 0.0
    delta = raw_fit - committed_fit
    return float(np.sum(delta * delta) / denominator)


def _candidate_from_local_basis(
    R: np.ndarray,
    *,
    peak_index: int,
    start: int,
    end: int,
    local_basis: np.ndarray,
    activation_threshold: float,
    threshold: float,
    n_min: int,
    lambda_i: float,
    lambda_activation: float,
    simplifier: dict[str, Any],
) -> CandidateFit:
    residual = np.asarray(R, dtype=float)
    basis_local = np.asarray(local_basis, dtype=float)
    if basis_local.shape != (end - start, 2):
        raise ValueError("local_basis must have shape interval_length x 2")
    basis = np.zeros((residual.shape[1], 2), dtype=float)
    basis[start:end, :] = basis_local
    coefficients = residual[:, start:end] @ basis_local
    raw_gain = float(np.sum(coefficients * coefficients))
    norms = np.linalg.norm(coefficients, axis=1)
    active_mask = norms > float(activation_threshold)
    if int(np.count_nonzero(active_mask)) < int(n_min):
        active_mask = np.zeros(residual.shape[0], dtype=bool)
    active_coefficients = coefficients.copy()
    active_coefficients[~active_mask, :] = 0.0
    active_gain = float(np.sum(active_coefficients * active_coefficients))
    score = active_gain - float(threshold) - float(lambda_i) * (end - start) - float(lambda_activation) * int(
        np.count_nonzero(active_mask)
    )
    return CandidateFit(
        peak_index=int(peak_index),
        start=int(start),
        end=int(end),
        basis=basis,
        coefficients=active_coefficients,
        active_mask=active_mask,
        raw_gain=raw_gain,
        active_gain=active_gain,
        score=float(score),
        threshold=float(threshold),
        simplifier=simplifier,
    )


def _simplifier_from_metric(metric: dict[str, Any]) -> dict[str, Any]:
    if not metric.get("local_simplifier_enabled", False):
        return {}
    return {
        "enabled": True,
        "active_count": int(metric.get("local_simplifier_active_count", 0)),
        "min_gain_per_point_m2": float(metric.get("local_simplifier_min_gain_per_point_m2", 0.0)),
        "mean_approximation_points": float(metric.get("local_simplifier_mean_points", 0.0)),
        "median_approximation_points": float(metric.get("local_simplifier_median_points", 0.0)),
        "max_observed_approximation_points": int(metric.get("local_simplifier_max_points", 0)),
        "initial_error_m2": float(metric.get("local_simplifier_initial_error_m2", 0.0)),
        "residual_error_m2": float(metric.get("local_simplifier_residual_error_m2", 0.0)),
        "reduced_error_m2": float(metric.get("local_simplifier_reduced_error_m2", 0.0)),
        "relative_reconstruction_loss": float(metric.get("local_simplifier_relative_reconstruction_loss", 0.0)),
        "point_count_histogram": dict(metric.get("local_simplifier_point_count_histogram", {}) or {}),
    }


def _explained_fraction(X: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.sum(X * X))
    if denominator <= 0.0:
        return 0.0
    return float(1.0 - np.sum(residual * residual) / denominator)
