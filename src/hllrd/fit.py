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
    ridge: float = 1e-6
    endpoint_trim_threshold: float = 0.05
    duplicate_iou_threshold: float = 0.8
    keep_next_longer: bool = False
    center_method: str = "median"
    peak_backtrack_enabled: bool = True
    peak_backtrack_rise_fraction: float = 0.05
    local_simplifier_enabled: bool = True
    local_simplifier_gain_sigma: float = 128.0
    local_simplifier_max_points: int = 4


@dataclass(frozen=True)
class HLLRDV2Config(HLLRDV1Config):
    lag_enabled: bool = True
    max_lag_stations: int = 8
    lag_direction: str = "both"
    lag_penalty: float = 0.0
    max_extend_stations: int = 0
    extend_direction: str = "nonnegative"
    extend_penalty: float = 0.0
    registration_iterations: int = 5
    registration_tolerance: int = 0


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
    lag_offsets: np.ndarray | None = None
    extension_offsets: np.ndarray | None = None

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
    trace_tangent_center: np.ndarray | None = None
    trace_tangent_reconstruction: np.ndarray | None = None
    trace_tangent_coefficients: np.ndarray | None = None


@dataclass(frozen=True)
class HLLRDTransformResult:
    coefficients: np.ndarray
    reconstruction: np.ndarray
    residual: np.ndarray
    explained_fraction: float
    active_counts: np.ndarray
    lag_offsets: np.ndarray | None = None
    extension_offsets: np.ndarray | None = None


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
        R = R - _candidate_reconstruction(trimmed)
        if original_energy <= 0.0 or trimmed.active_gain / original_energy < cfg.epsilon_gain:
            break

    dictionary = build_dictionary(tuple(selected), M)
    if dictionary.size == 0:
        coefficients = np.zeros((n, 0), dtype=float)
        reconstruction = np.zeros_like(X_centered)
        residual = X_centered.copy()
        explained_fraction = 0.0
        refit_events: tuple[HLLRDEvent, ...] = ()
    elif _has_registered_events(tuple(selected)):
        coefficients, reconstruction = refit_registered_coefficients(X_centered, tuple(selected), ridge=cfg.ridge)
        coefficients = apply_activation_threshold(
            coefficients,
            event_lengths=[event.length for event in selected],
            activation_energy_floor=activation_energy_floor,
            activation_scale=cfg.activation_scale,
        )
        reconstruction = reconstruct_registered_events(tuple(selected), coefficients)
        residual = X_centered - reconstruction
        explained_fraction = _explained_fraction(X_centered, residual)
        refit_events = _events_with_refit_coefficients(tuple(selected), coefficients)
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
        metadata=metadata or {},
    )


def fit_lag_registered_low_rank(
    X: np.ndarray,
    config: HLLRDV2Config | None = None,
    *,
    already_centered: bool = False,
    metadata: dict[str, Any] | None = None,
) -> HLLRDFitResult:
    return fit_localized_low_rank(
        X,
        config or HLLRDV2Config(),
        already_centered=already_centered,
        metadata=metadata,
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
        lag_offsets: np.ndarray | None = None
        extension_offsets: np.ndarray | None = None
    elif _model_uses_registration(model):
        lag_offsets, extension_offsets = infer_model_registrations(X_centered, model)
        coefficients, reconstruction = refit_registered_coefficients(
            X_centered,
            model.events,
            ridge=model.config.ridge,
            lag_offsets=lag_offsets,
            extension_offsets=extension_offsets,
        )
        coefficients = apply_activation_threshold(
            coefficients,
            event_lengths=[event.length for event in model.events],
            activation_energy_floor=model.activation_energy_floor,
            activation_scale=model.config.activation_scale,
        )
        reconstruction = reconstruct_registered_events(
            model.events,
            coefficients,
            lag_offsets=lag_offsets,
            extension_offsets=extension_offsets,
        )
    else:
        coefficients = refit_coefficients(X_centered, model.dictionary, ridge=model.config.ridge)
        coefficients = apply_activation_threshold(
            coefficients,
            event_lengths=[event.length for event in model.events],
            activation_energy_floor=model.activation_energy_floor,
            activation_scale=model.config.activation_scale,
        )
        reconstruction = coefficients @ model.dictionary.T
        lag_offsets = None
        extension_offsets = None
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
        lag_offsets=lag_offsets,
        extension_offsets=extension_offsets,
    )


def augment_with_trace_tangent_lift(
    model: HLLRDFitResult,
    tangent_residual: np.ndarray,
    *,
    center_method: str = "median",
    ridge: float | None = None,
    extra_residual_config: HLLRDV1Config | None = None,
) -> HLLRDFitResult:
    tangent = np.asarray(tangent_residual, dtype=float)
    if tangent.shape != model.reconstruction.shape:
        raise ValueError("tangent_residual shape must match model reconstruction")
    if center_method == "median":
        tangent_center = np.median(tangent, axis=0)
    elif center_method == "mean":
        tangent_center = np.mean(tangent, axis=0)
    elif center_method == "none":
        tangent_center = np.zeros(tangent.shape[1], dtype=float)
    else:
        raise ValueError("center_method must be 'median', 'mean', or 'none'")
    tangent_centered = tangent - tangent_center[None, :]
    fit_ridge = model.config.ridge if ridge is None else float(ridge)
    tangent_coefficients, tangent_reconstruction_centered = refit_registered_coefficients(
        tangent_centered,
        model.events,
        ridge=fit_ridge,
    )
    extra_metadata: dict[str, Any] = {"enabled": False}
    if extra_residual_config is not None:
        residual = tangent_centered - tangent_reconstruction_centered
        extra_fit = fit_localized_low_rank(
            residual,
            extra_residual_config,
            already_centered=True,
        )
        tangent_reconstruction_centered = tangent_reconstruction_centered + extra_fit.reconstruction
        extra_metadata = {
            "enabled": True,
            "events": len(extra_fit.events),
            "explained_residual_fraction": float(extra_fit.explained_fraction),
            "config": asdict(extra_residual_config),
        }
    tangent_reconstruction = tangent_reconstruction_centered + tangent_center[None, :]
    tangent_explained_fraction = _explained_fraction(tangent, tangent - tangent_reconstruction)
    metadata = {
        **model.metadata,
        "trace_tangent_lift": {
            "enabled": True,
            "center_method": center_method,
            "ridge": fit_ridge,
            "explained_fraction": float(tangent_explained_fraction),
            "extra_residual_events": extra_metadata,
        },
    }
    return replace(
        model,
        metadata=metadata,
        trace_tangent_center=tangent_center,
        trace_tangent_reconstruction=tangent_reconstruction,
        trace_tangent_coefficients=tangent_coefficients,
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
                activation_energy_floor=activation_energy_floor,
                n_min=n_min,
                config=cfg,
            )
            active_lags = _candidate_active_lags(candidate)
            active_extensions = _candidate_active_extensions(candidate)
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
                    "lag_registered": int(candidate.lag_offsets is not None),
                    "lag_min": int(np.min(active_lags)) if active_lags.size else 0,
                    "lag_max": int(np.max(active_lags)) if active_lags.size else 0,
                    "lag_abs_mean": float(np.mean(np.abs(active_lags))) if active_lags.size else 0.0,
                    "lag_histogram": _lag_histogram(active_lags),
                    "extension_registered": int(candidate.extension_offsets is not None),
                    "extension_min": int(np.min(active_extensions)) if active_extensions.size else 0,
                    "extension_max": int(np.max(active_extensions)) if active_extensions.size else 0,
                    "extension_abs_mean": float(np.mean(np.abs(active_extensions))) if active_extensions.size else 0.0,
                    "extension_histogram": _lag_histogram(active_extensions),
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
        event_reconstruction = _event_reconstruction(event)
        event_energy = float(np.sum(event_reconstruction * event_reconstruction))
        active_lags = _active_lag_offsets(event)
        active_extensions = _active_extension_offsets(event)
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
                "local_simplifier_point_count_histogram": event.simplifier.get("point_count_histogram", {}),
                "lag_registered": int(_event_uses_lag_registration(event)),
                "lag_min": int(np.min(active_lags)) if active_lags.size else 0,
                "lag_max": int(np.max(active_lags)) if active_lags.size else 0,
                "lag_mean": float(np.mean(active_lags)) if active_lags.size else 0.0,
                "lag_median": float(np.median(active_lags)) if active_lags.size else 0.0,
                "lag_abs_mean": float(np.mean(np.abs(active_lags))) if active_lags.size else 0.0,
                "lag_histogram": _lag_histogram(active_lags),
                "extension_registered": int(_event_uses_extension_registration(event)),
                "extension_min": int(np.min(active_extensions)) if active_extensions.size else 0,
                "extension_max": int(np.max(active_extensions)) if active_extensions.size else 0,
                "extension_mean": float(np.mean(active_extensions)) if active_extensions.size else 0.0,
                "extension_median": float(np.median(active_extensions)) if active_extensions.size else 0.0,
                "extension_abs_mean": float(np.mean(np.abs(active_extensions))) if active_extensions.size else 0.0,
                "extension_histogram": _lag_histogram(active_extensions),
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
    lag_offsets = (
        np.stack([_event_lag_offsets(event, result.residual.shape[0]) for event in result.events], axis=0)
        if event_count
        else np.zeros((0, result.residual.shape[0]), dtype=int)
    )
    extension_offsets = (
        np.stack([_event_extension_offsets(event, result.residual.shape[0]) for event in result.events], axis=0)
        if event_count
        else np.zeros((0, result.residual.shape[0]), dtype=int)
    )
    intervals = np.asarray([[event.start, event.end, event.peak_index] for event in result.events], dtype=int)
    metrics = {
        "explained_fraction": result.explained_fraction,
        "sigma_hat": result.sigma_hat,
        "activation_energy_floor": result.activation_energy_floor,
        "events": event_summary(result),
        "metadata": result.metadata,
        "trace_tangent_lift_enabled": result.trace_tangent_reconstruction is not None,
    }
    trace_tangent_center = (
        np.asarray(result.trace_tangent_center, dtype=float)
        if result.trace_tangent_center is not None
        else np.zeros(result.residual.shape[1], dtype=float)
    )
    trace_tangent_reconstruction = (
        np.asarray(result.trace_tangent_reconstruction, dtype=float)
        if result.trace_tangent_reconstruction is not None
        else np.zeros_like(result.reconstruction)
    )
    trace_tangent_coefficients = (
        np.asarray(result.trace_tangent_coefficients, dtype=float)
        if result.trace_tangent_coefficients is not None
        else np.zeros((result.residual.shape[0], 2 * event_count), dtype=float)
    )
    np.savez_compressed(
        path,
        dictionary=result.dictionary,
        coefficients=result.coefficients,
        reconstruction=result.reconstruction,
        residual=result.residual,
        column_center=result.column_center,
        trace_tangent_center=trace_tangent_center,
        trace_tangent_reconstruction=trace_tangent_reconstruction,
        trace_tangent_coefficients=trace_tangent_coefficients,
        basis=basis,
        event_coefficients=event_coefficients,
        active_masks=active_masks,
        lag_offsets=lag_offsets,
        extension_offsets=extension_offsets,
        intervals=intervals,
        flight_ids=np.asarray(flight_ids, dtype=str),
        config=np.asarray(json.dumps(asdict(result.config), sort_keys=True), dtype=str),
        metrics=np.asarray(json.dumps(metrics, sort_keys=True), dtype=str),
    )


def load_fit_result(path: Path) -> HLLRDFitResult:
    with np.load(path, allow_pickle=False) as data:
        config_payload = json.loads(str(np.asarray(data["config"]).item()))
        config_payload.pop("tau_z", None)
        config = _config_from_payload(config_payload)
        metrics = json.loads(str(np.asarray(data["metrics"]).item()))
        basis_stack = np.asarray(data["basis"], dtype=float)
        event_coefficients = np.asarray(data["event_coefficients"], dtype=float)
        active_masks = np.asarray(data["active_masks"], dtype=bool)
        lag_offsets = (
            np.asarray(data["lag_offsets"], dtype=int)
            if "lag_offsets" in data.files
            else np.zeros((basis_stack.shape[0], event_coefficients.shape[1]), dtype=int)
        )
        extension_offsets = (
            np.asarray(data["extension_offsets"], dtype=int)
            if "extension_offsets" in data.files
            else np.zeros((basis_stack.shape[0], event_coefficients.shape[1]), dtype=int)
        )
        intervals = np.asarray(data["intervals"], dtype=int)
        events: list[HLLRDEvent] = []
        metric_rows = metrics.get("events", [])
        for index in range(basis_stack.shape[0]):
            metric = metric_rows[index] if index < len(metric_rows) else {}
            event_lags = lag_offsets[index] if index < lag_offsets.shape[0] else None
            event_extensions = extension_offsets[index] if index < extension_offsets.shape[0] else None
            lag_registered = bool(metric.get("lag_registered", False))
            extension_registered = bool(metric.get("extension_registered", False))
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
                    lag_offsets=np.asarray(event_lags, dtype=int).copy() if lag_registered and event_lags is not None else None,
                    extension_offsets=(
                        np.asarray(event_extensions, dtype=int).copy()
                        if extension_registered and event_extensions is not None
                        else None
                    ),
                )
            )
        trace_tangent_enabled = bool(metrics.get("trace_tangent_lift_enabled", False))
        trace_tangent_center = (
            np.asarray(data["trace_tangent_center"], dtype=float)
            if trace_tangent_enabled and "trace_tangent_center" in data.files
            else None
        )
        trace_tangent_reconstruction = (
            np.asarray(data["trace_tangent_reconstruction"], dtype=float)
            if trace_tangent_enabled and "trace_tangent_reconstruction" in data.files
            else None
        )
        trace_tangent_coefficients = (
            np.asarray(data["trace_tangent_coefficients"], dtype=float)
            if trace_tangent_enabled and "trace_tangent_coefficients" in data.files
            else None
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
            trace_tangent_center=trace_tangent_center,
            trace_tangent_reconstruction=trace_tangent_reconstruction,
            trace_tangent_coefficients=trace_tangent_coefficients,
        )


def local_rank2_lag_registered_candidate(
    R: np.ndarray,
    *,
    peak_index: int,
    start: int,
    end: int,
    activation_threshold: float,
    threshold: float,
    n_min: int = 5,
    lambda_i: float = 0.0,
    lambda_activation: float = 0.0,
    lag_penalty: float = 0.0,
    max_lag_stations: int = 8,
    lag_direction: str = "both",
    max_extend_stations: int = 0,
    extend_direction: str = "nonnegative",
    extend_penalty: float = 0.0,
    registration_iterations: int = 5,
    registration_tolerance: int = 0,
) -> CandidateFit:
    residual = np.asarray(R, dtype=float)
    n, M = residual.shape
    if not (0 <= start < end <= M):
        raise ValueError("candidate interval must satisfy 0 <= start < end <= M")
    local = residual[:, start:end]
    basis_local = _rank2_basis_from_local_block(local)
    lag_values = _candidate_lag_values(
        max_lag_stations=max_lag_stations,
        lag_direction=lag_direction,
        start=start,
        end=end,
        M=M,
    )
    extension_values = _candidate_extension_values(
        max_extend_stations=max_extend_stations,
        extend_direction=extend_direction,
        start=start,
        end=end,
        M=M,
    )
    lag_offsets = np.zeros(n, dtype=int)
    extension_offsets = np.zeros(n, dtype=int)
    coefficients = np.zeros((n, 2), dtype=float)
    active_mask = np.ones(n, dtype=bool)
    iterations = max(1, int(registration_iterations))
    tolerance = max(0, int(registration_tolerance))

    for _iteration in range(iterations):
        previous_lags = lag_offsets.copy()
        previous_extensions = extension_offsets.copy()
        coefficients, lag_offsets, extension_offsets, _row_scores = _best_registrations_for_basis(
            residual,
            start=start,
            end=end,
            peak_index=peak_index,
            local_basis=basis_local,
            lag_values=lag_values,
            extension_values=extension_values,
            lag_penalty=lag_penalty,
            extension_penalty=extend_penalty,
        )
        norms = np.linalg.norm(coefficients, axis=1)
        active_mask = norms > float(activation_threshold)
        if int(np.count_nonzero(active_mask)) < int(n_min):
            active_mask = np.zeros(n, dtype=bool)
            break
        registered = _registered_rows_for_registrations(residual, start, end, lag_offsets, extension_offsets)
        basis_local = _rank2_basis_from_local_block(registered[active_mask])
        max_lag_change = int(np.max(np.abs(lag_offsets - previous_lags), initial=0))
        max_extension_change = int(np.max(np.abs(extension_offsets - previous_extensions), initial=0))
        if max(max_lag_change, max_extension_change) <= tolerance:
            break

    coefficients, lag_offsets, extension_offsets, _row_scores = _best_registrations_for_basis(
        residual,
        start=start,
        end=end,
        peak_index=peak_index,
        local_basis=basis_local,
        lag_values=lag_values,
        extension_values=extension_values,
        lag_penalty=lag_penalty,
        extension_penalty=extend_penalty,
    )
    norms = np.linalg.norm(coefficients, axis=1)
    active_mask = norms > float(activation_threshold)
    if int(np.count_nonzero(active_mask)) < int(n_min):
        active_mask = np.zeros(n, dtype=bool)
    active_coefficients = coefficients.copy()
    active_coefficients[~active_mask, :] = 0.0
    raw_gain = float(np.sum(coefficients * coefficients))
    active_gain = float(np.sum(active_coefficients * active_coefficients))
    lag_cost = float(lag_penalty) * float(np.sum(np.abs(lag_offsets[active_mask])))
    extension_cost = float(extend_penalty) * float(np.sum(np.abs(extension_offsets[active_mask])))
    score = active_gain - float(threshold) - float(lambda_i) * (end - start) - float(lambda_activation) * int(
        np.count_nonzero(active_mask)
    ) - lag_cost - extension_cost
    basis = np.zeros((M, 2), dtype=float)
    basis[start:end, :] = basis_local
    simplifier = {
        "lag_registered": True,
        "lag_histogram": _lag_histogram(lag_offsets[active_mask]),
        "lag_mean": float(np.mean(lag_offsets[active_mask])) if np.any(active_mask) else 0.0,
        "lag_abs_mean": float(np.mean(np.abs(lag_offsets[active_mask]))) if np.any(active_mask) else 0.0,
        "extension_registered": int(max(0, int(max_extend_stations)) > 0),
        "extension_histogram": _lag_histogram(extension_offsets[active_mask]),
        "extension_mean": float(np.mean(extension_offsets[active_mask])) if np.any(active_mask) else 0.0,
        "extension_abs_mean": float(np.mean(np.abs(extension_offsets[active_mask]))) if np.any(active_mask) else 0.0,
    }
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
        lag_offsets=lag_offsets,
        extension_offsets=extension_offsets if int(max_extend_stations) > 0 else None,
    )


def refit_registered_coefficients(
    X: np.ndarray,
    events: tuple[HLLRDEvent, ...],
    *,
    ridge: float,
    lag_offsets: np.ndarray | None = None,
    extension_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(X, dtype=float)
    n, M = matrix.shape
    if not events:
        return np.zeros((n, 0), dtype=float), np.zeros_like(matrix)
    coefficients = np.zeros((n, 2 * len(events)), dtype=float)
    reconstruction = np.zeros_like(matrix)
    for row_index in range(n):
        dictionary = _row_dictionary(
            events,
            row_index,
            M,
            lag_offsets=lag_offsets,
            extension_offsets=extension_offsets,
        )
        if dictionary.size == 0:
            continue
        gram = dictionary.T @ dictionary
        regularized = gram + float(ridge) * np.eye(gram.shape[0])
        row_coefficients = np.linalg.solve(regularized, dictionary.T @ matrix[row_index])
        coefficients[row_index, :] = row_coefficients
        reconstruction[row_index, :] = row_coefficients @ dictionary.T
    return coefficients, reconstruction


def reconstruct_registered_events(
    events: tuple[HLLRDEvent, ...],
    coefficients: np.ndarray,
    *,
    lag_offsets: np.ndarray | None = None,
    extension_offsets: np.ndarray | None = None,
) -> np.ndarray:
    if not events:
        return np.zeros((coefficients.shape[0], 0), dtype=float)
    n = coefficients.shape[0]
    M = events[0].basis.shape[0]
    reconstruction = np.zeros((n, M), dtype=float)
    for row_index in range(n):
        dictionary = _row_dictionary(
            events,
            row_index,
            M,
            lag_offsets=lag_offsets,
            extension_offsets=extension_offsets,
        )
        if dictionary.size:
            reconstruction[row_index, :] = coefficients[row_index, :] @ dictionary.T
    return reconstruction


def infer_model_registrations(X: np.ndarray, model: HLLRDFitResult) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(X, dtype=float)
    lag_table = np.zeros((matrix.shape[0], len(model.events)), dtype=int)
    extension_table = np.zeros((matrix.shape[0], len(model.events)), dtype=int)
    for event_index, event in enumerate(model.events):
        if not _event_uses_registration(event):
            continue
        lag_values = _candidate_lag_values(
            max_lag_stations=_config_max_lag(model.config, event),
            lag_direction=_config_lag_direction(model.config),
            start=event.start,
            end=event.end,
            M=matrix.shape[1],
        )
        extension_values = _candidate_extension_values(
            max_extend_stations=_config_max_extend(model.config, event),
            extend_direction=_config_extend_direction(model.config),
            start=event.start,
            end=event.end,
            M=matrix.shape[1],
        )
        local_basis = event.basis[event.start : event.end, :]
        _coefficients, event_lags, event_extensions, _scores = _best_registrations_for_basis(
            matrix,
            start=event.start,
            end=event.end,
            peak_index=event.peak_index,
            local_basis=local_basis,
            lag_values=lag_values,
            extension_values=extension_values,
            lag_penalty=_config_lag_penalty(model.config),
            extension_penalty=_config_extend_penalty(model.config),
        )
        lag_table[:, event_index] = event_lags
        extension_table[:, event_index] = event_extensions
    return lag_table, extension_table


def refit_lagged_coefficients(
    X: np.ndarray,
    events: tuple[HLLRDEvent, ...],
    *,
    ridge: float,
    lag_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return refit_registered_coefficients(X, events, ridge=ridge, lag_offsets=lag_offsets)


def reconstruct_lagged_events(
    events: tuple[HLLRDEvent, ...],
    coefficients: np.ndarray,
    *,
    lag_offsets: np.ndarray | None = None,
) -> np.ndarray:
    return reconstruct_registered_events(events, coefficients, lag_offsets=lag_offsets)


def infer_model_lags(X: np.ndarray, model: HLLRDFitResult) -> np.ndarray:
    lag_offsets, _extension_offsets = infer_model_registrations(X, model)
    return lag_offsets


def _lag_registration_enabled(config: HLLRDV1Config) -> bool:
    return (
        isinstance(config, HLLRDV2Config)
        and bool(config.lag_enabled)
        and (int(config.max_lag_stations) > 0 or int(config.max_extend_stations) > 0)
    )


def _config_from_payload(payload: dict[str, Any]) -> HLLRDV1Config:
    v2_keys = {
        "lag_enabled",
        "max_lag_stations",
        "lag_direction",
        "lag_penalty",
        "max_extend_stations",
        "extend_direction",
        "extend_penalty",
        "registration_iterations",
        "registration_tolerance",
    }
    if any(key in payload for key in v2_keys):
        return HLLRDV2Config(**payload)
    return HLLRDV1Config(**payload)


def _model_uses_lag_registration(model: HLLRDFitResult) -> bool:
    return any(_event_uses_lag_registration(event) for event in model.events)


def _model_uses_registration(model: HLLRDFitResult) -> bool:
    return any(_event_uses_registration(event) for event in model.events)


def _event_uses_lag_registration(event: HLLRDEvent) -> bool:
    return event.lag_offsets is not None or bool(event.simplifier.get("lag_registered", False))


def _event_uses_extension_registration(event: HLLRDEvent) -> bool:
    return event.extension_offsets is not None or bool(event.simplifier.get("extension_registered", False))


def _event_uses_registration(event: HLLRDEvent) -> bool:
    return _event_uses_lag_registration(event) or _event_uses_extension_registration(event)


def _event_lag_offsets(event: HLLRDEvent, n: int) -> np.ndarray:
    if event.lag_offsets is None:
        return np.zeros(n, dtype=int)
    values = np.asarray(event.lag_offsets, dtype=int)
    if values.shape[0] == n:
        return values.copy()
    padded = np.zeros(n, dtype=int)
    limit = min(n, values.shape[0])
    padded[:limit] = values[:limit]
    return padded


def _event_extension_offsets(event: HLLRDEvent, n: int) -> np.ndarray:
    if event.extension_offsets is None:
        return np.zeros(n, dtype=int)
    values = np.asarray(event.extension_offsets, dtype=int)
    if values.shape[0] == n:
        return values.copy()
    padded = np.zeros(n, dtype=int)
    limit = min(n, values.shape[0])
    padded[:limit] = values[:limit]
    return padded


def _active_lag_offsets(event: HLLRDEvent) -> np.ndarray:
    if event.lag_offsets is None:
        return np.zeros(0, dtype=int)
    values = np.asarray(event.lag_offsets, dtype=int)
    active = np.asarray(event.active_mask, dtype=bool)
    if active.shape[0] != values.shape[0]:
        return values
    return values[active]


def _active_extension_offsets(event: HLLRDEvent) -> np.ndarray:
    if event.extension_offsets is None:
        return np.zeros(0, dtype=int)
    values = np.asarray(event.extension_offsets, dtype=int)
    active = np.asarray(event.active_mask, dtype=bool)
    if active.shape[0] != values.shape[0]:
        return values
    return values[active]


def _candidate_active_lags(candidate: CandidateFit) -> np.ndarray:
    if candidate.lag_offsets is None:
        return np.zeros(0, dtype=int)
    values = np.asarray(candidate.lag_offsets, dtype=int)
    active = np.asarray(candidate.active_mask, dtype=bool)
    if active.shape[0] != values.shape[0]:
        return values
    return values[active]


def _candidate_active_extensions(candidate: CandidateFit) -> np.ndarray:
    if candidate.extension_offsets is None:
        return np.zeros(0, dtype=int)
    values = np.asarray(candidate.extension_offsets, dtype=int)
    active = np.asarray(candidate.active_mask, dtype=bool)
    if active.shape[0] != values.shape[0]:
        return values
    return values[active]


def _has_lagged_events(events: tuple[HLLRDEvent, ...]) -> bool:
    return any(_event_uses_lag_registration(event) for event in events)


def _has_registered_events(events: tuple[HLLRDEvent, ...]) -> bool:
    return any(_event_uses_registration(event) for event in events)


def _candidate_lag_values(
    *,
    max_lag_stations: int,
    lag_direction: str,
    start: int,
    end: int,
    M: int,
) -> np.ndarray:
    max_lag = max(0, int(max_lag_stations))
    if lag_direction == "both":
        values = np.arange(-max_lag, max_lag + 1, dtype=int)
    elif lag_direction == "nonnegative":
        values = np.arange(0, max_lag + 1, dtype=int)
    elif lag_direction == "nonpositive":
        values = np.arange(-max_lag, 1, dtype=int)
    else:
        raise ValueError("lag_direction must be 'both', 'nonnegative', or 'nonpositive'")
    valid = values[(start + values >= 0) & (end + values <= M)]
    if valid.size == 0:
        return np.asarray([0], dtype=int)
    if 0 not in set(int(value) for value in valid):
        valid = np.sort(np.append(valid, 0))
    return valid


def _candidate_extension_values(
    *,
    max_extend_stations: int,
    extend_direction: str,
    start: int,
    end: int,
    M: int,
) -> np.ndarray:
    max_extend = max(0, int(max_extend_stations))
    if max_extend == 0:
        return np.asarray([0], dtype=int)
    if extend_direction == "both":
        values = np.arange(-max_extend, max_extend + 1, dtype=int)
    elif extend_direction == "nonnegative":
        values = np.arange(0, max_extend + 1, dtype=int)
    elif extend_direction == "nonpositive":
        values = np.arange(-max_extend, 1, dtype=int)
    else:
        raise ValueError("extend_direction must be 'both', 'nonnegative', or 'nonpositive'")
    base_length = end - start
    valid = values[(base_length + values >= 2) & (start + base_length + values <= M)]
    if valid.size == 0:
        return np.asarray([0], dtype=int)
    if 0 not in set(int(value) for value in valid):
        valid = np.sort(np.append(valid, 0))
    return valid


def _best_lags_for_basis(
    R: np.ndarray,
    *,
    start: int,
    end: int,
    local_basis: np.ndarray,
    lag_values: np.ndarray,
    lag_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(R, dtype=float)
    basis = np.asarray(local_basis, dtype=float)
    n = residual.shape[0]
    coefficients = np.zeros((n, 2), dtype=float)
    lag_offsets = np.zeros(n, dtype=int)
    scores = np.full(n, -np.inf, dtype=float)
    for row_index in range(n):
        best_coefficients = np.zeros(2, dtype=float)
        best_lag = 0
        best_score = -np.inf
        for lag in lag_values:
            lag_int = int(lag)
            observed = residual[row_index, start + lag_int : end + lag_int]
            row_coefficients = observed @ basis
            gain = float(np.dot(row_coefficients, row_coefficients))
            score = gain - float(lag_penalty) * abs(lag_int)
            if score > best_score:
                best_score = score
                best_lag = lag_int
                best_coefficients = row_coefficients
        coefficients[row_index, :] = best_coefficients
        lag_offsets[row_index] = best_lag
        scores[row_index] = best_score
    return coefficients, lag_offsets, scores


def _best_registrations_for_basis(
    R: np.ndarray,
    *,
    start: int,
    end: int,
    peak_index: int,
    local_basis: np.ndarray,
    lag_values: np.ndarray,
    extension_values: np.ndarray,
    lag_penalty: float,
    extension_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(R, dtype=float)
    basis = np.asarray(local_basis, dtype=float)
    n, M = residual.shape
    coefficients = np.zeros((n, 2), dtype=float)
    lag_offsets = np.zeros(n, dtype=int)
    extension_offsets = np.zeros(n, dtype=int)
    scores = np.full(n, -np.inf, dtype=float)
    peak_offset = int(peak_index) - int(start)

    for extension in np.asarray(extension_values, dtype=int):
        extension_int = int(extension)
        deformed_basis = _deformed_local_basis(basis, extension_int, peak_offset=peak_offset)
        target_length = deformed_basis.shape[0]
        gram = deformed_basis.T @ deformed_basis
        regularized = gram + 1.0e-9 * np.eye(gram.shape[0])
        for lag in np.asarray(lag_values, dtype=int):
            lag_int = int(lag)
            observed_start = int(start) + lag_int
            observed_end = observed_start + target_length
            if observed_start < 0 or observed_end > M:
                continue
            observed = residual[:, observed_start:observed_end]
            row_coefficients = np.linalg.solve(regularized, deformed_basis.T @ observed.T).T
            fitted = row_coefficients @ deformed_basis.T
            gains = np.sum(fitted * fitted, axis=1)
            candidate_scores = gains - float(lag_penalty) * abs(lag_int) - float(extension_penalty) * abs(extension_int)
            improved = candidate_scores > scores
            if np.any(improved):
                coefficients[improved, :] = row_coefficients[improved, :]
                lag_offsets[improved] = lag_int
                extension_offsets[improved] = extension_int
                scores[improved] = candidate_scores[improved]
    return coefficients, lag_offsets, extension_offsets, scores


def _registered_rows_for_lags(
    R: np.ndarray,
    start: int,
    end: int,
    lag_offsets: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(R, dtype=float)
    lags = np.asarray(lag_offsets, dtype=int)
    registered = np.zeros((residual.shape[0], end - start), dtype=float)
    for row_index, lag in enumerate(lags):
        registered[row_index, :] = residual[row_index, start + int(lag) : end + int(lag)]
    return registered


def _registered_rows_for_registrations(
    R: np.ndarray,
    start: int,
    end: int,
    lag_offsets: np.ndarray,
    extension_offsets: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(R, dtype=float)
    lags = np.asarray(lag_offsets, dtype=int)
    extensions = np.asarray(extension_offsets, dtype=int)
    base_length = int(end) - int(start)
    registered = np.zeros((residual.shape[0], base_length), dtype=float)
    for row_index, (lag, extension) in enumerate(zip(lags, extensions, strict=False)):
        observed_start = int(start) + int(lag)
        observed_end = observed_start + base_length + int(extension)
        if observed_start < 0 or observed_end > residual.shape[1] or observed_end <= observed_start:
            continue
        observed = residual[row_index, observed_start:observed_end]
        registered[row_index, :] = _resample_1d(observed, base_length)
    return registered


def _coefficients_for_fixed_lags(
    R: np.ndarray,
    start: int,
    end: int,
    local_basis: np.ndarray,
    lag_offsets: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(R, dtype=float)
    coefficients = np.zeros((residual.shape[0], 2), dtype=float)
    for row_index, lag in enumerate(np.asarray(lag_offsets, dtype=int)):
        observed_start = start + int(lag)
        observed_end = end + int(lag)
        if 0 <= observed_start < observed_end <= residual.shape[1]:
            coefficients[row_index, :] = residual[row_index, observed_start:observed_end] @ local_basis
    return coefficients


def _coefficients_for_fixed_registrations(
    R: np.ndarray,
    start: int,
    end: int,
    peak_index: int,
    local_basis: np.ndarray,
    lag_offsets: np.ndarray,
    extension_offsets: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(R, dtype=float)
    basis_local = np.asarray(local_basis, dtype=float)
    coefficients = np.zeros((residual.shape[0], 2), dtype=float)
    peak_offset = int(peak_index) - int(start)
    for row_index, (lag, extension) in enumerate(
        zip(np.asarray(lag_offsets, dtype=int), np.asarray(extension_offsets, dtype=int), strict=False)
    ):
        deformed_basis = _deformed_local_basis(basis_local, int(extension), peak_offset=peak_offset)
        observed_start = int(start) + int(lag)
        observed_end = observed_start + deformed_basis.shape[0]
        if 0 <= observed_start < observed_end <= residual.shape[1]:
            gram = deformed_basis.T @ deformed_basis
            regularized = gram + 1.0e-9 * np.eye(gram.shape[0])
            coefficients[row_index, :] = np.linalg.solve(regularized, deformed_basis.T @ residual[row_index, observed_start:observed_end])
    return coefficients


def _candidate_reconstruction(candidate: CandidateFit) -> np.ndarray:
    if candidate.lag_offsets is None and candidate.extension_offsets is None:
        return candidate.coefficients @ candidate.basis.T
    event = _event_from_candidate(candidate)
    return _event_reconstruction(event)


def _event_reconstruction(event: HLLRDEvent) -> np.ndarray:
    coefficients = np.asarray(event.coefficients, dtype=float)
    if event.lag_offsets is None and event.extension_offsets is None:
        return coefficients @ event.basis.T
    return reconstruct_registered_events((event,), coefficients)


def _row_dictionary(
    events: tuple[HLLRDEvent, ...],
    row_index: int,
    M: int,
    *,
    lag_offsets: np.ndarray | None = None,
    extension_offsets: np.ndarray | None = None,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for event_index, event in enumerate(events):
        lag = _event_row_lag(event, row_index, lag_offsets=lag_offsets, event_index=event_index)
        extension = _event_row_extension(
            event,
            row_index,
            extension_offsets=extension_offsets,
            event_index=event_index,
        )
        row_basis = _row_event_basis(event, M, lag=lag, extension=extension)
        columns.extend([row_basis[:, 0], row_basis[:, 1]])
    if not columns:
        return np.zeros((M, 0), dtype=float)
    return np.column_stack(columns)


def _event_row_lag(
    event: HLLRDEvent,
    row_index: int,
    *,
    lag_offsets: np.ndarray | None = None,
    event_index: int = 0,
) -> int:
    if lag_offsets is not None:
        table = np.asarray(lag_offsets, dtype=int)
        if table.ndim == 2 and row_index < table.shape[0] and event_index < table.shape[1]:
            return int(table[row_index, event_index])
        if table.ndim == 1 and row_index < table.shape[0]:
            return int(table[row_index])
    if event.lag_offsets is None:
        return 0
    event_lags = np.asarray(event.lag_offsets, dtype=int)
    if row_index >= event_lags.shape[0]:
        return 0
    return int(event_lags[row_index])


def _event_row_extension(
    event: HLLRDEvent,
    row_index: int,
    *,
    extension_offsets: np.ndarray | None = None,
    event_index: int = 0,
) -> int:
    if extension_offsets is not None:
        table = np.asarray(extension_offsets, dtype=int)
        if table.ndim == 2 and row_index < table.shape[0] and event_index < table.shape[1]:
            return int(table[row_index, event_index])
        if table.ndim == 1 and row_index < table.shape[0]:
            return int(table[row_index])
    if event.extension_offsets is None:
        return 0
    event_extensions = np.asarray(event.extension_offsets, dtype=int)
    if row_index >= event_extensions.shape[0]:
        return 0
    return int(event_extensions[row_index])


def _row_event_basis(event: HLLRDEvent, M: int, *, lag: int, extension: int) -> np.ndarray:
    local_basis = np.asarray(event.basis[event.start : event.end, :], dtype=float)
    peak_offset = int(event.peak_index) - int(event.start)
    deformed_local = _deformed_local_basis(local_basis, int(extension), peak_offset=peak_offset)
    start = int(event.start) + int(lag)
    end = start + deformed_local.shape[0]
    row_basis = np.zeros((M, 2), dtype=float)
    source_left = max(0, -start)
    source_right = deformed_local.shape[0] - max(0, end - M)
    target_start = max(0, start)
    target_end = min(M, end)
    if target_end > target_start and source_right > source_left:
        row_basis[target_start:target_end, :] = deformed_local[source_left:source_right, :]
    return row_basis


def _deformed_local_basis(local_basis: np.ndarray, extension: int, *, peak_offset: int) -> np.ndarray:
    local = np.asarray(local_basis, dtype=float)
    if local.ndim != 2:
        raise ValueError("local_basis must be a 2D array")
    if local.shape[0] < 2:
        return local.copy()
    extension_int = int(extension)
    target_length = max(2, local.shape[0] + extension_int)
    if target_length == local.shape[0]:
        return local.copy()
    if extension_int < 0:
        return _resample_local_basis(local, target_length)
    pivot = int(np.clip(int(peak_offset), 0, local.shape[0] - 1))
    hold = np.repeat(local[pivot : pivot + 1, :], extension_int, axis=0)
    return np.vstack((local[: pivot + 1, :], hold, local[pivot + 1 :, :]))


def _resample_local_basis(local_basis: np.ndarray, target_length: int) -> np.ndarray:
    local = np.asarray(local_basis, dtype=float)
    return np.column_stack([_resample_1d(local[:, component], target_length) for component in range(local.shape[1])])


def _resample_1d(values: np.ndarray, target_length: int) -> np.ndarray:
    source_values = np.asarray(values, dtype=float)
    length = max(1, int(target_length))
    if source_values.size == length:
        return source_values.copy()
    if source_values.size == 0:
        return np.zeros(length, dtype=float)
    if source_values.size == 1:
        return np.full(length, float(source_values[0]), dtype=float)
    source = np.linspace(0.0, 1.0, source_values.size)
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, source_values)


def _shifted_basis(basis: np.ndarray, lag: int) -> np.ndarray:
    base = np.asarray(basis, dtype=float)
    shifted = np.zeros_like(base)
    lag_int = int(lag)
    if lag_int == 0:
        return base.copy()
    if abs(lag_int) >= base.shape[0]:
        return shifted
    if lag_int > 0:
        shifted[lag_int:, :] = base[:-lag_int, :]
    else:
        shifted[:lag_int, :] = base[-lag_int:, :]
    return shifted


def _lag_histogram(lag_offsets: np.ndarray) -> dict[str, int]:
    values = np.asarray(lag_offsets, dtype=int)
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def _config_max_lag(config: HLLRDV1Config, event: HLLRDEvent) -> int:
    if isinstance(config, HLLRDV2Config):
        return max(0, int(config.max_lag_stations))
    if event.lag_offsets is None:
        return 0
    return int(np.max(np.abs(event.lag_offsets), initial=0))


def _config_max_extend(config: HLLRDV1Config, event: HLLRDEvent) -> int:
    if isinstance(config, HLLRDV2Config):
        return max(0, int(config.max_extend_stations))
    if event.extension_offsets is None:
        return 0
    return int(np.max(np.abs(event.extension_offsets), initial=0))


def _config_lag_direction(config: HLLRDV1Config) -> str:
    if isinstance(config, HLLRDV2Config):
        return config.lag_direction
    return "both"


def _config_extend_direction(config: HLLRDV1Config) -> str:
    if isinstance(config, HLLRDV2Config):
        return config.extend_direction
    return "nonnegative"


def _config_lag_penalty(config: HLLRDV1Config) -> float:
    if isinstance(config, HLLRDV2Config):
        return float(config.lag_penalty)
    return 0.0


def _config_extend_penalty(config: HLLRDV1Config) -> float:
    if isinstance(config, HLLRDV2Config):
        return float(config.extend_penalty)
    return 0.0


def _score_interval(
    R: np.ndarray,
    *,
    peak_index: int,
    peak_start_index: int | None = None,
    length: int,
    sigma_hat: float,
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
    activation_energy_floor: float,
    n_min: int,
    config: HLLRDV1Config,
) -> CandidateFit:
    n, _M = R.shape
    threshold = analytic_null_threshold(end - start, n, sigma_hat, config.c_null)
    activation_threshold = activation_threshold_for_length(end - start, activation_energy_floor, config.activation_scale)
    if _lag_registration_enabled(config):
        return local_rank2_lag_registered_candidate(
            R,
            peak_index=peak_index,
            start=start,
            end=end,
            activation_threshold=activation_threshold,
            threshold=threshold,
            n_min=n_min,
            lambda_i=config.lambda_i,
            lambda_activation=config.lambda_activation,
            lag_penalty=config.lag_penalty,
            max_lag_stations=config.max_lag_stations,
            lag_direction=config.lag_direction,
            max_extend_stations=config.max_extend_stations,
            extend_direction=config.extend_direction,
            extend_penalty=config.extend_penalty,
            registration_iterations=config.registration_iterations,
            registration_tolerance=config.registration_tolerance,
        )
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
        lag_offsets=candidate.lag_offsets,
        extension_offsets=candidate.extension_offsets,
        lag_penalty=config.lag_penalty if _lag_registration_enabled(config) else 0.0,
        extension_penalty=config.extend_penalty if _lag_registration_enabled(config) and isinstance(config, HLLRDV2Config) else 0.0,
    )
    if simplified_candidate.score <= 0.0:
        return candidate
    return simplified_candidate


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
        lag_offsets=None if candidate.lag_offsets is None else np.asarray(candidate.lag_offsets, dtype=int).copy(),
        extension_offsets=(
            None
            if candidate.extension_offsets is None
            else np.asarray(candidate.extension_offsets, dtype=int).copy()
        ),
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
                lag_offsets=None if event.lag_offsets is None else np.asarray(event.lag_offsets, dtype=int).copy(),
                extension_offsets=(
                    None
                    if event.extension_offsets is None
                    else np.asarray(event.extension_offsets, dtype=int).copy()
                ),
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
    lag_offsets: np.ndarray | None = None,
    extension_offsets: np.ndarray | None = None,
    lag_penalty: float = 0.0,
    extension_penalty: float = 0.0,
) -> CandidateFit:
    residual = np.asarray(R, dtype=float)
    basis_local = np.asarray(local_basis, dtype=float)
    if basis_local.shape != (end - start, 2):
        raise ValueError("local_basis must have shape interval_length x 2")
    basis = np.zeros((residual.shape[1], 2), dtype=float)
    basis[start:end, :] = basis_local
    if lag_offsets is None and extension_offsets is None:
        coefficients = residual[:, start:end] @ basis_local
        lag_values = None
        extension_values = None
    else:
        lag_values = np.zeros(residual.shape[0], dtype=int) if lag_offsets is None else np.asarray(lag_offsets, dtype=int)
        extension_values = (
            np.zeros(residual.shape[0], dtype=int)
            if extension_offsets is None
            else np.asarray(extension_offsets, dtype=int)
        )
        coefficients = _coefficients_for_fixed_registrations(
            residual,
            start,
            end,
            peak_index,
            basis_local,
            lag_values,
            extension_values,
        )
    raw_gain = float(np.sum(coefficients * coefficients))
    norms = np.linalg.norm(coefficients, axis=1)
    active_mask = norms > float(activation_threshold)
    if int(np.count_nonzero(active_mask)) < int(n_min):
        active_mask = np.zeros(residual.shape[0], dtype=bool)
    active_coefficients = coefficients.copy()
    active_coefficients[~active_mask, :] = 0.0
    active_gain = float(np.sum(active_coefficients * active_coefficients))
    lag_cost = 0.0
    if lag_values is not None and float(lag_penalty) > 0.0:
        lag_cost = float(lag_penalty) * float(np.sum(np.abs(lag_values[active_mask])))
    extension_cost = 0.0
    if extension_values is not None and float(extension_penalty) > 0.0:
        extension_cost = float(extension_penalty) * float(np.sum(np.abs(extension_values[active_mask])))
    score = active_gain - float(threshold) - float(lambda_i) * (end - start) - float(lambda_activation) * int(
        np.count_nonzero(active_mask)
    ) - lag_cost - extension_cost
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
        lag_offsets=None if lag_values is None else lag_values.copy(),
        extension_offsets=None if extension_values is None else extension_values.copy(),
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
        "point_count_histogram": dict(metric.get("local_simplifier_point_count_histogram", {}) or {}),
    }


def _explained_fraction(X: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.sum(X * X))
    if denominator <= 0.0:
        return 0.0
    return float(1.0 - np.sum(residual * residual) / denominator)
