from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class CandidateFit:
    peak_index: int
    start: int
    end: int
    basis: np.ndarray
    coefficients: np.ndarray
    active_mask: np.ndarray
    raw_gain: float
    active_gain: float
    score: float
    threshold: float
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


def length_grid(M: int, L_min: int | None = None, L_max: int | None = None) -> tuple[int, ...]:
    if M < 1:
        raise ValueError("M must be positive")
    minimum = max(2, max(5, int(np.ceil(0.02 * M))) if L_min is None else int(L_min))
    maximum = max(minimum, int(np.ceil(0.25 * M)) if L_max is None else int(L_max))
    values: list[int] = []
    current = minimum
    while current < maximum:
        values.append(current)
        current *= 2
    values.append(maximum)
    return tuple(dict.fromkeys(min(value, M) for value in values))


def smooth_energy(energy: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(energy, dtype=float)
    if window <= 1 or values.size < 3:
        return values.copy()
    width = min(int(window), values.size)
    if width % 2 == 0:
        width -= 1
    if width <= 1:
        return values.copy()
    kernel = np.ones(width, dtype=float) / width
    pad = width // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def find_residual_energy_peaks(
    R: np.ndarray,
    *,
    kappa_peak: float = 2.0,
    min_peak_distance: int | None = None,
    smoothing_window: int = 5,
) -> np.ndarray:
    residual = np.asarray(R, dtype=float)
    if residual.ndim != 2:
        raise ValueError("R must be a 2D matrix")
    energy = np.mean(residual * residual, axis=0)
    smoothed = smooth_energy(energy, smoothing_window)
    median = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - median)))
    threshold = median + float(kappa_peak) * mad
    distance = max(1, int(min_peak_distance or 1))
    peaks, _properties = find_peaks(smoothed, height=threshold, distance=distance)
    if peaks.size == 0 and smoothed.size and float(np.max(smoothed)) > threshold:
        peaks = np.asarray([int(np.argmax(smoothed))], dtype=int)
    return peaks[np.argsort(smoothed[peaks])[::-1]]


def backtrack_peak_rise_start(
    energy: np.ndarray,
    peak_index: int,
    *,
    baseline: float | None = None,
    rise_fraction: float = 0.05,
) -> int:
    values = np.asarray(energy, dtype=float)
    if values.ndim != 1:
        raise ValueError("energy must be a 1D array")
    if values.size == 0:
        raise ValueError("energy must be non-empty")
    peak = int(np.clip(int(peak_index), 0, values.size - 1))
    reference = float(np.median(values)) if baseline is None else float(baseline)
    peak_value = float(values[peak])
    if not np.isfinite(peak_value) or peak_value <= reference:
        return peak

    fraction = max(0.0, min(1.0, float(rise_fraction)))
    threshold = reference + fraction * (peak_value - reference)
    start = peak
    while start > 0 and float(values[start - 1]) >= threshold:
        start -= 1
    return int(start)


def centered_interval(peak_index: int, length: int, M: int) -> tuple[int, int]:
    width = max(1, min(int(length), int(M)))
    start = int(peak_index) - width // 2
    start = max(0, min(start, M - width))
    return start, start + width


def analytic_null_threshold(length: int, n: int, sigma_hat: float, c_null: float) -> float:
    return float(c_null) * float(sigma_hat) ** 2 * (int(n) + int(length))


def local_rank2_candidate(
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
) -> CandidateFit:
    residual = np.asarray(R, dtype=float)
    n, M = residual.shape
    if not (0 <= start < end <= M):
        raise ValueError("candidate interval must satisfy 0 <= start < end <= M")
    local = residual[:, start:end]
    U, singular_values, Wt = np.linalg.svd(local, full_matrices=False)
    rank = min(2, Wt.shape[0])
    local_basis = np.zeros((end - start, 2), dtype=float)
    if rank:
        local_basis[:, :rank] = Wt[:rank, :].T
    basis = np.zeros((M, 2), dtype=float)
    basis[start:end, :] = local_basis

    coefficients = local @ local_basis
    raw_gain = float(np.sum(singular_values[:rank] ** 2))
    norms = np.linalg.norm(coefficients, axis=1)
    active_mask = norms > float(activation_threshold)
    if int(np.count_nonzero(active_mask)) < int(n_min):
        active_mask = np.zeros(n, dtype=bool)
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
    )


def interval_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    left = max(first[0], second[0])
    right = min(first[1], second[1])
    intersection = max(0, right - left)
    union = max(first[1], second[1]) - min(first[0], second[0])
    if union <= 0:
        return 0.0
    return intersection / union


def is_duplicate_interval(
    candidate: tuple[int, int],
    selected: tuple[int, int],
    *,
    iou_threshold: float = 0.8,
) -> bool:
    candidate_length = candidate[1] - candidate[0]
    selected_length = selected[1] - selected[0]
    if candidate_length <= 0 or selected_length <= 0:
        return False
    length_ratio = candidate_length / selected_length
    return interval_iou(candidate, selected) > iou_threshold and 0.5 < length_ratio < 2.0
