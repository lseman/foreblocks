"""Signal-analysis utilities for transformer tuner heuristics.

Extracted from tuner.py to keep the main module focused on TunerConfig,
Pydantic report models, and the TransformerTuner class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _choose_nearest_candidate(
    value: float, candidates: list[int], max_value: float = float("inf")
) -> int:
    viable = [c for c in candidates if c <= max_value]
    pool = viable if viable else candidates
    return min(pool, key=lambda c: abs(c - value))


def _to_1d_array(series: Any) -> np.ndarray:
    if isinstance(series, torch.Tensor):
        arr = series.detach().cpu().numpy()
    else:
        arr = np.asarray(series, dtype=np.float64)

    if arr.ndim == 0:
        raise ValueError("series must contain at least one observation")
    if arr.ndim == 1:
        values = arr
    elif arr.ndim == 2:
        values = arr.reshape(-1) if 1 in arr.shape else arr.mean(axis=-1)
    else:
        values = arr.reshape(arr.shape[0], -1).mean(axis=-1)

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        raise ValueError("series must contain at least one finite observation")

    return values[finite_mask].astype(np.float64, copy=False)


def _standard_deviation(values: np.ndarray) -> float:
    return 0.0 if values.size <= 1 else float(np.std(values, ddof=0))


def _difference(values: np.ndarray) -> np.ndarray:
    return np.empty(0, dtype=np.float64) if values.size <= 1 else np.diff(values)


def _linear_detrend(values: np.ndarray) -> tuple[np.ndarray, float]:
    if values.size <= 2:
        return values - values.mean(), 0.0

    x = np.arange(values.size, dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    trend = A @ coef
    residual = values - trend
    trend_strength = float(_clamp(1.0 - np.var(residual) / (np.var(values) + 1e-12)))
    return residual, trend_strength


def _autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values - values.mean()
    variance = float(np.dot(centered, centered))
    if variance <= 1e-12:
        return np.zeros(max_lag + 1, dtype=np.float64)

    corr = np.correlate(centered, centered, mode="full")
    mid = corr.size // 2
    acf = corr[mid : mid + max_lag + 1] / variance
    return acf.astype(np.float64, copy=False)


def _find_acf_peaks(acf: np.ndarray, max_peaks: int = 6) -> list[tuple[int, float]]:
    peaks: list[tuple[int, float]] = []
    if acf.size <= 3:
        return peaks

    for lag in range(2, acf.size - 1):
        val = float(acf[lag])
        if val < 0.1:
            continue
        if val >= acf[lag - 1] and val >= acf[lag + 1]:
            peaks.append((lag, val))

    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:max_peaks]


def _spectral_entropy(normalized_power: np.ndarray) -> float:
    if normalized_power.size == 0:
        return 1.0
    power = normalized_power[normalized_power > 0]
    if power.size == 0:
        return 1.0
    entropy = -float(np.sum(power * np.log(power)))
    return entropy / max(np.log(power.size), 1e-8)


def _lempel_ziv_complexity(values: np.ndarray, alphabet_size: int = 8) -> float:
    if values.size <= 2:
        return 0.5

    min_v, max_v = np.min(values), np.max(values)
    if max_v - min_v < 1e-12:
        return 0.0

    symbols = np.digitize(values, np.linspace(min_v, max_v, alphabet_size + 1)[:-1]) - 1
    symbols = np.clip(symbols, 0, alphabet_size - 1)

    seq_str = "".join(map(str, symbols))
    i = 0
    complexity = 0
    seen: set[str] = set()

    while i < len(seq_str):
        for j in range(i + 1, len(seq_str) + 1):
            substring = seq_str[i:j]
            if substring not in seen:
                seen.add(substring)
                complexity += 1
                i = j - 1
                break
        else:
            i = len(seq_str)

    max_possible = len(seq_str) / np.log2(max(len(seq_str), 2))
    normalized = complexity / max(max_possible, 1e-8)
    return float(np.clip(normalized, 0.0, 2.0))


def _cwt_energy_profile(
    values: np.ndarray, scales: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    if values.size < 16:
        return np.array([1.0] * scales, dtype=np.float64), np.array([0], dtype=np.int64)

    centered = (values - values.mean()) / (np.std(values) + 1e-8)
    scale_list = np.logspace(0, 2, scales, base=2, dtype=np.float64)
    energies = np.zeros(scales, dtype=np.float64)

    for i, scale in enumerate(scale_list):
        sigma = scale / (2 * np.pi)
        t = np.arange(-4 * sigma, 4 * sigma, 1)
        if len(t) < 2:
            t = np.array([-1, 0, 1], dtype=np.float64)

        wavelet = np.exp(-0.5 * (t / sigma) ** 2) * np.cos(2 * np.pi * t / scale)
        wavelet /= np.sqrt(np.sum(wavelet**2))

        conv = np.convolve(centered, wavelet, mode="same")
        energies[i] = float(np.sqrt(np.mean(conv**2)))

    energies /= np.sum(energies) + 1e-12
    dominant_idx = np.argmax(energies)

    return energies, np.array([dominant_idx], dtype=np.int64)


__all__ = [
    "_autocorrelation",
    "_choose_nearest_candidate",
    "_cwt_energy_profile",
    "_clamp",
    "_difference",
    "_find_acf_peaks",
    "_lempel_ziv_complexity",
    "_linear_detrend",
    "_spectral_entropy",
    "_standard_deviation",
    "_to_1d_array",
]
