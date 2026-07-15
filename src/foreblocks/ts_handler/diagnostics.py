"""foreblocks.ts_handler.diagnostics.

Diagnostic and statistical analysis functions for time-series preprocessing.

"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import entropy, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, pacf

from foreblocks.ts_handler.utils import _prepare_series_for_diagnostics


def detect_stationarity(data: np.ndarray, D: int) -> list[float]:
    pvals: list[float] = []
    for i in range(D):
        clean, _ = _prepare_series_for_diagnostics(data[:, i], max_points=2048)
        if len(clean) <= 10:
            pvals.append(1.0)
            continue
        try:
            pvals.append(float(adfuller(clean, autolag="AIC")[1]))
        except Exception:
            pvals.append(1.0)
    return pvals


def detect_seasonality(data: np.ndarray, D: int) -> tuple[list[bool], list[int | None]]:
    seasonal_flags: list[bool] = []
    detected_periods: list[int | None] = []

    for i in range(D):
        clean, stride = _prepare_series_for_diagnostics(data[:, i], max_points=4096)
        if len(clean) < 10:
            seasonal_flags.append(False)
            detected_periods.append(None)
            continue

        norm = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)

        try:
            nperseg = min(256, max(16, len(norm)))
            freqs, psd = welch(norm, nperseg=nperseg)
            if not np.any(psd > 0):
                seasonal_flags.append(False)
                detected_periods.append(None)
                continue

            peaks, _ = find_peaks(psd, height=0.1 * np.max(psd))
            if len(peaks) == 0:
                seasonal_flags.append(False)
                detected_periods.append(None)
                continue

            peak_freq = float(freqs[peaks[np.argmax(psd[peaks])]])
            period = int(round((1.0 / peak_freq) * stride)) if peak_freq > 0 else None

            acf_vals = acf(norm, nlags=min(100, len(norm) // 2), fft=True)
            acf_peaks, _ = find_peaks(acf_vals, height=0.2)
            strength = float(np.max(acf_vals[acf_peaks])) if len(acf_peaks) > 0 else 0.0
            is_seasonal = strength > 0.3
        except Exception:
            is_seasonal, period = False, None

        seasonal_flags.append(bool(is_seasonal))
        detected_periods.append(period if is_seasonal else None)

    return seasonal_flags, detected_periods


def analyze_signal_quality(data: np.ndarray, D: int) -> tuple[list[float], list[float]]:
    flatness_scores: list[float] = []
    snr_scores: list[float] = []

    for i in range(D):
        clean, _ = _prepare_series_for_diagnostics(data[:, i], max_points=4096)
        if len(clean) < 10:
            flatness_scores.append(1.0)
            snr_scores.append(0.0)
            continue

        norm = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)
        spec = np.abs(np.fft.rfft(norm)) ** 2
        spec = spec[1 : max(2, len(spec) // 2)]
        if len(spec) == 0:
            flatness_scores.append(1.0)
            snr_scores.append(0.0)
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            flat = float(np.exp(np.mean(np.log(spec + 1e-8))) / (np.mean(spec) + 1e-8))
        snr = float(np.max(spec) / (np.mean(spec) + 1e-8))

        flatness_scores.append(flat)
        snr_scores.append(snr)

    return flatness_scores, snr_scores


def score_ljung_box(data: np.ndarray, D: int) -> list[float]:
    """Formal test for whether the series is structurally autoregressive vs white noise."""
    pvals: list[float] = []
    for i in range(D):
        clean, _ = _prepare_series_for_diagnostics(data[:, i], max_points=2048)
        if len(clean) < 30:
            pvals.append(1.0)
            continue
        try:
            # Test up to 10 lags, take the minimum p-value indicating any structure
            res = acorr_ljungbox(clean, lags=[min(10, len(clean) // 3)], return_df=True)
            pvals.append(float(res["lb_pvalue"].min()))
        except Exception:
            pvals.append(1.0)
    return pvals


def score_pacf(data: np.ndarray, D: int) -> list[int]:
    scores: list[int] = []
    for i in range(D):
        clean, _ = _prepare_series_for_diagnostics(data[:, i], max_points=2048)
        if len(clean) < 30:
            scores.append(0)
            continue
        try:
            pacf_vals = pacf(clean, nlags=min(20, len(clean) // 3), method="ywm")
            scores.append(int(np.sum(np.abs(pacf_vals[1:]) > 0.2)))
        except Exception:
            scores.append(0)
    return scores


def estimate_ewt_bands(data: np.ndarray, D: int) -> list[int]:
    band_estimates: list[int] = []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) < 20:
            band_estimates.append(3)
            continue
        hist, _ = np.histogram(clean, bins=20, density=True)
        hist = np.maximum(hist, 1e-10)
        hist /= np.sum(hist)
        ent = float(entropy(hist))
        band_estimates.append(int(np.clip(ent * 2, 2, 10)))
    return band_estimates


def _get_iterative_imputer_class() -> Any | None:
    try:
        from fancyimpute import IterativeImputer

        return IterativeImputer
    except ImportError:
        try:
            import sklearn.experimental.enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            return IterativeImputer
        except ImportError:
            return None
