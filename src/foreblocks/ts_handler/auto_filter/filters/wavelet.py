"""foreblocks.ts_handler.auto_filter.filters.wavelet.

Wavelet denoising: BayesShrink + garrote thresholding, PyWavelets or pure-NumPy Haar.

"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from foreblocks.ts_handler.auto_filter.filters.utils import _as_series
from foreblocks.ts_handler.auto_filter.registry import register_filter

try:
    import pywt
except ImportError:  # pragma: no cover - exercised when dependency is absent.
    pywt = None


def _garrote_threshold(c: np.ndarray, thr: float) -> np.ndarray:
    a = np.abs(c)
    return np.where(a <= thr, 0.0, c * (1.0 - (thr * thr) / (a * a + 1e-12)))


def _haar_dwt_multilevel(
    x: np.ndarray, levels: int
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    approx = np.asarray(x, dtype=float).copy()
    details, lengths = [], []
    for _ in range(levels):
        lengths.append(len(approx))
        if len(approx) < 2:
            break
        if len(approx) % 2 == 1:
            approx = np.pad(approx, (0, 1), mode="edge")
        a = (approx[0::2] + approx[1::2]) / np.sqrt(2.0)
        d = (approx[0::2] - approx[1::2]) / np.sqrt(2.0)
        details.append(d)
        approx = a
    return approx, details, lengths


def _haar_idwt_multilevel(
    approx: np.ndarray,
    details: list[np.ndarray],
    lengths: list[int],
) -> np.ndarray:
    rec = np.asarray(approx, dtype=float).copy()
    for d, orig_len in zip(reversed(details), reversed(lengths)):
        up = np.empty(2 * len(d), dtype=float)
        up[0::2] = (rec + d) / np.sqrt(2.0)
        up[1::2] = (rec - d) / np.sqrt(2.0)
        rec = up[:orig_len]
    return rec


def _haar_wavelet_denoise_once(x: np.ndarray, levels: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return x.copy()
    max_levels = max(1, int(np.floor(np.log2(len(x)))) - 1)
    levels = int(np.clip(levels, 1, max_levels))

    approx, details, lengths = _haar_dwt_multilevel(x, levels=levels)
    if not details:
        return x.copy()

    den_details = _bayes_garrote_thresholds(details)
    return _haar_idwt_multilevel(approx, den_details, lengths)


def _bayes_garrote_thresholds(
    details: list[np.ndarray],
) -> list[np.ndarray]:
    """BayesShrink threshold + non-negative garrote shrinkage, per detail level.

    Noise σ is estimated from the finest detail band via the robust MAD
    estimator (Donoho & Johnstone). Per band we take the smaller of the
    BayesShrink and universal thresholds, with a mild geometric relaxation at
    coarser scales so low-frequency structure is preserved.
    """
    if not details:
        return details
    sigma = np.median(np.abs(details[0])) / 0.6745 + 1e-12
    out = []
    for j, d in enumerate(details):
        sigma_y = np.std(d) + 1e-12
        sigma_x = np.sqrt(max(sigma_y**2 - sigma**2, 0.0))
        bayes_thr = np.inf if sigma_x < 1e-12 else (sigma**2) / sigma_x
        universal_thr = sigma * np.sqrt(2.0 * np.log(d.size + 1.0))
        thr = min(bayes_thr, universal_thr) * (0.92**j)
        out.append(_garrote_threshold(d, thr))
    return out


def _pywt_denoise_once(x: np.ndarray, wavelet: str, levels: int) -> np.ndarray:
    coeffs = pywt.wavedec(x, wavelet, mode="periodization", level=levels)
    approx, details = coeffs[0], list(coeffs[1:])
    details = _bayes_garrote_thresholds(details)
    rec = pywt.waverec([approx, *details], wavelet, mode="periodization")
    return np.asarray(rec, dtype=float)[: len(x)]


@register_filter("Wavelet (Bayes+Garrote)")
def wavelet_denoise(
    ts: pd.Series,
    levels: int = 3,
    cycle_spins: int = 4,
    wavelet: str = "sym8",
) -> pd.Series:
    """Translation-invariant wavelet denoising with BayesShrink + garrote.

    Uses a real Daubechies/symlet basis (``wavelet``, default ``sym8``) via
    pywt — far fewer staircase artefacts than a Haar basis — with cycle
    spinning (averaging over circular shifts) for translation invariance.
    Falls back to a pure-NumPy Haar transform if pywt is unavailable.

    Parameters
    ----------
    levels:
        Number of decomposition levels (clamped to the signal length).
    cycle_spins:
        Number of circular shifts averaged for translation invariance.
    wavelet:
        pywt wavelet name (e.g. ``"sym8"``, ``"db4"``). Ignored on the Haar
        fallback.
    """
    x = ts.values.astype(float)
    n = len(x)
    if n < 8:
        return ts.copy()
    max_levels = max(1, int(np.floor(np.log2(n))) - 1)
    levels = int(np.clip(levels, 1, max_levels))
    spins = max(1, int(cycle_spins))

    if pywt is not None:
        try:
            denoise_once = lambda v: _pywt_denoise_once(
                v, wavelet, levels
            )  # noqa: E731
        except Exception:
            denoise_once = lambda v: _haar_wavelet_denoise_once(
                v, levels=levels
            )  # noqa: E731
    else:
        denoise_once = lambda v: _haar_wavelet_denoise_once(
            v, levels=levels
        )  # noqa: E731

    acc = np.zeros_like(x)
    for shift in range(spins):
        xs = np.roll(x, shift)
        try:
            rec = denoise_once(xs)
        except Exception:
            rec = _haar_wavelet_denoise_once(xs, levels=levels)
        acc += np.roll(rec[:n], -shift)
    return _as_series(acc / spins, ts.index, name="wavelet")
