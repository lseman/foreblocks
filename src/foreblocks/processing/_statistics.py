"""Small time-series statistics implemented with NumPy and SciPy.

Only the modes used by preprocessing are supported. Constant series produce
NaN correlations and test statistics, since their variance is zero.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len
from scipy.linalg import solve_toeplitz
from scipy.stats import chi2


def _series(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x):
        raise ValueError("expected a nonempty 1-D series")
    return x


def acf(values, nlags, *, fft=True):
    """Biased, mean-centered autocorrelations through ``nlags`` inclusive."""
    x = _series(values)
    if int(nlags) != nlags or nlags < 0:
        raise ValueError("nlags must be a nonnegative integer")
    nlags = min(int(nlags), len(x) - 1)
    centered = x - x.mean()
    if fft:
        size = next_fast_len(2 * len(x) - 1)
        spectrum = np.fft.rfft(centered, n=size)
        covariance = np.fft.irfft(spectrum * spectrum.conj(), n=size)[: nlags + 1]
    else:
        covariance = np.array(
            [
                np.dot(centered[lag:], centered[: len(x) - lag])
                for lag in range(nlags + 1)
            ]
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return covariance / covariance[0]


def pacf(values, nlags, *, method="ywm"):
    """Partial autocorrelations using biased Yule–Walker equations."""
    x = _series(values)
    if method != "ywm":
        raise ValueError("only method='ywm' is supported")
    if int(nlags) != nlags or not 0 <= nlags < len(x) // 2:
        raise ValueError("nlags must be nonnegative and less than half the sample size")
    correlations = acf(x, nlags, fft=False)
    result = np.ones(int(nlags) + 1)
    if not np.isfinite(correlations).all():
        result[1:] = np.nan
        return result
    for lag in range(1, int(nlags) + 1):
        result[lag] = solve_toeplitz(correlations[:lag], correlations[1 : lag + 1])[-1]
    return result


def ljung_box(values, lags):
    """Return cumulative Q statistics and chi-square p-values at given lags.

    An integer requests all lags 1..lags; a sequence requests specific lags.
    Uses zero fitted-model degrees of freedom, as all preprocessing callers do.
    """
    x = _series(values)
    requested = np.arange(1, int(lags) + 1) if np.isscalar(lags) else np.asarray(lags)
    if (
        requested.ndim != 1
        or not len(requested)
        or not np.isfinite(requested).all()
        or np.any(requested != requested.astype(int))
        or np.any(requested < 1)
        or np.any(requested >= len(x))
    ):
        raise ValueError("lags must be integers between 1 and sample size - 1")
    if np.isscalar(lags) and int(lags) != lags:
        raise ValueError("lags must be integers")
    requested = requested.astype(int)
    maximum = int(requested.max())
    correlations = acf(x, maximum, fft=False)[1:]
    q = (
        len(x)
        * (len(x) + 2)
        * np.cumsum(correlations**2 / (len(x) - np.arange(1, maximum + 1)))
    )
    selected = q[requested - 1]
    return selected, chi2.sf(selected, requested)


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, using zero for short or constant inputs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(c) else float(c)
