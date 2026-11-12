"""Robust local linear LOWESS with optional Numba acceleration.

Uses nearest-neighbor tricube weights, bisquare residual reweighting, and
linear interpolation between fits separated by ``delta``. Storage is O(n).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:

    def njit(**kwargs):
        return lambda function: function


@njit(cache=True)
def _lowess_sorted(y, x, frac, it, delta):
    n = len(y)
    fitted = y.copy()
    if n < 2:
        return fitted
    neighbors = min(n, max(2, int(frac * n + 1e-10)))
    robust = np.ones(n)
    weights = np.empty(neighbors)
    for iteration in range(it + 1):
        left = 0
        i = 0
        previous = -1
        while i < n:
            while left + neighbors < n and x[i] > (x[left] + x[left + neighbors]) / 2:
                left += 1
            radius = max(x[i] - x[left], x[left + neighbors - 1] - x[i])
            total = 0.0
            positive = 0
            for j in range(neighbors):
                distance = abs(x[left + j] - x[i]) / radius if radius > 0 else 1.0
                w = (1.0 - distance**3) ** 3 * robust[left + j]
                weights[j] = w
                total += w
                positive += w > 1e-12
            fitted[i] = y[i]
            if positive >= 2:
                mean_x = 0.0
                for j in range(neighbors):
                    weights[j] /= total
                    mean_x += weights[j] * x[left + j]
                variance = 0.0
                for j in range(neighbors):
                    variance += weights[j] * (x[left + j] - mean_x) ** 2
                variance = max(variance, 1e-12)
                estimate = 0.0
                for j in range(neighbors):
                    estimate += (
                        weights[j]
                        * y[left + j]
                        * (1.0 + (x[i] - mean_x) * (x[left + j] - mean_x) / variance)
                    )
                fitted[i] = estimate
            if previous >= 0 and i > previous + 1:
                for j in range(previous + 1, i):
                    fraction = (x[j] - x[previous]) / (x[i] - x[previous])
                    fitted[j] = (1 - fraction) * fitted[previous] + fraction * fitted[i]
            previous = i
            # Repeated positions share a fit. Fit the last point within delta
            # next, then interpolate the points skipped between these anchors.
            end = i + 1
            while end < n and x[end] <= x[i] + delta:
                if x[end] == x[i]:
                    fitted[end] = fitted[i]
                    previous = end
                end += 1
            if previous == n - 1:
                break
            i = max(previous + 1, min(end, n - 1) - 1)
        if iteration < it:
            residuals = np.abs(y - fitted)
            median = np.median(residuals)
            for j in range(n):
                scaled = (
                    min(residuals[j] / (6 * median), 1.0)
                    if median > 0
                    else (1.0 if residuals[j] > 0 else 0.0)
                )
                robust[j] = (1 - scaled * scaled) ** 2
    return fitted


def _lowess_smooth(values, positions, *, frac, it, delta=0.0):
    """Smooth paired 1-D arrays, retaining input order and missing positions."""
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(positions, dtype=np.float64)
    if y.ndim != 1 or x.ndim != 1 or y.shape != x.shape:
        raise ValueError("values and positions must be matching 1-D arrays")
    if not 0 <= frac <= 1:
        raise ValueError("frac must be in [0, 1]")
    if int(it) != it or it < 0:
        raise ValueError("it must be a nonnegative integer")
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    result = np.full(y.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(y) & np.isfinite(x))
    order = valid[np.argsort(x[valid])]
    if len(order):
        result[order] = _lowess_sorted(
            y[order], x[order], float(frac), int(it), float(delta)
        )
    return result
