"""Common helpers for the EMD/VMD decomposition package.

This module provides low-level utilities used across both the EMD and VMD families:
signal energy computation, boundary handling, FFT management, signal analysis,
mode processing, and fractal dimension estimation.

Most users should import from the package root (``foretools.decomposition.emd``)
rather than from this submodule directly.
"""

from __future__ import annotations

import numpy as np


def _energy(x: np.ndarray) -> float:
    """Compute the L2 energy (squared norm) of a signal.

    Parameters
    ----------
    x : array-like
        Input signal (any shape).

    Returns
    -------
    float
        Sum of squared values.
    """
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def _normalise(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation.

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    np.ndarray
        Normalised copy of *x*.
    """
    arr = np.asarray(x, dtype=np.float64)
    std = float(np.std(arr)) + 1e-12
    return (arr - np.mean(arr)) / std


def _reconstruct_error(original: np.ndarray, modes: list[np.ndarray]) -> float:
    """Reconstruction error as a fraction of original energy.

    Parameters
    ----------
    original : array-like
        The original signal that was decomposed.
    modes : list of array-like
        Decomposed modes (IMFs or VMD modes).

    Returns
    -------
    float
        ||x - sum(modes)||² / ||x||² in [0, 1].
    """
    x = np.asarray(original, dtype=np.float64)
    total_E = float(np.sum(x**2)) + 1e-12
    recon = np.sum(np.stack([np.asarray(m, dtype=np.float64) for m in modes], axis=0), axis=0)
    return float(np.sum((x - recon) ** 2) / total_E)


def _mode_energy_ratio(mode: np.ndarray, original: np.ndarray) -> float:
    """Fraction of original signal energy carried by one mode.

    Parameters
    ----------
    mode : array-like
        A single decomposed mode.
    original : array-like
        The original signal.

    Returns
    -------
    float
        Energy(mode) / Energy(original).
    """
    m = np.asarray(mode, dtype=np.float64)
    x = np.asarray(original, dtype=np.float64)
    total_E = float(np.sum(x**2)) + 1e-12
    return float(np.sum(m**2) / total_E)


def _is_imf(
    candidate: np.ndarray,
    min_oscillations: int = 3,
    symmetry_threshold: float = 0.5,
) -> bool:
    """Rough check whether a signal looks like a valid IMF.

    An IMF should have (approximately) equal numbers of extrema and zero-crossings,
    and the mean of its upper/lower envelopes should be near zero.

    Parameters
    ----------
    candidate : array-like
        Signal to test.
    min_oscillations : int
        Minimum number of zero-crossings required.
    symmetry_threshold : float
        Maximum allowed envelope asymmetry (as fraction of signal std).

    Returns
    -------
    bool
        True if the signal passes basic IMF criteria.
    """
    from scipy.signal import find_peaks

    x = np.asarray(candidate, dtype=np.float64)
    if x.size < 10:
        return False

    # Count extrema and zero-crossings
    _, _ = find_peaks(x)
    _, min_p = find_peaks(-x)
    n_extrema = len(_) + len(min_p)

    # Zero crossings of the mean-removed signal
    x_centered = x - np.mean(x)
    n_zc = int(np.sum(np.abs(np.diff(np.signbit(x_centered))) > 0))

    if n_extrema < min_oscillations or n_zc < min_oscillations:
        return False

    # Check envelope symmetry (rough): mean of the signal should be near zero
    mean_abs = np.mean(np.abs(x - np.mean(x)))
    std_x = float(np.std(x)) + 1e-12
    if mean_abs / std_x > symmetry_threshold:
        return False

    return True


def _validate_signal(
    signal: np.ndarray,
    min_length: int = 8,
    require_finite: bool = True,
) -> np.ndarray:
    """Validate and normalise an input signal.

    Parameters
    ----------
    signal : array-like
        Input signal.
    min_length : int
        Minimum number of samples required.
    require_finite : bool
        If True, raise ValueError on non-finite values.

    Returns
    -------
    np.ndarray
        Validated 1-D float64 array.

    Raises
    ------
    ValueError
        If the signal is invalid (wrong shape, too short, or non-finite).
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D signal, got {x.ndim}-D array")
    if x.size < min_length:
        raise ValueError(
            f"Signal too short: need at least {min_length} samples, got {x.size}"
        )
    if require_finite and not np.all(np.isfinite(x)):
        bad = np.where(~np.isfinite(x))[0]
        raise ValueError(f"Non-finite values at indices {bad[:10].tolist()}" + ("..." if len(bad) > 10 else ""))
    return x


__all__ = [
    "_energy",
    "_normalise",
    "_reconstruct_error",
    "_mode_energy_ratio",
    "_is_imf",
    "_validate_signal",
]
