# ewt_compat_fixed_optimized_no_numba.py
# -----------------------------------------------------------------------------
# Optimized "Compatibility-first" EWT1D – matching original ewtpy / Gilles' 2018 behavior
#   without using Numba at all.
#
# Optimizations included:
# - Use np.fft.rfft for one-sided spectrum (faster + less memory)
# - Use scipy.signal.fftconvolve for "average" regularization (much faster than np.convolve for longer kernels)
# - Keep gaussian_filter for gaussian case (already efficient)
# - Fully vectorized LocalMax + improved LocalMaxMin (no slow per-segment Python loops where avoidable)
# - Avoid redundant .ravel() / copies when possible
# - Minor cleanups for clarity & micro-speed
# -----------------------------------------------------------------------------
from __future__ import annotations
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import fftconvolve
except ImportError as e:
    raise ImportError(
        "This implementation requires SciPy (gaussian_filter & fftconvolve). "
        "Install with: pip install scipy"
    ) from e


def EWT1D(
    f: np.ndarray,
    N: int = 5,
    log: int = 0,
    detect: str = "locmax",
    completion: int = 0,
    reg: str = "average",
    lengthFilter: int = 10,
    sigmaFilter: float = 5.0,
):
    """
    Returns
    -------
    ewt : (len(f), nbands) real ndarray
    mfb : (len(fMirr), nbands) ndarray – Fourier-domain filters
    boundaries : (nbounds,) ndarray in radians [0, pi]
                 with the same quirky mapping as reference
    """
    f = np.asarray(f, dtype=float).flatten()   # ensure 1D, float64
    n = len(f)
    if n < 2:
        raise ValueError("Input signal must have length >= 2.")
    N = int(N)
    if N < 1:
        raise ValueError("N must be >= 1.")

    # 1. One-sided magnitude (faster with rfft)
    ff = np.abs(np.fft.rfft(f))

    # 2. Boundary detection on one-sided spectrum
    boundaries_idx = EWT_Boundaries_Detect(
        ff,
        log=log,
        detect=detect,
        N=N,
        reg=reg,
        lengthFilter=lengthFilter,
        sigmaFilter=sigmaFilter,
    )

    # 3. Convert indices → radians (exact reference style)
    boundaries = boundaries_idx * np.pi / round(len(ff))

    if completion == 1 and len(boundaries) < N - 1:
        boundaries = EWT_Boundaries_Completion(boundaries, NT=N - 1)

    # 4. Mirror extension (exact reference)
    ltemp = (n + 1) // 2
    fMirr = np.r_[f[ltemp-1 :: -1], f, f[-2 : -ltemp-1 : -1]]
    ffMirr = np.fft.fft(fMirr)

    # 5. Meyer filter bank
    mfb = EWT_Meyer_FilterBank(boundaries, len(ffMirr))

    # 6. Filter & ifft (vectorized over bands)
    Y = np.conj(mfb) * ffMirr[:, None]
    ewt_full = np.fft.ifft(Y, axis=0).real

    # Crop back to original length (exact reference slicing)
    ewt = ewt_full[ltemp-1 : ltemp-1 + n, :]

    return ewt, mfb, boundaries


def EWT_Boundaries_Detect(
    ff: np.ndarray,
    log: int,
    detect: str,
    N: int,
    reg: str,
    lengthFilter: int,
    sigmaFilter: float,
):
    ff = np.asarray(ff, dtype=float).flatten()

    if log == 1:
        ff_log = np.log(ff + 1e-100)   # avoid -inf, close to reference
        presig_input = ff_log
    else:
        presig_input = ff

    # Regularization
    if reg == "average":
        if lengthFilter > 1:
            kernel = np.ones(lengthFilter, dtype=float) / lengthFilter
            presig = fftconvolve(presig_input, kernel, mode='same')
        else:
            presig = presig_input
    elif reg == "gaussian":
        presig = gaussian_filter(presig_input, sigma=sigmaFilter)
    else:
        presig = presig_input

    if detect == "locmax":
        bounds = LocalMax(presig, N)
    elif detect == "locmaxmin":
        bounds = LocalMaxMin(presig, N, fm=None)
    elif detect == "locmaxminf":
        bounds = LocalMaxMin(presig, N, fm=ff)
    else:
        raise ValueError("detect must be: 'locmax', 'locmaxmin', 'locmaxminf'")

    return bounds + 1.0   # the famous +1 shift from reference


def LocalMax(ff: np.ndarray, N: int) -> np.ndarray:
    ff = np.asarray(ff, dtype=float).flatten()
    nb = max(0, int(N) - 1)
    if nb == 0 or len(ff) < 3:
        return np.array([])

    # Strict local maxima
    is_max = (ff[2:] < ff[1:-1]) & (ff[:-2] < ff[1:-1])
    values = np.zeros_like(ff)
    values[1:-1] = ff[1:-1] * is_max.astype(float)

    # Top nb maxima (indices, descending strength)
    idx = np.argsort(values)[-nb:][::-1]
    idx = idx[np.nonzero(values[idx])[0]]  # remove zeros if any

    if len(idx) == 0:
        return np.array([])

    bounds = np.zeros(len(idx), dtype=float)
    prev = 0
    for i, cur in enumerate(idx):
        bounds[i] = (prev + cur) / 2.0
        prev = cur

    return bounds


def LocalMaxMin(f: np.ndarray, N: int, fm: np.ndarray | None = None) -> np.ndarray:
    f = np.asarray(f, dtype=float).flatten()
    if fm is None:
        fm = f
    else:
        fm = np.asarray(fm, dtype=float).flatten()

    if len(f) != len(fm) or len(f) < 3:
        return np.array([])

    nb = max(0, int(N) - 1)
    if nb == 0:
        return np.array([])

    # Strict local maxima on f
    is_max = (f[2:] < f[1:-1]) & (f[:-2] < f[1:-1])
    locmax = np.zeros_like(f)
    locmax[1:-1] = f[1:-1] * is_max.astype(float)

    # Top nb maxima indices (descending)
    Imax = np.argsort(locmax)[-nb:][::-1]
    Imax = Imax[locmax[Imax] > 0]  # remove invalid
    if len(Imax) == 0:
        return np.array([])

    # Strict local minima on fm
    is_min = (fm[2:] > fm[1:-1]) & (fm[:-2] > fm[1:-1])
    locmin = np.full_like(fm, np.inf)
    locmin[1:-1] = fm[1:-1] * is_min.astype(float)

    # Per interval – middle min (with tie handling)
    bounds = np.zeros(len(Imax), dtype=float)
    prev = 0

    for i, peak in enumerate(Imax):
        a = 1 if i == 0 else Imax[i-1] + 1
        b = peak + 1

        seg = locmin[a:b]
        if len(seg) == 0:
            bounds[i] = (a + b - 1) / 2.0
            prev = peak
            continue

        finite = np.isfinite(seg)
        if not finite.any():
            bounds[i] = (a + b - 1) / 2.0
            prev = peak
            continue

        seg_valid = seg[finite]
        idx_valid = np.flatnonzero(finite)

        order = np.argsort(seg_valid)
        mid = len(order) // 2
        chosen_rel = idx_valid[order[mid]]

        bounds[i] = a + chosen_rel - 1
        prev = peak

    return bounds


def EWT_Boundaries_Completion(boundaries: np.ndarray, NT: int) -> np.ndarray:
    boundaries = np.asarray(boundaries, dtype=float).flatten()
    if len(boundaries) == 0:
        return np.arange(1, NT + 1) * np.pi / (NT + 1)

    NT = int(NT)
    current = len(boundaries)
    if current >= NT:
        return boundaries[:NT]

    delta = (np.pi - boundaries[-1]) / (NT - current + 1)
    extra = boundaries[-1] + delta * np.arange(1, NT - current + 2)
    return np.r_[boundaries, extra[1:]]


def EWT_Meyer_FilterBank(boundaries: np.ndarray, Nsig: int) -> np.ndarray:
    boundaries = np.asarray(boundaries, dtype=float).flatten()
    Npic = len(boundaries)

    # Gamma computation (exact reference logic)
    gamma = 1.0
    for k in range(Npic - 1):
        r = (boundaries[k + 1] - boundaries[k]) / (boundaries[k + 1] + boundaries[k])
        gamma = min(gamma, r)
    r_last = (np.pi - boundaries[-1]) / (np.pi + boundaries[-1])
    gamma = min(gamma, r_last)
    gamma *= (1.0 - 1.0 / Nsig)

    mfb = np.zeros((Nsig, Npic + 1), dtype=float)

    # Frequency grid (exact reference style)
    w = np.fft.fftshift(np.linspace(0, 2 * np.pi * (1 - 1/Nsig), Nsig))
    Mi = Nsig // 2
    w[:Mi] -= 2 * np.pi
    aw = np.abs(w)

    # Scaling function (low-pass)
    an = 1 / (2 * gamma * boundaries[0])
    pbn = (1 + gamma) * boundaries[0]
    mbn = (1 - gamma) * boundaries[0]

    yms = np.where(aw <= mbn, 1.0,
           np.where(aw <= pbn, np.cos(np.pi/2 * EWT_beta(an * (aw - mbn))), 0.0))

    mfb[:, 0] = np.fft.ifftshift(yms)

    # Wavelets
    for k in range(Npic - 1):
        mfb[:, k + 1] = EWT_Meyer_Wavelet(boundaries[k], boundaries[k + 1], gamma, Nsig, aw, w)
    mfb[:, Npic] = EWT_Meyer_Wavelet(boundaries[-1], np.pi, gamma, Nsig, aw, w)

    return mfb


def EWT_beta(x: float | np.ndarray) -> float | np.ndarray:
    """Vectorized beta polynomial"""
    x = np.asarray(x)
    y = np.zeros_like(x)
    mask = (x >= 0) & (x <= 1)
    xm = x[mask]
    xm2 = xm * xm
    xm3 = xm2 * xm
    xm4 = xm3 * xm
    y[mask] = xm4 * (35.0 - 84.0 * xm + 70.0 * xm2 - 20.0 * xm3)
    y[x > 1] = 1.0
    return y


def EWT_Meyer_Wavelet(wn: float, wm: float, gamma: float, Nsig: int, aw: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Precompute aw & w outside if calling many times (but usually called few times)"""
    an = 1.0 / (2.0 * gamma * wn)
    am = 1.0 / (2.0 * gamma * wm)
    pbn = (1.0 + gamma) * wn
    mbn = (1.0 - gamma) * wn
    pbm = (1.0 + gamma) * wm
    mbm = (1.0 - gamma) * wm

    ymw = np.zeros(Nsig, dtype=float)

    # flat transition + flat passband + small transition
    ymw[(aw >= pbn) & (aw <= mbm)] = 1.0

    trans_high = (aw >= mbm) & (aw <= pbm)
    ymw[trans_high] = np.cos(np.pi / 2 * EWT_beta(am * (aw[trans_high] - mbm)))

    trans_low = (aw >= mbn) & (aw <= pbn)
    ymw[trans_low] = np.sin(np.pi / 2 * EWT_beta(an * (aw[trans_low] - mbn)))

    return np.fft.ifftshift(ymw)