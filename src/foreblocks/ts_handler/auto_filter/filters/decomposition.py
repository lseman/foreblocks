"""foreblocks.ts_handler.auto_filter.filters.decomposition.

Seasonal/decomposition filters: STL residual wavelet, VMD, CEEMDAN+VMD.

"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from foreblocks.ts_handler.auto_filter.filters.utils import (
    _as_series,
    _resize_to_match_length,
    _valid_odd_window,
)
from foreblocks.ts_handler.auto_filter.filters.wavelet import wavelet_denoise
from foreblocks.ts_handler.auto_filter.registry import register_filter

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:  # pragma: no cover - statsmodels ships this in supported envs.
    STL = None

try:
    from PyEMD import CEEMDAN, EMD
except ImportError:  # pragma: no cover - exercised when dependency is absent.
    CEEMDAN = None
    EMD = None

try:
    from vmdpy import VMD
except ImportError:  # pragma: no cover - exercised when dependency is absent.
    VMD = None

_SEED = 42


@register_filter("STL Residual Wavelet")
def stl_residual_denoise(
    ts: pd.Series,
    period: int = 24,
    seasonal: int = 13,
    resid_levels: int = 2,
    cycle_spins: int = 3,
    robust: bool = True,
) -> pd.Series:
    if STL is None:
        warnings.warn(
            "statsmodels STL is unavailable; falling back to wavelet denoising.",
            stacklevel=2,
        )
        return wavelet_denoise(ts, levels=resid_levels, cycle_spins=cycle_spins)

    y = ts.values.astype(float)
    if len(y) < max(8, period * 2):
        return wavelet_denoise(ts, levels=resid_levels, cycle_spins=cycle_spins)

    period = int(np.clip(period, 2, max(2, len(y) // 2)))
    seasonal = _valid_odd_window(len(y), int(seasonal), minimum=7)
    if seasonal <= period and seasonal % 2 == 0:
        seasonal += 1

    fit = STL(y, period=period, seasonal=seasonal, robust=robust).fit()
    resid = pd.Series(fit.resid, index=ts.index, name=ts.name)
    resid_clean = wavelet_denoise(
        resid,
        levels=resid_levels,
        cycle_spins=cycle_spins,
    ).values
    return _as_series(
        fit.trend + fit.seasonal + resid_clean, ts.index, name="stl_wavelet"
    )


@register_filter("VMD", slow=True)
def vmd_filter(
    ts: pd.Series,
    K: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    drop_modes: int = 1,
) -> pd.Series:
    compat_module = sys.modules.get("foreblocks.ts_handler.auto_filter")
    vmd_fn = getattr(compat_module, "VMD", VMD)
    if vmd_fn is None:
        warnings.warn(
            "vmdpy is unavailable; falling back to wavelet denoising.",
            stacklevel=2,
        )
        return wavelet_denoise(ts)

    x = ts.values.astype(float)
    grand_mean = float(np.mean(x))
    x_centered = x - grand_mean
    K = int(np.clip(K, 2, min(8, max(2, len(ts) // 8))))
    drop_modes = int(np.clip(drop_modes, 1, K - 1))

    u, _, omega = vmd_fn(x_centered, alpha, tau, K, DC=0, init=1, tol=tol)
    final_omega = omega[-1]
    drop_idx = np.argsort(final_omega)[-drop_modes:]
    keep_mask = np.ones(K, dtype=bool)
    keep_mask[drop_idx] = False
    recon = _resize_to_match_length(np.sum(u[keep_mask], axis=0), len(ts))
    return _as_series(recon + grand_mean, ts.index, name="vmd")


@register_filter("CEEMDAN+VMD", slow=True)
def ceemdan_vmd_filter(
    ts: pd.Series,
    trials: int = 50,
    epsilon: float = 0.005,
    K: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
) -> pd.Series:
    x = ts.values.astype(float)
    grand_mean = np.mean(x)
    x_centered = x - grand_mean
    compat_module = sys.modules.get("foreblocks.ts_handler.auto_filter")
    ceemdan_cls = getattr(compat_module, "CEEMDAN", CEEMDAN)
    emd_cls = getattr(compat_module, "EMD", EMD)
    vmd_fn = getattr(compat_module, "VMD", VMD)

    # --- Stage 1: CEEMDAN ---
    try:
        ceemdan = ceemdan_cls(trials=trials, epsilon=epsilon)
        ceemdan.noise_seed(_SEED)
        imfs = ceemdan(x_centered)
        if imfs.ndim == 1:
            ceemdan_recon = imfs.copy()
        elif imfs.shape[0] >= 2:
            ceemdan_recon = np.sum(imfs[1:], axis=0)
        else:
            ceemdan_recon = imfs[0].copy()
    except Exception as exc:
        warnings.warn(
            f"CEEMDAN failed ({exc}); falling back to EMD.",
            stacklevel=2,
        )
        try:
            emd = emd_cls()
            imfs = emd.emd(x_centered)
            ceemdan_recon = (
                np.sum(imfs[1:], axis=0) if imfs.shape[0] >= 2 else imfs[0].copy()
            )
        except Exception:
            ceemdan_recon = moving_average(
                _as_series(x_centered, ts.index), window=9
            ).values

    # --- Stage 2: VMD ---
    try:
        u, _, omega = vmd_fn(ceemdan_recon, alpha, tau, K, DC=0, init=1, tol=tol)
        final_omega = omega[-1]
        highest_freq_idx = int(np.argmax(final_omega))
        keep_mask = np.ones(K, dtype=bool)
        keep_mask[highest_freq_idx] = False
        vmd_recon = _resize_to_match_length(np.sum(u[keep_mask], axis=0), len(ts))
    except Exception as exc:
        warnings.warn(
            f"VMD stage failed ({exc}); using CEEMDAN reconstruction directly.",
            stacklevel=2,
        )
        vmd_recon = ceemdan_recon

    return _as_series(vmd_recon + grand_mean, ts.index, name="ceemdan_vmd")


# Fallback for moving_average used in ceemdan_vmd_filter
def moving_average(ts: pd.Series, window: int = 7) -> pd.Series:
    from foreblocks.ts_handler.auto_filter.filters.utils import _valid_odd_window

    window = _valid_odd_window(len(ts), window, minimum=3)
    return ts.rolling(window=window, center=True, min_periods=1).mean()
