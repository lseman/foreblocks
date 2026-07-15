"""foreblocks.ts_handler.auto_filter.filters.

Curated collection of signal processing filters for auto-selection.

Provides moving average, Gaussian, Savitzky-Golay, Butterworth lowpass,
Kalman, VMD, wavelet, CEEMDAN, STL, and other denoising methods. Each
filter is registered with the auto-selection registry and supports
both univariate and multivariate time series.

Core API:
- moving_average, gaussian_filter, savgol_filter, butter_lowpass: classic filters
- kalman_rts_smoother: Kalman fixed-interval RTS smoother
- wavelet_denoise, vmd_filter, ceemdan_vae_filter: decomposition-based filters
- lowess_filter, robust_loess_filter: nonparametric smoothers
- stl_residual_denoise: STL-based residual denoising

"""

from __future__ import annotations

from foreblocks.ts_handler.auto_filter.filters.bilateral import bilateral_filter
from foreblocks.ts_handler.auto_filter.filters.classical import (
    butter_lowpass,
    gaussian_filter,
    moving_average,
    savgol_filter,
)
from foreblocks.ts_handler.auto_filter.filters.decomposition import (
    ceemdan_vmd_filter,
    stl_residual_denoise,
    vmd_filter,
)
from foreblocks.ts_handler.auto_filter.filters.deep import train_dae, train_vae
from foreblocks.ts_handler.auto_filter.filters.kalman_rts import kalman_rts_smoother
from foreblocks.ts_handler.auto_filter.filters.lowess import (
    lowess_filter,
    robust_loess_filter,
)
from foreblocks.ts_handler.auto_filter.filters.penalized import (
    hp_filter,
    l1_trend_filter,
    whittaker_smoother,
)
from foreblocks.ts_handler.auto_filter.filters.smoothers import (
    gaussian_process_smoother,
    non_local_means_filter,
)
from foreblocks.ts_handler.auto_filter.filters.ssa import ssa_filter
from foreblocks.ts_handler.auto_filter.filters.tv import tv_denoise
from foreblocks.ts_handler.auto_filter.filters.utils import (
    _autocorr,
    _safe_corr,
    _valid_odd_window,
)
from foreblocks.ts_handler.auto_filter.filters.wavelet import wavelet_denoise

__all__ = [
    "_autocorr",
    "_safe_corr",
    "_valid_odd_window",
    "bilateral_filter",
    "butter_lowpass",
    "ceemdan_vmd_filter",
    "gaussian_filter",
    "gaussian_process_smoother",
    "hp_filter",
    "kalman_rts_smoother",
    "l1_trend_filter",
    "lowess_filter",
    "moving_average",
    "non_local_means_filter",
    "robust_loess_filter",
    "savgol_filter",
    "ssa_filter",
    "stl_residual_denoise",
    "train_dae",
    "train_vae",
    "tv_denoise",
    "vmd_filter",
    "wavelet_denoise",
    "whittaker_smoother",
]
