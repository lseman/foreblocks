"""foreblocks.ts_handler.auto_filter.

Automatic time-series denoising with filter selection.

Provides a curated set of signal processing filters with automatic selection
based on signal characteristics. Supports wavelet, Kalman, lowess, Savitzky-
Golay, and deep learning-based denoising.

Core API:
- auto_filter: main auto-selection denoising function
- suggest_weights: heuristically suggest ScoringWeights from signal characteristics
- tune_weights: auto-tune ScoringWeights using Optuna
- tune_filter: search filter family and per-filter hyperparameters jointly with Optuna
- filter_metrics: compute seven quality metrics for a denoised series
- plot_results: three-panel plot of best filter / top-k comparison / ranking bar chart
- ScoringWeights: dataclass for filter scoring weights
- TuneFilterResult: dataclass for tune_filter results
- register_filter: register custom filters in the auto-selection registry

"""

from __future__ import annotations

from foreblocks.ts_handler.auto_filter.filters import (
    bilateral_filter,
    butter_lowpass,
    ceemdan_vmd_filter,
    gaussian_filter,
    gaussian_process_smoother,
    hp_filter,
    kalman_rts_smoother,
    l1_trend_filter,
    lowess_filter,
    non_local_means_filter,
    robust_loess_filter,
    savgol_filter,
    ssa_filter,
    stl_residual_denoise,
    train_dae,
    train_vae,
    tv_denoise,
    vmd_filter,
    wavelet_denoise,
    whittaker_smoother,
)
from foreblocks.ts_handler.auto_filter.heuristics import suggest_weights
from foreblocks.ts_handler.auto_filter.metrics import ScoringWeights, filter_metrics
from foreblocks.ts_handler.auto_filter.registry import register_filter
from foreblocks.ts_handler.auto_filter.runner import auto_filter
from foreblocks.ts_handler.auto_filter.tuning import TuneFilterResult, tune_filter, tune_weights
from foreblocks.ts_handler.auto_filter.visualization import plot_results

__all__ = [
    "auto_filter",
    "bilateral_filter",
    "butter_lowpass",
    "ceemdan_vmd_filter",
    "filter_metrics",
    "gaussian_filter",
    "gaussian_process_smoother",
    "hp_filter",
    "kalman_rts_smoother",
    "l1_trend_filter",
    "lowess_filter",
    "non_local_means_filter",
    "plot_results",
    "register_filter",
    "robust_loess_filter",
    "savgol_filter",
    "suggest_weights",
    "ssa_filter",
    "stl_residual_denoise",
    "ScoringWeights",
    "train_dae",
    "train_vae",
    "tune_filter",
    "tune_weights",
    "tv_denoise",
    "TuneFilterResult",
    "vmd_filter",
    "wavelet_denoise",
    "whittaker_smoother",
]
