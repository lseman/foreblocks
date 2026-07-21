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
from foreblocks.ts_handler.auto_filter.registry import (
    _FILTER_REGISTRY,
    _SLOW_FILTERS,
    register_filter,
)
from foreblocks.ts_handler.auto_filter import runner as _runner
from foreblocks.ts_handler.auto_filter import tuning as _tuning
from foreblocks.ts_handler.auto_filter.tuning import (
    _TUNE_FILTER_FAMILIES as _TUNE_FILTER_FAMILIES,
    _TUNE_FILTER_SLOW_FAMILIES as _TUNE_FILTER_SLOW_FAMILIES,
    TuneFilterResult,
)
from foreblocks.ts_handler.auto_filter.visualization import plot_results


def auto_filter(*args, **kwargs):
    _runner._FILTER_REGISTRY = _FILTER_REGISTRY
    _runner._SLOW_FILTERS = _SLOW_FILTERS
    _runner.filter_metrics = filter_metrics
    return _runner.auto_filter(*args, **kwargs)


def tune_weights(*args, **kwargs):
    _tuning._FILTER_REGISTRY = _FILTER_REGISTRY
    _tuning._SLOW_FILTERS = _SLOW_FILTERS
    return _tuning.tune_weights(*args, **kwargs)


def tune_filter(*args, **kwargs):
    _tuning._suggest_filter_and_params = _suggest_filter_and_params
    _tuning._run_parametrized_filter = _run_parametrized_filter
    _tuning.filter_metrics = filter_metrics
    return _tuning.tune_filter(*args, **kwargs)


_suggest_filter_and_params = _tuning._suggest_filter_and_params
_run_parametrized_filter = _tuning._run_parametrized_filter

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
