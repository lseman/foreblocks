"""foreblocks.ts_handler.filters.

Time-series filtering and denoising toolkit.

Provides adaptive Savitzky-Golay, Kalman, lowess, Wiener, EMD, SSA, and
STL filters with robust NaN handling, parallel processing, and edge-case
safety. Supports both univariate and multivariate signals with configurable
pre-centering and deterministic execution.

Core API:
- adaptive_savgol_filter: edge-aware Savitzky-Golay with robust pre-centering
- kalman_filter: Kalman smoothing and filtering
- lowess_filter: locally weighted scatterplot smoothing
- wiener_filter: Wiener deconvolution
- emd_filter: Empirical Mode Decomposition denoising
- ssa_filter: Singular Spectrum Analysis
- stl_filter: STL-based trend-seasonal decomposition

"""

from __future__ import annotations

from foreblocks.ts_handler.filters.emd import emd_filter
from foreblocks.ts_handler.filters.kalman import kalman_filter
from foreblocks.ts_handler.filters.lowess import lowess_filter
from foreblocks.ts_handler.filters.savgol import adaptive_savgol_filter
from foreblocks.ts_handler.filters.ssa import ssa_filter
from foreblocks.ts_handler.filters.stl import stl_filter
from foreblocks.ts_handler.filters.wiener import wiener_filter

__all__ = [
    "adaptive_savgol_filter",
    "emd_filter",
    "kalman_filter",
    "lowess_filter",
    "ssa_filter",
    "stl_filter",
    "wiener_filter",
]
