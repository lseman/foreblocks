"""Contracts retained by the NumPy and pandas filtering adapters."""

import numpy as np
import pandas as pd
import pytest

from foreblocks.processing.auto_filter.filters.kalman_rts import kalman_rts_smoother
from foreblocks.processing.auto_filter.filters.lowess import (
    lowess_filter as series_lowess,
    robust_loess_filter,
)
from foreblocks.processing.filters.kalman import kalman_filter
from foreblocks.processing.filters.lowess import lowess_filter


def test_lowess_adapters_agree_and_preserve_missing_positions():
    x = np.sin(np.arange(48) / 5)
    x[[3, 24]] = np.nan
    original = x.copy()
    index = pd.date_range("2025-01-01", periods=len(x), freq="h")
    result = series_lowess(pd.Series(x, index=index), frac=0.3, it=2)
    np.testing.assert_allclose(
        result.to_numpy(), lowess_filter(x, frac=0.3, it=2)[:, 0], equal_nan=True
    )
    pd.testing.assert_index_equal(result.index, index)
    assert result.name == "lowess"
    np.testing.assert_array_equal(np.isnan(result), np.isnan(x))
    np.testing.assert_array_equal(x, original)


def test_robust_lowess_keeps_minimum_iteration_policy():
    ts = pd.Series(np.sin(np.arange(48) / 5))
    ts.iloc[20] = 100
    result = robust_loess_filter(ts, frac=0.3, it=0)
    expected = series_lowess(ts, frac=0.3, it=3).rename("robust_loess")
    pd.testing.assert_series_equal(result, expected)


def test_lowess_insufficient_data_returns_nan_per_column():
    x = np.column_stack([np.arange(20, dtype=float), np.full(20, np.nan)])
    x[:15, 1] = 1
    result = lowess_filter(x, min_points=16)
    assert np.isfinite(result[:, 0]).all()
    assert np.isnan(result[:, 1]).all()


def test_kalman_models_retain_distinct_numerical_results():
    x = np.array([0.0, 1.0, 4.0, 2.0, 3.0, 5.0])
    # Recorded from the original scalar and local-linear-trend implementations.
    np.testing.assert_allclose(
        kalman_filter(x, n_iter=1, min_points=1)[:, 0],
        [
            4.5067733676,
            3.1286977932,
            2.5505602831,
            2.3701942590,
            2.2809620256,
            2.2476895101,
        ],
    )
    index = pd.date_range("2025-01-01", periods=len(x), freq="h")
    result = kalman_rts_smoother(pd.Series(x, index=index), q=0.1, r=0.5)
    np.testing.assert_allclose(
        result.to_numpy(),
        [
            0.3380761298,
            1.2709232569,
            2.2522343013,
            2.8654153319,
            3.6552017287,
            4.5843416385,
        ],
    )
    pd.testing.assert_index_equal(result.index, index)
    assert result.name == "kalman_rts"


def test_array_kalman_preserves_nan_and_skips_insufficient_columns():
    x = np.column_stack([np.arange(32, dtype=float), np.full(32, np.nan)])
    x[4, 0] = np.nan
    x[0, 1] = 2
    original = x.copy()
    result = kalman_filter(x)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(x))
    np.testing.assert_array_equal(result[:, 1], x[:, 1])
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("length", [0, 1, 3])
def test_trend_kalman_short_series_is_unchanged(length):
    ts = pd.Series(np.arange(length, dtype=float), name="original")
    pd.testing.assert_series_equal(kalman_rts_smoother(ts), ts)
