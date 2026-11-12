"""Behavioral contracts for both SSA filter interfaces."""

import numpy as np
import pandas as pd
import pytest

from foreblocks.processing.auto_filter.filters.ssa import ssa_filter as series_ssa
from foreblocks.processing.filters.ssa import ssa_filter as array_ssa


def test_interfaces_agree_with_explicit_window_and_rank():
    x = np.random.default_rng(42).normal(size=64)
    index = pd.date_range("2025-01-01", periods=len(x), freq="h")
    result = series_ssa(pd.Series(x, index=index), window=12, n_components=3)
    np.testing.assert_allclose(
        result.to_numpy(), array_ssa(x, window_length=12, n_components=3)[:, 0]
    )
    pd.testing.assert_index_equal(result.index, index)
    assert result.name == "ssa"


def test_full_rank_reconstructs_multivariate_input_without_mutating_it():
    x = np.random.default_rng(7).normal(size=(48, 3))
    original = x.copy()
    result = array_ssa(x, window_length=10, n_components=10)
    np.testing.assert_allclose(result, original, atol=1e-12)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("fill_nans", [True, False])
def test_array_nan_policy(fill_nans):
    x = np.arange(48, dtype=float)
    x[[3, 20]] = np.nan
    result = array_ssa(x, window_length=8, fill_nans_for_filter=fill_nans)[:, 0]
    np.testing.assert_array_equal(np.isnan(result), np.isnan(x))
    if fill_nans:
        assert np.isfinite(result[~np.isnan(x)]).all()
    else:
        np.testing.assert_array_equal(result, x)


def test_short_inputs_keep_interface_specific_shapes_and_names():
    x = np.arange(6, dtype=float)
    ts = pd.Series(x, name="original")
    pd.testing.assert_series_equal(series_ssa(ts), ts)
    np.testing.assert_array_equal(array_ssa(x, window_length=20), x[:, None])


def test_rank_zero_keeps_interface_specific_policy():
    x = np.ones(32)
    np.testing.assert_array_equal(array_ssa(x, n_components=0), np.zeros((32, 1)))
    np.testing.assert_allclose(series_ssa(pd.Series(x), n_components=0), x)


def test_svd_failure_falls_back_only_for_array_interface(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD failed")

    monkeypatch.setattr(np.linalg, "svd", fail)
    x = np.arange(32, dtype=float)
    np.testing.assert_array_equal(array_ssa(x), x[:, None])
    with pytest.raises(np.linalg.LinAlgError, match="SVD failed"):
        series_ssa(pd.Series(x))
