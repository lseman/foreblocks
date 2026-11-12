"""Numerical contracts and optional statsmodels parity checks."""

import numpy as np
import pytest

from foreblocks.processing._statistics import acf, ljung_box, pacf
from foreblocks.processing.filters._lowess import _lowess_smooth


def test_known_statistics_and_degenerate_inputs():
    np.testing.assert_allclose(acf([1, 2, 3], 2), [1, 0, -0.5], atol=1e-15)
    np.testing.assert_allclose(pacf([1, 2, 3, 4, 5, 6], 1), [1, 0.5])
    q, p = ljung_box([1, 2, 3], [1, 2])
    np.testing.assert_allclose(q, [0, 3.75], atol=1e-15)
    assert p[0] == 1
    assert np.isnan(acf(np.ones(20), 4)).all()
    assert np.isnan(ljung_box(np.ones(20), 4)[0]).all()
    np.testing.assert_array_equal(pacf(np.ones(20), 0), [1])


def test_lowess_linear_missing_and_input_order():
    x = np.array([9.0, 0.0, 3.0, 1.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0])
    y = 3 * x + 2
    original = y.copy()
    y[3] = np.nan
    fitted = _lowess_smooth(y, x, frac=0.7, it=0, delta=2)
    valid = np.isfinite(y)
    np.testing.assert_allclose(fitted[valid], original[valid], atol=1e-12)
    assert np.isnan(fitted[3])
    np.testing.assert_array_equal(y[valid], original[valid])
    assert _lowess_smooth([], [], frac=0.5, it=0).size == 0
    np.testing.assert_array_equal(_lowess_smooth([2], [0], frac=0, it=2), [2])


@pytest.mark.parametrize("frac", [0, 0.08, 0.3, 1])
@pytest.mark.parametrize("it", [0, 2, 4])
@pytest.mark.parametrize("delta", [0, 0.5, 5, 1000])
def test_lowess_matches_statsmodels(frac, it, delta):
    reference = pytest.importorskip("statsmodels.nonparametric.smoothers_lowess")
    rng = np.random.default_rng(8)
    x = np.cumsum(rng.uniform(0.1, 2, 100))
    y = np.sin(x / 5) + rng.normal(0, 0.2, len(x))
    y[20] = 20
    y[3] = np.nan
    x[8] = np.nan
    permutation = rng.permutation(len(x))
    x, y = x[permutation], y[permutation]
    expected = reference.lowess(
        y, x, frac=frac, it=it, delta=delta, return_sorted=False
    )
    actual = _lowess_smooth(y, x, frac=frac, it=it, delta=delta)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("kind", ["constant", "linear", "duplicate"])
def test_lowess_degenerate_parity(kind):
    reference = pytest.importorskip("statsmodels.nonparametric.smoothers_lowess")
    x = np.arange(40, dtype=float)
    y = np.ones(40) if kind == "constant" else 2 * x + 3
    if kind == "duplicate":
        x //= 3
    with np.errstate(invalid="ignore", divide="ignore"):
        actual = _lowess_smooth(y, x, frac=0.3, it=2, delta=3)
        expected = reference.lowess(y, x, frac=0.3, it=2, delta=3, return_sorted=False)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("n", [31, 100, 257])
@pytest.mark.parametrize("fft", [True, False])
def test_statistics_match_statsmodels(n, fft):
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    diagnostic = pytest.importorskip("statsmodels.stats.diagnostic")
    x = np.random.default_rng(n).normal(size=n) + 5
    lags = min(12, n // 2 - 1)
    np.testing.assert_allclose(
        acf(x, lags, fft=fft), stattools.acf(x, nlags=lags, fft=fft), atol=1e-14
    )
    np.testing.assert_allclose(
        pacf(x, lags), stattools.pacf(x, nlags=lags, method="ywm"), atol=1e-14
    )
    for requested in [lags, [1, 4, lags]]:
        expected = diagnostic.acorr_ljungbox(x, lags=requested, return_df=True)
        q, p = ljung_box(x, requested)
        np.testing.assert_allclose(q, expected.lb_stat, atol=1e-13)
        np.testing.assert_allclose(p, expected.lb_pvalue, atol=1e-13)


@pytest.mark.parametrize("lags", [0, -1, 20, [1, 2.5], []])
def test_ljung_box_rejects_invalid_lags(lags):
    with pytest.raises(ValueError):
        ljung_box(np.arange(20), lags)


def test_lowess_and_statistics_without_statsmodels_or_numba():
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import importlib.abc
        import sys

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'statsmodels', 'numba'}:
                    raise ModuleNotFoundError(fullname)

        sys.meta_path.insert(0, BlockOptional())
        import numpy as np
        from foreblocks.processing._statistics import acf, pacf, ljung_box
        from foreblocks.processing.filters._lowess import _lowess_smooth

        x = np.arange(32, dtype=float)
        y = 2 * x + 3
        np.testing.assert_allclose(
            _lowess_smooth(y, x, frac=0.4, it=0, delta=3), y, atol=1e-12
        )
        assert acf(x, 4).shape == (5,)
        assert pacf(x, 4).shape == (5,)
        assert np.isfinite(ljung_box(x, 4)).all()
        assert 'statsmodels' not in sys.modules
        assert 'numba' not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
