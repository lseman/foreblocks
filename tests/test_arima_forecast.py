import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import foretools.arima.arima as arima
from foretools.arima.arima import (
    _as_2d,
    _invert_difference_forecast,
    difference,
    future_difference_exog,
)


def test_invert_difference_forecast_recovers_level_horizon() -> None:
    y = np.array([10.0, 11.0, 13.0, 16.0])
    diff_forecast = np.array([4.0, 5.0, 6.0])

    forecast = _invert_difference_forecast(y, diff_forecast, d=1, D=0, s=1)

    np.testing.assert_allclose(forecast, np.array([20.0, 25.0, 31.0]))
    np.testing.assert_allclose(
        difference(np.concatenate([y, forecast]), d=1, D=0, s=1)[-3:],
        diff_forecast,
    )


def test_invert_difference_forecast_recovers_seasonal_level_horizon() -> None:
    y = np.array([10.0, 20.0, 11.0, 21.0, 12.0, 22.0])
    diff_forecast = np.array([1.5, 2.5, 3.5])

    forecast = _invert_difference_forecast(y, diff_forecast, d=0, D=1, s=2)

    np.testing.assert_allclose(forecast, np.array([13.5, 24.5, 17.0]))
    np.testing.assert_allclose(
        difference(np.concatenate([y, forecast]), d=0, D=1, s=2)[-3:],
        diff_forecast,
    )


def test_future_difference_exog_uses_history_for_default_arimax_order() -> None:
    history = np.array([[1.0], [2.0], [4.0], [7.0]])
    future = np.array([[7.0], [7.0], [7.0]])

    diff_future = future_difference_exog(history, future, d=1, D=0, s=1)

    np.testing.assert_allclose(diff_future, np.array([[0.0], [0.0], [0.0]]))


def test_as_2d_accepts_clean_exog_without_interpolation() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    out = _as_2d(X, n=2)

    np.testing.assert_allclose(out, X)


def test_jax_stationary_transform_has_static_slices() -> None:
    pytest.importorskip("jax")
    if not arima._HAS_JAX:
        pytest.skip("foretools ARIMA imported without JAX support")

    jnp = arima.jnp
    value, grad = arima.jax.value_and_grad(
        lambda raw: jnp.sum(arima._jax_constrain_stationary(raw) ** 2)
    )(jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad)))
