"""foretools.arima - State-space SARIMAX with JAX autodiff and NumPy acceleration.

Provides seasonal ARIMA (SARIMAX) modeling via Kalman filter state-space estimation,
with optional JAX autodiff for gradient-based optimization and NumPy JIT-accelerated
computation paths.

Features
--------
- **SARIMAX**: Full seasonal specification (p,d,q)(P,D,Q,s) with exogenous regressors
- **JAX autodiff**: Gradient computation via ``jax.value_and_grad`` (requires ``jax`` + optional ``optax``)
- **NumPy acceleration**: JIT-compiled Kalman filter and state-space construction (requires ``numba``)
- **Auto-search**: Hyndman–Khandakar stepwise model selection with AICc criterion
- **Forecasting**: Analytical prediction intervals via covariance propagation

Quick Start
-----------
.. code-block:: python

    from foretools.arima import SarimaxSpec, SarimaxScratch, AutoConfig, auto_sarimax_stepwise

    # Manual fit
    spec = SarimaxSpec(order=(1, 1, 1), seasonal_order=(1, 0, 0, 7))
    model = SarimaxScratch(spec)
    fit = model.fit(y_data, exog=X_data)
    forecast = model.forecast(y_data, steps=7, exog_future=X_future)

    # Auto-search
    fit = auto_sarimax_stepwise(y_data, seasonal_period=7)

Dependencies
------------
- Required: NumPy, SciPy
- Optional: JAX (autodiff gradients), Optax (L-BFGS optimizer), Numba (JIT acceleration)

Public API
----------
- **SarimaxSpec**: Frozen dataclass specifying the ARIMA model structure
- **SarimaxFit**: Immutable result of fitting a SARIMAX model
- **SarimaxScratch**: State-space SARIMAX estimator (fit, filter, forecast)
- **AutoConfig**: Configuration for stepwise auto-search
- **auto_sarimax_stepwise**: Hyndman–Khandakar style automated model selection

Classes
-------
.. autosummary::
   SarimaxSpec
   SarimaxFit
   SarimaxScratch
   AutoConfig
"""

from __future__ import annotations

__all__ = [
    # Core dataclasses
    "SarimaxSpec",
    "SarimaxFit",
    "AutoConfig",
    # Main estimator class
    "SarimaxScratch",
    # Public functions
    "auto_sarimax_stepwise",
    "difference",
    "difference_exog",
    "future_difference_exog",
    "aicc",
]

# Import all public symbols from the implementation module.
# These are re-exported so users can do:
#   from foretools.arima import SarimaxSpec, auto_sarimax_stepwise
from .arima import (  # noqa: F401
    AutoConfig,
    SarimaxFit,
    SarimaxScratch,
    SarimaxSpec,
    auto_sarimax_stepwise,
    aicc,
    difference,
    difference_exog,
    future_difference_exog,
)
