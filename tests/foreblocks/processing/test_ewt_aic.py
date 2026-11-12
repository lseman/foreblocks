"""Regression coverage for ARMA-based EWT component selection."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from foreblocks.processing.transforms import ewt


def test_aic_selection_converts_arma_order_and_selects_best_component(monkeypatch):
    import statsmodels.tsa.stattools as stattools

    signal = np.linspace(0, 10, 40)
    imfs = np.column_stack([signal * 0.1, signal * 0.2, signal * 0.3])
    selected_orders = iter([(1, 2), (2, 1), (0, 0)])
    scores = iter([12.0, 3.0, 8.0])
    calls = []

    def select_order(residual, **kwargs):
        assert kwargs == dict(ic="aic", max_ar=2, max_ma=2)
        return {"aic_min_order": next(selected_orders)}

    def arima(residual, *, order):
        calls.append((residual.copy(), order))
        if len(order) != 3:
            raise ValueError("ARIMA requires (p, d, q)")
        score = next(scores)
        return SimpleNamespace(fit=lambda: SimpleNamespace(aic=score))

    monkeypatch.setattr(stattools, "arma_order_select_ic", select_order)
    monkeypatch.setattr(ewt.sm.tsa, "ARIMA", arima)

    assert ewt._select_best_imf_by_aic(signal, imfs) == 1
    assert [order for _, order in calls] == [(1, 0, 2), (2, 0, 1), (0, 0, 0)]
    for i, (residual, _) in enumerate(calls):
        np.testing.assert_allclose(residual, signal - imfs[:, i])
