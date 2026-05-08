import numpy as np
import pandas as pd

from foreblocks.ts_handler import auto_filter as af


def _as_array(weights: af.ScoringWeights) -> np.ndarray:
    return np.array(
        [
            weights.fidelity_mse,
            weights.roughness,
            weights.residual_autocorr,
            weights.derivative_corr,
        ],
        dtype=float,
    )


def test_suggest_weights_are_normalized_and_finite_for_degenerate_inputs() -> None:
    series = pd.Series([np.nan, 1.0, np.inf, 1.0, -np.inf, 1.0])

    weights = af.suggest_weights(series)
    values = _as_array(weights)

    assert np.all(np.isfinite(values))
    assert np.isclose(values.sum(), 1.0)
    assert np.all(values > 0.0)


def test_suggest_weights_relax_fidelity_for_noisy_series() -> None:
    rng = np.random.default_rng(123)
    t = np.linspace(0.0, 8.0 * np.pi, 256)
    clean = pd.Series(np.sin(t) + 0.03 * rng.normal(size=t.size))
    noisy = pd.Series(np.sin(t) + 1.0 * rng.normal(size=t.size))

    clean_weights = af.suggest_weights(clean)
    noisy_weights = af.suggest_weights(noisy)

    assert noisy_weights.fidelity_mse < clean_weights.fidelity_mse
    assert noisy_weights.roughness > clean_weights.roughness
    assert noisy_weights.residual_autocorr > clean_weights.residual_autocorr


def test_suggest_weights_preserve_shape_for_trended_series() -> None:
    rng = np.random.default_rng(321)
    t = np.arange(160, dtype=float)
    trended = pd.Series(0.08 * t + 0.15 * rng.normal(size=t.size))

    weights = af.suggest_weights(trended)

    assert weights.derivative_corr > weights.roughness
    assert weights.fidelity_mse > weights.residual_autocorr


def test_auto_filter_scores_custom_registry(monkeypatch) -> None:
    ts = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0])

    def identity(series: pd.Series) -> pd.Series:
        return series.copy()

    def flat(series: pd.Series) -> pd.Series:
        return pd.Series(np.full(len(series), series.mean()), index=series.index)

    monkeypatch.setattr(af, "_FILTER_REGISTRY", {"identity": identity, "flat": flat})
    monkeypatch.setattr(af, "_SLOW_FILTERS", set())

    best_name, best_series, score_table = af.auto_filter(
        ts,
        fast=True,
        weights=af.ScoringWeights(
            fidelity_mse=0.0,
            roughness=1.0,
            residual_autocorr=0.0,
            derivative_corr=0.0,
        ),
    )

    assert best_name == "flat"
    assert np.allclose(best_series.values, ts.mean())
    assert list(score_table.index) == ["flat", "identity"]
    assert "score" in score_table.columns
