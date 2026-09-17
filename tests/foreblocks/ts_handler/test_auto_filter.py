import numpy as np
import pandas as pd

from foreblocks.ts_handler import auto_filter as af


def _as_array(weights: af.ScoringWeights) -> np.ndarray:
    return np.array(
        [
            weights.fidelity_mse,
            weights.gcv,
            weights.roughness,
            weights.residual_autocorr,
            weights.spectral_distance,
            weights.residual_iid,
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


def test_suggest_weights_protect_level_shifts() -> None:
    rng = np.random.default_rng(456)
    step = np.r_[np.zeros(96), np.full(96, 4.0)]
    series = pd.Series(step + 0.08 * rng.normal(size=step.size))

    weights = af.suggest_weights(series)

    assert weights.derivative_corr > weights.residual_autocorr
    assert weights.roughness < weights.derivative_corr


def test_suggest_weights_prioritizes_smoothing_for_white_noise() -> None:
    rng = np.random.default_rng(789)
    series = pd.Series(rng.normal(size=256))

    weights = af.suggest_weights(series)

    assert weights.fidelity_mse < 0.30
    assert weights.roughness + weights.residual_autocorr > weights.derivative_corr


def test_suggest_weights_force_majority_smoothing_share_for_spiky_series() -> None:
    rng = np.random.default_rng(987)
    base = rng.normal(scale=0.4, size=256)
    spikes = np.zeros(256)
    spikes[::17] = rng.normal(loc=0.0, scale=5.0, size=len(spikes[::17]))
    series = pd.Series(base + spikes).diff().fillna(0.0)

    weights = af.suggest_weights(series)

    # Smoothing criteria stay in the majority on this very noisy / spike-heavy
    # input, but with a softer cap (was 0.60 / 0.15 / 0.20) so the auto-selection
    # cannot collapse the signal toward the mean when the chosen filter has no
    # parametric tuning.
    assert weights.roughness + weights.residual_autocorr >= 0.40
    assert weights.fidelity_mse <= 0.22
    assert weights.derivative_corr >= 0.10


def test_suggest_weights_can_explain_recommendation() -> None:
    rng = np.random.default_rng(246)
    t = np.linspace(0.0, 6.0 * np.pi, 192)
    series = pd.Series(np.sin(t) + 0.55 * rng.normal(size=t.size))

    weights, explanation = af.suggest_weights(series, explain=True)

    assert isinstance(weights, af.ScoringWeights)
    assert set(explanation["weights"]) == {
        "fidelity_mse",
        "gcv",
        "roughness",
        "residual_autocorr",
        "spectral_distance",
        "residual_iid",
        "derivative_corr",
    }
    assert {"noise", "periodicity", "trend", "memory"} <= set(
        explanation["diagnostics"]
    )
    assert {"structure", "smoothing_pressure", "shape_pressure"} <= set(
        explanation["derived"]
    )
    assert explanation["reasons"]
    assert all(isinstance(reason, str) for reason in explanation["reasons"])


def test_filter_metrics_returns_seven_axes() -> None:
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 4.0 * np.pi, 256)
    original = pd.Series(np.sin(t) + 0.3 * rng.normal(size=t.size))
    denoised = pd.Series(np.sin(t))

    metrics = af.filter_metrics(denoised, original)

    assert set(metrics) == {
        "fidelity_mse",
        "gcv",
        "roughness",
        "residual_autocorr",
        "spectral_distance",
        "residual_iid",
        "derivative_corr",
    }
    assert all(np.isfinite(v) for v in metrics.values())
    # Wavelet-grade denoising of pure-Gaussian-noisy sin should be inside
    # plausible ranges for each axis.
    assert metrics["fidelity_mse"] > 0
    assert metrics["spectral_distance"] >= 0
    assert 0.0 <= metrics["derivative_corr"] <= 1.0


def test_suggest_weights_emphasize_spectral_for_periodic_series() -> None:
    rng = np.random.default_rng(55)
    t = np.linspace(0.0, 12.0 * np.pi, 1024)
    seasonal = np.sin(t) + 0.6 * np.sin(3.0 * t)
    series = pd.Series(seasonal + 0.15 * rng.normal(size=t.size))

    weights = af.suggest_weights(series)

    # A strongly periodic signal should put more weight on spectral_distance
    # and derivative_corr than on raw roughness.
    assert weights.spectral_distance > 0.10
    assert weights.spectral_distance + weights.derivative_corr > weights.roughness


def test_registry_roster_is_pruned_and_extended() -> None:
    names = set(af._FILTER_REGISTRY)
    # Pruned filters must be gone.
    assert {
        "Moving Average",
        "Exponential Smoothing",
        "FFT Denoising",
        "EMD+VMD Baseline",
    }.isdisjoint(names)
    # Newer/stronger filters must be registered.
    assert {
        "Robust LOESS",
        "Whittaker-Eilers",
        "L1 Trend Filter",
        "Gaussian Process",
        "Non-local Means 1D",
        "Hodrick-Prescott",
        "Variational Autoencoder",
    } <= names


def test_tune_filter_family_roster_includes_stronger_candidates() -> None:
    assert {"gp", "nlm", "hp", "kalman", "stl", "vmd"} <= set(af._TUNE_FILTER_FAMILIES)
    assert {"ceemdan_vmd", "vae"} <= set(af._TUNE_FILTER_SLOW_FAMILIES)


def test_new_filters_preserve_length_and_smooth() -> None:
    rng = np.random.default_rng(7)
    t = np.arange(200, dtype=float)
    clean = 0.01 * t + np.sin(2 * np.pi * t / 24)
    series = pd.Series(clean + 0.3 * rng.normal(size=t.size))

    for fn in (af.whittaker_smoother, af.l1_trend_filter, af.robust_loess_filter):
        out = fn(series)
        assert len(out) == len(series)
        assert np.isfinite(out.to_numpy()).all()
        # Each should reduce roughness (std of first differences) vs the input.
        assert np.std(np.diff(out.to_numpy())) < np.std(np.diff(series.to_numpy()))


def test_wavelet_uses_real_basis_when_available() -> None:
    rng = np.random.default_rng(13)
    t = np.linspace(0.0, 8.0 * np.pi, 512)
    series = pd.Series(np.sin(t) + 0.3 * rng.normal(size=t.size))

    out = af.wavelet_denoise(series, wavelet="sym8")
    assert len(out) == len(series)
    assert np.isfinite(out.to_numpy()).all()
    # Denoised output should track the underlying sine more closely than noise.
    assert float(series.corr(out)) > 0.8


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
            gcv=0.0,
            roughness=1.0,
            residual_autocorr=0.0,
            spectral_distance=0.0,
            residual_iid=0.0,
            derivative_corr=0.0,
        ),
        target_band={},
    )

    assert best_name == "flat"
    assert np.allclose(best_series.values, ts.mean())
    assert list(score_table.index) == ["flat", "identity"]
    assert "score" in score_table.columns


def test_auto_filter_parallel_matches_sequential_registry(monkeypatch) -> None:
    ts = pd.Series(np.sin(np.linspace(0.0, 2.0 * np.pi, 64)))

    def identity(series: pd.Series) -> pd.Series:
        return series.copy()

    def smooth(series: pd.Series) -> pd.Series:
        return series.rolling(window=5, center=True, min_periods=1).mean()

    monkeypatch.setattr(
        af, "_FILTER_REGISTRY", {"identity": identity, "smooth": smooth}
    )
    monkeypatch.setattr(af, "_SLOW_FILTERS", set())

    sequential = af.auto_filter(
        ts,
        fast=True,
        target_band={},
        n_jobs=1,
        use_mc_gcv=False,
    )
    parallel = af.auto_filter(
        ts,
        fast=True,
        target_band={},
        n_jobs=2,
        parallel_backend="thread",
        use_mc_gcv=False,
    )

    assert parallel[0] == sequential[0]
    assert parallel[2].index.tolist() == sequential[2].index.tolist()
    assert np.allclose(parallel[1].to_numpy(), sequential[1].to_numpy())


def test_auto_filter_progress_matches_sequential_registry(monkeypatch) -> None:
    ts = pd.Series(np.sin(np.linspace(0.0, 2.0 * np.pi, 64)))

    def identity(series: pd.Series) -> pd.Series:
        return series.copy()

    def smooth(series: pd.Series) -> pd.Series:
        return series.rolling(window=5, center=True, min_periods=1).mean()

    monkeypatch.setattr(
        af, "_FILTER_REGISTRY", {"identity": identity, "smooth": smooth}
    )
    monkeypatch.setattr(af, "_SLOW_FILTERS", set())

    sequential = af.auto_filter(
        ts,
        fast=True,
        target_band={},
        n_jobs=1,
        use_mc_gcv=False,
    )
    with_progress = af.auto_filter(
        ts,
        fast=True,
        target_band={},
        n_jobs=1,
        use_mc_gcv=False,
        progress=True,
    )

    assert with_progress[0] == sequential[0]
    assert with_progress[2].index.tolist() == sequential[2].index.tolist()
    assert np.allclose(with_progress[1].to_numpy(), sequential[1].to_numpy())


def test_vmd_based_filters_resize_off_by_one_reconstructions(monkeypatch) -> None:
    ts = pd.Series(np.linspace(-1.0, 1.0, 9))

    class DummyEMD:
        MAX_ITERATION = 0

        def emd(self, values: np.ndarray, max_imf: int | None = None) -> np.ndarray:
            return np.stack([0.1 * values, 0.9 * values], axis=0)

    class DummyCEEMDAN:  # noqa: D401 - test double
        def __init__(self, trials: int, epsilon: float) -> None:
            self.trials = trials
            self.epsilon = epsilon

        def noise_seed(self, seed: int) -> None:
            self.seed = seed

        def __call__(self, values: np.ndarray) -> np.ndarray:
            return np.stack([0.2 * values, 0.8 * values], axis=0)

    def fake_vmd(
        values: np.ndarray,
        alpha: float,
        tau: float,
        K: int,
        DC: int,
        init: int,
        tol: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        short = len(values) - 1
        modes = np.vstack([
            np.linspace(0.0, 0.2, short),
            np.linspace(0.2, 0.4, short),
            np.linspace(0.4, 0.6, short),
            np.linspace(0.6, 0.8, short),
        ])
        return (
            modes[:K],
            np.zeros((K, short)),
            np.tile(np.arange(K, dtype=float), (2, 1)),
        )

    monkeypatch.setattr(af, "EMD", DummyEMD, raising=False)
    monkeypatch.setattr(af, "CEEMDAN", DummyCEEMDAN, raising=False)
    monkeypatch.setattr(af, "VMD", fake_vmd, raising=False)

    ceemdan = af.ceemdan_vmd_filter(ts, K=4)

    assert len(ceemdan) == len(ts)
    assert np.isfinite(ceemdan.to_numpy()).all()


def test_tune_weights_guardrails_avoid_oversmoothed_winner(monkeypatch) -> None:
    rng = np.random.default_rng(2026)
    t = np.linspace(0.0, 6.0 * np.pi, 192)
    ts = pd.Series(np.sin(t) + 0.2 * rng.normal(size=t.size))

    def oversmoothed(series: pd.Series) -> pd.Series:
        return pd.Series(
            np.full(len(series), series.mean()), index=series.index, name="flat"
        )

    def balanced(series: pd.Series) -> pd.Series:
        values = pd.Series(series).rolling(window=5, center=True, min_periods=1).mean()
        return pd.Series(values.to_numpy(), index=series.index, name="balanced")

    monkeypatch.setattr(
        af, "_FILTER_REGISTRY", {"flat": oversmoothed, "balanced": balanced}
    )
    monkeypatch.setattr(af, "_SLOW_FILTERS", set())

    tuned = af.tune_weights(
        ts,
        n_trials=30,
        fast=True,
        seed=123,
        warm_start=True,
        min_derivative_corr=0.80,
        max_rel_mae=0.35,
        min_roughness_ratio=0.10,
    )
    best_name, _, _ = af.auto_filter(ts, fast=True, weights=tuned)

    assert best_name == "balanced"


def test_tune_filter_band_penalty_dominates_raw_metric_scale(monkeypatch) -> None:
    ts = pd.Series(np.sin(np.linspace(0.0, 4.0 * np.pi, 96)), name="signal")

    def fake_suggest(
        trial, families: tuple[str, ...], value_scale: float
    ) -> tuple[str, dict[str, float | int]]:
        if trial.number == 0:
            return "bad", {}
        return "good", {}

    def fake_run(
        name: str, params: dict[str, float | int], series: pd.Series
    ) -> pd.Series:
        if name == "bad":
            return pd.Series(np.zeros(len(series)), index=series.index, name=name)
        return pd.Series(series.to_numpy() * 0.95, index=series.index, name=name)

    def fake_metrics(
        denoised: pd.Series,
        original: pd.Series,
        max_lag: int = 20,
        filter_fn=None,
        *,
        use_mc_gcv: bool = True,
    ) -> dict[str, float]:
        if denoised.name == "bad":
            return {
                "fidelity_mse": 0.0,
                "gcv": 0.0,
                "roughness": 0.0,
                "residual_autocorr": 0.0,
                "spectral_distance": 0.0,
                "residual_iid": 0.0,
                "derivative_corr": 0.30,
            }
        return {
            "fidelity_mse": 0.0,
            "gcv": 0.0,
            "roughness": 0.0,
            "residual_autocorr": 1000.0,
            "spectral_distance": 1000.0,
            "residual_iid": 1000.0,
            "derivative_corr": 0.95,
        }

    monkeypatch.setattr(af, "_suggest_filter_and_params", fake_suggest)
    monkeypatch.setattr(af, "_run_parametrized_filter", fake_run)
    monkeypatch.setattr(af, "filter_metrics", fake_metrics)

    result = af.tune_filter(
        ts,
        n_trials=2,
        families=("bad", "good"),
        rel_mae_band=(0.02, 0.12),
        roughness_ratio_band=(0.35, 1.10),
        min_derivative_corr=0.90,
    )

    assert result.name == "good"
    assert result.band_penalty == 0.0
