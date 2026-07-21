"""foreblocks.ts_handler.auto_filter.tuning.

Optuna-based tuning functions for filter weights and filter parameters.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

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
    savgol_filter,
    ssa_filter,
    stl_residual_denoise,
    train_vae,
    tv_denoise,
    vmd_filter,
    wavelet_denoise,
    whittaker_smoother,
)
from foreblocks.ts_handler.auto_filter.heuristics import suggest_weights
from foreblocks.ts_handler.auto_filter.metrics import ScoringWeights, filter_metrics
from foreblocks.ts_handler.auto_filter.registry import _FILTER_REGISTRY, _SLOW_FILTERS


@dataclass
class TuneFilterResult:
    name: str
    params: dict[str, float | int]
    series: pd.Series
    metrics: dict[str, float]
    rel_mae: float
    roughness_ratio: float
    objective_value: float
    band_penalty: float = 0.0


_TUNE_FILTER_FAMILIES = (
    "gaussian",
    "tv",
    "bilateral",
    "savgol",
    "butter",
    "wavelet",
    "lowess",
    "gp",
    "nlm",
    "hp",
    "kalman",
    "stl",
    "vmd",
    "ssa",
    "whittaker",
    "l1_trend",
)

_TUNE_FILTER_SLOW_FAMILIES = ("ceemdan_vmd", "vae")

_TUNE_FILTER_BAND_PENALTY_WEIGHT = 10.0


def _run_parametrized_filter(
    name: str, params: dict[str, float | int], ts: pd.Series
) -> pd.Series:
    if name == "gaussian":
        return gaussian_filter(ts, sigma=float(params["sigma"]))
    if name == "tv":
        return tv_denoise(ts, weight=float(params["weight"]))
    if name == "bilateral":
        return bilateral_filter(
            ts,
            sigma_t=float(params["sigma_t"]),
            sigma_v=float(params["sigma_v"]),
        )
    if name == "savgol":
        return savgol_filter(
            ts,
            window=int(params["window"]),
            polyorder=int(params["polyorder"]),
        )
    if name == "butter":
        return butter_lowpass(
            ts,
            cutoff=float(params["cutoff"]),
            order=int(params["order"]),
        )
    if name == "wavelet":
        return wavelet_denoise(
            ts,
            levels=int(params["levels"]),
            cycle_spins=int(params["cycle_spins"]),
            wavelet=str(params["wavelet"]),
        )
    if name == "lowess":
        return lowess_filter(ts, frac=float(params["frac"]), it=int(params["it"]))
    if name == "gp":
        return gaussian_process_smoother(
            ts,
            length_scale=float(params["length_scale"]),
            noise=float(params["noise"]),
            max_inducing=int(params["max_inducing"]),
        )
    if name == "nlm":
        return non_local_means_filter(
            ts,
            patch_radius=int(params["patch_radius"]),
            search_radius=int(params["search_radius"]),
            h=float(params["h"]),
        )
    if name == "hp":
        return hp_filter(ts, lamb=float(params["lamb"]))
    if name == "kalman":
        return kalman_rts_smoother(
            ts,
            q=float(params["q"]),
            r=float(params["r"]),
        )
    if name == "stl":
        return stl_residual_denoise(
            ts,
            period=int(params["period"]),
            seasonal=int(params["seasonal"]),
            resid_levels=int(params["resid_levels"]),
            cycle_spins=int(params["cycle_spins"]),
        )
    if name == "vmd":
        return vmd_filter(
            ts,
            K=int(params["K"]),
            alpha=float(params["alpha"]),
            tau=float(params["tau"]),
            drop_modes=int(params["drop_modes"]),
        )
    if name == "ceemdan_vmd":
        return ceemdan_vmd_filter(
            ts,
            trials=int(params["trials"]),
            epsilon=float(params["epsilon"]),
            K=int(params["K"]),
            alpha=float(params["alpha"]),
            tau=float(params["tau"]),
        )
    if name == "vae":
        return train_vae(
            ts,
            window=int(params["window"]),
            epochs=int(params["epochs"]),
            noise_std=float(params["noise_std"]),
            beta=float(params["beta"]),
            latent_size=int(params["latent_size"]),
        )
    if name == "ssa":
        return ssa_filter(
            ts,
            window=int(params["window"]),
            n_components=int(params["n_components"]),
        )
    if name == "whittaker":
        return whittaker_smoother(
            ts, lam=float(params["lam"]), order=int(params["order"])
        )
    if name == "l1_trend":
        return l1_trend_filter(ts, lam=float(params["lam"]))
    raise ValueError(f"Unknown filter family: {name}")


def _suggest_filter_and_params(
    trial: optuna.Trial,
    families: tuple[str, ...],
    value_scale: float,
) -> tuple[str, dict[str, float | int]]:
    name = trial.suggest_categorical("filter_name", list(families))

    if name == "gaussian":
        return name, {"sigma": trial.suggest_float("sigma", 0.6, 6.0, log=True)}
    if name == "tv":
        return name, {"weight": trial.suggest_float("weight", 0.03, 1.2, log=True)}
    if name == "bilateral":
        return name, {
            "sigma_t": trial.suggest_float("sigma_t", 1.0, 14.0),
            "sigma_v": trial.suggest_float(
                "sigma_v", 0.2 * value_scale, 2.5 * value_scale
            ),
        }
    if name == "savgol":
        window = trial.suggest_int("window", 5, 31, step=2)
        polyorder = trial.suggest_int("polyorder", 2, min(4, window - 1))
        return name, {"window": window, "polyorder": polyorder}
    if name == "butter":
        return name, {
            "cutoff": trial.suggest_float("cutoff", 0.02, 0.25),
            "order": trial.suggest_int("order", 2, 5),
        }
    if name == "wavelet":
        return name, {
            "levels": trial.suggest_int("levels", 1, 4),
            "cycle_spins": trial.suggest_int("cycle_spins", 1, 6),
            "wavelet": trial.suggest_categorical("wavelet", ["db4", "sym8", "coif3"]),
        }
    if name == "lowess":
        return name, {
            "frac": trial.suggest_float("frac", 0.02, 0.20),
            "it": trial.suggest_int("it", 0, 3),
        }
    if name == "gp":
        return name, {
            "length_scale": trial.suggest_float("gp_length_scale", 2.0, 96.0, log=True),
            "noise": trial.suggest_float("gp_noise", 0.01, 0.35, log=True),
            "max_inducing": trial.suggest_categorical(
                "gp_max_inducing", [96, 160, 256]
            ),
        }
    if name == "nlm":
        return name, {
            "patch_radius": trial.suggest_int("nlm_patch_radius", 1, 5),
            "search_radius": trial.suggest_int("nlm_search_radius", 12, 96, step=12),
            "h": trial.suggest_float("nlm_h", 0.05 * value_scale, 2.0 * value_scale),
        }
    if name == "hp":
        return name, {"lamb": trial.suggest_float("hp_lamb", 10.0, 1e5, log=True)}
    if name == "kalman":
        return name, {
            "q": trial.suggest_float("kalman_q", 1e-8, 1.0, log=True),
            "r": trial.suggest_float("kalman_r", 1e-6, 5.0, log=True),
        }
    if name == "stl":
        period = trial.suggest_categorical("stl_period", [12, 24, 48, 168])
        return name, {
            "period": int(period),
            "seasonal": trial.suggest_int("stl_seasonal", 7, 31, step=2),
            "resid_levels": trial.suggest_int("stl_resid_levels", 1, 3),
            "cycle_spins": trial.suggest_int("stl_cycle_spins", 1, 4),
        }
    if name == "vmd":
        return name, {
            "K": trial.suggest_int("vmd_K", 3, 7),
            "alpha": trial.suggest_float("vmd_alpha", 200.0, 8000.0, log=True),
            "tau": trial.suggest_float("vmd_tau", 0.0, 0.2),
            "drop_modes": trial.suggest_int("vmd_drop_modes", 1, 2),
        }
    if name == "ceemdan_vmd":
        return name, {
            "trials": trial.suggest_int("ceemdan_trials", 8, 32),
            "epsilon": trial.suggest_float("ceemdan_epsilon", 0.001, 0.02, log=True),
            "K": trial.suggest_int("ceemdan_vmd_K", 3, 6),
            "alpha": trial.suggest_float("ceemdan_vmd_alpha", 500.0, 8000.0, log=True),
            "tau": trial.suggest_float("ceemdan_vmd_tau", 0.0, 0.2),
        }
    if name == "vae":
        return name, {
            "window": trial.suggest_int("vae_window", 9, 41, step=2),
            "epochs": trial.suggest_int("vae_epochs", 8, 24),
            "noise_std": trial.suggest_float("vae_noise_std", 0.04, 0.25),
            "beta": trial.suggest_float("vae_beta", 0.002, 0.08, log=True),
            "latent_size": trial.suggest_int("vae_latent_size", 3, 12),
        }
    if name == "whittaker":
        return name, {
            "lam": trial.suggest_float("lam", 1.0, 1e5, log=True),
            "order": trial.suggest_int("order", 1, 3),
        }
    if name == "l1_trend":
        return name, {"lam": trial.suggest_float("l1_lam", 0.1, 50.0, log=True)}
    if name == "ssa":
        window = trial.suggest_int("ssa_window", 12, 120, step=4)
        n_components = trial.suggest_int("ssa_components", 1, 8)
        return name, {"window": window, "n_components": n_components}
    raise ValueError(f"Unknown filter family: {name}")


def _bounded_unit(value: float) -> float:
    value = max(float(value), 0.0)
    return value / (1.0 + value)


def _target_band_penalty_from_diagnostics(
    *,
    rel_mae: float,
    roughness_ratio: float,
    derivative_corr: float,
    rel_mae_band: tuple[float, float],
    roughness_ratio_band: tuple[float, float],
    min_derivative_corr: float,
) -> float:
    penalty = 0.0
    penalty += 4.0 * max(rel_mae_band[0] - rel_mae, 0.0)
    penalty += 4.0 * max(rel_mae - rel_mae_band[1], 0.0)
    penalty += 3.0 * max(roughness_ratio_band[0] - roughness_ratio, 0.0)
    penalty += 3.0 * max(roughness_ratio - roughness_ratio_band[1], 0.0)
    penalty += 4.0 * max(min_derivative_corr - derivative_corr, 0.0)
    return penalty


def _band_penalty(
    winner: pd.Series,
    original: pd.Series,
    *,
    target: dict[str, float],
) -> tuple[float, dict[str, float]]:
    o = original.values.astype(float)
    d = winner.values.astype(float)
    residual = o - d

    abs_mean = max(float(np.mean(np.abs(o))), 1e-12)
    rel_mae = float(np.mean(np.abs(residual))) / abs_mean

    orig_rough = max(float(np.std(np.diff(o))), 1e-12)
    win_rough = float(np.std(np.diff(d)))
    roughness_ratio = win_rough / orig_rough

    derivative_corr = abs(_safe_corr(np.diff(d), np.diff(o)))

    penalty = _target_band_penalty_from_diagnostics(
        rel_mae=rel_mae,
        roughness_ratio=roughness_ratio,
        derivative_corr=derivative_corr,
        rel_mae_band=(target["rel_mae_min"], target["rel_mae_max"]),
        roughness_ratio_band=(
            target["roughness_ratio_min"],
            target["roughness_ratio_max"],
        ),
        min_derivative_corr=target["derivative_corr_min"],
    )

    diagnostics = {
        "rel_mae": rel_mae,
        "roughness_ratio": roughness_ratio,
        "derivative_corr": derivative_corr,
    }
    return penalty, diagnostics


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def _unsupervised_proxy(
    winner: pd.Series,
    original: pd.Series,
    candidates: dict[str, pd.Series],
) -> float:
    lags = min(20, len(original) // 5)
    lb_stats: list[float] = []

    for s in candidates.values():
        resid = (original - s).values.astype(float)
        try:
            lb_stats.append(_ljung_box_stat(resid, lags=lags))
        except Exception:
            lb_stats.append(float("nan"))

    names = list(candidates.keys())
    winner_name = winner.name
    if winner_name not in names:
        for i, s in enumerate(candidates.values()):
            if np.allclose(s.values, winner.values, atol=1e-10):
                winner_idx = i
                break
        else:
            winner_idx = 0
    else:
        winner_idx = names.index(winner_name)

    def _rank_norm_list(vals: list[float]) -> list[float]:
        arr = np.array(vals, dtype=float)
        finite = np.isfinite(arr)
        if finite.sum() < 2:
            return [0.5] * len(vals)
        ranks = np.empty(len(arr))
        order = np.argsort(arr[finite])
        finite_indices = np.where(finite)[0]
        for rank, fi in enumerate(order):
            ranks[finite_indices[fi]] = rank / (finite.sum() - 1)
        ranks[~finite] = 1.0
        return ranks.tolist()

    lb_norm = _rank_norm_list(lb_stats)
    from foreblocks.ts_handler.auto_filter.runner import _DEFAULT_TARGET_BAND

    band_pen, _ = _band_penalty(winner, original, target=_DEFAULT_TARGET_BAND)
    return 0.5 * lb_norm[winner_idx] + band_pen


def _ljung_box_stat(residual: np.ndarray, lags: int) -> float:
    if len(residual) < lags + 2:
        return 0.0
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(residual, lags=lags, return_df=True)
        return float(lb["lb_stat"].iloc[-1])
    except Exception:
        from foreblocks.ts_handler.auto_filter.metrics import _autocorr

        return float(
            np.mean([abs(_autocorr(residual, lag)) for lag in range(1, lags + 1)])
        )


def _oversmoothing_penalty(
    winner: pd.Series,
    original: pd.Series,
    metrics: pd.Series,
    *,
    min_derivative_corr: float | None = None,
    min_rel_mae: float | None = None,
    max_rel_mae: float | None = None,
    min_roughness_ratio: float | None = None,
    max_roughness_ratio: float | None = None,
) -> float:
    penalty = 0.0

    if min_derivative_corr is not None:
        penalty += 4.0 * max(
            min_derivative_corr - float(metrics["derivative_corr"]), 0.0
        )

    rel_mae = float(np.mean(np.abs((original - winner).values.astype(float)))) / max(
        float(np.mean(np.abs(original.values.astype(float)))), 1e-12
    )
    if max_rel_mae is not None:
        penalty += 3.0 * max(rel_mae - max_rel_mae, 0.0)
    if min_rel_mae is not None:
        penalty += 3.0 * max(min_rel_mae - rel_mae, 0.0)

    original_roughness = float(np.std(np.diff(original.values.astype(float))))
    winner_roughness = float(np.std(np.diff(winner.values.astype(float))))
    roughness_ratio = winner_roughness / max(original_roughness, 1e-12)
    if min_roughness_ratio is not None:
        penalty += 2.5 * max(min_roughness_ratio - roughness_ratio, 0.0)
    if max_roughness_ratio is not None:
        penalty += 2.5 * max(roughness_ratio - max_roughness_ratio, 0.0)

    return penalty


def tune_weights(
    ts: pd.Series,
    n_trials: int = 100,
    fast: bool = True,
    seed: int = 42,
    verbose: bool = False,
    warm_start: bool = True,
    min_derivative_corr: float | None = 0.90,
    min_rel_mae: float | None = 0.02,
    max_rel_mae: float | None = 0.12,
    min_roughness_ratio: float | None = 0.35,
    max_roughness_ratio: float | None = 0.92,
) -> ScoringWeights:
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    active = {
        name: fn
        for name, fn in _FILTER_REGISTRY.items()
        if not (fast and name in _SLOW_FILTERS)
    }

    candidates: dict[str, pd.Series] = {}
    for name, fn in active.items():
        try:
            out = fn(ts).rename(name)
            candidates[name] = out
        except Exception as exc:
            warnings.warn(f"tune_weights: filter '{name}' failed: {exc}", stacklevel=2)

    if len(candidates) < 2:
        raise RuntimeError("Fewer than 2 filters succeeded; cannot tune weights.")

    metrics_rows = {
        name: filter_metrics(series, ts, filter_fn=active.get(name))
        for name, series in candidates.items()
    }
    mdf_base = pd.DataFrame(metrics_rows).T

    n_weights = 7

    def _weights_from_raw(raw: np.ndarray) -> ScoringWeights:
        return ScoringWeights(
            fidelity_mse=float(raw[0]),
            gcv=float(raw[1]),
            roughness=float(raw[2]),
            residual_autocorr=float(raw[3]),
            spectral_distance=float(raw[4]),
            residual_iid=float(raw[5]),
            derivative_corr=float(raw[6]),
        )

    def objective(trial: optuna.Trial) -> float:
        logits = np.array(
            [trial.suggest_float(f"w{i}", 0.01, 1.0) for i in range(n_weights)]
        )
        logits /= logits.sum()
        w = _weights_from_raw(logits)
        scores = _compute_scores(mdf_base, w)
        best_name = str(scores.idxmin())
        winner = candidates[best_name]
        base_value = _unsupervised_proxy(winner, ts, candidates)
        penalty = _oversmoothing_penalty(
            winner,
            ts,
            mdf_base.loc[best_name],
            min_derivative_corr=min_derivative_corr,
            min_rel_mae=min_rel_mae,
            max_rel_mae=max_rel_mae,
            min_roughness_ratio=min_roughness_ratio,
            max_roughness_ratio=max_roughness_ratio,
        )
        return base_value + penalty

    from foreblocks.ts_handler.auto_filter.metrics import _compute_scores

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    if warm_start:
        sw = suggest_weights(ts)
        study.enqueue_trial(
            {
                "w0": sw.fidelity_mse,
                "w1": sw.gcv,
                "w2": sw.roughness,
                "w3": sw.residual_autocorr,
                "w4": sw.spectral_distance,
                "w5": sw.residual_iid,
                "w6": sw.derivative_corr,
            }
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best = study.best_params
    raw = np.array([best[f"w{i}"] for i in range(n_weights)], dtype=float)
    raw /= raw.sum()
    tuned = _weights_from_raw(raw)

    if verbose:
        print(f"\nTuned ScoringWeights (n_trials={n_trials}):")
        print(f"  fidelity_mse      = {tuned.fidelity_mse:.4f}")
        print(f"  gcv               = {tuned.gcv:.4f}")
        print(f"  roughness         = {tuned.roughness:.4f}")
        print(f"  residual_autocorr = {tuned.residual_autocorr:.4f}")
        print(f"  spectral_distance = {tuned.spectral_distance:.4f}")
        print(f"  residual_iid      = {tuned.residual_iid:.4f}")
        print(f"  derivative_corr   = {tuned.derivative_corr:.4f}")
        print(f"  best trial value  = {study.best_value:.6f}")

    return tuned


def tune_filter(
    ts: pd.Series,
    n_trials: int = 60,
    seed: int = 42,
    verbose: bool = False,
    progress: bool = False,
    families: tuple[str, ...] = _TUNE_FILTER_FAMILIES,
    rel_mae_band: tuple[float, float] = (0.02, 0.12),
    roughness_ratio_band: tuple[float, float] = (0.35, 0.92),
    min_derivative_corr: float = 0.90,
) -> TuneFilterResult:
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    x = np.asarray(ts.to_numpy(), dtype=float)
    abs_mean = max(float(np.mean(np.abs(x))), 1e-12)
    orig_roughness = max(float(np.std(np.diff(x))), 1e-12)
    value_scale = float(np.median(np.abs(x - np.median(x)))) + 1e-6

    def _run(name: str, params: dict[str, float | int]) -> pd.Series:
        return _run_parametrized_filter(name, params, ts)

    def objective(trial: optuna.Trial) -> float:
        name, params = _suggest_filter_and_params(trial, families, value_scale)
        filtered = _run(name, params)
        metrics = filter_metrics(filtered, ts, use_mc_gcv=False)
        residual = ts - filtered
        rel_mae = float(residual.abs().mean()) / abs_mean
        roughness_ratio = float(np.std(np.diff(filtered.to_numpy()))) / orig_roughness

        derivative_corr = float(metrics["derivative_corr"])
        band_penalty = _target_band_penalty_from_diagnostics(
            rel_mae=rel_mae,
            roughness_ratio=roughness_ratio,
            derivative_corr=derivative_corr,
            rel_mae_band=rel_mae_band,
            roughness_ratio_band=roughness_ratio_band,
            min_derivative_corr=min_derivative_corr,
        )

        lags = max(min(20, len(ts) // 5), 1)
        whiteness = float(metrics["residual_autocorr"]) / lags

        base_loss = (
            0.45 * _bounded_unit(whiteness)
            + 0.20 * _bounded_unit(float(metrics["residual_iid"]))
            + 0.15 * _bounded_unit(float(metrics["roughness"]))
            + 0.10 * _bounded_unit(float(metrics["spectral_distance"]))
            - 0.15 * derivative_corr
        )
        loss = _TUNE_FILTER_BAND_PENALTY_WEIGHT * band_penalty + base_loss

        trial.set_user_attr("filter_name", name)
        trial.set_user_attr("params", params)
        trial.set_user_attr("metrics", dict(metrics))
        trial.set_user_attr("rel_mae", rel_mae)
        trial.set_user_attr("roughness_ratio", roughness_ratio)
        trial.set_user_attr("band_penalty", band_penalty)
        trial.set_user_attr("base_loss", base_loss)
        return float(loss)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=(progress and True),
    )

    best = study.best_trial
    best_name = best.user_attrs["filter_name"]
    best_params = best.user_attrs["params"]
    best_series = _run_parametrized_filter(best_name, best_params, ts).rename(
        ts.name if ts.name is not None else best_name
    )

    return TuneFilterResult(
        name=best_name,
        params=best_params,
        series=best_series,
        metrics=best.user_attrs["metrics"],
        rel_mae=float(best.user_attrs["rel_mae"]),
        roughness_ratio=float(best.user_attrs["roughness_ratio"]),
        objective_value=float(study.best_value),
        band_penalty=float(best.user_attrs["band_penalty"]),
    )
