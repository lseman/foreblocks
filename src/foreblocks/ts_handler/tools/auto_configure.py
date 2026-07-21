"""foreblocks.ts_handler.auto_configure.

Auto-configuration logic for time-series preprocessing.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from foreblocks.ts_handler.utils import (
    _as_2d,
    _mean_abs_autocorr,
    _mean_fill_2d,
    _rank_normalize,
    _safe_corr_1d,
    _select_diagnostic_features,
)


def summarize_configuration(params: dict[str, Any]) -> None:
    print(
        "\n"
        + tabulate(
            [
                ["Dataset Dimensions", params["dimensions"]],
                ["Missing Values", f"{params['missing_rate']:.2%}"],
                [
                    "Stationarity",
                    "Non-stationary" if params["detrend"] else "Stationary",
                ],
                ["Seasonality", "Present" if params["seasonal"] else "Not detected"],
                [
                    "Transformation",
                    "Log (selective)" if params["log_transform"] else "None",
                ],
                ["Scaling", params.get("scaling_method", "standard")],
                [
                    "Signal Processing",
                    params["filter_method"] if params["apply_filter"] else "None",
                ],
                ["Imputation", params["impute_method"] or "None"],
                ["Outlier Detection", params["outlier_method"]],
                ["Outlier Threshold", f"{params['outlier_threshold']:.2f}"],
                ["Decomposition", f"{params['ewt_bands']} bands"],
            ],
            headers=["Parameter", "Configuration"],
            tablefmt="pretty",
        )
    )


def _auto_filter_weights(stats: dict[str, Any]) -> dict[str, float]:
    weights = {
        "fidelity_mse": 0.40,
        "roughness": 0.10,
        "residual_autocorr": 0.25,
        "derivative_corr": 0.25,
    }

    if stats["med_snr"] < 1.5:
        weights["roughness"] += 0.10
        weights["residual_autocorr"] += 0.05
        weights["fidelity_mse"] -= 0.05
        weights["derivative_corr"] -= 0.10

    if stats["seasonal_fraction"] > 0.25 or stats["strong_periods"] >= 1:
        weights["derivative_corr"] += 0.10
        weights["roughness"] -= 0.05
        weights["fidelity_mse"] -= 0.05

    if stats["heavy_tails_fraction"] > 0.35 or stats["extreme_ratio"] > 0.08:
        weights["fidelity_mse"] -= 0.05
        weights["roughness"] += 0.05

    total = sum(weights.values())
    return {k: float(v / total) for k, v in weights.items()}


def _filter_eval_kwargs(method: str, stats: dict[str, Any]) -> dict[str, Any]:
    method = (method or "none").lower()
    kwargs: dict[str, Any] = {
        "fill_nans_for_filter": True,
        "n_jobs": 1,
    }
    if method == "stl":
        period = max(2, min(365, int(stats.get("dominant_period", 7) or 7)))
        kwargs.update(
            {
                "period": period,
                "seasonal": max(5, min(13, period if period % 2 == 1 else period + 1)),
                "robust": True,
            }
        )
    elif method == "ssa":
        kwargs.update(
            {
                "window_length": min(
                    max(7, stats.get("T_eval", 24) // 4),
                    max(7, stats.get("T_eval", 24) // 4),
                ),
                "n_components": 2,
            }
        )
    elif method == "lowess":
        kwargs.update({"frac": 0.05 if stats["T"] > 400 else 0.08})
    elif method == "wiener":
        kwargs.update({"mysize": min(15, max(5, 12 // 2 * 2 + 1))})
    elif method == "savgol":
        kwargs.update({"robust_center": True})
    return kwargs


def _filter_candidate_pool(stats: dict[str, Any]) -> list[str]:
    candidates = ["none", "savgol"]

    if stats["strong_periods"] >= 1 or stats["seasonal_fraction"] > 0.25:
        candidates.extend(["stl", "ssa"])
    if stats["med_snr"] < 1.8:
        candidates.extend(["wiener", "savgol"])
    if stats["is_autoregressive"]:
        candidates.extend(["savgol", "lowess"])
    if stats["heavy_tails_fraction"] > 0.25:
        candidates.extend(["ssa", "lowess"])
    if stats["missing_rate"] < 0.2 and stats["T"] > 250:
        candidates.append("kalman")

    ordered: list[str] = []
    for name in candidates:
        lname = (name or "none").lower()
        if lname not in ordered:
            ordered.append(lname)
    return ordered


def _sample_for_filter_selection(data: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    x = _as_2d(data)
    feat_idx = _select_diagnostic_features(x.shape[1], max_features=min(8, x.shape[1]))
    sampled = x[:, feat_idx]
    max_points = 1024 if stats["T"] > 2000 else 2048
    if sampled.shape[0] > max_points:
        step = max(1, int(np.ceil(sampled.shape[0] / max_points)))
        sampled = sampled[::step]
    return sampled


def _score_filter_candidate(
    original: np.ndarray, denoised: np.ndarray
) -> dict[str, float]:
    orig = _as_2d(original)
    den = _as_2d(denoised)
    residual = orig - den
    deriv_orig = np.diff(orig, axis=0)
    deriv_den = np.diff(den, axis=0)

    derivative_corrs = (
        [
            abs(_safe_corr_1d(deriv_den[:, j], deriv_orig[:, j]))
            for j in range(orig.shape[1])
        ]
        if orig.shape[0] > 1
        else [0.0]
    )

    residual_autocorr = [
        _mean_abs_autocorr(residual[:, j], max_lag=min(20, max(1, orig.shape[0] // 5)))
        for j in range(orig.shape[1])
    ]

    roughness = [
        float(np.std(np.diff(den[:, j]))) if den.shape[0] > 1 else 0.0
        for j in range(den.shape[1])
    ]

    fidelity = [
        float(np.mean((den[:, j] - orig[:, j]) ** 2)) for j in range(orig.shape[1])
    ]

    return {
        "fidelity_mse": float(np.mean(fidelity)),
        "roughness": float(np.mean(roughness)),
        "residual_autocorr": float(np.mean(residual_autocorr)),
        "derivative_corr": float(np.mean(derivative_corrs)),
    }


def _auto_select_filter_method(
    data: np.ndarray,
    stats: dict[str, Any],
    filter_method: str,
    apply_filter_fn: callable,
    verbose: bool,
) -> dict[str, Any]:
    x_eval = _sample_for_filter_selection(data, stats)
    x_eval = _mean_fill_2d(_mean_fill_2d(x_eval))
    candidates = _filter_candidate_pool(stats)
    metrics_rows: list[dict[str, Any]] = []

    for method in candidates:
        try:
            if method == "none":
                filtered = x_eval.copy()
            else:
                filtered = apply_filter_fn(
                    x_eval,
                    method=method,
                    **_filter_eval_kwargs(method, stats),
                )
                filtered = _mean_fill_2d(filtered)
            row = {"method": method}
            row.update(_score_filter_candidate(x_eval, filtered))
            metrics_rows.append(row)
        except Exception as exc:
            if verbose:
                print(f"[Preprocessing] Filter candidate '{method}' skipped: {exc}")

    if not metrics_rows:
        return {
            "best_method": filter_method,
            "apply_filter": False,
            "scores": pd.DataFrame(),
        }

    scores = pd.DataFrame(metrics_rows).set_index("method")
    weights = _auto_filter_weights(stats)
    scores["score"] = (
        weights["fidelity_mse"] * _rank_normalize(scores["fidelity_mse"].values)
        + weights["roughness"] * _rank_normalize(scores["roughness"].values)
        + weights["residual_autocorr"]
        * _rank_normalize(scores["residual_autocorr"].values)
        + weights["derivative_corr"]
        * _rank_normalize(scores["derivative_corr"].values, invert=True)
    )
    scores = scores.sort_values("score")

    best_method = str(scores.index[0])
    none_score = (
        float(scores.loc["none", "score"]) if "none" in scores.index else float("inf")
    )
    best_score = float(scores.iloc[0]["score"])
    improvement = none_score - best_score

    noisy_enough = (
        stats["med_snr"] < 2.0
        or stats["heavy_tails_fraction"] > 0.20
        or stats["extreme_ratio"] > 0.04
        or stats["seasonal_fraction"] > 0.25
    )
    apply_filter = best_method != "none" and (improvement > 0.08 or noisy_enough)

    return {
        "best_method": best_method,
        "apply_filter": bool(apply_filter),
        "best_score": best_score,
        "none_score": none_score,
        "improvement": float(improvement),
        "weights": weights,
        "scores": scores,
    }
