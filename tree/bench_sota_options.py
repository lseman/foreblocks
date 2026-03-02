#!/usr/bin/env python3
"""Benchmark ForeForest against strong baselines.

Runs regression and binary classification checks with:
- ForeForest axis baseline
- ForeForest "modern" config (k-way + oblique + GOSS + DART)
- sklearn histogram GBDT baselines
- optional xgboost/lightgbm/catboost if installed
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score


def _configure_import_paths() -> None:
    root = Path(__file__).resolve().parent
    preferred = [root / "build", root.parent / "build"]
    existing = []
    for p in preferred:
        if p.exists():
            existing.append(p)
    # Keep priority order: root/build, then parent/build.
    for p in reversed(existing):
        sys.path.insert(0, str(p))


_configure_import_paths()
import foreforest  # noqa: E402


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def logloss(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, 1.0 - 1.0e-12)
    return float(-np.mean(y_true * np.log(prob) + (1.0 - y_true) * np.log(1.0 - prob)))


def make_regression_data(
    n: int = 20000, p: int = 16, seed: int = 7, missing_rate: float = 0.03
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)

    X[:, 0] = rng.integers(0, 6, size=n)
    X[:, 1] = rng.integers(0, 4, size=n)

    y = (
        2.2 * (X[:, 0] == 3)
        - 1.6 * (X[:, 0] == 1)
        + 1.3 * (X[:, 1] == 2)
        + 0.8 * X[:, 2] * X[:, 3]
        - 1.1 * np.sin(X[:, 4])
        + 0.35 * (X[:, 5] ** 2)
        + rng.normal(0.0, 0.25, size=n)
    ).astype(np.float64)

    miss = rng.random((n, p)) < missing_rate
    X[miss] = np.nan
    return X, y


def make_binary_data(
    n: int = 20000, p: int = 16, seed: int = 123, missing_rate: float = 0.03
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)

    X[:, 0] = rng.integers(0, 6, size=n)
    X[:, 1] = rng.integers(0, 4, size=n)

    z = (
        1.9 * (X[:, 0] == 3)
        - 1.4 * (X[:, 0] == 1)
        + 1.1 * (X[:, 1] == 2)
        + 0.8 * X[:, 2] * X[:, 3]
        - 1.0 * np.sin(X[:, 4])
        + 0.30 * (X[:, 5] ** 2)
        + rng.normal(0.0, 0.40, size=n)
    )
    y = (z > 0.0).astype(np.float64)

    miss = rng.random((n, p)) < missing_rate
    X[miss] = np.nan
    return X, y


def split_tvt(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(train_ratio * n)
    n_valid = int(valid_ratio * n)
    i_train = perm[:n_train]
    i_valid = perm[n_train : n_train + n_valid]
    i_test = perm[n_train + n_valid :]
    return X[i_train], y[i_train], X[i_valid], y[i_valid], X[i_test], y[i_test]


def _build_hist_cfg() -> Any:
    hcfg = foreforest.HistogramConfig()
    hcfg.method = "adaptive"
    hcfg.max_bins = 256
    hcfg.use_missing_bin = True
    hcfg.adaptive_binning = True
    hcfg.use_parallel = False
    return hcfg


def _build_tree_cfg(tree_overrides: dict[str, Any]) -> Any:
    cfg = foreforest.TreeConfig()
    cfg.max_depth = 8
    cfg.max_leaves = 63
    cfg.min_samples_split = 20
    cfg.min_samples_leaf = 10
    cfg.min_child_weight = 1e-3
    cfg.lambda_ = 1.0
    cfg.alpha_ = 0.0
    cfg.gamma_ = 0.0
    cfg.growth = foreforest.Growth.LeafWise
    cfg.split_mode = foreforest.SplitMode.Histogram
    cfg.exact_cutover = 2048
    cfg.enable_kway_splits = False
    cfg.enable_oblique_splits = False
    cfg.kway_max_groups = 8
    cfg.oblique_k_features = 4
    cfg.oblique_ridge = 1e-3
    cfg.axis_vs_oblique_guard = 1.02
    cfg.subsample_bytree = 1.0
    cfg.subsample_bylevel = 1.0
    cfg.subsample_bynode = 1.0
    cfg.subsample_with_replacement = True
    cfg.subsample_importance_scale = False
    cfg.goss.enabled = False
    cfg.goss.top_rate = 0.2
    cfg.goss.other_rate = 0.1
    cfg.goss.min_node_size = 1024
    for k, v in tree_overrides.items():
        if k.startswith("goss_"):
            setattr(cfg.goss, k[len("goss_") :], v)
        else:
            setattr(cfg, k, v)
    return cfg


def _build_forest_cfg(
    objective: Any, tree_cfg: Any, forest_overrides: dict[str, Any]
) -> Any:
    cfg = foreforest.ForeForestConfig()
    cfg.mode = foreforest.Mode.GBDT
    cfg.objective = objective
    cfg.n_estimators = 300
    cfg.learning_rate = 0.05
    cfg.colsample_bytree = 0.8
    cfg.colsample_bynode = 0.8
    cfg.gbdt_use_subsample = True
    cfg.gbdt_row_subsample = 0.8
    cfg.early_stopping_enabled = True
    cfg.early_stopping_rounds = 30
    cfg.early_stopping_min_delta = 0.0
    cfg.dart_enabled = False
    cfg.dart_drop_rate = 0.10
    cfg.dart_max_drop = 3
    cfg.dart_normalize = True
    cfg.hist_cfg = _build_hist_cfg()
    cfg.tree_cfg = tree_cfg
    for k, v in forest_overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_foreforest_regression(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tree_overrides: dict[str, Any],
    forest_overrides: dict[str, Any],
) -> dict[str, Any]:
    tree_cfg = _build_tree_cfg(tree_overrides)
    ff_cfg = _build_forest_cfg(
        foreforest.Objective.SquaredError, tree_cfg, forest_overrides
    )
    model = foreforest.ForeForest(ff_cfg)

    t0 = time.perf_counter()
    model.fit_complete(X_train, y_train, X_valid, y_valid)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_valid = np.asarray(model.predict(X_valid), dtype=np.float64)
    pred_test = np.asarray(model.predict(X_test), dtype=np.float64)
    pred_s = time.perf_counter() - t1

    return {
        "name": name,
        "fit_s": fit_s,
        "pred_s": pred_s,
        "metric_valid": mse(y_valid, pred_valid),
        "metric_test": mse(y_test, pred_test),
        "trees": int(model.size()),
        "best_iter": int(model.best_iteration()),
        "early_stop": bool(model.early_stopped()),
    }


def run_foreforest_binary(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tree_overrides: dict[str, Any],
    forest_overrides: dict[str, Any],
) -> dict[str, Any]:
    tree_cfg = _build_tree_cfg(tree_overrides)
    ff_cfg = _build_forest_cfg(
        foreforest.Objective.BinaryLogloss, tree_cfg, forest_overrides
    )
    model = foreforest.ForeForest(ff_cfg)

    t0 = time.perf_counter()
    model.fit_complete(X_train, y_train, X_valid, y_valid)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    prob_valid = np.asarray(model.predict(X_valid), dtype=np.float64)
    prob_test = np.asarray(model.predict(X_test), dtype=np.float64)
    pred_s = time.perf_counter() - t1

    auc_valid = float(roc_auc_score(y_valid, prob_valid))
    auc_test = float(roc_auc_score(y_test, prob_test))
    return {
        "name": name,
        "fit_s": fit_s,
        "pred_s": pred_s,
        "metric_valid": logloss(y_valid, prob_valid),
        "metric_test": logloss(y_test, prob_test),
        "auc_valid": auc_valid,
        "auc_test": auc_test,
        "trees": int(model.size()),
        "best_iter": int(model.best_iteration()),
        "early_stop": bool(model.early_stopped()),
    }


def run_sklearn_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=8,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        early_stopping=True,
        random_state=99,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)
    pred_s = time.perf_counter() - t1
    n_iter = int(getattr(model, "n_iter_", 0))
    return {
        "name": "sklearn_hgbr",
        "fit_s": fit_s,
        "pred_s": pred_s,
        "metric_valid": mse(y_valid, pred_valid),
        "metric_test": mse(y_test, pred_test),
        "trees": n_iter,
        "best_iter": n_iter,
        "early_stop": True,
    }


def run_sklearn_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=8,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        early_stopping=True,
        random_state=99,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    prob_valid = model.predict_proba(X_valid)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]
    pred_s = time.perf_counter() - t1
    n_iter = int(getattr(model, "n_iter_", 0))
    return {
        "name": "sklearn_hgbc",
        "fit_s": fit_s,
        "pred_s": pred_s,
        "metric_valid": logloss(y_valid, prob_valid),
        "metric_test": logloss(y_test, prob_test),
        "auc_valid": float(roc_auc_score(y_valid, prob_valid)),
        "auc_test": float(roc_auc_score(y_test, prob_test)),
        "trees": n_iter,
        "best_iter": n_iter,
        "early_stop": True,
    }


def maybe_run_external_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        xgb = None  # type: ignore
    if xgb is not None:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=99,
            eval_metric="rmse",
            verbosity=0,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred_valid = model.predict(X_valid)
        pred_test = model.predict(X_test)
        pred_s = time.perf_counter() - t1
        rows.append(
            {
                "name": "xgboost_hist",
                "fit_s": fit_s,
                "pred_s": pred_s,
                "metric_valid": mse(y_valid, pred_valid),
                "metric_test": mse(y_test, pred_test),
                "trees": int(getattr(model, "n_estimators", 0)),
                "best_iter": int(getattr(model, "best_iteration", 0) or 0),
                "early_stop": False,
            }
        )

    try:
        import lightgbm as lgb  # type: ignore
    except Exception:
        lgb = None  # type: ignore
    if lgb is not None:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=99,
            verbosity=-1,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred_valid = model.predict(X_valid)
        pred_test = model.predict(X_test)
        pred_s = time.perf_counter() - t1
        rows.append(
            {
                "name": "lightgbm",
                "fit_s": fit_s,
                "pred_s": pred_s,
                "metric_valid": mse(y_valid, pred_valid),
                "metric_test": mse(y_test, pred_test),
                "trees": int(getattr(model, "n_estimators", 0)),
                "best_iter": int(getattr(model, "best_iteration_", 0) or 0),
                "early_stop": False,
            }
        )

    try:
        import catboost as cb  # type: ignore
    except Exception:
        cb = None  # type: ignore
    if cb is not None:
        model = cb.CatBoostRegressor(
            loss_function="RMSE",
            iterations=300,
            learning_rate=0.05,
            depth=8,
            random_seed=99,
            verbose=False,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred_valid = model.predict(X_valid)
        pred_test = model.predict(X_test)
        pred_s = time.perf_counter() - t1
        rows.append(
            {
                "name": "catboost",
                "fit_s": fit_s,
                "pred_s": pred_s,
                "metric_valid": mse(y_valid, pred_valid),
                "metric_test": mse(y_test, pred_test),
                "trees": int(getattr(model, "tree_count_", 0)),
                "best_iter": int(
                    getattr(model, "get_best_iteration", lambda: 0)() or 0
                ),
                "early_stop": False,
            }
        )

    return rows


def maybe_run_external_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        xgb = None  # type: ignore
    if xgb is not None:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=99,
            eval_metric="logloss",
            verbosity=0,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        prob_valid = model.predict_proba(X_valid)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]
        pred_s = time.perf_counter() - t1
        rows.append(
            {
                "name": "xgboost_hist_bin",
                "fit_s": fit_s,
                "pred_s": pred_s,
                "metric_valid": logloss(y_valid, prob_valid),
                "metric_test": logloss(y_test, prob_test),
                "auc_valid": float(roc_auc_score(y_valid, prob_valid)),
                "auc_test": float(roc_auc_score(y_test, prob_test)),
                "trees": int(getattr(model, "n_estimators", 0)),
                "best_iter": int(getattr(model, "best_iteration", 0) or 0),
                "early_stop": False,
            }
        )

    try:
        import lightgbm as lgb  # type: ignore
    except Exception:
        lgb = None  # type: ignore
    if lgb is not None:
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=99,
            verbosity=-1,
            force_col_wise=True,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        prob_valid = model.predict_proba(X_valid)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]
        pred_s = time.perf_counter() - t1
        rows.append(
            {
                "name": "lightgbm_bin",
                "fit_s": fit_s,
                "pred_s": pred_s,
                "metric_valid": logloss(y_valid, prob_valid),
                "metric_test": logloss(y_test, prob_test),
                "auc_valid": float(roc_auc_score(y_valid, prob_valid)),
                "auc_test": float(roc_auc_score(y_test, prob_test)),
                "trees": int(getattr(model, "n_estimators", 0)),
                "best_iter": int(getattr(model, "best_iteration_", 0) or 0),
                "early_stop": False,
            }
        )

    try:
        import catboost as cb  # type: ignore
    except Exception:
        cb = None  # type: ignore
    if cb is not None:
        model = cb.CatBoostClassifier(
            loss_function="Logloss",
            iterations=300,
            learning_rate=0.05,
            depth=8,
            random_seed=99,
            verbose=False,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        prob_valid = model.predict_proba(X_valid)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]
        pred_s = time.perf_counter() - t1
        rows.append(
            {
                "name": "catboost_bin",
                "fit_s": fit_s,
                "pred_s": pred_s,
                "metric_valid": logloss(y_valid, prob_valid),
                "metric_test": logloss(y_test, prob_test),
                "auc_valid": float(roc_auc_score(y_valid, prob_valid)),
                "auc_test": float(roc_auc_score(y_test, prob_test)),
                "trees": int(getattr(model, "tree_count_", 0)),
                "best_iter": int(
                    getattr(model, "get_best_iteration", lambda: 0)() or 0
                ),
                "early_stop": False,
            }
        )

    return rows


def print_regression_table(rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: r["metric_test"])
    print("\nRegression benchmark (lower MSE is better)")
    print(
        "name                     fit_s   pred_s   mse_valid  mse_test   trees best_iter early_stop"
    )
    for r in rows:
        print(
            f"{r['name']:<24} {r['fit_s']:>6.3f}  {r['pred_s']:>6.3f}  "
            f"{r['metric_valid']:>9.5f} {r['metric_test']:>9.5f}  "
            f"{r['trees']:>5d} {r['best_iter']:>9d} {str(r['early_stop']):>10}"
        )


def print_binary_table(rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: r["metric_test"])
    print("\nBinary benchmark (lower logloss / higher AUC is better)")
    print(
        "name                     fit_s   pred_s   ll_valid   ll_test   auc_valid  auc_test   trees best_iter early_stop"
    )
    for r in rows:
        print(
            f"{r['name']:<24} {r['fit_s']:>6.3f}  {r['pred_s']:>6.3f}  "
            f"{r['metric_valid']:>8.5f} {r['metric_test']:>8.5f}  "
            f"{r['auc_valid']:>8.5f} {r['auc_test']:>8.5f}  "
            f"{r['trees']:>5d} {r['best_iter']:>9d} {str(r['early_stop']):>10}"
        )


def main() -> None:
    print("foreforest module:", foreforest.__file__)

    Xr, yr = make_regression_data()
    Xr_tr, yr_tr, Xr_va, yr_va, Xr_te, yr_te = split_tvt(
        Xr, yr, train_ratio=0.70, valid_ratio=0.15, seed=99
    )

    reg_rows: list[dict[str, Any]] = []
    reg_rows.append(
        run_foreforest_regression(
            "foreforest_axis",
            Xr_tr,
            yr_tr,
            Xr_va,
            yr_va,
            Xr_te,
            yr_te,
            tree_overrides={},
            forest_overrides={"dart_enabled": False},
        )
    )
    reg_rows.append(
        run_foreforest_regression(
            "foreforest_kway_oblique_goss",
            Xr_tr,
            yr_tr,
            Xr_va,
            yr_va,
            Xr_te,
            yr_te,
            tree_overrides={
                "enable_kway_splits": True,
                "kway_max_groups": 8,
                "enable_oblique_splits": True,
                "oblique_k_features": 4,
                "axis_vs_oblique_guard": 1.01,
                "goss_enabled": True,
                "goss_top_rate": 0.2,
                "goss_other_rate": 0.1,
                "goss_min_node_size": 1024,
            },
            forest_overrides={"dart_enabled": False},
        )
    )
    reg_rows.append(
        run_foreforest_regression(
            "foreforest_modern_dart",
            Xr_tr,
            yr_tr,
            Xr_va,
            yr_va,
            Xr_te,
            yr_te,
            tree_overrides={
                "enable_kway_splits": True,
                "kway_max_groups": 8,
                "enable_oblique_splits": True,
                "oblique_k_features": 4,
                "axis_vs_oblique_guard": 1.01,
                "goss_enabled": True,
                "goss_top_rate": 0.2,
                "goss_other_rate": 0.1,
                "goss_min_node_size": 1024,
            },
            forest_overrides={
                "dart_enabled": True,
                "dart_drop_rate": 0.05,
                "dart_max_drop": 3,
                "dart_normalize": False,
            },
        )
    )
    reg_rows.append(
        run_foreforest_regression(
            "foreforest_oblivious",
            Xr_tr,
            yr_tr,
            Xr_va,
            yr_va,
            Xr_te,
            yr_te,
            tree_overrides={
                "growth": foreforest.Growth.Oblivious,
            },
            forest_overrides={"dart_enabled": False},
        )
    )
    reg_rows.append(
        run_foreforest_regression(
            "foreforest_sota_oblique",
            Xr_tr,
            yr_tr,
            Xr_va,
            yr_va,
            Xr_te,
            yr_te,
            tree_overrides={
                "enable_oblique_splits": True,
                "oblique_k_features": 4,
                "oblique_newton_steps": 3,
                "oblique_l1": 0.05,
            },
            forest_overrides={"dart_enabled": False},
        )
    )
    reg_rows.append(
        run_foreforest_regression(
            "foreforest_huber",
            Xr_tr,
            yr_tr,
            Xr_va,
            yr_va,
            Xr_te,
            yr_te,
            tree_overrides={},
            forest_overrides={
                "dart_enabled": False,
                "objective": foreforest.Objective.HuberError,
                "huber_delta": 1.5,
            },
        )
    )
    reg_rows.append(run_sklearn_regression(Xr_tr, yr_tr, Xr_va, yr_va, Xr_te, yr_te))
    reg_rows.extend(
        maybe_run_external_regressors(Xr_tr, yr_tr, Xr_va, yr_va, Xr_te, yr_te)
    )
    print_regression_table(reg_rows)

    Xb, yb = make_binary_data()
    Xb_tr, yb_tr, Xb_va, yb_va, Xb_te, yb_te = split_tvt(
        Xb, yb, train_ratio=0.70, valid_ratio=0.15, seed=202
    )

    bin_rows: list[dict[str, Any]] = []
    bin_rows.append(
        run_foreforest_binary(
            "foreforest_axis_bin",
            Xb_tr,
            yb_tr,
            Xb_va,
            yb_va,
            Xb_te,
            yb_te,
            tree_overrides={},
            forest_overrides={"dart_enabled": False},
        )
    )
    bin_rows.append(
        run_foreforest_binary(
            "foreforest_kway_oblique_goss_bin",
            Xb_tr,
            yb_tr,
            Xb_va,
            yb_va,
            Xb_te,
            yb_te,
            tree_overrides={
                "enable_kway_splits": True,
                "kway_max_groups": 8,
                "enable_oblique_splits": True,
                "oblique_k_features": 4,
                "axis_vs_oblique_guard": 1.01,
                "goss_enabled": True,
                "goss_top_rate": 0.2,
                "goss_other_rate": 0.1,
                "goss_min_node_size": 1024,
            },
            forest_overrides={"dart_enabled": False},
        )
    )
    bin_rows.append(
        run_foreforest_binary(
            "foreforest_modern_dart_bin",
            Xb_tr,
            yb_tr,
            Xb_va,
            yb_va,
            Xb_te,
            yb_te,
            tree_overrides={
                "enable_kway_splits": True,
                "kway_max_groups": 8,
                "enable_oblique_splits": True,
                "oblique_k_features": 4,
                "axis_vs_oblique_guard": 1.01,
                "goss_enabled": True,
                "goss_top_rate": 0.2,
                "goss_other_rate": 0.1,
                "goss_min_node_size": 1024,
            },
            forest_overrides={
                "dart_enabled": True,
                "dart_drop_rate": 0.05,
                "dart_max_drop": 3,
                "dart_normalize": False,
            },
        )
    )
    bin_rows.append(
        run_foreforest_binary(
            "foreforest_oblivious_bin",
            Xb_tr,
            yb_tr,
            Xb_va,
            yb_va,
            Xb_te,
            yb_te,
            tree_overrides={
                "growth": foreforest.Growth.Oblivious,
            },
            forest_overrides={"dart_enabled": False},
        )
    )
    bin_rows.append(
        run_foreforest_binary(
            "foreforest_sota_oblique_bin",
            Xb_tr,
            yb_tr,
            Xb_va,
            yb_va,
            Xb_te,
            yb_te,
            tree_overrides={
                "enable_oblique_splits": True,
                "oblique_k_features": 4,
                "oblique_newton_steps": 3,
                "oblique_l1": 0.05,
            },
            forest_overrides={"dart_enabled": False},
        )
    )
    bin_rows.append(
        run_foreforest_binary(
            "foreforest_focal_bin",
            Xb_tr,
            yb_tr,
            Xb_va,
            yb_va,
            Xb_te,
            yb_te,
            tree_overrides={},
            forest_overrides={
                "dart_enabled": False,
                "objective": foreforest.Objective.BinaryFocalLoss,
                "focal_gamma": 2.0,
            },
        )
    )
    bin_rows.append(
        run_foreforest_binary(
            "foreforest_efb_bin",
            Xb_tr,
            yb_tr,
            Xb_va,
            yb_va,
            Xb_te,
            yb_te,
            tree_overrides={},
            forest_overrides={
                "dart_enabled": False,
                "efb_enabled": True,
                "efb_max_conflict_rate": 0.05,
            },
        )
    )
    bin_rows.append(run_sklearn_binary(Xb_tr, yb_tr, Xb_va, yb_va, Xb_te, yb_te))
    bin_rows.extend(maybe_run_external_binary(Xb_tr, yb_tr, Xb_va, yb_va, Xb_te, yb_te))
    print_binary_table(bin_rows)

    print("\nInstalled optional boosters:")
    for mod in ["lightgbm", "xgboost", "catboost"]:
        try:
            m = __import__(mod)
            print(f"- {mod}: OK ({getattr(m, '__version__', 'unknown')})")
        except Exception:
            print(f"- {mod}: not installed")


if __name__ == "__main__":
    main()
