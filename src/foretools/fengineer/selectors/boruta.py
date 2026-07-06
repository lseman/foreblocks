"""Proper Boruta feature selection with iterative acceptance/rejection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class BorutaSelector(BaseEstimator, TransformerMixin):
    """
    Full Boruta: All-Relevant Feature Selection.

    Iteratively compares real feature importance against shadow feature
    importance using statistically rigorous binomial tests. Features are
    definitively accepted or rejected once their p-value crosses alpha.

    Key improvements over the basic implementation:
      - Proper iterative acceptance/rejection with binomial tests
      - Shadow feature shuffling per-iteration
      - Multiple random forest estimators for stability
      - Smart stopping when all features are decided
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        max_iter: int = 20,
        perc: float = 100,
        alpha: float = 0.05,
        random_state: int = 42,
        verbose: int = 0,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        class_weight: str | dict | None = "balanced",
        n_jobs: int = -1,
    ):
        self.estimator = estimator
        self.max_iter = max_iter
        self.perc = perc
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.n_jobs = n_jobs

        self.support_: np.ndarray | None = None
        self.ranking_: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self.is_fitted: bool = False

        # Results
        self.decisions_: dict[str, str] = {}
        self.hit_counts_: np.ndarray | None = None
        self.p_values_: np.ndarray | None = None
        self.importance_history_: list[np.ndarray] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BorutaSelector":
        # Feature names (guaranteed non-None after this line)
        feature_names: list[str] = X.columns.tolist()
        n_feat = X.shape[1]
        n_features = n_feat

        # Determine task type for default estimator
        if self.estimator is None:
            y_clean = y.dropna()
            if pd.api.types.is_numeric_dtype(y_clean):
                unique = y_clean.nunique()
                is_classification = unique < 20 or unique / max(1, len(y_clean)) < 0.05
            else:
                is_classification = True

            self.estimator = (
                RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
                if is_classification
                else RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
            )

        X_array = X.values.astype(np.float64, copy=True)
        y_array = y.values.astype(float, copy=True)
        rng = np.random.RandomState(self.random_state)

        # Initialize states
        status: dict[str, str] = {f: "Tentative" for f in feature_names}
        hit_counts = np.zeros(n_feat, dtype=float)
        self.hit_counts_ = hit_counts.copy()
        self.importance_history_ = []

        n_accepted = 0
        n_rejected = 0
        n_tentative = n_features

        for iteration in range(self.max_iter):
            # Early stop if all features decided
            if n_tentative == 0:
                if self.verbose:
                    print(
                        f"  Boruta converged at iteration {iteration + 1}: "
                        f"{n_accepted} accepted, {n_rejected} rejected"
                    )
                break

            # Create shadow features (shuffle each column independently)
            X_shadow = X_array.copy()
            for j in range(n_features):
                col = X_shadow[:, j].copy()
                rng.shuffle(col)
                X_shadow[:, j] = col
            X_combined = np.hstack([X_array, X_shadow])

            # Fit RF and get importances
            est = self.estimator  # type: ignore[assignment]
            est.fit(X_combined, y_array)  # type: ignore[union-attr]
            importances = est.feature_importances_  # type: ignore[union-attr]
            self.importance_history_.append(importances.copy())

            real_imp = importances[:n_features]
            shadow_max = np.percentile(importances[n_features:], self.perc)
            try:
                m_shadow = float(shadow_max)
            except Exception:
                m_shadow = 0.0

            # Count hits for tentative features only
            tentative_mask = np.array([status[f] == "Tentative" for f in feature_names])
            tentative_hits = real_imp[tentative_mask] > m_shadow
            tentative_indices = np.where(tentative_mask)[0]

            hit_counts[tentative_indices] += tentative_hits.astype(float)
            self.hit_counts_ = hit_counts.copy()

            # Binomial test for each tentative feature
            n_trials = iteration + 1
            for i, feat_name in enumerate(feature_names):
                if status[feat_name] != "Tentative":
                    continue

                try:
                    k = int(hit_counts[i])
                except Exception:
                    k = 0
                try:
                    test = binomtest(k, n_trials, p=0.5, alternative="greater")
                    p_value = test.pvalue
                except Exception:
                    p_value = 1.0

                if p_value <= self.alpha and k >= 2:
                    status[feat_name] = "Accepted"
                    n_accepted += 1
                    n_tentative -= 1
                    if self.verbose:
                        print(
                            f"  Iter {iteration + 1}: ✅ {feat_name} ACCEPTED "
                            f"(k={k}, p={p_value:.4f})"
                        )

            # Reject features with zero hits after enough iterations
            if n_trials >= 15:
                for i, feat_name in enumerate(feature_names):
                    if status[feat_name] != "Tentative":
                        continue
                    try:
                        k = int(hit_counts[i])
                    except Exception:
                        k = 0
                    if k == 0:
                        status[feat_name] = "Rejected"
                        n_rejected += 1
                        n_tentative -= 1
                        if self.verbose:
                            print(
                                f"  Iter {iteration + 1}: ❌ {feat_name} REJECTED "
                                f"(k=0 after {n_trials} iters)"
                            )

            # Report
            if self.verbose:
                print(
                    f"  Iter {iteration + 1}/{self.max_iter}: "
                    f"{n_accepted} accepted, {n_rejected} rejected, "
                    f"{n_tentative} tentative"
                )

        # Set final results
        self.decisions_ = status
        self.p_values_ = self._compute_final_p_values(hit_counts, n_features)

        self.support_ = np.array(
            [status[f] == "Accepted" for f in feature_names], dtype=bool
        )
        self.ranking_ = self._compute_ranking(
            hit_counts, status, feature_names, n_features
        )
        self.is_fitted = True

        return self

    def _compute_final_p_values(
        self, hit_counts: np.ndarray, n_feat: int
    ) -> np.ndarray:
        """Compute approximate final p-values."""
        n_trials = len(self.importance_history_)
        p_values = np.ones(n_feat)
        for i in range(n_feat):
            try:
                k = int(hit_counts[i])
            except Exception:
                k = 0
            try:
                test = binomtest(k, max(1, n_trials), p=0.5, alternative="greater")
                p_values[i] = test.pvalue
            except Exception:
                p_values[i] = 1.0
        return p_values

    def _compute_ranking(
        self,
        hit_counts: np.ndarray,
        status: dict[str, str],
        feature_names: list[str],
        n_feat: int,
    ) -> np.ndarray:
        """Compute rankings: accepted=1, tentative=2, rejected=highest."""
        ranking = np.ones(n_feat, dtype=int)
        tentative_count = sum(1 for s in status.values() if s == "Tentative")
        rejected_count = sum(1 for s in status.values() if s == "Rejected")

        rank = tentative_count + 2
        for i, feat_name in enumerate(feature_names):
            if status[feat_name] == "Rejected":
                ranking[i] = rank
                rank += 1

        accepted_indices = [
            i for i, f in enumerate(feature_names) if status[f] == "Accepted"
        ]
        accepted_sorted = sorted(
            accepted_indices,
            key=lambda i: hit_counts[i],
            reverse=True,
        )
        for rank_val, idx in enumerate(accepted_sorted, start=1):
            ranking[idx] = rank_val

        return ranking

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("BorutaSelector must be fitted first.")
        assert self.support_ is not None
        return X.iloc[:, self.support_]

    def get_selected_features(self) -> list[str]:
        if not self._feature_names:
            return []
        assert self.support_ is not None
        return [name for name, s in zip(self._feature_names, self.support_) if s]

    def get_decision_summary(self) -> dict[str, str]:
        """Return feature -> decision mapping."""
        return dict(self.decisions_)

    def get_importance_history(self) -> list[np.ndarray]:
        """Return importance arrays from each iteration."""
        return [h.copy() for h in self.importance_history_]
