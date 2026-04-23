from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .adaptive_mi import AdaptiveMI


class AdaptiveMRMR:
    """
    AdaptiveMI-backed minimum-redundancy maximum-relevance selector.

    Supports both the common difference criterion (MID) and quotient
    criterion (MIQ), with optional raw-MI scoring from AdaptiveMI.
    """

    def __init__(
        self,
        scorer: AdaptiveMI | None = None,
        *,
        criterion: str = "mid",
        candidate_pool: int = 128,
        redundancy_weight: float = 1.0,
        redundancy_eps: float = 1e-8,
        use_raw_mi: bool = False,
        stable_relevance: bool = True,
        cv: int = 5,
        min_freq: float = 0.5,
        task: str = "regression",
        random_state: int = 42,
    ):
        self.scorer = scorer or AdaptiveMI(random_state=random_state)
        self.criterion = str(criterion).lower()
        self.candidate_pool = int(candidate_pool)
        self.redundancy_weight = float(redundancy_weight)
        self.redundancy_eps = float(redundancy_eps)
        self.use_raw_mi = bool(use_raw_mi)
        self.stable_relevance = bool(stable_relevance)
        self.cv = int(cv)
        self.min_freq = float(min_freq)
        self.task = str(task).lower()
        self.random_state = int(random_state)

        self.relevance_scores_: pd.Series | None = None
        self.selection_scores_: pd.Series | None = None
        self.selected_features_: list[str] = []
        self.redundancy_matrix_: pd.DataFrame | None = None

        if self.criterion not in {"mid", "miq"}:
            raise ValueError(
                f"Unknown criterion={criterion!r}. Expected 'mid' or 'miq'."
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        min_features: int = 1,
        max_features: int | None = None,
        mi_threshold: float = 0.01,
        min_samples: int = 10,
    ) -> AdaptiveMRMR:
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            self.selected_features_ = []
            self.relevance_scores_ = pd.Series(dtype=float)
            self.selection_scores_ = pd.Series(dtype=float)
            self.redundancy_matrix_ = pd.DataFrame(dtype=float)
            return self

        X_clean, y_clean = self._prepare_data(X[numerical_cols], y)
        if len(X_clean) < int(min_samples):
            self.selected_features_ = numerical_cols.tolist()
            self.relevance_scores_ = pd.Series(dtype=float)
            self.selection_scores_ = pd.Series(dtype=float)
            self.redundancy_matrix_ = pd.DataFrame(dtype=float)
            return self

        if self.stable_relevance:
            relevance = self._compute_relevance_stable(X_clean, y_clean)
        else:
            relevance = self._compute_relevance_fast(X_clean, y_clean)

        self.relevance_scores_ = relevance
        ranked_features = relevance.index.tolist()
        if not ranked_features:
            self.selected_features_ = []
            self.selection_scores_ = pd.Series(dtype=float)
            self.redundancy_matrix_ = pd.DataFrame(dtype=float)
            return self

        target_count = self._resolve_target_feature_count(
            relevance,
            min_features=min_features,
            max_features=max_features,
            mi_threshold=mi_threshold,
        )
        pool_size = max(target_count, min(self.candidate_pool, len(ranked_features)))
        candidate_pool = ranked_features[:pool_size]
        redundancy = self._compute_redundancy_matrix(X_clean[candidate_pool])

        selected, scores = self._greedy_select(
            relevance.loc[candidate_pool], redundancy, target_count
        )
        if not selected:
            fallback_count = max(1, int(min_features))
            selected = ranked_features[:fallback_count]
            scores = {feature: float(relevance[feature]) for feature in selected}

        self.selected_features_ = selected
        self.selection_scores_ = pd.Series(scores).sort_values(ascending=False)
        self.redundancy_matrix_ = redundancy
        return self

    def _prepare_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]

        if self.task == "classification":
            if not pd.api.types.is_numeric_dtype(y_aligned):
                encoder = LabelEncoder()
                y_clean = pd.Series(
                    encoder.fit_transform(y_aligned), index=y_aligned.index
                )
            else:
                y_clean = y_aligned.astype("int32")
        else:
            y_clean = pd.to_numeric(y_aligned, errors="coerce")

        valid_mask = y_clean.notna() & np.isfinite(y_clean)
        if not valid_mask.any():
            return pd.DataFrame(), pd.Series(dtype=float)

        X_clean = X_aligned[valid_mask].astype("float32", copy=False)
        y_clean = y_clean[valid_mask]
        return X_clean, y_clean

    def _compute_relevance_fast(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        scores = self.scorer.score_pairwise(
            X.values, y.values, return_raw_mi=self.use_raw_mi
        )
        relevance = pd.Series(scores, index=X.columns)
        return relevance.fillna(0.0).clip(lower=0.0).sort_values(ascending=False)

    def _compute_relevance_stable(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        n = len(X)
        cv = max(2, min(self.cv, max(2, n // 20)))
        if cv < 2:
            return self._compute_relevance_fast(X, y)

        if self._is_classification_target(y):
            splitter = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=self.random_state
            )
            split_iter = splitter.split(X, y)
        else:
            splitter = KFold(
                n_splits=cv, shuffle=True, random_state=self.random_state
            )
            split_iter = splitter.split(X)

        names = list(X.columns)
        fold_scores: dict[str, list[float]] = {name: [] for name in names}
        positive_freq: dict[str, int] = {name: 0 for name in names}

        for train_idx, _ in split_iter:
            X_fold = X.iloc[train_idx]
            y_fold = y.iloc[train_idx]
            scores = self.scorer.score_pairwise(
                X_fold.values, y_fold.values, return_raw_mi=self.use_raw_mi
            )
            fold_series = pd.Series(scores, index=names).fillna(0.0).clip(lower=0.0)
            for name, value in fold_series.items():
                if np.isfinite(value):
                    fold_scores[name].append(float(value))
                    if value > 0:
                        positive_freq[name] += 1

        min_freq = int(np.ceil(self.min_freq * cv))
        aggregated = {}
        for name in names:
            if not fold_scores[name]:
                aggregated[name] = 0.0
                continue
            if positive_freq[name] < min_freq:
                aggregated[name] = 0.0
                continue
            aggregated[name] = float(np.median(fold_scores[name]))

        relevance = pd.Series(aggregated, index=names)
        return relevance.fillna(0.0).clip(lower=0.0).sort_values(ascending=False)

    def _compute_redundancy_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        columns = list(X.columns)
        n_features = len(columns)
        matrix = np.zeros((n_features, n_features), dtype=float)
        values = [X[col].to_numpy(copy=False) for col in columns]

        for i in range(n_features):
            matrix[i, i] = 1.0
            for j in range(i + 1, n_features):
                score = float(
                    self.scorer.score(
                        values[i],
                        values[j],
                        return_raw_mi=self.use_raw_mi,
                    )
                )
                if not np.isfinite(score):
                    score = 0.0
                matrix[i, j] = matrix[j, i] = max(0.0, score)

        return pd.DataFrame(matrix, index=columns, columns=columns)

    def _greedy_select(
        self,
        relevance: pd.Series,
        redundancy: pd.DataFrame,
        target_count: int,
    ) -> tuple[list[str], dict[str, float]]:
        selected: list[str] = []
        scores: dict[str, float] = {}
        remaining = list(relevance.index)

        while remaining and len(selected) < target_count:
            best_feature = None
            best_score = -np.inf
            best_relevance = -np.inf

            for feature in remaining:
                rel = float(relevance[feature])
                if not selected:
                    score = rel
                else:
                    avg_redundancy = float(redundancy.loc[feature, selected].mean())
                    score = self._combine(rel, avg_redundancy)

                if score > best_score or (
                    np.isclose(score, best_score) and rel > best_relevance
                ):
                    best_feature = feature
                    best_score = score
                    best_relevance = rel

            if best_feature is None:
                break

            selected.append(best_feature)
            scores[best_feature] = float(best_score)
            remaining.remove(best_feature)

        return selected, scores

    def _combine(self, relevance: float, redundancy: float) -> float:
        weighted_redundancy = self.redundancy_weight * float(redundancy)
        if self.criterion == "miq":
            return float(relevance / (weighted_redundancy + self.redundancy_eps))
        return float(relevance - weighted_redundancy)

    @staticmethod
    def _is_classification_target(y: pd.Series) -> bool:
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            return True
        n = max(1, len(y))
        k = y.nunique(dropna=True)
        return (k <= 20) or (k / n < 0.05)

    @staticmethod
    def _resolve_target_feature_count(
        scores: pd.Series,
        *,
        min_features: int,
        max_features: int | None,
        mi_threshold: float,
    ) -> int:
        above_threshold = int((scores > mi_threshold).sum())
        min_count = max(1, int(min_features))
        max_count = len(scores) if max_features is None else max(min_count, int(max_features))
        target = above_threshold if above_threshold > 0 else min_count
        return int(np.clip(target, min_count, min(max_count, len(scores))))
