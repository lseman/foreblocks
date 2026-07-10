"""Clustering-based feature generator.

Generates cluster membership IDs and distance/probability features
via KMeans, GMM, HDBSCAN, and UMAP.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .aux import BaseFeatureTransformer, require_fitted

if TYPE_CHECKING:
    pass


class ClusteringTransformer(BaseFeatureTransformer):
    """
    Clustering-based feature generator.

    Fits multiple clustering strategies and produces:
      - Cluster IDs (membership)
      - Distance-to-centroid features (KMeans)
      - Membership probability features (GMM)
      - Approximate cluster strengths (HDBSCAN)
      - Embedding coordinates (UMAP)
    """

    def __init__(
        self,
        config: Any,
        strategies: tuple[str, ...] | None = None,
    ):
        super().__init__(config)
        config_strategies = getattr(config, "clustering_strategies", None)
        if strategies is not None:
            self.strategies = tuple(strategies)
        elif config_strategies:
            self.strategies = tuple(config_strategies)
        else:
            self.strategies = ("kmeans", "gmm")

        self.scaler_ = None
        self.models_: dict[str, Any] = {}
        self.cluster_features_: list[str] = []
        self.fill_values_ = pd.Series(dtype="float64")

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "ClusteringTransformer":
        if not getattr(self.config, "create_clustering", False):
            self.is_fitted = True
            return self

        num_cols = self.get_numerical_cols(X)
        if len(num_cols) < 2:
            self.is_fitted = True
            return self

        Xnum = X[num_cols].copy()
        self.fill_values_ = Xnum.median(numeric_only=True)
        Xnum = Xnum.fillna(self.fill_values_)

        max_feat = min(
            getattr(self.config, "clustering_max_features", 20), len(num_cols)
        )
        self.cluster_features_ = Xnum.var().nlargest(max_feat).index.tolist()
        if len(self.cluster_features_) < 2:
            self.is_fitted = True
            return self

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(Xnum[self.cluster_features_])

        n_clusters = max(2, min(self.config.n_clusters, len(X_scaled) // 10))
        n_clusters = min(n_clusters, len(X_scaled))
        if n_clusters < 2:
            self.is_fitted = True
            return self

        # --- Fit models ---
        if "kmeans" in self.strategies:
            try:
                km = KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    random_state=self.config.random_state,
                )
                km.fit(X_scaled)
                self.models_["kmeans"] = km
            except Exception as e:
                warnings.warn(f"KMeans clustering failed: {e}")

        if "gmm" in self.strategies:
            try:
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    random_state=self.config.random_state,
                )
                gmm.fit(X_scaled)
                self.models_["gmm"] = gmm
            except Exception as e:
                warnings.warn(f"GMM clustering failed: {e}")

        if "hdbscan" in self.strategies:
            try:
                import hdbscan as _hdbscan

                hdb = _hdbscan.HDBSCAN(min_cluster_size=15)
                hdb.fit(X_scaled)
                self.models_["hdbscan"] = hdb
            except Exception as e:
                warnings.warn(f"HDBSCAN clustering failed: {e}")

        if "umap" in self.strategies:
            try:
                import umap as _umap

                reducer = _umap.UMAP(
                    n_components=3,
                    random_state=self.config.random_state,
                )
                reducer.fit(X_scaled)
                self.models_["umap"] = reducer
            except Exception as e:
                warnings.warn(f"UMAP embedding failed: {e}")

        self.is_fitted = True
        return self

    @require_fitted
    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        if not self.models_ or not self.cluster_features_:
            return self._empty_df(X.index)

        Xnum = pd.DataFrame(index=X.index)
        for col in self.cluster_features_:
            if col in X.columns:
                Xnum[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                Xnum[col] = self.fill_values_.get(col, 0.0)

        Xnum = Xnum.fillna(
            self.fill_values_.reindex(self.cluster_features_).fillna(0.0)
        )
        X_scaled = self.scaler_.transform(Xnum[self.cluster_features_])

        feats: dict[str, np.ndarray] = {}

        for strat, model in self.models_.items():
            if strat == "kmeans":
                feats["kmeans_id"] = model.predict(X_scaled)
                dist = model.transform(X_scaled)
                for i in range(dist.shape[1]):
                    feats[f"kmeans_dist_{i}"] = dist[:, i]

            elif strat == "gmm":
                feats["gmm_id"] = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)
                for i in range(probs.shape[1]):
                    feats[f"gmm_prob_{i}"] = probs[:, i]

            elif strat == "hdbscan":
                try:
                    import hdbscan as _hdbscan

                    labels, strengths = _hdbscan.approximate_predict(model, X_scaled)
                    feats["hdbscan_id"] = labels
                    feats["hdbscan_strength"] = strengths
                except Exception:
                    continue

            elif strat == "umap":
                embedding = model.transform(X_scaled)
                for i in range(embedding.shape[1]):
                    feats[f"umap_{i}"] = embedding[:, i]

        return pd.DataFrame(feats, index=X.index, dtype=np.float32)
