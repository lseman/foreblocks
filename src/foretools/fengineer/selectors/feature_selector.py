"""Multi-stage feature selection pipeline.

Composes single selectors (``MISelector``, ``MRMRSelector``, ``BorutaSelector``,
``AdvancedRFECV``) and runs them in priority order until one succeeds.
Maintains backward-compatible attributes with the legacy :class:`FeatureSelector`
API.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from foretools.aux.adaptive_mi import AdaptiveMI

from .base import FeatureSelectorABC
from .boruta import BorutaSelector
from .mi_selector import MISelector
from .mrmr_selector import MRMRSelector
from .redundancy import RedundancyPruner
from .rfecv import AdvancedRFECV, RFECVConfig

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator (replaces the old monolithic FeatureSelector)
# ──────────────────────────────────────────────────────────────────────────────


class PipelineSelector:
    """
    Multi-stage feature selection pipeline.

    Creates individual selector instances (``MISelector``, ``MRMRSelector``,
    ``BorutaSelector``, ``AdvancedRFECV``) and tries them in priority order
    until one succeeds.  Maintains a backward-compatible attribute set with
    the legacy ``FeatureSelector`` API.

    Example
    -------
    >>> selector = PipelineSelector(config)
    >>> selector.fit(X, y)
    >>> selector.get_selected_features()
    >>> selector.transform(X_test)
    """

    # ── public attrs (exposed for backward compat) ──────────────────────
    selected_features_: list[str]
    selection_method_: str
    mi_scores_: pd.Series | None
    mrmr_scores_: pd.Series | None
    rfecv_selector_: AdvancedRFECV | None
    boruta_selector_: BorutaSelector | None
    mrmr_selector_: AdaptiveMRMR | None
    logger: logging.Logger

    # ── init ────────────────────────────────────────────────────────────

    def __init__(self, config: Any) -> None:
        self.config = config
        self.selected_features_ = []
        self.selection_method_ = "mi"
        self.mi_scores_ = None
        self.mrmr_scores_ = None
        self.rfecv_selector_ = None
        self.boruta_selector_ = None
        self.mrmr_selector_ = None

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        level = getattr(
            logging,
            str(getattr(config, "log_level", "INFO")).upper(),
            logging.INFO,
        )
        self.logger.setLevel(level)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(h)

        # Selector choice
        self.selector_method = str(getattr(config, "selector_method", "mi")).lower()
        self.use_rfecv = bool(getattr(config, "use_rfecv", True))

        # MI / mRMR shared config
        self._ami_scorer = AdaptiveMI(
            subsample=min(getattr(config, "max_rows_score", 2000), 2000),
            spearman_gate=getattr(config, "mi_spearman_gate", 0.05),
            min_overlap=getattr(config, "mi_min_overlap", 50),
            ks=(3, 5, 10),
            n_bins=getattr(config, "mi_bins", 16),
            random_state=getattr(config, "random_state", 42),
        )
        self.use_stable_mi = bool(getattr(config, "selector_stable_mi", True))
        self.selector_cv = int(getattr(config, "selector_cv", 5))
        self.selector_min_freq = float(getattr(config, "selector_min_freq", 0.5))
        self.selector_redundancy_prune = bool(
            getattr(config, "selector_redundancy_prune", True)
        )
        self.selector_redundancy_threshold = float(
            getattr(config, "selector_redundancy_threshold", 0.98)
        )
        self.selector_redundancy_pool = int(
            getattr(config, "selector_redundancy_pool", 200)
        )

        # RFECV params
        self._rfecv_params: dict[str, Any] = {
            "step": getattr(config, "rfecv_step", 0.1),
            "cv": getattr(config, "rfecv_cv", 5),
            "min_features_to_select": getattr(config, "rfecv_min_features", None),
            "max_features_to_select": getattr(config, "rfecv_max_features", None),
            "patience": getattr(config, "rfecv_patience", 5),
            "use_ensemble": getattr(config, "rfecv_use_ensemble", True),
            "stability_selection": getattr(config, "rfecv_stability_selection", True),
            "verbose": 0,
            "random_state": getattr(config, "random_state", 42),
        }

        # mRMR params
        self._mrmr_criterion = str(getattr(config, "mrmr_criterion", "mid")).lower()
        self._mrmr_candidate_pool = int(
            getattr(config, "mrmr_candidate_pool", self.selector_redundancy_pool)
        )
        self._mrmr_redundancy_weight = float(
            getattr(config, "mrmr_redundancy_weight", 1.0)
        )
        self._mrmr_redundancy_eps = float(getattr(config, "mrmr_redundancy_eps", 1e-8))
        self._mrmr_use_raw_mi = bool(getattr(config, "mrmr_use_raw_mi", False))

    # ── fit / transform ─────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PipelineSelector":
        """Fit the pipeline, trying selectors in priority order."""
        n_features = X.shape[1]
        method = self._resolve_method(n_features)

        # 1. Boruta
        if method == "boruta":
            self.logger.info("🌲 Running Boruta feature selection...")
            if self._try_boruta(X, y):
                self.selection_method_ = "boruta"
                return self
            self.logger.warning("   ⚠️  Boruta failed, falling back")

        # 2. mRMR
        if method == "mrmr":
            self.logger.info("🧠 Using AdaptiveMI mRMR selection...")
            self.selection_method_ = "mrmr"
            self._try_mrmr(X, y)
            return self

        # 3. RFECV
        if method == "rfecv":
            self.logger.info("🔄 Using RFECV selection...")
            if self._try_rfecv(X, y):
                self.selection_method_ = "rfecv"
                return self
            self.logger.warning("   ⚠️  RFECV failed, falling back")

        # 4. MI (default fallback)
        self.logger.info("📊 Using Mutual Information selection...")
        self.selection_method_ = "mi"
        self._try_mi(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with only selected columns."""
        avail = [f for f in self.selected_features_ if f in X.columns]
        if not avail:
            return pd.DataFrame(index=X.index)
        return X[avail]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self) -> list[str]:
        return list(self.selected_features_)

    def get_scores(self) -> pd.Series | None:
        """Return scores from the active selector."""
        method = self.selection_method_
        if method == "rfecv" and self.rfecv_selector_ is not None:
            scores = self.rfecv_selector_.get_scores()
            if scores is not None:
                return scores
        if method == "mrmr" and self.mrmr_selector_ is not None:
            return self.mrmr_selector_.get_scores()
        return self.mi_scores_

    def get_feature_scores(self) -> pd.Series | None:
        """Alias of :meth:`get_scores`."""
        return self.get_scores()

    def get_top_features(self, n: int = 10) -> list[str]:
        scores = self.get_scores()
        if scores is None or scores.empty:
            return self.selected_features_[:n]
        return list(scores.head(n).index)

    # ── selector factories ──────────────────────────────────────────────

    def _make_mi_selector(self, X: pd.DataFrame, y: pd.Series) -> MISelector:
        return MISelector(
            config=self.config,
            scorer=self._ami_scorer,
            threshold=float(getattr(self.config, "mi_threshold", 0.01)),
            use_stable=self.use_stable_mi,
            cv=self.selector_cv,
            min_freq=self.selector_min_freq,
            redundancy_pruner=(
                RedundancyPruner(
                    threshold=self.selector_redundancy_threshold,
                    pool_size=self.selector_redundancy_pool,
                )
                if self.selector_redundancy_prune
                else None
            ),
            min_features=int(getattr(self.config, "min_features", 1)),
            max_features=int(getattr(self.config, "max_features", X.shape[1])),
            task=str(getattr(self.config, "task", "regression")),
            random_state=int(getattr(self.config, "random_state", 42)),
        )

    def _make_mrmr_selector(self, X: pd.DataFrame, y: pd.Series) -> MRMRSelector:
        return MRMRSelector(
            config=self.config,
            scorer=self._ami_scorer,
            criterion=self._mrmr_criterion,
            candidate_pool=self._mrmr_candidate_pool,
            redundancy_weight=self._mrmr_redundancy_weight,
            redundancy_eps=self._mrmr_redundancy_eps,
            use_raw_mi=self._mrmr_use_raw_mi,
            stable_relevance=self.use_stable_mi,
            cv=self.selector_cv,
            min_freq=self.selector_min_freq,
            task=str(getattr(self.config, "task", "regression")),
            random_state=int(getattr(self.config, "random_state", 42)),
        )

    def _make_boruta_selector(self) -> BorutaSelector:
        return BorutaSelector(
            max_iter=int(getattr(self.config, "boruta_max_iter", 20)),
            random_state=int(getattr(self.config, "random_state", 42)),
            verbose=0,
        )

    def _make_rfecv_selector(self) -> AdvancedRFECV:
        cfg = RFECVConfig(**{k: v for k, v in self._rfecv_params.items()})
        return AdvancedRFECV(config=cfg)

    # ── try blocks ──────────────────────────────────────────────────────

    def _try_boruta(self, X: pd.DataFrame, y: pd.Series) -> bool:
        try:
            num = X.select_dtypes(include=[np.number]).columns
            if len(num) < 2:
                return False
            X_num = X[num].fillna(X[num].median())
            y_clean = self._clean_target(y)
            idx = X_num.index.intersection(y_clean.index)
            if len(idx) < 10:
                return False
            sel = self._make_boruta_selector()
            sel.fit(X_num.loc[idx], y_clean.loc[idx])
            self.selected_features_ = sel.get_selected_features()
            self.boruta_selector_ = sel
            self.logger.info(
                f"   ✅ Boruta selected {len(self.selected_features_)} features"
            )
            return True
        except Exception as e:
            self.logger.warning(f"   Boruta failed: {e}")
            return False

    def _try_mrmr(self, X: pd.DataFrame, y: pd.Series) -> None:
        num = X.select_dtypes(include=[np.number]).columns
        if not num:
            return
        sel = self._make_mrmr_selector(X, y)
        sel.fit(X[num], y)
        self.selected_features_ = sel.get_selected_features()
        self.mrmr_selector_ = sel._mrmr  # underlying AdaptiveMRMR for compat
        self.mi_scores_ = sel.get_relevance_scores()
        self.mrmr_scores_ = sel.get_scores()

    def _try_rfecv(self, X: pd.DataFrame, y: pd.Series) -> bool:
        try:
            num = X.select_dtypes(include=[np.number]).columns
            if len(num) < 2:
                return False
            X_num = X[num].copy().fillna(X[num].median())
            y_clean = self._clean_target(y)
            idx = X_num.index.intersection(y_clean.index)
            if len(idx) < 10:
                return False
            X_num, y_clean = X_num.loc[idx], y_clean.loc[idx]

            # Drop remaining NaN rows
            combined = pd.concat([X_num, y_clean], axis=1).dropna()
            if len(combined) < 10:
                return False
            X_num = combined.iloc[:, :-1]
            y_clean = combined.iloc[:, -1]

            sel = self._make_rfecv_selector()
            sel.fit(X_num, y_clean)
            self.selected_features_ = sel.get_selected_features()
            self.rfecv_selector_ = sel
            self.logger.info(
                f"   ✅ RFECV selected {len(self.selected_features_)} features"
            )
            return True
        except Exception as e:
            self.logger.warning(f"   RFECV failed: {e}")
            return False

    def _try_mi(self, X: pd.DataFrame, y: pd.Series) -> None:
        num = X.select_dtypes(include=[np.number]).columns.tolist()
        if not num:
            return
        sel = self._make_mi_selector(X, y)
        sel.fit(X[num], y)
        self.selected_features_ = sel.get_selected_features()
        self.mi_scores_ = sel.get_scores()

    # ── helpers ─────────────────────────────────────────────────────────

    def _resolve_method(self, n_features: int) -> str:
        """Resolve which selector to try first."""
        allowed = {"auto", "mi", "mrmr", "rfecv", "boruta"}
        method = self.selector_method
        if method not in allowed:
            warnings.warn(
                f"Unknown selector_method='{method}'. Falling back to 'auto'."
            )
            method = "auto"

        if method == "boruta":
            return "boruta"

        if method == "auto":
            if getattr(self.config, "use_boruta", False):
                return "boruta"

            backend = str(getattr(self.config, "backend", "auto")).lower()
            cap = int(getattr(self.config, "selector_auto_rfecv_max_features", 80))

            if backend in {"linear", "neural"}:
                resolved = str(
                    getattr(self.config, f"selector_auto_{backend}_method", "mrmr")
                ).lower()
                if resolved == "rfecv":
                    return "rfecv" if self.use_rfecv and n_features <= cap else "mrmr"
                return resolved

            if backend in {"tree", "gbdt"}:
                resolved = str(
                    getattr(self.config, "selector_auto_tree_method", "mi")
                ).lower()
                if resolved == "rfecv":
                    return "rfecv" if self.use_rfecv and n_features <= cap else "mi"
                return resolved

            return "mi"

        return method

    @staticmethod
    def _clean_target(y: pd.Series) -> pd.Series:
        y = y.copy()
        if y.isna().any():
            task = str(getattr(y, "_task", "regression"))
            if task == "classification":
                mode_val = y.mode()
                y = y.fillna(mode_val[0]) if len(mode_val) > 0 else y.dropna()
            else:
                med = y.median()
                y = y.fillna(med) if not pd.isna(med) else y.dropna()
        return y

    @staticmethod
    def _is_classification(y: pd.Series) -> bool:
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            return True
        k = y.nunique(dropna=True)
        n = max(1, len(y))
        return (k <= 20) or (k / n < 0.05)


# Backward-compatibility alias
FeatureSelector = PipelineSelector
