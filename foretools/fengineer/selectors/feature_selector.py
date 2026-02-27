import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from foretools.aux.adaptive_mi import AdaptiveMI
from .rfecv import AdvancedRFECV, RFECVConfig


class FeatureSelector:
    def __init__(self, config: Any):
        self.config = config
        self.mi_scores_: Optional[pd.Series] = None
        self.shap_scores_: Optional[pd.Series] = None
        self.selected_features_: List[str] = []

        # Selector integration
        self.selector_method = str(getattr(config, "selector_method", "mi")).lower()
        self.use_rfecv = getattr(config, 'use_rfecv', True)
        self.rfecv_selector_ = None
        self.selection_method_ = "mi"  # Default to MI
        
        # RFECV parameters from config
        self.rfecv_params = {
            'step': getattr(config, 'rfecv_step', 0.1),
            'cv': getattr(config, 'rfecv_cv', 5),
            'min_features_to_select': getattr(config, 'rfecv_min_features', None),
            'max_features_to_select': getattr(config, 'rfecv_max_features', None),
            'patience': getattr(config, 'rfecv_patience', 5),
            'use_ensemble': getattr(config, 'rfecv_use_ensemble', True),
            'stability_selection': getattr(config, 'rfecv_stability_selection', True),
            'verbose': 0,
            'random_state': getattr(config, 'random_state', 42)
        }

        self.ami_scorer = AdaptiveMI(
            subsample=min(getattr(config, "max_rows_score", 2000), 2000),
            spearman_gate=getattr(config, "mi_spearman_gate", 0.05),
            min_overlap=getattr(config, "mi_min_overlap", 50),
            ks=(3, 5, 10),
            n_bins=getattr(config, "mi_bins", 16),
            random_state=getattr(config, "random_state", 42),
        )

        self._feature_cache: Dict[str, float] = {}
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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """Fit feature selector with configurable method selection."""
        method = self._resolve_selector_method()

        if method == "boruta":
            print("🌲 Running Boruta feature selection...")
            success = self._fit_boruta(X, y)
            if success:
                self.selection_method_ = "boruta"
                return self
            print("   ⚠️  Boruta failed, falling back to MI selection")

        elif method == "rfecv":
            print("🔄 Using RFECV feature selection...")
            success = self._fit_rfecv(X, y)
            if success:
                self.selection_method_ = "rfecv"
                return self
            print("   ⚠️  RFECV failed, falling back to MI selection")

        # Fallback/default MI-based selection
        print("📊 Using Mutual Information feature selection...")
        self.selection_method_ = "mi"
        return self._fit_mi(X, y)

    def _resolve_selector_method(self) -> str:
        """Resolve selector strategy with backward compatibility."""
        allowed = {"auto", "mi", "rfecv", "boruta"}
        method = self.selector_method
        if method not in allowed:
            warnings.warn(
                f"Unknown selector_method='{method}'. Falling back to 'auto'."
            )
            method = "auto"

        # Preserve historical behavior when using auto:
        # boruta (if enabled) -> rfecv (if enabled) -> mi
        if method == "auto":
            if getattr(self.config, "use_boruta", False):
                return "boruta"
            if self.use_rfecv:
                return "rfecv"
            return "mi"

        # Explicit method overrides legacy booleans.
        return method

    def _fit_boruta(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Fit Boruta selector."""
        try:
            from foretools.fengineer.selectors.boruta import BorutaSelector
            
            # Prepare data (Boruta needs numerical and no NaNs)
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) < 2: return False
            
            X_numeric = X[numerical_cols].fillna(X[numerical_cols].median())
            y_clean = y.fillna(y.mode()[0] if self._is_classification_target(y) else y.median())
            
            common_idx = X_numeric.index.intersection(y_clean.index)
            X_final = X_numeric.loc[common_idx]
            y_final = y_clean.loc[common_idx]
            
            selector = BorutaSelector(
                max_iter=getattr(self.config, 'boruta_max_iter', 20),
                random_state=getattr(self.config, 'random_state', 42),
                verbose=0
            )
            selector.fit(X_final, y_final)
            self.selected_features_ = selector.get_selected_features()
            self.boruta_selector_ = selector
            
            print(f"   ✅ Boruta selected {len(self.selected_features_)} features")
            return True
        except Exception as e:
            warnings.warn(f"Boruta selection failed: {e}")
            return False
  
    def _fit_rfecv(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Fit RFECV selector."""
        try:
            # Auto-calculate parameters if not specified
            n_features = X.shape[1]
            if self.rfecv_params['min_features_to_select'] is None:
                self.rfecv_params['min_features_to_select'] = max(1, n_features // 20)
            if self.rfecv_params['max_features_to_select'] is None:
                self.rfecv_params['max_features_to_select'] = min(100, n_features)
            
            # Create RFECV config
            rfecv_config = RFECVConfig(**self.rfecv_params)
            
            # Initialize and fit RFECV
            self.rfecv_selector_ = AdvancedRFECV(config=rfecv_config)
            
            # Use only numerical features for RFECV
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) < 2:
                return False
                
            X_numeric = X[numerical_cols].copy()
            X_filled = X_numeric.fillna(X_numeric.median())
            
            # Clean target variable - handle NaN values
            y_clean = y.copy()
            
            # Handle missing values in target
            if y_clean.isna().any():
                if getattr(self.config, "task", "regression") == "classification":
                    # For classification, use mode
                    mode_val = y_clean.mode()
                    if len(mode_val) > 0:
                        y_clean = y_clean.fillna(mode_val[0])
                    else:
                        # If no mode, drop NaN rows
                        valid_mask = y_clean.notna()
                        y_clean = y_clean[valid_mask]
                        X_filled = X_filled.loc[valid_mask]
                else:
                    # For regression, use median
                    median_val = y_clean.median()
                    if not pd.isna(median_val):
                        y_clean = y_clean.fillna(median_val)
                    else:
                        # If no median, drop NaN rows
                        valid_mask = y_clean.notna()
                        y_clean = y_clean[valid_mask]
                        X_filled = X_filled.loc[valid_mask]
            
            # Ensure we have enough data after cleaning
            if len(X_filled) < 10 or len(y_clean) < 10:
                return False
            
            # Align indices
            common_idx = X_filled.index.intersection(y_clean.index)
            if len(common_idx) < 10:
                return False
                
            X_final = X_filled.loc[common_idx]
            y_final = y_clean.loc[common_idx]
            
            # Final check for any remaining NaN values
            if X_final.isna().any().any() or y_final.isna().any():
                # Drop any remaining NaN rows
                combined_df = pd.concat([X_final, y_final], axis=1)
                combined_clean = combined_df.dropna()
                
                if len(combined_clean) < 10:
                    return False
                    
                X_final = combined_clean.iloc[:, :-1]
                y_final = combined_clean.iloc[:, -1]
            
            self.rfecv_selector_.fit(X_final, y_final)
            
            # Get selected features
            self.selected_features_ = self.rfecv_selector_.get_selected_features()
            
            print(f"   ✅ RFECV selected {len(self.selected_features_)} features")
            return True
            
        except Exception as e:
            warnings.warn(f"RFECV selection failed: {e}")
            return False
    
    def _fit_mi(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """Fit MI selector (original implementation)."""
        # Get numerical columns once
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            self.selected_features_ = []
            return self

        # Prepare data more efficiently
        X_clean, y_clean = self._prepare_data_fast(X[numerical_cols], y)

        if len(X_clean) < getattr(self.config, "min_samples", 10):
            self.selected_features_ = numerical_cols.tolist()
            return self

        # Compute MI scores (stable across folds if enabled)
        if self.use_stable_mi:
            self.mi_scores_ = self._compute_mi_scores_stable(X_clean, y_clean)
        else:
            self.mi_scores_ = self._compute_mi_scores_fast(X_clean, y_clean)

        # Feature selection
        mi_threshold = getattr(self.config, "mi_threshold", 0.01)
        selected_mask = self.mi_scores_ > mi_threshold

        if selected_mask.any():
            self.selected_features_ = self.mi_scores_[selected_mask].index.tolist()
        else:
            # Fallback to top features
            min_features = getattr(self.config, "min_features", 1)
            self.selected_features_ = self.mi_scores_.head(min_features).index.tolist()

        # Optional redundancy pruning on top-MI pool
        if self.selector_redundancy_prune and self.selected_features_:
            self.selected_features_ = self._prune_redundant_features(
                X_clean, self.selected_features_
            )

        # Limit max features
        max_features = getattr(self.config, "max_features", len(numerical_cols))
        if len(self.selected_features_) > max_features:
            self.selected_features_ = self.selected_features_[:max_features]

        return self

    def _prepare_data_fast(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        # Fast index alignment
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]

        # Vectorized validity check
        if getattr(self.config, "task", "regression") == "classification":
            if not pd.api.types.is_numeric_dtype(y_aligned):
                le = LabelEncoder()
                y_clean = pd.Series(le.fit_transform(y_aligned), index=y_aligned.index)
            else:
                y_clean = y_aligned.astype("int32")
        else:
            y_clean = pd.to_numeric(y_aligned, errors="coerce")

        # Remove invalid rows in one operation
        valid_mask = y_clean.notna() & np.isfinite(y_clean)
        if not valid_mask.any():
            return pd.DataFrame(), pd.Series(dtype=float)

        X_clean = X_aligned[valid_mask].astype("float32", copy=False)
        y_clean = y_clean[valid_mask]

        return X_clean, y_clean

    def _compute_mi_scores_fast(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        # Convert to numpy arrays once
        X_values = X.values
        y_values = y.values

        # Batch MI computation
        scores = self.ami_scorer.score_pairwise(X_values, y_values)

        # Create series and sort in one step
        mi_scores = pd.Series(scores, index=X.columns)
        mi_scores = mi_scores.fillna(0.0).clip(lower=0.0).sort_values(ascending=False)

        return mi_scores

    @staticmethod
    def _is_classification_target(y: pd.Series) -> bool:
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            return True
        n = max(1, len(y))
        k = y.nunique(dropna=True)
        return (k <= 20) or (k / n < 0.05)

    def _compute_mi_scores_stable(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Compute MI with fold stability aggregation."""
        n = len(X)
        cv = max(2, min(self.selector_cv, max(2, n // 20)))
        if cv < 2:
            return self._compute_mi_scores_fast(X, y)

        if self._is_classification_target(y):
            splitter = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=getattr(self.config, "random_state", 42)
            )
            split_iter = splitter.split(X, y)
        else:
            splitter = KFold(
                n_splits=cv, shuffle=True, random_state=getattr(self.config, "random_state", 42)
            )
            split_iter = splitter.split(X)

        names = list(X.columns)
        fold_scores: Dict[str, List[float]] = {c: [] for c in names}
        freq: Dict[str, int] = {c: 0 for c in names}

        for tr_idx, _ in split_iter:
            Xf = X.iloc[tr_idx]
            yf = y.iloc[tr_idx]
            scores = self.ami_scorer.score_pairwise(Xf.values, yf.values)
            s = pd.Series(scores, index=names).fillna(0.0).clip(lower=0.0)
            for c, v in s.items():
                if np.isfinite(v):
                    fold_scores[c].append(float(v))
                    if v > 0:
                        freq[c] += 1

        min_freq = int(np.ceil(self.selector_min_freq * cv))
        agg = {}
        for c in names:
            if not fold_scores[c]:
                agg[c] = 0.0
                continue
            if freq[c] < min_freq:
                agg[c] = 0.0
                continue
            agg[c] = float(np.median(fold_scores[c]))

        mi_scores = pd.Series(agg, index=names).fillna(0.0).clip(lower=0.0)
        return mi_scores.sort_values(ascending=False)

    def _prune_redundant_features(
        self, X: pd.DataFrame, ranked_features: List[str]
    ) -> List[str]:
        """Greedy correlation pruning while preserving MI ranking order."""
        if len(ranked_features) < 2:
            return ranked_features

        pool = ranked_features[: max(2, self.selector_redundancy_pool)]
        corr_thr = self.selector_redundancy_threshold
        Xp = X[pool]
        corr = Xp.corr().abs().fillna(0.0)

        kept: List[str] = []
        for f in pool:
            drop = False
            for k in kept:
                if corr.at[f, k] >= corr_thr:
                    drop = True
                    break
            if not drop:
                kept.append(f)

        # Keep original order for any features outside pool
        tail = [f for f in ranked_features if f not in pool]
        return kept + tail

    def get_selected_features(self) -> List[str]:
        return self.selected_features_.copy()

    def get_feature_scores(self) -> Optional[pd.Series]:
        """Get feature importance scores from the selection method used."""
        if self.selection_method_ == "rfecv" and self.rfecv_selector_ is not None:
            # Get importance scores from RFECV
            if hasattr(self.rfecv_selector_, 'feature_importances_'):
                # Get the features that were actually used in RFECV
                numerical_cols = self.rfecv_selector_._feature_names
                if numerical_cols and len(self.rfecv_selector_.feature_importances_) == len(numerical_cols):
                    scores = pd.Series(
                        self.rfecv_selector_.feature_importances_, 
                        index=numerical_cols
                    )
                    return scores.sort_values(ascending=False)
        
        # Fallback to MI scores
        if hasattr(self, 'mi_scores_') and self.mi_scores_ is not None:
            return self.mi_scores_
        
        return None

    def get_top_features(self, n: int = 10) -> List[str]:
        if self.mi_scores_ is None:
            return self.selected_features_[:n]
        return self.mi_scores_.head(n).index.tolist()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            return pd.DataFrame(index=X.index)

        # Fast column filtering
        available_features = [f for f in self.selected_features_ if f in X.columns]

        if not available_features:
            return pd.DataFrame(index=X.index)

        return X[available_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
