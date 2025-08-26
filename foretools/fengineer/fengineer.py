import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from foretools.aux.adaptive_mi import AdaptiveMI
from foretools.fengineer.rfecv import *
from foretools.fengineer.transformers import *

# Type hints for better code clarity
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]
ModelLike = Any



class CorrelationFilter:
    """Removes highly correlated features with multiple selection strategies."""

    def __init__(
        self,
        threshold: float = 0.95,
        method: str = "variance",
        min_features: int = 2,
        handle_missing: bool = True,
    ):
        """
        Args:
            threshold: Correlation threshold above which to remove features
            method: Strategy for choosing which feature to drop ('variance', 'target_corr', 'random')
            min_features: Minimum number of features to keep
            handle_missing: Whether to handle missing values before correlation
        """
        self.threshold = threshold
        self.method = method
        self.min_features = min_features
        self.handle_missing = handle_missing
        self.features_to_drop_: List[str] = []
        self.correlation_pairs_: List[Tuple[str, str, float]] = []
        self.feature_rankings_: Dict[str, float] = {}

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CorrelationFilter":
        """Fit correlation filter."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) < 2:
            self.features_to_drop_ = []
            return self

        try:
            # Handle missing values if requested
            X_corr = X[numerical_cols].copy()
            if self.handle_missing:
                X_corr = X_corr.fillna(X_corr.median())

            # Calculate correlation matrix
            corr_matrix = X_corr.corr()

            # Handle NaN correlations (constant features)
            corr_matrix = corr_matrix.fillna(0)

            # Find highly correlated pairs
            upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            high_corr_mask = (corr_matrix.abs() > self.threshold) & upper_tri
            high_corr_pairs = np.where(high_corr_mask)

            # Store correlation pairs for inspection
            self.correlation_pairs_ = []
            feature_drop_scores = {}

            for i, j in zip(*high_corr_pairs):
                col1, col2 = corr_matrix.index[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                self.correlation_pairs_.append((col1, col2, abs(corr_value)))

                # Calculate drop scores based on method
                if self.method == "variance":
                    score1 = X_corr[col1].var()
                    score2 = X_corr[col2].var()
                    keep_col = col1 if score1 > score2 else col2
                    drop_col = col2 if score1 > score2 else col1

                elif self.method == "target_corr" and y is not None:
                    # Keep feature more correlated with target
                    try:
                        score1 = abs(X_corr[col1].corr(y))
                        score2 = abs(X_corr[col2].corr(y))
                        keep_col = col1 if score1 > score2 else col2
                        drop_col = col2 if score1 > score2 else col1
                    except:
                        # Fallback to variance if target correlation fails
                        score1 = X_corr[col1].var()
                        score2 = X_corr[col2].var()
                        keep_col = col1 if score1 > score2 else col2
                        drop_col = col2 if score1 > score2 else col1

                else:  # random or fallback
                    drop_col = np.random.choice([col1, col2])
                    keep_col = col1 if drop_col == col2 else col2

                # Track drop scores (higher score = more likely to drop)
                feature_drop_scores[drop_col] = feature_drop_scores.get(drop_col, 0) + 1

            # Sort features by drop frequency and select final drops
            candidate_drops = sorted(
                feature_drop_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Ensure we don't drop too many features
            max_drops = len(numerical_cols) - self.min_features
            self.features_to_drop_ = [col for col, _ in candidate_drops[:max_drops]]

            # Store feature rankings for inspection
            remaining_features = set(numerical_cols) - set(self.features_to_drop_)
            self.feature_rankings_ = {
                col: feature_drop_scores.get(col, 0) for col in remaining_features
            }

        except Exception as e:
            warnings.warn(f"Correlation analysis failed: {e}")
            self.features_to_drop_ = []
            self.correlation_pairs_ = []

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove correlated features."""
        return X.drop(columns=self.features_to_drop_, errors="ignore")
    
# Enhanced SOTA Feature Engineer with RFECV integration
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Comprehensive state-of-the-art feature engineering pipeline.

    Features:
    - Modular transformer architecture
    - Configuration-driven approach
    - Comprehensive feature creation and selection
    - RFECV integration for advanced feature selection
    - Robust error handling
    - Memory efficient processing
    - Extensive logging and monitoring
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.transformers_ = {}
        self.correlation_filter_ = None
        self.selector_ = None
        self.final_scaler_ = None
        self.fitted_ = False

        # Track feature creation statistics
        self.feature_stats_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Fit comprehensive feature engineering pipeline."""
        X = X.copy()
        if y is not None:
            y = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()

        print(
            f"🚀 Fitting SOTA Feature Engineer on {X.shape[0]} rows, {X.shape[1]} columns"
        )

        # Initialize all transformers
        self.transformers_ = {
            #"datetime": DateTimeTransformer(self.config),
            "mathematical": MathematicalTransformer(self.config),
            "interactions": InteractionTransformer(self.config),
            "categorical": CategoricalTransformer(self.config),
            "binning": BinningTransformer(self.config),
            #"clustering": ClusteringTransformer(self.config),
            "statistical": StatisticalTransformer(self.config),
            #'fourier': FourierTransformer(self.config),
            #'autoencoder': AutoencoderTransformer(self.config),
            "rff": RandomFourierFeaturesTransformer(self.config),
        }

        # Fit transformers sequentially
        current_X = X.copy()
        for name, transformer in self.transformers_.items():
            print(f"🔧 Fitting {name} transformer...")
            # try:
            transformer.fit(current_X, y)
            # Update current_X with new features for next transformer
            transformed = transformer.transform(current_X)
            if not transformed.empty:
                current_X = pd.concat([current_X, transformed], axis=1)
                self.feature_stats_[name] = transformed.shape[1]
            else:
                self.feature_stats_[name] = 0


        # Apply correlation filtering
        print("🔍 Applying correlation filtering...")
        self.correlation_filter_ = CorrelationFilter(self.config.corr_threshold)
        self.correlation_filter_.fit(current_X)
        current_X = self.correlation_filter_.transform(current_X)

        # Feature selection
        if y is not None and not current_X.empty:
            print("🎯 Performing feature selection...")
            self.selector_ = FeatureSelector(self.config)
            self.selector_.fit(current_X, y)
            selected_features = self.selector_.get_selected_features()

            if selected_features:
                current_X = current_X[selected_features]

                # Fit final scaler
                if self.config.use_quantile_transform:
                    print("📊 Fitting final quantile transformer...")
                    self.final_scaler_ = QuantileTransformer(
                        n_quantiles=min(1000, max(10, len(current_X) // 2)),
                        output_distribution="normal",
                        random_state=self.config.random_state,
                    )
                    X_filled = current_X.fillna(current_X.median())
                    self.final_scaler_.fit(X_filled)

                print(f"✅ Selected {len(selected_features)} features after filtering")
            else:
                print("⚠️  No features selected - pipeline may need tuning")

        # Print feature creation summary
        self._print_feature_summary()

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted pipeline."""
        check_is_fitted(self, "fitted_")

        current_X = X.copy()

        # Apply all transformers
        for name, transformer in self.transformers_.items():
            try:
                transformed = transformer.transform(current_X)
                if not transformed.empty:
                    current_X = pd.concat([current_X, transformed], axis=1)
            except Exception as e:
                warnings.warn(f"Transform failed for {name}: {e}")

        # Apply correlation filtering
        if self.correlation_filter_ is not None:
            current_X = self.correlation_filter_.transform(current_X)

        # Apply feature selection
        if self.selector_ is not None:
            selected_features = self.selector_.get_selected_features()
            available_features = [
                col for col in selected_features if col in current_X.columns
            ]

            if available_features:
                current_X = current_X[available_features]

                # Apply final scaling
                if self.final_scaler_ is not None:
                    X_filled = current_X.fillna(current_X.median())
                    X_scaled = self.final_scaler_.transform(X_filled)
                    return pd.DataFrame(
                        X_scaled, columns=current_X.columns, index=X.index
                    )

        return current_X if not current_X.empty else pd.DataFrame(index=X.index)

    def _print_feature_summary(self):
        """Print summary of feature creation."""
        print("\n📈 Feature Creation Summary:")
        print("-" * 40)
        total_features = 0
        for name, count in self.feature_stats_.items():
            print(f"{name.title():<15}: {count:>4} features")
            total_features += count
        print("-" * 40)
        print(f"{'Total':<15}: {total_features:>4} features")

        if self.correlation_filter_:
            dropped = len(self.correlation_filter_.features_to_drop_)
            print(f"{'Corr. Dropped':<15}: {dropped:>4} features")

        if self.selector_:
            selected = len(self.selector_.get_selected_features())
            print(f"{'Final Selected':<15}: {selected:>4} features")
            
            # Show selection method used if RFECV was used
            if hasattr(self.selector_, 'selection_method_'):
                print(f"{'Selection Method':<15}: {self.selector_.selection_method_}")
        print()

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance from feature selection analysis."""
        if self.selector_ and hasattr(self.selector_, "get_feature_scores"):
            return self.selector_.get_feature_scores()
        elif self.selector_ and hasattr(self.selector_, "mi_scores_"):
            return self.selector_.mi_scores_
        return None

    def plot_feature_importance(
        self, top_k: int = 20, figsize: Tuple[int, int] = (12, 8)
    ):
        """Plot feature importance with enhanced visualization."""
        import matplotlib.pyplot as plt

        importance_scores = self.get_feature_importance()
        if importance_scores is None:
            print("❌ No feature importance scores available.")
            return

        top_scores = importance_scores.head(top_k)

        plt.figure(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_scores)))
        bars = plt.barh(range(len(top_scores)), top_scores.values, color=colors)

        plt.title(
            f"🎯 Top {top_k} Features by Importance",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Feature Importance Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.yticks(range(len(top_scores)), top_scores.index)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_scores.values)):
            plt.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.grid(axis="x", alpha=0.3)
        plt.show()

    def get_transformation_report(self) -> Dict[str, Any]:
        """Get detailed report of transformations applied."""
        report = {
            "feature_stats": self.feature_stats_.copy(),
            "config": self.config.__dict__.copy(),
            "correlation_dropped": (
                len(self.correlation_filter_.features_to_drop_)
                if self.correlation_filter_
                else 0
            ),
            "selected_features": (
                len(self.selector_.get_selected_features()) if self.selector_ else 0
            ),
        }

        # Add RFECV-specific info if available
        if self.selector_ and hasattr(self.selector_, 'rfecv_selector_') and self.selector_.rfecv_selector_:
            report["rfecv_results"] = self.selector_.rfecv_selector_.get_performance_summary()

        if self.selector_ and hasattr(self.selector_, "get_feature_scores"):
            scores = self.selector_.get_feature_scores()
            if scores is not None:
                report["top_features"] = scores.head(10).to_dict()
        elif self.selector_ and hasattr(self.selector_, "mi_scores_"):
            report["top_features"] = self.selector_.mi_scores_.head(10).to_dict()

        return report

    def plot_rfecv_results(self, **kwargs):
        """Plot RFECV results if RFECV was used."""
        if (self.selector_ and 
            hasattr(self.selector_, 'rfecv_selector_') and 
            self.selector_.rfecv_selector_ is not None):
            self.selector_.rfecv_selector_.plot_cv_scores(**kwargs)
        else:
            print("RFECV was not used or is not available for plotting.")


class FeatureSelector:
    def __init__(self, config: Any):
        self.config = config
        self.mi_scores_: Optional[pd.Series] = None
        self.shap_scores_: Optional[pd.Series] = None
        self.selected_features_: List[str] = []

        # RFECV integration
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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """Fit feature selector with optional RFECV."""
        
        # Determine if we should use RFECV based on dataset characteristics
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # Use RFECV if explicitly enabled or if dataset is suitable
        should_use_rfecv = self.use_rfecv
        
        if should_use_rfecv:
            print("🔄 Using RFECV feature selection...")
            success = self._fit_rfecv(X, y)
            if success:
                self.selection_method_ = "rfecv"
                return self
            else:
                print("   ⚠️  RFECV failed, falling back to MI selection")
        
        # Fallback to original MI-based selection
        print("📊 Using Mutual Information feature selection...")
        self.selection_method_ = "mi"
        return self._fit_mi(X, y)
  
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

        # Compute MI scores
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

