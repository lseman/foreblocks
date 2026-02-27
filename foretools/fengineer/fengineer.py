import warnings
import inspect
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from foretools.fengineer.filters import CorrelationFilter
from foretools.fengineer.selectors import FeatureSelector
from foretools.fengineer.transformers import (
    BinningTransformer,
    CategoricalTransformer,
    FeatureConfig,
    InteractionTransformer,
    MathematicalTransformer,
    RandomFourierFeaturesTransformer,
    StatisticalTransformer,
)

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
        self.output_features_ = []
        self.output_fill_values_ = pd.Series(dtype="float64")
        self.fitted_ = False

        # Track feature creation statistics
        self.feature_stats_ = {}

    def _build_transformers(self) -> Dict[str, Any]:
        """Build ordered transformer registry from config flags."""
        registry = {
            "mathematical": (
                getattr(self.config, "create_math_features", True),
                MathematicalTransformer(self.config),
            ),
            "interactions": (
                getattr(self.config, "create_interactions", True),
                InteractionTransformer(self.config),
            ),
            "categorical": (
                getattr(self.config, "create_categorical", True),
                CategoricalTransformer(self.config),
            ),
            "binning": (
                getattr(self.config, "create_binning", True),
                BinningTransformer(self.config),
            ),
            "statistical": (
                getattr(self.config, "create_statistical", True),
                StatisticalTransformer(self.config),
            ),
            "rff": (
                getattr(self.config, "create_rff", False),
                RandomFourierFeaturesTransformer(
                    self.config,
                    n_components=getattr(self.config, "rff_n_components", 50),
                    gamma=getattr(self.config, "rff_gamma", "auto"),
                    kernel=getattr(self.config, "rff_kernel", "rbf"),
                    max_features=getattr(self.config, "rff_max_features", 50),
                    handle_missing_features=getattr(
                        self.config, "rff_handle_missing_features", "impute"
                    ),
                ),
            ),
        }
        return {name: tf for name, (enabled, tf) in registry.items() if enabled}

    @staticmethod
    def _transform_with_optional_y(
        transformer: Any, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> pd.DataFrame:
        """Pass y only for transformers that support it (e.g., target-kfold categorical)."""
        if y is None:
            return transformer.transform(X)
        try:
            sig = inspect.signature(transformer.transform)
            if "y" in sig.parameters:
                return transformer.transform(X, y=y)
        except (TypeError, ValueError):
            pass
        return transformer.transform(X)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Fit comprehensive feature engineering pipeline."""
        X = X.copy()
        if y is not None:
            y = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()

        print(
            f"🚀 Fitting SOTA Feature Engineer on {X.shape[0]} rows, {X.shape[1]} columns"
        )

        # Initialize enabled transformers in a deterministic order
        self.transformers_ = self._build_transformers()

        # Fit transformers sequentially
        current_X = X.copy()
        for name, transformer in self.transformers_.items():
            print(f"🔧 Fitting {name} transformer...")
            # try:
            transformer.fit(current_X, y)
            # Update current_X with new features for next transformer
            transformed = self._transform_with_optional_y(transformer, current_X, y)
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
        self.output_features_ = current_X.columns.tolist()
        # Keep training-time fill values so transform does not depend on tiny batch stats.
        self.output_fill_values_ = current_X.median(numeric_only=True)
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
            if selected_features:
                missing_selected = [c for c in selected_features if c not in current_X.columns]
                if missing_selected:
                    current_X = current_X.copy()
                    for col in missing_selected:
                        current_X[col] = np.nan
                current_X = current_X[selected_features]

        # Enforce fitted output schema/order for robust inference on small/unseen batches.
        if self.output_features_:
            missing_output = [c for c in self.output_features_ if c not in current_X.columns]
            if missing_output:
                current_X = current_X.copy()
                for col in missing_output:
                    current_X[col] = np.nan
            current_X = current_X[self.output_features_]
            current_X = current_X.apply(pd.to_numeric, errors="coerce")

            fill_values = self.output_fill_values_.reindex(self.output_features_)
            fill_values = fill_values.fillna(0.0)
            current_X = current_X.fillna(fill_values)

        # Apply final scaling
        if self.final_scaler_ is not None and not current_X.empty:
            X_scaled = self.final_scaler_.transform(current_X)
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
