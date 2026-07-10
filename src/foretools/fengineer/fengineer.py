import inspect
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from foretools.fengineer.filters import CorrelationFilter
from foretools.fengineer.selectors import FeatureSelector
from foretools.fengineer.transformers import (
    AutoencoderConfig,
    AutoencoderTransformer,
    BinningTransformer,
    CategoricalTransformer,
    ClusteringTransformer,
    DateTimeTransformer,
    FeatureConfig,
    FourierTransformer,
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

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.transformers_ = {}
        self.correlation_filter_ = None
        self.selector_ = None
        self.final_scaler_ = None
        self.autoencoder_ = None
        self.output_features_ = []
        self.output_fill_values_ = pd.Series(dtype="float64")
        self.fitted_ = False

        # Track feature creation statistics
        self.feature_stats_ = {}

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(
            getattr(logging, str(self.config.log_level).upper(), logging.INFO)
        )
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def _resolve_backend(self) -> str:
        backend = str(getattr(self.config, "backend", "auto")).lower()
        allowed = {"auto", "linear", "tree", "gbdt", "neural"}
        if backend not in allowed:
            self._log(
                f"Unknown backend '{backend}', falling back to 'auto'",
                level="warning",
            )
            return "auto"
        return backend

    def _log(self, message: str, level: str = "info") -> None:
        if not getattr(self.config, "verbose", True):
            return
        log_fn = getattr(self.logger, level, self.logger.info)
        log_fn(f"[FeatureEngineer] {message}")

    def _is_transformer_enabled(self, name: str, flag: str) -> bool:
        if not getattr(self.config, flag, False):
            return False

        backend = self._resolve_backend()
        if backend in {"tree", "gbdt"} and name in {
            "interactions",
            "rff",
            "fourier",
            "clustering",
        }:
            return False
        if backend == "neural" and name == "binning":
            return False
        return True

    def _should_use_quantile_transform(self) -> bool:
        if not getattr(self.config, "use_quantile_transform", True):
            return False
        return self._resolve_backend() not in {"tree", "gbdt"}

    def _build_transformers(self) -> dict[str, Any]:
        """Build ordered transformer registry from config flags."""
        registry = {
            "datetime": (
                self._is_transformer_enabled("datetime", "create_datetime"),
                DateTimeTransformer(
                    self.config,
                    include_cyclical=getattr(
                        self.config, "datetime_include_cyclical", True
                    ),
                    include_flags=getattr(self.config, "datetime_include_flags", True),
                    include_elapsed=getattr(
                        self.config, "datetime_include_elapsed", True
                    ),
                    group_key=getattr(self.config, "datetime_group_key", None),
                    country_holidays=getattr(
                        self.config, "datetime_country_holidays", None
                    ),
                ),
            ),
            "categorical": (
                self._is_transformer_enabled("categorical", "create_categorical"),
                CategoricalTransformer(self.config),
            ),
            "fourier": (
                self._is_transformer_enabled("fourier", "create_fourier"),
                FourierTransformer(self.config),
            ),
            "mathematical": (
                self._is_transformer_enabled("mathematical", "create_math_features"),
                MathematicalTransformer(self.config),
            ),
            "binning": (
                self._is_transformer_enabled("binning", "create_binning"),
                BinningTransformer(self.config),
            ),
            "statistical": (
                self._is_transformer_enabled("statistical", "create_statistical"),
                StatisticalTransformer(self.config),
            ),
            "interactions": (
                self._is_transformer_enabled("interactions", "create_interactions"),
                InteractionTransformer(self.config),
            ),
            "rff": (
                self._is_transformer_enabled("rff", "create_rff"),
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
            "clustering": (
                self._is_transformer_enabled("clustering", "create_clustering"),
                ClusteringTransformer(self.config),
            ),
        }
        # Autoencoder (if enabled) - special: compresses all features into latent space
        if getattr(self.config, "use_autoencoder", False):
            try:
                ae_latent_dim = int(getattr(self.config, "ae_latent_dim", 8))
            except Exception:
                ae_latent_dim = 8
            ae_encoder_raw = self._parse_list_str(
                getattr(self.config, "ae_encoder_arch", "64,32")
            )
            ae_decoder_raw = self._parse_list_str(
                getattr(self.config, "ae_decoder_arch", "32,64")
            )

            # Ensure encoder ends with latent_dim
            if ae_encoder_raw and ae_encoder_raw[-1] != ae_latent_dim:
                ae_encoder_raw = ae_encoder_raw + [ae_latent_dim]
            elif not ae_encoder_raw:
                ae_encoder_raw = [
                    max(8, ae_latent_dim * 4),
                    max(4, ae_latent_dim * 2),
                    ae_latent_dim,
                ]

            # Ensure decoder starts from latent_dim and ends at input_dim
            if ae_decoder_raw and ae_decoder_raw[0] != ae_latent_dim:
                ae_decoder_raw = [ae_latent_dim] + ae_decoder_raw

            ae_cfg = AutoencoderConfig(
                enabled=True,
                latent_dim=ae_latent_dim,
                encoder_arch=ae_encoder_raw,
                decoder_arch=ae_decoder_raw,
                activation=getattr(self.config, "ae_activation", "relu"),
                dropout=getattr(self.config, "ae_dropout", 0.1),
                learning_rate=getattr(self.config, "ae_learning_rate", 1e-3),
                batch_size=getattr(self.config, "ae_batch_size", 64),
                epochs=getattr(self.config, "ae_epochs", 50),
                patience=getattr(self.config, "ae_patience", 10),
                max_features=getattr(self.config, "ae_max_features", 100),
                min_features=getattr(self.config, "ae_min_features", 4),
                random_state=getattr(self.config, "random_state", 42),
            )
            registry["autoencoder"] = (
                True,
                AutoencoderTransformer(self.config, ae_cfg),
            )

        return {name: tf for name, (enabled, tf) in registry.items() if enabled}

    @staticmethod
    def _parse_list_str(val: str | int | float) -> list[int]:
        try:
            s = str(val).strip()
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        except Exception:
            return [64, 32]

    @staticmethod
    def _transform_with_optional_y(
        transformer: Any, X: pd.DataFrame, y: pd.Series | None
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

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None):
        """Fit comprehensive feature engineering pipeline."""
        X = X.copy()
        if y is not None:
            y = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()

        self._log(
            f"🚀 Fitting FeatureEngineer on {X.shape[0]} rows, {X.shape[1]} columns"
        )
        self.feature_stats_ = {}

        self.transformers_ = self._build_transformers()
        current_X = X.copy()

        for name, transformer in self.transformers_.items():
            self._log(f"🔧 Fitting {name} transformer...")
            try:
                transformer.fit(current_X, y)
                transformed = self._transform_with_optional_y(transformer, current_X, y)
                if transformed is not None and not transformed.empty:
                    # Autoencoder replaces the entire feature matrix
                    if isinstance(transformer, (AutoencoderTransformer,)):
                        current_X = transformed
                        self.feature_stats_[name] = transformed.shape[1]
                    else:
                        current_X = pd.concat([current_X, transformed], axis=1)
                        self.feature_stats_[name] = transformed.shape[1]
                else:
                    self.feature_stats_[name] = 0
            except Exception as exc:
                self.logger.warning(
                    f"Transformer '{name}' failed: {exc}. Skipping this step."
                )
                self.feature_stats_[name] = 0

        self._log("🔍 Applying correlation filtering...")
        self.correlation_filter_ = CorrelationFilter(
            threshold=self.config.corr_threshold,
            method=getattr(self.config, "corr_filter_method", "variance"),
            dependence_metric=getattr(self.config, "corr_dependence_metric", "pearson"),
            random_state=getattr(self.config, "random_state", 42),
            mi_subsample=min(getattr(self.config, "max_rows_score", 2000), 2000),
            mi_min_overlap=getattr(self.config, "mi_min_overlap", 50),
            mi_bins=getattr(self.config, "mi_bins", 16),
        )
        self.correlation_filter_.fit(current_X, y=y)
        current_X = self.correlation_filter_.transform(current_X)

        # Feature selection
        if y is not None and not current_X.empty:
            self._log("🎯 Performing feature selection...")
            self.selector_ = FeatureSelector(self.config)
            self.selector_.fit(current_X, y)
            selected_features = self.selector_.get_selected_features()

            if selected_features:
                current_X = current_X[selected_features]

                # Fit final scaler
                if self._should_use_quantile_transform():
                    print("📊 Fitting final quantile transformer...")
                    self.final_scaler_ = QuantileTransformer(
                        n_quantiles=min(1000, max(10, len(current_X) // 2)),
                        output_distribution="normal",
                        random_state=self.config.random_state,
                    )
                    X_filled = current_X.fillna(current_X.median())
                    self.final_scaler_.fit(X_filled)

                self._log(
                    f"✅ Selected {len(selected_features)} features after filtering"
                )
            else:
                self._log(
                    "⚠️  No features selected - pipeline may need tuning",
                    level="warning",
                )

        # Print feature creation summary
        self.output_features_ = current_X.columns.tolist()
        # Keep training-time fill values so transform does not depend on tiny batch stats.
        self.output_fill_values_ = current_X.median(numeric_only=True)
        self._print_feature_summary()

        self.fitted_ = True
        return self

    def get_summary(self) -> dict[str, Any]:
        """Return a dictionary summarizing the fitted pipeline."""
        return {
            "backend": self._resolve_backend(),
            "feature_stats": self.feature_stats_.copy(),
            "output_features": self.output_features_.copy(),
            "selected_features": self.selector_.get_selected_features()
            if self.selector_
            else [],
            "correlation_dropped": len(self.correlation_filter_.features_to_drop_)
            if self.correlation_filter_
            else 0,
            "final_scaler": self.final_scaler_ is not None,
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted pipeline."""
        check_is_fitted(self, "fitted_")

        current_X = X.copy()

        # Apply all transformers
        for name, transformer in self.transformers_.items():
            try:
                transformed = transformer.transform(current_X)
                if transformed is not None and not transformed.empty:
                    # Autoencoder replaces the entire feature matrix
                    if isinstance(transformer, (AutoencoderTransformer,)):
                        current_X = transformed
                    else:
                        current_X = pd.concat([current_X, transformed], axis=1)
            except Exception as e:
                self.logger.warning(f"Transform failed for {name}: {e}")

        # Autoencoder replaces the entire matrix, skip correlation filter + selection
        has_autoencoder = any(
            isinstance(t, (AutoencoderTransformer,))
            for t in self.transformers_.values()
        )

        if not has_autoencoder:
            # Apply correlation filtering
            if self.correlation_filter_ is not None:
                current_X = self.correlation_filter_.transform(current_X)

            # Apply feature selection
            if self.selector_ is not None:
                selected_features = self.selector_.get_selected_features()
                if selected_features:
                    missing_selected = [
                        c for c in selected_features if c not in current_X.columns
                    ]
                    if missing_selected:
                        current_X = current_X.copy()
                        for col in missing_selected:
                            current_X[col] = np.nan
                    current_X = current_X[selected_features]

        # Enforce fitted output schema/order for robust inference on small/unseen batches.
        if self.output_features_:
            missing_output = [
                c for c in self.output_features_ if c not in current_X.columns
            ]
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
            return pd.DataFrame(X_scaled, columns=current_X.columns, index=X.index)

        return current_X if not current_X.empty else pd.DataFrame(index=X.index)

    def _print_feature_summary(self):
        """Log summary of feature creation."""
        total_features = sum(self.feature_stats_.values())
        self._log("\n📈 Feature Creation Summary:")
        self._log("-" * 40)
        for name, count in self.feature_stats_.items():
            self._log(f"{name.title():<15}: {count:>4} features")
        self._log("-" * 40)
        self._log(f"{'Total':<15}: {total_features:>4} features")

        if self.correlation_filter_:
            dropped = len(self.correlation_filter_.features_to_drop_)
            self._log(f"{'Corr. Dropped':<15}: {dropped:>4} features")

        if self.selector_:
            selected = len(self.selector_.get_selected_features())
            self._log(f"{'Final Selected':<15}: {selected:>4} features")
            if hasattr(self.selector_, "selection_method_"):
                self._log(
                    f"{'Selection Method':<15}: {self.selector_.selection_method_}"
                )

    def get_feature_importance(self) -> pd.Series | None:
        """Get feature importance from feature selection analysis."""
        if self.selector_ and hasattr(self.selector_, "get_feature_scores"):
            return self.selector_.get_feature_scores()
        elif self.selector_ and hasattr(self.selector_, "mi_scores_"):
            return self.selector_.mi_scores_
        return None

    def plot_feature_importance(
        self, top_k: int = 20, figsize: tuple[int, int] = (12, 8)
    ):
        """Plot feature importance with enhanced visualization."""
        import matplotlib.pyplot as plt

        importance_scores = self.get_feature_importance()
        if importance_scores is None:
            self.logger.warning("❌ No feature importance scores available.")
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

    def get_transformation_report(self) -> dict[str, Any]:
        """Get detailed report of transformations applied."""
        report = {
            "feature_stats": self.feature_stats_.copy(),
            "config": self.config.__dict__.copy(),
            "backend": self._resolve_backend(),
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
        if (
            self.selector_
            and hasattr(self.selector_, "rfecv_selector_")
            and self.selector_.rfecv_selector_
        ):
            report["rfecv_results"] = (
                self.selector_.rfecv_selector_.get_performance_summary()
            )

        if self.selector_ and hasattr(self.selector_, "get_feature_scores"):
            scores = self.selector_.get_feature_scores()
            if scores is not None:
                report["top_features"] = scores.head(10).to_dict()
        elif self.selector_ and hasattr(self.selector_, "mi_scores_"):
            report["top_features"] = self.selector_.mi_scores_.head(10).to_dict()

        return report

    def plot_rfecv_results(self, **kwargs):
        """Plot RFECV results if RFECV was used."""
        if (
            self.selector_
            and hasattr(self.selector_, "rfecv_selector_")
            and self.selector_.rfecv_selector_ is not None
        ):
            self.selector_.rfecv_selector_.plot_cv_scores(**kwargs)
        else:
            self.logger.warning("RFECV was not used or is not available for plotting.")
