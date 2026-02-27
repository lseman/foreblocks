from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


@dataclass
class RFECVConfig:
    """Configuration for RFECV parameters."""

    step: Union[int, float] = 0.1  # Features to remove each iteration
    cv: int = 5  # Cross-validation folds
    scoring: str = "auto"  # Scoring metric
    min_features_to_select: int = 1  # Minimum features to keep
    max_features_to_select: Optional[int] = None  # Maximum features (None = no limit)
    n_jobs: int = -1  # Parallel jobs
    verbose: int = 1  # Verbosity level
    random_state: int = 42

    # Advanced options
    patience: int = 5  # Early stopping patience
    improvement_threshold: float = 0.001  # Minimum improvement to continue
    stability_selection: bool = True  # Use stability across CV folds
    feature_importance_method: str = (
        "auto"  # 'auto', 'coef', 'feature_importances', 'permutation'
    )

    # Ensemble options
    use_ensemble: bool = True  # Use multiple estimators
    estimator_weights: Optional[Dict[str, float]] = None  # Weights for ensemble voting


class AdvancedRFECV(BaseEstimator, TransformerMixin):
    """
    Advanced Recursive Feature Elimination with Cross-Validation.

    This implementation provides several enhancements over sklearn's RFECV:
    - Multiple estimator support with ensemble voting
    - Stability selection across CV folds
    - Advanced early stopping criteria
    - Comprehensive scoring options
    - Feature importance tracking
    - Detailed performance analytics
    """

    def __init__(self, estimator=None, config: Optional[RFECVConfig] = None, **kwargs):
        self.estimator = estimator
        self.config = config or RFECVConfig(**kwargs)

        # Results storage
        self.support_ = None
        self.ranking_ = None
        self.n_features_ = None
        self.cv_scores_ = None
        self.feature_importances_ = None
        self.grid_scores_ = None

        # Internal state
        self._is_fitted = False
        self._feature_names = None
        self._task_type = None
        self._best_estimator = None
        self._cv_results = []
        self._feature_stability = {}

    def _determine_task_type(self, y: pd.Series) -> str:
        """Determine if this is classification or regression."""
        if y.dtype == "object" or y.dtype.name == "category":
            return "classification"

        unique_vals = y.nunique()
        n_samples = len(y)

        if unique_vals < 20 or unique_vals / n_samples < 0.05:
            return "classification"
        else:
            return "regression"

    def _get_default_estimators(self, task_type: str) -> Dict[str, Any]:
        """Get default estimators based on task type."""
        if task_type == "classification":
            return {
                "rf": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.config.random_state,
                    n_jobs=1,  # Will be parallelized at CV level
                ),
                "lr": LogisticRegression(
                    random_state=self.config.random_state,
                    max_iter=1000,
                    solver="liblinear",
                ),
            }
        else:  # regression
            return {
                "rf": RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.config.random_state,
                    n_jobs=1,
                ),
                "ridge": Ridge(random_state=self.config.random_state, alpha=1.0),
            }

    def _get_scorer(self, task_type: str, scoring: str) -> str:
        """Get appropriate scorer for the task."""
        if scoring == "auto":
            if task_type == "classification":
                return "accuracy"
            else:
                return "neg_mean_squared_error"
        return scoring

    def _get_cv_splitter(self, y: np.ndarray, task_type: str):
        """Get appropriate CV splitter."""
        if task_type == "classification":
            return StratifiedKFold(
                n_splits=self.config.cv,
                shuffle=True,
                random_state=self.config.random_state,
            )
        else:
            return KFold(
                n_splits=self.config.cv,
                shuffle=True,
                random_state=self.config.random_state,
            )

    def _extract_feature_importance(
        self, estimator, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Extract feature importance from fitted estimator."""
        method = self.config.feature_importance_method

        if method == "auto":
            if hasattr(estimator, "feature_importances_"):
                return estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                coef = estimator.coef_
                if coef.ndim > 1:
                    return np.abs(coef).mean(axis=0)
                return np.abs(coef)
            else:
                method = "permutation"

        if method == "feature_importances":
            return estimator.feature_importances_
        elif method == "coef":
            coef = estimator.coef_
            if coef.ndim > 1:
                return np.abs(coef).mean(axis=0)
            return np.abs(coef)
        elif method == "permutation":
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                estimator,
                X,
                y,
                n_repeats=5,
                random_state=self.config.random_state,
                n_jobs=1,
            )
            return result.importances_mean
        else:
            raise ValueError(f"Unknown importance method: {method}")

    def _calculate_step_size(self, n_features: int) -> int:
        """Calculate number of features to remove in this step."""
        if isinstance(self.config.step, int):
            return min(
                self.config.step, n_features - self.config.min_features_to_select
            )
        else:  # float
            step_size = max(1, int(n_features * self.config.step))
            return min(step_size, n_features - self.config.min_features_to_select)

    def _ensemble_feature_importance(
        self,
        estimators: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        feature_mask: np.ndarray,
    ) -> np.ndarray:
        """Calculate ensemble feature importance across multiple estimators."""
        importances = []
        weights = self.config.estimator_weights or {}

        X_subset = X[:, feature_mask]

        for name, estimator in estimators.items():
            # Clone and fit estimator
            est_clone = clone(estimator)
            est_clone.fit(X_subset, y)

            # Get importance
            importance = self._extract_feature_importance(est_clone, X_subset, y)

            # Apply weight
            weight = weights.get(name, 1.0)
            importances.append(importance * weight)

        # Average importances
        ensemble_importance = np.mean(importances, axis=0)

        # Map back to full feature space
        full_importance = np.zeros(len(feature_mask))
        full_importance[feature_mask] = ensemble_importance

        return full_importance

    def _stability_selection_step(
        self,
        estimators: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        feature_mask: np.ndarray,
        cv_splitter,
    ) -> Tuple[np.ndarray, float]:
        """Perform stability selection across CV folds."""
        n_features = np.sum(feature_mask)
        stability_scores = np.zeros(len(feature_mask))
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Get ensemble importance for this fold
            fold_importance = self._ensemble_feature_importance(
                estimators, X_train, y_train, feature_mask
            )

            # Update stability scores
            stability_scores += fold_importance

            # Calculate CV score for this fold
            if self.config.use_ensemble:
                fold_scores = []
                for name, estimator in estimators.items():
                    est_clone = clone(estimator)
                    X_train_subset = X_train[:, feature_mask]
                    X_val_subset = X_val[:, feature_mask]
                    est_clone.fit(X_train_subset, y_train)

                    if self._task_type == "classification":
                        pred = est_clone.predict(X_val_subset)
                        score = accuracy_score(y_val, pred)
                    else:
                        pred = est_clone.predict(X_val_subset)
                        score = -mean_squared_error(
                            y_val, pred
                        )  # Negative MSE for maximization

                    fold_scores.append(score)

                cv_scores.append(np.mean(fold_scores))
            else:
                # Use single estimator
                estimator = list(estimators.values())[0]
                est_clone = clone(estimator)
                X_train_subset = X_train[:, feature_mask]
                X_val_subset = X_val[:, feature_mask]
                est_clone.fit(X_train_subset, y_train)

                if self._task_type == "classification":
                    pred = est_clone.predict(X_val_subset)
                    score = accuracy_score(y_val, pred)
                else:
                    pred = est_clone.predict(X_val_subset)
                    score = -mean_squared_error(y_val, pred)

                cv_scores.append(score)

        # Average stability scores across folds
        stability_scores /= self.config.cv

        return stability_scores, np.mean(cv_scores)

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "AdvancedRFECV":
        """
        Fit RFECV with advanced features.

        Parameters:
        -----------
        X : DataFrame or array-like
            Training features.
        y : Series or array-like
            Target values.

        Returns:
        --------
        self : AdvancedRFECV
            Fitted estimator.
        """
        # Convert inputs
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Handle categorical targets
        if y_array.dtype == "object":
            le = LabelEncoder()
            y_array = le.fit_transform(y_array)

        # Determine task type
        y_series = pd.Series(y_array)
        self._task_type = self._determine_task_type(y_series)

        # Set up estimators
        if self.estimator is None:
            estimators = self._get_default_estimators(self._task_type)
        else:
            estimators = {"custom": self.estimator}

        # Set up CV
        cv_splitter = self._get_cv_splitter(y_array, self._task_type)
        scorer = self._get_scorer(self._task_type, self.config.scoring)

        if self.config.verbose > 0:
            print("🔄 Starting Advanced RFECV...")
            print(f"Task type: {self._task_type}")
            print(f"Features: {X_array.shape[1]}, Samples: {X_array.shape[0]}")
            print(f"Estimators: {list(estimators.keys())}")
            print(f"CV folds: {self.config.cv}")

        # Initialize tracking
        n_features = X_array.shape[1]
        feature_mask = np.ones(n_features, dtype=bool)
        self.grid_scores_ = []
        self._cv_results = []

        best_score = -np.inf
        best_n_features = n_features
        best_mask = feature_mask.copy()
        patience_counter = 0

        iteration = 0
        max_features = self.config.max_features_to_select or n_features

        # Main RFECV loop
        while np.sum(feature_mask) > self.config.min_features_to_select:
            current_n_features = np.sum(feature_mask)

            if current_n_features > max_features:
                # Skip scoring if we have too many features
                step_size = min(
                    self._calculate_step_size(current_n_features),
                    current_n_features - max_features,
                )

                # Just remove least important features without full CV
                importance = self._ensemble_feature_importance(
                    estimators, X_array, y_array, feature_mask
                )

                # Get indices of features to remove
                feature_indices = np.where(feature_mask)[0]
                feature_importances = importance[feature_mask]
                worst_indices = feature_indices[
                    np.argsort(feature_importances)[:step_size]
                ]
                feature_mask[worst_indices] = False

                if self.config.verbose > 0:
                    print(
                        f"Iteration {iteration + 1}: {current_n_features} -> {np.sum(feature_mask)} features (fast mode)"
                    )

                iteration += 1
                continue

            if self.config.verbose > 0:
                print(
                    f"\nIteration {iteration + 1}: Evaluating {current_n_features} features..."
                )

            # Perform stability selection or regular CV
            if self.config.stability_selection:
                importance, cv_score = self._stability_selection_step(
                    estimators, X_array, y_array, feature_mask, cv_splitter
                )
            else:
                # Regular cross-validation
                X_subset = X_array[:, feature_mask]
                scores = []

                for name, estimator in estimators.items():
                    est_scores = cross_val_score(
                        estimator,
                        X_subset,
                        y_array,
                        cv=cv_splitter,
                        scoring=scorer,
                        n_jobs=self.config.n_jobs,
                    )
                    scores.extend(est_scores)

                cv_score = np.mean(scores)

                # Get feature importance
                importance = self._ensemble_feature_importance(
                    estimators, X_array, y_array, feature_mask
                )

            # Store results
            self.grid_scores_.append(cv_score)
            self._cv_results.append(
                {
                    "iteration": iteration,
                    "n_features": current_n_features,
                    "cv_score": cv_score,
                    "feature_mask": feature_mask.copy(),
                    "importance": importance.copy(),
                }
            )

            if self.config.verbose > 0:
                print(f"CV Score: {cv_score:.4f}")

            # Check for improvement
            if cv_score > best_score + self.config.improvement_threshold:
                best_score = cv_score
                best_n_features = current_n_features
                best_mask = feature_mask.copy()
                patience_counter = 0

                if self.config.verbose > 0:
                    print(
                        f"✅ New best score: {best_score:.4f} with {best_n_features} features"
                    )
            else:
                patience_counter += 1

                if self.config.verbose > 0:
                    print(
                        f"⏳ No improvement ({patience_counter}/{self.config.patience})"
                    )

            # Early stopping
            if patience_counter >= self.config.patience:
                if self.config.verbose > 0:
                    print(
                        f"🛑 Early stopping: No improvement for {self.config.patience} iterations"
                    )
                break

            # Calculate step size and remove features
            step_size = self._calculate_step_size(current_n_features)

            if step_size <= 0:
                break

            # Remove least important features
            feature_indices = np.where(feature_mask)[0]
            feature_importances = importance[feature_mask]
            worst_indices = feature_indices[np.argsort(feature_importances)[:step_size]]
            feature_mask[worst_indices] = False

            iteration += 1

        # Set final results
        self.support_ = best_mask
        self.n_features_ = best_n_features

        # Create ranking
        self.ranking_ = np.ones(n_features, dtype=int)
        eliminated_order = []

        for result in reversed(self._cv_results):
            current_mask = result["feature_mask"]
            if not np.array_equal(current_mask, best_mask):
                # Features eliminated in this step
                eliminated = np.where((best_mask == False) & (current_mask == True))[0]
                eliminated_order.extend(eliminated)
                best_mask = current_mask

        # Assign rankings
        for rank, feature_idx in enumerate(eliminated_order):
            self.ranking_[feature_idx] = len(eliminated_order) - rank + 1

        # Store final feature importances
        self.feature_importances_ = self._ensemble_feature_importance(
            estimators, X_array, y_array, self.support_
        )

        # Fit best estimator
        if self.config.use_ensemble:
            # Use the first estimator as the best estimator
            self._best_estimator = clone(list(estimators.values())[0])
        else:
            self._best_estimator = clone(self.estimator or list(estimators.values())[0])

        X_final = X_array[:, self.support_]
        self._best_estimator.fit(X_final, y_array)

        self._is_fitted = True

        if self.config.verbose > 0:
            print("\n🎯 RFECV completed!")
            print(f"Best score: {best_score:.4f}")
            print(f"Selected features: {self.n_features_} out of {n_features}")
            if self._feature_names:
                selected_names = np.array(self._feature_names)[self.support_].tolist()
                print(
                    f"Feature names: {selected_names[:10]}{'...' if len(selected_names) > 10 else ''}"
                )

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Transform data by selecting only the chosen features."""
        if not self._is_fitted:
            raise ValueError("RFECV must be fitted before transform.")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.support_]
        else:
            return X[:, self.support_]

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Fit RFECV and transform data in one step."""
        return self.fit(X, y).transform(X)

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        if not self._is_fitted:
            raise ValueError("RFECV must be fitted first.")

        if self._feature_names:
            return np.array(self._feature_names)[self.support_].tolist()
        else:
            return [f"feature_{i}" for i in np.where(self.support_)[0]]

    def get_feature_ranking(self) -> pd.DataFrame:
        """Get detailed feature ranking with scores."""
        if not self._is_fitted:
            raise ValueError("RFECV must be fitted first.")

        df = pd.DataFrame(
            {
                "feature": self._feature_names
                or [f"feature_{i}" for i in range(len(self.ranking_))],
                "selected": self.support_,
                "ranking": self.ranking_,
                "importance": self.feature_importances_,
            }
        )

        return df.sort_values("ranking")

    def plot_cv_scores(self, figsize: Tuple[int, int] = (12, 6)):
        """Plot cross-validation scores vs number of features."""
        if not self._is_fitted:
            raise ValueError("RFECV must be fitted first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting.")
            return

        # Extract data for plotting
        n_features_list = [result["n_features"] for result in self._cv_results]
        cv_scores = [result["cv_score"] for result in self._cv_results]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # CV scores vs number of features
        ax1.plot(n_features_list, cv_scores, "bo-", linewidth=2, markersize=6)
        ax1.axvline(
            x=self.n_features_,
            color="red",
            linestyle="--",
            label=f"Selected: {self.n_features_} features",
        )
        ax1.set_xlabel("Number of Features")
        ax1.set_ylabel("Cross-Validation Score")
        ax1.set_title("RFECV: CV Score vs Number of Features")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Feature importance for selected features
        if self._feature_names:
            selected_features = np.array(self._feature_names)[self.support_]
            selected_importance = self.feature_importances_[self.support_]

            # Sort by importance
            sorted_idx = np.argsort(selected_importance)[::-1]
            top_features = selected_features[sorted_idx][:15]  # Top 15
            top_importance = selected_importance[sorted_idx][:15]

            bars = ax2.barh(
                range(len(top_features)), top_importance, color="skyblue", alpha=0.7
            )
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features)
            ax2.set_xlabel("Feature Importance")
            ax2.set_title("Top Selected Features")
            ax2.grid(axis="x", alpha=0.3)

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, top_importance)):
                ax2.text(
                    bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    ha="left",
                    va="center",
                    fontsize=9,
                )

            ax2.invert_yaxis()

        plt.tight_layout()
        plt.show()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self._is_fitted:
            raise ValueError("RFECV must be fitted first.")

        cv_scores = [result["cv_score"] for result in self._cv_results]
        n_features_list = [result["n_features"] for result in self._cv_results]

        return {
            "best_score": max(cv_scores),
            "best_n_features": self.n_features_,
            "total_iterations": len(self._cv_results),
            "score_improvement": max(cv_scores) - min(cv_scores) if cv_scores else 0,
            "feature_reduction_ratio": (n_features_list[0] - self.n_features_)
            / n_features_list[0]
            if n_features_list
            else 0,
            "selected_features": self.get_selected_features(),
            "cv_scores_history": cv_scores,
            "n_features_history": n_features_list,
        }
