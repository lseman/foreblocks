
from typing import Any, Dict

import pandas as pd

from .foreminer_aux import *


class FeatureEngineeringAnalyzer(AnalysisStrategy):
    """SOTA intelligent feature engineering with advanced ML techniques"""

    @property
    def name(self) -> str:
        return "feature_engineering"

    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        pass

        import numpy as np
        import pandas as pd
        import scipy.stats as stats
        from scipy.stats import jarque_bera, shapiro
        from sklearn.ensemble import (ExtraTreesRegressor,
                                      GradientBoostingRegressor,
                                      RandomForestRegressor)
        from statsmodels.stats.outliers_influence import \
            variance_inflation_factor
        from statsmodels.tsa.seasonal import STL

        try:
            import shap

            SHAP_AVAILABLE = True
        except ImportError:
            SHAP_AVAILABLE = False
        try:
            pass

            CATEGORY_ENCODERS_AVAILABLE = True
        except ImportError:
            CATEGORY_ENCODERS_AVAILABLE = False

        # Initialize both structures properly
        suggestions = {}
        detailed_results = {
            "transformations": {},
            "interactions": [],
            "encodings": {},
            "feature_ranking": {},
            "dimensionality_reduction": [],
            "time_series_features": {},
            "advanced_features": [],
        }

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Remove time and target columns from feature lists
        time_col = getattr(config, "time_col", None)
        target_col = getattr(config, "target", None)
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if time_col in categorical_cols:
            categorical_cols.remove(time_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        # === Advanced Distribution Analysis & Transformations ===
        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) < 10:
                continue

            transforms = []
            col_info = {
                "skewness": stats.skew(series),
                "kurtosis": stats.kurtosis(series),
                "cv": series.std() / (abs(series.mean()) + 1e-8),
                "outliers_pct": (
                    (
                        series
                        < (
                            series.quantile(0.25)
                            - 1.5 * (series.quantile(0.75) - series.quantile(0.25))
                        )
                    )
                    | (
                        series
                        > (
                            series.quantile(0.75)
                            + 1.5 * (series.quantile(0.75) - series.quantile(0.25))
                        )
                    )
                ).mean()
                * 100,
            }

            # Normality tests
            try:
                jb_stat, jb_p = jarque_bera(series.sample(min(5000, len(series))))
                sw_stat, sw_p = shapiro(series.sample(min(5000, len(series))))
                col_info["normality_jb_p"] = jb_p
                col_info["normality_sw_p"] = sw_p
            except:
                pass

            # Smart transformation suggestions based on distribution properties
            if abs(col_info["skewness"]) > 2:
                if series.min() > 0:
                    transforms.extend(
                        [f"np.log1p({col})", f"np.sqrt({col})", f"1/({col}+1e-8)"]
                    )
                else:
                    transforms.extend(
                        [
                            f"np.sign({col}) * np.log1p(np.abs({col}))",
                            f"scipy.stats.yeojohnson({col})[0]",
                        ]
                    )

            if col_info["kurtosis"] > 3:  # Heavy tails
                transforms.extend(
                    [
                        f"np.tanh({col}/series.std())",
                        f"scipy.stats.rankdata({col})/len({col})",
                    ]
                )

            if col_info["outliers_pct"] > 10:
                transforms.extend(
                    [
                        f"scipy.stats.mstats.winsorize({col}, limits=(0.05, 0.05))",
                        f"RobustScaler().fit_transform({col}.values.reshape(-1,1)).flatten()",
                    ]
                )

            if col_info["cv"] > 3:  # High variability
                transforms.extend(
                    [
                        f"pd.qcut({col}, q=10, labels=False, duplicates='drop')",
                        f"PowerTransformer(method='yeo-johnson').fit_transform({col}.values.reshape(-1,1)).flatten()",
                    ]
                )

            # Advanced statistical transformations
            if series.min() >= 0:
                transforms.extend(
                    [f"np.power({col}, 1/3)", f"np.exp(-{col}/series.mean())"]
                )

            transforms.extend(
                [
                    f"({col} - series.mean()) / series.std()",  # Z-score
                    f"({col} - series.median()) / series.mad()",
                ]
            )  # Robust standardization

            # Store in detailed results
            detailed_results["transformations"][col] = {
                "stats": col_info,
                "recommended": transforms[:8],  # Top 8 transforms
            }

            # Backward compatibility - flat list for existing interface
            suggestions[col] = transforms[:4]

        # === Ensemble-based Feature Importance & Selection ===
        if target_col and target_col in data.columns:
            X = data[numeric_cols].fillna(data[numeric_cols].median())
            y = data[target_col].fillna(data[target_col].median())

            if len(X) > 50 and X.shape[1] > 1:
                # Multiple model ensemble for robust importance
                models = [
                    RandomForestRegressor(n_estimators=100, random_state=42),
                    GradientBoostingRegressor(n_estimators=100, random_state=42),
                    ExtraTreesRegressor(n_estimators=100, random_state=42),
                ]

                importance_scores = {}
                for model in models:
                    try:
                        model.fit(X, y)
                        for i, col in enumerate(numeric_cols):
                            importance_scores.setdefault(col, []).append(
                                model.feature_importances_[i]
                            )
                    except:
                        continue

                # Aggregate importance scores
                avg_importance = {
                    col: np.mean(scores) for col, scores in importance_scores.items()
                }
                detailed_results["feature_ranking"] = dict(
                    sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
                )

                # SHAP analysis if available
                if SHAP_AVAILABLE and len(X) < 1000:  # Limit for performance
                    try:
                        model = GradientBoostingRegressor(
                            n_estimators=50, random_state=42
                        ).fit(X, y)
                        explainer = shap.Explainer(model, X.sample(min(100, len(X))))
                        shap_values = explainer(X.sample(min(200, len(X))))
                        shap_importance = np.abs(shap_values.values).mean(axis=0)
                        detailed_results["feature_ranking"]["shap_importance"] = dict(
                            zip(numeric_cols, shap_importance)
                        )
                    except:
                        pass

        # === Intelligent Feature Interactions ===
        if len(numeric_cols) >= 2:
            top_features = (
                list(detailed_results.get("feature_ranking", {}).keys())[:10]
                or numeric_cols[:10]
            )

            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i + 1 :]:
                    if feat1 in data.columns and feat2 in data.columns:
                        # Calculate interaction strength
                        try:
                            corr = data[[feat1, feat2]].corr().iloc[0, 1]
                            if abs(corr) < 0.9:  # Avoid highly correlated pairs
                                interactions = [
                                    f"{feat1} * {feat2}",
                                    f"{feat1} / ({feat2} + 1e-8)",
                                    f"np.sqrt({feat1}**2 + {feat2}**2)",
                                    f"np.maximum({feat1}, {feat2})",
                                    f"np.minimum({feat1}, {feat2})",
                                    f"({feat1} + {feat2}) / 2",
                                    f"abs({feat1} - {feat2})",
                                    f"np.where({feat1} > {feat2}, 1, 0)",
                                ]
                                detailed_results["interactions"].extend(interactions)
                        except:
                            continue

            # Backward compatibility - add interactions to main suggestions
            suggestions["interactions"] = detailed_results["interactions"][:20]

        # === Advanced Categorical Encoding ===
        for col in categorical_cols:
            nunique = data[col].nunique()
            null_pct = data[col].isnull().mean() * 100

            encoding_strategies = []

            if 2 <= nunique <= 5:
                encoding_strategies.append("pd.get_dummies(drop_first=True)")
            elif nunique <= 20:
                encoding_strategies.extend(["LabelEncoder", "OrdinalEncoder"])
                if target_col and CATEGORY_ENCODERS_AVAILABLE:
                    encoding_strategies.extend(["TargetEncoder", "WOEEncoder"])
            else:
                encoding_strategies.extend(["HashingEncoder", "FrequencyEncoder"])
                if target_col and CATEGORY_ENCODERS_AVAILABLE:
                    encoding_strategies.append("CatBoostEncoder")

            # Add rare category handling for high cardinality
            if nunique > 10:
                encoding_strategies.append(f"RareCategoryEncoder(threshold=0.01)")

            # Add encoding suggestions to main dict for backward compatibility
            suggestions[col] = encoding_strategies[:3]  # Top 3 strategies
            detailed_results["encodings"][col] = {
                "cardinality": nunique,
                "null_percentage": null_pct,
                "strategies": encoding_strategies,
            }

        # === Multicollinearity Detection (VIF) ===
        if len(numeric_cols) > 1 and len(numeric_cols) < 50:
            try:
                X_vif = data[numeric_cols].fillna(data[numeric_cols].median())
                vif_data = pd.DataFrame()
                vif_data["Feature"] = numeric_cols
                vif_data["VIF"] = [
                    variance_inflation_factor(X_vif.values, i)
                    for i in range(X_vif.shape[1])
                ]

                high_vif = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
                if high_vif:
                    detailed_results["dimensionality_reduction"].append(
                        f"High VIF features (>10): {high_vif}"
                    )
                    detailed_results["dimensionality_reduction"].append(
                        "Consider PCA or feature selection"
                    )
            except:
                pass

        # === Time Series Feature Engineering ===
        if time_col and time_col in data.columns:
            time_series_features = []

            # Temporal features
            time_series_features.extend(
                [
                    f"df['{time_col}'].dt.hour",
                    f"df['{time_col}'].dt.dayofweek",
                    f"df['{time_col}'].dt.month",
                    f"df['{time_col}'].dt.quarter",
                    f"df['{time_col}'].dt.is_weekend",
                    f"df['{time_col}'].dt.dayofyear",
                ]
            )

            # Cyclical encoding
            time_series_features.extend(
                [
                    f"np.sin(2 * np.pi * df['{time_col}'].dt.hour / 24)",
                    f"np.cos(2 * np.pi * df['{time_col}'].dt.hour / 24)",
                    f"np.sin(2 * np.pi * df['{time_col}'].dt.dayofyear / 365)",
                    f"np.cos(2 * np.pi * df['{time_col}'].dt.dayofyear / 365)",
                ]
            )

            # Lag and window features for numeric columns
            for col in numeric_cols[:5]:  # Limit to top 5 for performance
                try:
                    series = data.set_index(time_col)[col].dropna()
                    if len(series) > 24:
                        # Decomposition analysis
                        stl = STL(series, seasonal=min(13, len(series) // 3)).fit()
                        trend_strength = np.var(stl.trend.dropna()) / (
                            np.var(series) + 1e-8
                        )
                        seasonal_strength = np.var(stl.seasonal) / (
                            np.var(series) + 1e-8
                        )

                        lag_features = [
                            f"df['{col}'].shift(1)",
                            f"df['{col}'].shift(7)",
                            f"df['{col}'].rolling(window=3).mean()",
                            f"df['{col}'].rolling(window=7).mean()",
                            f"df['{col}'].rolling(window=3).std()",
                            f"df['{col}'].expanding().mean()",
                            f"df['{col}'].pct_change()",
                        ]

                        if trend_strength > 0.3:
                            lag_features.append(f"STL_trend_component")
                        if seasonal_strength > 0.3:
                            lag_features.append(f"STL_seasonal_component")

                        detailed_results["time_series_features"][col] = lag_features
                except:
                    continue

            detailed_results["time_series_features"]["temporal"] = time_series_features

        # === Advanced Feature Creation Patterns ===
        advanced_patterns = [
            "Polynomial features (degree 2-3) for top predictors",
            "Binning continuous variables into quantiles",
            "Distance/similarity features between data points",
            "Aggregated features by categorical groupings",
            "Ratio features between related numeric columns",
            "Deviation from group means/medians",
            "Percentile rank transformations",
            "Fourier transform features for periodic patterns",
        ]
        detailed_results["advanced_features"] = advanced_patterns

        # === Feature Selection Recommendations ===
        selection_methods = [
            "Recursive Feature Elimination (RFE)",
            "SelectKBest with mutual information",
            "LASSO regularization for sparse selection",
            "Boruta algorithm for all-relevant features",
            "Permutation importance ranking",
        ]
        detailed_results["feature_selection_methods"] = selection_methods

        # Store detailed results for advanced users
        suggestions["_detailed"] = detailed_results

        return suggestions

