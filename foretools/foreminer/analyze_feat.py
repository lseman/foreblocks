from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .foreminer_aux import *


class FeatureEngineeringAnalyzer(AnalysisStrategy):
    """Intelligent feature engineering with robust stats, leak-aware suggestions, and time features."""

    @property
    def name(self) -> str:
        return "feature_engineering"

    # -------------------- Public API --------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        cfg = self._get_cfg(config)

        # Partition columns, excluding time/target from candidates
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
        for c in (cfg.time_col, cfg.target_col):
            if c in numeric_cols: numeric_cols.remove(c)
            if c in categorical_cols: categorical_cols.remove(c)

        suggestions: Dict[str, Any] = {}
        detailed: Dict[str, Any] = {
            "transformations": {},
            "interactions": [],
            "encodings": {},
            "feature_ranking": {},
            "dimensionality_reduction": [],
            "time_series_features": {},
            "advanced_features": [],
            "feature_selection_methods": [],
        }

        # 1) Numeric transforms (stats-driven)
        t_suggestions, t_details = self._suggest_numeric_transforms(
            data, numeric_cols, cfg
        )
        suggestions.update(t_suggestions)
        detailed["transformations"] = t_details

        # 2) Supervised ranking (+ optional SHAP)
        if cfg.target_col and cfg.target_col in data.columns and len(numeric_cols) > 1:
            franking = self._rank_features(data, numeric_cols, cfg)
            if franking:
                detailed["feature_ranking"] = franking

        # 3) Interactions (avoid collinear, optional MI gate)
        interactions = self._suggest_interactions(data, numeric_cols, detailed, cfg)
        detailed["interactions"] = interactions
        suggestions["interactions"] = interactions[:20]  # short list (back-compat)

        # 4) Categorical encodings (leak-aware: suggest only)
        enc_suggestions, enc_details = self._suggest_encodings(
            data, categorical_cols, cfg
        )
        suggestions.update(enc_suggestions)
        detailed["encodings"] = enc_details

        # 5) Multicollinearity (VIF)
        dimred_notes = self._vif_screen(data, numeric_cols)
        if dimred_notes:
            detailed["dimensionality_reduction"].extend(dimred_notes)

        # 6) Time-series features (optional STL)
        detailed["time_series_features"] = self._time_series_features(data, numeric_cols, cfg)

        suggestions["_detailed"] = detailed
        return suggestions

    # -------------------- Config & Utilities --------------------
    class _Cfg:
        def __init__(self, **kw):
            self.time_col: Optional[str] = kw.get("time_col")
            self.target_col: Optional[str] = kw.get("target")
            self.random_state: int = int(kw.get("random_state", 42))
            self.max_numeric_transforms: int = int(kw.get("max_numeric_transforms", 8))
            self.top_rank_for_interactions: int = int(kw.get("top_rank_for_interactions", 10))
            self.max_interactions: int = int(kw.get("max_interactions", 50))
            self.corr_gate: float = float(kw.get("interaction_corr_gate", 0.9))
            self.mi_gate: float = float(kw.get("interaction_mi_gate", 0.0))
            self.use_stl: bool = bool(kw.get("use_stl", True))
            self.stl_season_len: int = int(kw.get("stl_season_len", 13))
            self.lag_list: List[int] = list(kw.get("lags", [1, 7]))
            self.roll_windows: List[int] = list(kw.get("roll_windows", [3, 7]))
            self.limit_ts_cols: int = int(kw.get("limit_ts_cols", 5))
            self.enable_shap: bool = bool(kw.get("enable_shap", False))
            self.shap_max_n: int = int(kw.get("shap_max_n", 1000))

    def _get_cfg(self, config: AnalysisConfig) -> "_Cfg":
        return self._Cfg(**{k: getattr(config, k) for k in dir(config) if not k.startswith("_")})

    # -------------------- (1) Numeric transforms --------------------
    def _suggest_numeric_transforms(
        self, data: pd.DataFrame, numeric_cols: List[str], cfg: "_Cfg"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import scipy.stats as sps
        from scipy.stats import jarque_bera, shapiro

        suggestions: Dict[str, Any] = {}
        details: Dict[str, Any] = {}

        for col in numeric_cols:
            s = data[col].dropna()
            if s.size < 10:
                continue

            # Robust stats
            mean = float(s.mean())
            std = float(s.std(ddof=1))
            med = float(s.median())
            mad = float(getattr(s, "mad", lambda: np.median(np.abs(s - med)) * 1.4826)())
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            cv = std / (abs(mean) + 1e-12) if np.isfinite(std) else np.nan
            skew = float(sps.skew(s, bias=False))
            kurt = float(sps.kurtosis(s, fisher=False, bias=False))
            outlier_pct = float(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).mean() * 100.0)

            # Normality tests (sampled)
            ss = s.sample(min(5000, len(s)), random_state=cfg.random_state)
            jb_p = np.nan
            sw_p = np.nan
            try:
                _, jb_p = jarque_bera(ss)
            except Exception:
                pass
            try:
                _, sw_p = shapiro(ss)
            except Exception:
                pass

            transforms: List[str] = []
            # Skewness handling
            if abs(skew) > 2:
                if s.min() > 0:
                    transforms += [f"np.log1p({col})", f"np.sqrt({col})", f"1/({col}+1e-8)"]
                else:
                    transforms += [
                        f"np.sign({col})*np.log1p(np.abs({col}))",
                        f"scipy.stats.yeojohnson({col})[0]",
                    ]
            # Heavy tails & outliers
            if kurt > 3:
                transforms += [
                    f"np.tanh({col}/({std:+.6g}+1e-12))",
                    f"scipy.stats.rankdata({col})/len({col})",
                ]
            if outlier_pct > 10:
                transforms += [
                    f"scipy.stats.mstats.winsorize({col}, limits=(0.05,0.05))",
                    f"RobustScaler().fit_transform({col}.values.reshape(-1,1)).ravel()",
                ]
            # High variability
            if cv > 3:
                transforms += [
                    f"pd.qcut({col}, q=10, labels=False, duplicates='drop')",
                    f"PowerTransformer(method='yeo-johnson').fit_transform({col}.values.reshape(-1,1)).ravel()",
                ]
            # Extra-safe nonnegative transforms
            if s.min() >= 0:
                transforms += [f"np.cbrt({col})", f"np.exp(-{col}/({mean:+.6g}+1e-12))"]
            # Scaling
            transforms += [
                f"({col}-{mean:+.6g})/({std:+.6g}+1e-12)",
                f"({col}-{med:+.6g})/({mad:+.6g}+1e-12)",
            ]

            details[col] = {
                "stats": {
                    "count": int(s.size),
                    "mean": mean,
                    "std": std,
                    "median": med,
                    "mad": mad,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "cv": cv,
                    "skewness": skew,
                    "kurtosis": kurt,
                    "outliers_pct": outlier_pct,
                    "normality_jb_p": jb_p,
                    "normality_sw_p": sw_p,
                },
                "recommended": transforms[:cfg.max_numeric_transforms],
            }
            suggestions[col] = transforms[:4]  # back-compat short list

        return suggestions, details

    # -------------------- (2) Ranking (+ SHAP opt) --------------------
    def _rank_features(self, data: pd.DataFrame, numeric_cols: List[str], cfg: "_Cfg") -> Dict[str, float]:
        X = data[numeric_cols].copy().fillna(data[numeric_cols].median())
        y = data[cfg.target_col]
        if y.dtype.kind in "biufc":
            y = y.fillna(y.median())

        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        models = [
            RandomForestRegressor(n_estimators=200, random_state=cfg.random_state),
            GradientBoostingRegressor(n_estimators=150, random_state=cfg.random_state),
            ExtraTreesRegressor(n_estimators=300, random_state=cfg.random_state),
        ]

        imp: Dict[str, List[float]] = {c: [] for c in numeric_cols}
        for m in models:
            try:
                m.fit(X, y)
                fi = getattr(m, "feature_importances_", None)
                if fi is not None:
                    for c, v in zip(numeric_cols, fi):
                        imp[c].append(float(v))
            except Exception:
                continue

        avg_imp = {c: (float(np.mean(v)) if len(v) else 0.0) for c, v in imp.items()}
        ranked = dict(sorted(avg_imp.items(), key=lambda kv: kv[1], reverse=True))

        # Optional SHAP (read-only; doesn’t change ranking)
        if cfg.enable_shap and len(X) < cfg.shap_max_n:
            try:
                import shap
                from sklearn.ensemble import GradientBoostingRegressor
                rng = np.random.default_rng(cfg.random_state)
                gbr = GradientBoostingRegressor(n_estimators=100, random_state=cfg.random_state).fit(X, y)
                idx = rng.choice(len(X), size=min(200, len(X)), replace=False)
                Xs = X.iloc[idx]
                explainer = shap.Explainer(gbr, Xs)
                sv = explainer(Xs)
                shap_imp = np.abs(sv.values).mean(axis=0)
                ranked["shap_importance"] = dict(zip(numeric_cols, map(float, shap_imp)))
            except Exception:
                pass

        return ranked

    # -------------------- (3) Interactions --------------------
    def _suggest_interactions(
        self,
        data: pd.DataFrame,
        numeric_cols: List[str],
        detailed: Dict[str, Any],
        cfg: "_Cfg",
    ) -> List[str]:
        if len(numeric_cols) < 2:
            return []

        ranked = list(detailed.get("feature_ranking", {}).keys())[: cfg.top_rank_for_interactions]
        if not ranked:
            ranked = numeric_cols[: cfg.top_rank_for_interactions]

        sub = data[ranked]
        corr = sub.corr(method="pearson", min_periods=20)
        interactions: List[str] = []

        def _mi_disc(a: pd.Series, b: pd.Series, bins: int = 16) -> float:
            # stable, discretized MI
            qa = min(bins, max(2, a.nunique()))
            qb = min(bins, max(2, b.nunique()))
            aa = pd.qcut(a, q=qa, duplicates="drop").cat.codes
            bb = pd.qcut(b, q=qb, duplicates="drop").cat.codes
            joint = pd.crosstab(aa, bb, normalize=True)
            px = joint.sum(1).to_numpy()
            py = joint.sum(0).to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                pij = joint.to_numpy()
                log_term = np.log(pij / (px[:, None] * py[None, :] + 1e-12) + 1e-12)
                mi_val = float(np.nansum(pij * log_term))
            return max(mi_val, 0.0)

        for i, a in enumerate(ranked):
            for b in ranked[i + 1 :]:
                try:
                    if abs(corr.loc[a, b]) >= cfg.corr_gate:
                        continue
                    if cfg.mi_gate > 0:
                        if _mi_disc(sub[a].dropna(), sub[b].dropna()) < cfg.mi_gate:
                            continue
                    interactions += [
                        f"{a}*{b}",
                        f"{a}/({b}+1e-8)",
                        f"np.sqrt({a}**2+{b}**2)",
                        f"abs({a}-{b})",
                        f"({a}+{b})/2",
                    ]
                    if len(interactions) >= cfg.max_interactions:
                        return interactions
                except Exception:
                    continue

        return interactions

    # -------------------- (4) Cat encodings --------------------
    def _suggest_encodings(
        self, data: pd.DataFrame, categorical_cols: List[str], cfg: "_Cfg"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        suggestions: Dict[str, Any] = {}
        details: Dict[str, Any] = {}

        for col in categorical_cols:
            nunique = int(data[col].nunique(dropna=True))
            null_pct = float(data[col].isnull().mean() * 100.0)

            enc: List[str] = []
            if 2 <= nunique <= 5:
                enc.append("OneHot(drop_first=True)")
            elif nunique <= 20:
                enc += ["OrdinalEncoder"]
                if cfg.target_col:
                    enc += ["TargetEncoder (CV)", "WOEEncoder (CV)"]
            else:
                enc += ["HashingEncoder", "FrequencyEncoder"]
                if cfg.target_col:
                    enc += ["CatBoostEncoder (CV)"]

            if nunique > 10:
                enc.append("RareCategoryEncoder(threshold≈1%)")

            suggestions[col] = enc[:3]
            details[col] = {
                "cardinality": nunique,
                "null_percentage": null_pct,
                "strategies": enc,
                "note": "Use CV for supervised encoders to avoid leakage.",
            }

        return suggestions, details

    # -------------------- (5) VIF --------------------
    def _vif_screen(self, data: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        if not (1 < len(numeric_cols) < 80):
            return []
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            Xv = data[numeric_cols].copy().fillna(data[numeric_cols].median())
            V = [variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])]
            high_vif = [c for c, v in zip(numeric_cols, V) if v > 10]
            notes: List[str] = []
            if high_vif:
                notes.append(f"High VIF (>10): {high_vif}")
                notes.append("Consider PCA or dropping one per collinear group.")
            return notes
        except Exception:
            return []

    # -------------------- (6) Time series features --------------------
    def _time_series_features(
        self, data: pd.DataFrame, numeric_cols: List[str], cfg: "_Cfg"
    ) -> Dict[str, Any]:
        ts_feats: Dict[str, Any] = {}
        if not (cfg.time_col and cfg.time_col in data.columns):
            return ts_feats

        df = data.copy()
        if not np.issubdtype(df[cfg.time_col].dtype, np.datetime64):
            with pd.option_context("mode.chained_assignment", None):
                df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")

        # base temporal
        temporal = [
            f"df['{cfg.time_col}'].dt.hour",
            f"df['{cfg.time_col}'].dt.dayofweek",
            f"df['{cfg.time_col}'].dt.month",
            f"df['{cfg.time_col}'].dt.quarter",
            f"(df['{cfg.time_col}'].dt.dayofweek>=5).astype(int)  # is_weekend",
            f"df['{cfg.time_col}'].dt.dayofyear",
            f"np.sin(2*np.pi*df['{cfg.time_col}'].dt.hour/24)",
            f"np.cos(2*np.pi*df['{cfg.time_col}'].dt.hour/24)",
            f"np.sin(2*np.pi*df['{cfg.time_col}'].dt.dayofyear/365)",
            f"np.cos(2*np.pi*df['{cfg.time_col}'].dt.dayofyear/365)",
        ]
        ts_feats["temporal"] = temporal

        # per-series lag/roll + optional STL strengths
        use_cols = numeric_cols[: cfg.limit_ts_cols]
        if not use_cols:
            return ts_feats

        from statsmodels.tsa.seasonal import STL as _STL

        for col in use_cols:
            try:
                s = (
                    df[[cfg.time_col, col]]
                    .dropna()
                    .sort_values(cfg.time_col)
                    .set_index(cfg.time_col)[col]
                )
                feats = [f"df['{col}'].shift({L})" for L in cfg.lag_list]
                feats += [f"df['{col}'].rolling({w}).mean()" for w in cfg.roll_windows]
                feats += [f"df['{col}'].rolling({w}).std()" for w in cfg.roll_windows]
                feats += [f"df['{col}'].expanding().mean()", f"df['{col}'].pct_change()"]

                if cfg.use_stl and len(s) > cfg.stl_season_len * 3:
                    try:
                        res = _STL(s, seasonal=min(cfg.stl_season_len, max(7, len(s) // 6))).fit()
                        var = float(np.var(s))
                        t_strength = float(np.var(res.trend.dropna()) / (var + 1e-8))
                        s_strength = float(np.var(res.seasonal) / (var + 1e-8))
                        if t_strength > 0.3:
                            feats.append("STL_trend_component")
                        if s_strength > 0.3:
                            feats.append("STL_seasonal_component")
                    except Exception:
                        pass

                ts_feats[col] = feats
            except Exception:
                continue

        return ts_feats
