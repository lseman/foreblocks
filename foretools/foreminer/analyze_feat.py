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
        categorical_cols = data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        for c in (cfg.time_col, cfg.target_col):
            if c in numeric_cols:
                numeric_cols.remove(c)
            if c in categorical_cols:
                categorical_cols.remove(c)

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
        # print(interactions)
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
        # detailed["time_series_features"] = self._time_series_features(data, numeric_cols, cfg)

        suggestions["_detailed"] = detailed
        return suggestions

    # -------------------- Config & Utilities --------------------
    class _Cfg:
        def __init__(self, **kw):
            self.time_col: Optional[str] = kw.get("time_col")
            self.target_col: Optional[str] = kw.get("target")
            self.random_state: int = int(kw.get("random_state", 42))
            self.max_numeric_transforms: int = int(kw.get("max_numeric_transforms", 8))
            self.top_rank_for_interactions: int = int(
                kw.get("top_rank_for_interactions", 10)
            )
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
        return self._Cfg(
            **{k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        )

    # -------------------- (1) Numeric transforms --------------------
    def _suggest_numeric_transforms(
        self, data: pd.DataFrame, numeric_cols: List[str], cfg: "_Cfg"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Suggest numeric transforms per column with robust stats + modern mappings.
        Returns (suggestions, details) where suggestions[col] is a short list (<=4)
        and details[col]["recommended"] respects cfg.max_numeric_transforms.
        """
        import numpy as np
        import pandas as pd
        import scipy.stats as sps
        from scipy.stats import jarque_bera, shapiro

        # -------------------- helpers --------------------
        rng = np.random.default_rng(getattr(cfg, "random_state", None))

        def _fmt(x: float) -> str:
            # compact numeric literal for embedding in strings
            if not np.isfinite(x):
                return "np.nan"
            return f"{x:.6g}"

        def _sample_series(s: pd.Series, cap: int = 5000) -> np.ndarray:
            a = s.to_numpy()
            n = a.size
            if n <= cap:
                return a
            idx = rng.choice(n, size=cap, replace=False)
            return a[idx]

        def _winsor_limits_from_iqr(
            s: pd.Series, k: float = 3.0
        ) -> Tuple[float, float]:
            q1, q3 = np.percentile(s, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            return float(lo), float(hi)

        def _mad(arr: np.ndarray, med: float) -> float:
            return float(np.median(np.abs(arr - med)) * 1.4826)

        def _uniq_keep_order(xs: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in xs:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        # -------------------- config-ish constants --------------------
        short_list_k = 4  # back-compat
        max_k = getattr(cfg, "max_numeric_transforms", 8)
        shapiro_cap = 5000  # Shapiro not recommended for N>5000
        quantile_bins = 10  # for qcut suggestions
        rank_eps = 1e-9

        suggestions: Dict[str, Any] = {}
        details: Dict[str, Any] = {}

        for col in numeric_cols:
            s: pd.Series = data[col].dropna()
            if s.size < 10:
                continue

            a = s.to_numpy()
            n = a.size

            # ---- robust + classical stats
            mean = float(np.mean(a))
            std = float(np.std(a, ddof=1)) if n > 1 else 0.0
            med = float(np.median(a))
            mad = _mad(a, med)
            q1, q3 = np.percentile(a, [25, 75])
            iqr = float(q3 - q1)
            cv = float(std / (abs(mean) + 1e-12)) if np.isfinite(std) else np.nan
            rcv = float(iqr / (abs(med) + 1e-12))  # robust CV
            skew = float(sps.skew(a, bias=False))
            # fisher=False → Pearson definition (3 for normal)
            kurt = float(sps.kurtosis(a, fisher=False, bias=False))

            # Outlier rates (IQR and robust MAD z>3.5)
            iqr_lo, iqr_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_outlier_pct = float(((a < iqr_lo) | (a > iqr_hi)).mean() * 100.0)
            mad_z = np.abs((a - med) / (mad + 1e-12))
            mad_outlier_pct = float((mad_z > 3.5).mean() * 100.0)
            outlier_pct = max(iqr_outlier_pct, mad_outlier_pct)

            # Normality tests on a sample (deterministic)
            ss = _sample_series(s, cap=shapiro_cap)
            try:
                _, jb_p = jarque_bera(ss)
                jb_p = float(jb_p)
            except Exception:
                jb_p = np.nan
            try:
                # Only if small enough
                if ss.size <= shapiro_cap:
                    _, sw_p = shapiro(ss)
                    sw_p = float(sw_p)
                else:
                    sw_p = np.nan
            except Exception:
                sw_p = np.nan

            # ---- decision heuristics (priorities)
            strongly_non_normal = (
                (not np.isnan(jb_p) and jb_p < 1e-3)
                or (abs(skew) > 2.0)
                or (kurt > 6.0)
            )
            moderately_non_normal = (
                (not np.isnan(jb_p) and jb_p < 0.05)
                or (abs(skew) > 1.0)
                or (kurt > 3.5)
            )

            # ---- build transform menu (monotone & robust first)
            T: List[str] = []

            # 1) Rank/Gauss rank (handles heavy tails & weird marginals)
            #   - rank -> [0,1], gaussrank -> ~N(0,1)
            T.append(f"(scipy.stats.rankdata({col})/(len({col})+{_fmt(rank_eps)}))")
            T.append(
                f"scipy.stats.norm.ppf((scipy.stats.rankdata({col})-0.5)/(len({col})+{_fmt(rank_eps)}))"
            )

            # 2) Power transforms
            if s.min() > 0:
                # Box-Cox (positive only) and log1p
                T.append(
                    f"PowerTransformer(method='box-cox').fit_transform({col}.values.reshape(-1,1)).ravel()"
                )
                T.append(f"np.log1p({col})")
            # Yeo–Johnson works on R
            T.append(
                f"PowerTransformer(method='yeo-johnson').fit_transform({col}.values.reshape(-1,1)).ravel()"
            )

            # 3) Asinh / sqrt / cbrt (numerically gentle, defined on R or R+)
            # asinh is robust alternative to log
            T.append(f"np.arcsinh({col})")
            if s.min() >= 0:
                T.append(f"np.sqrt({col})")
                T.append(f"np.cbrt({col})")

            # 4) Winsorization / clipping (IQR-based)
            lo_iqr3, hi_iqr3 = _winsor_limits_from_iqr(s, k=3.0)
            T.append(f"np.clip({col}, {_fmt(lo_iqr3)}, {_fmt(hi_iqr3)})")
            # SciPy mstats winsorize (5% each side) as a strong option
            T.append(f"scipy.stats.mstats.winsorize({col}, limits=(0.05, 0.05))")

            # 5) Quantile mapping & discretization
            T.append(
                f"pd.qcut({col}, q={quantile_bins}, labels=False, duplicates='drop')"
            )
            # QuantileTransformer to uniform and normal
            T.append(
                f"QuantileTransformer(output_distribution='uniform', n_quantiles=100, random_state={getattr(cfg, 'random_state', None)}).fit_transform({col}.values.reshape(-1,1)).ravel()"
            )
            T.append(
                f"QuantileTransformer(output_distribution='normal', n_quantiles=100, random_state={getattr(cfg, 'random_state', None)}).fit_transform({col}.values.reshape(-1,1)).ravel()"
            )

            # 6) Variability-aware rescales
            T.append(f"({col}-{_fmt(mean)})/({_fmt(std)}+1e-12)")
            T.append(f"({col}-{_fmt(med)})/({_fmt(mad)}+1e-12)")
            # Exponential dampening (kept from your version, but safer on mean≈0)
            if abs(mean) > 1e-8:
                T.append(f"np.exp(-{col}/({_fmt(mean)}+1e-12))")

            # 7) Inversion (only if strictly positive and nonzero-heavy)
            if s.min() > 0 and (abs(skew) > 1.0 or strongly_non_normal):
                T.append(f"1/({col}+1e-8)")

            # 8) Domain-aware: if CVs are huge, emphasize stabilizers
            if cv > 3 or rcv > 1.5 or outlier_pct > 10:
                # push robust-centered scaling
                T.append(
                    f"(np.clip({col}, {_fmt(lo_iqr3)}, {_fmt(hi_iqr3)})-{_fmt(med)})/({_fmt(mad)}+1e-12)"
                )

            # Re-rank priority: emphasize monotone mappings for non-normal marginals
            if strongly_non_normal:
                priority = [
                    "PowerTransformer(method='yeo-johnson')",
                    "PowerTransformer(method='box-cox')",
                    "np.arcsinh(",
                    "np.log1p(",
                    "rankdata(",
                    "norm.ppf(",
                    "QuantileTransformer(output_distribution='normal'",
                    "QuantileTransformer(output_distribution='uniform'",
                    "winsorize(",
                    "np.clip(",
                    "pd.qcut(",
                    "/(_mad",  # robust scale
                    "/(_std",  # standard scale
                ]
            elif moderately_non_normal:
                priority = [
                    "PowerTransformer(method='yeo-johnson')",
                    "np.arcsinh(",
                    "rankdata(",
                    "QuantileTransformer(output_distribution='normal'",
                    "winsorize(",
                    "np.clip(",
                    "pd.qcut(",
                    "PowerTransformer(method='box-cox')" if s.min() > 0 else "",
                    "np.log1p(" if s.min() > 0 else "",
                ]
            else:
                priority = [
                    "/(_std",
                    "/(_mad",
                    "PowerTransformer(method='yeo-johnson')",
                    "rankdata(",
                    "np.arcsinh(",
                    "QuantileTransformer(output_distribution='normal'",
                    "np.clip(",
                    "winsorize(",
                ]
            priority = [p for p in priority if p]

            # Deduplicate and sort by simple priority key
            T = _uniq_keep_order(T)

            def _score(t: str) -> int:
                for i, key in enumerate(priority):
                    if key in t:
                        return i
                return len(priority) + 1

            T.sort(key=_score)

            recommended = T[:max_k]
            suggestions[col] = T[:short_list_k]

            details[col] = {
                "stats": {
                    "count": int(n),
                    "mean": mean,
                    "std": std,
                    "median": med,
                    "mad": mad,
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": iqr,
                    "cv": cv,
                    "robust_cv": rcv,
                    "skewness": skew,
                    "kurtosis": kurt,
                    "outliers_pct": outlier_pct,
                    "normality_jb_p": jb_p,
                    "normality_sw_p": sw_p,
                },
                "recommended": recommended,
            }

        return suggestions, details

    # -------------------- (2) Ranking (+ SHAP opt, CV perm) --------------------
    def _rank_features(
        self, data: pd.DataFrame, numeric_cols: List[str], cfg: "_Cfg"
    ) -> Dict[str, float]:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            HistGradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import KFold

        rs = getattr(cfg, "random_state", 42)
        n_splits = getattr(cfg, "rank_cv_splits", 5)
        n_repeats = getattr(cfg, "rank_perm_repeats", 3)
        max_n = getattr(cfg, "rank_sample_size", 20000)

        # weights for blending signals (sum need not be 1; we renormalize later)
        w_imp = getattr(cfg, "w_impurity", 0.35)
        w_perm = getattr(cfg, "w_permutation", 0.40)
        w_mi = getattr(cfg, "w_mutual_info", 0.15)
        w_lin = getattr(cfg, "w_linear", 0.10)

        # -------------------- Prep --------------------
        X = data[numeric_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        # Median impute per-column (OK for tree models, keeps scale for Ridge)
        X = X.fillna(X.median(numeric_only=True))

        y = pd.to_numeric(data[getattr(cfg, "target_col")], errors="coerce")
        if y.dtype.kind in "biufc":
            y = y.fillna(y.median())

        # Optional subsample for speed (deterministic)
        if len(X) > max_n:
            idx = np.random.RandomState(rs).choice(len(X), size=max_n, replace=False)
            X = X.iloc[idx]
            y = y.iloc[idx]

        # Drop constant columns (avoid MI/perm issues)
        nunique = X.nunique(dropna=False)
        keep = nunique[nunique > 1].index.tolist()
        X = X[keep]
        numeric_cols = keep
        if len(numeric_cols) == 0:
            return {}

        # Fast near-duplicate pruning (corr > 0.995 keep first)
        with np.errstate(all="ignore"):
            cmat = X.corr(method="spearman")
        to_drop = set()
        cols_order = list(cmat.columns)
        for i, c in enumerate(cols_order):
            if c in to_drop:
                continue
            # only scan upper triangle
            corr_row = cmat.loc[c].iloc[i + 1 :]
            dups = corr_row.index[np.abs(corr_row.values) > 0.995]
            for d in dups:
                to_drop.add(d)
        if to_drop:
            X = X.drop(columns=list(to_drop))
            numeric_cols = [c for c in numeric_cols if c not in to_drop]
            if len(numeric_cols) == 0:
                return {}

        # -------------------- Model zoo --------------------
        model_zoo = [
            ("rf", RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=-1)),
            ("etr", ExtraTreesRegressor(n_estimators=400, random_state=rs, n_jobs=-1)),
            ("gbr", GradientBoostingRegressor(n_estimators=200, random_state=rs)),
            ("hgb", HistGradientBoostingRegressor(max_depth=None, random_state=rs)),
        ]
        try:
            from lightgbm import LGBMRegressor

            model_zoo.append(("lgbm", LGBMRegressor(n_estimators=400, random_state=rs)))
        except Exception:
            pass
        try:
            from xgboost import XGBRegressor

            model_zoo.append(
                (
                    "xgb",
                    XGBRegressor(
                        n_estimators=400, random_state=rs, n_jobs=-1, verbosity=0
                    ),
                )
            )
        except Exception:
            pass

        # -------------------- Containers --------------------
        impurity: dict[str, list[float]] = {c: [] for c in numeric_cols}
        permute: dict[str, list[float]] = {c: [] for c in numeric_cols}

        # -------------------- CV loop: impurity + permutation on held-out --------------------
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
        for train_idx, test_idx in kf.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

            for name, m in model_zoo:
                try:
                    m.fit(Xtr, ytr)
                    fi = getattr(m, "feature_importances_", None)
                    if (
                        fi is not None
                        and np.all(np.isfinite(fi))
                        and len(fi) == X.shape[1]
                    ):
                        for c, v in zip(X.columns, fi):
                            impurity[c].append(float(max(v, 0.0)))
                    # permutation importance on validation fold
                    try:
                        pi = permutation_importance(
                            m, Xte, yte, n_repeats=n_repeats, random_state=rs, n_jobs=-1
                        )
                        if len(pi.importances_mean) == X.shape[1]:
                            for c, v in zip(X.columns, pi.importances_mean):
                                permute[c].append(float(max(v, 0.0)))
                    except Exception:
                        pass
                except Exception:
                    continue

        # -------------------- Mutual information --------------------
        try:
            mi_vals = mutual_info_regression(X, y, random_state=rs)
            mi_map = {c: float(max(v, 0.0)) for c, v in zip(X.columns, mi_vals)}
        except Exception:
            mi_map = {c: 0.0 for c in X.columns}

        # -------------------- Linear (Ridge) signal --------------------
        lin_map = {c: 0.0 for c in X.columns}
        try:
            ridge = RidgeCV(alphas=(1e-3, 1e-2, 1e-1, 1, 10))
            ridge.fit(X, y)
            for c, v in zip(X.columns, np.abs(ridge.coef_).astype(float)):
                lin_map[c] = float(v)
        except Exception:
            pass

        # -------------------- Aggregate & robust-normalize channels --------------------
        def _avg(d: dict[str, list[float]]) -> dict[str, float]:
            return {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in d.items()}

        imp_avg = _avg(impurity)
        perm_avg = _avg(permute)

        def _robust_norm(m: dict[str, float]) -> dict[str, float]:
            """Normalize to [0,1] robustly via (x - q05)/(q95 - q05); fallback to max-norm."""
            arr = np.array(list(m.values()), dtype=float)
            if len(arr) == 0:
                return {}
            q05, q95 = np.percentile(arr, [5, 95])
            scale = max(q95 - q05, 1e-12)
            normed = {
                k: float(np.clip((v - q05) / scale, 0.0, 1.0)) for k, v in m.items()
            }
            # if all zeros (degenerate), fallback to max-norm
            if sum(normed.values()) == 0.0:
                mx = arr.max()
                if mx > 0:
                    normed = {k: float(v / mx) for k, v in m.items()}
            return normed

        imp_n = _robust_norm(imp_avg)
        perm_n = _robust_norm(perm_avg)
        mi_n = _robust_norm(mi_map)
        lin_n = _robust_norm(lin_map)

        # Blend (then renormalize blended scores to [0,1] for readability)
        blended_raw = {
            c: w_imp * imp_n.get(c, 0.0)
            + w_perm * perm_n.get(c, 0.0)
            + w_mi * mi_n.get(c, 0.0)
            + w_lin * lin_n.get(c, 0.0)
            for c in X.columns
        }
        # final min-max normalization for interpretability
        vals = np.array(list(blended_raw.values()), dtype=float)
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if np.isfinite(vmax - vmin) and (vmax - vmin) > 1e-12:
            blended = {
                k: float((v - vmin) / (vmax - vmin)) for k, v in blended_raw.items()
            }
        else:
            blended = blended_raw

        ranked = dict(sorted(blended.items(), key=lambda kv: kv[1], reverse=True))

        # -------------------- Optional SHAP side-info (does NOT affect ranking) --------------------
        if getattr(cfg, "enable_shap", False) and len(X) < getattr(
            cfg, "shap_max_n", 12000
        ):
            try:
                import shap

                # choose a strong tree model for SHAP if available
                try:
                    from lightgbm import LGBMRegressor

                    model = LGBMRegressor(n_estimators=300, random_state=rs).fit(X, y)
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    gbr = GradientBoostingRegressor(
                        n_estimators=200, random_state=rs
                    ).fit(X, y)
                    explainer = shap.TreeExplainer(gbr)
                rng = np.random.default_rng(rs)
                m = min(1024, len(X))
                idx = rng.choice(len(X), size=m, replace=False)
                sv = explainer.shap_values(X.iloc[idx])
                shap_imp = np.mean(np.abs(sv), axis=0).astype(float)
                # store side-info under a reserved key to avoid feature-name collisions
                ranked["__shap_importance__"] = dict(
                    zip(X.columns, map(float, shap_imp))
                )
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
        import numpy as np
        import pandas as pd

        # ---------- Config (backward compatible, friendlier defaults) ----------
        top_k = getattr(cfg, "top_rank_for_interactions", 12)
        corr_gate = getattr(cfg, "corr_gate", 0.95)  # was 0.85
        mi_gate = getattr(cfg, "mi_gate", 0.0)
        dcor_gate = getattr(cfg, "dcor_gate", 0.0)  # was 0.05 (made opt-in)
        hsic_gate = getattr(cfg, "hsic_gate", 0.0)
        gain_gate = getattr(cfg, "target_gain_gate", 1e-4)  # was 0.002
        max_pairs = getattr(cfg, "max_interactions", 64)
        sample_n = getattr(cfg, "interaction_sample_size", 6000)
        allow_angle = getattr(cfg, "allow_angle", True)
        allow_minmax = getattr(cfg, "allow_minmax", True)
        allow_log1p = getattr(cfg, "allow_log1p", True)
        allow_rbf = getattr(cfg, "allow_rbf", False)
        rbf_sigma_mode = getattr(cfg, "rbf_sigma_mode", "median")  # "median" | "var"
        random_state = getattr(cfg, "random_state", 42)

        rng = np.random.default_rng(random_state)

        # Target handling (optional)
        target_col = getattr(cfg, "target_col", None)
        y = None
        if target_col is not None and target_col in data.columns:
            y = pd.to_numeric(data[target_col], errors="coerce")

        # Use ranking if provided, else fallback to numeric cols
        ranked = list(detailed.get("feature_ranking", {}).keys())[:top_k]
        if not ranked:
            ranked = numeric_cols[:top_k]

        cols = [c for c in ranked if c in data.columns]
        if len(cols) < 2:
            return []

        # Work on a sampled subset for robustness/speed
        work = data[cols].copy()
        if sample_n and len(work) > sample_n:
            work = work.sample(sample_n, random_state=random_state)
            if y is not None:
                y = y.loc[work.index]

        work = work.apply(pd.to_numeric, errors="coerce")
        y = y.astype(float) if y is not None else None

        # Precompute correlations (Pearson + Spearman)
        with np.errstate(all="ignore"):
            pearson = work.corr(method="pearson", min_periods=20)
            spearman = work.corr(method="spearman", min_periods=20)

        # ---------- Measures (aligned inputs) ----------
        def _align_pair(
            a_s: pd.Series, b_s: pd.Series
        ) -> tuple[np.ndarray, np.ndarray]:
            mask = a_s.notna() & b_s.notna()
            if mask.sum() < 30:
                return np.array([]), np.array([])
            aa = a_s[mask].astype(float).to_numpy()
            bb = b_s[mask].astype(float).to_numpy()
            return aa, bb

        def _mi_disc_aligned(aa: np.ndarray, bb: np.ndarray, bins: int = 16) -> float:
            # Discretized MI using quantile bins; robust and fast
            n = min(aa.size, bb.size)
            if n < 30:
                return 0.0
            if n > 8000:
                idx = rng.choice(n, 8000, replace=False)
                aa = aa[idx]
                bb = bb[idx]
                n = 8000
            qa = max(2, min(bins, len(np.unique(aa))))
            qb = max(2, min(bins, len(np.unique(bb))))
            try:
                aaq = pd.qcut(aa, q=qa, duplicates="drop").codes
                bbq = pd.qcut(bb, q=qb, duplicates="drop").codes
            except Exception:
                # Fallback to rank-based bins if qcut fails
                ra = pd.Series(aa).rank(method="average").to_numpy()
                rb = pd.Series(bb).rank(method="average").to_numpy()
                aaq = pd.qcut(ra, q=qa, duplicates="drop").codes
                bbq = pd.qcut(rb, q=qb, duplicates="drop").codes
            joint = pd.crosstab(aaq, bbq, normalize=True)
            px = joint.sum(1).to_numpy()
            py = joint.sum(0).to_numpy()
            pij = joint.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                mi_val = float(
                    np.nansum(
                        pij * np.log(pij / (px[:, None] * py[None, :] + 1e-12) + 1e-12)
                    )
                )
            return max(mi_val, 0.0)

        def _dcor_aligned(aa: np.ndarray, bb: np.ndarray) -> float:
            # Unbiased distance correlation (O(n^2)), subsample if large
            n = min(aa.size, bb.size)
            if n < 30:
                return 0.0
            if n > 3000:
                idx = rng.choice(n, 3000, replace=False)
                aa = aa[idx]
                bb = bb[idx]
                n = 3000
            A = np.abs(aa[:, None] - aa[None, :])
            B = np.abs(bb[:, None] - bb[None, :])
            A = A - A.mean(0)[None, :] - A.mean(1)[:, None] + A.mean()
            B = B - B.mean(0)[None, :] - B.mean(1)[:, None] + B.mean()
            dcov2 = (A * B).sum() / (n * n)
            dvarx = (A * A).sum() / (n * n)
            dvary = (B * B).sum() / (n * n)
            denom = np.sqrt(max(dvarx, 1e-12) * max(dvary, 1e-12))
            return float(np.sqrt(max(dcov2, 0.0)) / (denom + 1e-12))

        def _hsic_gaussian_aligned(aa: np.ndarray, bb: np.ndarray) -> float:
            # Normalized HSIC with median heuristic, subsampled
            n = min(aa.size, bb.size)
            if n < 80:
                return 0.0
            if n > 1200:
                idx = rng.choice(n, 1200, replace=False)
                aa = aa[idx]
                bb = bb[idx]
                n = 1200
            a = aa.reshape(-1, 1)
            b = bb.reshape(-1, 1)

            def _med_sigma(x):
                Dx = np.abs(x - x.T)
                m = np.median(Dx[Dx > 0]) if np.any(Dx > 0) else 1.0
                return m + 1e-12

            sig_a = _med_sigma(a)
            sig_b = _med_sigma(b)
            Ka = np.exp(-((a - a.T) ** 2) / (2 * sig_a**2))
            Kb = np.exp(-((b - b.T) ** 2) / (2 * sig_b**2))
            H = np.eye(n) - np.ones((n, n)) / n
            Ka = H @ Ka @ H
            Kb = H @ Kb @ H
            hsic = (Ka * Kb).sum() / ((n - 1) ** 2)
            na = np.linalg.norm(Ka, "fro")
            nb = np.linalg.norm(Kb, "fro")
            return float(hsic / ((na * nb / (n - 1) ** 2) + 1e-12))

        def _delta_r2(a_s: pd.Series, b_s: pd.Series, y_s: pd.Series) -> float:
            # ΔR² between ridge on [a,b] vs [a,b,a*b]; standardize for stability
            try:
                df = pd.concat([a_s, b_s, y_s], axis=1).dropna()
                if len(df) < 80:
                    return 0.0
                xa = df.iloc[:, 0].astype(float).to_numpy()
                xb = df.iloc[:, 1].astype(float).to_numpy()
                yy = df.iloc[:, 2].astype(float).to_numpy()

                def _std(z):
                    s = np.nanstd(z)
                    return (z - np.nanmean(z)) / (s + 1e-12)

                xa = _std(xa)
                xb = _std(xb)
                yy = _std(yy)
                x1 = np.c_[xa, xb]
                x2 = np.c_[xa, xb, xa * xb]
                lam = 1e-3

                def _ridge_r2(X, yv):
                    XtX = X.T @ X
                    XtX.flat[:: X.shape[1] + 1] += lam
                    beta = np.linalg.solve(XtX, X.T @ yv)
                    yhat = X @ beta
                    ss_res = np.sum((yv - yhat) ** 2)
                    ss_tot = np.sum((yv - yv.mean()) ** 2) + 1e-12
                    return 1.0 - ss_res / ss_tot

                r2_base = _ridge_r2(x1, yy)
                r2_full = _ridge_r2(x2, yy)
                return max(0.0, float(r2_full - r2_base))
            except Exception:
                return 0.0

        # ---------- Scan pairs ----------
        interactions: List[str] = []
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                try:
                    # 1) Skip highly collinear pairs
                    p = (
                        float(pearson.loc[a, b])
                        if not np.isnan(pearson.loc[a, b])
                        else 0.0
                    )
                    s = (
                        float(spearman.loc[a, b])
                        if not np.isnan(spearman.loc[a, b])
                        else 0.0
                    )
                    if max(abs(p), abs(s)) >= corr_gate:
                        continue

                    a_s = work[a].astype(float)
                    b_s = work[b].astype(float)

                    # Align once
                    aa, bb = _align_pair(a_s, b_s)
                    if aa.size == 0:
                        continue

                    # 2) Nonlinear dependence gates (opt-in by positive thresholds)
                    keep_score = 0.0

                    if mi_gate > 0:
                        mi_val = _mi_disc_aligned(aa, bb)
                        if mi_val >= mi_gate:
                            keep_score += mi_val

                    if dcor_gate > 0:
                        dcor_val = _dcor_aligned(aa, bb)
                        if dcor_val >= dcor_gate:
                            keep_score += dcor_val

                    if hsic_gate > 0:
                        hsic_val = _hsic_gaussian_aligned(aa, bb)
                        if hsic_val >= hsic_gate:
                            keep_score += hsic_val

                    if (
                        mi_gate > 0 or dcor_gate > 0 or hsic_gate > 0
                    ) and keep_score <= 0:
                        # none fired
                        continue

                    # 3) Target-aware signal: allow bypass if strong
                    gain = 0.0
                    if y is not None:
                        gain = _delta_r2(a_s, b_s, y)
                        if gain >= gain_gate:
                            keep_score = max(keep_score, gain)  # bypass the gate

                    if (
                        mi_gate > 0 or dcor_gate > 0 or hsic_gate > 0 or y is not None
                    ) and (keep_score <= 0):
                        continue

                    # 4) Emit transforms (strings for downstream compatibility)
                    cand = [
                        f"{a}*{b}",
                        f"{a}/({b}+1e-8)",
                        f"{b}/({a}+1e-8)",
                        f"np.sqrt(({a})**2+({b})**2)",
                        f"abs({a}-{b})",
                        f"({a}+{b})/2",
                    ]
                    if allow_minmax:
                        cand += [f"np.minimum({a},{b})", f"np.maximum({a},{b})"]
                    if allow_angle:
                        cand += [f"np.arctan2({b},{a})"]
                    if allow_log1p:
                        cand += [f"np.log1p(abs({a})+abs({b}))"]
                    if allow_rbf:
                        if rbf_sigma_mode == "median":
                            # median absolute deviation of (a-b) as scale
                            cand += [
                                f"np.exp(-(({a}-{b})**2)/(np.median(abs(({a}-{b})-np.median({a}-{b})))+1e-8)**2)"
                            ]
                        else:
                            cand += [f"np.exp(-(({a}-{b})**2)/(np.var({a}-{b})+1e-8))"]

                    for expr in cand:
                        interactions.append(expr)
                        if len(interactions) >= max_pairs:
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
