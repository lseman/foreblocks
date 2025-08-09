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

    # -------------------- (2) Ranking (+ SHAP opt, CV perm) --------------------
    def _rank_features(self, data: pd.DataFrame, numeric_cols: List[str], cfg: "_Cfg") -> Dict[str, float]:
        # ---- Prep ----
        X = data[numeric_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        y = data[cfg.target_col].astype(float)
        if y.dtype.kind in "biufc":
            y = y.fillna(y.median())

        import numpy as np
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
        # weights for blending signals
        w_imp = getattr(cfg, "w_impurity", 0.35)
        w_perm = getattr(cfg, "w_permutation", 0.40)
        w_mi  = getattr(cfg, "w_mutual_info", 0.15)
        w_lin = getattr(cfg, "w_linear", 0.10)

        # optional LightGBM / XGBoost if available
        model_zoo = [
            ("rf",  RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=-1)),
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
            model_zoo.append(("xgb", XGBRegressor(n_estimators=400, random_state=rs, n_jobs=-1, verbosity=0)))
        except Exception:
            pass

        # subsample rows for speed without hurting rank stability
        if len(X) > max_n:
            idx = np.random.RandomState(rs).choice(len(X), size=max_n, replace=False)
            X = X.iloc[idx]; y = y.iloc[idx]

        # containers
        impurity: dict[str, list[float]] = {c: [] for c in numeric_cols}
        permute:  dict[str, list[float]] = {c: [] for c in numeric_cols}

        # ---- CV loop: impurity + permutation importance on held-out ----
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
        for train_idx, test_idx in kf.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

            for name, m in model_zoo:
                try:
                    m.fit(Xtr, ytr)
                    fi = getattr(m, "feature_importances_", None)
                    if fi is not None and np.all(np.isfinite(fi)):
                        for c, v in zip(numeric_cols, fi):
                            impurity[c].append(float(max(v, 0.0)))
                    # permutation importance on validation fold
                    try:
                        pi = permutation_importance(m, Xte, yte, n_repeats=n_repeats, random_state=rs, n_jobs=-1)
                        for c, v in zip(numeric_cols, pi.importances_mean):
                            permute[c].append(float(max(v, 0.0)))
                    except Exception:
                        pass
                except Exception:
                    continue

        # ---- Mutual information (filter-like, single pass) ----
        try:
            mi_vals = mutual_info_regression(X, y, random_state=rs)
            mi_map = {c: float(max(v, 0.0)) for c, v in zip(numeric_cols, mi_vals)}
        except Exception:
            mi_map = {c: 0.0 for c in numeric_cols}

        # ---- Linear (Ridge) coefficient magnitude as a simple signal ----
        lin_map = {c: 0.0 for c in numeric_cols}
        try:
            ridge = RidgeCV(alphas=(1e-3, 1e-2, 1e-1, 1, 10))
            ridge.fit(X, y)
            for c, v in zip(numeric_cols, np.abs(ridge.coef_)):
                lin_map[c] = float(v)
        except Exception:
            pass

        # ---- Aggregate & normalize each channel to [0,1], then blend ----
        def _avg(d: dict[str, list[float]]) -> dict[str, float]:
            return {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in d.items()}

        imp_avg = _avg(impurity)
        perm_avg = _avg(permute)

        def _norm(m: dict[str, float]) -> dict[str, float]:
            arr = np.array(list(m.values()), dtype=float)
            s = arr.sum()
            if not np.isfinite(s) or s <= 0:
                # fallback to max-norm
                mx = arr.max() if len(arr) else 0.0
                return {k: (v / (mx + 1e-12) if mx > 0 else 0.0) for k, v in m.items()}
            return {k: float(v / s) for k, v in m.items()}

        imp_n  = _norm(imp_avg)
        perm_n = _norm(perm_avg)
        mi_n   = _norm(mi_map)
        lin_n  = _norm(lin_map)

        blended = {
            c: w_imp * imp_n.get(c, 0.0)
            + w_perm * perm_n.get(c, 0.0)
            + w_mi  * mi_n.get(c, 0.0)
            + w_lin * lin_n.get(c, 0.0)
            for c in numeric_cols
        }

        ranked = dict(sorted(blended.items(), key=lambda kv: kv[1], reverse=True))

        # ---- Optional SHAP (read-only; does NOT affect ranking) ----
        if getattr(cfg, "enable_shap", False) and len(X) < getattr(cfg, "shap_max_n", 12000):
            try:
                import shap

                # Use a strong tree model for SHAP if available
                if any(n == "lgbm" for n, _ in model_zoo):
                    from lightgbm import LGBMRegressor
                    model = LGBMRegressor(n_estimators=300, random_state=rs).fit(X, y)
                    explainer = shap.TreeExplainer(model)
                else:
                    # fallback to GBR for TreeExplainer
                    gbr = GradientBoostingRegressor(n_estimators=200, random_state=rs).fit(X, y)
                    explainer = shap.TreeExplainer(gbr)
                rng = np.random.default_rng(rs)
                m = min(1024, len(X))
                idx = rng.choice(len(X), size=m, replace=False)
                sv = explainer.shap_values(X.iloc[idx])
                shap_imp = np.mean(np.abs(sv), axis=0).astype(float)
                # store side-info without breaking the ranking dict usage downstream
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

        # ---------- Config (backward compatible) ----------
        top_k = getattr(cfg, "top_rank_for_interactions", 12)
        corr_gate = getattr(cfg, "corr_gate", 0.85)            # skip highly collinear pairs
        mi_gate = getattr(cfg, "mi_gate", 0.0)                 # keep if MI >= mi_gate
        dcor_gate = getattr(cfg, "dcor_gate", 0.05)            # small > 0 to filter noise
        hsic_gate = getattr(cfg, "hsic_gate", 0.0)             # set >0 to enable HSIC filtering
        gain_gate = getattr(cfg, "target_gain_gate", 0.002)    # ΔR² threshold to keep
        max_pairs = getattr(cfg, "max_interactions", 64)
        sample_n = getattr(cfg, "interaction_sample_size", 6000)
        allow_angle = getattr(cfg, "allow_angle", True)
        allow_minmax = getattr(cfg, "allow_minmax", True)
        allow_log1p = getattr(cfg, "allow_log1p", True)
        allow_rbf = getattr(cfg, "allow_rbf", False)           # off by default (can be heavy)
        rbf_sigma_mode = getattr(cfg, "rbf_sigma_mode", "median")  # "median" | "var"

        target_col = getattr(cfg, "target_col", None)
        y = None
        if target_col is not None and target_col in data.columns:
            y = data[target_col].astype(float)

        ranked = list(detailed.get("feature_ranking", {}).keys())[:top_k]
        if not ranked:
            ranked = numeric_cols[:top_k]

        # Work on a sampled subset for robustness/speed
        cols = [c for c in ranked if c in data.columns]
        work = data[cols].copy()
        if sample_n and len(work) > sample_n:
            work = work.sample(sample_n, random_state=42)
            if y is not None:
                y = y.loc[work.index]
        work = work.apply(pd.to_numeric, errors="coerce")
        y = y.astype(float) if y is not None else None

        # Precompute correlations (Pearson + Spearman)
        with np.errstate(all="ignore"):
            pearson = work.corr(method="pearson", min_periods=20)
            spearman = work.corr(method="spearman", min_periods=20)

        # ---------- Measures ----------
        def _mi_disc(a: pd.Series, b: pd.Series, bins: int = 16) -> float:
            a = a.dropna(); b = b.dropna()
            if len(a) < 30 or len(b) < 30:
                return 0.0
            n = min(len(a), len(b))
            if n > 8000:
                idx = np.random.RandomState(42).choice(n, 8000, replace=False)
                a = a.iloc[idx]; b = b.iloc[idx]
            qa = min(bins, max(2, a.nunique()))
            qb = min(bins, max(2, b.nunique()))
            aa = pd.qcut(a, q=qa, duplicates="drop").cat.codes
            bb = pd.qcut(b, q=qb, duplicates="drop").cat.codes
            joint = pd.crosstab(aa, bb, normalize=True)
            px = joint.sum(1).to_numpy()
            py = joint.sum(0).to_numpy()
            pij = joint.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                mi_val = float(np.nansum(pij * np.log(pij / (px[:, None] * py[None, :] + 1e-12) + 1e-12)))
            return max(mi_val, 0.0)

        def _dcor(a: np.ndarray, b: np.ndarray) -> float:
            # Unbiased distance correlation (O(n^2)), subsample if large
            a = a[~np.isnan(a) & ~np.isinf(a)]
            b = b[~np.isnan(b) & ~np.isinf(b)]
            n = min(len(a), len(b))
            if n < 30: return 0.0
            if n > 3000:
                rs = np.random.RandomState(42)
                idx = rs.choice(n, 3000, replace=False)
                a = a[idx]; b = b[idx]; n = 3000
            A = np.abs(a[:, None] - a[None, :])
            B = np.abs(b[:, None] - b[None, :])
            A = A - A.mean(0)[None, :] - A.mean(1)[:, None] + A.mean()
            B = B - B.mean(0)[None, :] - B.mean(1)[:, None] + B.mean()
            dcov2 = (A * B).sum() / (n * n)
            dvarx = (A * A).sum() / (n * n)
            dvary = (B * B).sum() / (n * n)
            denom = np.sqrt(max(dvarx, 1e-12) * max(dvary, 1e-12))
            return float(np.sqrt(max(dcov2, 0.0)) / (denom + 1e-12))

        def _hsic_gaussian(a: np.ndarray, b: np.ndarray) -> float:
            # Fast, normalized HSIC with median heuristic, subsampled
            n = min(len(a), len(b))
            if n < 80: return 0.0
            if n > 1200:
                rs = np.random.RandomState(42)
                idx = rs.choice(n, 1200, replace=False)
                a = a[idx]; b = b[idx]; n = 1200
            a = a.reshape(-1, 1); b = b.reshape(-1, 1)
            def _med_sigma(x):
                Dx = np.abs(x - x.T)
                m = np.median(Dx[Dx>0]) if np.any(Dx>0) else 1.0
                return m + 1e-12
            sig_a = _med_sigma(a); sig_b = _med_sigma(b)
            Ka = np.exp(-(np.square(a - a.T)) / (2 * sig_a ** 2))
            Kb = np.exp(-(np.square(b - b.T)) / (2 * sig_b ** 2))
            H = np.eye(n) - np.ones((n, n))/n
            Ka = H @ Ka @ H
            Kb = H @ Kb @ H
            hsic = (Ka * Kb).sum() / ((n - 1) ** 2)
            # Normalize to ~[0,1]
            na = np.linalg.norm(Ka, "fro"); nb = np.linalg.norm(Kb, "fro")
            return float(hsic / ((na * nb / (n - 1) ** 2) + 1e-12))

        def _delta_r2(a: pd.Series, b: pd.Series, y: pd.Series) -> float:
            # ΔR² between ridge on [a,b] vs [a,b,a*b]; standardize for stability
            try:
                df = pd.concat([a, b, y], axis=1).dropna()
                if len(df) < 80:
                    return 0.0
                xa = df.iloc[:, 0].astype(float).to_numpy()
                xb = df.iloc[:, 1].astype(float).to_numpy()
                yy = df.iloc[:, 2].astype(float).to_numpy()

                def _std(z): 
                    s = np.nanstd(z); 
                    return (z - np.nanmean(z)) / (s + 1e-12)

                xa = _std(xa); xb = _std(xb); yy = _std(yy)
                x1 = np.c_[xa, xb]
                x2 = np.c_[xa, xb, xa * xb]
                lam = 1e-3
                # Ridge via normal equations: (X^T X + λI)β = X^T y
                def _ridge_r2(X, y):
                    XtX = X.T @ X
                    XtX.flat[::X.shape[1]+1] += lam
                    beta = np.linalg.solve(XtX, X.T @ y)
                    yhat = X @ beta
                    ss_res = np.sum((y - yhat)**2)
                    ss_tot = np.sum((y - y.mean())**2) + 1e-12
                    return 1.0 - ss_res / ss_tot

                r2_base = _ridge_r2(x1, yy)
                r2_full = _ridge_r2(x2, yy)
                return max(0.0, float(r2_full - r2_base))
            except Exception:
                return 0.0

        interactions: List[str] = []
        kept_pairs = 0

        # ---------- Pair scan ----------
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                try:
                    # 1) Skip very high linear or rank correlation => redundant
                    p = float(pearson.loc[a, b]) if not np.isnan(pearson.loc[a, b]) else 0.0
                    s = float(spearman.loc[a, b]) if not np.isnan(spearman.loc[a, b]) else 0.0
                    if max(abs(p), abs(s)) >= corr_gate:
                        continue

                    a_s = work[a].astype(float)
                    b_s = work[b].astype(float)

                    # 2) Nonlinear dependence gates (keep only if at least one is informative)
                    keep_score = 0.0

                    if mi_gate > 0:
                        mi_val = _mi_disc(a_s, b_s)
                        if mi_val >= mi_gate:
                            keep_score += mi_val

                    dcor_val = _dcor(a_s.to_numpy(), b_s.to_numpy())
                    if dcor_val >= dcor_gate:
                        keep_score += dcor_val

                    if hsic_gate > 0:
                        hsic_val = _hsic_gaussian(a_s.to_numpy(), b_s.to_numpy())
                        if hsic_val >= hsic_gate:
                            keep_score += hsic_val

                    # If none of the dependency measures fired, skip
                    if keep_score <= 0 and (mi_gate > 0 or dcor_gate > 0 or hsic_gate > 0):
                        continue

                    # 3) Target-aware signal: does a*b add marginal value beyond a,b?
                    gain = 0.0
                    if y is not None:
                        gain = _delta_r2(a_s, b_s, y)
                        if gain < gain_gate:
                            # still allow if the nonlinear dependence was strong
                            if keep_score < (mi_gate + dcor_gate + hsic_gate) * 0.75:
                                continue

                    # 4) Emit a compact but expressive set of transforms
                    #    (strings only, to keep compatibility with your downstream pipeline)
                    cand: List[str] = [
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
                            cand += [f"np.exp(-(({a}-{b})**2)/(np.median(abs(({a}-{b})-np.median({a}-{b})))+1e-8)**2)"]
                        else:
                            cand += [f"np.exp(-(({a}-{b})**2)/(np.var({a}-{b})+1e-8))"]

                    for expr in cand:
                        interactions.append(expr)
                        if len(interactions) >= max_pairs:
                            return interactions

                    kept_pairs += 1
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
