from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.stats import jarque_bera, shapiro
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from ..aux.hsic import HSIC  # your class
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

    def _suggest_numeric_transforms(
        self, data: pd.DataFrame, numeric_cols: List[str], cfg: "_Cfg"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Suggest numeric transforms per column with robust scoring.
        Returns (suggestions, details) where suggestions[col] is a short list (<=4)
        and details[col]["recommended"] respects cfg.max_numeric_transforms.
        """
        rs = getattr(cfg, "random_state", 42)
        rng = np.random.default_rng(rs)

        short_list_k = 4
        max_k       = getattr(cfg, "max_numeric_transforms", 8)
        shapiro_cap = 5000
        sample_cap  = getattr(cfg, "transform_score_sample_size", 5000)
        qbins       = getattr(cfg, "transform_qcut_bins", 10)

        # Optional target-aware scoring via HSIC
        target_name   = getattr(cfg, "target_col", None)
        y_series      = pd.to_numeric(data[target_name], errors="coerce") if target_name in (data.columns if target_name else []) else None
        y_present     = y_series is not None and y_series.notna().sum() >= 5 and y_series.nunique() > 1
        hsic_weight   = float(getattr(cfg, "transform_hsic_weight", 0.25))  # contributes to ranking if y_present
        hsic_min_n    = getattr(cfg, "rank_hsic_min_samples", 50)
        hsic_est      = getattr(cfg, "rank_hsic_estimator", "biased")
        hsic_norm     = getattr(cfg, "rank_hsic_normalize", True)
        hsic_kx       = getattr(cfg, "rank_hsic_kernel_x", "rbf")
        hsic_ky       = getattr(cfg, "rank_hsic_kernel_y", "rbf")
        hsic_use_numba= getattr(cfg, "rank_hsic_use_numba", True)
        hsic_scorer   = HSIC(kernel_x=hsic_kx, kernel_y=hsic_ky, estimator=hsic_est, normalize=hsic_norm,
                            use_numba=hsic_use_numba, random_state=rs) if y_present else None

        def _fmt(x: float) -> str:
            if not np.isfinite(x): return "np.nan"
            return f"{x:.6g}"

        def _sample(a: np.ndarray, cap: int = sample_cap) -> np.ndarray:
            n = a.size
            if n <= cap: return a
            idx = rng.choice(n, size=cap, replace=False)
            return a[idx]

        def _winsor_limits_from_iqr(a: np.ndarray, k: float = 3.0) -> Tuple[float, float]:
            q1, q3 = np.percentile(a, [25, 75])
            iqr = q3 - q1
            return float(q1 - k * iqr), float(q3 + k * iqr)

        def _mad(a: np.ndarray, med: float) -> float:
            return float(np.median(np.abs(a - med)) * 1.4826)

        def _jb_p(a: np.ndarray) -> float:
            try:
                _, p = jarque_bera(a)
                return float(p)
            except Exception:
                return np.nan

        def _sw_p(a: np.ndarray) -> float:
            try:
                if a.size <= shapiro_cap:
                    _, p = shapiro(a)
                    return float(p)
                return np.nan
            except Exception:
                return np.nan

        # concrete functions (for *scoring*); map to output strings for your executor
        def signed_log1p(v):  # robust on R
            return np.sign(v) * np.log1p(np.abs(v))

        def asinh(v):
            return np.arcsinh(v)

        # Build transform catalog per column with safe guards; each item: (name_key, callable, string_repr)
        def _catalog_for(col: str, a: np.ndarray) -> List[Tuple[str, callable, str]]:
            out: List[Tuple[str, callable, str]] = []
            n = a.size
            med = float(np.median(a))
            mad = _mad(a, med)
            mean = float(np.mean(a))
            std  = float(np.std(a, ddof=1)) if n > 1 else 0.0
            lo_iqr3, hi_iqr3 = _winsor_limits_from_iqr(a, k=3.0)
            pos_min = np.nanmin(a) if a.size else np.nan

            # Rank and gauss-rank via sklearn (more stable), but mirror strings
            qt_u = QuantileTransformer(output_distribution="uniform", n_quantiles=min(100, max(10, n)), random_state=rs, subsample=int(1e9))
            qt_n = QuantileTransformer(output_distribution="normal",  n_quantiles=min(100, max(10, n)), random_state=rs, subsample=int(1e9))

            # PowerTransforms
            pt_yj = PowerTransformer(method="yeo-johnson", standardize=False)
            # Box-Cox only if strictly positive and non-degenerate
            can_boxcox = np.all(a > 0) and np.isfinite(a).sum() > 10

            try:
                # Fit objects on the *scoring* sample for stability
                s = _sample(a)
                qt_u.fit(s.reshape(-1,1))
                qt_n.fit(s.reshape(-1,1))
                pt_yj.fit(s.reshape(-1,1))
            except Exception:
                pass  # fall through; their callables may still raise and we guard later

            # 1) rank/uniform & gauss-rank
            out.append(("q_uniform", lambda v: qt_u.transform(v.reshape(-1,1)).ravel(),
                        f"QuantileTransformer(output_distribution='uniform', n_quantiles=100, random_state={rs}).fit_transform({col}.values.reshape(-1,1)).ravel()"))
            out.append(("q_normal",  lambda v: qt_n.transform(v.reshape(-1,1)).ravel(),
                        f"QuantileTransformer(output_distribution='normal', n_quantiles=100, random_state={rs}).fit_transform({col}.values.reshape(-1,1)).ravel()"))

            # 2) power / log family
            out.append(("yeo_johnson", lambda v: pt_yj.transform(v.reshape(-1,1)).ravel(),
                        f"PowerTransformer(method='yeo-johnson').fit_transform({col}.values.reshape(-1,1)).ravel()"))
            if can_boxcox:
                pt_bc = PowerTransformer(method="box-cox", standardize=False)
                try:
                    pt_bc.fit(_sample(a[a > 0]).reshape(-1,1))
                    out.append(("box_cox", lambda v: pt_bc.transform(v.reshape(-1,1)).ravel(),
                                f"PowerTransformer(method='box-cox').fit_transform({col}.values.reshape(-1,1)).ravel()"))
                    out.append(("log1p", lambda v: np.log1p(v), f"np.log1p({col})"))
                except Exception:
                    # if BC fails, still allow log1p if strictly positive
                    if np.all(a > 0):
                        out.append(("log1p", lambda v: np.log1p(v), f"np.log1p({col})"))

            # 3) robust monotone
            out.append(("asinh",       asinh,             f"np.arcsinh({col})"))
            if np.all(a >= 0):
                out.append(("sqrt",    np.sqrt,           f"np.sqrt({col})"))
            out.append(("cbrt",    np.cbrt,               f"np.cbrt({col})"))
            out.append(("signed_log1p", signed_log1p,     f"np.sign({col})*np.log1p(np.abs({col}))"))

            # 4) winsor / clip
            out.append(("clip_iqr3",  lambda v: np.clip(v, lo_iqr3, hi_iqr3),
                        f"np.clip({col}, {_fmt(lo_iqr3)}, {_fmt(hi_iqr3)})"))

            # 5) discretizations
            out.append(("qcut",       lambda v: pd.qcut(pd.Series(v), q=min(qbins, max(2, len(np.unique(v)))),
                                                        labels=False, duplicates='drop').to_numpy(),
                        f"pd.qcut({col}, q={qbins}, labels=False, duplicates='drop')"))

            # 6) scalings
            out.append(("zscore_mean_std", lambda v: (v - mean) / (std + 1e-12),
                        f"({col}-{_fmt(mean)})/({_fmt(std)}+1e-12)"))
            out.append(("zscore_med_mad",  lambda v: (v - med)  / (mad  + 1e-12),
                        f"({col}-{_fmt(med)})/({_fmt(mad)}+1e-12)"))

            # 7) inversion (dangerous; keep last/low priority; only if >0)
            if np.all(a > 0):
                out.append(("inverse", lambda v: 1.0/(v+1e-8), f"1/({col}+1e-8)"))

            # dedup by name_key
            seen = set(); uniq = []
            for t in out:
                if t[0] in seen: continue
                seen.add(t[0]); uniq.append(t)
            return uniq

        # scoring: combine distribution-fixing + optional HSIC-to-target
        def _score_transform(vec: np.ndarray, yvec: np.ndarray | None) -> Tuple[float, Dict[str, float]]:
            v = vec[np.isfinite(vec)]
            if v.size < 50:
                return -np.inf, {"jb_p": np.nan, "skew": np.nan, "kurt": np.nan, "hsic": np.nan}
            v = _sample(v)  # score on sample for speed
            # stats before/after are handled in caller; here we return absolute stats only
            jb = _jb_p(v)
            sk = float(sps.skew(v, bias=False)) if v.size > 2 else np.nan
            ku = float(sps.kurtosis(v, fisher=False, bias=False)) if v.size > 3 else np.nan
            score = 0.0
            # prefer high JB p, low |skew| and |kurt-3|
            if np.isfinite(jb): score += np.clip(jb, 0.0, 1.0) * 1.0
            if np.isfinite(sk): score += (1.0 / (1.0 + abs(sk))) * 0.5
            if np.isfinite(ku): score += (1.0 / (1.0 + abs(ku - 3.0))) * 0.5

            hs = np.nan
            if y_present and yvec is not None:
                # align to finite and sample together
                m = np.isfinite(vec) & np.isfinite(yvec)
                if m.sum() >= hsic_min_n:
                    xs = _sample(vec[m]); ys = _sample(yvec[m])
                    if xs.size >= hsic_min_n and ys.size >= hsic_min_n:
                        try:
                            hs = float(hsic_scorer.score(xs, ys))
                            if np.isfinite(hs):
                                score += hsic_weight * hs
                        except Exception:
                            hs = np.nan
            return score, {"jb_p": jb, "skew": sk, "kurt": ku, "hsic": (hs if y_present else np.nan)}

        suggestions: Dict[str, Any] = {}
        details: Dict[str, Any] = {}

        y_arr_full = y_series.to_numpy() if y_present else None

        for col in numeric_cols:
            s = pd.to_numeric(data[col], errors="coerce").dropna()
            if s.size < 10:
                continue
            a = s.to_numpy()
            n = a.size

            # baseline stats
            mean = float(np.mean(a)); std = float(np.std(a, ddof=1)) if n > 1 else 0.0
            med  = float(np.median(a)); mad = _mad(a, med)
            q1, q3 = np.percentile(a, [25, 75]); iqr = float(q3 - q1)
            skew0 = float(sps.skew(a, bias=False)); kurt0 = float(sps.kurtosis(a, fisher=False, bias=False))
            jb0   = _jb_p(_sample(a)); sw0 = _sw_p(_sample(a))

            # build and evaluate catalog
            cats = _catalog_for(col, a)
            scored: List[Tuple[str, float, Dict[str, float]]] = []

            # target vector aligned for this column (once)
            y_aligned = None
            if y_present:
                # align using index of non-na a
                y_aligned = y_arr_full[s.index]  # safe because s comes from data[col].dropna()

            for name_key, fn, as_str in cats:
                try:
                    vec = fn(a.copy())
                    # Some transforms can return pandas (qcut). Ensure ndarray float where possible.
                    if isinstance(vec, pd.Series): vec = vec.to_numpy()
                    vec = np.asarray(vec)
                    sc, meta = _score_transform(vec, y_aligned)
                    # bonus for strong monotone transforms when very skewed
                    if (abs(skew0) > 2.0 or kurt0 > 6.0) and any(k in name_key for k in ("yeo", "box", "log", "asinh", "q_normal")):
                        sc += 0.05
                    scored.append((as_str, sc, meta))
                except Exception:
                    continue

            if not scored:
                continue

            # sort by score desc
            scored.sort(key=lambda t: t[1], reverse=True)
            rec = [s for s, _, _ in scored[:max_k]]
            suggestions[col] = [s for s, _, _ in scored[:short_list_k]]

            details[col] = {
                "stats": {
                    "count": int(n),
                    "mean": mean, "std": std, "median": med, "mad": mad,
                    "q1": float(q1), "q3": float(q3), "iqr": iqr,
                    "skewness": skew0, "kurtosis": kurt0,
                    "normality_jb_p": float(jb0) if np.isfinite(jb0) else np.nan,
                    "normality_sw_p": float(sw0) if np.isfinite(sw0) else np.nan,
                },
                "scored": [
                    {"expr": s, "score": float(sc), **meta} for (s, sc, meta) in scored[:max_k]
                ],
                "recommended": rec,
            }

        return suggestions, details
        
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

        from ..aux.hsic import HSIC

        rs        = getattr(cfg, "random_state", 42)
        n_splits  = getattr(cfg, "rank_cv_splits", 5)
        n_repeats = getattr(cfg, "rank_perm_repeats", 3)
        max_n     = getattr(cfg, "rank_sample_size", 20000)

        # weights (we’ll add HSIC and renormalize later)
        w_imp  = getattr(cfg, "w_impurity", 0.30)
        w_perm = getattr(cfg, "w_permutation", 0.35)
        w_mi   = getattr(cfg, "w_mutual_info", 0.10)
        w_lin  = getattr(cfg, "w_linear", 0.10)
        w_hsic = getattr(cfg, "w_hsic",  0.15)  # NEW

        # -------------------- Prep --------------------
        X = data[numeric_cols].copy().apply(pd.to_numeric, errors="coerce")
        # Median impute per-column (works fine for trees; OK for Ridge if data roughly centered)
        X = X.fillna(X.median(numeric_only=True))

        target_name = getattr(cfg, "target_col", None)
        y = None
        if target_name and target_name in data.columns:
            y = pd.to_numeric(data[target_name], errors="coerce")

        # Optional subsample for speed (deterministic)
        if len(X) > max_n:
            idx = np.random.RandomState(rs).choice(len(X), size=max_n, replace=False)
            X = X.iloc[idx]
            if y is not None:
                y = y.iloc[idx]

        # Drop constant columns (avoid MI/perm issues)
        nunique = X.nunique(dropna=False)
        keep = nunique[nunique > 1].index.tolist()
        X = X[keep]
        numeric_cols = keep
        if len(numeric_cols) == 0:
            return {}

        # Fast near-duplicate pruning (|rho_spearman| > 0.995 keep first)
        with np.errstate(all="ignore"):
            cmat = X.corr(method="spearman")
        to_drop = set()
        cols_order = list(cmat.columns)
        for i, c in enumerate(cols_order):
            if c in to_drop:
                continue
            corr_row = cmat.loc[c].iloc[i + 1 :]
            dups = corr_row.index[np.abs(corr_row.values) > 0.995]
            to_drop.update(dups)
        if to_drop:
            X = X.drop(columns=list(to_drop))
            numeric_cols = [c for c in numeric_cols if c not in to_drop]
            if len(numeric_cols) == 0:
                return {}

        # Prepare y after pruning (and guard degenerate cases)
        y_present = y is not None
        if y_present:
            y = y.loc[X.index]
            if y.notna().sum() < 5 or y.nunique() <= 1:
                y_present = False
            else:
                y = y.fillna(y.median())
                # Standardize y to stabilize HSIC / permutation
                y = (y - y.mean()) / (y.std() + 1e-12)

        # -------------------- Model zoo --------------------
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

        # -------------------- Containers --------------------
        impurity: dict[str, list[float]] = {c: [] for c in numeric_cols}
        permute:  dict[str, list[float]] = {c: [] for c in numeric_cols}

        # -------------------- CV loop: impurity + permutation on held-out --------------------
        if y_present:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
            for train_idx, test_idx in kf.split(X):
                Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
                ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

                for _, m in model_zoo:
                    try:
                        m.fit(Xtr, ytr)
                        fi = getattr(m, "feature_importances_", None)
                        if fi is not None and np.all(np.isfinite(fi)) and len(fi) == X.shape[1]:
                            for c, v in zip(X.columns, fi):
                                impurity[c].append(float(max(v, 0.0)))
                        # permutation importance on validation fold
                        try:
                            pi = permutation_importance(m, Xte, yte, n_repeats=n_repeats, random_state=rs, n_jobs=-1)
                            if len(pi.importances_mean) == X.shape[1]:
                                for c, v in zip(X.columns, pi.importances_mean):
                                    permute[c].append(float(max(v, 0.0)))
                        except Exception:
                            pass
                    except Exception:
                        continue
        else:
            # No target: these channels will be zeroed
            pass

        # -------------------- Mutual information --------------------
        if y_present:
            try:
                mi_vals = mutual_info_regression(X, y, random_state=rs)
                mi_map = {c: float(max(v, 0.0)) for c, v in zip(X.columns, mi_vals)}
            except Exception:
                mi_map = {c: 0.0 for c in X.columns}
        else:
            mi_map = {c: 0.0 for c in X.columns}

        # -------------------- Linear (Ridge) signal --------------------
        lin_map = {c: 0.0 for c in X.columns}
        if y_present:
            try:
                ridge = RidgeCV(alphas=(1e-3, 1e-2, 1e-1, 1, 10))
                ridge.fit(X, y)
                for c, v in zip(X.columns, np.abs(ridge.coef_).astype(float)):
                    lin_map[c] = float(v)
            except Exception:
                pass

        # -------------------- HSIC-to-target (NEW) --------------------
        # robust against huge n: optional subsample
        hsic_map = {c: 0.0 for c in X.columns}
        if y_present:
            hsic_sample_n   = getattr(cfg, "rank_hsic_sample_size", 8000)
            hsic_estimator  = getattr(cfg, "rank_hsic_estimator", "biased")   # or "unbiased"
            hsic_normalize  = getattr(cfg, "rank_hsic_normalize", True)
            hsic_kernel_x   = getattr(cfg, "rank_hsic_kernel_x", "rbf")
            hsic_kernel_y   = getattr(cfg, "rank_hsic_kernel_y", "rbf")
            hsic_min_n      = getattr(cfg, "rank_hsic_min_samples", 50)
            use_numba       = getattr(cfg, "rank_hsic_use_numba", True)

            y_arr = y.to_numpy()
            idx = np.arange(len(X))
            if len(idx) > hsic_sample_n:
                rng = np.random.default_rng(rs)
                idx = rng.choice(idx, size=hsic_sample_n, replace=False)
            y_sub = y_arr[idx]

            scorer = HSIC(kernel_x=hsic_kernel_x, kernel_y=hsic_kernel_y,
                        estimator=hsic_estimator, normalize=hsic_normalize,
                        use_numba=use_numba, random_state=rs)
            for c in X.columns:
                x_sub = X[c].to_numpy()[idx]
                # guard tiny or constant slices
                if np.isfinite(x_sub).sum() < hsic_min_n:
                    continue
                if np.std(x_sub) < 1e-12:
                    continue
                try:
                    hsic_val = scorer.score(x_sub, y_sub)
                    if np.isfinite(hsic_val):
                        hsic_map[c] = float(max(0.0, hsic_val))
                except Exception:
                    continue

        # -------------------- Aggregate & robust-normalize channels --------------------
        def _avg(d: dict[str, list[float]]) -> dict[str, float]:
            return {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in d.items()}

        imp_avg = _avg(impurity)
        perm_avg = _avg(permute)

        def _robust_norm(m: dict[str, float]) -> dict[str, float]:
            """Normalize to [0,1] via (x - q05)/(q95 - q05); fallback to max-norm."""
            if not m:
                return {}
            arr = np.array(list(m.values()), dtype=float)
            q05, q95 = np.percentile(arr, [5, 95])
            scale = max(q95 - q05, 1e-12)
            normed = {k: float(np.clip((v - q05) / scale, 0.0, 1.0)) for k, v in m.items()}
            if sum(normed.values()) == 0.0:
                mx = arr.max()
                if mx > 0:
                    normed = {k: float(v / mx) for k, v in m.items()}
            return normed

        imp_n  = _robust_norm(imp_avg)
        perm_n = _robust_norm(perm_avg)
        mi_n   = _robust_norm(mi_map)
        lin_n  = _robust_norm(lin_map)
        hsic_n = _robust_norm(hsic_map)

        # Blend (then renormalize blended scores to [0,1] for readability)
        blended_raw = {
            c: w_imp  * imp_n.get(c, 0.0)
            + w_perm * perm_n.get(c, 0.0)
            + w_mi   * mi_n.get(c, 0.0)
            + w_lin  * lin_n.get(c, 0.0)
            + w_hsic * hsic_n.get(c, 0.0)
            for c in X.columns
        }
        vals = np.array(list(blended_raw.values()), dtype=float)
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if np.isfinite(vmax - vmin) and (vmax - vmin) > 1e-12:
            blended = {k: float((v - vmin) / (vmax - vmin)) for k, v in blended_raw.items()}
        else:
            blended = blended_raw

        ranked = dict(sorted(blended.items(), key=lambda kv: kv[1], reverse=True))

        # -------------------- Optional SHAP side-info (does NOT affect ranking) --------------------
        if getattr(cfg, "enable_shap", False) and y_present and len(X) < getattr(cfg, "shap_max_n", 12000):
            try:
                import shap
                try:
                    from lightgbm import LGBMRegressor
                    model = LGBMRegressor(n_estimators=300, random_state=rs).fit(X, y)
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    gbr = GradientBoostingRegressor(n_estimators=200, random_state=rs).fit(X, y)
                    explainer = shap.TreeExplainer(gbr)
                rng = np.random.default_rng(rs)
                m = min(1024, len(X))
                idx = rng.choice(len(X), size=m, replace=False)
                sv = explainer.shap_values(X.iloc[idx])
                shap_imp = np.mean(np.abs(sv), axis=0).astype(float)
                ranked["__shap_importance__"] = dict(zip(X.columns, map(float, shap_imp)))
            except Exception:
                pass

        return ranked
    
    def _suggest_interactions(
        self,
        data: pd.DataFrame,
        numeric_cols: List[str],
        detailed: Dict[str, Any],
        cfg: "_Cfg",
    ) -> List[str]:
        """
        SOTA interaction proposer:
        - HSIC-based pair screening (nonlinear, kernelized)
        - (Conditional) HSIC-based target ranking of candidate expressions
        - Target-gain filter
        - Rich expression catalog (basic, poly, trig, statistical, fourier, target-proxies)
        """
        from functools import lru_cache
        from itertools import combinations

        import numpy as np
        import pandas as pd

        from ..aux.hsic import HSIC

        rng = np.random.default_rng(getattr(cfg, "random_state", 42))

        # ---------- Config ----------
        top_k      = getattr(cfg, "top_rank_for_interactions", 12)
        corr_gate  = getattr(cfg, "corr_gate", 0.95)
        hsic_gate  = getattr(cfg, "hsic_gate", 0.10)     # minimal HSIC to keep a pair
        max_pairs  = getattr(cfg, "max_interactions", 64)
        sample_n   = getattr(cfg, "interaction_sample_size", 6000)

        allow_basic        = getattr(cfg, "allow_basic_ops", True)
        allow_poly         = getattr(cfg, "allow_polynomial", True)
        allow_trig         = getattr(cfg, "allow_trigonometric", True)
        allow_statistical  = getattr(cfg, "allow_statistical_moments", True)
        allow_fourier      = getattr(cfg, "allow_fourier_features", True)
        allow_target_proxies = getattr(cfg, "allow_target_encoding_proxies", True)

        enable_hl       = getattr(cfg, "enable_high_level_interactions", True)
        max_high_level  = getattr(cfg, "max_high_level", 16)
        hl_seed_pairs_top = getattr(cfg, "hl_seed_pairs_top", 6)
        hl_gain_gate    = getattr(cfg, "hl_gain_gate", 3e-4)

        # HSIC options (plumbed from cfg)
        hsic_kernel_x_pairs = getattr(cfg, "hsic_kernel_x_pairs", "rbf")
        hsic_kernel_y_pairs = getattr(cfg, "hsic_kernel_y_pairs", "rbf")
        hsic_kernel_x_tgt   = getattr(cfg, "hsic_kernel_x_tgt",   "rbf")
        hsic_kernel_y_tgt   = getattr(cfg, "hsic_kernel_y_tgt",   "rbf")
        hsic_estimator      = getattr(cfg, "hsic_estimator", "biased")   # "biased" | "unbiased" | "block" | "rff"...
        hsic_normalize      = getattr(cfg, "hsic_normalize", True)
        hsic_use_numba      = getattr(cfg, "hsic_use_numba", True)
        hsic_min_samples    = getattr(cfg, "hsic_min_samples", 50)

        # --- NEW: conditional HSIC controls ---
        cond_on      = getattr(cfg, "hsic_condition_on", None)    # None | "others" | List[str]
        cond_lam     = getattr(cfg, "hsic_cond_lambda", 1e-3)
        cond_min_n   = getattr(cfg, "hsic_cond_min_samples", 120)
        cond_std     = getattr(cfg, "hsic_cond_standardize", True)

        # ---------- Target ----------
        target_col = getattr(cfg, "target_col", None)
        y = None
        if target_col and target_col in data.columns:
            y = pd.to_numeric(data[target_col], errors="coerce")

        # ---------- Candidate features ----------
        ranked = list(detailed.get("feature_ranking", {}).keys())[:top_k]
        if not ranked:
            ranked = numeric_cols[:top_k]
        cols = [c for c in ranked if c in data.columns]
        if len(cols) < 2:
            return []

        work = data[cols].copy()
        if sample_n and len(work) > sample_n:
            work = work.sample(sample_n, random_state=getattr(cfg, "random_state", 42))
            if y is not None:
                y = y.loc[work.index]

        work = work.apply(pd.to_numeric, errors="coerce")
        y = y.astype(float) if y is not None else None

        # ---------- Correlation blocking ----------
        min_periods = max(20, len(work) // 100)
        with np.errstate(all="ignore"):
            corr_matrix = np.abs(work.corr(method="pearson", min_periods=min_periods).values)
        np.fill_diagonal(corr_matrix, 0)
        blocked_pairs = set()
        for i, j in zip(*np.where(np.triu(corr_matrix >= corr_gate, 1))):
            blocked_pairs.add((cols[i], cols[j]))

        # ---------- Preprocessing ----------
        col_data = {}
        for c in cols:
            s = work[c].astype(float)
            mask = s.notna().to_numpy()
            arr = s.to_numpy()
            if mask.sum() >= 2:
                mu, std = np.mean(arr[mask]), np.std(arr[mask])
                z_score = (arr - mu) / (std + 1e-12)
            else:
                z_score = np.zeros_like(arr)
            col_data[c] = {"array": arr, "mask": mask, "z_score": z_score}

        y_data = None
        if y is not None:
            y_mask = y.notna().to_numpy()
            y_arr = y.to_numpy()
            if y_mask.sum() >= 2:
                y_mu, y_std = np.mean(y_arr[y_mask]), np.std(y_arr[y_mask])
                y_standardized = (y_arr - y_mu) / (y_std + 1e-12)
            else:
                y_standardized = np.zeros_like(y_arr)
            y_data = {"array": y_arr, "mask": y_mask, "standardized": y_standardized}

        # ---------- HSIC scorers ----------
        hsic_pair = HSIC(kernel_x=hsic_kernel_x_pairs, kernel_y=hsic_kernel_y_pairs,
                        estimator=hsic_estimator, normalize=hsic_normalize,
                        use_numba=hsic_use_numba, random_state=getattr(cfg, "random_state", 42))
        hsic_tgt  = HSIC(kernel_x=hsic_kernel_x_tgt, kernel_y=hsic_kernel_y_tgt,
                        estimator=hsic_estimator, normalize=hsic_normalize,
                        use_numba=hsic_use_numba, random_state=getattr(cfg, "random_state", 42))

        # ---------- Core helpers ----------
        @lru_cache(maxsize=128)
        def _get_aligned_pair(a: str, b: str):
            mask = col_data[a]["mask"] & col_data[b]["mask"]
            if mask.sum() < hsic_min_samples:
                return None, None, None
            return col_data[a]["array"][mask], col_data[b]["array"][mask], mask

        def _target_gain(a: str, b: str) -> float:
            # keep your lightweight linear gain test
            if y_data is None: return 0.0
            aa, bb, mask = _get_aligned_pair(a, b)
            if mask is None:
                return 0.0
            mask = mask & y_data["mask"]
            if mask.sum() < max(60, hsic_min_samples):
                return 0.0
            Xa = col_data[a]["z_score"][mask]
            Xb = col_data[b]["z_score"][mask]
            yv = y_data["standardized"][mask]
            lam = 1e-3
            try:
                Xbse = np.column_stack([Xa, Xb])
                beta_b = np.linalg.solve(Xbse.T @ Xbse + lam*np.eye(2), Xbse.T @ yv)
                r2_b = np.corrcoef(Xbse @ beta_b, yv)[0,1]**2
                Xful = np.column_stack([Xa, Xb, Xa*Xb])
                beta_f = np.linalg.solve(Xful.T @ Xful + lam*np.eye(3), Xful.T @ yv)
                r2_f = np.corrcoef(Xful @ beta_f, yv)[0,1]**2
                return max(0.0, r2_f - r2_b)
            except Exception:
                return 0.0

        # --- NEW: helpers for conditional HSIC ---
        def _finite_mask_cols(arr2d: np.ndarray) -> np.ndarray:
            # arr2d: (n,d)
            return np.isfinite(arr2d).all(axis=1)

        def _build_Z(mask_base: np.ndarray, exclude: tuple) -> tuple:
            """
            Returns (Z_masked, final_mask) or (None, None).
            - cond_on == "others": use all 'cols' except 'exclude'
            - cond_on == list[str]: intersect with available df columns, excluding 'exclude'
            - applies standardization if cond_std
            """
            if not cond_on:
                return None, None
            if cond_on == "others":
                zcols = [c for c in cols if c not in exclude]
            else:
                zcols = [c for c in cond_on if c in data.columns and c not in exclude]
            if not zcols:
                return None, None

            Z_full = data[zcols].to_numpy()
            mZ = _finite_mask_cols(Z_full)
            m = (mask_base if mask_base is not None else np.ones(len(Z_full), bool)) & mZ
            if y_data is not None:
                m = m & y_data["mask"]  # align with target availability
            if m.sum() < cond_min_n:
                return None, None

            Zm = Z_full[m]
            if cond_std:
                mu = np.nanmean(Zm, axis=0, keepdims=True)
                sd = np.nanstd(Zm, axis=0, keepdims=True)
                Zm = (Zm - mu) / (sd + 1e-12)
            return Zm, m

        # ---------- Expression generator ----------
        def _generate_expressions(a: str, b: str) -> List[str]:
            exprs = []
            if allow_basic:
                exprs += [f"{a}*{b}", f"{a}/({b}+1e-8)", f"{b}/({a}+1e-8)",
                        f"abs({a}-{b})", f"({a}+{b})/2", f"np.sqrt({a}**2+{b}**2)",
                        f"np.minimum({a},{b})", f"np.maximum({a},{b})"]
            if allow_poly:
                exprs += [f"({a}+{b})**2", f"({a}-{b})**2", f"{a}**2+{b}**2",
                        f"{a}**2*{b}", f"{a}*{b}**2", f"{a}**3+{b}**3", f"({a}*{b})**2"]
            if allow_trig:
                exprs += [f"np.sin({a})*np.cos({b})", f"np.sin({a}+{b})",
                        f"np.cos({a}-{b})", f"np.arctan2({b},{a})",
                        f"np.sin({a}**2+{b}**2)"]
            if allow_statistical:
                exprs += [f"({a}*{b})/np.sqrt(({a}**2+1)*({b}**2+1))",
                        f"np.sign({a})*np.sign({b})*np.sqrt(abs({a}*{b}))",
                        f"({a}**3+{b}**3)/({a}**2+{b}**2+1e-8)",
                        f"({a}**4+{b}**4)/({a}**2+{b}**2+1e-8)**2",
                        f"np.exp(-({a}-{b})**2/(np.var([{a},{b}])+1e-8))"]
            if allow_fourier:
                exprs += [f"np.cos(2*np.pi*({a}+{b}))", f"np.cos(2*np.pi*({a}-{b}))",
                        f"np.cos(2*np.pi*{a})*np.cos(2*np.pi*{b})",
                        f"np.sin(2*np.pi*{a})*np.sin(2*np.pi*{b})",
                        f"np.cos(np.pi*{a})*np.sin(np.pi*{b})"]
            if allow_target_proxies and y_data is not None:
                exprs += [f"({a}*{b})*np.sign({a}+{b})",
                        f"np.where({a}*{b}>0, np.log1p(abs({a}*{b})), -np.log1p(abs({a}*{b})))",
                        f"({a}>np.median({a}))*({b}>np.median({b}))",
                        f"np.tanh({a})*np.tanh({b})"]
            return exprs

        # ---------- Candidate scoring ----------
        expr_candidates = {}
        pair_scores = []

        for a, b in combinations(cols, 2):
            if (a, b) in blocked_pairs:
                continue
            aa, bb, mask = _get_aligned_pair(a, b)
            if aa is None:
                continue

            # HSIC pair gate (nonlinear, kernel-based)
            try:
                hsic_pair_val = hsic_pair.score(aa, bb)
            except Exception:
                hsic_pair_val = 0.0
            if not np.isfinite(hsic_pair_val) or hsic_pair_val < hsic_gate:
                continue

            gain_val = _target_gain(a, b)
            score = hsic_pair_val + 10.0 * gain_val
            if score <= 0:
                continue

            exprs = _generate_expressions(a, b)
            for expr in exprs:
                expr_candidates[expr] = (a, b, mask)  # store names+mask to build values later
            pair_scores.append((a, b, score))

        if not expr_candidates:
            return []

        # ---------- Rank expressions by HSIC(expression, target) ----------
        expr_scores = []
        expr_list = list(expr_candidates.keys())

        if y_data is not None:
            for expr in expr_list:
                a, b, mask = expr_candidates[expr]
                if mask is None:
                    continue
                mask_t = mask & y_data["mask"]
                if mask_t.sum() < max(60, hsic_min_samples):
                    continue

                # locals for eval: a_vec, b_vec, np
                a_vec = data[a].to_numpy()[mask_t]
                b_vec = data[b].to_numpy()[mask_t]
                try:
                    # evaluate candidate vector
                    vec = eval(expr, {"np": np}, {a: a_vec, b: b_vec})
                    vec = np.asarray(vec, dtype=float)
                    yv  = y_data["array"][mask_t].astype(float)
                    if vec.size != yv.size or vec.size < hsic_min_samples:
                        continue

                    # ---- NEW: conditional HSIC if requested ----
                    if cond_on:
                        Zm, m_final = _build_Z(mask_t, exclude=(a, b))
                        if Zm is not None:
                            xf = vec[m_final]
                            yf = yv[m_final]
                            if xf.size >= max(hsic_min_samples, cond_min_n):
                                try:
                                    s = hsic_tgt.conditional_score(xf, yf, Zm, lam=cond_lam)
                                except Exception:
                                    s = hsic_tgt.score(vec, yv)  # safe fallback
                            else:
                                s = hsic_tgt.score(vec, yv)
                        else:
                            s = hsic_tgt.score(vec, yv)
                    else:
                        s = hsic_tgt.score(vec, yv)

                    if np.isfinite(s):
                        expr_scores.append((expr, float(s)))
                except Exception:
                    continue

            expr_scores.sort(key=lambda t: t[1], reverse=True)
            selected = [e for e, _ in expr_scores[:max_pairs]]
        else:
            # fallback: keep top by pair score if no target
            selected = [e for e in expr_list][:max_pairs]

        final_expressions = selected[:max_pairs]

        # ---------- High-level interactions (unchanged) ----------
        if enable_hl and len(cols) >= 3 and y_data is not None and len(final_expressions) < max_pairs:
            pair_scores.sort(key=lambda x: x[2], reverse=True)
            triplet_count = 0
            for a, b, _ in pair_scores[:hl_seed_pairs_top]:
                for c in cols:
                    if c in [a, b]:
                        continue
                    mask = col_data[a]['mask'] & col_data[b]['mask'] & col_data[c]['mask'] & y_data['mask']
                    if mask.sum() < 100:
                        continue
                    x1, x2, x3 = (col_data[a]['z_score'][mask],
                                col_data[b]['z_score'][mask],
                                col_data[c]['z_score'][mask])
                    yv = y_data['standardized'][mask]
                    base_r2 = max(
                        np.corrcoef(x1+x2, yv)[0,1]**2,
                        np.corrcoef(x1+x3, yv)[0,1]**2,
                        np.corrcoef(x2+x3, yv)[0,1]**2
                    )
                    candidates = {
                        f"{a}*{b}*{c}": x1 * x2 * x3,
                        f"({a}+{b}+{c})**2": (x1+x2+x3)**2,
                        f"np.tanh({a}+{b}+{c})": np.tanh(x1+x2+x3),
                        f"np.median([{a},{b},{c}], axis=0)": np.median(np.vstack([x1,x2,x3]), axis=0),
                    }
                    for expr, vec in candidates.items():
                        try:
                            r2 = np.corrcoef(vec, yv)[0,1]**2
                        except Exception:
                            r2 = 0
                        if r2 - base_r2 >= hl_gain_gate:
                            final_expressions.append(expr)
                            triplet_count += 1
                            if triplet_count >= max_high_level:
                                break
                if triplet_count >= max_high_level:
                    break

        return final_expressions[:max_pairs]

    # -------------------- (4) Cat encodings --------------------
    def _suggest_encodings(
        self, data: pd.DataFrame, categorical_cols: List[str], cfg: "_Cfg"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Heuristic, target-aware encoder suggestions.
        Returns:
        suggestions[col]: short list (<=3)
        details[col]: stats + full ranked strategy list + reason
        """
        suggestions: Dict[str, Any] = {}
        details: Dict[str, Any] = {}

        # ---- config / task ----
        target_col: Optional[str] = getattr(cfg, "target_col", None)
        task: str = getattr(cfg, "task", "auto")  # "regression" | "binary" | "multiclass" | "auto"
        time_col: Optional[str] = getattr(cfg, "time_col", None)     # for leakage-safe encoders
        group_col: Optional[str] = getattr(cfg, "group_col", None)   # group-aware CV, leakage guard
        rs = getattr(cfg, "random_state", 42)

        y = None
        y_kind = None
        if target_col and target_col in data.columns:
            y = data[target_col]
            # infer task if needed
            if task == "auto":
                if pd.api.types.is_numeric_dtype(y):
                    # decide binary vs regression by #unique AFTER dropping nans
                    nun = y.dropna().nunique()
                    if nun == 2:
                        y_kind = "binary"
                    elif nun <= 15 and all(float(v).is_integer() for v in y.dropna().unique()):
                        y_kind = "multiclass"
                    else:
                        y_kind = "regression"
                else:
                    nun = y.dropna().nunique()
                    y_kind = "binary" if nun == 2 else "multiclass"
            else:
                y_kind = task
        else:
            y_kind = None

        def _entropy(s: pd.Series) -> float:
            vc = s.value_counts(dropna=False)
            p = (vc / max(vc.sum(), 1)).to_numpy(dtype=float)
            p = p[p > 0]
            return float(-(p * np.log(p)).sum()) if p.size else 0.0

        def _dominance(s: pd.Series) -> float:
            vc = s.value_counts(dropna=True)
            return float(vc.iloc[0] / max(vc.sum(), 1)) if len(vc) else 0.0

        def _card_bucket(k: int) -> str:
            if k <= 5: return "very_low"
            if k <= 20: return "low"
            if k <= 50: return "medium"
            if k <= 200: return "high"
            return "extreme"

        for col in categorical_cols:
            s = data[col]
            nunique = int(s.nunique(dropna=True))
            null_pct = float(s.isna().mean() * 100.0)
            ent = _entropy(s)
            dom = _dominance(s)
            bucket = _card_bucket(nunique)

            enc: List[str] = []
            notes: List[str] = []

            # Always: suggest handling for missing/unseen
            if null_pct > 0:
                notes.append(f"Missing {null_pct:.1f}% → treat NaN as its own level + MissingIndicator.")
            notes.append("Set handle_unknown='infrequent_if_exist' / 'ignore' to avoid errors.")
            if group_col:
                notes.append("Use GroupKFold/LeaveOneGroupOut for supervised encoders (group leakage).")
            if time_col:
                notes.append("Use time-based CV (purged CV) for supervised encoders (temporal leakage).")

            # Baseline unsupervised picks by cardinality
            if bucket == "very_low":
                enc += ["OneHot(drop_first=True)", "OrdinalEncoder"]
            elif bucket == "low":
                enc += ["OneHot(drop_first=False)", "OrdinalEncoder", "BinaryEncoder"]
            elif bucket == "medium":
                enc += ["OneHot(drop_first=False)", "BinaryEncoder", "BaseNEncoder(base=4)"]
            elif bucket == "high":
                enc += ["FrequencyEncoder", "HashingEncoder(n_features≈min(2*k, 512))", "BinaryEncoder"]
            else:  # extreme
                enc += ["HashingEncoder(n_features≈max(1024, 2*k))", "FrequencyEncoder", "LeaveOneOutEncoder (CV)"]

            # Adjust for imbalance / dominance
            if dom >= 0.8:
                enc.insert(0, "RareCategoryEncoder(threshold≈1%)")
                notes.append("Strongly imbalanced categories → collapse rares first.")

            # Target-aware augmentations
            if y_kind is not None:
                if y_kind == "regression":
                    # CV Target encoders good; CatBoost handles high-cardinality well
                    if bucket in {"low", "medium"}:
                        enc = ["TargetEncoder (KFold CV, smoothing)"] + enc
                    elif bucket in {"high", "extreme"}:
                        enc = ["CatBoostEncoder (KFold CV, ordered)", "TargetEncoder (KFold CV, smoothing)"] + enc
                elif y_kind == "binary":
                    # WOE is principled for binary classification
                    if nunique <= 50:
                        enc = ["WOEEncoder (KFold CV, regularized)", "TargetEncoder (KFold CV, smoothing)"] + enc
                    else:
                        enc = ["CatBoostEncoder (KFold CV, ordered)", "TargetEncoder (KFold CV, smoothing)"] + enc
                else:  # multiclass
                    if bucket in {"low", "medium"}:
                        enc = ["TargetEncoder (KFold CV, per-class)", "OneHot(drop_first=False)"] + enc
                    else:
                        enc = ["CatBoostEncoder (KFold CV, ordered)"] + enc

            # Rare category handling for higher cardinalities
            if nunique > 20 and "RareCategoryEncoder(threshold≈1%)" not in enc:
                enc.append("RareCategoryEncoder(threshold≈1%)")

            # Uniqueness-heavy columns (IDs): avoid target encoders unless you have grouping/time CV
            if nunique > 0.9 * len(s):
                notes.append("Looks like an ID-like field → avoid supervised encoders; prefer hashing/frequency.")

            # Rank lightly by “fit”
            rank_keys = []
            for e in enc:
                score = 0
                # Prefer CV target encoders when supervised
                if "TargetEncoder" in e or "CatBoostEncoder" in e or "WOEEncoder" in e or "LeaveOneOut" in e:
                    score += 2 if y_kind else -1
                # Prefer hashing/frequency for high/extreme
                if bucket in {"high", "extreme"} and ("Hashing" in e or "Frequency" in e):
                    score += 2
                # Prefer OHE for very_low/low
                if bucket in {"very_low", "low"} and "OneHot" in e:
                    score += 2
                # Penalize OneHot in high-card
                if bucket in {"high", "extreme"} and "OneHot" in e:
                    score -= 3
                # Bonus for RareCategory when imbalance
                if dom >= 0.8 and "RareCategoryEncoder" in e:
                    score += 2
                # WOE only for binary
                if "WOE" in e and y_kind != "binary":
                    score -= 2
                rank_keys.append((e, score))

            rank_keys.sort(key=lambda t: t[1], reverse=True)
            ranked = [e for e, _ in rank_keys]

            suggestions[col] = ranked[:3]
            details[col] = {
                "cardinality": nunique,
                "null_percentage": float(null_pct),
                "dominant_category_pct": float(dom * 100.0),
                "entropy": float(ent),
                "strategies": ranked,
                "task": y_kind or "unsupervised",
                "notes": notes,
                "leakage_advice": (
                    "Use KFold(GroupKFold/TimeSeriesSplit) for supervised encoders; fit only on training folds."
                    if y_kind else "Unsupervised encoders only (no target)."
                ),
                "params_hints": {
                    "TargetEncoder": {"cv": "KFold/GroupKFold/TimeSeriesSplit", "smoothing": "1–50", "noise": "0.01–0.1"},
                    "CatBoostEncoder": {"cv": "KFold, ordered", "prior": "mean", "noise": "ordered"},
                    "WOEEncoder": {"binary_only": True, "min_samples": "≥50 per bin", "regularization": "strong"},
                    "HashingEncoder": {"n_features": "choose to keep collision rate <5%"},
                    "RareCategoryEncoder": {"threshold": "0.5%–2%", "merge": "to '__rare__'"},
                    "OneHot": {"drop_first": "True for linear models; False for trees"},
                    "OrdinalEncoder": {"handle_unknown": "use -1 or 'infrequent' bin"},
                    "FrequencyEncoder": {"mapping": "count or normalized frequency"},
                },
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
