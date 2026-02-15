from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from foretools.aux.adaptive_mi import AdaptiveMI

from .base import BaseFeatureTransformer


# Optional parallelism
try:
    from joblib import Parallel, delayed

    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False


class InteractionTransformer(BaseFeatureTransformer):
    """
    Turbo interaction & polynomial generator (improved, AdaptiveMI-compatible):
      - smarter pair pre-screen: variance × (1 - |corr|) heuristic
      - commutative op canonicalization (no duplicate A∘B vs B∘A)
      - light winsorization + robust _clean guardrails
      - cached arrays + train means (no leakage)
      - parallel scoring (joblib) with safe fallback
      - recipe-based transform (compute only selected)
      - optional redundancy pruning (rank-Pearson)
    """

    def __init__(self, config: Any):
        super().__init__(config)

        # discovered during fit
        self.numerical_cols_: List[str] = []
        self.col_means_: Dict[str, float] = {}

        # recipes
        self.selected_interactions_: List[Tuple[str, str, str]] = []  # (col1, op, col2)
        self.selected_polynomials_: List[Tuple[str, Union[float, str]]] = (
            []
        )  # (col, power)
        self.feature_scores_: Dict[str, float] = {}

        # speed/quality knobs
        self.max_interactions = getattr(config, "max_interactions", 100)
        self.max_polynomials = getattr(config, "max_polynomials", 50)
        self.max_pairs = getattr(config, "max_pairs_screen", 800)
        self.prescreen_topk = getattr(config, "interaction_prescreen_topk", 32)
        self.min_variance = getattr(config, "min_variance_threshold", 1e-6)
        self.redundancy_corr = getattr(config, "interaction_redundancy_corr", 0.985)
        self.enable_redundancy_prune = getattr(
            config, "interaction_prune_redundancy", False
        )
        self.enable_stability_selection = getattr(
            config, "interaction_stability_selection", True
        )
        self.stability_min_freq = float(
            getattr(config, "interaction_stability_min_freq", 0.5)
        )

        self.random_state = getattr(config, "random_state", 42)
        self.eps = 1e-8

        # Turbo controls
        self.fast_mode = getattr(config, "interaction_fast_mode", True)
        self.row_subsample = getattr(config, "interaction_row_subsample", 5000)
        self.n_jobs = getattr(config, "interaction_n_jobs", -1)
        self.prescreen_use_spearman = getattr(
            config, "interaction_prescreen_spearman", False
        )

        # operations (meta uses train means at transform)
        self.operations = {
            "sum": (
                getattr(config, "include_sum", True),
                lambda a, b, meta=None: a + b,
            ),
            "diff": (
                getattr(config, "include_diff", True),
                lambda a, b, meta=None: a - b,
            ),
            "prod": (
                getattr(config, "include_prod", True),
                lambda a, b, meta=None: a * b,
            ),
            "ratio": (
                getattr(config, "include_ratio", True),
                lambda a, b, meta=None: np.divide(
                    a,
                    b,
                    out=np.full_like(a, np.nan),
                    where=(np.abs(b) > self.eps) & np.isfinite(b),
                ),
            ),
            "norm_ratio": (
                getattr(config, "include_norm_ratio", True),
                lambda a, b, meta=None: (a - b)
                / (np.abs(a) + np.abs(b) + self.eps),
            ),
            "min": (getattr(config, "include_minmax", True), np.minimum),
            "max": (getattr(config, "include_minmax", True), np.maximum),
            "zdiff": (
                getattr(config, "include_zdiff", True),
                lambda a, b, meta=None: (
                    a - (meta["a_mean"] if meta else np.nanmean(a))
                )
                - (b - (meta["b_mean"] if meta else np.nanmean(b))),
            ),
            "log_ratio": (
                getattr(config, "include_logratio", True),
                lambda a, b, meta=None: np.log1p(np.abs(a) + self.eps)
                - np.log1p(np.abs(b) + self.eps),
            ),
            "root_prod": (
                getattr(config, "include_rootprod", True),
                lambda a, b, meta=None: np.sign(a * b) * np.sqrt(np.abs(a * b)),
            ),
        }
        # mark commutative ops to canonicalize (avoid A∘B and B∘A duplicates)
        self._commutative_ops = {"sum", "prod", "min", "max", "root_prod"}

        # polynomials
        self.powers = {
            "squared": (getattr(config, "include_square", True), 2.0),
            "sqrt": (getattr(config, "include_sqrt", True), 0.5),
            "cubed": (getattr(config, "include_cube", False), 3.0),
            "reciprocal": (getattr(config, "include_reciprocal", False), -1.0),
            "log": (getattr(config, "include_log", False), "log"),
        }

        # scorer (unchanged to preserve AdaptiveMI behavior)
        self.ami_scorer = AdaptiveMI(
            subsample=min(getattr(config, "max_rows_score", 2000), 2000),
            spearman_gate=getattr(config, "mi_spearman_gate", 0.05),
            min_overlap=getattr(config, "mi_min_overlap", 50),
            ks=(3, 5, 10),
            n_bins=getattr(config, "mi_bins", 16),
            random_state=self.random_state,
        )

        # light winsorization for stability (percentile caps)
        self._winsor_p = float(
            getattr(config, "interaction_winsor_p", 0.001)
        )  # 0.1% tails by default

    # ---------------- utils ----------------

    def _winsorize(self, arr: np.ndarray) -> np.ndarray:
        """Light symmetric winsorization to reduce extreme tails (keeps AMI stable)."""
        a = arr.copy()
        finite = np.isfinite(a)
        if finite.sum() < 20:
            return a
        lo = np.nanpercentile(a[finite], 100 * self._winsor_p)
        hi = np.nanpercentile(a[finite], 100 * (1 - self._winsor_p))
        a[finite] = np.clip(a[finite], lo, hi)
        return a

    def _clean(self, arr: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(arr, dtype=np.float32)
        # sanitize
        arr[~np.isfinite(arr)] = np.nan
        arr = np.where(np.abs(arr) > 1e10, np.nan, arr)
        # winsorize finite values
        arr = self._winsorize(arr)
        # guardrails
        finite = np.isfinite(arr)
        if finite.sum() < 10:
            return None
        if np.nanvar(arr) < self.min_variance:
            return None
        return arr

    def _poly(self, arr: np.ndarray, power: Union[float, str]) -> np.ndarray:
        with np.errstate(all="ignore"):
            if power == 0.5:
                out = np.where(arr >= 0, np.sqrt(arr), np.nan)
            elif power == "log":
                out = np.where(arr > 0, np.log1p(arr), np.nan)
            elif power == -1.0:
                out = np.divide(
                    1.0,
                    arr,
                    out=np.full_like(arr, np.nan),
                    where=(np.abs(arr) > self.eps) & np.isfinite(arr),
                )
            else:
                out = np.power(arr, power)
        return np.clip(out, -1e10, 1e10)

    def _score(self, arr: np.ndarray, y: Optional[np.ndarray]) -> float:
        vals = arr[np.isfinite(arr)]
        if vals.size < 20:
            return 0.0
        if y is None:
            # unsupervised fallback: dispersion score
            m, s = np.mean(vals), np.std(vals)
            cv = s / abs(m) if abs(m) > self.eps else s
            rng = (np.max(vals) - np.min(vals)) / (s + 1e-8)
            return float(max(0.0, 0.7 * cv + 0.3 * rng))
        mask = np.isfinite(arr) & np.isfinite(y)
        if mask.sum() < 20:
            return 0.0
        try:
            # keep AMI call intact
            return float(
                max(
                    0.0,
                    self.ami_scorer.score_pairwise(arr[mask].reshape(-1, 1), y[mask])[
                        0
                    ],
                )
            )
        except Exception:
            return 0.0

    def _subsample_idx(self, n_rows: int) -> np.ndarray:
        if not self.fast_mode or self.row_subsample is None:
            return np.arange(n_rows)
        k = min(int(self.row_subsample), n_rows)
        rng = np.random.RandomState(self.random_state)
        return np.sort(rng.choice(n_rows, size=k, replace=False))

    def _robust_var(self, x: np.ndarray) -> float:
        # IQR-based variance proxy to resist outliers
        x = x[np.isfinite(x)]
        if x.size < 5:
            return 0.0
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        return float(iqr * iqr)

    def _pair_heuristic(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Heuristic to rank column pairs before expensive feature gen/AMI:
          score = robust_var(a)*robust_var(b) * (1 - |corr|)
        Encourages informative-but-not-collinear pairs.
        """
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 25:
            return 0.0
        aa, bb = a[mask], b[mask]
        va = self._robust_var(aa)
        vb = self._robust_var(bb)
        if va <= 0 or vb <= 0:
            return 0.0
        sa = aa.std()
        sb = bb.std()
        corr = 0.0
        if sa > 1e-12 and sb > 1e-12:
            corr = float(np.corrcoef(aa, bb)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        return float((1.0 - abs(corr)) * va * vb)

    def _safe_abs_corr(self, a: np.ndarray, b: np.ndarray) -> float:
        """Absolute Pearson with finite-mask checks."""
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 25:
            return 0.0
        aa = a[m]
        bb = b[m]
        sa = aa.std()
        sb = bb.std()
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        c = float(np.corrcoef(aa, bb)[0, 1])
        if not np.isfinite(c):
            return 0.0
        return abs(c)

    def _prescreen_columns(self, X: pd.DataFrame, y: Optional[pd.Series]) -> List[str]:
        cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            return []
        # cheap variance-based prescreen
        scores: Dict[str, float] = {}
        for c in cols:
            vals = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            scores[c] = float(np.nanvar(vals))
        # optional supervised bump
        if self.prescreen_use_spearman and (y is not None):
            ys = pd.Series(y)
            for c in cols:
                s = pd.Series(X[c])
                m = s.notna() & ys.notna()
                if m.sum() >= 25:
                    rho = s[m].rank().corr(ys[m].rank())
                    if pd.notna(rho):
                        scores[c] += abs(float(rho))
        topk = min(self.prescreen_topk, len(cols))
        return [
            c
            for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[
                :topk
            ]
        ]

    # --------- candidate generation (fit-only, with subsample) ---------

    def _canonical_pair(self, c1: str, c2: str, op_name: str) -> Tuple[str, str, str]:
        """For commutative ops, sort column names so A∘B == B∘A."""
        if op_name in self._commutative_ops:
            if c2 < c1:
                c1, c2 = c2, c1
        return c1, op_name, c2

    def _screen_pairs(
        self,
        pair_cols: List[str],
        cache: Dict[str, np.ndarray],
        y_arr: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, str]]:
        """Rank pairs using the heuristic and keep up to max_pairs."""
        if len(pair_cols) < 2:
            return []

        scores = []

        # Optional target-aware partner gating to reduce combinatorial blow-up.
        # For each anchor feature, only keep top-N partners by target relevance.
        allowed_pairs = None
        if (y_arr is not None) and getattr(self.config, "pair_corr_with_y", True):
            col_rel = {}
            for c in pair_cols:
                col_rel[c] = self._safe_abs_corr(cache[c], y_arr)
            partners = int(max(1, getattr(self.config, "pair_max_per_feature", 32)))
            allowed_pairs = set()
            for c1 in pair_cols:
                ranked = sorted(
                    (c2 for c2 in pair_cols if c2 != c1),
                    key=lambda c2: col_rel.get(c2, 0.0),
                    reverse=True,
                )[:partners]
                for c2 in ranked:
                    allowed_pairs.add(tuple(sorted((c1, c2))))

        redundancy_cap = float(getattr(self.config, "corr_avoid_redundancy", 0.995))
        for c1, c2 in combinations(pair_cols, 2):
            if (allowed_pairs is not None) and (tuple(sorted((c1, c2))) not in allowed_pairs):
                continue
            a, b = cache[c1], cache[c2]
            ab_corr = self._safe_abs_corr(a, b)
            if ab_corr >= redundancy_cap:
                continue
            s = self._pair_heuristic(a, b)
            if s > 0:
                scores.append((s, c1, c2))
        scores.sort(reverse=True, key=lambda t: t[0])
        keep = scores[: self.max_pairs]
        return [(c1, c2) for _, c1, c2 in keep]

    def _generate_interactions_fit(
        self,
        pair_cols: List[str],
        cache: Dict[str, np.ndarray],
        y_arr: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        out = {}
        if len(pair_cols) < 2:
            return out

        # rank pairs first, to reduce unnecessary feature computations
        ranked_pairs = self._screen_pairs(pair_cols, cache, y_arr=y_arr)
        for c1, c2 in ranked_pairs:
            a = cache[c1]
            b = cache[c2]
            # train means for zdiff
            self.col_means_.setdefault(c1, float(np.nanmean(a)))
            self.col_means_.setdefault(c2, float(np.nanmean(b)))
            meta = {"a_mean": self.col_means_[c1], "b_mean": self.col_means_[c2]}

            for op_name, (flag, func) in self.operations.items():
                if not flag:
                    continue
                cc1, opn, cc2 = self._canonical_pair(c1, c2, op_name)
                key = f"{cc1}__{opn}__{cc2}"
                if key in out:
                    continue
                try:
                    feat = func(a, b, meta)
                except TypeError:
                    feat = func(a, b)
                feat = self._clean(feat)
                if feat is not None:
                    out[key] = feat
        return out

    def _generate_polynomials_fit(
        self, cols: List[str], cache: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        out = {}
        for c in cols:
            arr = cache[c]
            self.col_means_.setdefault(c, float(np.nanmean(arr)))
            for suffix, (flag, power) in self.powers.items():
                if not flag:
                    continue
                feat = self._poly(arr, power)
                feat = self._clean(feat)
                if feat is not None:
                    out[f"{c}__{suffix}"] = feat
        return out

    # ---------------- scoring ----------------

    def _score_dict(
        self, feats: Dict[str, np.ndarray], y_arr: Optional[np.ndarray]
    ) -> Dict[str, float]:
        names = list(feats.keys())
        arrays = [feats[n] for n in names]
        if _HAVE_JOBLIB and (self.n_jobs is not None) and (self.n_jobs != 0):
            n_jobs = self.n_jobs
            scores = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._score)(arr, y_arr) for arr in arrays
            )
        else:
            scores = [self._score(arr, y_arr) for arr in arrays]
        return {n: s for n, s in zip(names, scores)}

    def _select_topk(
        self, feats: Dict[str, np.ndarray], y_arr: Optional[np.ndarray], k: int
    ) -> List[str]:
        if not feats or k <= 0:
            return []
        scores = self._score_dict(feats, y_arr)
        self.feature_scores_.update(scores)
        ordered = [
            n
            for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if s > 0
        ]
        return ordered[:k]

    def _aggregate_stability_score(self, values: List[float]) -> float:
        if not values:
            return 0.0
        mode = str(getattr(self.config, "importance_agg", "median")).lower()
        arr = np.asarray(values, dtype=float)
        if mode == "mean":
            return float(np.mean(arr))
        if mode == "max":
            return float(np.max(arr))
        return float(np.median(arr))

    def _stable_select_topk(
        self, feats: Dict[str, np.ndarray], y_arr: Optional[np.ndarray], k: int
    ) -> List[str]:
        """Stability selection across folds for robust interaction ranking."""
        if (not feats) or (k <= 0):
            return []
        if y_arr is None:
            return self._select_topk(feats, y_arr, k)

        n = len(y_arr)
        n_splits = int(max(2, min(getattr(self.config, "n_splits", 5), n // 20)))
        if n_splits < 2:
            return self._select_topk(feats, y_arr, k)

        # Pick CV splitter by task.
        y_valid = y_arr[np.isfinite(y_arr)]
        is_classification = (
            str(getattr(self.config, "task", "regression")).lower() == "classification"
        )
        if is_classification and y_valid.size > 0:
            unique = np.unique(y_valid)
            if unique.size > 1:
                splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
            else:
                splitter = KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
        else:
            splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )

        names = list(feats.keys())
        freq = {n: 0 for n in names}
        fold_scores: Dict[str, List[float]] = {n: [] for n in names}
        per_fold_topk = int(max(1, getattr(self.config, "min_selected_per_fold", 20)))

        # Build split iterator with finite targets only.
        valid_idx = np.where(np.isfinite(y_arr))[0]
        if valid_idx.size < max(40, 2 * n_splits):
            return self._select_topk(feats, y_arr, k)

        y_valid_all = y_arr[valid_idx]
        if isinstance(splitter, StratifiedKFold):
            split_iter = splitter.split(valid_idx, y_valid_all)
        else:
            split_iter = splitter.split(valid_idx)

        for _, te_local in split_iter:
            idx = valid_idx[te_local]
            y_fold = y_arr[idx]
            fold_feat = {n: feats[n][idx] for n in names}
            sdict = self._score_dict(fold_feat, y_fold)
            ordered = [
                n
                for n, s in sorted(sdict.items(), key=lambda x: x[1], reverse=True)
                if s > 0
            ][:per_fold_topk]
            for n in ordered:
                freq[n] += 1
                fold_scores[n].append(float(sdict[n]))

        min_freq = int(np.ceil(self.stability_min_freq * n_splits))
        candidates = []
        for n in names:
            if freq[n] >= min_freq and fold_scores[n]:
                agg = self._aggregate_stability_score(fold_scores[n])
                candidates.append((freq[n], agg, n))

        # Fallback to global ranking if strict stability gate removes everything.
        if not candidates:
            return self._select_topk(feats, y_arr, k)

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = [n for _, _, n in candidates[:k]]

        # Store aggregate scores for reportability.
        for _, agg, n in candidates:
            self.feature_scores_[n] = float(agg)
        return selected

    # ---------------- sklearn API ----------------

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "InteractionTransformer":
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        if not self.numerical_cols_:
            self.is_fitted = True
            return self

        # subsample rows once
        idx = self._subsample_idx(len(X))
        X_sub = X.iloc[idx, :]

        # cache numeric columns as float arrays (subsampled)
        cache: Dict[str, np.ndarray] = {}
        for c in self.numerical_cols_:
            cache[c] = pd.to_numeric(X_sub[c], errors="coerce").to_numpy(
                dtype=np.float32
            )

        y_arr = None
        if y is not None:
            y_arr = pd.Series(y).iloc[idx].to_numpy(dtype=float)
            y_arr[~np.isfinite(y_arr)] = np.nan  # AMI handles mask

        # prescreen columns (on full X; cheap) and then pair-rank on subsample cache
        pair_cols = self._prescreen_columns(X, y)

        # generate candidates on subsample & cache
        cand_inter, cand_poly = {}, {}
        if getattr(self.config, "create_interactions", True):
            cand_inter = self._generate_interactions_fit(pair_cols, cache, y_arr=y_arr)
        if getattr(self.config, "create_polynomials", True):
            cand_poly = self._generate_polynomials_fit(self.numerical_cols_, cache)

        # optional redundancy prune among candidates
        if self.enable_redundancy_prune:
            cand_inter = self._fast_redundancy_prune(cand_inter)
            cand_poly = self._fast_redundancy_prune(cand_poly)

        # select
        if self.enable_stability_selection:
            sel_inter_names = self._stable_select_topk(
                cand_inter, y_arr, self.max_interactions
            )
            sel_poly_names = self._stable_select_topk(
                cand_poly, y_arr, self.max_polynomials
            )
        else:
            sel_inter_names = self._select_topk(
                cand_inter, y_arr, self.max_interactions
            )
            sel_poly_names = self._select_topk(
                cand_poly, y_arr, self.max_polynomials
            )

        # to recipes
        self.selected_interactions_ = []
        for nm in sel_inter_names:
            c1, op_name, c2 = nm.split("__", 2)
            self.selected_interactions_.append((c1, op_name, c2))

        self.selected_polynomials_ = []
        for nm in sel_poly_names:
            col, suffix = nm.split("__", 1)
            power = self.powers[suffix][1] if suffix in self.powers else None
            if power is not None:
                self.selected_polynomials_.append((col, power))

        self.is_fitted = True
        return self

    # faster streaming redundancy prune (Spearman-ish via Pearson on ranks; stop early)
    def _fast_redundancy_prune(
        self, feats: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if not feats:
            return feats
        names = list(feats.keys())
        kept = []
        # pre-rank to approximate Spearman cheaply
        ranked = {}
        for n in names:
            v = feats[n]
            mask = np.isfinite(v)
            r = np.full_like(v, np.nan, dtype=np.float32)
            # rank only finite elements
            r[mask] = np.argsort(np.argsort(v[mask])).astype(np.float32)
            ranked[n] = r

        for n in names:
            r = ranked[n]
            drop = False
            for k in kept:
                rk = ranked[k]
                m = np.isfinite(r) & np.isfinite(rk)
                if m.sum() < 25:
                    continue
                # Pearson on ranks ~= Spearman
                a = r[m]
                b = rk[m]
                sa = a.std()
                sb = b.std()
                if sa < 1e-12 or sb < 1e-12:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])
                if abs(corr) >= self.redundancy_corr:
                    drop = True
                    break
            if not drop:
                kept.append(n)
        return {k: feats[k] for k in kept}

    def _compute_interaction(
        self, X: pd.DataFrame, c1: str, op_name: str, c2: str
    ) -> Optional[np.ndarray]:
        # canonicalize at transform too, to match recipe naming
        if op_name in self._commutative_ops and c2 < c1:
            c1, c2 = c2, c1
        a = pd.to_numeric(X[c1], errors="coerce").to_numpy(dtype=np.float32)
        b = pd.to_numeric(X[c2], errors="coerce").to_numpy(dtype=np.float32)
        meta = {
            "a_mean": self.col_means_.get(c1, float(np.nanmean(a))),
            "b_mean": self.col_means_.get(c2, float(np.nanmean(b))),
        }
        func = self.operations[op_name][1]
        try:
            feat = func(a, b, meta)
        except TypeError:
            feat = func(a, b)
        return self._clean(feat)

    def _compute_polynomial(
        self, X: pd.DataFrame, col: str, power: Union[float, str]
    ) -> Optional[np.ndarray]:
        arr = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=np.float32)
        return self._clean(self._poly(arr, power))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            return pd.DataFrame(index=X.index)

        feats: Dict[str, np.ndarray] = {}

        for c1, op_name, c2 in self.selected_interactions_:
            if c1 in X.columns and c2 in X.columns:
                arr = self._compute_interaction(X, c1, op_name, c2)
                if arr is not None:
                    # keep canonical name
                    if op_name in self._commutative_ops and c2 < c1:
                        c1, c2 = c2, c1
                    feats[f"{c1}__{op_name}__{c2}"] = arr

        for col, power in self.selected_polynomials_:
            if col in X.columns:
                arr = self._compute_polynomial(X, col, power)
                if arr is not None:
                    suffix = next(
                        (k for k, (_, p) in self.powers.items() if p == power),
                        str(power),
                    )
                    feats[f"{col}__{suffix}"] = arr

        return (
            pd.DataFrame(feats, index=X.index, dtype=np.float32)
            if feats
            else pd.DataFrame(index=X.index)
        )

