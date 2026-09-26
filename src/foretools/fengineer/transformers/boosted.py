"""OpenFE-style feature generation scored by incremental gain over a GBDT.

Reference
---------
Zhang, T. et al. "OpenFE: Automated Feature Generation with Expert-level
Performance", ICML 2023.

The key idea is that a candidate feature is only useful if it adds
information *beyond what a strong model already extracts from the base
features*.  Univariate relevance (MI / correlation with y) mostly rewards
candidates that re-encode their parents.  Instead we:

1. Fit a base GBDT and take its out-of-fold margins as ``init_score``.
2. For every candidate, boost a few small Newton-step trees on that single
   candidate starting from ``init_score`` and measure the validation-loss
   reduction ("feature boosting").
3. Prune candidates with successive halving over growing row budgets, so the
   bulk of candidates is only ever evaluated on small samples.
4. Select greedily with conditional gain: split the rows in two halves and,
   at each step, score the remaining survivors by the cross-fitted gain over
   the *current* margins (fit on one half, evaluate on the other, and swap).
   The winner's trees are then added to the margins out-of-fold, so a
   candidate that is redundant with an already selected one shows ~zero
   gain.  Lazy evaluation (stale gains are upper bounds in practice) keeps
   the number of re-scorings small.  Near-duplicates (rank correlation above
   ``dedup_threshold``) are dropped before this stage to save work.
   This stage runs only on the halving *training* rows, whose evaluation
   played no part in picking the survivors, and a candidate must beat the
   best gain of row-shuffled "shadow" copies of the pool (a null threshold
   in the spirit of Boruta / null importances), which controls the
   multiple-testing noise of screening hundreds of candidates.

Candidate operators
-------------------
- binary numeric: ``add``, ``sub``, ``mul``, ``div`` (both orders)
- group-by aggregates of a numeric column within a key: ``gmean``, ``gstd``,
  ``gmin``, ``gmax``, ``gmedian``; and row-relative ones: deviation from
  (``gdev``) and ratio to (``gratio``) the group mean, and the percentile
  rank within the group's training distribution (``grank``)
- categorical: frequency encoding of a key (``freq``) and of the combination
  of two keys (``combfreq``)

Unary monotone transforms are omitted on purpose: tree ensembles are
invariant to them, so they carry no incremental gain.
"""

from __future__ import annotations

import heapq
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeRegressor

from .support import BaseFeatureTransformer

_GROUP_AGGS = ("mean", "std", "min", "max", "median")
_NAN_KEY = "__nan__"


def _key_series(s: pd.Series) -> pd.Series:
    """Stable string keys for grouping, identical at fit and transform time."""
    return s.astype("object").where(s.notna(), _NAN_KEY).astype(str)


class BoostedFeatureGenerator(BaseFeatureTransformer):
    """Generate features that improve a GBDT, OpenFE-style.

    Parameters
    ----------
    config : FeatureConfig
        Reads ``config.boosted`` (a :class:`BoostedConfig`), ``config.task``
        and ``config.random_state``.

    Attributes
    ----------
    recipes_ : list[tuple[str, str, str, str | None]]
        Selected recipes ``(kind, op, a, b)`` in output order.
    feature_names_ : list[str]
        Output column names, aligned with ``recipes_``.
    feature_gains_ : dict[str, float]
        Relative validation-loss reduction of each surviving candidate at the
        last successive-halving stage.
    conditional_gains_ : dict[str, float]
        Cross-fitted relative loss reduction of each selected feature, given
        the base model and the features selected before it.
    base_loss_ : float
        Validation loss of the out-of-fold base-model margins.
    null_gain_ : float
        Selection threshold of the greedy stage (best shadow gain, at least
        ``min_gain``).
    """

    def __init__(self, config: Any, **overrides: Any):
        super().__init__(config)
        cfg = getattr(config, "boosted", None)

        def opt(name: str, default: Any) -> Any:
            if name in overrides:
                return overrides[name]
            return getattr(cfg, name, default) if cfg is not None else default

        self.max_features = int(opt("max_features", 30))
        self.max_base_features = int(opt("max_base_features", 20))
        self.max_group_keys = int(opt("max_group_keys", 5))
        self.max_group_cardinality = int(opt("max_group_cardinality", 1000))
        self.binary_ops = tuple(opt("binary_ops", ("add", "sub", "mul", "div")))
        self.group_aggs = tuple(
            opt("group_aggs", (*_GROUP_AGGS, "dev", "ratio", "rank"))
        )
        self.include_freq = bool(opt("include_freq", True))
        self.include_combine = bool(opt("include_combine", True))
        self.max_rows = int(opt("max_rows", 20000))
        self.min_rows = int(opt("min_rows", 200))
        self.halving_min_rows = int(opt("halving_min_rows", 1000))
        self.max_halving_rounds = int(opt("max_halving_rounds", 4))
        self.valid_fraction = float(opt("valid_fraction", 0.25))
        self.dedup_threshold = float(opt("dedup_threshold", 0.95))
        self.oof_folds = int(opt("oof_folds", 3))
        self.base_max_iter = int(opt("base_max_iter", 100))
        self.n_rounds = int(opt("n_rounds", 3))
        self.learning_rate = float(opt("learning_rate", 0.3))
        self.tree_max_leaves = int(opt("tree_max_leaves", 16))
        self.reg_lambda = float(opt("reg_lambda", 1.0))
        self.min_gain = float(opt("min_gain", 1e-4))
        self.confirm = bool(opt("confirm", True))
        self.confirm_pool = int(opt("confirm_pool", 3 * self.max_features))
        self.n_shadows = int(opt("n_shadows", 30))
        self.random_state = int(opt("random_state", getattr(config, "random_state", 42)))

        self.recipes_: list[tuple[str, str, str, str | None]] = []
        self.feature_names_: list[str] = []
        self.feature_gains_: dict[str, float] = {}
        self.conditional_gains_: dict[str, float] = {}
        self.base_loss_: float = float("nan")
        self.null_gain_: float = float("nan")
        self.group_tables_: dict[tuple[str, str], pd.DataFrame] = {}
        self.group_sorted_: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self.freq_tables_: dict[tuple[str, ...], pd.Series] = {}

    # ------------------------------------------------------------------
    # task / loss helpers
    # ------------------------------------------------------------------

    def _resolve_task(self, y: pd.Series) -> str:
        task = str(getattr(self.config, "task", "regression")).lower()
        if task == "classification" or not pd.api.types.is_numeric_dtype(y):
            return "multiclass" if y.nunique() > 2 else "binary"
        return "regression"

    def _encode_target(self, y: pd.Series) -> np.ndarray:
        if self.task_ == "regression":
            return y.to_numpy(dtype=float).reshape(-1, 1)
        codes = np.searchsorted(self.classes_, y.to_numpy())
        if self.task_ == "binary":
            return codes.astype(float).reshape(-1, 1)
        return np.eye(len(self.classes_))[codes]

    def _probs(self, margin: np.ndarray) -> np.ndarray:
        if self.task_ == "binary":
            return 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30)))
        z = margin - margin.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _loss(self, margin: np.ndarray, Y: np.ndarray) -> float:
        if self.task_ == "regression":
            return float(np.mean((margin - Y) ** 2))
        p = np.clip(self._probs(margin), 1e-12, 1 - 1e-12)
        if self.task_ == "binary":
            return float(-np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p)))
        return float(-np.mean(np.sum(Y * np.log(p), axis=1)))

    def _grad_hess(self, margin: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.task_ == "regression":
            return margin - Y, np.ones_like(margin)
        p = self._probs(margin)
        return p - Y, np.maximum(p * (1 - p), 1e-12)

    # ------------------------------------------------------------------
    # base model
    # ------------------------------------------------------------------

    def _make_base_model(self):
        params = dict(max_iter=self.base_max_iter, random_state=self.random_state)
        if self.task_ == "regression":
            return HistGradientBoostingRegressor(**params)
        return HistGradientBoostingClassifier(**params)

    def _model_margin(self, model, X: np.ndarray) -> np.ndarray:
        if self.task_ == "regression":
            return model.predict(X).reshape(-1, 1)
        m = model.decision_function(X)
        return m.reshape(-1, 1) if m.ndim == 1 else m

    def _prior_margin(self, Y: np.ndarray, n: int) -> np.ndarray:
        mean = Y.mean(axis=0, keepdims=True)
        if self.task_ == "regression":
            return np.repeat(mean, n, axis=0)
        p = np.clip(mean, 1e-6, 1 - 1e-6)
        m = np.log(p / (1 - p)) if self.task_ == "binary" else np.log(p)
        return np.repeat(m, n, axis=0)

    def _splitter(self, n_splits: int):
        if self.task_ == "regression":
            return KFold(n_splits, shuffle=True, random_state=self.random_state)
        return StratifiedKFold(n_splits, shuffle=True, random_state=self.random_state)

    def _oof_margin(self, B: np.ndarray, y_lab: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Cross-fitted base-model margins, so residuals are honest on every row."""
        margin = np.empty((len(B), Y.shape[1]), dtype=float)
        try:
            for tr, te in self._splitter(self.oof_folds).split(B, y_lab):
                model = self._make_base_model().fit(B[tr], y_lab[tr])
                if self.task_ == "multiclass" and len(model.classes_) != Y.shape[1]:
                    raise ValueError("fold is missing classes")
                margin[te] = self._model_margin(model, B[te])
            return margin
        except Exception as exc:
            self.logger.warning(f"Base OOF model failed ({exc}); using prior margin.")
            return self._prior_margin(Y, len(B))

    def _rank_base_columns(
        self, B: np.ndarray, y_lab: np.ndarray, cols: list[str]
    ) -> list[str]:
        """Order base columns by permutation importance of a GBDT on held-out rows."""
        try:
            strat = y_lab if self.task_ != "regression" else None
            B_tr, B_va, y_tr, y_va = train_test_split(
                B, y_lab, test_size=0.3, random_state=self.random_state, stratify=strat
            )
            model = self._make_base_model().fit(B_tr, y_tr)
            scoring = (
                "neg_mean_squared_error" if self.task_ == "regression" else "neg_log_loss"
            )
            imp = permutation_importance(
                model,
                B_va[:5000],
                y_va[:5000],
                scoring=scoring,
                n_repeats=2,
                random_state=self.random_state,
            ).importances_mean
            order = np.argsort(-imp)
            return [cols[i] for i in order]
        except Exception as exc:
            self.logger.warning(f"Base importance ranking failed ({exc}).")
            return list(cols)

    # ------------------------------------------------------------------
    # candidates
    # ------------------------------------------------------------------

    def _group_table(self, X: pd.DataFrame, x: str, g: str) -> pd.DataFrame:
        key = (g, x)
        if key not in self.group_tables_:
            vals = pd.to_numeric(X[x], errors="coerce").astype(float)
            aggs = [a for a in _GROUP_AGGS if a in self.group_aggs or a == "mean"]
            keys = _key_series(X[g])
            self.group_tables_[key] = vals.groupby(keys).agg(aggs)
            if "rank" in self.group_aggs:
                self.group_sorted_[key] = {
                    k: np.sort(v.dropna().to_numpy())
                    for k, v in vals.groupby(keys)
                }
        return self.group_tables_[key]

    def _freq_table(self, X: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
        if keys not in self.freq_tables_:
            combo = self._combined_key(X, keys)
            self.freq_tables_[keys] = combo.value_counts(normalize=True)
        return self.freq_tables_[keys]

    @staticmethod
    def _combined_key(X: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
        combo = _key_series(X[keys[0]])
        for k in keys[1:]:
            combo = combo + "\x1f" + _key_series(X[k])
        return combo

    @staticmethod
    def _recipe_name(recipe: tuple[str, str, str, str | None]) -> str:
        kind, op, a, b = recipe
        if kind == "bin":
            return f"{a}__{op}__{b}"
        if kind == "grp":
            return f"{a}__g{op}__{b}"
        if kind == "comb":
            return f"{a}__combfreq__{b}"
        return f"{a}__freq"

    def _compute(
        self, X: pd.DataFrame, recipe: tuple[str, str, str, str | None]
    ) -> np.ndarray:
        kind, op, a, b = recipe
        if kind == "bin":
            va = pd.to_numeric(X[a], errors="coerce").to_numpy(dtype=float)
            vb = pd.to_numeric(X[b], errors="coerce").to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                if op == "add":
                    out = va + vb
                elif op == "sub":
                    out = va - vb
                elif op == "mul":
                    out = va * vb
                else:
                    out = np.divide(
                        va, vb, out=np.full_like(va, np.nan), where=np.abs(vb) > 1e-12
                    )
        elif kind == "grp":
            table = self.group_tables_[(b, a)]
            keys = _key_series(X[b])
            x = pd.to_numeric(X[a], errors="coerce").to_numpy(dtype=float)
            if op == "dev":
                out = x - keys.map(table["mean"]).to_numpy(dtype=float)
            elif op == "ratio":
                gm = keys.map(table["mean"]).to_numpy(dtype=float)
                with np.errstate(all="ignore"):
                    out = np.divide(
                        x, gm, out=np.full_like(x, np.nan), where=np.abs(gm) > 1e-12
                    )
            elif op == "rank":
                sorted_vals = self.group_sorted_[(b, a)]
                out = np.full(len(x), np.nan)
                for k, idx in keys.groupby(keys).indices.items():
                    ref = sorted_vals.get(k)
                    if ref is not None and ref.size:
                        out[idx] = np.searchsorted(ref, x[idx], side="right") / ref.size
                out[~np.isfinite(x)] = np.nan
            else:
                out = keys.map(table[op]).to_numpy(dtype=float)
        else:
            keys_t = (a, b) if kind == "comb" else (a,)
            table = self.freq_tables_[keys_t]
            combo = self._combined_key(X, keys_t)
            out = combo.map(table).fillna(0.0).to_numpy(dtype=float)
        out = np.asarray(out, dtype=float)
        out[~np.isfinite(out)] = np.nan
        return out

    def _generate_candidates(
        self, X: pd.DataFrame, num_ranked: list[str], key_ranked: list[str]
    ) -> list[tuple[str, str, str, str | None]]:
        nums = num_ranked[: self.max_base_features]
        keys = key_ranked[: self.max_group_keys]
        cands: list[tuple[str, str, str, str | None]] = []

        for a, b in combinations(nums, 2):
            for op in self.binary_ops:
                cands.append(("bin", op, a, b))
                if op == "div":
                    cands.append(("bin", op, b, a))

        for g in keys:
            for x in nums:
                if x == g:
                    continue
                self._group_table(X, x, g)
                for agg in self.group_aggs:
                    cands.append(("grp", agg, x, g))

        if self.include_freq:
            for g in keys:
                self._freq_table(X, (g,))
                cands.append(("freq", "freq", g, None))
        if self.include_combine:
            for g1, g2 in combinations(keys, 2):
                self._freq_table(X, (g1, g2))
                cands.append(("comb", "freq", g1, g2))
        return cands

    # ------------------------------------------------------------------
    # feature boosting evaluator
    # ------------------------------------------------------------------

    def _boost(
        self,
        x: np.ndarray,
        tr: np.ndarray,
        va: np.ndarray,
        margin: np.ndarray,
    ) -> tuple[float, np.ndarray | None]:
        """Boost Newton-step trees on one candidate, starting from ``margin``.

        Trees are fitted on ``tr`` and applied to ``va``.  Returns the relative
        loss reduction on ``va`` and the updated ``va`` margins (None when the
        candidate is unusable).
        """
        x_tr, x_va = x[tr], x[va]
        finite = np.isfinite(x_tr)
        if finite.sum() < 20 or np.nanstd(x_tr[finite]) < 1e-12:
            return 0.0, None
        # Trees are split-invariant to monotone maps, so a sentinel below the
        # observed range is an exact stand-in for a "missing" branch.
        lo, hi = np.nanmin(x_tr), np.nanmax(x_tr)
        fill = lo - (hi - lo) - 1.0
        x_tr = np.where(np.isfinite(x_tr), x_tr, fill).reshape(-1, 1)
        x_va = np.where(np.isfinite(x_va), x_va, fill).reshape(-1, 1)

        m_tr = margin[tr].copy()
        m_va = margin[va].copy()
        Y_tr, Y_va = self._Y[tr], self._Y[va]
        base = self._loss(m_va, Y_va)
        if base <= 0:
            return 0.0, None

        min_leaf = max(5, len(tr) // 100)
        for _ in range(self.n_rounds):
            G, H = self._grad_hess(m_tr, Y_tr)
            tree = DecisionTreeRegressor(
                max_leaf_nodes=self.tree_max_leaves,
                min_samples_leaf=min_leaf,
                random_state=self.random_state,
            )
            tree.fit(x_tr, -G if G.shape[1] > 1 else -G.ravel())
            leaf_tr = tree.apply(x_tr)
            leaf_va = tree.apply(x_va)
            n_nodes = tree.tree_.node_count
            values = np.zeros((n_nodes, G.shape[1]))
            for k in range(G.shape[1]):
                sg = np.bincount(leaf_tr, weights=G[:, k], minlength=n_nodes)
                sh = np.bincount(leaf_tr, weights=H[:, k], minlength=n_nodes)
                values[:, k] = -sg / (sh + self.reg_lambda)
            m_tr += self.learning_rate * values[leaf_tr]
            m_va += self.learning_rate * values[leaf_va]

        return float((base - self._loss(m_va, Y_va)) / base), m_va

    def _successive_halving(
        self,
        X_s: pd.DataFrame,
        cands: list[tuple[str, str, str, str | None]],
        tr_all: np.ndarray,
        va_all: np.ndarray,
    ) -> list[tuple[tuple[str, str, str, str | None], float]]:
        schedule = [len(tr_all)]
        while schedule[0] // 2 >= self.halving_min_rows:
            schedule.insert(0, schedule[0] // 2)
        schedule = schedule[-max(1, self.max_halving_rounds):]
        keep_min = max(1, self.confirm_pool)
        ratio = len(va_all) / max(1, len(tr_all))

        scored: list[tuple[tuple[str, str, str, str | None], float]] = []
        for stage, n_tr in enumerate(schedule):
            tr = tr_all[:n_tr]
            va = va_all[: max(20, int(round(n_tr * ratio)))]
            scored = []
            for recipe in cands:
                try:
                    gain, _ = self._boost(
                        self._compute(X_s, recipe), tr, va, self._margin
                    )
                except Exception:
                    gain = 0.0
                scored.append((recipe, gain))
            scored.sort(key=lambda t: t[1], reverse=True)
            if stage < len(schedule) - 1:
                n_keep = max(keep_min, int(np.ceil(len(scored) / 2)))
                cands = [r for r, _ in scored[:n_keep]]
        return [(r, g) for r, g in scored if g > self.min_gain]

    def _dedup(
        self,
        X_s: pd.DataFrame,
        survivors: list[tuple[tuple[str, str, str, str | None], float]],
        rows: np.ndarray,
    ) -> list[tuple[tuple[str, str, str, str | None], float]]:
        """Greedy (by gain) removal of candidates rank-correlated with a kept one."""
        if self.dedup_threshold >= 1.0 or len(survivors) < 2:
            return survivors
        rows = rows[:5000]
        R = pd.DataFrame(
            {i: self._compute(X_s.iloc[rows], r) for i, (r, _) in enumerate(survivors)}
        ).rank()
        R = R.fillna(R.mean())
        R = (R - R.mean()) / R.std(ddof=0).replace(0, np.nan)
        R = R.fillna(0.0).to_numpy()
        kept: list[int] = []
        for i in range(R.shape[1]):
            if kept:
                corr = np.abs(R[:, kept].T @ R[:, i]) / len(R)
                if corr.max() >= self.dedup_threshold:
                    continue
            kept.append(i)
        return [survivors[i] for i in kept]

    def _cross_fit(
        self, x: np.ndarray, halves: tuple[np.ndarray, np.ndarray], margin: np.ndarray
    ) -> tuple[float, np.ndarray | None]:
        """Mean gain over both half-swaps, plus the out-of-fold updated margins."""
        a, b = halves
        g_b, m_b = self._boost(x, a, b, margin)
        g_a, m_a = self._boost(x, b, a, margin)
        if m_a is None or m_b is None:
            return 0.0, None
        new_margin = margin.copy()
        new_margin[a], new_margin[b] = m_a, m_b
        n_a, n_b = len(a), len(b)
        return float((g_a * n_a + g_b * n_b) / (n_a + n_b)), new_margin

    def _greedy_select(
        self,
        X_s: pd.DataFrame,
        pool: list[tuple[tuple[str, str, str, str | None], float]],
        rows: np.ndarray,
    ) -> list[tuple[tuple[str, str, str, str | None], float]]:
        """Lazy-greedy forward selection by cross-fitted conditional gain."""
        rows = np.random.RandomState(self.random_state).permutation(rows)
        half = len(rows) // 2
        halves = (np.sort(rows[:half]), np.sort(rows[half:]))
        xs = [self._compute(X_s, r) for r, _ in pool]
        margin = self._margin.copy()

        # Null threshold: best cross-fitted gain among row-shuffled candidates.
        rng = np.random.RandomState(self.random_state + 1)
        threshold = self.min_gain
        for i in rng.permutation(len(xs))[: self.n_shadows]:
            shadow = xs[i].copy()
            shadow[rows] = shadow[rng.permutation(rows)]
            threshold = max(threshold, self._cross_fit(shadow, halves, margin)[0])
        self.null_gain_ = float(threshold)

        # Heap entries: (-gain, index, step at which the gain was computed).
        # Halving gains seed the priorities; they are re-scored before use.
        heap = [(-g, i, -1) for i, (_, g) in enumerate(pool)]
        heapq.heapify(heap)
        fresh: dict[int, np.ndarray] = {}
        selected: list[tuple[tuple[str, str, str, str | None], float]] = []

        while heap and len(selected) < self.max_features:
            neg_g, i, step = heapq.heappop(heap)
            if step != len(selected):
                gain, new_margin = self._cross_fit(xs[i], halves, margin)
                if new_margin is not None and gain > threshold:
                    fresh[i] = new_margin
                    heapq.heappush(heap, (-gain, i, len(selected)))
                continue
            selected.append((pool[i][0], -neg_g))
            margin = fresh[i]
            fresh = {}
        return selected

    # ------------------------------------------------------------------
    # sklearn-style API
    # ------------------------------------------------------------------

    def _base_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        nums, keys = [], []
        for c in X.columns:
            s = X[c]
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                v = pd.to_numeric(s, errors="coerce").astype(float)
                if v.notna().mean() < 0.05 or v.nunique() < 2:
                    continue
                nums.append(c)
                nu = v.nunique()
                if nu <= self.max_group_cardinality and np.allclose(
                    v.dropna() % 1, 0
                ):
                    keys.append(c)
            elif (
                isinstance(s.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)  # pandas>=3 default "str" dtype
            ):
                nu = s.nunique(dropna=False)
                if 2 <= nu <= self.max_group_cardinality:
                    keys.append(c)
        return nums, keys

    def _base_matrix(self, X: pd.DataFrame, nums: list[str], keys: list[str]) -> tuple[np.ndarray, list[str]]:
        cols, mats = [], []
        for c in nums:
            cols.append(c)
            mats.append(pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float))
        for c in keys:
            if c in nums:
                continue
            codes = pd.Categorical(_key_series(X[c])).codes.astype(float)
            cols.append(c)
            mats.append(codes)
        return np.column_stack(mats), cols

    def _split(
        self, idx: np.ndarray, y_lab: np.ndarray, frac: float
    ) -> tuple[np.ndarray, np.ndarray]:
        strat = y_lab[idx] if self.task_ != "regression" else None
        try:
            return train_test_split(
                idx, test_size=frac, random_state=self.random_state, stratify=strat
            )
        except ValueError:
            return train_test_split(idx, test_size=frac, random_state=self.random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BoostedFeatureGenerator":
        self.recipes_, self.feature_names_ = [], []
        self.feature_gains_, self.conditional_gains_ = {}, {}
        self.group_tables_, self.group_sorted_, self.freq_tables_ = {}, {}, {}
        self.is_fitted = True
        if y is None:
            return self

        y_s = pd.Series(np.asarray(y), index=X.index)
        valid = y_s.notna().to_numpy()
        nums, keys = self._base_columns(X)
        if valid.sum() < self.min_rows or not (nums or keys):
            return self

        self.task_ = self._resolve_task(y_s[valid])
        rng = np.random.RandomState(self.random_state)
        pos = np.flatnonzero(valid)
        if len(pos) > self.max_rows:
            pos = np.sort(rng.choice(pos, self.max_rows, replace=False))
        X_s = X.iloc[pos]
        y_lab = y_s.iloc[pos].to_numpy()
        if self.task_ != "regression":
            self.classes_ = np.unique(y_lab)
            if len(self.classes_) < 2:
                return self
        self._Y = self._encode_target(pd.Series(y_lab))

        B, base_cols = self._base_matrix(X_s, nums, keys)
        self._margin = self._oof_margin(B, y_lab, self._Y)

        ranked = self._rank_base_columns(B, y_lab, base_cols)
        num_ranked = [c for c in ranked if c in nums]
        key_ranked = [c for c in ranked if c in keys]
        cands = self._generate_candidates(X, num_ranked, key_ranked)
        if not cands:
            del self._margin, self._Y
            return self

        idx = np.arange(len(pos))
        tr, va = self._split(idx, y_lab, self.valid_fraction)
        tr, va = rng.permutation(tr), rng.permutation(va)
        self.base_loss_ = self._loss(self._margin[va], self._Y[va])

        survivors = self._successive_halving(X_s, cands, tr, va)
        self.feature_gains_ = {self._recipe_name(r): g for r, g in survivors}

        if self.confirm and survivors:
            pool = self._dedup(X_s, survivors, tr)[: self.confirm_pool]
            selected = self._greedy_select(X_s, pool, np.sort(tr))
            self.conditional_gains_ = {self._recipe_name(r): g for r, g in selected}
        else:
            selected = survivors
        selected = selected[: self.max_features]

        self.recipes_ = [r for r, _ in selected]
        self.feature_names_ = [self._recipe_name(r) for r in self.recipes_]

        # Drop fitted tables that no selected recipe needs.
        need_groups = {(r[3], r[2]) for r in self.recipes_ if r[0] == "grp"}
        need_freq = {
            (r[2], r[3]) if r[0] == "comb" else (r[2],)
            for r in self.recipes_
            if r[0] in {"comb", "freq"}
        }
        self.group_tables_ = {k: v for k, v in self.group_tables_.items() if k in need_groups}
        need_rank = {(r[3], r[2]) for r in self.recipes_ if r[:2] == ("grp", "rank")}
        self.group_sorted_ = {k: v for k, v in self.group_sorted_.items() if k in need_rank}
        self.freq_tables_ = {k: v for k, v in self.freq_tables_.items() if k in need_freq}
        del self._margin, self._Y
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        if not self.is_fitted or not self.recipes_:
            return pd.DataFrame(index=X.index)
        feats = {}
        for recipe, name in zip(self.recipes_, self.feature_names_):
            _, _, a, b = recipe
            if a not in X.columns or (b is not None and b not in X.columns):
                continue
            feats[name] = self._compute(X, recipe)
        if not feats:
            return pd.DataFrame(index=X.index)
        return pd.DataFrame(feats, index=X.index, dtype=np.float32)
