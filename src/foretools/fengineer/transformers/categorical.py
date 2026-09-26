from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import chi2, f_regression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .support import BaseFeatureTransformer


class CategoricalTransformer(BaseFeatureTransformer):
    """
    Modern categorical transformer:
      - Strategies: 'auto', 'onehot', 'freq', 'hashing', 'target_kfold', 'ordinal', 'loo', 'james-stein'
      - Leakage-safe target encoding: every target-based strategy is computed
        out-of-fold on training rows and with full-data statistics at
        inference, using the same shrinkage formula in both cases
      - Smoothing: m-estimate (``smoothing_prior`` float) or empirical-Bayes
        (``smoothing_prior="auto"``, the variance-ratio shrinkage used by
        sklearn's ``TargetEncoder``)
      - Multiclass targets (``task="classification"`` or non-numeric labels
        with >2 classes) get one-vs-rest encodings, one column per class
      - Robust rare handling: frequency threshold, min count, top-k cap
      - Stable output schema across fit/transform

    Notes
    -----
    'loo' is kept for API compatibility but is computed out-of-fold with an
    unsmoothed mean: in-sample leave-one-out encodings are a decreasing
    function of the row's own target within each category, which leaks y.
    'james-stein' is the empirical-Bayes shrinkage and matches
    ``smoothing_prior="auto"``.
    """

    _TARGET_STRATEGIES = ("target_kfold", "loo", "james-stein")

    def __init__(
        self,
        config,
        strategies: tuple[str, ...] = ("auto",),
        rare_threshold: None | (
            float
        ) = None,  # fraction (0..1); falls back to config.rare_threshold
        min_count: int = 1,  # absolute count for rare categories
        top_k: int | None = None,  # keep top_k most frequent; others -> OTHER
        hashing_dim: int = 64,
        n_splits: int | None = None,  # falls back to config.categorical.n_splits
        target_min_samples: int | None = None,  # min rows to consider target encoding
        smoothing_prior: float | str | None = None,  # m-estimate weight or "auto"
        random_state: int | None = None,
    ):
        super().__init__(config)
        cat_cfg = getattr(config, "categorical", None)

        def _cfg(name: str, default: Any) -> Any:
            return getattr(cat_cfg, name, default) if cat_cfg is not None else default

        self.strategies = strategies
        self.rare_threshold = (
            rare_threshold
            if rare_threshold is not None
            else getattr(config, "rare_threshold", 0.01)
        )
        self.min_count = int(min_count)
        self.top_k = top_k or getattr(config, "cat_top_k", None)
        self.hashing_dim = int(hashing_dim)
        self.n_splits = int(n_splits if n_splits is not None else _cfg("n_splits", 5))
        self.target_min_samples = int(
            target_min_samples
            if target_min_samples is not None
            else _cfg("target_min_samples", 100)
        )
        smoothing = (
            smoothing_prior
            if smoothing_prior is not None
            else _cfg("smoothing_prior", 10.0)
        )
        self.smoothing_prior: float | str = (
            "auto" if str(smoothing).lower() == "auto" else float(smoothing)
        )
        self.random_state = int(
            random_state
            if random_state is not None
            else getattr(config, "random_state", 42)
        )
        self.use_stratified_kfold = bool(
            getattr(config, "cat_use_stratified_kfold", True)
        )
        self.target_noise_std = float(getattr(config, "cat_target_noise_std", 0.0))
        self.fold_strategy = str(getattr(config, "cat_fold_strategy", "auto")).lower()
        self.group_key = getattr(config, "cat_group_key", None)
        self.time_col = getattr(config, "cat_time_col", None)

        self.categorical_cols_: list[str] = []
        self.col_info_: dict[str, dict[str, Any]] = {}  # per-col strategy+artifacts
        self.is_fitted = False

    # -------------------------- utils --------------------------

    @staticmethod
    def _as_str_series(s: pd.Series) -> pd.Series:
        s_obj = s.astype("object").where(s.notna(), "MISSING")
        return s_obj.astype(str).replace({"": "MISSING"})

    @staticmethod
    def _is_classification_target(y: pd.Series) -> bool:
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            return True
        # heuristic for numeric target with few distinct labels
        n = max(1, len(y))
        k = y.nunique(dropna=True)
        return (k <= 20) or (k / n < 0.05)

    @staticmethod
    def _to_numeric_target(y: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(y):
            return pd.to_numeric(y, errors="coerce")
        y_codes = y.astype("category").cat.codes.astype(float)
        y_codes = y_codes.where(y_codes >= 0, np.nan)
        return y_codes

    def _apply_rare_policy(self, s: pd.Series) -> tuple[pd.Series, list[str]]:
        vc = s.value_counts(dropna=False)
        freq = vc / vc.sum()
        rare = set()

        # threshold by relative freq and absolute count
        rare |= set(freq[freq < self.rare_threshold].index)
        rare |= set(vc[vc < self.min_count].index)

        # top-k cap if requested
        if self.top_k is not None and self.top_k > 0 and len(vc) > self.top_k:
            keep = set(vc.nlargest(self.top_k).index)
            rare |= set(vc.index.difference(keep))

        if len(vc) - len(rare) < 2:
            rare = set()  # avoid collapsing almost all to OTHER

        if rare:
            s = s.where(~s.isin(rare), "OTHER")

        return s, sorted(map(str, rare))

    def _choose_auto_strategy(self, s: pd.Series, y: pd.Series | None) -> str:
        k = s.nunique(dropna=False)
        n = len(s)
        te_threshold = int(getattr(self.config, "target_encode_threshold", 10))
        backend = str(getattr(self.config, "backend", "auto")).lower()
        tree_onehot_max = int(getattr(self.config, "cat_tree_onehot_max_categories", 8))
        tree_ordinal_max = int(
            getattr(self.config, "cat_tree_ordinal_max_categories", 255)
        )

        if backend in {"tree", "gbdt"}:
            if k <= tree_onehot_max:
                return "onehot"
            if k <= tree_ordinal_max:
                return "ordinal"
            if k <= 5000:
                return "freq"
            return "hashing"

        # If target is available and enough samples + medium/high cardinality,
        # consider target encoding.
        if y is not None and n >= self.target_min_samples and te_threshold < k < 10000:
            try:
                y_num = self._to_numeric_target(pd.Series(y))
                y_median = y_num.median()
                y_fill = y_num.fillna(0.0 if pd.isna(y_median) else y_median)
                if self._is_classification_target(pd.Series(y)):  # classification
                    score = chi2(
                        pd.get_dummies(
                            s, drop_first=True, sparse=False, dtype=np.uint8
                        ),
                        y_fill,
                    )[0].mean()
                else:  # regression
                    # quick Ordinal for score proxy (safe for screening only)
                    oe = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    Xc = oe.fit_transform(s.to_frame())
                    score = f_regression(Xc, y_fill.values)[0].mean()
                if np.isfinite(score) and score > 0:
                    if getattr(self.config, "use_loo", False):
                        return "loo"
                    if getattr(self.config, "use_james_stein", False):
                        return "james-stein"
                    return "target_kfold"
            except Exception:
                pass

        # Low cardinality → onehot
        if k <= 12:
            return "onehot"
        # Medium → frequency
        if k <= 1000:
            return "freq"
        # Very high → hashing
        return "hashing"

    def _resolve_fold_strategy(self, X: pd.DataFrame) -> str:
        strategy = self.fold_strategy
        if strategy not in {"auto", "kfold", "group", "time"}:
            strategy = "auto"

        if strategy == "auto":
            if self.time_col and self.time_col in X.columns:
                return "time"
            if self.group_key and self.group_key in X.columns:
                return "group"
            return "kfold"

        if strategy == "group" and not (self.group_key and self.group_key in X.columns):
            return "kfold"
        if strategy == "time" and not (self.time_col and self.time_col in X.columns):
            return "kfold"
        return strategy

    def _iter_target_folds(
        self,
        X: pd.DataFrame,
        s_valid: pd.Series,
        y_valid: pd.Series,
        is_cls: bool,
    ):
        n_valid = int(len(s_valid))
        n_splits = min(self.n_splits, max(2, int(n_valid // 2)))
        strategy = self._resolve_fold_strategy(X.loc[s_valid.index])

        if strategy == "time":
            time_values = pd.to_datetime(
                X.loc[s_valid.index, self.time_col], errors="coerce", utc=True
            )
            valid_mask = time_values.notna()
            if int(valid_mask.sum()) < 3:
                return []

            order = np.argsort(time_values[valid_mask].astype("int64").to_numpy())
            ordered_index = time_values[valid_mask].index[order]
            ordered_pos = np.arange(len(ordered_index))
            split_n = min(n_splits, max(2, len(ordered_index) - 1))
            splitter = TimeSeriesSplit(n_splits=split_n)
            return [
                (
                    ordered_index.take(tr_pos).to_numpy(),
                    ordered_index.take(te_pos).to_numpy(),
                )
                for tr_pos, te_pos in splitter.split(ordered_pos)
            ]

        if strategy == "group":
            groups = pd.Series(X.loc[s_valid.index, self.group_key]).astype(str)
            unique_groups = groups.nunique(dropna=False)
            if unique_groups < 2:
                return []
            split_n = min(n_splits, int(unique_groups))
            splitter = GroupKFold(n_splits=split_n)
            return [
                (
                    s_valid.index.take(tr_idx).to_numpy(),
                    s_valid.index.take(te_idx).to_numpy(),
                )
                for tr_idx, te_idx in splitter.split(s_valid, y_valid, groups=groups)
            ]

        if self.use_stratified_kfold and is_cls:
            cls_counts = y_valid.value_counts()
            max_splits = int(cls_counts.min()) if not cls_counts.empty else 2
            n_splits = min(n_splits, max(2, max_splits))
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            return [
                (
                    s_valid.index.take(tr_idx).to_numpy(),
                    s_valid.index.take(te_idx).to_numpy(),
                )
                for tr_idx, te_idx in splitter.split(s_valid, y_valid)
            ]

        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        return [
            (
                s_valid.index.take(tr_idx).to_numpy(),
                s_valid.index.take(te_idx).to_numpy(),
            )
            for tr_idx, te_idx in splitter.split(s_valid)
        ]

    # -------------------------- fit --------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.categorical_cols_ = cats
        self.col_info_.clear()

        for col in cats:
            s = self._as_str_series(X[col])
            s, rare_list = self._apply_rare_policy(s)
            n_unique = s.nunique(dropna=False)

            # pick strategy
            if "auto" in self.strategies:
                strategy = self._choose_auto_strategy(s, y)
            else:
                strategy = self.strategies[0]

            info = {"strategy": strategy, "rare": rare_list}

            if strategy == "onehot":
                enc = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,  # sklearn >=1.2; if older, use sparse=False
                    dtype=np.uint8,
                )
                enc.fit(s.to_frame())
                feat_names = enc.get_feature_names_out([col]).tolist()
                info.update(encoder=enc, feature_names=feat_names)

            elif strategy == "ordinal":
                # more robust than LabelEncoder for unseen categories
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.int64,
                )
                enc.fit(s.to_frame())
                cats_known = [list(map(str, c)) for c in enc.categories_]
                info.update(
                    encoder=enc, categories=cats_known, feature_names=[f"{col}_ord"]
                )

            elif strategy == "freq":
                vc = s.value_counts(normalize=True, dropna=False)
                mapping = vc.to_dict()
                info.update(mapping=mapping, feature_names=[f"{col}_freq"])

            elif strategy == "hashing":
                # Stable column names for hashed dims
                n_feat = min(self.hashing_dim, max(2, n_unique))
                enc = FeatureHasher(
                    n_features=n_feat, input_type="string", alternate_sign=False
                )
                info.update(
                    encoder=enc,
                    n_features=n_feat,
                    feature_names=[f"{col}_hash_{i}" for i in range(n_feat)],
                )

            elif strategy in self._TARGET_STRATEGIES and y is not None:
                y_full = pd.Series(y).reindex(X.index)
                classes = self._multiclass_labels(y_full)
                targets = self._target_matrix(y_full, classes)

                feat_suffix = {
                    "target_kfold": "te",
                    "loo": "loo",
                    "james-stein": "js",
                }.get(strategy, "enc")
                target_suffixes = (
                    [f"_{c}" for c in classes] if classes is not None else [""]
                )

                # Full-data statistics define the inference-time encoding.
                priors, enc_maps = {}, {}
                for t_suffix, t_col in zip(target_suffixes, targets.columns):
                    prior, enc_map = self._fit_encoding(
                        s, targets[t_col], strategy
                    )
                    priors[t_suffix] = prior
                    enc_maps[t_suffix] = enc_map

                info.update(
                    classes=classes,
                    target_suffixes=target_suffixes,
                    priors=priors,
                    enc_maps=enc_maps,
                    feature_names=[
                        f"{col}_{feat_suffix}{t}" for t in target_suffixes
                    ],
                )
                info["is_classification_target"] = self._is_classification_target(
                    pd.Series(y)
                )
            else:
                # fallback (also used when a target strategy is requested without y)
                vc = s.value_counts(normalize=True, dropna=False)
                mapping = vc.to_dict()
                info.update(
                    strategy="freq", mapping=mapping, feature_names=[f"{col}_freq"]
                )

            self.col_info_[col] = info

        self.is_fitted = True
        return self

    # -------------------------- transform --------------------------

    def _multiclass_labels(self, y: pd.Series) -> list[Any] | None:
        """Class labels needing one-vs-rest encoding, or None for a single column.

        Binary and regression targets are encoded through their numeric mean.
        Multiclass means >2 labels and either non-numeric labels or an explicit
        ``task="classification"`` (numeric targets with few distinct values are
        often ordinal/regression, where the mean is meaningful).
        """
        y_valid = y.dropna()
        labels = pd.unique(y_valid)
        if len(labels) <= 2:
            return None
        explicit_cls = (
            str(getattr(self.config, "task", "regression")).lower() == "classification"
        )
        if explicit_cls or not pd.api.types.is_numeric_dtype(y_valid):
            try:
                return sorted(labels.tolist())
            except TypeError:
                return list(labels)
        return None

    def _target_matrix(self, y: pd.Series, classes: list[Any] | None) -> pd.DataFrame:
        """Numeric target columns: one per class (one-vs-rest) or the target itself."""
        if classes is None:
            return self._to_numeric_target(y).to_frame("y")
        valid = y.notna()
        return pd.DataFrame(
            {
                f"y{i}": (y == c).astype(float).where(valid, np.nan)
                for i, c in enumerate(classes)
            },
            index=y.index,
        )

    def _shrunk_means(
        self,
        stats: pd.DataFrame,
        prior: float,
        y_var: float,
        strategy: str,
    ) -> pd.Series:
        """Per-category encoding from (mean, var, count) statistics.

        - target_kfold: m-estimate ``(n*mean + a*prior) / (n + a)``, or
          empirical-Bayes when ``smoothing_prior == "auto"``
        - james-stein: empirical-Bayes ``lam = n*var_y / (n*var_y + var_cat)``
        - loo: unsmoothed mean (applied out-of-fold only)
        """
        n = stats["count"].astype(float)
        mean = stats["mean"].astype(float)

        if strategy == "loo":
            return mean

        use_eb = strategy == "james-stein" or self.smoothing_prior == "auto"
        if use_eb:
            if not np.isfinite(y_var) or y_var <= 0:
                return pd.Series(prior, index=stats.index, dtype=float)
            var_cat = stats["var"].astype(float)
            # Singletons have no within-category variance estimate; use the
            # pooled within-category variance instead of trusting them fully.
            multi = n > 1
            if multi.any():
                pooled = float(
                    ((n[multi] - 1) * var_cat[multi]).sum() / (n[multi] - 1).sum()
                )
            else:
                pooled = y_var
            var_cat = var_cat.where(multi & var_cat.notna(), pooled)
            lam = (n * y_var) / (n * y_var + var_cat)
            lam = lam.fillna(1.0).clip(0.0, 1.0)
            return lam * mean + (1.0 - lam) * prior

        alpha = float(self.smoothing_prior)
        return (n * mean + alpha * prior) / (n + alpha)

    def _fit_encoding(
        self, s: pd.Series, y_num: pd.Series, strategy: str
    ) -> tuple[float, dict[Any, float]]:
        """Return (prior, category -> encoding) fitted on the given rows."""
        valid = y_num.notna()
        prior = float(y_num[valid].mean()) if valid.any() else 0.0
        if not np.isfinite(prior):
            prior = 0.0
        if not valid.any():
            return prior, {}
        y_var = float(y_num[valid].var(ddof=0))
        stats = (
            pd.DataFrame({"cat": s[valid], "y": y_num[valid]})
            .groupby("cat")["y"]
            .agg(["mean", "var", "count"])
        )
        return prior, self._shrunk_means(stats, prior, y_var, strategy).to_dict()

    def _target_encode_transform(
        self,
        X: pd.DataFrame,
        s: pd.Series,
        y: pd.Series | None,
        info: dict[str, Any],
        index: pd.Index,
    ) -> pd.DataFrame:
        """Out-of-fold encoding when y is given (train), fitted maps otherwise."""
        strategy = info["strategy"]
        names = info["feature_names"]
        suffixes = info["target_suffixes"]

        if y is None:
            return pd.DataFrame(
                {
                    name: s.map(info["enc_maps"][t]).astype(float).fillna(
                        info["priors"][t]
                    ).to_numpy()
                    for name, t in zip(names, suffixes)
                },
                index=index,
            )

        y_full = pd.Series(y).reindex(index)
        targets = self._target_matrix(y_full, info.get("classes"))
        valid = targets.notna().all(axis=1)
        is_cls = bool(info.get("is_classification_target", False))

        fold_pairs = []
        if valid.sum() >= 4:
            fold_pairs = self._iter_target_folds(
                X, s.loc[valid], y_full.loc[valid], is_cls
            )

        out: dict[str, pd.Series] = {}
        for name, t, t_col in zip(names, suffixes, targets.columns):
            y_num = targets[t_col]
            prior = float(y_num[valid].mean()) if valid.any() else info["priors"][t]
            col_out = pd.Series(prior, index=index, dtype=float)
            for tr_keys, te_keys in fold_pairs:
                fold_prior, enc_map = self._fit_encoding(
                    s.loc[tr_keys], y_num.loc[tr_keys], strategy
                )
                te_index = pd.Index(te_keys)
                col_out.loc[te_index] = (
                    s.loc[te_index].map(enc_map).astype(float).fillna(fold_prior).values
                )
            if self.target_noise_std > 0:
                rng = np.random.RandomState(self.random_state)
                col_out = col_out + rng.normal(0.0, self.target_noise_std, len(col_out))
            out[name] = col_out.fillna(prior)

        return pd.DataFrame(out, index=index)

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        if not self.categorical_cols_:
            return pd.DataFrame(index=X.index)

        outputs = []
        for col in self.categorical_cols_:
            info = self.col_info_.get(col)
            if info is None:
                continue

            s = self._as_str_series(X[col])

            # replicate rare policy from fit (ANY category in `rare` -> OTHER)
            rare = set(info.get("rare", []))
            if rare:
                s = s.where(~s.isin(rare), "OTHER")

            strategy = info["strategy"]

            if strategy == "onehot":
                enc = info["encoder"]
                arr = enc.transform(s.to_frame())
                df = pd.DataFrame(arr, columns=info["feature_names"], index=X.index)
                outputs.append(df.astype(np.uint8))

            elif strategy == "ordinal":
                enc = info["encoder"]
                arr = enc.transform(s.to_frame())
                outputs.append(
                    pd.DataFrame(
                        {info["feature_names"][0]: arr.ravel()}, index=X.index
                    ).astype(np.int64)
                )

            elif strategy == "freq":
                mapping = info["mapping"]
                vals = s.map(mapping).fillna(0.0).astype(float)
                outputs.append(
                    pd.DataFrame({info["feature_names"][0]: vals}, index=X.index)
                )

            elif strategy == "hashing":
                enc = info["encoder"]
                mat = enc.transform(s.tolist())  # input_type="string"
                # densify with fixed column names
                df = pd.DataFrame(
                    mat.toarray(), columns=info["feature_names"], index=X.index
                )
                outputs.append(df)

            elif strategy in self._TARGET_STRATEGIES:
                df = self._target_encode_transform(X, s, y, info, X.index)
                outputs.append(df.astype(float))

            else:  # fallback to frequency
                mapping = info.get("mapping", {})
                vals = s.map(mapping).fillna(0.0).astype(float)
                outputs.append(pd.DataFrame({f"{col}_freq": vals}, index=X.index))

        return pd.concat(outputs, axis=1) if outputs else pd.DataFrame(index=X.index)
