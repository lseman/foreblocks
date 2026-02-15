from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import chi2, f_regression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .base import BaseFeatureTransformer


class CategoricalTransformer(BaseFeatureTransformer):
    """
    Modern categorical transformer:
      - Strategies: 'auto', 'onehot', 'freq', 'hashing', 'target_kfold', 'ordinal', 'loo', 'james-stein'
      - Leakage-safe target encoding (k-fold, with prior smoothing)
      - Leave-One-Out (LOO) encoding for small-leakage high cardinality
      - James-Stein shrinkage encoding
      - Robust rare handling: frequency threshold, min count, top-k cap
      - Stable output schema across fit/transform
    """

    def __init__(
        self,
        config,
        strategies: Tuple[str, ...] = ("auto",),
        rare_threshold: Optional[
            float
        ] = None,  # fraction (0..1); falls back to config.rare_threshold
        min_count: int = 1,  # absolute count for rare categories
        top_k: Optional[int] = None,  # keep top_k most frequent; others -> OTHER
        hashing_dim: int = 64,
        n_splits: int = 5,
        target_min_samples: int = 100,  # only consider target encoding if enough data
        smoothing_prior: float = 10.0,  # target encoding prior weight
        random_state: int = 42,
    ):
        super().__init__(config)
        self.strategies = strategies
        self.rare_threshold = (
            rare_threshold
            if rare_threshold is not None
            else getattr(config, "rare_threshold", 0.01)
        )
        self.min_count = int(min_count)
        self.top_k = top_k or getattr(config, "cat_top_k", None)
        self.hashing_dim = int(hashing_dim)
        self.n_splits = int(n_splits)
        self.target_min_samples = int(target_min_samples)
        self.smoothing_prior = float(smoothing_prior)
        self.random_state = int(random_state)
        self.use_stratified_kfold = bool(
            getattr(config, "cat_use_stratified_kfold", True)
        )
        self.target_noise_std = float(getattr(config, "cat_target_noise_std", 0.0))

        self.categorical_cols_: List[str] = []
        self.col_info_: Dict[str, Dict[str, Any]] = {}  # per-col strategy+artifacts
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

    def _apply_rare_policy(self, s: pd.Series) -> Tuple[pd.Series, List[str]]:
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

    def _choose_auto_strategy(self, s: pd.Series, y: Optional[pd.Series]) -> str:
        k = s.nunique(dropna=False)
        n = len(s)
        te_threshold = int(getattr(self.config, "target_encode_threshold", 10))

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
                    if getattr(self.config, 'use_loo', False):
                        return "loo"
                    if getattr(self.config, 'use_james_stein', False):
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

    # -------------------------- fit --------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
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

            elif strategy in ["target_kfold", "loo", "james-stein"] and y is not None:
                # store per-category stats for smoothing + kfold plan
                y_series = self._to_numeric_target(pd.Series(y).reindex(X.index))
                global_mean = float(np.nanmean(y_series.values))
                # Fit-time we only keep category means and counts
                grp = pd.DataFrame({"cat": s, "y": y_series}).groupby("cat")["y"]
                cat_mean = grp.mean().to_dict()
                cat_count = grp.size().to_dict()
                
                feat_suffix = {
                    "target_kfold": "te",
                    "loo": "loo",
                    "james-stein": "js"
                }.get(strategy, "enc")
                
                info.update(
                    global_mean=global_mean,
                    cat_mean=cat_mean, 
                    cat_count=cat_count, 
                    feature_names=[f"{col}_{feat_suffix}"]
                )
                info["is_classification_target"] = self._is_classification_target(
                    pd.Series(y)
                )
            else:
                # fallback
                vc = s.value_counts(normalize=True, dropna=False)
                mapping = vc.to_dict()
                info.update(mapping=mapping, feature_names=[f"{col}_freq"])

            self.col_info_[col] = info

        self.is_fitted = True
        return self

    # -------------------------- transform --------------------------

    def _target_kfold_transform(
        self,
        s: pd.Series,
        y: Optional[pd.Series],
        info: Dict[str, Any],
        index: pd.Index,
    ):
        # If y is provided at transform (train), do KFold out-of-fold encoding.
        # If not (inference), use smoothed mean from fit stats.
        name = info["feature_names"][0]
        prior = info["global_mean"]
        alpha = self.smoothing_prior

        if y is not None:
            y_series = self._to_numeric_target(pd.Series(y).reindex(index))
            valid = y_series.notna()
            out = pd.Series(prior, index=index, dtype=float)
            if valid.sum() < 4:
                return pd.DataFrame({name: out.values}, index=index)

            s_valid = s.loc[valid]
            y_valid = y_series.loc[valid]
            n_splits = min(self.n_splits, max(2, int(valid.sum() // 2)))
            is_cls = bool(info.get("is_classification_target", False))

            if self.use_stratified_kfold and is_cls:
                cls_counts = y_valid.value_counts()
                max_splits = int(cls_counts.min()) if not cls_counts.empty else 2
                n_splits = min(n_splits, max(2, max_splits))
                splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                split_iter = splitter.split(s_valid, y_valid)
            else:
                splitter = KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                split_iter = splitter.split(s_valid)

            out = pd.Series(index=index, dtype=float)
            for tr_idx, te_idx in split_iter:
                tr_c = s_valid.iloc[tr_idx]
                tr_y = y_valid.iloc[tr_idx]
                stats = (
                    pd.DataFrame({"cat": tr_c, "y": tr_y})
                    .groupby("cat")["y"]
                    .agg(["mean", "count"])
                )
                # smoothing
                smoothed = (stats["count"] * stats["mean"] + alpha * prior) / (
                    stats["count"] + alpha
                )
                enc_map = smoothed.to_dict()
                te_index = s_valid.index[te_idx]
                out.loc[te_index] = (
                    s_valid.iloc[te_idx].map(enc_map).fillna(prior).values
                )
            if self.target_noise_std > 0:
                rng = np.random.RandomState(self.random_state)
                noise = rng.normal(0.0, self.target_noise_std, size=len(out))
                out = out + noise
            out = out.fillna(prior)
            return pd.DataFrame({name: out.values}, index=index)
        else:
            # test-time: use global fit stats with smoothing
            cat_mean = info["cat_mean"]
            cat_count = info["cat_count"]
            vals = []
            for v in s.values:
                m = cat_mean.get(v, prior)
                c = cat_count.get(v, 0)
                vals.append((c * m + alpha * prior) / (c + alpha))
            return pd.DataFrame({name: vals}, index=index)

    def _loo_transform(
        self,
        s: pd.Series,
        y: Optional[pd.Series],
        info: Dict[str, Any],
        index: pd.Index,
    ):
        """Leave-One-Out encoding: (TargetSum - CurrentY) / (Count - 1)"""
        name = info["feature_names"][0]
        prior = info["global_mean"]
        
        if y is not None:
            y_series = self._to_numeric_target(pd.Series(y).reindex(index))
            # Calculate counts and sums per category
            cat_stats = pd.DataFrame({'cat': s, 'y': y_series}).groupby('cat')['y'].agg(['sum', 'count'])
            
            # Map stats back to rows
            row_sum = s.map(cat_stats['sum']).fillna(0)
            row_count = s.map(cat_stats['count']).fillna(0)
            
            # (TotalSum - SelfY) / (TotalCount - 1)
            # Avoid division by zero
            loo = (row_sum - y_series.fillna(0)) / (row_count - 1).replace(0, 1)
            # Fill cases where count was 1 with prior
            loo = loo.where(row_count > 1, prior)
            
            # Add noise for regularization
            if self.target_noise_std > 0:
                rng = np.random.RandomState(self.random_state)
                loo += rng.normal(0, self.target_noise_std, size=len(loo))
            
            return pd.DataFrame({name: loo.values}, index=index)
        else:
            # Inference: use pre-calculated means
            cat_mean = info["cat_mean"]
            return pd.DataFrame({name: s.map(cat_mean).fillna(prior).values}, index=index)

    def _james_stein_transform(
        self,
        s: pd.Series,
        y: Optional[pd.Series],
        info: Dict[str, Any],
        index: pd.Index,
    ):
        """James-Stein encoder: Shrinkage toward the global mean."""
        name = info["feature_names"][0]
        prior = info["global_mean"]
        
        # Calculate stats
        if y is not None:
            y_series = self._to_numeric_target(pd.Series(y).reindex(index))
            global_var = y_series.var()
            if global_var == 0 or np.isnan(global_var):
                return pd.DataFrame({name: [prior]*len(index)}, index=index)
                
            stats = pd.DataFrame({'cat': s, 'y': y_series}).groupby('cat')['y'].agg(['mean', 'var', 'count'])
            
            # Weight = 1 - (var_within / (var_within + var_between))
            # Simplified JS: Weight = 1 - (pooled_var / (pooled_var + category_var * count))
            # We use a common version: B = VarAcrossCategories / (VarAcrossCategories + VarWithinCategory / Count)
            
            # Map means and counts to rows
            row_mean = s.map(stats['mean']).fillna(prior)
            row_var = s.map(stats['var']).fillna(0)
            row_count = s.map(stats['count']).fillna(0)
            
            # Shrinkage factor
            B = global_var / (global_var + (row_var / row_count.replace(0, 1)))
            B = B.clip(0, 1) # Ensure weight is between 0 and 1
            
            js = (1 - B) * prior + B * row_mean
            return pd.DataFrame({name: js.values}, index=index)
        else:
            # Simplified inference for JS
            cat_mean = info["cat_mean"]
            return pd.DataFrame({name: s.map(cat_mean).fillna(prior).values}, index=index)

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
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

            elif strategy == "target_kfold":
                df = self._target_kfold_transform(s, y, info, X.index)
                outputs.append(df.astype(float))

            elif strategy == "loo":
                df = self._loo_transform(s, y, info, X.index)
                outputs.append(df.astype(float))

            elif strategy == "james-stein":
                df = self._james_stein_transform(s, y, info, X.index)
                outputs.append(df.astype(float))

            else:  # fallback to frequency
                mapping = info.get("mapping", {})
                vals = s.map(mapping).fillna(0.0).astype(float)
                outputs.append(pd.DataFrame({f"{col}_freq": vals}, index=X.index))

        return pd.concat(outputs, axis=1) if outputs else pd.DataFrame(index=X.index)
