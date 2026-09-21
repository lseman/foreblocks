"""Abstract base class and shared utilities for feature transformers.

All transformers inherit from :class:`BaseFeatureTransformer` which provides:
- Consistent fit/transform API via sklearn mixin
- Safe numeric coercion and validation
- Common statistical helpers (winsorization, median imputation, variance checks)
- Fit-result caching
- Classification target detection
"""

from __future__ import annotations

import functools
import hashlib
import logging
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd

from .config import FeatureConfig

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound="BaseFeatureTransformer")


class BaseFeatureTransformer(ABC):
    """
    Abstract base class for feature transformers.

    Provides common utilities and a consistent API:
      - ``fit(X, y)`` → self
      - ``transform(X, y=None)`` → DataFrame
      - ``fit_transform(X, y)`` → DataFrame
    """

    config: FeatureConfig
    is_fitted: bool

    def __init__(self, config: Any):
        self.config = config
        self.is_fitted = False
        self.logger: logging.Logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    # ── abstract interface ──────────────────────────────────────────────

    @abstractmethod
    def fit(self: T, X: pd.DataFrame, y: pd.Series | None = None) -> T:
        """Fit the transformer. Must set self.is_fitted = True."""
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Transform data. Returns DataFrame with same index as X."""
        ...

    def fit_transform(
        self: T, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)

    # ── shared helpers ──────────────────────────────────────────────────

    @staticmethod
    def get_numerical_cols(X: pd.DataFrame) -> list[str]:
        """Return column names of numeric dtypes."""
        return X.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def get_categorical_cols(X: pd.DataFrame) -> list[str]:
        """Return column names of object/category dtypes."""
        return X.select_dtypes(include=["object", "category"]).columns.tolist()

    @staticmethod
    def coerce_numeric(s: pd.Series, dtype: type = np.float64) -> np.ndarray:
        """Safely coerce a Series to a numeric numpy array."""
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=dtype, copy=True)

    @staticmethod
    def winsorize(arr: np.ndarray, p: float = 0.001) -> np.ndarray:
        """Light symmetric winsorization. Returns copy."""
        if p <= 0:
            return arr.copy()
        a = arr.copy()
        finite = np.isfinite(a)
        if finite.sum() < 20:
            return a
        lo = np.nanpercentile(a[finite], 100 * p)
        hi = np.nanpercentile(a[finite], 100 * (1 - p))
        a[finite] = np.clip(a[finite], lo, hi)
        return a

    @staticmethod
    def median_impute(
        X: pd.DataFrame, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Return a copy with NaN filled by column medians."""
        cols = columns if columns is not None else X.columns.tolist()
        X2 = X.copy()
        medians = X2[cols].median()
        X2[cols] = X2[cols].fillna(medians)
        return X2

    @staticmethod
    def check_variance(arr: np.ndarray, min_var: float = 1e-6) -> bool:
        """Return True if array has sufficient variance."""
        finite = arr[np.isfinite(arr)]
        if finite.size < 3:
            return False
        return float(np.nanvar(finite)) >= min_var

    @staticmethod
    def is_classification_target(y: pd.Series) -> bool:
        """Heuristic: classification vs regression target."""
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            return True
        k = y.nunique(dropna=True)
        n = max(1, len(y))
        return (k <= 20) or (k / n < 0.05)

    @staticmethod
    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation with finite-mask checks. Returns 0 on failure."""
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            return 0.0
        aa, bb = a[m], b[m]
        sa, sb = aa.std(), bb.std()
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        c = float(np.corrcoef(aa, bb)[0, 1])
        return c if np.isfinite(c) else 0.0

    @staticmethod
    def _empty_df(index: pd.Index) -> pd.DataFrame:
        """Return an empty DataFrame with the given index."""
        return pd.DataFrame(index=index)

    # ── caching (optional, opt-in via decorator) ────────────────────────

    def _fit_cache_key(self, X: pd.DataFrame, y: pd.Series | None) -> str:
        """Generate a cache key based on data content."""
        hasher = hashlib.md5()
        cfg_state = getattr(self.config, "__dict__", None)
        if cfg_state is not None:
            hasher.update(str(sorted(cfg_state.items())).encode())
        for col in sorted(X.columns):
            if col in X.columns:
                vals = self.coerce_numeric(X[col])
                finite = vals[np.isfinite(vals)]
                if finite.size > 0:
                    hasher.update(
                        f"{col}:{finite.mean():.6f}:{finite.std():.6f}:{finite.size}".encode()
                    )
        if y is not None:
            y_vals = self.coerce_numeric(pd.Series(y))
            finite = y_vals[np.isfinite(y_vals)]
            if finite.size > 0:
                hasher.update(
                    f"y:{finite.mean():.6f}:{finite.std():.6f}:{finite.size}".encode()
                )
        return hasher.hexdigest()


def cached_fit(
    attr: str = "_fit_cache",
    key_fn: str | None = None,
):
    """Decorator that caches fit() results.

    Usage
    -----
    >>> class MyTF(BaseFeatureTransformer):
    ...     @cached_fit(attr="_fit_cache", key_fn="_fit_cache_key")
    ...     def fit(self, X, y=None):
    ...         ...

    Parameters
    ----------
    attr:
        Name of the attribute to store the cache dict (default: ``"_fit_cache"``).
    key_fn:
        Name of the method on self that produces the cache key. Defaults to
        ``"_fit_cache_key"`` which is available on ``BaseFeatureTransformer``.
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self: BaseFeatureTransformer, *args, **kwargs):
            X = args[0] if args else kwargs.get("X")
            y = args[1] if len(args) > 1 else kwargs.get("y")
            if X is None:
                return method(self, *args, **kwargs)

            cache_key = self._fit_cache_key(X, y)
            cache = getattr(self, attr, None)
            if cache is None:
                cache = {}
                setattr(self, attr, cache)

            if cache_key in cache:
                # Restore fitted state
                for k, v in cache[cache_key].items():
                    setattr(self, k, v)
                self.is_fitted = True
                return self

            # Run fit and cache result
            result = method(self, *args, **kwargs)
            if self.is_fitted:
                cache[cache_key] = {
                    k: v
                    for k, v in self.__dict__.items()
                    if not k.startswith("_fit_cache") and k not in ("logger",)
                }
            return result

        return wrapper

    return decorator


def require_fitted(method):
    """Decorator that raises ValueError if transform is called before fit."""

    @functools.wraps(method)
    def wrapper(self: BaseFeatureTransformer, *args, **kwargs):
        if not self.is_fitted:
            raise ValueError(
                f"{self.__class__.__name__} has not been fitted. Call fit() first."
            )
        return method(self, *args, **kwargs)

    return wrapper
