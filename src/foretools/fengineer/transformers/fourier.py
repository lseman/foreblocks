"""Fourier feature transformer for periodic patterns."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import fft

from .aux import BaseFeatureTransformer, require_fitted

if TYPE_CHECKING:
    pass


class FourierTransformer(BaseFeatureTransformer):
    """Creates Fourier features for periodic patterns in numerical columns."""

    _TEMPORAL_TOKENS = (
        "time",
        "date",
        "day",
        "week",
        "month",
        "quarter",
        "year",
        "hour",
        "minute",
        "second",
        "season",
        "elapsed",
    )

    @classmethod
    def _looks_generated(cls, col: str) -> bool:
        prefixes = ("row_", "rff_", "kmeans_", "gmm_", "umap_", "hdbscan_")
        suffixes = ("_bin", "_te")
        return "__" in col or col.startswith(prefixes) or col.endswith(suffixes)

    @classmethod
    def _temporal_priority(cls, col: str) -> int:
        lowered = col.lower()
        return int(any(token in lowered for token in cls._TEMPORAL_TOKENS))

    def _select_source_columns(self, X: pd.DataFrame) -> list[str]:
        cols = self.get_numerical_cols(X)
        if not cols:
            return []

        if getattr(self.config, "fourier_exclude_generated_sources", True):
            cols = [c for c in cols if not self._looks_generated(c)]
            if not cols:
                return []

        max_cols = int(getattr(self.config, "fourier_max_source_features", 12))
        ranking = []
        for col in cols:
            vals = pd.to_numeric(X[col], errors="coerce")
            ranking.append(
                (
                    self._temporal_priority(col),
                    float(vals.var(skipna=True)),
                    col,
                )
            )
        ranking.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [col for _, _, col in ranking[: max(1, min(max_cols, len(ranking)))]]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FourierTransformer":
        self.numerical_cols_ = self._select_source_columns(X)
        self.fourier_configs_: dict[str, dict] = {}

        if not getattr(self.config, "create_fourier", False):
            self.is_fitted = True
            return self

        for col in self.numerical_cols_:
            if col not in X.columns:
                continue
            data = X[col].fillna(X[col].median())
            if data.var() < 1e-6:
                continue

            try:
                data_norm = (data - data.mean()) / (data.std() + 1e-8)
                fft_vals = fft.fft(data_norm.values)
                freqs = fft.fftfreq(len(data_norm))
                magnitude = np.abs(fft_vals)
                top_freq_idx = np.argsort(magnitude)[
                    -(getattr(self.config, "n_fourier_terms", 3) + 1) - 1 : -1
                ]
                valid_frequencies = [
                    freqs[idx] for idx in top_freq_idx if freqs[idx] != 0
                ]
                if valid_frequencies:
                    self.fourier_configs_[col] = {
                        "frequencies": valid_frequencies[
                            : getattr(self.config, "n_fourier_terms", 3)
                        ],
                        "mean": data.mean(),
                        "std": data.std(),
                    }
            except Exception as e:
                warnings.warn(f"Fourier analysis failed for {col}: {e}")

        self.is_fitted = True
        return self

    @require_fitted
    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        if (
            not getattr(self.config, "create_fourier", False)
            or not self.fourier_configs_
        ):
            return self._empty_df(X.index)

        features = {}
        for col, cfg in self.fourier_configs_.items():
            if col not in X.columns:
                continue
            data = X[col].fillna(cfg["mean"])
            for i, freq in enumerate(cfg["frequencies"]):
                features[f"{col}_fourier_cos_{i}"] = np.cos(
                    2 * np.pi * freq * np.arange(len(data))
                )
                features[f"{col}_fourier_sin_{i}"] = np.sin(
                    2 * np.pi * freq * np.arange(len(data))
                )

        return pd.DataFrame(features, index=X.index)
