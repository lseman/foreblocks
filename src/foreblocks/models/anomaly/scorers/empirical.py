"""Fitted NumPy ECOD, COPOD and histogram-based outlier scorers.

Inputs are feature matrices [N, D] or windows [N, T, D]. Windows are
flattened, preserving each lag/channel as a separate marginal. Larger scores
mean more anomalous. Prediction uses only training distributions, so results
are independent of query batch composition (unlike transductive variants).

References:
    ECOD: https://arxiv.org/abs/2201.00382
    COPOD: https://arxiv.org/abs/2009.09463
    HBOS: Goldstein and Dengel, KI 2012, Histogram-based Outlier Score.
"""

from __future__ import annotations

import numpy as np


class _EmpiricalScorer:
    """Shared shape validation and scalar score aggregation."""

    def _matrix(self, values: np.ndarray, *, fitting: bool = False) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim not in (2, 3) or any(size == 0 for size in x.shape[1:]):
            raise ValueError("Expected [N, D] features or [N, T, D] windows.")
        if not np.isfinite(x).all():
            raise ValueError("Input must contain only finite values.")
        if fitting:
            if len(x) == 0:
                raise ValueError("At least one training sample is required.")
            self.sample_shape_ = x.shape[1:]
        elif not hasattr(self, "sample_shape_"):
            raise RuntimeError("Scorer is not fitted.")
        elif x.shape[1:] != self.sample_shape_:
            raise ValueError(
                f"Expected sample shape {self.sample_shape_}, got {x.shape[1:]}."
            )
        return x.reshape(len(x), int(np.prod(x.shape[1:])))

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        """Return one score per sample using the fitted reference distribution."""
        return self.feature_scores(values).sum(axis=1)


class ECOD(_EmpiricalScorer):
    """Empirical CDF outlier detection with skew-aware, two-sided tails.

    Tail probabilities include ties and are floored at 1 / (n_train + 1)
    for unseen extremes. Constant marginals score zero at their fitted value.
    ``feature_scores`` exposes each flattened feature's score contribution.
    """

    def fit(self, values: np.ndarray) -> ECOD:
        x = self._matrix(values, fitting=True)
        self.sorted_ = np.sort(x, axis=0)
        # Rescale before taking moments to avoid overflow for large inputs.
        scale = np.maximum(np.max(np.abs(x), axis=0), 1.0)
        centered = x / scale - (x / scale).mean(axis=0)
        self.skew_sign_ = np.sign(np.mean(centered**3, axis=0))
        self.decision_scores_ = self.decision_function(values)
        return self

    def _tails(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.sorted_)
        left = np.empty_like(x)
        right = np.empty_like(x)
        floor = 1.0 / (n + 1)
        for j in range(x.shape[1]):
            reference = self.sorted_[:, j]
            left[:, j] = np.searchsorted(reference, x[:, j], side="right") / n
            right[:, j] = (n - np.searchsorted(reference, x[:, j], side="left")) / n
        return -np.log(np.maximum(left, floor)), -np.log(np.maximum(right, floor))

    def feature_scores(self, values: np.ndarray) -> np.ndarray:
        """Return [N, flattened_features] nonnegative score contributions."""
        left, right = self._tails(self._matrix(values))
        skew_tail = left * (self.skew_sign_ <= 0) + right * (self.skew_sign_ >= 0)
        return np.maximum(np.maximum(left, right), skew_tail)


class COPOD(ECOD):
    """Empirical copula outlier scores with skew-aware tail selection.

    Uses the maximum of the mean two-sided surprisal and the skew-selected
    tail per marginal, then sums across marginals. The fitted training
    distribution is held fixed for unseen samples.
    """

    def feature_scores(self, values: np.ndarray) -> np.ndarray:
        left, right = self._tails(self._matrix(values))
        skew_tail = left * (self.skew_sign_ < 0) + right * (self.skew_sign_ >= 0)
        return np.maximum((left + right) / 2.0, skew_tail)


class HBOS(_EmpiricalScorer):
    """Histogram-based outlier scores with independent equal-width marginals.

    Densities are normalized by each marginal's peak and regularized by
    ``alpha`` before taking negative logs. Outside the training range, a
    marginal has zero density. Constant marginals use exact-value matching.
    This explicit boundary policy differs from HBOS variants with tolerance
    bands around the outer histogram bins.
    """

    def __init__(self, n_bins: int = 10, alpha: float = 0.1) -> None:
        if (
            isinstance(n_bins, bool)
            or not isinstance(n_bins, (int, np.integer))
            or n_bins < 2
        ):
            raise ValueError("n_bins must be an integer >= 2.")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        self.n_bins = n_bins
        self.alpha = float(alpha)

    def fit(self, values: np.ndarray) -> HBOS:
        x = self._matrix(values, fitting=True)
        self.histograms_ = []
        for col in x.T:
            if col.min() == col.max():
                self.histograms_.append((None, np.array([col[0]])))
            else:
                counts, edges = np.histogram(col, bins=self.n_bins)
                self.histograms_.append((counts / counts.max(), edges))
        self.decision_scores_ = self.decision_function(values)
        return self

    def feature_scores(self, values: np.ndarray) -> np.ndarray:
        x = self._matrix(values)
        scores = np.empty_like(x)
        for j, (density, edges) in enumerate(self.histograms_):
            if density is None:
                relative_density = (x[:, j] == edges[0]).astype(float)
            else:
                indices = np.searchsorted(edges, x[:, j], side="right") - 1
                indices = np.clip(indices, 0, len(density) - 1)
                inside = (x[:, j] >= edges[0]) & (x[:, j] <= edges[-1])
                relative_density = np.where(inside, density[indices], 0.0)
            scores[:, j] = -np.log((relative_density + self.alpha) / (1 + self.alpha))
        return scores


def ecod_score(
    windows: np.ndarray, *, reference: np.ndarray | None = None
) -> np.ndarray:
    """Score windows against ``reference``; omit it for training-set scores."""
    return (
        ECOD()
        .fit(windows if reference is None else reference)
        .decision_function(windows)
    )


def copod_score(
    windows: np.ndarray, *, reference: np.ndarray | None = None
) -> np.ndarray:
    """Score windows against ``reference``; omit it for training-set scores."""
    return (
        COPOD()
        .fit(windows if reference is None else reference)
        .decision_function(windows)
    )


def hbos_score(
    windows: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    n_bins: int = 10,
    alpha: float = 0.1,
) -> np.ndarray:
    """Score windows against a fitted histogram reference."""
    return (
        HBOS(n_bins=n_bins, alpha=alpha)
        .fit(windows if reference is None else reference)
        .decision_function(windows)
    )
