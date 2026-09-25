"""MiniRocket: (almost) deterministic random convolutional kernels
(Dempster et al., 2021).

https://arxiv.org/abs/2012.08791
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from foreblocks.features._validation import as_panel
from foreblocks.features.rocket import _kernels

_KERNELS = _kernels.NUM_MINIROCKET_KERNELS


def fit_dilations(
    input_length: int, num_features: int, max_dilations_per_kernel: int
) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced dilations and the number of biases (features) per dilation."""
    features_per_kernel = num_features // _KERNELS
    if features_per_kernel < 1:
        raise ValueError(f"num_features must be at least {_KERNELS}.")
    true_max = min(features_per_kernel, max_dilations_per_kernel)
    multiplier = features_per_kernel / true_max
    max_exponent = np.log2((input_length - 1) / 8)
    dilations, counts = np.unique(
        np.logspace(0, max_exponent, true_max, base=2).astype(np.int64),
        return_counts=True,
    )
    counts = (counts * multiplier).astype(np.int64)
    remainder = features_per_kernel - counts.sum()
    i = 0
    while remainder > 0:
        counts[i] += 1
        remainder -= 1
        i = (i + 1) % len(counts)
    return dilations, counts


def golden_quantiles(n: int) -> np.ndarray:
    """Low-discrepancy quantiles `{k * phi mod 1}`, k = 1..n."""
    return ((np.arange(1, n + 1) * ((np.sqrt(5) + 1) / 2)) % 1).astype(np.float64)


class MiniRocketParameters:
    """Fitted dilations, channel subsets and biases for one representation."""

    def __init__(self, x: np.ndarray, num_features: int, max_dilations_per_kernel: int, rng):
        n_series, n_channels, length = x.shape
        if length < 9:
            raise ValueError("MiniRocket needs series of length >= 9.")
        if n_series < 1:
            raise ValueError("MiniRocket fits biases from data; at least one series is required.")
        self.dilations, self.features_per_dilation = fit_dilations(
            length, num_features, max_dilations_per_kernel
        )
        n_combinations = len(self.dilations) * _KERNELS
        max_exponent = np.log2(min(n_channels, 9) + 1)
        counts = (2 ** rng.uniform(0, max_exponent, n_combinations)).astype(np.int64)
        self.channel_offsets = np.concatenate([[0], np.cumsum(counts)])
        self.channel_indices = np.concatenate(
            [rng.choice(n_channels, c, replace=False) for c in counts]
        ).astype(np.int64)
        self.n_channels = n_channels
        # Pruning mask over (dilation, kernel) combinations; see `select`.
        self.active = np.ones(n_combinations, dtype=np.bool_)
        n_features = _KERNELS * int(self.features_per_dilation.sum())
        self.biases = _kernels.minirocket_fit_biases(
            x,
            rng.integers(n_series, size=n_combinations),
            self.dilations,
            self.features_per_dilation,
            golden_quantiles(n_features),
            self.channel_offsets,
            self.channel_indices,
        )

    @property
    def n_combinations(self) -> int:
        return len(self.active)

    def feature_combinations(self) -> np.ndarray:
        """(dilation, kernel) combination index of each active feature, in
        output order (one pooling block)."""
        per_comb = np.repeat(self.features_per_dilation, _KERNELS)
        combs = np.repeat(np.arange(self.n_combinations), per_comb)
        return combs[self.active[combs]]

    def select(self, combinations) -> MiniRocketParameters:
        """Copy that computes only the given (dilation, kernel) combinations."""
        pruned = copy.copy(self)
        pruned.active = np.zeros_like(self.active)
        pruned.active[np.asarray(combinations, dtype=np.int64)] = True
        pruned.active &= self.active
        return pruned

    def transform(self, x: np.ndarray, n_pool: int = 1) -> np.ndarray:
        """`n_pool` blocks of the first `n_pool` operators in
        `_kernels.POOLING_OPERATORS` (PPV, MPV, MIPV, LSPV, GMP)."""
        if x.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {x.shape[1]}.")
        return _kernels.minirocket_transform(
            x,
            self.dilations,
            self.features_per_dilation,
            self.biases,
            self.channel_offsets,
            self.channel_indices,
            int(n_pool),
            self.active,
        )


class MiniRocket(TransformerMixin, BaseEstimator):
    """84 fixed kernels (weights in {-1, 2}) x log-spaced dilations, with
    biases drawn as quantiles of training convolution outputs; pooled to PPV.

    `num_features` is rounded down to a multiple of 84. Multivariate input
    `[N, C, L]` sums each kernel over a random channel subset. Requires
    `L >= 9`.

    Output: `[N, 84 * (num_features // 84)]`.
    """

    def __init__(
        self,
        num_features: int = 10_000,
        max_dilations_per_kernel: int = 32,
        seed: int | None = None,
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.seed = seed

    def fit(self, X, y=None) -> MiniRocket:
        rng = np.random.default_rng(self.seed)
        self.parameters_ = MiniRocketParameters(
            as_panel(X, min_length=9), self.num_features, self.max_dilations_per_kernel, rng
        )
        self.n_features_out_ = len(self.parameters_.biases)
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "parameters_"):
            raise RuntimeError("Call fit before transform")
        return self.parameters_.transform(as_panel(X, min_length=9, allow_empty=True))

    @property
    def kernel_groups_(self) -> np.ndarray:
        """Kernel (dilation, kernel combination) id of each output feature."""
        return self.parameters_.feature_combinations()

    def select_kernels(self, kernels) -> MiniRocket:
        """Fitted copy that only computes the features of `kernels` (ids from
        `kernel_groups_`), in the original column order."""
        pruned = copy.copy(self)
        pruned.parameters_ = self.parameters_.select(kernels)
        pruned.n_features_out_ = len(pruned.parameters_.feature_combinations())
        return pruned
