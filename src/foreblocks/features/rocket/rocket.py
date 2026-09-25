"""ROCKET: random convolutional kernel transform (Dempster et al., 2020).

https://arxiv.org/abs/1910.13051
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from foreblocks.features._validation import as_panel
from foreblocks.features.rocket import _kernels


class Rocket(TransformerMixin, BaseEstimator):
    """Random dilated kernels pooled to (PPV, max) per kernel.

    Each kernel draws length from {7, 9, 11}, mean-centered Normal(0, 1)
    weights, a Uniform(-1, 1) bias, a log-uniform dilation and, with
    probability 1/2, "same" padding. Multivariate input `[N, C, L]` draws a
    random channel subset per kernel. The transform is data-independent, so
    `fit` only reads the input's channel count and length.

    Output: `[N, 2 * num_kernels]`, interleaved (PPV, max) per kernel.
    """

    def __init__(self, num_kernels: int = 10_000, seed: int | None = None):
        self.num_kernels = num_kernels
        self.seed = seed

    def fit(self, X, y=None) -> Rocket:
        x = as_panel(X, allow_empty=True)
        _, n_channels, length = x.shape
        rng = np.random.default_rng(self.seed)
        lengths = rng.choice(np.array([7, 9, 11], dtype=np.int32), self.num_kernels)
        channel_counts = np.array(
            [
                int(2 ** rng.uniform(0, np.log2(min(n_channels, k) + 1)))
                for k in lengths
            ],
            dtype=np.int64,
        )
        weights, channels = [], []
        biases = rng.uniform(-1, 1, self.num_kernels).astype(np.float32)
        dilations = np.empty(self.num_kernels, dtype=np.int64)
        paddings = np.empty(self.num_kernels, dtype=np.int64)
        for k, (klen, count) in enumerate(zip(lengths, channel_counts)):
            w = rng.normal(0, 1, (count, klen))
            weights.append((w - w.mean(axis=1, keepdims=True)).ravel())
            channels.append(rng.choice(n_channels, count, replace=False))
            exponent = max(np.log2((length - 1) / (klen - 1)), 0.0)
            dilations[k] = max(int(2 ** rng.uniform(0, exponent)), 1)
            span = (klen - 1) * dilations[k]
            # Series shorter than the kernel span keep "same" padding so every
            # kernel has at least one output position.
            pad = rng.integers(2) == 1 or span >= length
            paddings[k] = span // 2 if pad else 0

        self.lengths_ = lengths.astype(np.int64)
        self.weights_ = np.concatenate(weights).astype(np.float32)
        self.weight_offsets_ = np.concatenate([[0], np.cumsum(lengths * channel_counts)])
        self.biases_ = biases
        self.dilations_ = dilations
        self.paddings_ = paddings
        self.channel_indices_ = np.concatenate(channels).astype(np.int64)
        self.channel_offsets_ = np.concatenate([[0], np.cumsum(channel_counts)])
        self.n_channels_in_ = n_channels
        self.n_features_out_ = 2 * self.num_kernels
        return self

    @property
    def kernel_groups_(self) -> np.ndarray:
        """Kernel id of each output feature (PPV and max share a kernel)."""
        return np.repeat(np.arange(len(self.lengths_)), 2)

    def select_kernels(self, kernels) -> Rocket:
        """Fitted copy that only computes `kernels` (ids from `kernel_groups_`),
        in the original column order."""
        keep = np.unique(np.asarray(kernels, dtype=np.int64))
        pruned = copy.copy(self)
        weight_sizes = np.diff(self.weight_offsets_)
        channel_counts = np.diff(self.channel_offsets_)
        pruned.weights_ = np.concatenate(
            [self.weights_[self.weight_offsets_[k] : self.weight_offsets_[k + 1]] for k in keep]
        ).astype(np.float32)
        pruned.weight_offsets_ = np.concatenate([[0], np.cumsum(weight_sizes[keep])])
        pruned.channel_indices_ = np.concatenate(
            [self.channel_indices_[self.channel_offsets_[k] : self.channel_offsets_[k + 1]] for k in keep]
        ).astype(np.int64)
        pruned.channel_offsets_ = np.concatenate([[0], np.cumsum(channel_counts[keep])])
        pruned.lengths_ = self.lengths_[keep]
        pruned.biases_ = self.biases_[keep]
        pruned.dilations_ = self.dilations_[keep]
        pruned.paddings_ = self.paddings_[keep]
        pruned.n_features_out_ = 2 * len(keep)
        return pruned

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("Call fit before transform")
        x = as_panel(X, allow_empty=True)
        if x.shape[1] != self.n_channels_in_:
            raise ValueError(f"Expected {self.n_channels_in_} channels, got {x.shape[1]}.")
        return _kernels.rocket_transform(
            x,
            self.weights_,
            self.weight_offsets_,
            self.lengths_,
            self.biases_,
            self.dilations_,
            self.paddings_,
            self.channel_offsets_,
            self.channel_indices_,
        )
