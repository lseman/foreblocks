"""MultiRocket: MiniRocket kernels with multiple pooling operators over the
series and its first difference (Tan et al., 2022).

https://arxiv.org/abs/2102.00457
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from foreblocks.features._validation import as_panel
from foreblocks.features.rocket.minirocket import MiniRocketParameters

POOLING_OPERATORS = ("ppv", "mpv", "mipv", "lspv")


class MultiRocket(TransformerMixin, BaseEstimator):
    """MiniRocket kernels fitted separately on the series and on its first
    difference, each pooled with four operators over `z = conv - bias`:

    - PPV: proportion of positive values
    - MPV: mean of positive values (0 when there are none)
    - MIPV: mean index of positive values (-1 when there are none)
    - LSPV: longest stretch of consecutive positive values

    MPV follows the paper's definition (mean of positive `z`); the reference
    code accumulates `conv + bias` instead. `num_features` is the total
    target; each representation gets `num_features // 8` kernel/bias pairs,
    rounded down to a multiple of 84. Requires `L >= 10`.

    Output: `[N, 8 * 84 * (num_features // 8 // 84)]`, laid out as
    `[series: PPV|MPV|MIPV|LSPV, difference: PPV|MPV|MIPV|LSPV]`.
    """

    def __init__(
        self,
        num_features: int = 50_000,
        max_dilations_per_kernel: int = 32,
        seed: int | None = None,
    ):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.seed = seed

    def fit(self, X, y=None) -> MultiRocket:
        x = as_panel(X, min_length=10)
        rng = np.random.default_rng(self.seed)
        per_representation = self.num_features // (2 * len(POOLING_OPERATORS))
        self.parameters_ = MiniRocketParameters(
            x, per_representation, self.max_dilations_per_kernel, rng
        )
        self.diff_parameters_ = MiniRocketParameters(
            np.ascontiguousarray(np.diff(x, axis=-1)),
            per_representation,
            self.max_dilations_per_kernel,
            rng,
        )
        self.n_features_out_ = len(POOLING_OPERATORS) * (
            len(self.parameters_.biases) + len(self.diff_parameters_.biases)
        )
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "parameters_"):
            raise RuntimeError("Call fit before transform")
        x = as_panel(X, min_length=10, allow_empty=True)
        diff = np.ascontiguousarray(np.diff(x, axis=-1))
        n_pool = len(POOLING_OPERATORS)
        return np.concatenate(
            [
                self.parameters_.transform(x, n_pool),
                self.diff_parameters_.transform(diff, n_pool),
            ],
            axis=1,
        )

    @property
    def kernel_groups_(self) -> np.ndarray:
        """Kernel id of each output feature: (dilation, kernel) combinations of
        the series, then of the difference (offset by the series' count)."""
        n_pool = len(POOLING_OPERATORS)
        offset = self.parameters_.n_combinations
        return np.concatenate(
            [
                np.tile(self.parameters_.feature_combinations(), n_pool),
                np.tile(self.diff_parameters_.feature_combinations() + offset, n_pool),
            ]
        )

    def select_kernels(self, kernels) -> MultiRocket:
        """Fitted copy that only computes the features of `kernels` (ids from
        `kernel_groups_`), in the original column order."""
        kernels = np.asarray(kernels, dtype=np.int64)
        offset = self.parameters_.n_combinations
        pruned = copy.copy(self)
        pruned.parameters_ = self.parameters_.select(kernels[kernels < offset])
        pruned.diff_parameters_ = self.diff_parameters_.select(
            kernels[kernels >= offset] - offset
        )
        pruned.n_features_out_ = len(POOLING_OPERATORS) * (
            len(pruned.parameters_.feature_combinations())
            + len(pruned.diff_parameters_.feature_combinations())
        )
        return pruned
