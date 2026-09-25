"""HYDRA: competing convolutional kernels (Dempster et al., 2023).

https://arxiv.org/abs/2203.13652 — reference code: https://github.com/angus924/hydra

Kernels are arranged in `g` groups of `k`. At every time step each group
"votes" for its kernels: the kernel with the largest response accumulates that
response (soft max count) and the one with the smallest response gets +1 (hard
min count). Half the groups see the series, half its first difference, at
every power-of-two dilation. Counting kernel wins makes Hydra a dictionary
method built on random convolutions; it pairs well with MultiRocket.

Features are non-negative, sparse counts; scale them with `SparseScaler`
rather than `StandardScaler`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin

from foreblocks.features._validation import as_panel

_KERNEL_LENGTH = 9


class Hydra(TransformerMixin, BaseEstimator):
    """`[N, L]` or `[N, C, L]` -> `[N, num_dilations * min(2, g) * 2 * (g // 2) * k]`.

    Weights are Normal(0, 1), mean-centered and L1-normalized per kernel.
    Dilations are `2 ** i` for `i = 0..floor(log2((L - 1) / 8))`, with "same"
    padding. Multivariate input sums a random subset of
    `clip(C // 2, 2, max_num_channels)` channels per group, as in the reference
    multivariate implementation; univariate input matches the univariate
    reference. The transform is data-independent: `fit` only reads the input
    shape.
    """

    def __init__(
        self,
        k: int = 8,
        g: int = 64,
        max_num_channels: int = 8,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int = 256,
    ):
        self.k = k
        self.g = g
        self.max_num_channels = max_num_channels
        self.seed = seed
        self.device = device
        self.batch_size = batch_size

    def fit(self, X, y=None) -> Hydra:
        x = as_panel(X, allow_empty=True)
        _, n_channels, length = x.shape
        if self.k < 1 or self.g < 1:
            raise ValueError("k and g must be positive.")
        generator = torch.Generator()
        generator.manual_seed(
            int(np.random.SeedSequence(self.seed).generate_state(1)[0])
        )
        self.device_ = torch.device(self.device or "cpu")
        max_exponent = np.log2(max(length - 1, 1) / (_KERNEL_LENGTH - 1))
        self.dilations_ = 2 ** np.arange(max(int(max_exponent), 0) + 1)
        self.paddings_ = (_KERNEL_LENGTH - 1) * self.dilations_ // 2
        self.n_representations_ = 2 if self.g > 1 else 1
        self.groups_per_representation_ = self.g // self.n_representations_
        g_rep = self.groups_per_representation_
        self.weights_ = []
        self.channel_indices_ = []
        channels_per_group = int(np.clip(n_channels // 2, 2, self.max_num_channels))
        for _ in self.dilations_:
            w = torch.randn(
                self.n_representations_, self.k * g_rep, 1, _KERNEL_LENGTH,
                generator=generator,
            )
            w -= w.mean(-1, keepdim=True)
            w /= w.abs().sum(-1, keepdim=True)
            self.weights_.append(w.to(self.device_))
            if n_channels > 1:
                idx = torch.randint(
                    0, n_channels, (self.n_representations_, g_rep, channels_per_group),
                    generator=generator,
                ).sort().values
                self.channel_indices_.append(idx.to(self.device_))
        self.n_channels_in_ = n_channels
        self.n_features_out_ = (
            len(self.dilations_) * self.n_representations_ * 2 * g_rep * self.k
        )
        return self

    def _transform_batch(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        g_rep, k = self.groups_per_representation_, self.k
        diff = torch.diff(x, dim=-1) if self.n_representations_ > 1 else None
        multivariate = self.n_channels_in_ > 1
        out = []
        for i, (dilation, padding) in enumerate(zip(self.dilations_, self.paddings_)):
            for r in range(self.n_representations_):
                source = x if r == 0 else diff
                if multivariate:
                    # [n, g_rep, channels_per_group, L] -> [n, g_rep, L]
                    source = source[:, self.channel_indices_[i][r]].sum(2)
                z = F.conv1d(
                    source,
                    self.weights_[i][r],
                    dilation=int(dilation),
                    padding=int(padding),
                    groups=g_rep if multivariate else 1,
                ).view(n, g_rep, k, -1)
                max_values, max_indices = z.max(2)
                count_max = torch.zeros(n, g_rep, k, device=z.device)
                count_max.scatter_add_(-1, max_indices, max_values)
                _, min_indices = z.min(2)
                count_min = torch.zeros(n, g_rep, k, device=z.device)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(z[:, :, 0]))
                out.append(count_max)
                out.append(count_min)
        return torch.cat(out, 1).view(n, -1)

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("Call fit before transform")
        x = as_panel(X, allow_empty=True)
        if x.shape[1] != self.n_channels_in_:
            raise ValueError(f"Expected {self.n_channels_in_} channels, got {x.shape[1]}.")
        if len(x) == 0:
            return np.empty((0, self.n_features_out_), dtype=np.float32)
        parts = []
        with torch.no_grad():
            for start in range(0, len(x), max(1, int(self.batch_size))):
                batch = torch.from_numpy(x[start : start + self.batch_size]).to(self.device_)
                parts.append(self._transform_batch(batch).cpu().numpy())
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


class SparseScaler(TransformerMixin, BaseEstimator):
    """Hydra's feature scaler: `sqrt(clip(x, 0))`, then standardized, with
    zeros kept at zero (`mask`) and the scale inflated by
    `(fraction of zeros) ** exponent` so mostly-zero features are damped.
    """

    def __init__(self, mask: bool = True, exponent: float = 4.0):
        self.mask = mask
        self.exponent = exponent

    def fit(self, X, y=None) -> SparseScaler:
        x = np.sqrt(np.clip(np.asarray(X, dtype=np.float64), 0, None))
        epsilon = (x == 0).mean(axis=0) ** self.exponent + 1e-8
        self.mean_ = x.mean(axis=0)
        ddof = 1 if len(x) > 1 else 0
        self.scale_ = x.std(axis=0, ddof=ddof) + epsilon
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("Call fit before transform")
        x = np.sqrt(np.clip(np.asarray(X, dtype=np.float64), 0, None))
        centered = x - self.mean_
        if self.mask:
            centered = centered * (x != 0)
        return (centered / self.scale_).astype(np.float32)
