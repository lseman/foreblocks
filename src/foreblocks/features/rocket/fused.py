"""ROCKET kernels with MultiRocket-style PPV/MPV pooling, on fused kernels.

A dilated random-convolution transform in the MultiRocket family (PPV and
mean-of-positive-values pooling of the series and its first difference,
across many independently sampled dilations). Kernel length is fixed with
Normal(0, 1) weights and Uniform(-1, 1) biases (ROCKET's kernel scheme)
rather than MiniRocket's deterministic kernels, which lets the convolution
and pooling run as one fused Numba (CPU) or Triton (CUDA) pass. Use
`MultiRocket` for the published method.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin

from foreblocks.features._validation import input_length
from foreblocks.features.rocket import _fused_kernels


class FusedRocket(TransformerMixin, BaseEstimator):
    """Univariate `[N, L]` -> `[N, 4 * num_kernels]` features:
    {PPV, MPV} x {series, first difference}.

    The transform is data-independent: `fit` accepts the series array or just
    its length.
    """

    def __init__(
        self,
        num_kernels: int = 2_000,
        kernel_length: int = 9,
        seed: int | None = 42,
        device: str | None = None,
        # "auto" uses the fused Triton kernel on CUDA and the fused Numba kernel
        # on CPU; "torch" forces the per-dilation `conv1d` reference path.
        backend: str = "auto",
        # Bounds each conv1d call's output (batch * channels * length) so a
        # dilation bucket that happens to collect many kernels (log-uniform
        # dilation sampling packs many kernels into the low-dilation buckets)
        # cannot allocate an unbounded tensor and OOM.
        max_pool_elements: int = 32 * 1024 * 1024,
    ):
        self.num_kernels = num_kernels
        self.kernel_length = kernel_length
        self.seed = seed
        self.device = device
        self.backend = backend
        self.max_pool_elements = max_pool_elements

    def fit(self, X, y=None) -> FusedRocket:
        length = input_length(X)
        rng = np.random.default_rng(self.seed)
        self.device_ = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        max_exponent = np.log2((length - 1) / (self.kernel_length - 1))
        dilations = np.floor(2 ** rng.uniform(0, max_exponent, size=self.num_kernels))
        dilations = np.clip(dilations.astype(int), 1, None)
        weights = rng.normal(size=(self.num_kernels, self.kernel_length)).astype(
            np.float32
        )
        weights -= weights.mean(axis=1, keepdims=True)
        biases = rng.uniform(-1, 1, size=self.num_kernels).astype(np.float32)

        self.kernels_: dict[int, torch.Tensor] = {}
        self.biases_: dict[int, torch.Tensor] = {}
        for dilation in np.unique(dilations):
            mask = dilations == dilation
            self.kernels_[int(dilation)] = (
                torch.from_numpy(weights[mask]).unsqueeze(1).to(self.device_)
            )
            self.biases_[int(dilation)] = torch.from_numpy(biases[mask]).to(
                self.device_
            )
        # Flat kernel bank in the per-dilation-bucket feature order `_pool`
        # emits (ascending dilation, original order within a bucket), so every
        # backend produces identically ordered features.
        order = np.argsort(dilations, kind="stable")
        self.weights_ = np.ascontiguousarray(weights[order])
        self.biases_flat_ = np.ascontiguousarray(biases[order])
        self.dilations_ = np.ascontiguousarray(dilations[order], dtype=np.int64)
        self.backend_ = self._resolve_backend()
        if self.backend_ == "triton":
            self._weights_t = torch.from_numpy(self.weights_).to(self.device_)
            self._biases_t = torch.from_numpy(self.biases_flat_).to(self.device_)
            self._dilations_t = torch.from_numpy(self.dilations_).to(self.device_)
        self.output_dim = self.num_kernels * 4
        self.n_features_out_ = self.output_dim
        return self

    def _resolve_backend(self) -> str:
        if self.backend not in {"auto", "torch", "numba", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', 'numba' or 'triton'.")
        if self.backend != "auto":
            if self.backend == "triton" and (
                self.device_.type != "cuda" or not _fused_kernels.triton_available()
            ):
                raise ValueError("The triton backend needs triton and a CUDA device.")
            return self.backend
        if self.device_.type == "cuda":
            return "triton" if _fused_kernels.triton_available() else "torch"
        return "numba" if self.device_.type == "cpu" else "torch"

    def _pool_fused(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`x`: `[B, L]`. Same features as `_pool`, via the fused kernels."""
        if self.backend_ == "numba":
            return _fused_kernels.pool_numba(
                x, self.weights_, self.biases_flat_, self.dilations_
            )
        ppv, mpv = _fused_kernels.pool_triton(
            torch.from_numpy(np.ascontiguousarray(x)).to(self.device_),
            self._weights_t,
            self._biases_t,
            self._dilations_t,
        )
        return ppv.cpu().numpy(), mpv.cpu().numpy()

    def _pool(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x`: `[B, 1, L]`. Returns per-kernel `(PPV, MPV)`, each `[B, num_kernels]`."""
        batch, _, length = x.shape
        channel_chunk = max(1, self.max_pool_elements // max(batch * length, 1))
        ppv_parts, mpv_parts = [], []
        for dilation, kernel in self.kernels_.items():
            bias = self.biases_[dilation]
            padding = (self.kernel_length - 1) * dilation // 2
            ppv_chunks, mpv_chunks = [], []
            for start in range(0, kernel.shape[0], channel_chunk):
                out = F.conv1d(
                    x,
                    kernel[start : start + channel_chunk],
                    bias=bias[start : start + channel_chunk],
                    dilation=dilation,
                    padding=padding,
                )
                positive = out > 0
                ppv_chunks.append(positive.float().mean(dim=-1))
                count = positive.sum(dim=-1).clamp(min=1)
                mpv_chunks.append(out.clamp(min=0).sum(dim=-1) / count)
            ppv_parts.append(torch.cat(ppv_chunks, dim=1))
            mpv_parts.append(torch.cat(mpv_chunks, dim=1))
        return torch.cat(ppv_parts, dim=1), torch.cat(mpv_parts, dim=1)

    def transform(self, X, batch_size: int = 64) -> np.ndarray:
        """`X`: `[N, L]`. Returns `[N, output_dim]` pooled features."""
        if not getattr(self, "kernels_", None):
            raise RuntimeError("Call fit before transform")
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("FusedRocket expects univariate series with shape [samples, time].")
        diffs = np.diff(x, axis=-1)
        if getattr(self, "backend_", "torch") != "torch":
            # Fused kernels hold no activations, so larger batches are cheap.
            step = max(batch_size, 256)
            features = []
            for start in range(0, len(x), step):
                ppv_s, mpv_s = self._pool_fused(x[start : start + step])
                ppv_d, mpv_d = self._pool_fused(diffs[start : start + step])
                features.append(np.concatenate([ppv_s, mpv_s, ppv_d, mpv_d], axis=1))
            return np.concatenate(features, axis=0)
        features = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                signal = (
                    torch.from_numpy(x[start : start + batch_size])
                    .unsqueeze(1)
                    .to(self.device_)
                )
                diff = (
                    torch.from_numpy(diffs[start : start + batch_size])
                    .unsqueeze(1)
                    .to(self.device_)
                )
                ppv_s, mpv_s = self._pool(signal)
                ppv_d, mpv_d = self._pool(diff)
                batch = torch.cat([ppv_s, mpv_s, ppv_d, mpv_d], dim=1)
                features.append(batch.cpu().numpy())
        return np.concatenate(features, axis=0)
