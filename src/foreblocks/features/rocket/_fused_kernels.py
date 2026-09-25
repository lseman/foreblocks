"""Fused dilated-convolution + PPV/MPV pooling kernels for `FusedRocket`.

Each kernel computes, for every (series, random kernel) pair, the "same"
zero-padded dilated convolution and reduces it straight to the proportion of
positive values (PPV) and mean of positive values (MPV), never materializing
the `[batch, kernels, length]` activation tensor. This replaces one `conv1d`
call per dilation bucket (hundreds of launches, each writing and re-reading
its full output) with a single pass.

- `pool_numba`: CPU, parallel over (series, kernel block) with Numba.
- `pool_triton`: CUDA, one Triton program per (kernel, series).

Both match `F.conv1d(x, w, b, dilation=d, padding=(klen - 1) * d // 2)`
followed by `FusedRocket._pool`'s reductions, up to float rounding.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import prange

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - triton is an optional dependency
    triton = None


_KERNEL_BLOCK = 16
_TILE = 4096


@numba.njit(parallel=True, cache=True, fastmath=True)
def _pool_numba(x, weights, biases, dilations, ppv, mpv):
    n_series, length = x.shape
    n_kernels, klen = weights.shape
    n_blocks = (n_kernels + _KERNEL_BLOCK - 1) // _KERNEL_BLOCK
    for task in prange(n_series * n_blocks):
        row = task // n_blocks
        first = (task % n_blocks) * _KERNEL_BLOCK
        series = x[row]
        out = np.empty(_TILE, dtype=np.float32)
        for k in range(first, min(first + _KERNEL_BLOCK, n_kernels)):
            dilation = dilations[k]
            pad = (klen - 1) * dilation // 2
            bias = biases[k]
            count = 0
            total = 0.0
            # Time tiles keep the scratch row in L1; tap-major accumulation over
            # each tap's valid range keeps the inner loop contiguous and
            # branch-free (vectorizable).
            # Views indexed from 0 (`dst[u]`, `src[u]`) avoid Numba's
            # negative-index wraparound check, which blocks vectorization.
            for start in range(0, length, _TILE):
                stop = min(start + _TILE, length)
                tile = out[: stop - start]
                tile[:] = 0.0
                for j in range(klen):
                    w = weights[k, j]
                    shift = j * dilation - pad
                    lo = max(start, -shift)
                    hi = min(stop, length - shift)
                    if hi <= lo:
                        continue
                    dst = out[lo - start : hi - start]
                    src = series[lo + shift : hi + shift]
                    for u in range(hi - lo):
                        dst[u] += w * src[u]
                for t in range(stop - start):
                    value = tile[t] + bias
                    positive = value > 0
                    count += 1 if positive else 0
                    total += value if positive else 0.0
            ppv[row, k] = count / length
            mpv[row, k] = total / max(count, 1)


def pool_numba(
    x: np.ndarray, weights: np.ndarray, biases: np.ndarray, dilations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`x`: `[N, L]` float32. Returns `(PPV, MPV)`, each `[N, num_kernels]`."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    ppv = np.empty((len(x), len(weights)), dtype=np.float32)
    mpv = np.empty_like(ppv)
    _pool_numba(x, weights, biases, dilations, ppv, mpv)
    return ppv, mpv


if triton is not None:

    @triton.jit
    def _pool_triton_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        d_ptr,
        ppv_ptr,
        mpv_ptr,
        length,
        x_stride,
        n_kernels,
        KLEN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        k = tl.program_id(0)
        row = tl.program_id(1)
        dilation = tl.load(d_ptr + k)
        bias = tl.load(b_ptr + k)
        pad = (KLEN - 1) * dilation // 2
        base = x_ptr + row.to(tl.int64) * x_stride
        count = tl.zeros([BLOCK], dtype=tl.int32)
        total = tl.zeros([BLOCK], dtype=tl.float32)
        for start in range(0, length, BLOCK):
            t = start + tl.arange(0, BLOCK)
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for j in tl.static_range(KLEN):
                w = tl.load(w_ptr + k * KLEN + j)
                p = t + j * dilation - pad
                acc += w * tl.load(base + p, mask=(p >= 0) & (p < length), other=0.0)
            acc += bias
            positive = (acc > 0) & (t < length)
            count += positive.to(tl.int32)
            total += tl.where(positive, acc, 0.0)
        c = tl.sum(count, axis=0)
        out = row.to(tl.int64) * n_kernels + k
        tl.store(ppv_ptr + out, c.to(tl.float32) / length)
        tl.store(mpv_ptr + out, tl.sum(total, axis=0) / tl.maximum(c, 1).to(tl.float32))


def pool_triton(x, weights, biases, dilations):
    """`x`: `[N, L]` contiguous float32 CUDA tensor; kernel parameters as
    CUDA tensors. Returns `(PPV, MPV)`, each `[N, num_kernels]`.
    """
    import torch

    n_series, length = x.shape
    n_kernels, klen = weights.shape
    ppv = torch.empty((n_series, n_kernels), device=x.device, dtype=torch.float32)
    mpv = torch.empty_like(ppv)
    _pool_triton_kernel[(n_kernels, n_series)](
        x,
        weights,
        biases,
        dilations,
        ppv,
        mpv,
        length,
        x.stride(0),
        n_kernels,
        KLEN=klen,
        BLOCK=1024,
    )
    return ppv, mpv


def triton_available() -> bool:
    return triton is not None
