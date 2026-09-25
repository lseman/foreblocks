"""Amplitude-envelope extraction and a small dilated temporal encoder.

`compute_envelope` is plain numpy/scipy preprocessing (Hilbert magnitude,
antialiased resampling), reusable outside the model. `EnvelopeEncoder` is a
compact dilated Conv1d stack over the resampled envelope, following the
proposal's "small dilated temporal CNN [to capture] burst repetition and
persistence" (`datasets/_paper_review/model_proposal.md`).

`foreblocks.nn.blocks.tcn.CausalTCNBlock` was considered and rejected: its
`LayerNorm(channels)` is applied to a channel-first `[B, channels, T]`
tensor, so it raises unless `T == channels` by coincidence (confirmed with a
smoke test). `DilatedConvBlock` below uses `GroupNorm(1, channels)`, which is
shape-safe for channel-first conv tensors.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import hilbert, resample_poly


def compute_envelope(x: np.ndarray, fs: int, target_hz: int) -> np.ndarray:
    """Hilbert-magnitude envelope of `x` ([..., T]), antialiased-resampled
    from `fs` to `target_hz` along the last axis.
    """
    x = np.asarray(x)
    if x.ndim < 1 or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite waveforms with at least two samples.")
    if (
        isinstance(fs, bool)
        or isinstance(target_hz, bool)
        or not isinstance(fs, (int, np.integer))
        or not isinstance(target_hz, (int, np.integer))
        or fs <= 0
        or target_hz <= 0
    ):
        raise ValueError("fs and target_hz must be positive integers.")
    up_down = math.gcd(target_hz, fs)
    up, down = target_hz // up_down, fs // up_down

    def envelope(rows: np.ndarray) -> np.ndarray:
        magnitude = np.abs(hilbert(rows, axis=-1))
        # FIR resampling can ring slightly below zero; an amplitude is nonnegative.
        return np.maximum(resample_poly(magnitude, up, down, axis=-1), 0)

    rows = x.reshape(-1, x.shape[-1])
    n_chunks = min(len(rows) // _MIN_ROWS_PER_THREAD, _N_THREADS)
    if n_chunks < 2:
        return envelope(x)
    # scipy's FFT and upfirdn release the GIL, so row chunks run in parallel
    # with results identical to the single-call path.
    parts = _executor().map(envelope, np.array_split(rows, n_chunks))
    return np.concatenate(list(parts)).reshape(*x.shape[:-1], -1)


_N_THREADS = min(32, os.cpu_count() or 1)
_MIN_ROWS_PER_THREAD = 16
_EXECUTOR: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(_N_THREADS, thread_name_prefix="envelope")
    return _EXECUTOR


class DilatedConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = self.norm(z)
        z = self.act(z)
        z = self.dropout(z)
        return x + z


class EnvelopeEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        n_levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Conv1d(1, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                DilatedConvBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
                for level in range(n_levels)
            ]
        )
        self.output_dim = hidden_dim * 2

    def forward(self, envelope: torch.Tensor) -> torch.Tensor:
        """`envelope`: `[B, T_env]` or `[B, 1, T_env]`. Returns
        `[B, output_dim]`.
        """
        x = envelope if envelope.dim() == 3 else envelope.unsqueeze(1)
        z = self.stem(x)
        for block in self.blocks:
            z = block(z)
        mean = z.mean(dim=-1)
        peak = z.amax(dim=-1)
        return torch.cat([mean, peak], dim=-1)
