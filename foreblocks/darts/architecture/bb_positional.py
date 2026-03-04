"""Positional encodings: RotaryPositionalEncoding, PositionalEncoding."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

__all__ = ["RotaryPositionalEncoding", "PositionalEncoding"]


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding with odd-dim support and per-device dynamic caches."""

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 500000.0):
        super().__init__()
        self.dim = int(dim)
        self.rotary_dim = int(self.dim - (self.dim % 2))
        self.max_seq_len = int(max_seq_len) * 2
        self.base = float(base)

        if self.rotary_dim > 0:
            inv_freq = 1.0 / (
                self.base
                ** (
                    torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                    / self.rotary_dim
                )
            )
        else:
            inv_freq = torch.zeros(0, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._device_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._init_cache(self.max_seq_len)

    def _cache_key(self, device: torch.device) -> str:
        if device.type == "cuda":
            index = 0 if device.index is None else int(device.index)
            return f"cuda:{index}"
        return device.type

    def _compute_chunk(
        self, start: int, end: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim == 0:
            empty = torch.zeros(
                end - start, 0, device=device, dtype=self.inv_freq.dtype
            )
            return empty, empty

        t = torch.arange(start, end, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin

    def _init_cache(self, max_len: int) -> None:
        cos, sin = self._compute_chunk(0, int(max_len), self.inv_freq.device)
        self.register_buffer("cached_cos", cos, persistent=False)
        self.register_buffer("cached_sin", sin, persistent=False)
        key = self._cache_key(self.inv_freq.device)
        self._device_cache[key] = {"cos": self.cached_cos, "sin": self.cached_sin}

    def _ensure_cache(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self._cache_key(device)

        if key not in self._device_cache:
            base_cos = self.cached_cos.to(device)
            base_sin = self.cached_sin.to(device)
            self._device_cache[key] = {"cos": base_cos, "sin": base_sin}

        cache = self._device_cache[key]
        cos = cache["cos"]
        sin = cache["sin"]
        cached_len = int(cos.size(0))

        if seq_len > cached_len:
            new_len = max(seq_len, cached_len * 2)
            add_cos, add_sin = self._compute_chunk(cached_len, new_len, device)
            cos = torch.cat([cos, add_cos], dim=0)
            sin = torch.cat([sin, add_sin], dim=0)
            cache["cos"] = cos
            cache["sin"] = sin

        return cache["cos"][:seq_len], cache["sin"][:seq_len]

    def forward(
        self, seq_len: int, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.inv_freq.device
        return self._ensure_cache(int(seq_len), device)

    def apply_rotary_pos_emb(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply RoPE to the largest even sub-dimension and passthrough any tail dims."""
        seq_len = x.size(-2)
        head_dim = x.size(-1)
        rotary_dim = min(self.rotary_dim, head_dim - (head_dim % 2))
        if rotary_dim <= 0:
            return x

        half = rotary_dim // 2
        cos = cos[:seq_len, :half].view(1, 1, seq_len, half).to(dtype=x.dtype)
        sin = sin[:seq_len, :half].view(1, 1, seq_len, half).to(dtype=x.dtype)

        x_rot = x[..., :rotary_dim]
        x_even = x_rot[..., :half]
        x_odd = x_rot[..., half:rotary_dim]
        rotated = torch.cat(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
        )

        if rotary_dim < head_dim:
            return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)
        return rotated

    def get_embeddings_for_length(
        self, seq_len: int, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(seq_len, device)


class PositionalEncoding(nn.Module):
    """Optimized positional encoding"""

    def __init__(self, d_model: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model

        if d_model % 2 != 0:
            raise ValueError(f"d_model {d_model} must be even for positional encoding")

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(base) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            pe_extended = self._compute_extended_pe(seq_len, x.device, x.dtype)
            return x + pe_extended[:, :seq_len]
        pe_slice = self.pe[:, :seq_len]
        if pe_slice.device != x.device or pe_slice.dtype != x.dtype:
            pe_slice = pe_slice.to(device=x.device, dtype=x.dtype)
        return x + pe_slice

    def _compute_extended_pe(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=dtype)
        position = torch.arange(0, seq_len, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=dtype, device=device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def extend_max_len(self, new_max_len: int):
        if new_max_len > self.pe.size(1):
            device = self.pe.device
            dtype = self.pe.dtype
            new_pe = self._compute_extended_pe(new_max_len, device, dtype)
            self.register_buffer("pe", new_pe, persistent=False)
