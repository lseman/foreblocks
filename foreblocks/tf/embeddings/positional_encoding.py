import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with cache and optional explicit positions."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 10000,
        scale: float = 1.0,
        cache_limit: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.scale = scale
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.cache_limit = cache_limit

        self.register_buffer(
            "pe", self._build_table(d_model, max_len), persistent=False
        )
        self._pe_cache: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _build_table(
        D: int,
        T: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        pos = torch.arange(T, dtype=dtype, device=device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, D, 2, dtype=dtype, device=device)
            * (-math.log(10000.0) / D)
        )
        pe = torch.empty(T, D, dtype=dtype, device=device)
        ang = pos * div
        pe[:, 0::2] = torch.sin(ang)
        if D % 2 == 1:
            pe[:, 1::2] = torch.cos(ang[:, :-1])
            pe[:, -1] = 0
        else:
            pe[:, 1::2] = torch.cos(ang)
        return pe.unsqueeze(0)

    def _ensure_table(self, D: int, T: int, device: torch.device) -> torch.Tensor:
        if D == self.d_model and self.pe.size(1) >= T:
            return self.pe[:, :T].to(device=device)
        cached = self._pe_cache.get(D, None)
        if cached is None or cached.size(1) < T:
            tbl = self._build_table(D, max(T, 8), device=device)
            if D <= 2048:
                if len(self._pe_cache) >= self.cache_limit:
                    self._pe_cache.pop(next(iter(self._pe_cache)))
                self._pe_cache[D] = tbl
            return tbl[:, :T]
        return cached[:, :T].to(device=device)

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if pos is not None:
            if pos.dim() == 1:
                pos = pos.unsqueeze(0).expand(B, -1)
            max_idx = int(pos.max().item())
            base = self._ensure_table(D, max_idx + 1, device).squeeze(0)
            pe = F.embedding(pos.to(device=device, dtype=torch.long), base)
        else:
            pe = self._ensure_table(D, T, device).expand(B, -1, -1)

        pe = pe.to(dtype)
        out = x + pe * self.scale
        return self.dropout(out) if self.dropout else out


__all__ = ["PositionalEncoding"]
