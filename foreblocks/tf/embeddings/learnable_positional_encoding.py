import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding with optional low-rank factors."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        initialization: str = "normal",
        scale_strategy: str = "fixed",
        scale_value: Optional[float] = None,
        use_layer_norm: bool = True,
        norm_strategy: str = "pre_add",
        low_rank_dim: Optional[int] = None,
        per_head_scale: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.low_rank_dim = low_rank_dim
        self.norm_strategy = norm_strategy

        if low_rank_dim is None:
            self.pe = nn.Parameter(self._init_pe(initialization, (max_len, d_model)))
        else:
            self.U = nn.Parameter(self._init_pe(initialization, (max_len, low_rank_dim)))
            self.V = nn.Parameter(self._init_pe(initialization, (low_rank_dim, d_model)))

        if scale_strategy == "learnable":
            init_scale = scale_value or math.sqrt(d_model)
            self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        elif scale_strategy == "fixed":
            self.register_buffer(
                "scale",
                torch.tensor(scale_value or math.sqrt(d_model), dtype=torch.float32),
            )
        else:
            self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))

        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.per_head_scale = nn.Parameter(torch.ones(d_model)) if per_head_scale else None
        self._cache: Dict[int, torch.Tensor] = {}

    def _init_pe(self, mode: str, shape: tuple) -> torch.Tensor:
        if mode == "normal":
            return torch.randn(shape) * math.sqrt(2.0 / shape[-1])
        if mode == "uniform":
            bound = math.sqrt(6.0 / shape[-1])
            return torch.empty(shape).uniform_(-bound, bound)
        if mode == "zero":
            return torch.zeros(shape)
        return torch.randn(shape) * 0.02

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        if self.low_rank_dim is None:
            if positions is None:
                if T not in self._cache:
                    self._cache[T] = self.pe[:T].unsqueeze(0)
                pe = self._cache[T].to(device=device)
            else:
                pe = F.embedding(positions.to(device=device, dtype=torch.long), self.pe)
        else:
            if positions is None:
                pe = (self.U[:T] @ self.V).unsqueeze(0).expand(B, -1, -1)
            else:
                Usel = F.embedding(positions.to(device=device, dtype=torch.long), self.U)
                pe = torch.bmm(Usel, self.V.unsqueeze(0).expand(B, -1, -1))

        pe = pe.to(dtype)
        if self.per_head_scale is not None:
            pe = pe * self.per_head_scale.to(dtype=dtype, device=device)

        if self.layer_norm and self.norm_strategy == "pre_add":
            x = self.layer_norm(x)

        scale = self.scale.to(dtype=dtype, device=device)
        x = x + pe * scale

        if self.layer_norm and self.norm_strategy == "post_add":
            x = self.layer_norm(x)

        return self.dropout(x) if self.dropout else x


__all__ = ["LearnablePositionalEncoding"]
