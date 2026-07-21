"""Shared building blocks for Gated Delta attention variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDepthwiseConv(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        convolved = self.conv(x.transpose(1, 2))[:, :, :length]
        return F.silu(convolved.transpose(1, 2).contiguous())


class HeadRMSNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        weight = self.weight
        if x.dim() == 4:
            return output * weight.unsqueeze(0).unsqueeze(2)
        expanded = weight.repeat(x.size(0) // weight.size(0), 1).unsqueeze(1)
        return output * expanded


class GatedDeltaStateMixin:
    h: int
    dk: int
    dv: int

    def _init_state(
        self, batch_heads: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(batch_heads, self.dk, self.dv, device=device, dtype=dtype)

    @staticmethod
    def _l2_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return F.normalize(value, p=2.0, dim=-1, eps=eps)


__all__ = ["CausalDepthwiseConv", "GatedDeltaStateMixin", "HeadRMSNorm"]
