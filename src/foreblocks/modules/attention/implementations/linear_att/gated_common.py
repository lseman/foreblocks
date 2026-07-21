"""Shared building blocks for Gated Delta attention variants."""

from __future__ import annotations

from typing import Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GatedDeltaOwner(Protocol):
    h: int
    dk: int
    dv: int
    use_short_conv: bool
    q_proj: nn.Module
    k_proj: nn.Module
    v_proj: nn.Module
    q_conv: nn.Module
    k_conv: nn.Module
    v_conv: nn.Module
    g_down: nn.Module
    g_up: nn.Module

    def _forward_recurrent(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


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


class GatedDeltaExecutionMixin:
    """Shared projection, output-gating, state, and standalone execution."""

    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        owner = cast(_GatedDeltaOwner, cast(object, self))
        q_raw = owner.q_proj(x)
        k_raw = owner.k_proj(x)
        v_raw = owner.v_proj(x)
        if owner.use_short_conv:
            q_raw = owner.q_conv(q_raw)
            k_raw = owner.k_conv(k_raw)
            v_raw = owner.v_conv(v_raw)
        return q_raw, k_raw, v_raw

    def _output_gate(self, x: torch.Tensor) -> torch.Tensor:
        owner = cast(_GatedDeltaOwner, cast(object, self))
        batch, length = x.shape[:2]
        return (
            owner.g_up(F.silu(owner.g_down(x)))
            .view(batch, length, owner.h, owner.dv)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _init_state(
        self, batch_heads: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        owner = cast(_GatedDeltaOwner, cast(object, self))
        return torch.zeros(
            batch_heads, owner.dk, owner.dv, device=device, dtype=dtype
        )

    @staticmethod
    def _l2_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return F.normalize(value, p=2.0, dim=-1, eps=eps)

    def forward_standalone(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        owner = cast(_GatedDeltaOwner, cast(object, self))
        return owner._forward_recurrent(x, state)


__all__ = ["CausalDepthwiseConv", "GatedDeltaExecutionMixin", "HeadRMSNorm"]
