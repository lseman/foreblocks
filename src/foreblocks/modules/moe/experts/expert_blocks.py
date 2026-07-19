"""foreblocks.modules.moe.experts.expert_blocks.

Mixture-of-Experts feed-forward expert implementations.

Provides SwiGLU and standard FFN expert classes with optional dropout, plus
a multi-token prediction (MTP) head bank for auxiliary horizon prediction.
Use as the expert building blocks in MoE layers where different tokens route
to different FFNs.

Core API:
- MoE_SwiGLUExpert: SwiGLU expert with shared w12 gate/projection
- MoE_FFNExpert: standard FFN expert with configurable activation
- MTPHead: multi-token/horizon prediction head bank

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PackedExpertBank(nn.Module):
    """Canonical expert-axis parameter storage for grouped MoE execution."""

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_ff: int,
        *,
        use_swiglu: bool = True,
        activation: str = "gelu",
        dropout: float = 0.0,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        width = 2 * d_ff if use_swiglu else d_ff
        self.w12 = nn.Parameter(torch.empty(num_experts, d_model, width))
        self.w3 = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        for expert_idx in range(num_experts):
            nn.init.xavier_uniform_(self.w12[expert_idx])
            nn.init.xavier_uniform_(self.w3[expert_idx])
        self.num_experts = int(num_experts)
        self.d_ff = int(d_ff)
        self.use_swiglu = bool(use_swiglu)
        self.activation = getattr(F, activation.lower(), F.gelu)
        self.dropout_p = float(dropout)
        self.expert_dropout_p = float(expert_dropout)

    def __len__(self) -> int:
        return self.num_experts

    def forward_expert(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        hidden = x @ self.w12[expert_idx]
        if self.use_swiglu:
            gate, value = hidden.split(self.d_ff, dim=-1)
            hidden = F.silu(gate) * value
        else:
            hidden = self.activation(hidden)
        if self.training and self.dropout_p > 0:
            hidden = F.dropout(hidden, p=self.dropout_p, training=True)
        out = hidden @ self.w3[expert_idx]
        if self.training and self.expert_dropout_p > 0:
            out = F.dropout(out, p=self.expert_dropout_p, training=True)
        return out


class MoE_SwiGLUExpert(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.0, expert_dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_p = float(dropout)
        self.expert_dropout_p = float(expert_dropout)
        self._needs_dropout = dropout > 0
        self._needs_edrop = expert_dropout > 0
        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.w12(x)
        g, u = gu.split(self.d_ff, dim=-1)
        h = F.silu(g) * u
        if self.training and self._needs_dropout:
            h = F.dropout(h, p=self.dropout_p, training=True)
        out = self.w3(h)
        if self.training and self._needs_edrop:
            out = F.dropout(out, p=self.expert_dropout_p, training=True)
        return out


class MoE_FFNExpert(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout_p = float(dropout)
        self.expert_dropout_p = float(expert_dropout)
        self._needs_dropout = dropout > 0
        self._needs_edrop = expert_dropout > 0
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = getattr(F, activation.lower(), F.gelu)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        if self.training and self._needs_dropout:
            x = F.dropout(x, p=self.dropout_p, training=True)
        if self.training and self._needs_edrop:
            x = F.dropout(x, p=self.expert_dropout_p, training=True)
        return self.fc2(x)


class MTPHead(nn.Module):
    """Simple multi-token/multi-horizon prediction head bank."""

    def __init__(self, d_model: int, n_extra: int = 3, init_scale: float = 0.02):
        super().__init__()
        self.n_extra = int(n_extra)
        self.extra_heads = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(self.n_extra)]
        )
        std = float(init_scale) / math.sqrt(max(self.n_extra + 1, 1))
        for h in self.extra_heads:
            nn.init.normal_(h.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_extra <= 0:
            return x.new_zeros((x.size(0), 0, x.size(-1)))
        preds = [h(x) for h in self.extra_heads]
        return torch.stack(preds, dim=1)


__all__ = ["MTPHead", "MoE_FFNExpert", "MoE_SwiGLUExpert", "PackedExpertBank"]
