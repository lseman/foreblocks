from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if self.training and self._needs_edrop:
            h = F.dropout(h, p=self.expert_dropout_p, training=True)
        return self.w3(h)


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


__all__ = ["MTPHead", "MoE_FFNExpert", "MoE_SwiGLUExpert"]
