"""foreblocks.anomaly.models.anomaly_transformer.

Association-discrepancy transformer for reconstruction-based anomaly scoring.

AnomalyTransformer learns two attention distributions — series attention from data and
prior attention from temporal proximity — then scores anomalies via their KL divergence.
Higher divergence between learned and expected attention patterns indicates anomalous
timesteps. Use for window-level reconstruction scoring where anomaly manifests as
unusual temporal attention patterns.

Core API:
- AnomalyTransformer: association-discrepancy transformer model
- association_discrepancy: compute KL-divergence anomaly scores from attention maps

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.anomaly.models.base import choose_heads


@dataclass
class AnomalyTransformerForward:
    reconstruction: torch.Tensor
    series: list[torch.Tensor]
    prior: list[torch.Tensor]


class _AssociationAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.sigma = nn.Linear(self.d_model, self.n_heads)
        self.out = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(int(window_size), dtype=torch.float32)
        distance = (positions[:, None] - positions[None, :]).abs()
        self.register_buffer("distance", distance, persistent=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim**-0.5
        series = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        series = self.dropout(series)

        sigma = F.softplus(self.sigma(x)).transpose(1, 2).unsqueeze(-1) + 1e-4
        distance = self.distance[:length, :length].to(x.device)
        prior = torch.exp(
            -distance.pow(2).view(1, 1, length, length) / (2.0 * sigma.pow(2))
        )
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(series, v).transpose(1, 2).contiguous()
        context = context.view(bsz, length, self.d_model)
        return self.out(context), series, prior


class _AnomalyTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = _AssociationAttention(d_model, n_heads, window_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, series, prior = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, series, prior


class AnomalyTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 128,
        n_heads: int | None = None,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        n_heads = choose_heads(self.d_model, n_heads)
        dim_feedforward = int(dim_feedforward or max(4 * self.d_model, 128))

        self.input_projection = nn.Linear(self.n_features, self.d_model)
        self.position = nn.Parameter(torch.zeros(1, self.window_size, self.d_model))
        self.layers = nn.ModuleList(
            _AnomalyTransformerLayer(
                self.d_model,
                n_heads,
                self.window_size,
                dim_feedforward,
                dropout,
            )
            for _ in range(int(n_layers))
        )
        self.output_projection = nn.Linear(self.d_model, self.n_features)

    def forward(self, x: torch.Tensor) -> AnomalyTransformerForward:
        h = self.input_projection(x) + self.position[:, : x.shape[1]]
        series: list[torch.Tensor] = []
        prior: list[torch.Tensor] = []
        for layer in self.layers:
            h, layer_series, layer_prior = layer(h)
            series.append(layer_series)
            prior.append(layer_prior)
        return AnomalyTransformerForward(self.output_projection(h), series, prior)

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).reconstruction


def association_discrepancy(out: AnomalyTransformerForward) -> torch.Tensor:
    values = []
    for series, prior in zip(out.series, out.prior, strict=False):
        series = series.clamp_min(1e-8)
        prior = prior.clamp_min(1e-8)
        kl_sp = (series * (series.log() - prior.detach().log())).sum(dim=-1)
        kl_ps = (prior.detach() * (prior.detach().log() - series.log())).sum(dim=-1)
        values.append((kl_sp + kl_ps).mean(dim=1))
    return torch.stack(values, dim=0).mean(dim=0).mean(dim=1)


__all__ = [
    "AnomalyTransformer",
    "AnomalyTransformerForward",
    "association_discrepancy",
]
