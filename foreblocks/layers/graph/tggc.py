from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Tensor
from .common import is_batched_adj
from .common import xavier_zero_bias


def symmetric_normalize_adjacency(
    adj: Tensor,
    add_self_loops: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """
    Symmetric normalization for [N, N] or [B, N, N] adjacency tensors.
    """
    if adj.dim() not in (2, 3):
        raise ValueError("adj must have shape [N, N] or [B, N, N].")

    if add_self_loops:
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        adj = adj + (eye.unsqueeze(0) if is_batched_adj(adj) else eye)

    deg = adj.sum(dim=-1).clamp_min_(eps)
    deg_inv_sqrt = deg.pow(-0.5)

    if is_batched_adj(adj):
        scale = deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
    else:
        scale = deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
    return adj * scale


def normalized_laplacian_from_adjacency(
    adj: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    L = I - D^{-1/2} A D^{-1/2}
    """
    norm_adj = symmetric_normalize_adjacency(adj, add_self_loops=False, eps=eps)
    eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
    return eye.unsqueeze(0) - norm_adj if is_batched_adj(adj) else eye - norm_adj


def build_frequency_indices(
    seq_len: int,
    num_modes: int,
    mode_select_method: Literal["lowest", "random"] = "lowest",
    device: torch.device | None = None,
) -> Tensor:
    max_modes = seq_len // 2 + 1
    num_modes = min(int(num_modes), max_modes)
    if num_modes <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    if mode_select_method == "random":
        idx = torch.randperm(max_modes, device=device)[:num_modes]
        return idx.sort().values
    if mode_select_method != "lowest":
        raise ValueError("mode_select_method must be 'lowest' or 'random'.")
    return torch.arange(num_modes, device=device)


class MovingAverage(nn.Module):
    """
    Boundary-replicated moving average over [B, T, N, C] tensors.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        if self.kernel_size <= 1:
            return x

        bsz, steps, num_nodes, channels = x.shape
        left = (self.kernel_size - 1) - ((self.kernel_size - 1) // 2)
        right = (self.kernel_size - 1) // 2

        front = x[:, :1, :, :].expand(bsz, left, num_nodes, channels)
        end = x[:, -1:, :, :].expand(bsz, right, num_nodes, channels)
        x_pad = torch.cat([front, x, end], dim=1)

        pooled = x_pad.permute(0, 2, 3, 1).reshape(bsz * num_nodes * channels, 1, -1)
        pooled = self.pool(pooled)
        return pooled.reshape(bsz, num_nodes, channels, steps).permute(0, 3, 1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class LatentCorrelationLayer(nn.Module):
    """
    Learns a soft node correlation graph from per-node temporal histories.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        symmetric: bool = True,
        activation_slope: float = 0.2,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.symmetric = bool(symmetric)

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.weight_key = nn.Parameter(torch.empty(hidden_size, 1))
        self.weight_query = nn.Parameter(torch.empty(hidden_size, 1))
        nn.init.xavier_uniform_(self.weight_key)
        nn.init.xavier_uniform_(self.weight_query)

        self.leaky_relu = nn.LeakyReLU(activation_slope)
        self.attn_dropout = nn.Dropout(attention_dropout)

    def self_graph_attention(self, node_embeddings: Tensor) -> Tensor:
        key = node_embeddings @ self.weight_key
        query = node_embeddings @ self.weight_query
        scores = self.leaky_relu(key + query.transpose(1, 2))
        return self.attn_dropout(F.softmax(scores, dim=-1))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError("Expected x with shape [B, T, N, C].")
        bsz, steps, num_nodes, channels = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, received {channels}."
            )

        x_seq = x.permute(0, 2, 1, 3).reshape(bsz * num_nodes, steps, channels)
        x_seq = x_seq.transpose(0, 1).contiguous()
        out, _ = self.gru(x_seq)
        node_embed = out[-1].reshape(bsz, num_nodes, self.hidden_size)

        attn = self.self_graph_attention(node_embed)
        adj = attn.mean(dim=0)
        if self.symmetric:
            adj = 0.5 * (adj + adj.transpose(0, 1))
        return adj


class GraphGegenbauerConv(nn.Module):
    """
    Gegenbauer graph polynomial filter over the node axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        order: int = 4,
        alpha: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        if order < 0:
            raise ValueError("order must be >= 0.")
        if alpha <= 0:
            raise ValueError("alpha must be > 0.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.order = int(order)
        self.alpha = float(alpha)

        self.theta = nn.Parameter(
            torch.empty(self.order + 1, in_channels, out_channels)
        )
        nn.init.xavier_uniform_(self.theta)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    @staticmethod
    def _graph_matmul(mat: Tensor, x: Tensor) -> Tensor:
        if is_batched_adj(mat):
            return torch.einsum("bnm,btmc->btnc", mat, x)
        return torch.einsum("nm,btmc->btnc", mat, x)

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        p0 = x
        out = torch.einsum("btnc,co->btno", p0, self.theta[0])

        if self.order == 0:
            return out + self.bias if self.bias is not None else out

        p1 = 2.0 * self.alpha * self._graph_matmul(adj_norm, x)
        out = out + torch.einsum("btnc,co->btno", p1, self.theta[1])

        pkm2, pkm1 = p0, p1
        for k in range(2, self.order + 1):
            ap = self._graph_matmul(adj_norm, pkm1)
            pk = (
                2.0 * (k + self.alpha - 1.0) * ap - (k + 2.0 * self.alpha - 2.0) * pkm2
            ) / float(k)
            out = out + torch.einsum("btnc,co->btno", pk, self.theta[k])
            pkm2, pkm1 = pkm1, pk

        return out + self.bias if self.bias is not None else out


class ComplexLinearModes(nn.Module):
    """
    Complex linear mixing across selected FFT modes.
    """

    def __init__(self, num_modes: int, channels: int, per_channel: bool = True):
        super().__init__()
        self.num_modes = int(num_modes)
        self.channels = int(channels)
        self.per_channel = bool(per_channel)

        if self.per_channel:
            self.weight_real = nn.Parameter(torch.empty(channels, num_modes, num_modes))
            self.weight_imag = nn.Parameter(torch.empty(channels, num_modes, num_modes))
        else:
            self.weight_real = nn.Parameter(torch.empty(num_modes, num_modes))
            self.weight_imag = nn.Parameter(torch.empty(num_modes, num_modes))
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)

    def forward(self, x: Tensor) -> Tensor:
        weight = torch.complex(self.weight_real, self.weight_imag)
        if self.per_channel:
            xp = x.permute(0, 2, 3, 1)
            yp = torch.einsum("bncm,cmk->bnck", xp, weight)
            return yp.permute(0, 3, 1, 2)

        xp = x.permute(0, 2, 3, 1)
        yp = torch.einsum("bncm,mk->bnck", xp, weight)
        return yp.permute(0, 3, 1, 2)


class TemporalSpectralFilter(nn.Module):
    """
    FFT -> mode selection -> complex linear mixing -> inverse FFT.
    """

    def __init__(
        self,
        seq_len: int,
        channels: int,
        num_modes: int,
        mode_select_method: Literal["lowest", "random"] = "lowest",
        per_channel: bool = True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.channels = int(channels)
        self.num_modes = min(int(num_modes), self.seq_len // 2 + 1)
        self.mode_select_method = mode_select_method
        self.mode_filter = ComplexLinearModes(
            self.num_modes,
            channels,
            per_channel=per_channel,
        )
        self.register_buffer(
            "mode_idx", torch.empty(0, dtype=torch.long), persistent=False
        )

    def _indices(self, device: torch.device) -> Tensor:
        if self.mode_idx.numel() != self.num_modes or self.mode_idx.device != device:
            self.mode_idx = build_frequency_indices(
                seq_len=self.seq_len,
                num_modes=self.num_modes,
                mode_select_method=self.mode_select_method,
                device=device,
            )
        return self.mode_idx

    def forward(self, x: Tensor) -> Tensor:
        bsz, steps, _, channels = x.shape
        if steps != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, received {steps}."
            )
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, received {channels}.")

        xf = torch.fft.rfft(x, dim=1)
        idx = self._indices(x.device)
        xs = xf.index_select(dim=1, index=idx)
        ys = self.mode_filter(xs)

        yf = torch.zeros_like(xf)
        yf.index_copy_(1, idx, ys)
        return torch.fft.irfft(yf, n=steps, dim=1)


class FineTemporalSpectralFilter(nn.Module):
    def __init__(
        self,
        seq_len: int,
        channels: int,
        num_modes: int,
        decomp_kernel: int = 25,
        mode_select_method: Literal["lowest", "random"] = "lowest",
    ):
        super().__init__()
        self.decomp = SeriesDecomposition(decomp_kernel)
        self.filter = TemporalSpectralFilter(
            seq_len=seq_len,
            channels=channels,
            num_modes=num_modes,
            mode_select_method=mode_select_method,
            per_channel=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        seasonal, trend = self.decomp(x)
        return trend + self.filter(seasonal)


class TGGCBlock(nn.Module):
    """
    TGGC block adapted to [B, T, N, F] tensors.
    """

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        hidden_channels: int,
        order: int = 4,
        gegen_alpha: float = 1.0,
        coarse_modes: int = 16,
        fine_modes: int = 16,
        use_fine_filter: bool = True,
        decomp_kernel: int = 25,
        mode_select_method: Literal["lowest", "random"] = "lowest",
        dropout: float = 0.1,
        expansion: int = 2,
    ):
        super().__init__()
        self.graph_conv = GraphGegenbauerConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            order=order,
            alpha=gegen_alpha,
        )
        self.coarse_filter = TemporalSpectralFilter(
            seq_len=seq_len,
            channels=hidden_channels,
            num_modes=coarse_modes,
            mode_select_method=mode_select_method,
            per_channel=True,
        )
        self.use_fine_filter = bool(use_fine_filter)
        self.fine_filter = (
            FineTemporalSpectralFilter(
                seq_len=seq_len,
                channels=hidden_channels,
                num_modes=fine_modes,
                decomp_kernel=decomp_kernel,
                mode_select_method=mode_select_method,
            )
            if self.use_fine_filter
            else None
        )

        self.residual_proj = (
            nn.Identity()
            if in_channels == hidden_channels
            else nn.Linear(in_channels, hidden_channels)
        )
        if isinstance(self.residual_proj, nn.Linear):
            xavier_zero_bias(self.residual_proj)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        ffn_hidden = hidden_channels * int(expansion)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_channels),
            nn.Dropout(dropout),
        )
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                xavier_zero_bias(module)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        residual = self.residual_proj(x)
        z = self.graph_conv(x, adj_norm)
        z = self.coarse_filter(z)
        if self.fine_filter is not None:
            z = self.fine_filter(z)

        x = self.norm1(residual + self.dropout(z))
        return self.norm2(x + self.ffn(x))


@dataclass
class TGGCModernConfig:
    num_nodes: int
    seq_len: int
    horizon: int
    in_channels: int
    out_channels: int = 1
    hidden_channels: int = 64
    num_blocks: int = 3
    gegen_order: int = 4
    gegen_alpha: float = 1.0
    coarse_modes: int = 16
    fine_modes: int = 16
    use_fine_filter: bool = True
    decomp_kernel: int = 25
    mode_select_method: Literal["lowest", "random"] = "lowest"
    graph_hidden_size: int = 64
    graph_gru_layers: int = 1
    graph_dropout: float = 0.0
    attention_dropout: float = 0.0
    block_dropout: float = 0.1
    ff_expansion: int = 2
    use_learned_graph_residual: bool = True
    static_adjacency_weight: float = 0.0


class TGGCModern(nn.Module):
    """
    Full TGGC wrapper over [B, T, N, C_in] inputs.

    Returns:
      - predictions: [B, horizon, N, C_out]
      - learned adjacency: [N, N]
    """

    def __init__(self, cfg: TGGCModernConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_corr = LatentCorrelationLayer(
            in_channels=cfg.in_channels,
            hidden_size=cfg.graph_hidden_size,
            num_layers=cfg.graph_gru_layers,
            dropout=cfg.graph_dropout,
            attention_dropout=cfg.attention_dropout,
            symmetric=True,
        )
        self.input_proj = nn.Linear(cfg.in_channels, cfg.hidden_channels)
        xavier_zero_bias(self.input_proj)

        self.blocks = nn.ModuleList(
            [
                TGGCBlock(
                    seq_len=cfg.seq_len,
                    in_channels=cfg.hidden_channels,
                    hidden_channels=cfg.hidden_channels,
                    order=cfg.gegen_order,
                    gegen_alpha=cfg.gegen_alpha,
                    coarse_modes=cfg.coarse_modes,
                    fine_modes=cfg.fine_modes,
                    use_fine_filter=cfg.use_fine_filter,
                    decomp_kernel=cfg.decomp_kernel,
                    mode_select_method=cfg.mode_select_method,
                    dropout=cfg.block_dropout,
                    expansion=cfg.ff_expansion,
                )
                for _ in range(cfg.num_blocks)
            ]
        )

        self.time_head = nn.Linear(cfg.seq_len, cfg.horizon)
        self.out_head = nn.Linear(cfg.hidden_channels, cfg.out_channels)
        xavier_zero_bias(self.time_head)
        xavier_zero_bias(self.out_head)

    def _compute_graph(
        self,
        x: Tensor,
        static_adjacency: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        adj_learned = self.latent_corr(x)
        if static_adjacency is not None:
            if static_adjacency.shape != adj_learned.shape:
                raise ValueError(
                    "static_adjacency must match learned adjacency shape [N, N]."
                )
            if self.cfg.static_adjacency_weight > 0:
                w = float(self.cfg.static_adjacency_weight)
                adj = (1.0 - w) * adj_learned + w * static_adjacency.to(adj_learned)
            else:
                adj = adj_learned
        else:
            adj = adj_learned

        if self.cfg.use_learned_graph_residual:
            eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
            adj = adj + eye

        adj = 0.5 * (adj + adj.transpose(0, 1))
        adj = adj.clamp_min(0.0)
        adj_norm = symmetric_normalize_adjacency(adj, add_self_loops=False)
        return adj_learned, adj_norm

    def forward(
        self,
        x: Tensor,
        static_adjacency: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if x.dim() != 4:
            raise ValueError("Expected x with shape [B, T, N, C].")
        if x.size(1) != self.cfg.seq_len:
            raise ValueError(
                f"Expected sequence length {self.cfg.seq_len}, received {x.size(1)}."
            )
        if x.size(2) != self.cfg.num_nodes:
            raise ValueError(
                f"Expected {self.cfg.num_nodes} nodes, received {x.size(2)}."
            )
        if x.size(3) != self.cfg.in_channels:
            raise ValueError(
                f"Expected {self.cfg.in_channels} channels, received {x.size(3)}."
            )

        adj_learned, adj_norm = self._compute_graph(
            x, static_adjacency=static_adjacency
        )
        z = self.input_proj(x)
        for block in self.blocks:
            z = block(z, adj_norm)

        z = z.permute(0, 2, 3, 1)
        z = self.time_head(z)
        z = z.permute(0, 3, 1, 2)
        return self.out_head(z), adj_learned


__all__ = [
    "ComplexLinearModes",
    "FineTemporalSpectralFilter",
    "GraphGegenbauerConv",
    "LatentCorrelationLayer",
    "MovingAverage",
    "SeriesDecomposition",
    "TGGCBlock",
    "TGGCModern",
    "TGGCModernConfig",
    "TemporalSpectralFilter",
    "build_frequency_indices",
    "normalized_laplacian_from_adjacency",
    "symmetric_normalize_adjacency",
]
