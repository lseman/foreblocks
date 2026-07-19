"""foreblocks.layers.graph.spatiotemporal.graph_wavenet.

Real GraphWaveNet spatio-temporal blocks with diffusion-style graph propagation.

Implements GraphWaveNet-style residual blocks combining gated temporal
convolution with diffusion-style graph propagation (forward and backward
diffusion) and adaptive adjacency learning.

Core API:
- DiffusionConv: diffusion-style graph convolution
- GraphWaveNetBlock: spatio-temporal residual block with adaptive graphs

"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.graph.common import (
    Tensor,
    crop_residual_to_match,
    ensure_adj,
    is_batched_adj,
    xavier_zero_bias,
)
from foreblocks.layers.graph.norms import make_feature_norm


class DiffusionConv(nn.Module):
    """Diffusion-style graph convolution from GraphWaveNet.

    Computes:
    h^{(l+1)} = α * P * h^{(l)} + (1-α) * P^T * h^{(l)}

    Where P is the row-normalized adjacency matrix.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gcn_depth: int = 2,
        prop_alpha: float = 0.05,
        dropout: float = 0.0,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.gcn_depth = int(gcn_depth)
        self.prop_alpha = float(prop_alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.add_self_loops = bool(add_self_loops)
        self.proj = nn.Linear(in_channels, out_channels)
        xavier_zero_bias(self.proj)

    def _row_normalize_adjacency(self, adj: Tensor) -> Tensor:
        if adj.dim() not in (2, 3):
            raise ValueError("adj must have shape [N, N] or [B, N, N]")

        if self.add_self_loops:
            eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
            if is_batched_adj(adj):
                adj = adj + eye.unsqueeze(0)
            else:
                adj = adj + eye

        denom = adj.sum(dim=-1, keepdim=True).clamp_min_(1e-9)
        return adj / denom

    def _node_propagate(self, x: Tensor, adj: Tensor) -> Tensor:
        if is_batched_adj(adj):
            if adj.size(0) != x.size(0):
                raise ValueError("Batched adjacency must match x.size(0).")
            return torch.einsum("btmf,bnm->btnf", x, adj)
        return torch.einsum("btmf,nm->btnf", x, adj)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        a = self._row_normalize_adjacency(adj)
        h = x
        for _ in range(self.gcn_depth):
            h = self.prop_alpha * x + (1.0 - self.prop_alpha) * self._node_propagate(h, a)
        return self.dropout(self.proj(h))


class BackwardDiffusionConv(nn.Module):
    """Backward diffusion graph convolution (transpose adjacency)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gcn_depth: int = 2,
        prop_alpha: float = 0.05,
        dropout: float = 0.0,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.gcn_depth = int(gcn_depth)
        self.prop_alpha = float(prop_alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.add_self_loops = bool(add_self_loops)
        self.proj = nn.Linear(in_channels, out_channels)
        xavier_zero_bias(self.proj)

    def _row_normalize_adjacency(self, adj: Tensor) -> Tensor:
        if adj.dim() not in (2, 3):
            raise ValueError("adj must have shape [N, N] or [B, N, N]")

        if self.add_self_loops:
            eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
            if is_batched_adj(adj):
                adj = adj + eye.unsqueeze(0)
            else:
                adj = adj + eye

        denom = adj.sum(dim=-1, keepdim=True).clamp_min_(1e-9)
        return adj / denom

    def _node_propagate(self, x: Tensor, adj: Tensor) -> Tensor:
        if is_batched_adj(adj):
            if adj.size(0) != x.size(0):
                raise ValueError("Batched adjacency must match x.size(0).")
            return torch.einsum("btmf,bnm->btnf", x, adj)
        return torch.einsum("btmf,nm->btnf", x, adj)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # Use transpose adjacency for backward diffusion
        a = self._row_normalize_adjacency(adj.transpose(-1, -2))
        h = x
        for _ in range(self.gcn_depth):
            h = self.prop_alpha * x + (1.0 - self.prop_alpha) * self._node_propagate(h, a)
        return self.dropout(self.proj(h))


class GraphWaveNetTemporalConv(nn.Module):
    """Gated temporal convolution with dilated causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_factor: int = 1,
        kernel_set: Sequence[int] = (2, 3, 6, 7),
        dropout: float = 0.0,
        preserve_length: bool = True,
    ):
        super().__init__()
        kernels = tuple(int(k) for k in kernel_set)
        if not kernels:
            raise ValueError("kernel_set must contain at least one kernel size.")
        if any(k <= 0 for k in kernels):
            raise ValueError("kernel sizes must be positive.")

        self.kernel_set = kernels
        self.preserve_length = bool(preserve_length)
        self.dilation_factor = int(dilation_factor)

        base = out_channels // len(kernels)
        remainder = out_channels % len(kernels)
        branch_channels = [
            base + (1 if i < remainder else 0) for i in range(len(kernels))
        ]

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        for branch_out, kernel in zip(branch_channels, kernels):
            # Filter conv
            f_conv = nn.Conv2d(
                in_channels,
                branch_out,
                kernel_size=(1, kernel),
                dilation=(1, self.dilation_factor),
                bias=True,
            )
            nn.init.xavier_uniform_(f_conv.weight)
            if f_conv.bias is not None:
                nn.init.zeros_(f_conv.bias)
            self.filter_convs.append(f_conv)

            # Gate conv
            g_conv = nn.Conv2d(
                in_channels,
                branch_out,
                kernel_size=(1, kernel),
                dilation=(1, self.dilation_factor),
                bias=True,
            )
            nn.init.xavier_uniform_(g_conv.weight)
            if g_conv.bias is not None:
                nn.init.zeros_(g_conv.bias)
            self.gate_convs.append(g_conv)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x_cf = x.permute(0, 3, 2, 1).contiguous()
        filter_outs: list[Tensor] = []
        gate_outs: list[Tensor] = []

        for kernel, f_conv, g_conv in zip(self.kernel_set, self.filter_convs, self.gate_convs):
            if self.preserve_length:
                pad_t = self.dilation_factor * (kernel - 1)
                x_in = F.pad(x_cf, (pad_t, 0, 0, 0))
            else:
                x_in = x_cf

            f_out = f_conv(x_in)
            g_out = g_conv(x_in)

            filter_outs.append(torch.tanh(f_out))
            gate_outs.append(torch.sigmoid(g_out))

        if not self.preserve_length:
            target_t = filter_outs[-1].size(-1)
            filter_outs = [out[..., -target_t:] for out in filter_outs]
            gate_outs = [out[..., -target_t:] for out in gate_outs]

        filter_y = torch.cat(filter_outs, dim=1)
        gate_y = torch.cat(gate_outs, dim=1)

        y = filter_y * gate_y
        return self.dropout(y).permute(0, 3, 2, 1).contiguous()


class GraphWaveNetBlock(nn.Module):
    """
    Real GraphWaveNet-style residual block over [B, T, N, F] tensors.

    The block combines a gated temporal convolution with diffusion-style
    graph propagation (forward and backward diffusion) and can fall back
    to a learned adaptive adjacency when no graph is provided at call time.
    """

    def __init__(
        self,
        num_nodes: int,
        channels: int,
        conv_channels: int | None = None,
        skip_channels: int | None = None,
        gcn_depth: int = 2,
        prop_alpha: float = 0.05,
        dropout: float = 0.3,
        dilation_factor: int = 1,
        kernel_set: Sequence[int] = (2, 3, 6, 7),
        preserve_length: bool = True,
        use_adaptive_graph: bool = True,
        adaptive_k: int | None = None,
        adaptive_embed_dim: int | None = None,
        adaptive_alpha: float = 3.0,
        add_self_loops: bool = True,
        use_graph_norm: bool = False,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.channels = int(channels)
        self.conv_channels = int(conv_channels or channels)
        self.skip_channels = int(skip_channels or 0)
        self.use_adaptive_graph = bool(use_adaptive_graph)
        self.add_self_loops = bool(add_self_loops)

        self.temporal = GraphWaveNetTemporalConv(
            in_channels=self.channels,
            out_channels=self.conv_channels,
            dilation_factor=dilation_factor,
            kernel_set=kernel_set,
            dropout=dropout,
            preserve_length=preserve_length,
        )

        self.graph_forward = DiffusionConv(
            self.conv_channels,
            self.channels,
            gcn_depth=gcn_depth,
            prop_alpha=prop_alpha,
            dropout=dropout,
            add_self_loops=add_self_loops,
        )
        self.graph_backward = BackwardDiffusionConv(
            self.conv_channels,
            self.channels,
            gcn_depth=gcn_depth,
            prop_alpha=prop_alpha,
            dropout=dropout,
            add_self_loops=add_self_loops,
        )

        self.residual_proj = nn.Linear(self.conv_channels, self.channels)
        xavier_zero_bias(self.residual_proj)

        self.skip_proj = (
            nn.Linear(self.conv_channels, self.skip_channels)
            if self.skip_channels > 0
            else None
        )
        if self.skip_proj is not None:
            xavier_zero_bias(self.skip_proj)

        self.norm = make_feature_norm(self.channels, use_graph_norm=use_graph_norm)

        if self.use_adaptive_graph:
            # Simple adaptive graph constructor using node embeddings
            self.embed_dim = int(adaptive_embed_dim or max(8, self.channels))
            self.emb_forward = nn.Embedding(num_nodes, self.embed_dim)
            self.emb_backward = nn.Embedding(num_nodes, self.embed_dim)
            self.lin_forward = nn.Linear(self.embed_dim, self.embed_dim)
            self.lin_backward = nn.Linear(self.embed_dim, self.embed_dim)
            nn.init.xavier_uniform_(self.emb_forward.weight)
            nn.init.xavier_uniform_(self.emb_backward.weight)
            xavier_zero_bias(self.lin_forward)
            xavier_zero_bias(self.lin_backward)
            self.adaptive_k = int(adaptive_k or num_nodes)
            self.adaptive_alpha = float(adaptive_alpha)
        else:
            self.emb_forward = None
            self.emb_backward = None
            self.lin_forward = None
            self.lin_backward = None

    def _build_adaptive_adj(self, x: Tensor) -> Tensor:
        device = x.device
        num_nodes = self.num_nodes
        node_idx = torch.arange(num_nodes, device=device, dtype=torch.long)

        nodevec1 = F.tanh(self.adaptive_alpha * self.lin_forward(self.emb_forward(node_idx)))
        nodevec2 = F.tanh(self.adaptive_alpha * self.lin_backward(self.emb_backward(node_idx)))

        score = nodevec1 @ nodevec2.transpose(0, 1) - nodevec2 @ nodevec1.transpose(0, 1)
        adj = F.relu(torch.tanh(self.adaptive_alpha * score))

        if self.adaptive_k <= 0 or self.adaptive_k >= num_nodes:
            return adj

        noise = torch.rand_like(adj) * 1e-2
        top_idx = (adj + noise).topk(self.adaptive_k, dim=-1).indices
        mask = torch.zeros_like(adj)
        mask.scatter_(1, top_idx, 1.0)
        return adj * mask

    def _resolve_adjacency(
        self,
        x: Tensor,
        adj: Tensor | None,
        edge_index: Tensor | None,
        edge_weight: Tensor | None,
    ) -> Tensor:
        if adj is not None or edge_index is not None:
            return ensure_adj(
                adj=adj,
                edge_index=edge_index,
                num_nodes=x.size(2),
                edge_weight=edge_weight,
                batch_size=(
                    x.size(0) if adj is None and edge_index is not None else None
                ),
                dtype=x.dtype,
                device=x.device,
            )
        if self.emb_forward is None:
            raise ValueError(
                "adj or edge_index is required when adaptive graph is disabled."
            )
        return self._build_adaptive_adj(x)

    def forward(
        self,
        x: Tensor,
        adj: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        graph_adj = self._resolve_adjacency(x, adj, edge_index, edge_weight)
        z = self.temporal(x)
        skip = self.skip_proj(z) if self.skip_proj is not None else None

        h_forward = self.graph_forward(z, graph_adj)
        h_backward = self.graph_backward(z, graph_adj)

        h = h_forward + h_backward

        residual = crop_residual_to_match(x, h)
        out = self.norm(residual + h)
        return out, skip
