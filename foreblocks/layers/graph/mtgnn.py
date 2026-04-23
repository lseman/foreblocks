from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Tensor
from .common import ensure_adj
from .common import is_batched_adj
from .common import xavier_zero_bias
from .norms import make_feature_norm


def _default_node_idx(num_nodes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_nodes, device=device, dtype=torch.long)


def _crop_residual_to_match(x: Tensor, ref: Tensor) -> Tensor:
    if x.size(1) == ref.size(1):
        return x
    if x.size(1) < ref.size(1):
        raise ValueError("Residual input is shorter than the temporal reference tensor.")
    return x[:, -ref.size(1) :, :, :]


def _row_normalize_adjacency(
    adj: Tensor,
    *,
    add_self_loops: bool = True,
    eps: float = 1e-9,
) -> Tensor:
    if adj.dim() not in (2, 3):
        raise ValueError("adj must have shape [N, N] or [B, N, N]")

    if add_self_loops:
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        if is_batched_adj(adj):
            adj = adj + eye.unsqueeze(0)
        else:
            adj = adj + eye

    denom = adj.sum(dim=-1, keepdim=True).clamp_min_(eps)
    return adj / denom


def _node_propagate(x: Tensor, adj: Tensor) -> Tensor:
    """
    x: [B, T, N, F]
    adj: [N, N] or [B, N, N]
    returns: [B, T, N, F]
    """
    if is_batched_adj(adj):
        if adj.size(0) != x.size(0):
            raise ValueError("Batched adjacency must match x.size(0).")
        return torch.einsum("btmf,bnm->btnf", x, adj)
    return torch.einsum("btmf,nm->btnf", x, adj)


def _conv2d_xavier(module: nn.Conv2d) -> None:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class MTGNNGraphConstructor(nn.Module):
    """
    Adaptive sparse graph learner from MTGNN-style node embeddings.

    Returns a directed adjacency over the selected node subset.
    """

    def __init__(
        self,
        num_nodes: int,
        k: int,
        embed_dim: int,
        alpha: float = 3.0,
        static_feat: Tensor | None = None,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.k = int(k)
        self.embed_dim = int(embed_dim)
        self.alpha = float(alpha)

        if static_feat is not None:
            if static_feat.dim() != 2 or static_feat.size(0) != num_nodes:
                raise ValueError("static_feat must have shape [num_nodes, feat_dim].")
            feat_dim = int(static_feat.size(1))
            self.register_buffer("static_feat", static_feat.detach().clone())
            self.lin1 = nn.Linear(feat_dim, embed_dim)
            self.lin2 = nn.Linear(feat_dim, embed_dim)
            self.emb1 = None
            self.emb2 = None
        else:
            self.register_buffer("static_feat", None)
            self.emb1 = nn.Embedding(num_nodes, embed_dim)
            self.emb2 = nn.Embedding(num_nodes, embed_dim)
            self.lin1 = nn.Linear(embed_dim, embed_dim)
            self.lin2 = nn.Linear(embed_dim, embed_dim)
            nn.init.xavier_uniform_(self.emb1.weight)
            nn.init.xavier_uniform_(self.emb2.weight)

        xavier_zero_bias(self.lin1)
        xavier_zero_bias(self.lin2)

    def _node_features(self, node_idx: torch.Tensor) -> tuple[Tensor, Tensor]:
        if self.static_feat is None:
            assert self.emb1 is not None and self.emb2 is not None
            return self.emb1(node_idx), self.emb2(node_idx)

        static_feat = self.static_feat.to(device=node_idx.device)
        node_feat = static_feat.index_select(0, node_idx)
        return node_feat, node_feat

    def full_adjacency(self, node_idx: torch.Tensor | None = None) -> Tensor:
        device = (
            self.lin1.weight.device
            if self.static_feat is None
            else self.static_feat.device
        )
        node_idx = (
            _default_node_idx(self.num_nodes, device)
            if node_idx is None
            else node_idx.to(device=device, dtype=torch.long)
        )
        nodevec1, nodevec2 = self._node_features(node_idx)
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        score = nodevec1 @ nodevec2.transpose(0, 1) - nodevec2 @ nodevec1.transpose(0, 1)
        return F.relu(torch.tanh(self.alpha * score))

    def forward(self, node_idx: torch.Tensor | None = None) -> Tensor:
        adj = self.full_adjacency(node_idx=node_idx)
        if self.k <= 0 or self.k >= adj.size(-1):
            return adj

        noise = torch.rand_like(adj) * 1e-2
        top_idx = (adj + noise).topk(self.k, dim=-1).indices
        mask = torch.zeros_like(adj)
        mask.scatter_(1, top_idx, 1.0)
        return adj * mask


class MTGNNProp(nn.Module):
    """
    Single-state MTGNN propagation operator.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gdep: int,
        alpha: float = 0.05,
        dropout: float = 0.0,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.gdep = int(gdep)
        self.alpha = float(alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.add_self_loops = bool(add_self_loops)
        self.proj = nn.Linear(in_channels, out_channels)
        xavier_zero_bias(self.proj)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        a = _row_normalize_adjacency(adj, add_self_loops=self.add_self_loops)
        h = x
        for _ in range(self.gdep):
            h = self.alpha * x + (1.0 - self.alpha) * _node_propagate(h, a)
        return self.dropout(self.proj(h))


class MTGNNMixProp(nn.Module):
    """
    Mix-hop propagation from MTGNN, adapted to [B, T, N, F] tensors.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gdep: int,
        alpha: float = 0.05,
        dropout: float = 0.0,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.gdep = int(gdep)
        self.alpha = float(alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.add_self_loops = bool(add_self_loops)
        self.proj = nn.Linear((self.gdep + 1) * in_channels, out_channels)
        xavier_zero_bias(self.proj)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        a = _row_normalize_adjacency(adj, add_self_loops=self.add_self_loops)
        h = x
        hops = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1.0 - self.alpha) * _node_propagate(h, a)
            hops.append(h)
        return self.dropout(self.proj(torch.cat(hops, dim=-1)))


class MTGNNDilatedInception(nn.Module):
    """
    Multi-kernel temporal convolution block from MTGNN.

    Input/output shape: [B, T, N, F]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_factor: int = 1,
        kernel_set: Sequence[int] = (2, 3, 6, 7),
        preserve_length: bool = True,
        bias: bool = True,
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

        self.tconv = nn.ModuleList()
        for branch_out, kernel in zip(branch_channels, kernels):
            conv = nn.Conv2d(
                in_channels,
                branch_out,
                kernel_size=(1, kernel),
                dilation=(1, self.dilation_factor),
                bias=bias,
            )
            _conv2d_xavier(conv)
            self.tconv.append(conv)

    def forward(self, x: Tensor) -> Tensor:
        x_cf = x.permute(0, 3, 2, 1).contiguous()
        outputs: list[Tensor] = []
        for kernel, conv in zip(self.kernel_set, self.tconv):
            if self.preserve_length:
                pad_t = self.dilation_factor * (kernel - 1)
                x_in = F.pad(x_cf, (pad_t, 0, 0, 0))
            else:
                x_in = x_cf
            outputs.append(conv(x_in))

        if not self.preserve_length:
            target_t = outputs[-1].size(-1)
            outputs = [out[..., -target_t:] for out in outputs]

        y = torch.cat(outputs, dim=1)
        return y.permute(0, 3, 2, 1).contiguous()


class MTGNNTemporalGatedUnit(nn.Module):
    """
    MTGNN temporal filter/gate pair.
    """

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
        self.filter_conv = MTGNNDilatedInception(
            in_channels,
            out_channels,
            dilation_factor=dilation_factor,
            kernel_set=kernel_set,
            preserve_length=preserve_length,
        )
        self.gate_conv = MTGNNDilatedInception(
            in_channels,
            out_channels,
            dilation_factor=dilation_factor,
            kernel_set=kernel_set,
            preserve_length=preserve_length,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        filt = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        return self.dropout(filt * gate)


class MTGNNBlock(nn.Module):
    """
    Reusable MTGNN spatiotemporal residual block.

    The block keeps MTGNN's temporal gated unit and bidirectional mix-hop graph
    propagation, but uses repo-native [B, T, N, F] tensors and a simpler
    feature-only normalization step so it can be composed more freely.
    """

    def __init__(
        self,
        channels: int,
        conv_channels: int | None = None,
        skip_channels: int | None = None,
        gcn_depth: int = 2,
        prop_alpha: float = 0.05,
        dropout: float = 0.3,
        dilation_factor: int = 1,
        kernel_set: Sequence[int] = (2, 3, 6, 7),
        use_graph_conv: bool = True,
        use_reverse_graph: bool = True,
        preserve_length: bool = True,
        use_graph_norm: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.conv_channels = int(conv_channels or channels)
        self.skip_channels = int(skip_channels or 0)
        self.use_graph_conv = bool(use_graph_conv)
        self.use_reverse_graph = bool(use_reverse_graph)

        self.temporal = MTGNNTemporalGatedUnit(
            in_channels=self.channels,
            out_channels=self.conv_channels,
            dilation_factor=dilation_factor,
            kernel_set=kernel_set,
            dropout=dropout,
            preserve_length=preserve_length,
        )
        self.residual_proj = nn.Linear(self.conv_channels, self.channels)
        xavier_zero_bias(self.residual_proj)

        if self.use_graph_conv:
            self.graph_forward = MTGNNMixProp(
                self.conv_channels,
                self.channels,
                gdep=gcn_depth,
                alpha=prop_alpha,
                dropout=dropout,
            )
            self.graph_reverse = (
                MTGNNMixProp(
                    self.conv_channels,
                    self.channels,
                    gdep=gcn_depth,
                    alpha=prop_alpha,
                    dropout=dropout,
                )
                if self.use_reverse_graph
                else None
            )
        else:
            self.graph_forward = None
            self.graph_reverse = None

        self.skip_proj = (
            nn.Linear(self.conv_channels, self.skip_channels)
            if self.skip_channels > 0
            else None
        )
        if self.skip_proj is not None:
            xavier_zero_bias(self.skip_proj)

        self.norm = make_feature_norm(self.channels, use_graph_norm=use_graph_norm)

    def forward(self, x: Tensor, adj: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        z = self.temporal(x)
        skip = self.skip_proj(z) if self.skip_proj is not None else None

        if self.use_graph_conv:
            if adj is None:
                raise ValueError("adj is required when use_graph_conv=True.")
            h = self.graph_forward(z, adj)
            if self.graph_reverse is not None:
                h = h + self.graph_reverse(z, adj.transpose(-1, -2))
        else:
            h = self.residual_proj(z)

        residual = _crop_residual_to_match(x, h)
        out = self.norm(residual + h)
        return out, skip


class GraphWaveNetBlock(nn.Module):
    """
    Graph WaveNet-style residual block over [B, T, N, F] tensors.

    The block combines a gated temporal convolution with diffusion-style graph
    propagation and can fall back to a learned adaptive adjacency when no graph
    is provided at call time.
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

        self.temporal = MTGNNTemporalGatedUnit(
            in_channels=self.channels,
            out_channels=self.conv_channels,
            dilation_factor=dilation_factor,
            kernel_set=kernel_set,
            dropout=dropout,
            preserve_length=preserve_length,
        )
        self.graph_forward = MTGNNMixProp(
            self.conv_channels,
            self.channels,
            gdep=gcn_depth,
            alpha=prop_alpha,
            dropout=dropout,
            add_self_loops=add_self_loops,
        )
        self.graph_reverse = MTGNNMixProp(
            self.conv_channels,
            self.channels,
            gdep=gcn_depth,
            alpha=prop_alpha,
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
            self.graph_constructor = MTGNNGraphConstructor(
                num_nodes=self.num_nodes,
                k=adaptive_k or self.num_nodes,
                embed_dim=adaptive_embed_dim or max(8, self.channels),
                alpha=adaptive_alpha,
            )
        else:
            self.graph_constructor = None

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
                batch_size=x.size(0) if adj is None and edge_index is not None else None,
                dtype=x.dtype,
                device=x.device,
            )
        if self.graph_constructor is None:
            raise ValueError("adj or edge_index is required when adaptive graph is disabled.")
        return self.graph_constructor().to(device=x.device, dtype=x.dtype)

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

        h = self.graph_forward(z, graph_adj)
        h = h + self.graph_reverse(z, graph_adj.transpose(-1, -2))

        residual = _crop_residual_to_match(x, h)
        out = self.norm(residual + h)
        return out, skip


__all__ = [
    "GraphWaveNetBlock",
    "MTGNNBlock",
    "MTGNNDilatedInception",
    "MTGNNGraphConstructor",
    "MTGNNMixProp",
    "MTGNNProp",
    "MTGNNTemporalGatedUnit",
]
