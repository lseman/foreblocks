from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn

from .common import Tensor, crop_residual_to_match, ensure_adj, xavier_zero_bias
from .mtgnn import (
    MTGNNGraphConstructor,
    MTGNNMixProp,
    MTGNNTemporalGatedUnit,
)
from .norms import make_feature_norm


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
                batch_size=x.size(0)
                if adj is None and edge_index is not None
                else None,
                dtype=x.dtype,
                device=x.device,
            )
        if self.graph_constructor is None:
            raise ValueError(
                "adj or edge_index is required when adaptive graph is disabled."
            )
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

        residual = crop_residual_to_match(x, h)
        out = self.norm(residual + h)
        return out, skip
