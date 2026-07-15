"""foreblocks.layers.graph.layers.sage.

GraphSAGE convolution layer implementation.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.graph.common import (
    ActivationType,
    add_self_loops,
    ensure_adj,
    normalize_row,
    xavier_zero_bias,
)
from foreblocks.layers.graph.layers.message_passing import (
    GraphConvBase,
    MessagePassing,
    _add_self_loops_edge_index,
    _combine_edge_weights,
    _dense_message_passing,
    _edge_index_spmm,
    _normalize_row_edge_weight,
    CachedNorm,
)


class SAGEConv(GraphConvBase, MessagePassing):
    """GraphSAGE-style neighbor mixing with mean-style aggregation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: ActivationType = "gelu",
        dropout: float = 0.0,
        use_graph_norm: bool = True,
        pre_norm: bool | None = None,
        norm_strategy: str | None = None,
        residual: bool = False,
        fuse_linear: bool = False,
        use_sparse: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            activation=activation,
            dropout=dropout,
            use_graph_norm=use_graph_norm,
            pre_norm=pre_norm,
            norm_strategy=norm_strategy,
            residual=residual,
            aggr="mean",
        )
        self.fuse_linear = fuse_linear
        self.use_sparse = use_sparse
        self.cached_norm = CachedNorm()
        if fuse_linear:
            self.proj_weight = nn.Parameter(torch.empty(out_channels, in_channels))
            self.proj_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
            nn.init.xavier_uniform_(self.proj_weight)
            if self.proj_bias is not None:
                nn.init.zeros_(self.proj_bias)
            self.lin = nn.Linear(out_channels * 2, out_channels, bias=bias)
            xavier_zero_bias(self.lin)
        else:
            self.lin = nn.Linear(in_channels * 2, out_channels, bias=bias)
            xavier_zero_bias(self.lin)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _, N, _ = x.shape

        x = self.pre_norm_layer(x)
        x_res = x
        if self.fuse_linear:
            x = F.linear(x, self.proj_weight, self.proj_bias)
        combined_weight = _combine_edge_weights(edge_weight, edge_attr)

        if edge_index is not None and adj is None:
            edge_index = edge_index.to(device=x.device)
            edge_index, sparse_weight = _add_self_loops_edge_index(
                edge_index,
                N,
                combined_weight,
                dtype=x.dtype,
                device=x.device,
            )
            sparse_weight = self.cached_norm(
                _normalize_row_edge_weight,
                edge_index,
                N,
                sparse_weight,
                dtype=x.dtype,
                device=x.device,
            )
            neigh = _edge_index_spmm(
                x,
                edge_index,
                N,
                edge_weight=sparse_weight,
                aggr="add",
                use_sparse=self.use_sparse,
            )
        else:
            A = ensure_adj(
                adj,
                edge_index,
                N,
                combined_weight,
                batch_size=(B if (adj is None and edge_index is not None) else None),
                dtype=x.dtype,
                device=x.device,
            )
            A = normalize_row(add_self_loops(A))
            neigh = _dense_message_passing(x, A)

        y = self.lin(torch.cat([x, neigh], dim=-1))
        residual = self.res_lin(x_res) if self.residual else None
        return self._apply_norm_act_drop(y, residual)
