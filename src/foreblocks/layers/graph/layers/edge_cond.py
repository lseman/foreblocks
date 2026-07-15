"""foreblocks.layers.graph.layers.edge_cond.

Edge-conditioned GCN layer implementation.

"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.graph.common import is_batched_adj, xavier_zero_bias
from foreblocks.layers.graph.norms import make_activation, make_norm_pair


class EdgeCondGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 16,
        bias: bool = True,
        activation: Literal["relu", "gelu", "silu"] = "gelu",
        dropout: float = 0.0,
        use_graph_norm: bool = True,
        pre_norm: bool | None = None,
        norm_strategy: str | None = None,
    ):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, in_channels * out_channels),
        )
        self.node_lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = make_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm_strategy, self.pre_norm_layer, self.post_norm_layer = make_norm_pair(
            in_channels,
            out_channels,
            use_graph_norm=use_graph_norm,
            norm_strategy=norm_strategy,
            pre_norm=pre_norm,
        )

        xavier_zero_bias(self.node_lin)
        for module in self.edge_net:
            if isinstance(module, nn.Linear):
                xavier_zero_bias(module)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        _, _, N, Fin = x.shape
        Fout = self.node_lin.out_features

        x = self.pre_norm_layer(x)

        edge_feat = adj.unsqueeze(-1)
        if is_batched_adj(adj):
            edge_w = self.edge_net(edge_feat).view(adj.size(0), N, N, Fin, Fout)
        else:
            edge_w = self.edge_net(edge_feat).view(N, N, Fin, Fout).unsqueeze(0)

        agg = torch.einsum("btni,bnsij->btnj", x, edge_w)

        y = self.drop(self.act(self.node_lin(x) + agg))
        return self.post_norm_layer(y)
