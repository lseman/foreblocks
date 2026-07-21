"""Graph Isomorphism Network convolution layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.graph.common import ActivationType, ensure_adj, is_batched_adj
from foreblocks.layers.graph.conv.message_passing import (
    GraphConvBase,
    MessagePassing,
    _check_edge_index,
    _dense_message_passing,
    _edge_index_spmm,
    _edge_weight_view,
    _scatter_add,
)


class GINConv(GraphConvBase, MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        eps: float = 0.0,
        train_eps: bool = False,
        activation: ActivationType = "gelu",
        dropout: float = 0.0,
        use_graph_norm: bool = True,
        pre_norm: bool | None = None,
        norm_strategy: str | None = None,
        residual: bool = False,
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
            aggr="add",
        )
        hidden_channels = hidden_channels or out_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        eps_tensor = torch.tensor(float(eps))
        if train_eps:
            self.eps = nn.Parameter(eps_tensor)
        else:
            self.register_buffer("eps", eps_tensor)

    def _aggregate(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _, num_nodes, _ = x.shape
        if edge_index is not None and adj is None:
            return _edge_index_spmm(
                x,
                edge_index.to(device=x.device),
                num_nodes,
                edge_weight=edge_weight,
                aggr="add",
            )
        adjacency = ensure_adj(
            adj,
            edge_index,
            num_nodes,
            edge_weight,
            batch_size=(batch if adj is None else None),
            dtype=x.dtype,
            device=x.device,
        )
        return _dense_message_passing(x, adjacency)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pre_norm_layer(x)
        residual = self.res_lin(x) if self.residual else None
        neighbors = self._aggregate(x, adj, edge_index, edge_weight)
        out = self.mlp((1.0 + self.eps) * x + neighbors)
        return self._apply_norm_act_drop(out, residual)


class GINEConv(GINConv):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.edge_encoder = nn.Linear(edge_dim, in_channels)
        nn.init.xavier_uniform_(self.edge_encoder.weight)
        nn.init.zeros_(self.edge_encoder.bias)

    def _edge_messages(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        num_nodes = x.size(2)
        edge_index = edge_index.to(device=x.device)
        _check_edge_index(edge_index, num_nodes)
        src, dst = edge_index
        encoded = self.edge_encoder(edge_attr.to(device=x.device, dtype=x.dtype))
        if encoded.dim() == 2:
            encoded = encoded.view(1, 1, encoded.size(0), encoded.size(1))
        elif encoded.dim() == 3:
            if encoded.size(0) != x.size(0):
                raise ValueError("batched edge_attr must match x.size(0)")
            encoded = encoded.unsqueeze(1)
        elif encoded.dim() != 4:
            raise ValueError("edge_attr must have shape [E,D], [B,E,D], or [B,T,E,D]")
        messages = F.relu(x[:, :, src, :] + encoded)
        weight = _edge_weight_view(edge_weight, x, src.numel())
        if weight is not None:
            messages = messages * weight
        return _scatter_add(messages, dst, num_nodes)

    def _dense_edge_messages(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.edge_encoder(edge_attr.to(device=x.device, dtype=x.dtype))
        if encoded.dim() == 3:
            encoded = encoded.unsqueeze(0)
        if encoded.dim() != 4:
            raise ValueError("dense edge_attr must have shape [N,N,D] or [B,N,N,D]")
        if encoded.size(0) not in (1, x.size(0)):
            raise ValueError("batched edge_attr must match x.size(0)")
        messages = F.relu(x.unsqueeze(2) + encoded.unsqueeze(1))
        weights = adj.to(device=x.device, dtype=x.dtype)
        if not is_batched_adj(weights):
            weights = weights.unsqueeze(0)
        return (messages * weights[:, None, :, :, None]).sum(dim=3)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("GINEConv requires edge_attr")
        x = self.pre_norm_layer(x)
        residual = self.res_lin(x) if self.residual else None
        if edge_index is not None and adj is None:
            neighbors = self._edge_messages(x, edge_index, edge_attr, edge_weight)
        else:
            adjacency = ensure_adj(
                adj,
                edge_index,
                x.size(2),
                edge_weight,
                batch_size=(x.size(0) if adj is None else None),
                dtype=x.dtype,
                device=x.device,
            )
            neighbors = self._dense_edge_messages(x, adjacency, edge_attr)
        out = self.mlp((1.0 + self.eps) * x + neighbors)
        return self._apply_norm_act_drop(out, residual)
