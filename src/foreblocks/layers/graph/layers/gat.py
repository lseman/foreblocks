"""foreblocks.layers.graph.layers.gat.

GAT convolution layer implementation.

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.graph.common import (
    add_self_loops,
    ensure_adj,
    dtype_neg_inf,
    is_batched_adj,
    xavier_zero_bias,
)
from foreblocks.layers.graph.layers.message_passing import GraphConvBase, MessagePassing


def _attention_bias_from_adj(
    adj: torch.Tensor,
    *,
    batch_size: int,
    steps: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if adj.dim() not in (2, 3):
        raise ValueError("adj must have shape [N, N] or [B, N, N]")

    valid = adj > 0
    safe_adj = torch.where(valid, adj, torch.ones_like(adj))
    bias = torch.log(safe_adj).to(dtype=dtype)
    bias.masked_fill_(~valid, dtype_neg_inf(dtype))

    if is_batched_adj(adj):
        return (
            bias.unsqueeze(1)
            .expand(batch_size, steps, adj.size(-2), adj.size(-1))
            .reshape(batch_size * steps, 1, adj.size(-2), adj.size(-1))
        )
    return bias.view(1, 1, adj.size(-2), adj.size(-1)).expand(
        batch_size * steps, 1, adj.size(-2), adj.size(-1)
    )


class GATConv(GraphConvBase, MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops_flag: bool = True,
        use_graph_norm: bool = True,
        pre_norm: bool | None = None,
        norm_strategy: str | None = None,
        residual: bool = False,
        use_gatv2: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            activation="none",
            dropout=dropout,
            use_graph_norm=use_graph_norm,
            pre_norm=pre_norm,
            norm_strategy=norm_strategy,
            residual=residual,
            aggr="add",
        )
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.H = heads
        self.Dh = out_channels // heads
        self.concat = concat
        self.neg_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops_flag = add_self_loops_flag
        self.use_gatv2 = use_gatv2

        self.q_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.out_proj = (
            nn.Identity()
            if self.concat
            else nn.Linear(self.Dh, out_channels, bias=False)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.attn_fc: nn.Linear | None = None
        if use_gatv2:
            self.attn_fc = nn.Linear(self.Dh, 1, bias=False)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.attn_fc is not None:
            xavier_zero_bias(self.attn_fc)
        if isinstance(self.out_proj, nn.Linear):
            xavier_zero_bias(self.out_proj)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, N, _ = x.shape
        x = self.pre_norm_layer(x)
        q = self.q_proj(x).view(B, T, N, self.H, self.Dh)
        k = self.k_proj(x).view(B, T, N, self.H, self.Dh)
        v = self.v_proj(x).view(B, T, N, self.H, self.Dh)

        combined_weight = edge_weight
        if edge_attr is not None:
            if edge_weight is None:
                combined_weight = edge_attr
            else:
                combined_weight = edge_weight * edge_attr
        A = ensure_adj(
            adj,
            edge_index,
            N,
            combined_weight,
            batch_size=(B if (adj is None and edge_index is not None) else None),
            dtype=x.dtype,
            device=x.device,
        )
        if self.add_self_loops_flag:
            A = add_self_loops(A)

        q = q.permute(0, 1, 3, 2, 4).reshape(B * T, self.H, N, self.Dh)
        k = k.permute(0, 1, 3, 2, 4).reshape(B * T, self.H, N, self.Dh)
        v = v.permute(0, 1, 3, 2, 4).reshape(B * T, self.H, N, self.Dh)

        attn_bias = _attention_bias_from_adj(
            A,
            batch_size=B,
            steps=T,
            dtype=x.dtype,
        )
        if self.use_gatv2:
            assert self.attn_fc is not None
            q = q.unsqueeze(3)
            k = k.unsqueeze(2)
            qk = F.leaky_relu(q + k, negative_slope=self.neg_slope)
            attn_logits = self.attn_fc(qk).squeeze(-1)
            attn_logits = attn_logits / math.sqrt(self.Dh)
            attn_logits = attn_logits + attn_bias
            attn = torch.softmax(attn_logits, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.einsum("bhij,bhjf->bhif", attn, v)
            out = out.view(B, T, self.H, N, self.Dh).permute(0, 1, 3, 2, 4)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=(self.dropout if self.training else 0.0),
            )
            out = out.view(B, T, self.H, N, self.Dh).permute(0, 1, 3, 2, 4)

        if self.concat:
            out = out.reshape(B, T, N, self.H * self.Dh)
        else:
            out = self.out_proj(out.mean(dim=3))

        out = out + self.bias
        residual = self.res_lin(x) if self.residual else None
        return self._apply_norm_act_drop(out, residual)


class GATv2Conv(GATConv):
    """GATv2 convolution with query-conditioned dynamic attention."""

    def __init__(self, *args, **kwargs):
        kwargs["use_gatv2"] = True
        super().__init__(*args, **kwargs)
