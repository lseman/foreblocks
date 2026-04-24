from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    ActivationType,
    AggType,
    Tensor,
    add_self_loops,
    dtype_neg_inf,
    ensure_adj,
    is_batched_adj,
    normalize_gcn,
    normalize_row,
    xavier_zero_bias,
)
from .norms import make_activation, make_norm_pair

try:
    from torch_scatter import (
        scatter_add as torch_scatter_add,
    )
    from torch_scatter import (
        scatter_max as torch_scatter_max,
    )
except ImportError:
    torch_scatter_add = None
    torch_scatter_max = None


def _check_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must be shaped [2, E], got {tuple(edge_index.shape)}"
        )
    if edge_index.numel() == 0:
        return
    if edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes:
        raise ValueError("edge_index contains out-of-range node ids")


def _dense_message_passing(x: Tensor, adj: Tensor) -> Tensor:
    if is_batched_adj(adj):
        return torch.einsum("btjf,bij->btif", x, adj)
    return torch.einsum("btjf,ij->btif", x, adj)


def _attention_bias_from_adj(
    adj: Tensor,
    *,
    batch_size: int,
    steps: int,
    dtype: torch.dtype,
) -> Tensor:
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


def _edge_weight_view(
    edge_weight: Tensor | None,
    x: Tensor,
    num_edges: int,
) -> Tensor | None:
    if edge_weight is None:
        return None
    if edge_weight.dim() == 1:
        if edge_weight.size(0) != num_edges:
            raise ValueError("edge_weight must have shape [E] when unbatched")
        return edge_weight.view(1, 1, num_edges, 1).to(device=x.device, dtype=x.dtype)
    if edge_weight.dim() == 2:
        if edge_weight.size(1) != num_edges:
            raise ValueError("edge_weight must have shape [B, E] when batched")
        if edge_weight.size(0) != x.size(0):
            raise ValueError("batched edge_weight must match x.size(0)")
        return edge_weight.unsqueeze(1).unsqueeze(-1).to(device=x.device, dtype=x.dtype)
    raise ValueError("edge_weight must have shape [E] or [B, E]")


def _combine_edge_weights(
    edge_weight: Tensor | None,
    edge_attr: Tensor | None,
) -> Tensor | None:
    if edge_attr is None:
        return edge_weight
    if edge_weight is None:
        return edge_attr
    return edge_weight * edge_attr


def _scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if src.size(2) != index.numel():
        raise ValueError("src edge dimension must match edge_index length")
    if torch_scatter_add is not None:
        return torch_scatter_add(src, index, dim=2, dim_size=dim_size)
    out = src.new_zeros(src.size(0), src.size(1), dim_size, src.size(-1))
    expanded_index = index.view(1, 1, -1, 1).expand_as(src)
    out.scatter_add_(2, expanded_index, src)
    return out


def _scatter_max(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if src.size(2) != index.numel():
        raise ValueError("src edge dimension must match edge_index length")
    if torch_scatter_max is not None:
        values, _ = torch_scatter_max(src, index, dim=2, dim_size=dim_size)
        return values
    fill = torch.full(
        (src.size(0), src.size(1), dim_size, src.size(-1)),
        dtype_neg_inf(src.dtype),
        device=src.device,
        dtype=src.dtype,
    )
    expanded_index = index.view(1, 1, -1, 1).expand_as(src)
    fill.scatter_reduce_(2, expanded_index, src, reduce="amax", include_self=True)
    return fill


def _scatter_aggregate(
    src: Tensor, index: Tensor, dim_size: int, reduce: AggType
) -> Tensor:
    if reduce == "add":
        return _scatter_add(src, index, dim_size)
    if reduce == "mean":
        out = _scatter_add(src, index, dim_size)
        ones = torch.ones(
            src.size(0), src.size(1), src.size(2), 1, device=src.device, dtype=src.dtype
        )
        counts = _scatter_add(ones, index, dim_size).clamp_min_(1.0)
        return out / counts
    if reduce == "max":
        return _scatter_max(src, index, dim_size)
    raise ValueError(f"Unsupported reduce type: {reduce}")


class CachedNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self._last_edge_index = None
        self._last_edge_weight = None
        self._cached_result: Tensor | None = None

    def forward(
        self,
        norm_fn,
        edge_index: Tensor,
        edge_weight: Tensor | None,
        *args,
        **kwargs,
    ) -> Tensor:
        if (
            edge_index is self._last_edge_index
            and edge_weight is self._last_edge_weight
            and self._cached_result is not None
        ):
            return self._cached_result
        self._last_edge_index = edge_index
        self._last_edge_weight = edge_weight
        self._cached_result = norm_fn(edge_index, edge_weight, *args, **kwargs)
        return self._cached_result


def _edge_index_spmm(
    x: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    aggr: AggType = "add",
    use_sparse: bool = False,
) -> Tensor:
    _check_edge_index(edge_index, num_nodes)
    src = edge_index[0]
    dst = edge_index[1]

    if use_sparse and (edge_weight is None or edge_weight.dim() == 1):
        try:
            values = (
                torch.ones(src.numel(), device=x.device, dtype=x.dtype)
                if edge_weight is None
                else edge_weight.to(device=x.device, dtype=x.dtype)
            )
            sparse_adj = torch.sparse_coo_tensor(
                torch.stack([dst, src], dim=0),
                values,
                (num_nodes, num_nodes),
                device=x.device,
                dtype=x.dtype,
            )
            x_flat = x.reshape(-1, num_nodes, x.size(-1))
            out = torch.stack(
                [torch.sparse.mm(sparse_adj, x_flat[i]) for i in range(x_flat.size(0))],
                dim=0,
            )
            return out.view(x.size(0), x.size(1), num_nodes, x.size(-1))
        except RuntimeError:
            pass

    msg = x[:, :, src, :]
    weight = _edge_weight_view(edge_weight, x, src.numel())
    if weight is not None:
        msg = msg * weight
    return _scatter_aggregate(msg, dst, num_nodes, aggr)


def _add_self_loops_edge_index(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor | None]:
    loops = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
    loop_index = torch.stack([loops, loops], dim=0)
    full_edge_index = torch.cat([edge_index.to(device=device), loop_index], dim=1)
    if edge_weight is None:
        return full_edge_index, None

    edge_weight = edge_weight.to(device=device, dtype=dtype)
    if edge_weight.dim() == 1:
        loop_weight = torch.ones(num_nodes, device=device, dtype=dtype)
    elif edge_weight.dim() == 2:
        if edge_weight.size(1) != edge_index.size(1):
            raise ValueError(
                "batched edge_weight must have shape [B, E] where E == edge_index.size(1)"
            )
        loop_weight = torch.ones(
            edge_weight.size(0), num_nodes, device=device, dtype=dtype
        )
    else:
        raise ValueError(
            f"edge_weight must have shape [E] or [B, E], got shape {tuple(edge_weight.shape)}"
        )
    return full_edge_index, torch.cat([edge_weight, loop_weight], dim=-1)


def _normalize_gcn_edge_weight(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    src = edge_index[0]
    dst = edge_index[1]
    if edge_weight is None:
        weight = torch.ones(src.numel(), device=device, dtype=dtype)
    else:
        weight = edge_weight.to(device=device, dtype=dtype)

    if weight.dim() == 1:
        deg = torch.zeros(num_nodes, device=device, dtype=dtype)
        deg.scatter_add_(0, dst, weight)
        inv = deg.clamp(min=1e-9).pow(-0.5)
        inv[torch.isinf(inv)] = 0.0
        return inv[dst] * weight * inv[src]

    if weight.dim() == 2:
        deg = torch.zeros(weight.size(0), num_nodes, device=device, dtype=dtype)
        dst_index = dst.view(1, -1).expand(weight.size(0), -1)
        deg.scatter_add_(1, dst_index, weight)
        inv = deg.clamp(min=1e-9).pow(-0.5)
        inv[torch.isinf(inv)] = 0.0
        return inv[:, dst] * weight * inv[:, src]

    raise ValueError(
        f"edge_weight must have shape [E] or [B, E], got shape {tuple(weight.shape)}"
    )


def _normalize_row_edge_weight(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    dst = edge_index[1]
    if edge_weight is None:
        weight = torch.ones(dst.numel(), device=device, dtype=dtype)
    else:
        weight = edge_weight.to(device=device, dtype=dtype)

    if weight.dim() == 1:
        deg = torch.zeros(num_nodes, device=device, dtype=dtype)
        deg.scatter_add_(0, dst, weight)
        return weight / deg.clamp(min=1e-9)[dst]

    if weight.dim() == 2:
        deg = torch.zeros(weight.size(0), num_nodes, device=device, dtype=dtype)
        dst_index = dst.view(1, -1).expand(weight.size(0), -1)
        deg.scatter_add_(1, dst_index, weight)
        return weight / deg.clamp(min=1e-9)[:, dst]

    raise ValueError(
        f"edge_weight must have shape [E] or [B, E], got shape {tuple(weight.shape)}"
    )


class MessagePassing(nn.Module):
    """PyG-style message passing over tensors shaped [B, T, N, F]."""

    def __init__(self, aggr: AggType = "add"):
        super().__init__()
        if aggr not in ("add", "mean", "max"):
            raise ValueError("aggr must be one of {'add','mean','max'}")
        self.aggr: AggType = aggr

    @torch.no_grad()
    def _check(self, x: Tensor, A: Tensor) -> None:
        assert x.dim() == 4, f"x must be [B,T,N,F], got {x.shape}"
        num_nodes = x.size(-2)
        if is_batched_adj(A):
            assert A.size(0) == x.size(0) and A.size(1) == A.size(2) == num_nodes
        else:
            assert A.size(0) == A.size(1) == num_nodes

    def message(self, x_j: Tensor, x_i: Tensor, **kwargs) -> Tensor:
        return x_j

    def aggregate(self, m: Tensor, **kwargs) -> Tensor:
        return m

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return aggr_out

    def propagate(
        self,
        x: Tensor,
        adj: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        pre_aggregated: bool = False,
    ) -> Tensor:
        B, T, N, _ = x.shape
        if pre_aggregated:
            m = self.message(x_j=x, x_i=x)
            out = self.aggregate(m)
            return self.update(out, x)

        if edge_index is not None and adj is None:
            _check_edge_index(edge_index, N)
            neigh = _edge_index_spmm(
                x,
                edge_index.to(device=x.device),
                N,
                edge_weight=edge_weight,
                aggr=self.aggr,
            )
        else:
            A = ensure_adj(
                adj=adj,
                edge_index=edge_index,
                num_nodes=N,
                edge_weight=edge_weight,
                batch_size=(B if (adj is None and edge_index is not None) else None),
                dtype=x.dtype,
                device=x.device,
            )
            self._check(x, A)
            neigh = _dense_message_passing(x, A)
        m = self.message(x_j=neigh, x_i=x)
        out = self.aggregate(m)
        return self.update(out, x)


class GraphConvBase:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: ActivationType = "gelu",
        dropout: float = 0.0,
        use_graph_norm: bool = True,
        pre_norm: bool | None = None,
        norm_strategy: str | None = None,
        residual: bool = False,
        aggr: AggType = "add",
    ):
        super().__init__(aggr=aggr)
        self.act = make_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm_strategy, self.pre_norm_layer, self.post_norm_layer = make_norm_pair(
            in_channels,
            out_channels,
            use_graph_norm=use_graph_norm,
            norm_strategy=norm_strategy,
            pre_norm=pre_norm,
        )
        self.residual = residual
        self.res_lin: nn.Module = nn.Identity()
        if residual and in_channels != out_channels:
            self.res_lin = nn.Linear(in_channels, out_channels, bias=False)
            xavier_zero_bias(self.res_lin)

    def _apply_norm_act_drop(
        self,
        x: Tensor,
        residual: Tensor | None = None,
    ) -> Tensor:
        y = self.drop(self.act(x))
        if residual is not None:
            y = y + residual
        return self.post_norm_layer(y)


class GCNConv(GraphConvBase, MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        add_self_loops_flag: bool = True,
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
            aggr="add",
        )
        self.add_self_loops_flag = add_self_loops_flag
        self.fuse_linear = fuse_linear
        self.use_sparse = use_sparse
        self.cached_norm = CachedNorm()
        if fuse_linear:
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
            self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            self.lin = None
        else:
            self.lin = nn.Linear(in_channels, out_channels, bias=bias)
            xavier_zero_bias(self.lin)

    def forward(
        self,
        x: Tensor,
        adj: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        edge_attr: Tensor | None = None,
        pre_normalized: bool = False,
    ) -> Tensor:
        B, _, N, _ = x.shape
        x = self.pre_norm_layer(x)
        x_res = x
        if self.fuse_linear:
            x = F.linear(x, self.weight, self.bias)

        combined_weight = _combine_edge_weights(edge_weight, edge_attr)
        if edge_index is not None and adj is None:
            edge_index = edge_index.to(device=x.device)
            _check_edge_index(edge_index, N)
            sparse_weight = combined_weight
            if self.add_self_loops_flag:
                edge_index, sparse_weight = _add_self_loops_edge_index(
                    edge_index,
                    N,
                    sparse_weight,
                    dtype=x.dtype,
                    device=x.device,
                )
            if not pre_normalized:
                sparse_weight = self.cached_norm(
                    _normalize_gcn_edge_weight,
                    edge_index,
                    N,
                    sparse_weight,
                    dtype=x.dtype,
                    device=x.device,
                )
            agg = _edge_index_spmm(
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
            if self.add_self_loops_flag:
                A = add_self_loops(A)
            if not pre_normalized:
                A = normalize_gcn(A)
            agg = _dense_message_passing(x, A)

        if self.fuse_linear:
            y = agg
        else:
            y = self.lin(agg)
        residual = self.res_lin(x_res) if self.residual else None
        return self._apply_norm_act_drop(y, residual)


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
        x: Tensor,
        adj: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        B, _, N, _ = x.shape

        x = self.pre_norm_layer(x)
        x_res = x
        if self.fuse_linear:
            x = F.linear(x, self.proj_weight, self.proj_bias)
        combined_weight = _combine_edge_weights(edge_weight, edge_attr)

        if edge_index is not None and adj is None:
            edge_index = edge_index.to(device=x.device)
            _check_edge_index(edge_index, N)
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

        if self.fuse_linear:
            y = F.linear(torch.cat([x, neigh], dim=-1), self.weight, self.bias)
        else:
            y = self.lin(torch.cat([x, neigh], dim=-1))
        residual = self.res_lin(x_res) if self.residual else None
        return self._apply_norm_act_drop(y, residual)


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
            self.attn_fc = nn.Linear(self.Dh * 2, 1, bias=False)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.attn_fc is not None:
            xavier_zero_bias(self.attn_fc)
        if isinstance(self.out_proj, nn.Linear):
            xavier_zero_bias(self.out_proj)

    def forward(
        self,
        x: Tensor,
        adj: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        B, T, N, _ = x.shape
        x = self.pre_norm_layer(x)
        q = self.q_proj(x).view(B, T, N, self.H, self.Dh)
        k = self.k_proj(x).view(B, T, N, self.H, self.Dh)
        v = self.v_proj(x).view(B, T, N, self.H, self.Dh)

        combined_weight = _combine_edge_weights(edge_weight, edge_attr)
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
            q = q.unsqueeze(3).expand(-1, -1, -1, N, -1)
            k = k.unsqueeze(2).expand(-1, -1, N, -1, -1)
            qk = torch.cat([q, k], dim=-1)
            attn_logits = self.attn_fc(qk).squeeze(-1)
            attn_logits = attn_logits / math.sqrt(self.Dh)
            attn_logits = attn_logits + attn_bias
            attn = torch.softmax(attn_logits, dim=-1)
            out = torch.einsum("bhtij,bhtjf->bhtif", attn, v)
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

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
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


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0:
            return x + residual
        keep_prob = 1 - self.drop_prob
        mask = torch.bernoulli(
            torch.full((x.size(0), 1, 1, 1), keep_prob, device=x.device)
        )
        return (x / keep_prob) * mask + residual


class JumpKnowledge(nn.Module):
    def __init__(
        self,
        mode: Literal["none", "last", "sum", "max", "concat", "lstm", "attn"] = "none",
        hidden_size: int | None = None,
        output_size: int | None = None,
        num_layers_hint: int | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.hidden = hidden_size
        self.output = output_size
        self.concat_proj: nn.Linear | None = None
        self.attn_score: nn.Linear | None = None
        self.input_proj: nn.Module | None = None
        self.out_proj: nn.Module = nn.Identity()

        if mode == "lstm":
            if hidden_size is None:
                raise ValueError("hidden_size required for LSTM JK")
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.out_proj = (
                nn.Identity()
                if output_size is None or output_size == hidden_size
                else nn.Linear(hidden_size, output_size)
            )
            if isinstance(self.out_proj, nn.Linear):
                xavier_zero_bias(self.out_proj)
        elif mode == "concat" and (num_layers_hint and hidden_size and output_size):
            self.concat_proj = nn.Linear(num_layers_hint * hidden_size, output_size)
            xavier_zero_bias(self.concat_proj)
        elif mode == "attn":
            self.attn_score = nn.LazyLinear(1)
            self.out_proj = (
                nn.Identity() if output_size is None else nn.LazyLinear(output_size)
            )

    def _lazy_init_attn_score(self, in_dim: int, like: Tensor) -> None:
        if self.attn_score is None:
            self.attn_score = nn.Linear(in_dim, 1).to(like.device, like.dtype)
            xavier_zero_bias(self.attn_score)

    def _lazy_init_output_proj(self, in_dim: int, like: Tensor) -> None:
        if self.output is not None and isinstance(self.out_proj, nn.Identity):
            self.out_proj = nn.Linear(in_dim, self.output).to(like.device, like.dtype)
            xavier_zero_bias(self.out_proj)

    def _lazy_init_concat(self, concat_dim: int, like: Tensor) -> None:
        if self.concat_proj is None:
            if self.output is None:
                raise ValueError("output_size must be set for concat JK")
            self.concat_proj = nn.Linear(concat_dim, self.output).to(
                like.device, like.dtype
            )
            xavier_zero_bias(self.concat_proj)

    def _lazy_init_input_proj(self, in_dim: int, like: Tensor) -> None:
        if self.input_proj is None:
            if self.hidden is None:
                raise ValueError("hidden_size must be set for LSTM JK")
            if in_dim != self.hidden:
                self.input_proj = nn.Linear(in_dim, self.hidden).to(
                    like.device, like.dtype
                )
                xavier_zero_bias(self.input_proj)
            else:
                self.input_proj = nn.Identity()

    def forward(self, xs: list[Tensor]) -> Tensor:
        if not xs:
            raise ValueError("JK received empty list")
        base = xs[0].shape[:-1]
        for idx, t in enumerate(xs[1:], 1):
            if t.shape[:-1] != base:
                raise ValueError(
                    f"JK shape mismatch at {idx}: {t.shape} vs {xs[0].shape}"
                )

        if self.mode in ("none", "last"):
            return xs[-1]
        if self.mode == "sum":
            y = xs[0].clone()
            for t in xs[1:]:
                y.add_(t)
            return y
        if self.mode == "max":
            y = xs[0]
            for t in xs[1:]:
                y = torch.maximum(y, t)
            return y
        if self.mode == "concat":
            y = torch.cat(xs, dim=-1)
            self._lazy_init_concat(y.size(-1), y)
            return self.concat_proj(y)
        if self.mode == "lstm":
            B, T, N, D = xs[0].shape
            L = len(xs)
            seq = torch.stack(xs, dim=2).reshape(B * T * N, L, D)
            self._lazy_init_input_proj(D, xs[0])
            assert self.input_proj is not None
            seq = self.input_proj(seq)
            out, _ = self.lstm(seq)
            y = out[:, -1, :].reshape(B, T, N, self.hidden)
            return self.out_proj(y)
        if self.mode == "attn":
            y = torch.stack(xs, dim=2)
            B, T, L, N, D = y.shape
            self._lazy_init_attn_score(D, y)
            assert self.attn_score is not None
            scores = self.attn_score(y).squeeze(-1)
            weights = torch.softmax(scores, dim=2)
            y = (y * weights.unsqueeze(-1)).sum(dim=2)
            self._lazy_init_output_proj(D, y)
            return self.out_proj(y)
        raise ValueError(f"Unknown JK mode: {self.mode}")


__all__ = [
    "EdgeCondGCN",
    "GATConv",
    "GCNConv",
    "JumpKnowledge",
    "MessagePassing",
    "SAGEConv",
    "StochasticDepth",
]
