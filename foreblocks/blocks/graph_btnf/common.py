from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

Tensor = torch.Tensor
AggType = Literal["add", "mean", "max"]
ActivationType = Literal["relu", "gelu", "silu", "none"]


def is_batched_adj(adj: Tensor) -> bool:
    return adj.dim() == 3


def add_self_loops(adj: Tensor) -> Tensor:
    if is_batched_adj(adj):
        batch_size, num_nodes, _ = adj.shape
        eye = (
            torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
            .unsqueeze(0)
            .expand(batch_size, num_nodes, num_nodes)
        )
        return adj + eye
    num_nodes = adj.size(0)
    return adj + torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)


def normalize_gcn(adj: Tensor, eps: float = 1e-9) -> Tensor:
    deg = adj.sum(-1)
    inv = deg.clamp(min=eps).pow(-0.5)
    inv[torch.isinf(inv)] = 0.0
    if is_batched_adj(adj):
        return adj * inv.unsqueeze(-1) * inv.unsqueeze(-2)
    return adj * inv.unsqueeze(0) * inv.unsqueeze(1)


def normalize_row(adj: Tensor, eps: float = 1e-9) -> Tensor:
    deg = adj.sum(-1).clamp(min=eps)
    if is_batched_adj(adj):
        return adj / deg.unsqueeze(-1)
    return adj / deg.unsqueeze(1)


def to_dense_from_edge_index(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    batch_size: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    device = device or edge_index.device
    src, dst = edge_index[0], edge_index[1]
    if batch_size is None:
        adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
        if edge_weight is None:
            adj[dst, src] = 1.0
        else:
            adj[dst, src] = edge_weight
        return adj

    adj = torch.zeros(batch_size, num_nodes, num_nodes, device=device, dtype=dtype)
    if edge_weight is None:
        adj[:, dst, src] = 1.0
    else:
        adj[:, dst, src] = edge_weight
    return adj


def ensure_adj(
    adj: Tensor | None,
    edge_index: Tensor | None,
    num_nodes: int,
    edge_weight: Tensor | None,
    batch_size: int | None,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if adj is not None:
        return adj
    if edge_index is None:
        raise ValueError("Either adj or edge_index must be provided.")
    return to_dense_from_edge_index(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )


def xavier_zero_bias(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def dtype_neg_inf(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        return -65504.0
    if dtype == torch.bfloat16:
        return -3.38e38
    return -1e9


def safe_eye(n: int, like: Tensor) -> Tensor:
    return torch.eye(n, device=like.device, dtype=like.dtype)


__all__ = [
    "ActivationType",
    "AggType",
    "Tensor",
    "add_self_loops",
    "dtype_neg_inf",
    "ensure_adj",
    "is_batched_adj",
    "normalize_gcn",
    "normalize_row",
    "safe_eye",
    "to_dense_from_edge_index",
    "xavier_zero_bias",
]
