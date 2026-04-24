from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

Tensor = torch.Tensor
AggType = Literal["add", "mean", "max"]
ActivationType = Literal["relu", "gelu", "silu", "none"]


def is_batched_adj(adj: Tensor) -> bool:
    return adj.dim() == 3


def add_self_loops(adj: Tensor) -> Tensor:
    """Add self-loops to a dense adjacency matrix.

    Supports both unbatched adjacency matrices shaped [N, N] and batched
    adjacency matrices shaped [B, N, N]. For batched inputs, the identity
    matrix is constructed using ``torch.diag_embed`` for efficiency and
    correctness across devices and dtypes.
    """
    if is_batched_adj(adj):
        batch_size, num_nodes, _ = adj.shape
        eye = torch.diag_embed(
            torch.ones(batch_size, num_nodes, device=adj.device, dtype=adj.dtype)
        )
        return adj + eye
    num_nodes = adj.size(0)
    return adj + torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)


def normalize_gcn(adj: Tensor, eps: float = 1e-9) -> Tensor:
    """Compute symmetric normalized adjacency for graph convolution.

    The returned tensor is D^{-1/2} A D^{-1/2}, where D is the degree matrix
    of the adjacency. Batched adjacency tensors are normalized independently
    for each batch element.
    """
    deg = adj.sum(-1)
    inv = deg.clamp(min=eps).pow(-0.5)
    inv[torch.isinf(inv)] = 0.0
    if is_batched_adj(adj):
        return adj * inv.unsqueeze(-1) * inv.unsqueeze(-2)
    return adj * inv.unsqueeze(0) * inv.unsqueeze(1)


def normalize_row(adj: Tensor, eps: float = 1e-9) -> Tensor:
    """Row-normalize adjacency matrices so each row sums to one.

    Supports both unbatched adjacency matrices shaped [N, N] and batched
    adjacency matrices shaped [B, N, N].
    """
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
    """Convert an edge index representation to a dense adjacency matrix.

    If ``batch_size`` is supplied, this will return a batched adjacency tensor
    shaped [B, N, N]. Otherwise it returns an unbatched adjacency matrix
    shaped [N, N].
    """
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
    """Return a dense adjacency matrix, converting from sparse edge_index if needed."""
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
    """Initialize a linear module with Xavier weight and zero bias."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def dtype_neg_inf(dtype: torch.dtype) -> float:
    """Return a negative infinity sentinel appropriate for the given dtype."""
    if dtype == torch.float16:
        return -65504.0
    if dtype == torch.bfloat16:
        return -3.38e38
    return -1e9


def safe_eye(n: int, like: Tensor) -> Tensor:
    """Create an identity matrix on the same device and dtype as ``like``."""
    return torch.eye(n, device=like.device, dtype=like.dtype)


def crop_residual_to_match(x: Tensor, ref: Tensor) -> Tensor:
    if x.size(1) == ref.size(1):
        return x
    if x.size(1) < ref.size(1):
        raise ValueError(
            "Residual input is shorter than the temporal reference tensor."
        )
    return x[:, -ref.size(1) :, :, :]


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
    "crop_residual_to_match",
]
