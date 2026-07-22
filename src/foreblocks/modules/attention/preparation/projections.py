"""QKV head projection isolated from attention orchestration."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class ProjectionContext(Protocol):
    d_model: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    use_mla: bool
    q_proj: nn.Module
    k_proj: nn.Module
    v_proj: nn.Module
    kv_down_proj: nn.Module | None
    k_up_proj: nn.Module | None
    v_up_proj: nn.Module | None


class QKVProjector:
    """Project model-space inputs into query, key, and value heads."""

    def __init__(self, context: ProjectionContext) -> None:
        self.context = context

    def project(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        context = self.context
        batch_size, query_length, _ = query.shape
        key_length = key.shape[1]
        q = context.q_proj(query).view(
            batch_size, query_length, context.n_heads, context.head_dim
        ).transpose(1, 2)

        if context.use_mla:
            if (
                context.kv_down_proj is None
                or context.k_up_proj is None
                or context.v_up_proj is None
            ):
                raise RuntimeError("MLA projections are not initialized")
            latent = context.kv_down_proj(key)
            k = context.k_up_proj(latent).view(
                batch_size, key_length, context.n_kv_heads, context.head_dim
            ).transpose(1, 2)
            v = context.v_up_proj(latent).view(
                batch_size, key_length, context.n_kv_heads, context.head_dim
            ).transpose(1, 2)
            return q, k, v, latent

        k = context.k_proj(key).view(
            batch_size, key_length, context.n_kv_heads, context.head_dim
        ).transpose(1, 2)
        v = context.v_proj(value).view(
            batch_size, key_length, context.n_kv_heads, context.head_dim
        ).transpose(1, 2)
        return q, k, v, None


__all__ = ["ProjectionContext", "QKVProjector"]
