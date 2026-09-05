"""Storage allocation strategies for paged attention caches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PagedStorage:
    key: Tensor | None
    value: Tensor | None
    beta: Tensor | None
    latent: Tensor | None
    position: Tensor


class DensePagedStorage:
    @staticmethod
    def allocate(batch, heads, blocks, block_size, head_dim, *, device, dtype):
        shape = (batch, heads, blocks, block_size, head_dim)
        return PagedStorage(
            key=torch.empty(shape, device=device, dtype=dtype),
            value=torch.empty(shape, device=device, dtype=dtype),
            beta=torch.zeros(
                (batch, heads, blocks, block_size), device=device, dtype=dtype
            ),
            latent=None,
            position=torch.empty(
                (batch, blocks, block_size), device=device, dtype=torch.long
            ),
        )


class LatentPagedStorage:
    @staticmethod
    def allocate(batch, blocks, block_size, latent_dim, *, device, dtype):
        return PagedStorage(
            key=None,
            value=None,
            beta=None,
            latent=torch.empty(
                (batch, blocks, block_size, latent_dim), device=device, dtype=dtype
            ),
            position=torch.empty(
                (batch, blocks, block_size), device=device, dtype=torch.long
            ),
        )


__all__ = ["DensePagedStorage", "LatentPagedStorage", "PagedStorage"]
