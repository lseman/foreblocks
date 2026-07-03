"""foreblocks.layers.embeddings.alibi_bias.

ALiBi (Attention with Linear Biases) positional encoding.

Implements ALiBi from Press et al. (2022) — a position-independent
positional encoding that applies a linear distance penalty to attention
scores per head. Compatible with chunked attention and linear attention
backends. Provides length extrapolation without position limit.

Core API:
- ALiBiPositionalBias: ALiBi positional bias layer

"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ALiBiPositionalBias(nn.Module):
    """
    ALiBi positional bias applied to attention scores.

    The bias for each head h is:
        bias[i, j] = -slope_h * |i - j|
    where slope_h = 2^(-8*h/H) and H is the number of heads.

    This provides a monotonic distance penalty that encourages
    local attention while allowing global reach.

    Args:
        num_heads: Number of attention heads (for slope calculation).
        max_seq_len: Maximum sequence length for cache sizing.
        slopes: Optional list of per-head slopes (auto-computed if None).
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 4096,
        slopes: List[float] | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        if slopes is None:
            # Standard ALiBi slopes: 2^(-8*h/H)
            slopes = [2.0 ** (-(8 + h) / num_heads) for h in range(num_heads)]
        self.register_buffer(
            "slopes",
            torch.tensor(slopes, dtype=torch.float32).view(1, num_heads, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "bias_buffer",
            torch.empty(0, 0, 0, 0),  # lazily allocated
            persistent=False,
        )

    def _update_bias(self, seqlen_q: int, seqlen_k: int, device: torch.device) -> None:
        """Lazily allocate and compute the ALiBi bias matrix."""
        cached = self.bias_buffer
        if (
            cached.numel() > 0
            and cached.shape[-2] >= seqlen_q
            and cached.shape[-1] >= seqlen_k
            and cached.device == device
        ):
            return  # cache valid

        # [1, H, Tq, Tk] bias matrix
        rel_pos = torch.arange(seqlen_q, device=device, dtype=torch.float32).unsqueeze(
            1
        ) - torch.arange(  # [Tq, 1]
            seqlen_k, device=device, dtype=torch.float32
        ).unsqueeze(
            0
        )  # [1, Tk]  # [Tq, Tk]
        abs_rel = torch.abs(rel_pos)  # [Tq, Tk]
        bias = -(self.slopes * abs_rel.unsqueeze(0)).to(device=device)  # [1, H, Tq, Tk]
        self.bias_buffer = bias

    def forward(
        self,
        seqlen_q: int,
        seqlen_k: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute ALiBi bias matrix.

        Args:
            seqlen_q: Query sequence length.
            seqlen_k: Key sequence length.
            device: Target device.

        Returns:
            Bias tensor [1, num_heads, seqlen_q, seqlen_k].
        """
        if device is None:
            device = self.slopes.device
        self._update_bias(seqlen_q, seqlen_k, device)
        return self.bias_buffer[:, :, :seqlen_q, :seqlen_k]

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, max_seq_len={self.max_seq_len}, "
            f"slopes={self.slopes.squeeze().tolist()}"
        )
