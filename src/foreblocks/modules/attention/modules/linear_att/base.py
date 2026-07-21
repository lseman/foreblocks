"""foreblocks.modules.attention.modules.linear_att.base.

Shared utilities and mixins for linear attention backends.

Provides RoPEMixin for per-backend rotary position embedding on Q/K tensors,
causal Conv1d helper, and FeatureMapRegistry for configurable linear attention
feature maps (ELU+1, ReLU, SiLU, RFF, etc.). All linear backends inherit from
RoPEMixin; FeatureMapRegistry is used by the RDA backend.

Core API:
- RoPEMixin: per-backend RoPE on Q/K with lazy embedding initialization
- FeatureMapRegistry: factory for linear-attention feature maps
- _causal_conv1d: causal Conv1d + activation helper

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPEMixin:
    def _init_pos_encoding(self) -> None:
        self._rotary_emb: nn.Module | None = None

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "pos_encoding_type", "sinusoidal") != "rope":
            return q, k

        from foreblocks.layers.embeddings.rope_alibi_helpers import (
            create_rotary_embedding,
        )
        from foreblocks.layers.embeddings.rotary import apply_rotary_emb

        Lq = q.shape[2]
        Lk = k.shape[2]
        seqlen = max(Lq, Lk)
        if self._rotary_emb is None:
            self._rotary_emb = create_rotary_embedding(
                head_dim=self.d_head, max_seq_len=seqlen
            )
        # RotaryEmbedding builds its cos/sin cache lazily and on-device.
        self._rotary_emb._update_cos_sin_cache(seqlen, device=q.device, dtype=q.dtype)
        cos = self._rotary_emb._cos_cached
        sin = self._rotary_emb._sin_cached
        if cos is None or sin is None:
            raise RuntimeError("RotaryEmbedding cache was not initialized")

        q_rot = apply_rotary_emb(q.transpose(1, 2), cos[:Lq], sin[:Lq]).transpose(1, 2)
        k_rot = apply_rotary_emb(k.transpose(1, 2), cos[:Lk], sin[:Lk]).transpose(1, 2)
        return q_rot, k_rot


def _causal_conv1d(
    x: torch.Tensor, conv: nn.Conv1d, activation: nn.Module, kernel_size: int
) -> torch.Tensor:
    T0 = x.size(1)
    x = x.transpose(1, 2).contiguous()  # [B, D, T]
    x = conv(x)[:, :, :T0].contiguous()  # crop causal padding
    return activation(x).transpose(1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Feature maps for RDA backend
# ─────────────────────────────────────────────────────────────────────────────


class FeatureMapRegistry:
    @staticmethod
    def make(name: str, d_head: int, num_features: int | None = None):
        if name == "elu":
            return lambda x: F.elu(x) + 1.0
        if name == "relu":
            return F.relu
        if name == "silu":
            return F.silu
        if name == "leaky_relu":
            return F.leaky_relu
        if name == "rff":
            # Random Fourier features — Performed-style
            omega = nn.Parameter(
                torch.randn(1, d_head, num_features or d_head)
                * (1.0 / (num_features or d_head)) ** 0.5,
                requires_grad=False,
            )
            return lambda x: (
                torch.exp(-0.5 * (x**2).sum(-1, keepdim=True))
                * torch.cos(torch.einsum("...d,df->...f", x, omega))
            )
        if name == "tanh":
            return lambda x: torch.tanh(x) + 1.0  # keep non-negative
        if name == "cos_cos":
            # Cosine-Cosine: phi(x) = cos(x), but requires L2-normalised inputs
            return torch.cos
        raise ValueError(f"Unknown feature_map: {name}")
