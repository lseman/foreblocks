"""foreblocks.modules.attention.modules.linear_att.rda.

This module implements the rda pieces for its package.
It belongs to the linear and gated attention backends area of Foreblocks.
It exposes classes such as RDABackend.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from foreblocks.modules.attention.modules.linear_att.base import (
    FeatureMapRegistry,
    RoPEMixin,
)


try:
    from foreblocks.ops.attention.chunked_causal_linear_attention import (
        chunked_causal_linear_attn,
    )
    from foreblocks.ops.attention.fla_linear_attention import (
        can_use_fla_linear_attn,
        fla_recurrent_linear_attn_forward,
    )
except Exception:
    chunked_causal_linear_attn = None  # type: ignore[assignment]
    can_use_fla_linear_attn = None  # type: ignore[assignment]
    fla_recurrent_linear_attn_forward = None  # type: ignore[assignment]



class RDABackend(RoPEMixin, nn.Module):
    """RDA with configurable feature map + incremental recurrent decode."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        feature_map: str = "elu",
        num_features: Optional[int] = None,
        pos_encoding_type: str = "sinusoidal",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5
        self.pos_encoding_type = pos_encoding_type
        self._init_pos_encoding()

        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.feature_fn = FeatureMapRegistry.make(
            feature_map, self.d_head, num_features
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        layer_state: dict | None = None,
    ) -> tuple[torch.Tensor, None, dict | None]:
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        q = (
            self.q_proj(query).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        )  # B H Lq Dh
        k = (
            self.k_proj(key).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        )  # B H Lk Dh
        v = (
            self.v_proj(value).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        )  # B H Lk Dh

        q = q * self.scale
        k = k * self.scale

        # RoPE on Q/K before the feature map (no-op unless pos_encoding_type="rope")
        q, k = self._apply_rope(q, k)

        q_prime = self.feature_fn(q)
        k_prime = self.feature_fn(k)

        # Key-padding mask
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # B 1 Lk 1
            k_prime = k_prime.masked_fill(pad_mask, 0.0)
            v = v.masked_fill(pad_mask, 0.0)

        # ── Incremental recurrent decode ────────────────────────────────
        if layer_state is not None and is_causal:
            return self._incremental(q_prime, k_prime, v, layer_state)

        # ── Causal global ───────────────────────────────────────────────
        if is_causal:
            if (
                not torch.is_grad_enabled()
                and can_use_fla_linear_attn is not None
                and fla_recurrent_linear_attn_forward is not None
                and can_use_fla_linear_attn(q_prime, k_prime, v)
            ):
                out_heads = fla_recurrent_linear_attn_forward(
                    q_prime, k_prime, v, eps=1e-6
                )
            elif chunked_causal_linear_attn is not None and not torch.jit.is_scripting():
                # Chunk-parallel scan: differentiable, no O(B·H·T·F·Dh) intermediate.
                out_heads = chunked_causal_linear_attn(
                    q_prime, k_prime, v, chunk_size=128, eps=1e-6
                )
            else:
                # Fallback — creates O(B·H·T·F·Dh) intermediate (import unavailable)
                k_cum = torch.cumsum(k_prime, dim=2)
                kv_cum = torch.cumsum(k_prime.unsqueeze(-1) * v.unsqueeze(-2), dim=2)
                denom = torch.sum(q_prime * k_cum, dim=-1, keepdim=True)
                numer = torch.einsum("bhlf,bhlfd->bhld", q_prime, kv_cum)
                out_heads = numer / (denom + 1e-6)
        else:
            # Non-causal: fused einsum, O(B·H·L·d²) without O(L²) intermediate
            k_sum = k_prime.sum(dim=2)
            kv_sum = torch.einsum("bhlf,bhld->bhfd", k_prime, v)
            denom = torch.matmul(q_prime, k_sum.unsqueeze(-1))
            numer = torch.matmul(q_prime, kv_sum)
            out_heads = numer / (denom + 1e-6)

        out = out_heads.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.dropout(self.out_proj(out))
        return out, None, None

    def _incremental(
        self,
        q_prime: torch.Tensor,
        k_prime: torch.Tensor,
        v: torch.Tensor,
        layer_state: dict,
    ) -> tuple[torch.Tensor, None, dict]:
        prev_k = layer_state.get("k_sum")
        prev_kv = layer_state.get("kv_sum")

        k_cum = torch.cumsum(k_prime, dim=2)
        kv_cum = torch.cumsum(k_prime.unsqueeze(-1) * v.unsqueeze(-2), dim=2)

        if prev_k is not None:
            k_cum = k_cum + prev_k.unsqueeze(2)
            kv_cum = kv_cum + prev_kv.unsqueeze(2)

        denom = torch.sum(q_prime * k_cum, dim=-1, keepdim=True)
        numer = torch.einsum("bhlf,bhlfd->bhld", q_prime, kv_cum)
        out_heads = numer / (denom + 1e-6)

        layer_state["k_sum"] = k_cum[:, :, -1]
        layer_state["kv_sum"] = kv_cum[:, :, -1]

        B, _, Lq, _ = out_heads.shape
        out = out_heads.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.dropout(self.out_proj(out))
        return out, None, layer_state
