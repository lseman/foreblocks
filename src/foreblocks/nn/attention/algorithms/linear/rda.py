"""foreblocks.nn.attention.algorithms.linear.rda.

RDA (Riemannian Distance Attention) with configurable feature map and incremental recurrent decode.

Uses an ELU+1 (or other) feature map to convert Q/K into positive feature space,
then accumulates k_sum and kv_sum for O(L·d²) causal attention. Supports
incremental recurrent decoding for autoregressive generation.

Core API:
- RDABackend: RDA with configurable feature_map and incremental decode

"""

from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.kernels.attention.linear import (
    can_use_fused_recurrent_linear_attn,
    fused_recurrent_causal_linear_attn,
)
from foreblocks.nn.attention.algorithms.linear.base import (
    FeatureMapRegistry,
    RoPEMixin,
    mask_padded_tokens,
    validate_key_padding_mask,
    validate_linear_attention_mask,
)

try:
    from foreblocks.integrations.fla.linear import (
        can_use_fla_linear_attn,
        fla_recurrent_linear_attn_forward,
    )
    from foreblocks.ops.attention.chunked_causal_linear_attention import (
        chunked_causal_linear_attn,
    )
except Exception:
    chunked_causal_linear_attn = None  # type: ignore[assignment]
    can_use_fla_linear_attn = None  # type: ignore[assignment]
    fla_recurrent_linear_attn_forward = None  # type: ignore[assignment]


class RDABackend(RoPEMixin, nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        feature_map: str = "elu",
        num_features: int | None = None,
        pos_encoding_type: str = "sinusoidal",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5
        self.pos_encoding_type = pos_encoding_type
        self._init_pos_encoding()

        if n_heads <= 0 or d_model % n_heads:
            raise ValueError("n_heads must be positive and divide d_model")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.feature_fn = FeatureMapRegistry.build(
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
        validate_linear_attention_mask(attn_mask, query, key, is_causal)
        if is_causal and Lq != Lk:
            raise ValueError("causal RDA requires matching query and key lengths")
        validate_key_padding_mask(key_padding_mask, B, Lk, query.device)
        key, value = (mask_padded_tokens(x, key_padding_mask) for x in (key, value))
        if is_causal:
            query = mask_padded_tokens(query, key_padding_mask)

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
        offset = (
            layer_state.get("rda_position", 0)
            if layer_state is not None and is_causal
            else 0
        )
        q, k = self._apply_rope(q, k, offset, key_padding_mask if is_causal else None)

        q_prime = self.feature_fn(q)
        k_prime = self.feature_fn(k)

        # Key-padding mask
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # B 1 Lk 1
            k_prime = k_prime.masked_fill(pad_mask, 0.0)
            v = v.masked_fill(pad_mask, 0.0)

        # ── Incremental recurrent decode ────────────────────────────────
        if layer_state is not None and is_causal:
            result = self._incremental(q_prime, k_prime, v, layer_state)
            layer_state["rda_position"] = offset + (
                Lq if key_padding_mask is None else (~key_padding_mask).sum(-1)
            )
            return mask_padded_tokens(result[0], key_padding_mask), result[1], result[2]

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
            elif (
                chunked_causal_linear_attn is not None and not torch.jit.is_scripting()
            ):
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
        if is_causal:
            out = mask_padded_tokens(out, key_padding_mask)
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

        if (
            q_prime.size(2) == 1
            and not torch.is_grad_enabled()
            and can_use_fused_recurrent_linear_attn(q_prime, k_prime, v)
        ):
            out_heads, (next_k, next_kv) = fused_recurrent_causal_linear_attn(
                q_prime,
                k_prime,
                v,
                initial_state=(prev_k, prev_kv) if prev_k is not None else None,
                output_final_state=True,
            )
            layer_state.update(k_sum=next_k, kv_sum=next_kv)
            B, _, Lq, _ = out_heads.shape
            out = out_heads.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
            return self.dropout(self.out_proj(out)), None, layer_state

        acc_dtype = torch.promote_types(q_prime.dtype, torch.float32)
        q_acc, k_acc, v_acc = (t.to(acc_dtype) for t in (q_prime, k_prime, v))
        k_cum = torch.cumsum(k_acc, dim=2)
        kv_cum = torch.cumsum(k_acc.unsqueeze(-1) * v_acc.unsqueeze(-2), dim=2)

        if prev_k is not None:
            k_cum = k_cum + prev_k.unsqueeze(2)
            kv_cum = kv_cum + prev_kv.unsqueeze(2)

        denom = torch.sum(q_acc * k_cum, dim=-1, keepdim=True)
        numer = torch.einsum("bhlf,bhlfd->bhld", q_acc, kv_cum)
        out_heads = (numer / (denom + 1e-6)).to(v.dtype)

        layer_state["k_sum"] = k_cum[:, :, -1]
        layer_state["kv_sum"] = kv_cum[:, :, -1]

        B, _, Lq, _ = out_heads.shape
        out = out_heads.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.dropout(self.out_proj(out))
        return out, None, layer_state
