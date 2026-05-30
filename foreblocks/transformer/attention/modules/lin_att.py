from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from foreblocks.transformer.kernels.triton_helpers import (
        HAS_TRITON as _HAS_TRITON,
    )
    from foreblocks.transformer.kernels.triton_helpers import (
        triton_causal_linear_attn,
    )
except Exception:
    _HAS_TRITON = False
    triton_causal_linear_attn = None  # type: ignore[assignment]


class LinearAttention(nn.Module):
    """
    State-of-the-art linear attention using positive kernel approximation (ELU+1 feature map).
    O(L * d^2) complexity; drop-in for MultiAttention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        attention_type: str = "standard",  # Ignored; always linear
        freq_modes: int = 16,  # Ignored
        cross_attention: bool = False,  # Ignored (separate Q/K/V projs anyway)
        feature_map: str = "elu",  # "elu" (default), "relu", or "rff" (random Fourier features)
        num_features: int | None = None,  # For "rff"; else uses d_head
        # Positional encoding options
        pos_encoding_type: str = "sinusoidal",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.feature_map = feature_map
        self.num_features = num_features or self.d_head
        self.cross_attention = cross_attention  # For future incremental tweaks
        self.scale = self.d_head**-0.5
        self.pos_encoding_type = pos_encoding_type

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Positional encoding modules (initialized on demand)
        self._rotary_emb: nn.Module | None = None
        self._alibi_bias: nn.Module | None = None

        if self.feature_map == "rff":
            # Random projection for unbiased softmax approx (Performer-style)
            self.omega = nn.Parameter(
                torch.randn(n_heads, self.d_head, self.num_features)
                * (1.0 / self.num_features**0.5),
                requires_grad=False,
            )

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map phi(x)."""
        if self.feature_map == "elu":
            return F.elu(x) + 1.0
        elif self.feature_map == "relu":
            return F.relu(x)
        elif self.feature_map == "rff":
            # Simplified RFF: cos(proj) * exp(-||x||^2 / 2); project first
            proj = torch.einsum("b h l d, h d f -> b h l f", x, self.omega)
            norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
            return torch.exp(-0.5 * norm_sq) * torch.cos(proj)
        else:
            raise ValueError(f"Unknown feature_map: {self.feature_map}")

    def _incremental(
        self,
        q_prime: torch.Tensor,  # B H Lq F
        k_prime: torch.Tensor,  # B H Lk F
        v: torch.Tensor,  # B H Lk Dh
        layer_state: dict,
    ) -> torch.Tensor:
        """
        Causal linear attention with a carried recurrent state.

        State (stored under "lin_kv" / "lin_k"):
            k_sum  [B, H, F]      = Σ_j φ(k_j)
            kv_sum [B, H, F, Dh]  = Σ_j φ(k_j) v_jᵀ

        For each query position the prefix sums must include keys up to and
        including that position, so we cumsum the new chunk and add the prior
        running totals. Equivalent to the cumsum branch, but resumable.
        """
        prev_k = layer_state.get("lin_k")  # B H F  or None
        prev_kv = layer_state.get("lin_kv")  # B H F Dh or None

        # Running prefix sums within this chunk (inclusive).
        k_cum = torch.cumsum(k_prime, dim=2)  # B H Lq F
        kv_cum = torch.cumsum(
            k_prime.unsqueeze(-1) * v.unsqueeze(-2), dim=2
        )  # B H Lq F Dh

        if prev_k is not None:
            k_cum = k_cum + prev_k.unsqueeze(2)
            kv_cum = kv_cum + prev_kv.unsqueeze(2)

        denom = torch.sum(q_prime * k_cum, dim=-1, keepdim=True)  # B H Lq 1
        numer = torch.einsum("bhlf,bhlfd->bhld", q_prime, kv_cum)  # B H Lq Dh
        out_heads = numer / (denom + 1e-6)

        # Persist the running totals (last position of the inclusive cumsum).
        layer_state["lin_k"] = k_cum[:, :, -1]
        layer_state["lin_kv"] = kv_cum[:, :, -1]
        return out_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        layer_state: dict | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # Linear projections + reshape/transpose
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

        # Apply positional encoding (RoPE on Q/K before feature map)
        if self.pos_encoding_type == "rope":
            if self._rotary_emb is None:
                # Initialize rotary embedding on first use
                from foreblocks.transformer.embeddings.rope_alibi_helpers import (
                    create_rotary_embedding,
                )
                self._rotary_emb = create_rotary_embedding(
                    head_dim=self.d_head, max_seq_len=max(Lq, Lk)
                )
            # Initialize cache (RotaryEmbedding computes lazily)
            self._rotary_emb._update_cos_sin_cache(
                max(Lq, Lk), device=q.device, dtype=q.dtype
            )
            q, k = _apply_rope_to_qk(q, k, self._rotary_emb)

        # Apply feature map
        q_prime = self._feature_map(q)  # B H Lq F (F= num_features or Dh)
        k_prime = self._feature_map(k)  # B H Lk F

        # Handle padding mask
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # B 1 Lk 1
            k_prime = k_prime.masked_fill(pad_mask, 0.0)
            v = v.masked_fill(pad_mask, 0.0)

        # ── Incremental causal decoding (KV-cache via recurrent state) ────────
        # Carries the linear-attention state (Σφ(k), Σφ(k)vᵀ) across steps so
        # each call costs O(F·Dh) instead of recomputing the full prefix.
        if layer_state is not None and is_causal:
            out_heads = self._incremental(q_prime, k_prime, v, layer_state)
            out = (
                out_heads.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
            )
            out = self.dropout(self.out_proj(out))
            return out, None, layer_state

        # Compute linear attention (global sum or causal cumsum)
        if is_causal:
            # Causal: sequential scan or cumsum
            assert Lq == Lk, "Causal mode requires Lq == Lk"
            if (
                _HAS_TRITON
                and q_prime.is_cuda
                and not torch.is_grad_enabled()
                and not torch.jit.is_scripting()
            ):
                # Streaming Triton scan: O(B·H·T·(F+Dh)) memory, no huge intermediate
                out_heads = triton_causal_linear_attn(q_prime, k_prime, v, eps=1e-6)
            else:
                # PyTorch fallback — creates O(B·H·T·F·Dh) intermediate
                k_cum = torch.cumsum(k_prime, dim=2)  # B H L F
                kv_cum = torch.cumsum(
                    k_prime.unsqueeze(-1) * v.unsqueeze(-2), dim=2
                )  # B H L F Dh
                denom = torch.sum(q_prime * k_cum, dim=-1, keepdim=True)  # B H L 1
                # Per-timestep contraction over the feature axis F:
                #   numer[b,h,l,d] = Σ_f q'[b,h,l,f] · kv_cum[b,h,l,f,d]
                numer = torch.einsum("bhlf,bhlfd->bhld", q_prime, kv_cum)  # B H L Dh
                out_heads = numer / (denom + 1e-6)
        else:
            # Non-causal: fused einsum avoids O(B·H·L·F·Dh) intermediate tensor
            k_sum = k_prime.sum(dim=2)  # B H F
            kv_sum = torch.einsum("bhlf,bhld->bhfd", k_prime, v)  # B H F Dh
            denom = torch.matmul(q_prime, k_sum.unsqueeze(-1))  # B H L 1
            numer = torch.matmul(q_prime, kv_sum)  # B H L Dh
            out_heads = numer / (denom + 1e-6)

        # Apply ALiBi bias to attention scores (before softmax/normalization)
        if self.pos_encoding_type == "alibi" and self._alibi_bias is not None:
            # Linear attention doesn't have explicit attention scores,
            # so ALiBi is not applicable here. The bias would need to be
            # applied differently (e.g., as a multiplicative factor).
            pass
        elif self.pos_encoding_type == "alibi" and self._alibi_bias is None:
            # Initialize ALiBi on first use
            from foreblocks.transformer.embeddings.rope_alibi_helpers import (
                create_alibi_bias,
            )
            self._alibi_bias = create_alibi_bias(
                num_heads=self.n_heads, max_seq_len=max(Lq, Lk)
            )

        # Reshape + project + dropout
        out = out_heads.transpose(1, 2).contiguous().view(B, Lq, self.d_model)  # B L D
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, None, None


def _apply_rope_to_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    rotary_emb: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K tensors.

    Args:
        q: Query tensor [B, H, L, D]
        k: Key tensor [B, H, L, D]
        rotary_emb: RotaryEmbedding module with cached cos/sin

    Returns:
        (q_rotated, k_rotated) same shapes as input
    """
    B, H, L, D = q.shape

    # Get cached cos/sin from RotaryEmbedding
    # rotary_emb._cos_cached: [seqlen, D/2] → [L, D/2]
    cos = rotary_emb._cos_cached[:L].unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
    sin = rotary_emb._sin_cached[:L].unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]

    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # Rotate the first D/2 dims, keep rest unchanged
    q_rot = _rotate_half(q[..., : D // 2])
    q_out = torch.cat(
        [q[..., : D // 2] * cos + q_rot * sin, q[..., D // 2 :]], dim=-1
    )

    k_rot = _rotate_half(k[..., : D // 2])
    k_out = torch.cat(
        [k[..., : D // 2] * cos + k_rot * sin, k[..., D // 2 :]], dim=-1
    )

    return q_out, k_out
