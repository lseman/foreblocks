"""
ModernLinearAttention — modular linear attention with swappable backends.

Backends
--------
1. "rda"      : RDA (Riemannian Distance Attention) — original ELU+1 kernel.
                  Configurable feature_map: "elu", "relu", "silu", "leaky_relu".
                  O(L·d²) global, supports incremental recurrent decode.

2. "gla"      : Gated Linear Attention (GLA, Yang et al. 2023).
                  Per-timestep decay gate gk = log-sigmoid(low-rank) / τ.
                  Parallel chunk mode + exact recurrent mode.
                  O(L·d²) global, supports incremental decode.

3. "deltanet" : DeltaNet (Yang et al. 2024).
                  L2-normalised Q, K with causal conv pre-processing and
                  learnable β gate. Parallel WY chunk + exact recurrent.
                  O(L·d²) global, supports incremental decode.

(Gated DeltaNet / Mamba-2 lives in the standalone ``gated_delta.GatedDeltaNet``
module — it has correct, faster chunk kernels, so it is not duplicated here.)

All backends implement the same drop-in API:
    (query, key, value, attn_mask, key_padding_mask, is_causal, layer_state)
    → (out, None, updated_state)

layer_state dict carries recurrent state under key "<backend>_state":
    { "rda_state": {"k_sum": ..., "kv_sum": ...} }
    { "gla_state": {"S": ...} }
    { "deltanet_state": {"S": ...} }
"""

from __future__ import annotations

from math import ceil
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from foreblocks.transformer.kernels.triton_helpers import (
        chunked_causal_linear_attn,
    )
except Exception:
    chunked_causal_linear_attn = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────


class RoPEMixin:
    """
    Per-backend RoPE on Q/K (applied after projection, on [B, H, L, d_head]).

    ALiBi is intentionally a no-op for linear attention: there is no explicit
    [B, H, Lq, Lk] score matrix to add a bias to, and materialising one would
    forfeit the O(L·d²) cost. ``pos_encoding_type="alibi"`` is accepted (so the
    config does not crash) but has no effect — matching LinearAttention.

    Mixin contract: the host module must set ``self.pos_encoding_type`` and
    ``self.d_head`` in ``__init__``; ``_init_pos_encoding()`` sets up the lazy
    state holders.
    """

    def _init_pos_encoding(self) -> None:
        self._rotary_emb: nn.Module | None = None

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to q, k of shape [B, H, L, d_head]. No-op unless enabled."""
        if getattr(self, "pos_encoding_type", "sinusoidal") != "rope":
            return q, k

        from foreblocks.transformer.embeddings.rope_alibi_helpers import (
            create_rotary_embedding,
        )
        from foreblocks.transformer.embeddings.rotary import apply_rotary_emb

        Lq = q.shape[2]
        Lk = k.shape[2]
        seqlen = max(Lq, Lk)
        if self._rotary_emb is None:
            self._rotary_emb = create_rotary_embedding(
                head_dim=self.d_head, max_seq_len=seqlen
            )
        # RotaryEmbedding builds its cos/sin cache lazily and on-device.
        self._rotary_emb._update_cos_sin_cache(
            seqlen, device=q.device, dtype=q.dtype
        )
        cos = self._rotary_emb._cos_cached
        sin = self._rotary_emb._sin_cached
        if cos is None or sin is None:
            raise RuntimeError("RotaryEmbedding cache was not initialized")

        q_rot = apply_rotary_emb(
            q.transpose(1, 2), cos[:Lq], sin[:Lq]
        ).transpose(1, 2)
        k_rot = apply_rotary_emb(
            k.transpose(1, 2), cos[:Lk], sin[:Lk]
        ).transpose(1, 2)
        return q_rot, k_rot


def _causal_conv1d(
    x: torch.Tensor, conv: nn.Conv1d, activation: nn.Module, kernel_size: int
) -> torch.Tensor:
    """Causal depthwise/pointwise Conv1d + activation, crop to original length."""
    T0 = x.size(1)
    x = x.transpose(1, 2).contiguous()  # [B, D, T]
    x = conv(x)[:, :, :T0].contiguous()  # crop causal padding
    return activation(x).transpose(1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Feature maps for RDA backend
# ─────────────────────────────────────────────────────────────────────────────


class FeatureMapRegistry:
    """Factory for linear-attention feature maps."""

    @staticmethod
    def make(name: str, d_head: int, num_features: Optional[int] = None):
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


# ─────────────────────────────────────────────────────────────────────────────
# Backend 1: RDA (Riemannian Distance Attention)
# ─────────────────────────────────────────────────────────────────────────────


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
            if chunked_causal_linear_attn is not None and not torch.jit.is_scripting():
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


# ─────────────────────────────────────────────────────────────────────────────
# Backend 2: Gated Linear Attention (GLA)
# ─────────────────────────────────────────────────────────────────────────────


class GLABackend(RoPEMixin, nn.Module):
    """
    Gated Linear Attention (Yang et al. 2023).

    Recurrence:
        S_t = exp(g_t) · S_{t-1} + k_t · v_t^T
        o_t = q_t · S_t

    where g_t = logsigmoid(low_rank(x)) / τ  (per-channel decay gate).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        gate_logit_normalizer: float = 16.0,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        use_output_gate: bool = True,
        mode: Literal["chunk", "recurrent"] = "chunk",
        chunk_size: int = 64,
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
        assert mode in ("chunk", "recurrent")

        self.gate_logit_normalizer = gate_logit_normalizer
        self.clamp_min = clamp_min
        self.use_output_gate = use_output_gate
        self.mode = mode
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.gk_proj = nn.Sequential(
            nn.Linear(d_model, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, d_model, bias=True),
        )

        if use_output_gate:
            self.g_proj = nn.Linear(d_model, d_model, bias=False)
            self.gate_fn = nn.SiLU()
        else:
            self.g_proj = None

        self.norm = nn.RMSNorm(self.d_head, eps=1e-5)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _chunk_forward(
        self,
        q: torch.Tensor,  # [B, H, L, D]
        k: torch.Tensor,  # [B, H, L, D]
        v: torch.Tensor,  # [B, H, L, D]
        gk: torch.Tensor,  # [B, H, L, D]  (log-decay)
        S: torch.Tensor,  # [B, H, D, D]
    ) -> torch.Tensor:
        B, H, L, D = q.shape
        n_chunks = ceil(L / self.chunk_size)
        last_size = L % self.chunk_size if L % self.chunk_size > 0 else self.chunk_size

        padding = self.chunk_size - last_size if last_size > 0 else 0
        # Right-pad the sequence dim (not head_dim): real tokens stay at [:L] so
        # chunk indices align with real positions; trailing pad is cropped away.
        q_p, k_p, v_p, gk_p = (
            F.pad(x, (0, 0, 0, padding)) for x in (q, k, v, gk)
        )  # each [B, H, L+pad, D]

        # Local cumulative sum of gk within each chunk
        gk_cumsum = gk_p.reshape(B, H, n_chunks, self.chunk_size, D).cumsum(3)
        gk_cumsum = gk_cumsum.reshape(B, H, L + padding, D)

        L_padded = L + padding
        o = torch.zeros(B, H, L_padded, D, device=q.device, dtype=q.dtype)
        S = S.clone()

        for idx in range(n_chunks):
            s = idx * self.chunk_size
            e = s + self.chunk_size  # full chunk; trailing pad cropped at the end
            C = e - s  # actual chunk size

            Q = q_p[:, :, s:e]
            K = k_p[:, :, s:e]
            V = v_p[:, :, s:e]
            G = gk_cumsum[:, :, s:e]

            # Inter-chunk: (Q*scale)*exp(G) @ S
            qg = (Q * self.scale) * torch.exp(G)
            o_inter = torch.einsum("bhlk,bhkv->bhlv", qg, S)

            # Intra-chunk: A = Q_hat @ K_hat^T
            G_base = gk_cumsum[:, :, idx * self.chunk_size]  # [B, H, D]
            Q_hat = Q * torch.exp(G - G_base.unsqueeze(2)) * self.scale
            K_hat = K * torch.exp(G_base.unsqueeze(2) - G)
            A = torch.matmul(Q_hat, K_hat.transpose(-1, -2))  # [B, H, C, C]
            A_mask = torch.tril(
                torch.ones(1, 1, C, C, device=q.device, dtype=torch.bool)
            )
            A = A.masked_fill(~A_mask, 0.0)
            o_intra = torch.einsum("bhls,bhsv->bhlv", A, V)

            o[:, :, s:e] = o_inter + o_intra

            # State carry
            gk_last = G[:, :, -1]
            S = S * torch.exp(gk_last).unsqueeze(-1)

            w = torch.exp(gk_last.unsqueeze(2) - G)
            K_scaled = K * w
            S = S + torch.einsum("bhlk,bhlv->bhkv", K_scaled, V)

        return o[:, :, :L], S  # crop trailing padding

    def _recurrent_forward(
        self,
        q: torch.Tensor,  # [B, H, L, D]
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        S: torch.Tensor,  # [B, H, D, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, L, D = q.shape
        outputs = []

        for t in range(L):
            k_t = k[:, :, t]  # [B, H, D]
            q_t = q[:, :, t]  # [B, H, D]
            v_t = v[:, :, t]  # [B, H, D]
            gk_t = gk[:, :, t]  # [B, H, D]

            update = torch.einsum("bhk,bhv->bhkv", k_t, v_t)
            S = torch.exp(gk_t).unsqueeze(-1) * S + update

            q_scaled = q_t / (self.d_head**0.5)
            o_t = torch.einsum("bhk,bhkv->bhv", q_scaled, S)
            outputs.append(o_t)

        return torch.stack(outputs, dim=2), S

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        layer_state: dict | None = None,
    ) -> tuple[torch.Tensor, None, dict | None]:
        B, L, _ = query.shape

        q = self.q_proj(query).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        gk = self.gk_proj(query).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # RoPE on Q/K after projection (no-op unless pos_encoding_type="rope")
        q, k = self._apply_rope(q, k)

        # Decay gate: g = log-sigmoid(low_rank) / τ, clamped
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        # Key-padding mask (zero out k, v)
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(2).unsqueeze(3)
            k = k.masked_fill(pad_mask, 0.0)
            v = v.masked_fill(pad_mask, 0.0)

        # Recurrent state
        S = None
        if layer_state is not None:
            raw = layer_state.get("gla_S")
            if isinstance(raw, torch.Tensor):
                S = raw.to(query.device, query.dtype)

        if S is None:
            S = query.new_zeros((B, self.n_heads, self.d_head, self.d_head))

        o, S_next = (
            self._chunk_forward(q, k, v, gk, S)
            if self.mode == "chunk"
            else self._recurrent_forward(q, k, v, gk, S)
        )
        # o: [B, H, L, D]

        # Output gate
        if self.use_output_gate and self.g_proj is not None:
            gate = self.gate_fn(self.g_proj(query)).view(
                B, L, self.n_heads, self.d_head
            )  # [B, L, H, D]
            o = o * gate.transpose(1, 2)  # [B, H, L, D] * [B, H, L, D]

        # Output norm
        o = self.norm(o)

        out = o.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.dropout(self.out_proj(out))

        return out, None, {"gla_S": S_next} if is_causal else {"gla_S": S_next}


# ─────────────────────────────────────────────────────────────────────────────
# Backend 3: DeltaNet
# ─────────────────────────────────────────────────────────────────────────────


class DeltaNetBackend(nn.Module):
    """
    DeltaNet (Yang et al. 2024).

    L2-normalised Q, K with causal conv pre-processing and learnable β gate.
    Recurrence:
        S_t = S_{t-1} - β_t · (S_{t-1}·k_t - v_t) · k_t^T
        o_t = q_t · S_t

    Parallel chunk via WY representation.

    Note: RoPE is intentionally not applied — DeltaNet relies on its causal
    conv + delta-rule positional structure. ``pos_encoding_type`` is accepted
    for a uniform backend API but does not add rotary embeddings.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        conv_size: int = 4,
        mode: Literal["chunk", "recurrent"] = "chunk",
        chunk_size: int = 64,
        pos_encoding_type: str = "sinusoidal",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5
        self.pos_encoding_type = pos_encoding_type

        assert d_model % n_heads == 0
        assert mode in ("chunk", "recurrent")

        self.conv_size = conv_size
        self.mode = mode
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)

        self.k_conv = self._causal_conv(d_model, conv_size)
        self.q_conv = self._causal_conv(d_model, conv_size)
        self.v_conv = self._causal_conv(d_model, conv_size)

        self.norm = nn.RMSNorm(self.d_head, eps=1e-5)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal_conv(d_model: int, kernel_size: int) -> nn.Conv1d:
        return nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size - 1,
            bias=False,
        )

    def _causal_conv_forward(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        return _causal_conv1d(x, conv, nn.SiLU(), self.conv_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        layer_state: dict | None = None,
    ) -> tuple[torch.Tensor, None, dict | None]:
        B, L, _ = query.shape

        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)

        # Causal conv pre-processing
        k = (
            self
            ._causal_conv_forward(k, self.k_conv)
            .reshape(B, L, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        q = (
            self
            ._causal_conv_forward(q, self.q_conv)
            .reshape(B, L, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self
            ._causal_conv_forward(v, self.v_conv)
            .reshape(B, L, self.n_heads, self.d_head)
            .transpose(1, 2)
        )

        # L2 normalise
        k = F.normalize(k, p=2, dim=-1)
        q = F.normalize(q, p=2, dim=-1)

        # β gate
        beta = self.b_proj(query).sigmoid().unsqueeze(-1)  # [B, L, H, 1]
        beta = beta.transpose(1, 2)  # [B, H, L, 1]

        # Recurrent state
        S = None
        if layer_state is not None:
            raw = layer_state.get("deltanet_S")
            if isinstance(raw, torch.Tensor):
                S = raw.to(query.device, query.dtype)

        if S is None:
            S = query.new_zeros((B, self.n_heads, self.d_head, self.d_head))

        o, S_next = (
            self._chunk_forward(q, k, v, S, beta)
            if self.mode == "chunk"
            else self._recurrent_forward(q, k, v, S, beta)
        )

        # Output norm
        o = self.norm(o)
        out = o.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.dropout(self.out_proj(out))

        return out, None, {"deltanet_S": S_next}

    def _recurrent_forward(self, q, k, v, S, beta):
        # State S is indexed [v, k] to match _chunk_forward's convention:
        #   pred_v = S @ k                       (current value estimate)
        #   S     += beta * (v - pred) ⊗ k       (delta-rule rank-1 update)
        #   o      = (q / √d) · Sᵀ               (readout, == q @ S along k)
        B, H, L, D = q.shape
        outputs = []
        scale = self.d_head**0.5

        for t in range(L):
            k_t, q_t, v_t = k[:, :, t], q[:, :, t], v[:, :, t]  # [B, H, D]
            beta_t = beta[:, :, t]  # [B, H, 1]

            pred = torch.einsum("bhvk,bhk->bhv", S, k_t)  # [B, H, D_v]
            delta = beta_t * (v_t - pred)  # [B, H, D_v]
            S = S + torch.einsum("bhv,bhk->bhvk", delta, k_t)  # [B, H, D_v, D_k]

            o_t = torch.einsum("bhvk,bhk->bhv", S, q_t / scale)  # [B, H, D_v]
            outputs.append(o_t)

        return torch.stack(outputs, dim=2), S  # [B, H, L, D]

    def _chunk_forward(self, q, k, v, S, beta):
        B, H, L, D = q.shape
        n_chunks = ceil(L / self.chunk_size)
        last_size = L % self.chunk_size if L % self.chunk_size > 0 else self.chunk_size

        padding = self.chunk_size - last_size if last_size > 0 else 0
        # Right-pad the sequence dim (not d_head): real tokens stay at [:L] so
        # causal alignment is preserved and the trailing pad is cropped away.
        q_p, k_p, v_p, beta_p = (F.pad(x, (0, 0, 0, padding)) for x in (q, k, v, beta))

        S = S.clone()
        L_padded = L + padding
        o = torch.zeros(B, H, L_padded, D, device=q.device, dtype=q.dtype)

        for idx in range(n_chunks):
            s = idx * self.chunk_size
            e = min(s + self.chunk_size, L_padded)
            C = e - s

            Q = q_p[:, :, s:e]
            K = k_p[:, :, s:e]
            V = v_p[:, :, s:e]
            B_t = beta_p[:, :, s:e]

            K_b = K * B_t  # β·k   [B, H, C, D]
            S_swapped = S.transpose(-1, -2)  # Sᵀ = [k, v]   [B, H, D, D]

            # WY representation of this chunk's delta updates.
            #   T = (I + tril(β·K·Kᵀ, -1))⁻¹
            #   u = T · (β·V − β·K·Sᵀ)   ("pseudo-values", β folded in)
            I_C = (
                torch
                .eye(C, device=Q.device, dtype=Q.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(B, H, 1, 1)
            )
            A_mat = I_C + torch.tril(K_b @ K.transpose(-1, -2), -1)
            T = torch.linalg.solve_triangular(
                A_mat.float(), I_C.float(), upper=False
            ).to(Q.dtype)
            U = T @ (B_t * V - K_b @ S_swapped)  # [B, H, C, D]  pseudo-values

            # Output: inter-chunk readout from the old state + causal intra-chunk
            # contribution of this chunk's own pseudo-values. Diagonal is included
            # since token t reads the state *after* writing its own update.
            Q_scaled = Q / self.d_head**0.5
            M = (
                torch
                .tril(torch.ones(C, C, device=Q.device, dtype=Q.dtype))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            A_intra = (Q_scaled @ K.transpose(-1, -2)) * M
            O = Q_scaled @ S_swapped + A_intra @ U  # [B, H, C, D]

            o[:, :, s:e] = O

            # State carry: S += Σ_t u_t ⊗ k_t   (S is [v, k])
            S = S + U.transpose(-1, -2) @ K

        # Crop to original sequence length
        o = o[:, :, :L]
        return o, S


# ─────────────────────────────────────────────────────────────────────────────
# Router: ModernLinearAttention
# ─────────────────────────────────────────────────────────────────────────────

_BACKEND_MAP = {
    "rda": RDABackend,
    "gla": GLABackend,
    "deltanet": DeltaNetBackend,
}


class ModernLinearAttention(nn.Module):
    """
    Modular linear attention with swappable backends.

    Parameters
    ----------
    d_model : int
    n_heads : int
    dropout : float
    backend : str
        One of "rda", "gla", "deltanet". (Gated DeltaNet lives in the
        standalone ``gated_delta.GatedDeltaNet`` module.)
    state : str, optional
        Feature map for "rda" backend: "elu", "relu", "silu", "leaky_relu",
        "rff", "tanh", "cos_cos". Default: "elu".
    mode : str, optional
        Computation mode for backends that support it: "chunk" (parallel) or
        "recurrent" (exact sequential). Default: "chunk".
    chunk_size : int, optional
        Chunk size for chunk-mode backends. Default: 64.
    **backend_kwargs :
        Extra kwargs forwarded to the backend constructor.

    Example
    -------
    >>> attn = ModernLinearAttention(
    ...     d_model=256, n_heads=8, dropout=0.1,
    ...     backend="gla", mode="chunk", chunk_size=64
    ... )
    >>> out, _, state = attn(q, k, v, is_causal=True, layer_state=None)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        backend: Literal["rda", "gla", "deltanet"] = "rda",
        state: str = "elu",
        mode: Literal["chunk", "recurrent"] = "chunk",
        chunk_size: int = 64,
        pos_encoding_type: str = "sinusoidal",
        **backend_kwargs,
    ):
        super().__init__()
        self.backend_name = backend
        self.d_model = d_model
        self.n_heads = n_heads
        self.mode = mode
        self.pos_encoding_type = pos_encoding_type

        if backend not in _BACKEND_MAP:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {sorted(_BACKEND_MAP.keys())}"
            )

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        # Pass relevant kwargs to backend
        kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            pos_encoding_type=pos_encoding_type,
        )
        if backend == "rda":
            kwargs["feature_map"] = state
            kwargs["num_features"] = backend_kwargs.get("num_features")
        elif backend == "gla":
            kwargs["mode"] = mode
            kwargs["chunk_size"] = chunk_size
            kwargs["gate_logit_normalizer"] = backend_kwargs.get(
                "gate_logit_normalizer", 16.0
            )
            kwargs["gate_low_rank_dim"] = backend_kwargs.get("gate_low_rank_dim", 16)
            kwargs["clamp_min"] = backend_kwargs.get("clamp_min")
            kwargs["use_output_gate"] = backend_kwargs.get("use_output_gate", True)
        elif backend == "deltanet":
            kwargs["mode"] = mode
            kwargs["chunk_size"] = chunk_size
            kwargs["conv_size"] = backend_kwargs.get("conv_size", 4)

        self.impl = _BACKEND_MAP[backend](**kwargs)

    @property
    def state_key(self) -> str:
        """Key used in layer_state dict for this backend's recurrent state."""
        return f"{self.backend_name}_S"

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
        # Positional encoding (RoPE/ALiBi) is handled inside each backend, after
        # its own Q/K projection — applying it here would double-project.
        return self.impl(
            query, key, value, attn_mask, key_padding_mask, is_causal, layer_state
        )

    def reset_state(self, layer_state: dict) -> None:
        """Clear recurrent state for this backend."""
        layer_state.pop(self.state_key, None)
