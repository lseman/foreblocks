"""foreblocks.modules.attention.modules.linear_att.gla.

Gated Linear Attention: per-timestep decay gate for recurrent state accumulation.

Implements the GLA recurrence (S_t = exp(g_t) · S_{t-1} + k_t · v_t^T) where
g_t = log-sigmoid(low-rank(x)) / τ is a per-channel decay gate. Supports both
parallel chunk mode and exact recurrent mode. Use when you need linear-time
sequence modeling with learned per-timestep forgetting.

Core API:
- GLABackend: gated linear attention with RoPE support and chunk/recurrent modes

"""

from __future__ import annotations

from math import ceil
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.modules.attention.modules.linear_att.base import RoPEMixin
from foreblocks.ops.attention import can_use_fla_gla, fla_gla_forward


class GLABackend(RoPEMixin, nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        gate_logit_normalizer: float = 16.0,
        gate_low_rank_dim: int = 16,
        clamp_min: float | None = None,
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

        if n_heads <= 0 or d_model % n_heads:
            raise ValueError("n_heads must be positive and divide d_model")
        if mode not in {"chunk", "recurrent"}:
            raise ValueError("mode must be 'chunk' or 'recurrent'")

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
        _, _, L, _ = q.shape
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

        # L2 normalise (matches DeltaNet/Kimi/GatedDeltaNet Q/K-norm convention)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

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

        if can_use_fla_gla(q, k, v, gk, S):
            o, S_next = fla_gla_forward(q, k, v, gk, S, self.scale, self.mode)
        else:
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

        return out, None, {"gla_S": S_next}
