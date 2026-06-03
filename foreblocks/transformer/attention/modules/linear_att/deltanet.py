from __future__ import annotations

from math import ceil
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import _causal_conv1d


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


