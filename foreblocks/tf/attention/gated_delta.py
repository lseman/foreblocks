# -*- coding: utf-8 -*-
"""
gated_delta.py — Gated Delta Network Attention

Based on: "Gated Delta Networks: Improving Mamba2 with Delta Rule" (Yang et al., 2024)
https://arxiv.org/abs/2412.06464

Architecture
------------
The Gated Delta Network extends DeltaNet with:
 1. **Per-head matrix state** S ∈ ℝ^{Dk × Dv} maintained by the delta rule.
 2. **Scalar forget gate α_t ∈ (0,1]** — element-wise decay of each row of S.
 3. **Write strength β_t ∈ [0,1]** — controls how much the new (k,v) pair is written.
 4. **Sigmoid output gate g_t** — data-dependent gating of retrieved memories.
 5. **Head-wise RMSNorm** before the output gate (stabilises large S values).
 6. **Optional short (causal) depthwise conv** on Q, K, V for local context.

State update (per-head):
    err_t  = v_t − S_{t-1} @ k_t                      (prediction error)
    S_t    = α_t · S_{t-1} + β_t · outer(err_t, k_t)  (delta rule + decay)

Output (per-head):
    o_t = g_t ⊙ RMSNorm(S_t @ q_t)

Three forward modes
-------------------
 * **Sequential** (default, exact, training) — step-by-step O(T·Dk·Dv) per head.
 * **Chunk-parallel** (optional, approximate intra-chunk) — reduce Python loop
   overhead; intra-chunk uses pre-S state (standard chunk-mode approximation).
   Exact sequential state update maintains correct S across chunks.
 * **Incremental** (layer_state dict) — single-step decoding for KV-cached
   generation. Reads/writes "gdn_state" from/to the provided dict.

Interface matches LinearAttention / KimiAttention:
    forward(query, key, value, attn_mask, key_padding_mask, is_causal, layer_state)
    → Tuple[Tensor, None, None]

For standalone use (internal KimiAttention-style) pass `x` directly:
    _forward_standalone(x, state) → (y, next_state)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _matvec_xt_kernel(
        S_ptr,
        x_ptr,
        y_ptr,
        Dk: tl.constexpr,
        Dv: tl.constexpr,
        strideSB,
        strideSi,
        strideSj,
        stridexB,
        stridexi,
        strideyB,
        strideyj,
        BLOCK_I: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        """y[b, j] = sum_i S[b, i, j] * x[b, i]."""
        pid_b = tl.program_id(0)
        pid_j = tl.program_id(1)
        offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
        mask_j = offs_j < Dv
        acc = tl.zeros([BLOCK_J], dtype=tl.float32)

        for i0 in range(0, Dk, BLOCK_I):
            offs_i = i0 + tl.arange(0, BLOCK_I)
            mask_i = offs_i < Dk
            x_vec = tl.load(
                x_ptr + pid_b * stridexB + offs_i * stridexi,
                mask=mask_i,
                other=0.0,
            ).to(tl.float32)
            S_tile = tl.load(
                S_ptr
                + pid_b * strideSB
                + offs_i[:, None] * strideSi
                + offs_j[None, :] * strideSj,
                mask=mask_i[:, None] & mask_j[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(S_tile * x_vec[:, None], axis=0)

        tl.store(y_ptr + pid_b * strideyB + offs_j * strideyj, acc, mask=mask_j)

    @triton.jit
    def _rank1_update_kernel(
        S_ptr,
        k_ptr,
        d_ptr,
        b_ptr,
        Dk: tl.constexpr,
        Dv: tl.constexpr,
        strideSB,
        strideSi,
        strideSj,
        stridekB,
        strideki,
        stridedB,
        stridedj,
        stridebB,
        BLOCK_I: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        """S[b] += beta[b] * k[b] ⊗ d[b]."""
        pid_b = tl.program_id(0)
        pid_j = tl.program_id(1)
        offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
        mask_j = offs_j < Dv

        b_scalar = tl.load(b_ptr + pid_b * stridebB).to(tl.float32)
        d_vec = tl.load(
            d_ptr + pid_b * stridedB + offs_j * stridedj,
            mask=mask_j,
            other=0.0,
        ).to(tl.float32)
        d_scaled = b_scalar * d_vec

        for i0 in range(0, Dk, BLOCK_I):
            offs_i = i0 + tl.arange(0, BLOCK_I)
            mask_i = offs_i < Dk
            k_vec = tl.load(
                k_ptr + pid_b * stridekB + offs_i * strideki,
                mask=mask_i,
                other=0.0,
            ).to(tl.float32)
            S_tile = tl.load(
                S_ptr
                + pid_b * strideSB
                + offs_i[:, None] * strideSi
                + offs_j[None, :] * strideSj,
                mask=mask_i[:, None] & mask_j[None, :],
                other=0.0,
            ).to(tl.float32)
            tl.store(
                S_ptr
                + pid_b * strideSB
                + offs_i[:, None] * strideSi
                + offs_j[None, :] * strideSj,
                S_tile + k_vec[:, None] * d_scaled[None, :],
                mask=mask_i[:, None] & mask_j[None, :],
            )

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


class _CausalConv(nn.Module):
    """Causal depthwise Conv1d + SiLU activation."""

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=self.pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        T = x.size(1)
        x = self.conv(x.transpose(1, 2))[:, :, :T]  # causal crop
        return F.silu(x.transpose(1, 2).contiguous())


class _HeadRMSNorm(nn.Module):
    """Per-head RMSNorm — applied to [B, H, T, Dv] or [BH, T, Dv]."""

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, *, Dv]  or  [BH, *, Dv]  — norm over last dim
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = x * rms
        # broadcast weight: handle both (B,H,T,Dv) and (BH,T,Dv)
        w = self.weight.view(*self.weight.shape)  # (H, Dv)
        if x.dim() == 4:
            # (B, H, T, Dv) — unsqueeze for broadcast
            out = out * w.unsqueeze(0).unsqueeze(2)
        else:
            # (BH, T, Dv) — we don't know H here, return as-is (caller reshapes)
            out = out * w.reshape(-1, w.size(-1)).repeat(
                x.size(0) // self.weight.size(0), 1
            ).unsqueeze(1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────


class GatedDeltaNet(nn.Module):
    """
    Gated Delta Network attention.

    Drop-in replacement for ``LinearAttention`` / ``KimiAttention``.
    Accepts the same ``forward(query, key, value, ...)`` signature used by
    ``TransformerEncoderLayer`` / ``TransformerDecoderLayer``.

    Parameters
    ----------
    d_model : int
        Total model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Output dropout probability.
    d_key : int | None
        Per-head key/query dimension. Defaults to ``d_model // n_heads``.
    d_val : int | None
        Per-head value dimension. Defaults to ``d_model // n_heads``.
    chunk_size : int
        Chunk length for chunk-parallel mode. 0 = pure sequential.
    use_short_conv : bool
        Apply a causal depthwise conv on Q, K, V before the attention recurrence.
    conv_kernel : int
        Kernel size for the short conv (default 4).
    alpha_min : float
        Minimum value for the forget gate α (numerical stability).
    beta_max : float
        Maximum value for the write gate β.
    eps : float
        Epsilon for head-wise RMSNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        # Ignored kwargs — keep API-compatible with MultiAttention / LinearAttention
        attention_type: str = "standard",
        freq_modes: int = 16,
        cross_attention: bool = False,
        # GDN-specific
        d_key: Optional[int] = None,
        d_val: Optional[int] = None,
        chunk_size: int = 64,
        use_short_conv: bool = True,
        conv_kernel: int = 4,
        alpha_min: float = 0.1,
        beta_max: float = 1.0,
        eps: float = 1e-6,
        use_triton: bool = False,
        triton_block_i: int = 64,
        triton_block_j: int = 128,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.h = n_heads
        self.dk = d_key or (d_model // n_heads)
        self.dv = d_val or (d_model // n_heads)
        self.chunk_size = int(chunk_size) if chunk_size and chunk_size > 1 else 0
        self.alpha_min = float(alpha_min)
        self.beta_max = float(beta_max)
        self.use_triton = bool(use_triton and _HAS_TRITON)
        self.triton_block_i = int(triton_block_i)
        self.triton_block_j = int(triton_block_j)

        # ── Projections ────────────────────────────────────────────────────
        self.q_proj = nn.Linear(d_model, self.h * self.dk, bias=False)
        self.k_proj = nn.Linear(d_model, self.h * self.dk, bias=False)
        self.v_proj = nn.Linear(d_model, self.h * self.dv, bias=False)
        self.o_proj = nn.Linear(self.h * self.dv, d_model, bias=False)

        # ── Gates ──────────────────────────────────────────────────────────
        # α: per-head scalar forget gate (one value per head per step)
        self.alpha_proj = nn.Linear(d_model, self.h, bias=True)
        # β: per-head scalar write strength
        self.beta_proj = nn.Linear(d_model, self.h, bias=True)
        # g: output gate, matched to value dim
        self.g_down = nn.Linear(d_model, self.h * self.dv // 2, bias=False)
        self.g_up = nn.Linear(self.h * self.dv // 2, self.h * self.dv, bias=False)

        # ── Normalisation & dropout ────────────────────────────────────────
        self.h_rms = _HeadRMSNorm(self.h, self.dv, eps=eps)
        self.drop = nn.Dropout(dropout)

        # ── Optional short conv on Q, K, V ────────────────────────────────
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.q_conv = _CausalConv(self.h * self.dk, conv_kernel)
            self.k_conv = _CausalConv(self.h * self.dk, conv_kernel)
            self.v_conv = _CausalConv(self.h * self.dv, conv_kernel)

        # ── Initialisation ─────────────────────────────────────────────────
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)
        # β bias: start near 0.5 (moderate write rate)
        nn.init.constant_(self.beta_proj.bias, 0.0)
        # α bias: start near 0.9 (slow forgetting)
        nn.init.constant_(self.alpha_proj.bias, 2.0)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_state(
        self, BH: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(BH, self.dk, self.dv, device=device, dtype=dtype)

    @staticmethod
    def _l2_norm(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Row-wise L2-normalise last dim."""
        return t / (t.norm(dim=-1, keepdim=True).clamp_min(eps))

    def _project(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Linear project and apply optional short conv. Returns (Q,K,V) raw."""
        q_raw = self.q_proj(x)  # [B, T, H*Dk]
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)
        if self.use_short_conv:
            q_raw = self.q_conv(q_raw)
            k_raw = self.k_conv(k_raw)
            v_raw = self.v_conv(v_raw)
        return q_raw, k_raw, v_raw

    def _gate_params(
        self, x: torch.Tensor, BH: int, T: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forget (α) and write (β) gates.

        Returns
        -------
        alpha : [BH, T, 1]  scalar decay per head per step
        beta  : [BH, T, 1]  scalar write strength per head per step
        """
        alpha = (
            torch.sigmoid(self.alpha_proj(x))  # [B, T, H]
            .clamp(min=self.alpha_min, max=1.0)
            .transpose(1, 2)  # [B, H, T]
            .contiguous()
            .reshape(BH, T, 1)
        )
        beta = (
            torch.sigmoid(self.beta_proj(x))  # [B, T, H]
            .clamp(max=self.beta_max)
            .transpose(1, 2)
            .contiguous()
            .reshape(BH, T, 1)
        )
        return alpha, beta

    def _output_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Data-dependent sigmoid output gate [B, H, T, Dv]."""
        B, T = x.size(0), x.size(1)
        return (
            torch.sigmoid(self.g_up(F.silu(self.g_down(x))))  # [B, T, H*Dv]
            .view(B, T, self.h, self.dv)
            .permute(0, 2, 1, 3)  # [B, H, T, Dv]
            .contiguous()
        )

    def _can_use_triton(self, S: torch.Tensor) -> bool:
        return (
            self.use_triton
            and _HAS_TRITON
            and (not self.training)
            and S.is_cuda
            and S.is_contiguous()
            and S.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )

    def _triton_matvec_xt(self, S: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute y = S^T @ x for batched [BH, Dk, Dv] and [BH, Dk]."""
        BH, Dk, Dv = S.shape
        y = torch.empty(BH, Dv, device=S.device, dtype=torch.float32)
        grid = (BH, triton.cdiv(Dv, self.triton_block_j))
        _matvec_xt_kernel[grid](
            S,
            x,
            y,
            Dk,
            Dv,
            S.stride(0),
            S.stride(1),
            S.stride(2),
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_I=self.triton_block_i,
            BLOCK_J=self.triton_block_j,
        )
        return y

    def _triton_rank1_update(
        self,
        S: torch.Tensor,
        k_t: torch.Tensor,
        delta: torch.Tensor,
        beta_t: torch.Tensor,
    ) -> None:
        """In-place S += beta * k ⊗ delta (all batched over BH)."""
        BH, Dk, Dv = S.shape
        grid = (BH, triton.cdiv(Dv, self.triton_block_j))
        _rank1_update_kernel[grid](
            S,
            k_t,
            delta,
            beta_t,
            Dk,
            Dv,
            S.stride(0),
            S.stride(1),
            S.stride(2),
            k_t.stride(0),
            k_t.stride(1),
            delta.stride(0),
            delta.stride(1),
            beta_t.stride(0),
            BLOCK_I=self.triton_block_i,
            BLOCK_J=self.triton_block_j,
        )

    def _delta_step_triton(
        self,
        S: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        q_t: torch.Tensor,
        alpha_t: torch.Tensor,
        beta_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-only Triton fast step; numerically aligned with _delta_step."""
        k32 = k_t.to(torch.float32).contiguous()
        v32 = v_t.to(torch.float32).contiguous()
        q32 = q_t.to(torch.float32).contiguous()
        a32 = alpha_t.to(torch.float32).contiguous()
        b32 = beta_t.to(torch.float32).contiguous()

        S_work = S.contiguous().to(torch.float32)

        v_hat = self._triton_matvec_xt(S_work, k32)
        delta = v32 - v_hat

        S_work.mul_(a32.unsqueeze(-1))
        self._triton_rank1_update(S_work, k32, delta, b32.squeeze(-1))

        o_t = self._triton_matvec_xt(S_work, q32).to(S.dtype)
        S_new = S_work.to(S.dtype)
        return o_t, S_new

    # ── Single recurrent step ─────────────────────────────────────────────────

    @staticmethod
    def _delta_step(
        S: torch.Tensor,  # [BH, Dk, Dv]  current state
        k_t: torch.Tensor,  # [BH, Dk]       normalised key
        v_t: torch.Tensor,  # [BH, Dv]       value
        q_t: torch.Tensor,  # [BH, Dk]       normalised query
        alpha_t: torch.Tensor,  # [BH, 1]        or [BH, Dk] scalar/vector decay
        beta_t: torch.Tensor,  # [BH, 1]        write strength
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Delta-rule update + retrieval.

        Returns (o_t [BH, Dv], S_new [BH, Dk, Dv]).
        """
        # Prediction error: how wrong is the current state at predicting v_t?
        # v_retrieved = S @ k  →  [BH, Dv]
        v_hat = torch.einsum("bid,bi->bd", S, k_t)  # [BH, Dv]
        err = v_t - v_hat  # [BH, Dv]

        # Outer product write: [BH, Dk, Dv]
        write = torch.einsum("bi,bd->bid", k_t, err)  # [BH, Dk, Dv]

        # State update: decay rows + add error-corrected outer product
        # alpha_t is [BH, 1] — broadcast over (Dk, Dv)
        S_new = alpha_t.unsqueeze(-1) * S + beta_t.unsqueeze(-1) * write

        # Retrieve from updated state
        o_t = torch.einsum("bid,bi->bd", S_new, q_t)  # [BH, Dv]

        return o_t, S_new

    # ── Chunk-parallel output (inter-chunk exact, intra-chunk approx) ─────────

    def _chunk_output(
        self,
        Q: torch.Tensor,  # [BH, C, Dk]
        K: torch.Tensor,  # [BH, C, Dk]
        V: torch.Tensor,  # [BH, C, Dv]
        alpha: torch.Tensor,  # [BH, C, 1]
        beta: torch.Tensor,  # [BH, C, 1]
        S_prev: torch.Tensor,  # [BH, Dk, Dv]
    ) -> torch.Tensor:  # [BH, C, Dv]
        """
        Parallel output computation for one chunk using S_prev.

        Inter-chunk: O_inter = cumulative alpha decay × (Q @ S_prev) contribution.
        Intra-chunk: causal attention weighted by cumulative alpha and beta.

        This is an approximation: intra-chunk uses S_prev (not per-step S).
        Set chunk_size=1 / chunk_size=0. for exact sequential behaviour.
        """
        C = Q.size(1)
        device = Q.device

        # ── Inter-chunk contribution ──────────────────────────────────────
        # cum_alpha[t] = product of alpha[0..t]  (for broadcasting over Dk axis)
        cum_alpha = torch.cumprod(alpha, dim=1)  # [BH, C, 1]
        # o_inter[t] = alpha_cum[t] * (S_prev^T @ q_t),  S: [BH,Dk,Dv], q: [BH,C,Dk]
        # einsum contracts over Dk ("k"): [BH,Dk,Dv] × [BH,C,Dk] → [BH,C,Dv]
        o_inter = torch.einsum("bkv,btk->btv", S_prev, Q) * cum_alpha

        # ── Intra-chunk attention ─────────────────────────────────────────
        # For i >= j:  A[i,j] = (Q[i] · K[j]) × β[j] × cum_alpha[i] / cum_alpha[j]
        log_cum = torch.log(cum_alpha.clamp_min(1e-7))  # [BH, C, 1]
        # relative log-decay from j to i: log_cum[i] - log_cum[j]
        decay_ij = torch.exp(
            log_cum.squeeze(-1).unsqueeze(2) - log_cum.squeeze(-1).unsqueeze(1)
        )  # [BH, C, C]

        # Causal lower-triangular mask
        causal_mask = torch.tril(torch.ones(C, C, device=device, dtype=torch.bool))
        decay_ij = decay_ij.masked_fill(~causal_mask.unsqueeze(0), 0.0)

        # Attention logits: [BH, C, C]
        attn = torch.bmm(Q, K.transpose(1, 2))  # [BH, C, C]
        # Weight by beta and inter-step decay
        attn = attn * (beta.squeeze(-1).unsqueeze(1)) * decay_ij

        o_intra = torch.bmm(attn, V)  # [BH, C, Dv]

        return o_inter + o_intra

    # ── Core recurrent forward (standalone, x-in)  ────────────────────────────

    def _forward_recurrent(
        self,
        x: torch.Tensor,  # [B, T, D]
        state: Optional[torch.Tensor] = None,  # [B, H, Dk, Dv]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full recurrence over T steps.
        Returns (y [B, T, D], next_state [B, H, Dk, Dv]).
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype
        BH = B * self.h

        # Project + (optional) short conv
        q_raw, k_raw, v_raw = self._project(x)

        # Reshape to [BH, T, D*]
        Q = self._l2_norm(
            q_raw.view(B, T, self.h, self.dk)
            .permute(0, 2, 1, 3)
            .reshape(BH, T, self.dk)
        )
        K = self._l2_norm(
            k_raw.view(B, T, self.h, self.dk)
            .permute(0, 2, 1, 3)
            .reshape(BH, T, self.dk)
        )
        V = (
            v_raw.view(B, T, self.h, self.dv)
            .permute(0, 2, 1, 3)
            .reshape(BH, T, self.dv)
        )

        alpha, beta = self._gate_params(x, BH, T)  # [BH, T, 1]

        # Initialise / reshape state
        S = (
            state.reshape(BH, self.dk, self.dv).contiguous().to(dtype=dtype)
            if state is not None
            else self._init_state(BH, device, dtype)
        )

        out = torch.zeros(BH, T, self.dv, device=device, dtype=dtype)
        C = self.chunk_size
        use_triton = self._can_use_triton(S)

        if C and T > C:
            # ── Chunk-parallel mode ───────────────────────────────────────
            for s in range(0, T, C):
                e = min(s + C, T)
                Q_c = Q[:, s:e]
                K_c = K[:, s:e]
                V_c = V[:, s:e]
                a_c = alpha[:, s:e]
                b_c = beta[:, s:e]

                out[:, s:e] = self._chunk_output(Q_c, K_c, V_c, a_c, b_c, S)

                # Exact sequential state update (detach: S is a recurrent buffer)
                S = S.detach()
                for t in range(e - s):
                    if use_triton:
                        _, S = self._delta_step_triton(
                            S,
                            K_c[:, t],
                            V_c[:, t],
                            Q_c[:, t],
                            a_c[:, t],
                            b_c[:, t],
                        )
                    else:
                        _, S = self._delta_step(
                            S,
                            K_c[:, t],
                            V_c[:, t],
                            Q_c[:, t],
                            a_c[:, t],
                            b_c[:, t],
                        )
        else:
            # ── Exact sequential mode ─────────────────────────────────────
            S = S.detach()
            for t in range(T):
                if use_triton:
                    out[:, t], S = self._delta_step_triton(
                        S, K[:, t], V[:, t], Q[:, t], alpha[:, t], beta[:, t]
                    )
                else:
                    out[:, t], S = self._delta_step(
                        S, K[:, t], V[:, t], Q[:, t], alpha[:, t], beta[:, t]
                    )

        # ── Output gate + head-wise RMSNorm ───────────────────────────────
        out_h = out.reshape(B, self.h, T, self.dv)  # [B, H, T, Dv]
        out_h = self.h_rms(out_h)
        g = self._output_gate(x)  # [B, H, T, Dv]
        out_h = out_h * g

        y = out_h.permute(0, 2, 1, 3).contiguous().reshape(B, T, self.h * self.dv)
        y = self.drop(self.o_proj(y))

        next_state = S.reshape(B, self.h, self.dk, self.dv)
        return y, next_state

    # ── Single-step incremental forward ───────────────────────────────────────

    def _step(
        self,
        x_t: torch.Tensor,  # [B, 1, D]
        state: Optional[torch.Tensor],  # [B, H, Dk, Dv]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step forward for incremental (KV-cached) decoding."""
        B, _, _ = x_t.shape
        device, dtype = x_t.device, x_t.dtype
        BH = B * self.h

        q_raw = self.q_proj(x_t)  # [B, 1, H*Dk] — no short conv during step
        k_raw = self.k_proj(x_t)
        v_raw = self.v_proj(x_t)

        Q = self._l2_norm(
            q_raw.view(B, 1, self.h, self.dk)
            .permute(0, 2, 1, 3)
            .reshape(BH, 1, self.dk)
        )
        K = self._l2_norm(
            k_raw.view(B, 1, self.h, self.dk)
            .permute(0, 2, 1, 3)
            .reshape(BH, 1, self.dk)
        )
        V = (
            v_raw.view(B, 1, self.h, self.dv)
            .permute(0, 2, 1, 3)
            .reshape(BH, 1, self.dv)
        )

        alpha, beta = self._gate_params(x_t, BH, 1)  # [BH, 1, 1]

        S = (
            state.reshape(BH, self.dk, self.dv).contiguous().to(dtype=dtype)
            if state is not None
            else self._init_state(BH, device, dtype)
        )

        if self._can_use_triton(S):
            o_t, S_new = self._delta_step_triton(
                S, K[:, 0], V[:, 0], Q[:, 0], alpha[:, 0], beta[:, 0]
            )
        else:
            o_t, S_new = self._delta_step(
                S, K[:, 0], V[:, 0], Q[:, 0], alpha[:, 0], beta[:, 0]
            )  # o_t: [BH, Dv]

        # gate & norm
        out_h = o_t.reshape(B, self.h, 1, self.dv)  # [B, H, 1, Dv]
        out_h = self.h_rms(out_h)
        g = self._output_gate(x_t)  # [B, H, 1, Dv]
        out_h = out_h * g

        y = out_h.permute(0, 2, 1, 3).contiguous().reshape(B, 1, self.h * self.dv)
        y = self.drop(self.o_proj(y))

        next_state = S_new.reshape(B, self.h, self.dk, self.dv)
        return y, next_state

    # ── Public forward — matches MultiAttention / LinearAttention API ──────────

    def forward(
        self,
        query: torch.Tensor,  # [B, Tq, D]
        key: torch.Tensor,  # [B, Tk, D]  (self-attn: same as query)
        value: torch.Tensor,  # [B, Tk, D]
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        layer_state: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Drop-in forward.

        Notes
        -----
        * ``attn_mask`` / ``key_padding_mask`` are handled by masking V:
          padded positions are zeroed before the recurrence so they contribute
          nothing to the state update.
        * ``is_causal`` is implicitly satisfied by the causal recurrence.
        * For cross-attention (``key != query``) the module uses ``query``
          as the input token stream and projects K, V from ``key`` / ``value``
          (standard cross-attention approximation for recurrent models).
        * ``layer_state`` (dict): reads/writes ``"gdn_state"`` key for
          single-step incremental decoding.
        """
        B, Tq, D = query.shape
        is_cross = key is not query and key.data_ptr() != query.data_ptr()

        # ── Incremental single-step (decoder KV-cache path) ───────────────
        if layer_state is not None and Tq == 1 and not self.training:
            prev_s = layer_state.get("gdn_state", None)
            y, next_s = self._step(query, prev_s)
            layer_state["gdn_state"] = next_s
            return y, None, None

        # ── Cross-attention: use query stream but project from KV tensors ──
        if is_cross:
            # In the recurrent view we treat the entire encoder memory as a
            # single "batch" retrieved via the query stream.
            # Simplest correct approach: run recurrence on query, but use
            # projected K/V from the encoder memory as key/value vectors at
            # each step — however for T_q != T_k this is ill-defined.
            # We fall back to: encode KV into a fixed state via a fast
            # summary pass, then decode with query.
            y = self._cross_attention_approx(query, key, value, key_padding_mask)
            return y, None, None

        # ── Apply padding mask to query input (zero out padded tokens) ────
        x = query  # self-attention: query == key == value
        if key_padding_mask is not None:
            # key_padding_mask: [B, T], True = ignore
            pad = key_padding_mask.unsqueeze(-1).to(dtype=x.dtype)  # [B, T, 1]
            x = x * (1.0 - pad)

        # ── Incremental multi-step (carry state across calls) ─────────────
        if layer_state is not None:
            prev_s = layer_state.get("gdn_state", None)
            y, next_s = self._forward_recurrent(x, prev_s)
            layer_state["gdn_state"] = next_s
            return y, None, None

        # ── Standard full-sequence forward ────────────────────────────────
        y, _ = self._forward_recurrent(x, state=None)
        return y, None, None

    # ── Cross-attention approximation ────────────────────────────────────────

    def _cross_attention_approx(
        self,
        query: torch.Tensor,  # [B, Tq, D]
        key: torch.Tensor,  # [B, Tm, D]   (memory)
        value: torch.Tensor,  # [B, Tm, D]
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Approximate cross-attention using the delta rule:
         1. Build a state S from (K, V) memory pairs using the delta rule.
         2. Retrieve from S using query vectors.

        This is O(Tm·Dk·Dv + Tq·Dk·Dv) — linear in both memory and query length.
        """
        B, Tq, D = query.shape
        _, Tm, _ = key.shape
        device, dtype = query.device, query.dtype
        BH = B * self.h

        # Project and norm K, V from memory
        k_raw = (
            self.k_proj(key)
            .view(B, Tm, self.h, self.dk)
            .permute(0, 2, 1, 3)
            .reshape(BH, Tm, self.dk)
        )
        v_raw = (
            self.v_proj(value)
            .view(B, Tm, self.h, self.dv)
            .permute(0, 2, 1, 3)
            .reshape(BH, Tm, self.dv)
        )
        K_m = self._l2_norm(k_raw)
        V_m = v_raw

        if key_padding_mask is not None:
            # Zero out padded key/value pairs
            pad = (
                key_padding_mask.unsqueeze(1).unsqueeze(-1).to(dtype=dtype)
            )  # [B,1,Tm,1]
            pad_bh = pad.expand(B, self.h, Tm, 1).reshape(BH, Tm, 1)
            K_m = K_m * (1.0 - pad_bh)
            V_m = V_m * (1.0 - pad_bh)

        # Beta: uniform write rate for memory ingestion
        beta_m = torch.ones(BH, Tm, 1, device=device, dtype=dtype) * 0.5

        # Build memory state (no decay: α=1 for full memory retention)
        S = self._init_state(BH, device, dtype)
        use_triton = self._can_use_triton(S)
        for t in range(Tm):
            alpha_t = torch.ones(BH, 1, device=device, dtype=dtype)
            if use_triton:
                _, S = self._delta_step_triton(
                    S, K_m[:, t], V_m[:, t], K_m[:, t], alpha_t, beta_m[:, t]
                )
            else:
                _, S = self._delta_step(
                    S, K_m[:, t], V_m[:, t], K_m[:, t], alpha_t, beta_m[:, t]
                )

        # Project Q from query stream and retrieve
        q_raw = (
            self.q_proj(query)
            .view(B, Tq, self.h, self.dk)
            .permute(0, 2, 1, 3)
            .reshape(BH, Tq, self.dk)
        )
        Q = self._l2_norm(q_raw)

        out = torch.zeros(BH, Tq, self.dv, device=device, dtype=dtype)
        for t in range(Tq):
            out[:, t] = torch.einsum("bid,bi->bd", S, Q[:, t])

        # Output gate + norm
        out_h = out.reshape(B, self.h, Tq, self.dv)
        out_h = self.h_rms(out_h)
        g = self._output_gate(query)  # [B, H, Tq, Dv]
        out_h = out_h * g

        y = out_h.permute(0, 2, 1, 3).contiguous().reshape(B, Tq, self.h * self.dv)
        return self.drop(self.o_proj(y))

    # ── Standalone x-in interface (matches KimiAttention style) ──────────────

    def forward_standalone(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standalone forward (KimiAttention-compatible interface).

        Parameters
        ----------
        x     : [B, T, D]
        state : [B, H, Dk, Dv]  or None

        Returns
        -------
        y          : [B, T, D]
        next_state : [B, H, Dk, Dv]
        """
        return self._forward_recurrent(x, state)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, heads={self.h}, "
            f"dk={self.dk}, dv={self.dv}, chunk_size={self.chunk_size}, "
            f"short_conv={self.use_short_conv}, "
            f"alpha_min={self.alpha_min}, beta_max={self.beta_max}"
        )
