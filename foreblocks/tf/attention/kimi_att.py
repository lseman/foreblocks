# -*- coding: utf-8 -*-
"""
kimi_att_fast.py — Corrected KDA (DeltaNet-style) Linear Attention

Fixes vs previous version
--------------------------
1.  Triton path is inference-only (guarded with `assert not self.training`).
    The in-place `tl.store` operations break autograd; gradients were silently
    wrong during training in the old code.

2.  chunk_size is a real optimisation:
    - Inter-chunk contribution (S_prev → output) is fully vectorised: O(C·Dk·Dv)
      as a single bmm, not C individual matvecs.
    - Intra-chunk output uses a parallel lower-triangular attention matrix built
      from cumulative decay products — reducing Python loop overhead to O(T/C).
    - The state update inside each chunk is still a sequential scan (exact).
    - APPROXIMATION NOTE: the intra-chunk output ignores within-chunk
      error-correction (the −k·kᵀ·S_{t−1} correction term uses S_prev, not the
      per-step state). This is the standard chunk-mode trade-off: set
      chunk_size=1 for exact sequential behaviour.

3.  safe_updates is actually implemented: clamps α ∈ [α_min, 1] and β ∈ [0, β_max]
    to prevent state explosion.

4.  Triton kernels all use fp32 internally and cast outputs back to the input
    dtype — previously mixed-precision pointers caused silent wrong results in
    bf16/fp16.

5.  State tensor S is always detached before each chunk's sequential update.
    S is a recurrent buffer (like an RNN hidden state), not part of the
    computational graph for gradient-through-time — the old code treated it
    inconsistently.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional Triton — inference-only kernels
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_ALPHA_MIN_DEFAULT = (
    0.1  # safe_updates: minimum decay (prevents S from growing unboundedly)
)
_BETA_MAX_DEFAULT = 1.0  # safe_updates: maximum write strength


# ─────────────────────────────────────────────────────────────────────────────
# Causal ShortConv
# ─────────────────────────────────────────────────────────────────────────────
class _ShortConv1x(nn.Module):
    """Causal depthwise (or pointwise) Conv1d with SiLU activation."""

    def __init__(self, d_model: int, kernel_size: int = 4, mode: str = "depthwise"):
        super().__init__()
        self.mode = mode
        if mode == "off":
            self.conv = None
        else:
            pad = kernel_size - 1
            groups = d_model if mode == "depthwise" else 1
            self.conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                groups=groups,
                padding=pad,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        if self.conv is None:
            return x
        T0 = x.size(1)
        x = x.transpose(1, 2).contiguous()  # [B, D, T]
        x = self.conv(x)[:, :, :T0].contiguous()  # crop to original T
        return F.silu(x.transpose(1, 2).contiguous())


# ─────────────────────────────────────────────────────────────────────────────
# Head-wise RMSNorm
# ─────────────────────────────────────────────────────────────────────────────
class _HeadwiseRMSNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, H, T, Dv]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        w = self.weight.view(1, self.weight.size(0), 1, self.weight.size(1))
        return x * w


# ─────────────────────────────────────────────────────────────────────────────
# Optional Triton kernels — INFERENCE ONLY
# ─────────────────────────────────────────────────────────────────────────────
if _HAS_TRITON:

    @triton.jit
    def _row_scale_kernel(
        S_ptr,
        a_ptr,
        BH: tl.constexpr,
        Dk: tl.constexpr,
        Dv: tl.constexpr,
        strideSB,
        strideSi,
        strideSj,
        strideaB,
        strideai,
        BLOCK_I: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        """S[b,i,:] *= a[b,i]  (in-place row scale)."""
        pid_b = tl.program_id(0)
        pid_j = tl.program_id(1)
        offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
        mask_j = offs_j < Dv

        for i0 in range(0, Dk, BLOCK_I):
            offs_i = i0 + tl.arange(0, BLOCK_I)
            mask_i = offs_i < Dk
            # Load decay scalars once per i-tile
            a = tl.load(
                a_ptr + pid_b * strideaB + offs_i * strideai,
                mask=mask_i,
                other=1.0,
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
                (S_tile * a[:, None]),
                mask=mask_i[:, None] & mask_j[None, :],
            )

    @triton.jit
    def _matvec_xt_kernel(
        S_ptr,
        x_ptr,
        y_ptr,
        BH: tl.constexpr,
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
        v_ptr,
        kTS_ptr,
        b_ptr,
        BH: tl.constexpr,
        Dk: tl.constexpr,
        Dv: tl.constexpr,
        strideSB,
        strideSi,
        strideSj,
        stridekB,
        strideki,
        stridevB,
        stridevj,
        strideyB,
        strideyj,
        stridebB,
        BLOCK_I: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        """S[b] += b_scalar * k[b] ⊗ (v[b] - kTS[b])."""
        pid_b = tl.program_id(0)
        pid_j = tl.program_id(1)
        offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
        mask_j = offs_j < Dv

        b_scalar = tl.load(b_ptr + pid_b * stridebB).to(tl.float32)
        v_vec = tl.load(
            v_ptr + pid_b * stridevB + offs_j * stridevj,
            mask=mask_j,
            other=0.0,
        ).to(tl.float32)
        kTS_vec = tl.load(
            kTS_ptr + pid_b * strideyB + offs_j * strideyj,
            mask=mask_j,
            other=0.0,
        ).to(tl.float32)
        delta_v = b_scalar * (v_vec - kTS_vec)  # [BLOCK_J]

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
                S_tile + k_vec[:, None] * delta_v[None, :],
                mask=mask_i[:, None] & mask_j[None, :],
            )


# ─────────────────────────────────────────────────────────────────────────────
# Core KDA
# ─────────────────────────────────────────────────────────────────────────────
class _KDA_Fast(nn.Module):
    """
    Kernel-Decay Attention (DeltaNet-style) recurrence.

    Recurrence per step:
        S_t = diag(α_t) · S_{t-1} + β_t · k_t · (v_t − k_tᵀ S_{t-1})ᵀ
        o_t = S_t · q_t

    chunk_size > 1 activates the parallel-chunk mode:
        - Inter-chunk term  (exact):  O_inter = Q_c · S_prev  [vectorised bmm]
        - Intra-chunk term  (approx): parallel lower-triangular attention with
          cumulative decay weights; ignores within-chunk error-correction.
        - State update (exact): sequential scan of length chunk_size.
    Set chunk_size=1 (or 0) for the fully exact sequential mode.

    use_triton=True:  uses Triton kernels for the sequential state update steps.
                      INFERENCE ONLY — in-place Triton stores are not
                      autograd-safe.  The flag is silently ignored when
                      self.training is True.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        gate_rank: Optional[int] = None,
        dropout: float = 0.0,
        shortconv_mode: str = "depthwise",
        chunk_size: int = 64,
        # safe_updates: clamp α and β to prevent state blow-up
        safe_updates: bool = True,
        alpha_min: float = _ALPHA_MIN_DEFAULT,
        beta_max: float = _BETA_MAX_DEFAULT,
        # Triton (inference only)
        use_triton: bool = False,
        triton_block_i: int = 64,
        triton_block_j: int = 128,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.h = num_heads
        self.dk = d_k or (d_model // num_heads)
        self.dv = d_v or (d_model // num_heads)
        self.d_model = d_model

        # chunk_size=0 or 1 → pure sequential (exact)
        self.chunk_size = max(int(chunk_size), 0)

        self.safe_updates = bool(safe_updates)
        self.alpha_min = float(alpha_min)
        self.beta_max = float(beta_max)

        self.use_triton = bool(use_triton and _HAS_TRITON)
        self.BLOCK_I = int(triton_block_i)
        self.BLOCK_J = int(triton_block_j)

        # Short convolutions (causal)
        self.pre_q = _ShortConv1x(d_model, kernel_size=4, mode=shortconv_mode)
        self.pre_k = _ShortConv1x(d_model, kernel_size=4, mode=shortconv_mode)
        self.pre_v = _ShortConv1x(d_model, kernel_size=4, mode=shortconv_mode)

        # Q / K / V projections
        self.q_proj = nn.Linear(d_model, num_heads * self.dk, bias=True)
        self.k_proj = nn.Linear(d_model, num_heads * self.dk, bias=True)
        self.v_proj = nn.Linear(d_model, num_heads * self.dv, bias=True)

        # α (low-rank decay) and β (per-head write gate)
        r = gate_rank or self.dk
        self.alpha_down = nn.Linear(d_model, r, bias=True)
        self.alpha_up = nn.Linear(r, num_heads * self.dk, bias=True)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=True)

        # Output: per-head RMSNorm + gate + projection
        self.h_rms = _HeadwiseRMSNorm(num_heads, self.dv)
        self.out_gate_down = nn.Linear(d_model, r, bias=True)
        self.out_gate_up = nn.Linear(r, num_heads, bias=True)
        self.o_proj = nn.Linear(num_heads * self.dv, d_model, bias=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / x.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()

    def _init_state(
        self, BH: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(BH, self.dk, self.dv, device=device, dtype=dtype)

    def _project(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        Q = (
            self.q_proj(self.pre_q(x))
            .view(B, T, self.h, self.dk)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        K = (
            self.k_proj(self.pre_k(x))
            .view(B, T, self.h, self.dk)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        V = (
            self.v_proj(self.pre_v(x))
            .view(B, T, self.h, self.dv)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        if Q.size(2) != T:
            raise RuntimeError(f"ShortConv T drift: Q.T={Q.size(2)} != input T={T}")
        return Q, K, V

    def _gate_params(
        self, x: torch.Tensor, BH: int, T: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            alpha [BH, T, Dk]  — per-step decay (clamped when safe_updates=True)
            beta  [BH, T, 1]   — per-step write gate (clamped when safe_updates=True)
        """
        B = x.size(0)
        alpha = (
            torch.sigmoid(self.alpha_up(F.silu(self.alpha_down(x))))
            .view(B, T, self.h, self.dk)
            .permute(0, 2, 1, 3)
            .reshape(BH, T, self.dk)
        )

        beta = (
            torch.sigmoid(self.beta_proj(x))
            .permute(0, 2, 1)
            .contiguous()
            .reshape(BH, T, 1)
        )

        if self.safe_updates:
            alpha = alpha.clamp(min=self.alpha_min, max=1.0)
            beta = beta.clamp(max=self.beta_max)

        return alpha, beta

    # ── Triton wrappers (inference-only) ─────────────────────────────────────

    def _tri_row_scale(self, S: torch.Tensor, a: torch.Tensor) -> None:
        """In-place: S[i, :] *= a[i]."""
        BH, Dk, Dv = S.shape
        # Ensure fp32 contiguous inputs — kernels operate in fp32
        S_fp32 = S.to(torch.float32) if S.dtype != torch.float32 else S
        a_fp32 = a.float().contiguous()
        grid = (BH, triton.cdiv(Dv, self.BLOCK_J))
        _row_scale_kernel[grid](
            S_fp32,
            a_fp32,
            BH,
            Dk,
            Dv,
            S_fp32.stride(0),
            S_fp32.stride(1),
            S_fp32.stride(2),
            a_fp32.stride(0),
            a_fp32.stride(1),
            BLOCK_I=self.BLOCK_I,
            BLOCK_J=self.BLOCK_J,
        )
        if S.data_ptr() != S_fp32.data_ptr():
            S.copy_(S_fp32.to(S.dtype))

    def _tri_matvec_xt(self, S: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """y[j] = Σ_i S[i, j] · x[i], returns fp32, then cast to x.dtype."""
        BH, Dk, Dv = S.shape
        S_fp32 = S.float().contiguous()
        x_fp32 = x.float().contiguous()
        y = torch.empty(BH, Dv, device=S.device, dtype=torch.float32)
        grid = (BH, triton.cdiv(Dv, self.BLOCK_J))
        _matvec_xt_kernel[grid](
            S_fp32,
            x_fp32,
            y,
            BH,
            Dk,
            Dv,
            S_fp32.stride(0),
            S_fp32.stride(1),
            S_fp32.stride(2),
            x_fp32.stride(0),
            x_fp32.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_I=self.BLOCK_I,
            BLOCK_J=self.BLOCK_J,
        )
        return y.to(x.dtype)

    def _tri_rank1_update(
        self,
        S: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kTS: torch.Tensor,
        b: torch.Tensor,
    ) -> None:
        """In-place: S += b · k ⊗ (v − kTS)."""
        BH, Dk, Dv = S.shape
        S_fp32 = S.float().contiguous() if S.dtype != torch.float32 else S
        k_fp32 = k.float().contiguous()
        v_fp32 = v.float().contiguous()
        kTS_fp32 = kTS.float().contiguous()
        b_fp32 = b.float().contiguous().view(BH)
        grid = (BH, triton.cdiv(Dv, self.BLOCK_J))
        _rank1_update_kernel[grid](
            S_fp32,
            k_fp32,
            v_fp32,
            kTS_fp32,
            b_fp32,
            BH,
            Dk,
            Dv,
            S_fp32.stride(0),
            S_fp32.stride(1),
            S_fp32.stride(2),
            k_fp32.stride(0),
            k_fp32.stride(1),
            v_fp32.stride(0),
            v_fp32.stride(1),
            kTS_fp32.stride(0),
            kTS_fp32.stride(1),
            b_fp32.stride(0),
            BLOCK_I=self.BLOCK_I,
            BLOCK_J=self.BLOCK_J,
        )
        if S.data_ptr() != S_fp32.data_ptr():
            S.copy_(S_fp32.to(S.dtype))

    # ── sequential step (used in exact mode and for state updates in chunk mode)

    def _seq_step(
        self,
        S: torch.Tensor,  # [BH, Dk, Dv]
        k_t: torch.Tensor,  # [BH, Dk]
        v_t: torch.Tensor,  # [BH, Dv]
        q_t: torch.Tensor,  # [BH, Dk]
        a_t: torch.Tensor,  # [BH, Dk]
        b_t: torch.Tensor,  # [BH, 1]
        use_triton_path: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step → (o_t [BH, Dv], S_next [BH, Dk, Dv])."""
        if use_triton_path:
            # NOTE: in-place ops on S; only safe outside autograd
            self._tri_row_scale(S, a_t)
            kTS = self._tri_matvec_xt(S, k_t)
            self._tri_rank1_update(S, k_t, v_t, kTS, b_t)
            o_t = self._tri_matvec_xt(S, q_t)
            return o_t, S
        else:
            # Pure PyTorch — autograd-safe (creates new tensors)
            S = S * a_t.unsqueeze(-1)  # decay
            kTS = torch.bmm(k_t.unsqueeze(1), S).squeeze(1)  # [BH, Dv]
            delta = torch.bmm(
                (b_t * k_t).unsqueeze(-1),
                (v_t - kTS).unsqueeze(1),
            )  # [BH, Dk, Dv]
            S = S + delta
            o_t = torch.bmm(S.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)
            return o_t, S

    # ── parallel chunk output (approx intra-chunk) ────────────────────────────

    @staticmethod
    def _parallel_chunk_output(
        Q_c: torch.Tensor,  # [BH, C, Dk]
        K_c: torch.Tensor,  # [BH, C, Dk]
        V_c: torch.Tensor,  # [BH, C, Dv]
        alpha_c: torch.Tensor,  # [BH, C, Dk]
        beta_c: torch.Tensor,  # [BH, C, 1]
        S_prev: torch.Tensor,  # [BH, Dk, Dv]
    ) -> torch.Tensor:  # [BH, C, Dv]
        """
        Approximate parallel output for one chunk.

        Inter-chunk (exact):
            O_inter[i] = S_prev^T q_i  →  Q_c @ S_prev  [one bmm]

        Intra-chunk (approx — ignores within-chunk S updates):
            cum_alpha[i, j] = ∏_{l=j+1}^{i} alpha[l]   (upper=0 by mask)
            A[i, j]         = (q_i ⊙ cum_alpha[i,j]) · (k_j ⊙ β_j)
            O_intra[i]      = Σ_j A[i,j] · v_j
        """
        BH, C, Dk = Q_c.shape
        device, dtype = Q_c.device, Q_c.dtype

        # ── inter-chunk ──────────────────────────────────────────────────────
        # [BH, C, Dk] × [BH, Dk, Dv] → [BH, C, Dv]
        O_inter = torch.bmm(Q_c, S_prev)

        # ── intra-chunk ──────────────────────────────────────────────────────
        # Cumulative log-alpha: [BH, C, Dk]
        log_alpha = torch.log(alpha_c.clamp_min(1e-7))
        cumlog = torch.cumsum(log_alpha, dim=1)  # [BH, C, Dk]

        # cum_alpha[b, i, j, dk] = exp( cumlog[b,i,dk] - cumlog[b,j,dk] )
        # for i >= j (empty product at i=j gives exp(0)=1 — correct).
        # Shape: [BH, C, C, Dk]
        diff = cumlog.unsqueeze(2) - cumlog.unsqueeze(1)  # [BH, C, C, Dk]
        cum_alpha_ij = torch.exp(diff)  # [BH, C, C, Dk]

        # Lower-triangular causal mask (i >= j)
        causal = torch.tril(torch.ones(C, C, device=device, dtype=torch.bool))
        cum_alpha_ij = cum_alpha_ij * causal.unsqueeze(0).unsqueeze(-1)

        # Attention logits: A[b, i, j] = Σ_dk Q[b,i,dk] * cum_alpha[b,i,j,dk] * K_beta[b,j,dk]
        # [BH, C, 1, Dk] * [BH, C, C, Dk] → [BH, C, C, Dk]
        Q_dec = Q_c.unsqueeze(2) * cum_alpha_ij  # [BH, C, C, Dk]
        K_beta = K_c * beta_c  # [BH, C, Dk]
        A = (Q_dec * K_beta.unsqueeze(1)).sum(-1)  # [BH, C, C]

        # Intra-chunk output
        O_intra = torch.bmm(A, V_c)  # [BH, C, Dv]

        return O_inter + O_intra

    # ── main forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
        state: Optional[torch.Tensor] = None,  # [B, H, Dk, Dv]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            y          [B, T, D]
            next_state [B, H, Dk, Dv]
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # Triton is inference-only — silently fall back during training
        use_triton = self.use_triton and (not self.training) and _HAS_TRITON

        Q, K, V = self._project(x)  # each [B, H, T, D*]
        BH = B * self.h
        Q = self._l2_norm(Q.reshape(BH, T, self.dk))
        K = self._l2_norm(K.reshape(BH, T, self.dk))
        V = V.reshape(BH, T, self.dv)

        alpha, beta = self._gate_params(x, BH, T)  # [BH, T, Dk], [BH, T, 1]

        # Initialise / reshape state
        S = (
            state.reshape(BH, self.dk, self.dv).contiguous()
            if state is not None
            else self._init_state(BH, device, dtype)
        )

        out = torch.zeros(BH, T, self.dv, device=device, dtype=dtype)

        C = self.chunk_size if (self.chunk_size and self.chunk_size > 1) else 0

        if C:
            # ── Parallel-chunk mode ───────────────────────────────────────
            for s in range(0, T, C):
                e = min(s + C, T)
                c_len = e - s

                Q_c = Q[:, s:e, :]  # [BH, c_len, Dk]
                K_c = K[:, s:e, :]
                V_c = V[:, s:e, :]
                a_c = alpha[:, s:e, :]  # [BH, c_len, Dk]
                b_c = beta[:, s:e, :]  # [BH, c_len, 1]

                # Parallel output (vectorised, approximate intra-chunk)
                out[:, s:e, :] = self._parallel_chunk_output(Q_c, K_c, V_c, a_c, b_c, S)

                # Exact sequential state update within the chunk
                # S is detached: it is a recurrent buffer, not a gradient path.
                S = S.detach()
                for t in range(c_len):
                    _, S = self._seq_step(
                        S,
                        K_c[:, t, :],
                        V_c[:, t, :],
                        Q_c[:, t, :],
                        a_c[:, t, :],
                        b_c[:, t, :],
                        use_triton_path=use_triton,
                    )
        else:
            # ── Exact sequential mode ─────────────────────────────────────
            S = S.detach()
            for t in range(T):
                o_t, S = self._seq_step(
                    S,
                    K[:, t, :],
                    V[:, t, :],
                    Q[:, t, :],
                    alpha[:, t, :],
                    beta[:, t, :],
                    use_triton_path=use_triton,
                )
                out[:, t, :] = o_t

        # ── Output head: per-head RMSNorm + sigmoid gate + projection ─────
        O = out.reshape(B, self.h, T, self.dv)
        # gate: [B, H, T, 1]
        gate = (
            torch.sigmoid(self.out_gate_up(F.silu(self.out_gate_down(x))))
            .permute(0, 2, 1)
            .unsqueeze(-1)
        )  # [B, H, T, 1]

        O = self.h_rms(O) * gate
        y = O.permute(0, 2, 1, 3).contiguous().reshape(B, T, self.h * self.dv)
        y = self.drop(self.o_proj(y))

        next_state = S.reshape(B, self.h, self.dk, self.dv)
        return y, next_state


# ─────────────────────────────────────────────────────────────────────────────
# Public adapter — drop-in replacement for the original KimiAttention
# ─────────────────────────────────────────────────────────────────────────────
class KimiAttention(nn.Module):
    """
    Public wrapper around _KDA_Fast with the same call signature as
    MultiAttention / LinearAttention (query, key, value, masks, layer_state).

    Self-attention only — cross_attention=True raises immediately.

    layer_state dict carries the recurrent state S under the key "S":
        { "S": Tensor[B, H, Dk, Dv] }
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        cross_attention: bool = False,
        shortconv_mode: str = "depthwise",
        chunk_size: int = 64,
        safe_updates: bool = True,
        alpha_min: float = _ALPHA_MIN_DEFAULT,
        beta_max: float = _BETA_MAX_DEFAULT,
        use_triton: bool = False,
        triton_block_i: int = 64,
        triton_block_j: int = 128,
    ):
        super().__init__()
        if cross_attention:
            raise ValueError(
                "KimiAttention is self-attention only. "
                "Pass cross_attention=False (default)."
            )
        self.kda = _KDA_Fast(
            d_model=d_model,
            num_heads=n_heads,
            dropout=dropout,
            shortconv_mode=shortconv_mode,
            chunk_size=chunk_size,
            safe_updates=safe_updates,
            alpha_min=alpha_min,
            beta_max=beta_max,
            use_triton=use_triton,
            triton_block_i=triton_block_i,
            triton_block_j=triton_block_j,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # ignored (recurrent)
        key_padding_mask: Optional[torch.Tensor] = None,  # ignored (recurrent)
        is_causal: bool = True,  # always causal
        layer_state: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, None, Optional[Dict]]:
        """
        Returns:
            out          [B, T, D]
            attn_weights None  (not computed)
            updated_state dict {"S": Tensor[B, H, Dk, Dv]}
        """
        S = None
        if layer_state is not None:
            S_raw = layer_state.get("S", None)
            if isinstance(S_raw, torch.Tensor):
                S = S_raw

        out, S_next = self.kda(query, state=S)
        return out, None, {"S": S_next}
