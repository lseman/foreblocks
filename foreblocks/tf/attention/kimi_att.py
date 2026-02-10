# -*- coding: utf-8 -*-
"""
kimi_att_fast.py — Optimized Kimi (KDA) Linear Attention with optional Triton

Highlights
----------
• Fast PyTorch path (head packing + bmm), as before.
• Optional Triton kernels for the state update & matvecs (use_triton=True).
• Correct causal ShortConv (crop to original T).
• Autograd-safe (no problematic in-place ops on graph-tracked tensors).
• Self-attention only (cross-attn is intentionally blocked).

Recommended knobs
-----------------
• AMP (bf16/fp16), torch.compile, TF32 matmul, larger batch/H.
• chunk_size=128 or 256 for long sequences (reduces Python overhead).
• use_triton=True if Triton is installed (pip install triton).

"""

from __future__ import annotations
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try Triton (optional)
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# ──────────────────────────────────────────────────────────────────────────────
# ShortConv (causal)
# ──────────────────────────────────────────────────────────────────────────────
class _ShortConv1x(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 4, mode: str = "depthwise"):
        super().__init__()
        self.mode = mode
        if mode == "off":
            self.conv = None
        else:
            pad = kernel_size - 1
            groups = d_model if mode == "depthwise" else 1
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=groups, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,T,D]
        if self.conv is None:
            return x
        B, T0, D = x.shape
        x = x.transpose(1, 2).contiguous()   # [B,D,T]
        x = self.conv(x)                     # [B,D,T0 + (k-1)]
        x = x[:, :, :T0].contiguous()        # crop to ORIGINAL T
        x = x.transpose(1, 2).contiguous()   # [B,T0,D]
        return F.silu(x)


# ──────────────────────────────────────────────────────────────────────────────
# Head-wise RMSNorm
# ──────────────────────────────────────────────────────────────────────────────
class _HeadwiseRMSNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,H,T,Dh]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        w = self.weight.view(1, self.weight.size(0), 1, self.weight.size(1))
        return x * w


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernels (optional)
# S is [BH, Dk, Dv]; k, q, v, a are [BH, Dk] or [BH, Dv]; b is [BH, 1]
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _row_scale_kernel(
        S_ptr, a_ptr,
        BH: tl.constexpr, Dk: tl.constexpr, Dv: tl.constexpr,
        strideSB, strideSi, strideSj,
        strideaB, strideai,
        BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    ):
        pid_b = tl.program_id(0)  # over BH
        pid_j = tl.program_id(1)  # over j tiles
        offs_i = tl.arange(0, BLOCK_I)
        offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)

        # loop i in tiles
        for i0 in range(0, Dk, BLOCK_I):
            i_idx = i0 + offs_i
            j_idx = offs_j

            mask_i = i_idx < Dk
            mask_j = j_idx < Dv
            a = tl.load(a_ptr + pid_b * strideaB + i_idx * strideai, mask=mask_i, other=1.0)
            # load tile S[i,j]
            S_tile = tl.load(
                S_ptr + pid_b * strideSB + i_idx[:, None] * strideSi + j_idx[None, :] * strideSj,
                mask=mask_i[:, None] & mask_j[None, :],
                other=0.0,
            )
            S_tile = S_tile * a[:, None]
            tl.store(
                S_ptr + pid_b * strideSB + i_idx[:, None] * strideSi + j_idx[None, :] * strideSj,
                S_tile,
                mask=mask_i[:, None] & mask_j[None, :],
            )

    @triton.jit
    def _rank1_update_kernel(
        S_ptr, k_ptr, v_ptr, kTS_ptr, b_ptr,
        BH: tl.constexpr, Dk: tl.constexpr, Dv: tl.constexpr,
        strideSB, strideSi, strideSj,
        stridekB, strideki,
        stridevB, stridevj,
        strideyB, strideyj,   # kTS strides
        stridebB, strideb1,
        BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_j = tl.program_id(1)
        offs_i = tl.arange(0, BLOCK_I)
        offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)

        b = tl.load(b_ptr + pid_b * stridebB + 0 * strideb1)
        v_tile = tl.load(v_ptr + pid_b * stridevB + offs_j * stridevj,
                         mask=offs_j < Dv, other=0.0)
        kTS_tile = tl.load(kTS_ptr + pid_b * strideyB + offs_j * strideyj,
                           mask=offs_j < Dv, other=0.0)

        for i0 in range(0, Dk, BLOCK_I):
            i_idx = i0 + offs_i
            j_idx = offs_j
            mask_i = i_idx < Dk
            mask_j = j_idx < Dv

            k_i = tl.load(k_ptr + pid_b * stridekB + i_idx * strideki, mask=mask_i, other=0.0)

            S_tile = tl.load(
                S_ptr + pid_b * strideSB + i_idx[:, None] * strideSi + j_idx[None, :] * strideSj,
                mask=mask_i[:, None] & mask_j[None, :],
                other=0.0,
            )
            # S += b * k ⊗ (v - kTS)
            delta = (b * k_i)[:, None] * (v_tile[None, :] - kTS_tile[None, :])
            S_tile = S_tile + delta
            tl.store(
                S_ptr + pid_b * strideSB + i_idx[:, None] * strideSi + j_idx[None, :] * strideSj,
                S_tile,
                mask=mask_i[:, None] & mask_j[None, :],
            )

    @triton.jit
    def _matvec_xt_kernel(
        S_ptr, x_ptr, y_ptr,
        BH: tl.constexpr, Dk: tl.constexpr, Dv: tl.constexpr,
        strideSB, strideSi, strideSj,
        stridexB, stridexi,
        strideyB, strideyj,
        BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    ):
        # computes y[j] = sum_i S[i,j] * x[i], for each BH
        pid_b = tl.program_id(0)
        pid_j = tl.program_id(1)
        offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
        mask_j = offs_j < Dv

        acc = tl.zeros([BLOCK_J], dtype=tl.float32)
        for i0 in range(0, Dk, BLOCK_I):
            offs_i = i0 + tl.arange(0, BLOCK_I)
            mask_i = offs_i < Dk
            x = tl.load(x_ptr + pid_b * stridexB + offs_i * stridexi, mask=mask_i, other=0.0)

            S_tile = tl.load(
                S_ptr + pid_b * strideSB + offs_i[:, None] * strideSi + offs_j[None, :] * strideSj,
                mask=mask_i[:, None] & mask_j[None, :],
                other=0.0,
            )
            acc += tl.sum(S_tile.to(tl.float32) * x[:, None].to(tl.float32), axis=0)

        tl.store(y_ptr + pid_b * strideyB + offs_j * strideyj, acc, mask=mask_j)


# ──────────────────────────────────────────────────────────────────────────────
# Core KDA (fast) with optional Triton
# ──────────────────────────────────────────────────────────────────────────────
class _KDA_Fast(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        gate_rank: Optional[int] = None,
        dropout: float = 0.0,
        shortconv_mode: str = "depthwise",
        chunk_size: int = 0,
        safe_updates: bool = True,
        use_triton: bool = False,      # <-- new
        triton_block_i: int = 64,
        triton_block_j: int = 128,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.h = num_heads
        self.dk = d_k or (d_model // num_heads)
        self.dv = d_v or (d_model // num_heads)
        self.d_model = d_model
        self.chunk_size = int(chunk_size) if chunk_size is not None else 0
        self.safe_updates = bool(safe_updates)
        self.use_triton = bool(use_triton and _HAS_TRITON)
        self.BLOCK_I = int(triton_block_i)
        self.BLOCK_J = int(triton_block_j)

        # ShortConv
        self.pre_q = _ShortConv1x(d_model, kernel_size=4, mode=shortconv_mode)
        self.pre_k = _ShortConv1x(d_model, kernel_size=4, mode=shortconv_mode)
        self.pre_v = _ShortConv1x(d_model, kernel_size=4, mode=shortconv_mode)

        # Projections
        self.q_proj = nn.Linear(d_model, num_heads * self.dk, bias=True)
        self.k_proj = nn.Linear(d_model, num_heads * self.dk, bias=True)
        self.v_proj = nn.Linear(d_model, num_heads * self.dv, bias=True)

        # α (low-rank) and β (per-head)
        r = gate_rank or self.dk
        self.alpha_down = nn.Linear(d_model, r, bias=True)
        self.alpha_up   = nn.Linear(r, num_heads * self.dk, bias=True)
        self.beta_proj  = nn.Linear(d_model, num_heads, bias=True)

        # Headwise output norm + gate
        self.h_rms = _HeadwiseRMSNorm(num_heads, self.dv)
        self.out_gate_down = nn.Linear(d_model, r, bias=True)
        self.out_gate_up   = nn.Linear(r, num_heads, bias=True)

        self.o_proj = nn.Linear(num_heads * self.dv, d_model, bias=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _l2(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps).sqrt())

    def _init_state(self, BH: int, device, dtype) -> torch.Tensor:
        return torch.zeros(BH, self.dk, self.dv, device=device, dtype=dtype)

    # ---- projections ---------------------------------------------------------
    def _project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qx = self.pre_q(x)
        kx = self.pre_k(x)
        vx = self.pre_v(x)
        B, T, _ = x.shape
        Q = self.q_proj(qx).view(B, T, self.h, self.dk).permute(0, 2, 1, 3).contiguous()
        K = self.k_proj(kx).view(B, T, self.h, self.dk).permute(0, 2, 1, 3).contiguous()
        V = self.v_proj(vx).view(B, T, self.h, self.dv).permute(0, 2, 1, 3).contiguous()
        assert Q.shape[2] == T == K.shape[2] == V.shape[2], "ShortConv T drift detected"
        return Q, K, V

    # ---- Triton wrappers -----------------------------------------------------
    def _triton_row_scale(self, S: torch.Tensor, a: torch.Tensor):
        BH, Dk, Dv = S.shape
        grid = (BH, triton.cdiv(Dv, self.BLOCK_J))
        _row_scale_kernel[grid](
            S, a,
            BH, Dk, Dv,
            S.stride(0), S.stride(1), S.stride(2),
            a.stride(0), a.stride(1),
            BLOCK_I=self.BLOCK_I, BLOCK_J=self.BLOCK_J,
            num_warps=4, num_stages=2,
        )

    def _triton_matvec_xt(self, S: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
        # y[j] = sum_i S[i,j]*x[i]
        BH, Dk, Dv = S.shape
        grid = (BH, triton.cdiv(Dv, self.BLOCK_J))
        _matvec_xt_kernel[grid](
            S, x, out,
            BH, Dk, Dv,
            S.stride(0), S.stride(1), S.stride(2),
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_I=self.BLOCK_I, BLOCK_J=self.BLOCK_J,
            num_warps=4, num_stages=2,
        )

    def _triton_rank1_update(self, S: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kTS: torch.Tensor, b: torch.Tensor):
        BH, Dk, Dv = S.shape
        grid = (BH, triton.cdiv(Dv, self.BLOCK_J))
        _rank1_update_kernel[grid](
            S, k, v, kTS, b,
            BH, Dk, Dv,
            S.stride(0), S.stride(1), S.stride(2),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            kTS.stride(0), kTS.stride(1),
            b.stride(0), b.stride(1) if b.ndim > 1 else 0,
            BLOCK_I=self.BLOCK_I, BLOCK_J=self.BLOCK_J,
            num_warps=4, num_stages=2,
        )

    # ---- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        x: [B,T,D]
        state: [B,H,Dk,Dv] or None
        returns y:[B,T,D], next_state:[B,H,Dk,Dv]
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        Q, K, V = self._project(x)
        BH = B * self.h
        Q = self._l2(Q.view(BH, T, self.dk))
        K = self._l2(K.view(BH, T, self.dk))
        V = V.view(BH, T, self.dv)

        alpha = torch.sigmoid(self.alpha_up(F.silu(self.alpha_down(x)))).view(B, T, self.h, self.dk)
        alpha = alpha.permute(0, 2, 1, 3).contiguous().view(BH, T, self.dk)
        beta = torch.sigmoid(self.beta_proj(x)).permute(0, 2, 1).contiguous().view(BH, T, 1)  # [BH,T,1]

        S = state.view(BH, self.dk, self.dv).contiguous() if state is not None else self._init_state(BH, device, dtype)
        out = x.new_zeros(BH, T, self.dv)

        starts = range(0, T, self.chunk_size) if (self.chunk_size and self.chunk_size > 0) else [0]
        for s in starts:
            e = min(s + (self.chunk_size or T), T)
            for t in range(s, e):
                q_t = Q[:, t, :]                      # [BH,Dk]
                k_t = K[:, t, :]                      # [BH,Dk]
                v_t = V[:, t, :]                      # [BH,Dv]
                a_t = alpha[:, t, :]                  # [BH,Dk]
                b_t = beta[:, t, :]                   # [BH,1]

                if self.use_triton:
                    # 1) S <- Diag(a_t) @ S
                    self._triton_row_scale(S, a_t)
                    # 2) kTS = k_t^T S
                    kTS = torch.empty(BH, self.dv, device=device, dtype=torch.float32)
                    self._triton_matvec_xt(S, k_t, kTS)
                    # 3) Rank-1 updates: S += b·k ⊗ (v - kTS)
                    self._triton_rank1_update(S, k_t, v_t, kTS, b_t)
                    # 4) o_t = S^T q_t
                    o_t = torch.empty(BH, self.dv, device=device, dtype=torch.float32)
                    self._triton_matvec_xt(S, q_t, o_t)
                else:
                    # ---- PyTorch fallback (bmm path) ----
                    # 1) decay
                    S = S * a_t.unsqueeze(-1)
                    # 2) k^T S
                    kTS = torch.bmm(k_t.unsqueeze(1), S).squeeze(1)  # [BH,Dv]
                    # 3) rank-1: S += b*k ⊗ (v - kTS)
                    delta = torch.bmm(b_t.unsqueeze(-1) * k_t.unsqueeze(-1), (v_t - kTS).unsqueeze(1))
                    S = S + delta
                    # 4) read
                    o_t = torch.bmm(S.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)

                out[:, t, :] = o_t.to(out.dtype)

        # reshape, normalize heads, gate, project
        O = out.view(B, self.h, T, self.dv)
        gate = torch.sigmoid(self.out_gate_up(F.silu(self.out_gate_down(x)))).permute(0, 2, 1).unsqueeze(-1)  # [B,H,T,1]
        O = self.h_rms(O) * gate
        y = O.permute(0, 2, 1, 3).contiguous().view(B, T, self.h * self.dv)
        y = self.drop(self.o_proj(y))
        return y, S.view(B, self.h, self.dk, self.dv)


# ──────────────────────────────────────────────────────────────────────────────
# Public adapter — same API as before
# ──────────────────────────────────────────────────────────────────────────────
class KimiAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        cross_attention: bool = False,
        shortconv_mode: str = "depthwise",
        chunk_size: int = 0,
        safe_updates: bool = True,
        use_triton: bool = False,          # <-- enable Triton path when available
        triton_block_i: int = 64,
        triton_block_j: int = 128,
    ):
        super().__init__()
        if cross_attention:
            raise ValueError("KimiAttention is self-attention only (cross_attention=False).")
        self.kda = _KDA_Fast(
            d_model=d_model,
            num_heads=n_heads,
            dropout=dropout,
            shortconv_mode=shortconv_mode,
            chunk_size=chunk_size,
            safe_updates=safe_updates,
            use_triton=use_triton,
            triton_block_i=triton_block_i,
            triton_block_j=triton_block_j,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        layer_state: Optional[Dict] = None,
    ):
        S = None
        if layer_state is not None and isinstance(layer_state.get("S", None), torch.Tensor):
            S = layer_state["S"]
        out, S = self.kda(query, state=S)
        updated = {"S": S}
        return out, None, updated
