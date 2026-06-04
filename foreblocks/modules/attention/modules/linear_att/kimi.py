"""
kimi_att.py — Kimi Delta Attention (KDA) per Kimi Linear

Reference: Kimi Linear: An Expressive, Efficient Attention Architecture
  - GitHub: https://github.com/MoonshotAI/Kimi-Linear
  - Paper:  https://arxiv.org/abs/2510.26692

Core recurrence (KDA — "DeltaNet with fine-grained gating"):
    S_t = (I - β_t · k_t k_t^T) · Diag(α_t) · S_{t-1} + β_t · k_t · v_t^T
    o_t = S_t^T · q_t

where:
  - S_t in R^{d_h × d_v} — recurrent state (same shape as Gated DeltaNet)
  - q_t, k_t in R^{d_h} — queries and keys (L2-normalized per official impl)
  - v_t in R^{d_v} — values
  - α_t in [0,1]^{d_h} — per-channel decay (diagonal, *vector* per head)
  - β_t in [0,1] — scalar gate controlling write strength

Gating mechanism (differs from Gated DeltaNet):
  - A_log in R^{H}  — constant per-head parameter (uniform [1,16], log-space)
  - dt_bias in R^{H×K}  — per-head per-key-dim inverse time step init
  - f_proj: hidden → head_v_dim → gate_dim  (bottleneck, no bias)
  - g = -exp(A_log) · softplus(f_proj(x) + dt_bias)   # log-space gate
  - α = exp(g.cumsum(dim=time))  # cumulative decay
  - b = b_proj(x)  # raw logits → β = sigmoid(b) inside kernel

  This gives *per-channel* forget gate (g has shape [B,T,H,K]) vs Gated
  DeltaNet's scalar per-head gate [B,T,H].

Architecture highlights (per Kimi Linear paper):
  - Hybrid: 3:1 KDA-to-MLA layer ratio
  - Uses short (causal) convolutions on q, k, v projections
  - Reduces KV cache by up to 75%, up to 6× decoding throughput
  - Trained on 5.7T tokens (48B total / 3B activated params)

Reference implementation: fla-org/flash-linear-attn (KimiDeltaAttention layer)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.ops.attention import can_use_fla_kda, fla_kda_forward


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_EPS: float = 1e-6
_DEFAULT_NORM_EPS: float = 1e-5
_DEFAULT_CHUNK_SIZE: int = 64


# ─────────────────────────────────────────────────────────────────────────────
# Causal ShortConv (mirrors fla.modules.ShortConvolution)
# ─────────────────────────────────────────────────────────────────────────────
class _ShortConv1d(nn.Module):
    """Causal depthwise or pointwise Conv1d with SiLU activation.

    Mirrors fla.modules.ShortConvolution interface:
    __init__(hidden_size, kernel_size, bias, activation)
    forward(x, cache=None, output_final_state=False, cu_seqlens=None)
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        assert activation == "silu", "Only 'silu' activation supported"
        pad = kernel_size - 1
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,  # depthwise
            padding=pad,
            bias=bias,
        )
        self.kernel_size = kernel_size

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, D, T] (already transposed from [B, T, D])
            cache: [B, D, kernel_size - 1] previous inputs
            output_final_state: whether to return final conv state
            cu_seqlens: unused (variable-length support placeholder)

        Returns:
            output: [B, D, T] with SiLU
            final_cache: [B, D, kernel_size - 1] or None
        """
        T0 = x.size(2)
        # Shift register: prepend cache
        if cache is not None:
            x = torch.cat([cache, x], dim=2)[:, :, : T0 + self.kernel_size - 1]
        else:
            zero_pad = torch.zeros(
                x.size(0), x.size(1), self.kernel_size - 1,
                device=x.device, dtype=x.dtype,
            )
            x = torch.cat([zero_pad, x], dim=2)

        out = self.conv(x)[:, :, :T0]
        out = F.silu(out)

        final_cache = None
        if output_final_state:
            final_cache = x[:, :, T0:]  # last (kernel_size - 1) inputs

        return out, final_cache


# ─────────────────────────────────────────────────────────────────────────────
# Head-wise RMSNorm (for output gating)
# ─────────────────────────────────────────────────────────────────────────────
class _HeadwiseRMSNorm(nn.Module):
    """Per-head RMSNorm: normalizes each head independently over d_v."""

    def __init__(self, num_heads: int, head_dim: int, eps: float = _DEFAULT_NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, T, Dv] → [B, H, T, Dv]"""
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        w = self.weight.view(1, self.weight.size(0), 1, self.weight.size(1))
        return x * w


# ─────────────────────────────────────────────────────────────────────────────
# KDA Sequential Step — Kimi Linear recurrence
# ─────────────────────────────────────────────────────────────────────────────
def _kda_seq_step(
    S: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    q_t: torch.Tensor,
    alpha_t: torch.Tensor,
    beta_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One KDA recurrence step (exact, autograd-safe PyTorch).

    S_t = (I - β_t · k_t k_t^T) · Diag(α_t) · S_{t-1} + β_t · k_t · v_t^T
    o_t = S_t^T · q_t
    """
    # Step 1: Decay  Diag(α_t) S_{t-1}
    S = alpha_t.unsqueeze(-1) * S  # [BH, Dk, Dv]

    # Step 2: Forget  S − β_t · k_t (k_t^T S)
    kTS = torch.bmm(k_t.unsqueeze(1), S).squeeze(1)  # [BH, Dv]
    S = S - beta_t.unsqueeze(-1) * torch.bmm(
        k_t.unsqueeze(-1), kTS.unsqueeze(1)
    ).squeeze(1)  # [BH, Dk, Dv]

    # Step 3: Write  S + β_t · k_t · v_t^T
    S = S + beta_t.unsqueeze(-1) * torch.bmm(
        k_t.unsqueeze(-1), v_t.unsqueeze(1)
    )  # [BH, Dk, Dv]

    # Output: o_t = S_t^T q_t
    o_t = torch.bmm(S.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)  # [BH, Dv]

    return o_t, S


# ─────────────────────────────────────────────────────────────────────────────
# KDA Chunk-Parallel — verified sequential within chunk
# ─────────────────────────────────────────────────────────────────────────────
def _kda_chunk_parallel(
    S: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    alpha: torch.Tensor,  # [BH, T, Dk] per-step decay in (0, 1]
    beta: torch.Tensor,   # [BH, T, 1] already sigmoid'd
    out: torch.Tensor,
    C: int,
) -> torch.Tensor:
    """
    Chunk-parallel KDA: processes each chunk using the verified
    sequential recurrence.  State is carried forward between chunks,
    giving an O(T / C) speedup from inter-chunk parallelism of the
    carry operation.

    Within a chunk, the KDA recurrence is applied step-by-step:
        S_t = α_t · S_{t-1}
        S_t = S_t - β_t · k_t · (k_t^T · S_t)
        S_t = S_t + β_t · k_t · v_t
        o_t = S_t^T · q_t
    """
    B, H, T, Dk = Q.shape[:4]
    Dv = V.shape[-1]
    BH = B * H

    # Flatten to [BH, T, D*]
    Qf = Q.view(BH, T, Dk)
    Kf = K.view(BH, T, Dk)
    Vf = V.view(BH, T, Dv)

    for s in range(0, T, C):
        e = min(s + C, T)
        L = e - s

        for l in range(L):
            idx = s + l
            k_t = Kf[:, idx, :]   # [BH, Dk]
            v_t = Vf[:, idx, :]   # [BH, Dv]
            q_t = Qf[:, idx, :]   # [BH, Dk]
            a_t = alpha[:, idx, :] # [BH, Dk]
            b_t = beta[:, idx, :]  # [BH, 1]

            # Sequential KDA step
            # 1) Decay
            S = a_t.unsqueeze(-1) * S
            # 2) Forget (rank-1 update)
            kTS = torch.bmm(k_t.unsqueeze(1), S).squeeze(1)  # [BH, Dv]
            S = S - b_t.unsqueeze(-1) * torch.bmm(
                k_t.unsqueeze(-1), kTS.unsqueeze(1)
            ).squeeze(1)
            # 3) Write
            S = S + b_t.unsqueeze(-1) * torch.bmm(
                k_t.unsqueeze(-1), v_t.unsqueeze(1)
            )
            # 4) Output
            out[:, idx, :] = torch.bmm(S.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)

    return S


# ─────────────────────────────────────────────────────────────────────────────
# Core KDA layer — Kimi Linear architecture
# ─────────────────────────────────────────────────────────────────────────────
class _KDA_Fast(nn.Module):
    """
    Kimi Delta Attention (KDA) — Kimi Linear core.

    Follows the official Kimi Delta Attention architecture:
      https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py

    Key differences from Gated DeltaNet:
      - Per-channel forget gate (g shape [B,T,H,K]) vs scalar [B,T,H]
      - A_log: constant per-head parameter (not learned per-step)
      - dt_bias: per-head per-key-dim init from inverse time step
      - f_proj: bottleneck (hidden → head_dim → gate_dim) for gate
      - g_proj: two-layer bottleneck for output gating
      - b_proj: single linear for beta (no low-rank)

    Recurrence:
      S_t = (I - β_t · k_t k_t^T) · Diag(α_t) · S_{t-1} + β_t · k_t · v_t^T
      o_t = S_t^T · q_t
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        expand_v: float = 1.0,
        dropout: float = 0.0,
        shortconv_mode: str = "depthwise",
        conv_kernel: int = 4,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        safe_updates: bool = True,
        alpha_min: float = 0.1,
        use_triton: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.h = num_heads
        self.dk = d_k or (d_model // num_heads)
        self.dv = int(d_v or int(d_model // num_heads * expand_v))
        self.d_model = d_model
        self.expand_v = expand_v

        self.chunk_size = max(int(chunk_size), 0)
        self.safe_updates = bool(safe_updates)
        self.alpha_min = float(alpha_min)
        self.use_triton = False

        # ── Short convolutions (causal, mirrors fla.modules.ShortConvolution) ──
        self.use_short_conv = shortconv_mode != "off"
        if self.use_short_conv:
            self.q_conv1d = _ShortConv1d(
                self.dk * self.h,
                kernel_size=conv_kernel,
                bias=False,
                activation="silu",
            )
            self.k_conv1d = _ShortConv1d(
                self.dk * self.h,
                kernel_size=conv_kernel,
                bias=False,
                activation="silu",
            )
            self.v_conv1d = _ShortConv1d(
                self.dv * self.h,
                kernel_size=conv_kernel,
                bias=False,
                activation="silu",
            )
        else:
            self.q_conv1d = self.k_conv1d = self.v_conv1d = None

        # ── Q / K / V projections (no bias — matches official) ──
        self.q_proj = nn.Linear(d_model, num_heads * self.dk, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * self.dk, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * self.dv, bias=False)

        # ── Gating: f_proj (for α) and b_proj (for β) ──
        # f_proj: bottleneck structure [d_model → head_dim → gate_dim]
        self.gate_dim = self.dk  # per value-head, per key-dim gating
        self.f_proj = nn.Sequential(
            nn.Linear(d_model, self.dv, bias=False),
            nn.Linear(self.dv, self.h * self.gate_dim, bias=False),
        )

        # b_proj: single linear [d_model → num_heads]
        self.b_proj = nn.Linear(d_model, self.h, bias=False)

        # ── A_log (constant per-head) and dt_bias (per-head per-key-dim) ──
        # A_log: uniform [1, 16] in linear space → log
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.h, dtype=torch.float32).uniform_(1, 16))
        )
        self.A_log._no_weight_decay = True

        # dt_bias: inverse time step initialization
        dt = torch.exp(
            torch.rand(self.h * self.dk, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # ── Output: g_proj (two-layer bottleneck) + RMSNorm + projection ──
        self.g_proj = nn.Sequential(
            nn.Linear(d_model, self.dv, bias=False),
            nn.Linear(self.dv, self.h * self.dv, bias=True),
        )
        self.h_rms = _HeadwiseRMSNorm(self.h, self.dv, eps=_DEFAULT_NORM_EPS)
        self.o_proj = nn.Linear(self.h * self.dv, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _l2_norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """L2 normalize last dimension."""
        return x / x.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()

    def _init_state(self, BH: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(BH, self.dk, self.dv, device=device, dtype=dtype)

    def _project_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project x → (Q, K, V) with short convolutions and reshape.

        Returns each as [B, T, H, D*].
        """
        B, T, _ = x.shape

        if self.use_short_conv:
            q_raw = self.q_proj(x).transpose(1, 2)  # [B, H*dk, T]
            k_raw = self.k_proj(x).transpose(1, 2)
            v_raw = self.v_proj(x).transpose(1, 2)

            q, _ = self.q_conv1d(q_raw, output_final_state=False)
            k, _ = self.k_conv1d(k_raw, output_final_state=False)
            v, _ = self.v_conv1d(v_raw, output_final_state=False)

            # SiLU activation after conv (official applies SiLU to projections)
            q = F.silu(q).transpose(1, 2)  # [B, T, H*dk]
            k = F.silu(k).transpose(1, 2)
            v = F.silu(v).transpose(1, 2)
        else:
            q = F.silu(self.q_proj(x))
            k = F.silu(self.k_proj(x))
            v = F.silu(self.v_proj(x))

        # Reshape: [B, T, H*D] → [B, T, H, D]
        q = q.view(B, T, self.h, self.dk).permute(0, 2, 1, 3)  # [B, H, T, Dk]
        k = k.view(B, T, self.h, self.dk).permute(0, 2, 1, 3)
        v = v.view(B, T, self.h, self.dv).permute(0, 2, 1, 3)

        return q, k, v

    def _compute_gate_params(
        self, x: torch.Tensor, BH: int, T: int,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            alpha_step [BH, T, Dk]  per-step decay in (0, 1]
            g_flat     [BH, T, Dk]  per-step log-decay
            beta       [BH, T, 1]
        """
        B = x.size(0)

        # Gate parameters: per-step decay α and β
        # Following Kimi Linear KDA gating:
        #   g = -exp(A_log) * softplus(f_proj(x) + dt_bias)   [log-space per-step decay]
        #   α_step = exp(g)                                    [per-step decay in (0,1]]
        #   α_cum = exp(cumsum(g))                             [cumulative decay]
        f_out = self.f_proj(x)  # [B, T, H*gate_dim]
        f_out = f_out.view(B, T, self.h, self.gate_dim)  # [B, T, H, K]

        # dt_bias: [H*K] → [1, 1, H, K]
        if dt_bias is not None:
            dt_b = dt_bias.view(1, 1, self.h, self.gate_dim)
        else:
            dt_b = self.dt_bias.view(1, 1, self.h, self.gate_dim)

        # A_log: [H] → [1, 1, H, 1] for broadcasting with [B, T, H, K]
        if A_log is not None:
            a_l = A_log.view(1, 1, self.h, 1)
        else:
            a_l = self.A_log.view(1, 1, self.h, 1)

        # g is the log of per-step decay (negative value)
        g = -a_l * F.softplus(f_out + dt_b)  # [B, T, H, K] (negative)

        # Per-step decay: α_step = exp(g) in (0, 1]
        alpha_step = g.exp()  # [B, T, H, K]

        if self.safe_updates:
            alpha_step = alpha_step.clamp(min=self.alpha_min, max=1.0)
            g = alpha_step.clamp_min(1e-12).log()

        # Reshape to [BH, T, D*]
        alpha_step = alpha_step.permute(0, 2, 1, 3).reshape(BH, T, self.dk)
        g_flat = g.permute(0, 2, 1, 3).reshape(BH, T, self.dk)  # [BH, T, Dk]

        # Beta: sigmoid(b_proj(x))
        b_raw = self.b_proj(x)  # [B, T, H]
        beta = torch.sigmoid(b_raw)  # [B, T, H]
        beta = beta.permute(0, 2, 1).contiguous().reshape(BH, T, 1)

        return alpha_step, g_flat, beta

    # ── main forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
        state: torch.Tensor | None = None,  # [B, H, Dk, Dv]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            alpha_step [BH, T, Dk]  per-step decay in (0, 1]
            g_flat     [BH, T, Dk]  per-step log-decay
            beta       [BH, T, 1]
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        Q, K, V = self._project_qkv(x)  # each [B, H, T, D*]

        # L2 normalize Q and K (matches official kernel)
        BH = B * self.h
        Q = self._l2_norm(Q.reshape(BH, T, self.dk)).view(B, self.h, T, self.dk)
        K = self._l2_norm(K.reshape(BH, T, self.dk)).view(B, self.h, T, self.dk)
        V_heads = V.contiguous()
        V = V_heads.reshape(BH, T, self.dv)

        # Cast gate params to input dtype (handles .double() / .half())
        A_log_cast = self.A_log.to(dtype)  # [H]
        dt_bias_cast = self.dt_bias.to(dtype)  # [H*K]
        alpha_step, g_flat, beta = self._compute_gate_params(x, BH, T, A_log_cast, dt_bias_cast)
        # alpha_step  [BH, T, Dk]  per-step decay in (0, 1]
        # g_flat      [BH, T, Dk]  per-step log-decay (for chunk WY)
        # beta        [BH, T, 1]

        # State
        S = (
            state.reshape(BH, self.dk, self.dv).contiguous()
            if state is not None
            else self._init_state(BH, device, dtype)
        )

        out = torch.zeros(BH, T, self.dv, device=device, dtype=dtype)

        C = self.chunk_size if self.chunk_size and self.chunk_size > 1 else 0
        S_final = None

        g_heads = g_flat.reshape(B, self.h, T, self.dk)
        beta_heads = beta.reshape(B, self.h, T)
        S_heads = S.reshape(B, self.h, self.dk, self.dv)
        if C and can_use_fla_kda(
            Q, K, V_heads, g_heads, beta_heads, S_heads, C, recurrent=False
        ):
            out_h, S_h = fla_kda_forward(
                Q,
                K,
                V_heads,
                g_heads,
                beta_heads,
                S_heads,
                scale=1.0,
                chunk_size=C,
                recurrent=False,
            )
            out = out_h.reshape(BH, T, self.dv).to(dtype=dtype)
            S_final = S_h.reshape(BH, self.dk, self.dv).to(dtype=dtype)
        elif (not C) and (not torch.is_grad_enabled()) and can_use_fla_kda(
            Q, K, V_heads, g_heads, beta_heads, S_heads, 0, recurrent=True
        ):
            out_h, S_h = fla_kda_forward(
                Q,
                K,
                V_heads,
                g_heads,
                beta_heads,
                S_heads,
                scale=1.0,
                chunk_size=0,
                recurrent=True,
            )
            out = out_h.reshape(BH, T, self.dv).to(dtype=dtype)
            S_final = S_h.reshape(BH, self.dk, self.dv).to(dtype=dtype)
        elif C:
            S = _kda_chunk_parallel(S, Q, K, V, alpha_step, beta, out, C)
            S_final = S
        else:
            # Exact sequential mode: use per-step decay
            S = S.detach()
            # Flatten Q, K, V from [B, H, T, D*] to [BH, T, D*]
            Q_flat = Q.reshape(BH, T, self.dk)
            K_flat = K.reshape(BH, T, self.dk)
            V_flat = V.reshape(BH, T, self.dv)
            for t in range(T):
                o_t, S = _kda_seq_step(
                    S,
                    K_flat[:, t, :],
                    V_flat[:, t, :],
                    Q_flat[:, t, :],
                    alpha_step[:, t, :],
                    beta[:, t, :],
                )
                out[:, t, :] = o_t
            S_final = S

        # ── Output head ──
        output_heads = out.reshape(B, self.h, T, self.dv)

        # Output gate: RMSNorm × sigmoid(g_proj(x))
        g_out = self.g_proj(x)  # [B, T, H*Dv]
        g_out = g_out.view(B, T, self.h, self.dv).permute(0, 2, 1, 3)  # [B, H, T, Dv]
        gate = torch.sigmoid(g_out)

        output_heads = self.h_rms(output_heads) * gate
        y = (
            output_heads
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B, T, self.h * self.dv)
        )
        y = self.drop(self.o_proj(y))

        next_state = S_final.reshape(B, self.h, self.dk, self.dv)
        return y, next_state


# ─────────────────────────────────────────────────────────────────────────────
# Public adapter — drop-in replacement for MultiAttention / LinearAttention
# ─────────────────────────────────────────────────────────────────────────────
class KimiAttention(nn.Module):
    """
    Public wrapper around _KDA_Fast with the same call signature as
    MultiAttention / LinearAttention.

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
        d_key: int | None = None,
        d_val: int | None = None,
        expand_v: float = 1.0,
        shortconv_mode: str = "depthwise",
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        conv_kernel: int = 4,
        safe_updates: bool = True,
        alpha_min: float = 0.1,
        use_triton: bool = False,
        pos_encoding_type: str = "sinusoidal",
    ):
        super().__init__()
        if cross_attention:
            raise ValueError(
                "KimiAttention is self-attention only. "
                "Pass cross_attention=False (default)."
            )
        self.pos_encoding_type = pos_encoding_type
        self._rotary_emb: Optional[nn.Module] = None

        self.kda = _KDA_Fast(
            d_model=d_model,
            num_heads=n_heads,
            d_k=d_key,
            d_v=d_val,
            expand_v=expand_v,
            dropout=dropout,
            shortconv_mode=shortconv_mode,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
            safe_updates=safe_updates,
            alpha_min=alpha_min,
            use_triton=use_triton,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,  # ignored (recurrent)
        key_padding_mask: torch.Tensor | None = None,  # ignored (recurrent)
        is_causal: bool = True,  # always causal
        layer_state: dict | None = None,
    ) -> Tuple[torch.Tensor, None, dict | None]:
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

        # Apply RoPE to query if enabled (KimiAttention is recurrent, ALiBi not applicable)
        # NOTE: RoPE for recurrent models requires special handling.
        # For now, this is a placeholder — KimiAttention handles positional
        # information internally via its delta-gating mechanism.
        if self.pos_encoding_type == "rope" and self._rotary_emb is None:
            B, T, D = query.shape
            from foreblocks.layers.embeddings.rope_alibi_helpers import (
                create_rotary_embedding,
            )

            self._rotary_emb = create_rotary_embedding(
                head_dim=D // self.kda.h, max_seq_len=T
            )

        # Self-attention: pass x as query (key/value unused, passed for compat)
        out, S_next = self.kda(query, state=S)
        return out, None, {"S": S_next}
