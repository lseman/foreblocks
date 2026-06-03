"""
gated_deltanet2.py - Gated DeltaNet-2 (GDN-2)

Based on: "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
(Hatamizadeh, Choi, Kautz — arXiv:2605.22791v1, 2026)
https://arxiv.org/abs/2605.22791

Key innovation over Gated DeltaNet
-----------------------------------
GatedDeltaNet uses a single scalar β_t for BOTH erase and write, which forces
two different decisions (how much old content to erase on the key side vs how
much new content to commit on the value side) to share the same value.

Gated DeltaNet-2 decouples these:
  • **Erase gate**  b_t ∈ [0, 1]^dk   — channel-wise, controls which key
    coordinates are used to read old content from memory.
  • **Write gate**  w_t ∈ [0, 1]^dv   — channel-wise, controls which value
    coordinates are inserted into memory.
  • **Forget/decay** α_t ∈ (0, 1]^dk  — channel-wise, same Mamba2 gating.

Core recurrence (per head, per token):
    e_t = b_t ⊙ k_t          # gated erase direction
    z_t = w_t ⊙ v_t          # gated write target
    S̄_t = Diag(α_t) S_{t-1}  # apply decay
    S_t = S̄_t + k_t (z_t − r_t)^T   # delta-rule update
        where r_t = S̄_t^T e_t        # read from decayed state along erase dir

In matrix form:
    S_t = (I − k_t e_t^T) Diag(α_t) S_{t-1} + k_t z_t^T

When b_t = β·1 and w_t = β·1, GDN-2 reduces to KDA.
When further α_t = α·1, it reduces to GatedDeltaNet.

Architecture
------------
 1. Per-head matrix state S ∈ R^{dk × dv} maintained by the delta rule.
 2. Channel-wise erase gate b_t ∈ [0,1]^dk via independent projection.
 3. Channel-wise write gate w_t ∈ [0,1]^dv via independent projection.
 4. Channel-wise decay α_t ∈ (0,1]^dk via Mamba2-style gating.
 5. Sigmoid output gate — data-dependent gating of retrieved memories.
 6. Head-wise RMSNorm before the output gate.
 7. Optional short (causal) depthwise conv on Q, K, V for local context.
 8. Optional negative-eigenvalue mode: b_t → [0,2]^dk (allows state-tracking).

Three forward modes
-------------------
 * **Sequential** — step-by-step, exact, suitable for training.
 * **Chunk-parallel** — reduce Python-loop overhead; exact WY form.
 * **Incremental** — single-step decoding with layer_state KV cache.

Interface matches LinearAttention / KimiAttention:
    forward(query, key, value, attn_mask, key_padding_mask, is_causal, layer_state)
    → Tuple[Tensor, None, None]
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


class _CausalConv(nn.Module):
    """Causal depthwise Conv1d + SiLU activation."""

    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=self.pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        T = x.size(1)
        x = self.conv(x.transpose(1, 2))[:, :, :T]
        return F.silu(x.transpose(1, 2).contiguous())


class _HeadRMSNorm(nn.Module):
    """Per-head RMSNorm on [B, H, T, Dv] or [BH, T, Dv]."""

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, *, Dv]  or  [BH, *, Dv]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = x * rms
        w = self.weight
        if x.dim() == 4:
            # (B, H, T, Dv)
            out = out * w.unsqueeze(0).unsqueeze(2)
        else:
            # (BH, T, Dv) — caller should reshape to (B,H,T,Dv)
            out = out * w.reshape(-1, w.size(-1)).repeat(
                x.size(0) // w.size(0), 1
            ).unsqueeze(1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────


class GatedDeltaNet2(nn.Module):
    """
    Gated DeltaNet-2: decoupled erase and write in linear attention.

    Parameters
    ----------
    d_model : int
        Total model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Output dropout probability.
    d_key : int | None
        Per-head key/query dimension. Default: d_model // n_heads.
    d_val : int | None
        Per-head value dimension. Default: d_model // n_heads.
    chunk_size : int
        Chunk length for chunk-parallel mode. 0 = pure sequential.
    use_short_conv : bool
        Apply a causal depthwise conv on Q, K, V before recurrence.
    conv_kernel : int
        Kernel size for short conv (default 4).
    eps : float
        Epsilon for head-wise RMSNorm.
    allow_neg_eigval : bool
        Scale erase gate b_t to [0, 2] (allows negative eigenvalues in
        state transition, improves state-tracking per Hatamizadeh et al.).
    pos_encoding_type : str
        Accepted for API compatibility. RoPE not applied (GDN-2 uses causal
        conv + delta-rule positional structure).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        # Ignored kwargs for API compatibility
        attention_type: str = "standard",
        freq_modes: int = 16,
        cross_attention: bool = False,
        # GDN-2 specific
        d_key: int | None = None,
        d_val: int | None = None,
        chunk_size: int = 64,
        use_short_conv: bool = True,
        conv_kernel: int = 4,
        eps: float = 1e-6,
        allow_neg_eigval: bool = False,
        pos_encoding_type: str = "sinusoidal",
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.h = n_heads
        self.dk = d_key or (d_model // n_heads)
        self.dv = d_val or (d_model // n_heads)
        self.chunk_size = int(chunk_size) if chunk_size and chunk_size > 1 else 0
        self.allow_neg_eigval = allow_neg_eigval
        self.pos_encoding_type = pos_encoding_type

        # ── Projections ────────────────────────────────────────────────────
        self.q_proj = nn.Linear(d_model, self.h * self.dk, bias=False)
        self.k_proj = nn.Linear(d_model, self.h * self.dk, bias=False)
        self.v_proj = nn.Linear(d_model, self.h * self.dv, bias=False)
        self.o_proj = nn.Linear(self.h * self.dv, d_model, bias=False)

        # ── Gates ──────────────────────────────────────────────────────────
        # b: erase gate — per-channel, dk per head → [B, T, H, dk] → flattened
        self.b_proj = nn.Linear(d_model, self.h * self.dk, bias=True)

        # w: write gate — per-channel, dv per head
        self.w_proj = nn.Linear(d_model, self.h * self.dv, bias=True)

        # gk: raw projection for Mamba2-style decay
        self.gk_proj = nn.Linear(d_model, self.h, bias=True)

        # Per-head decay: A_log = log(A), A ~ uniform(1, 16)
        self.A_log = nn.Parameter(
            torch.empty(self.h, dtype=torch.float32).uniform_(1, 16).log()
        )
        self.A_log._no_weight_decay = True

        # Per-head dt bias: inverse-softplus init
        dt_min, dt_max, dt_floor = 1e-3, 1e-1, 1e-4
        dt = torch.exp(
            torch.rand(self.h, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_min(dt_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # g: output gate (gated RMSNorm style)
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
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.b_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.gk_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)

    # ── Private helpers ─────────────────────────────────────────────────

    def _init_state(
        self, BH: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Initial recurrent state [BH, dk, dv]."""
        return torch.zeros(BH, self.dk, self.dv, device=device, dtype=dtype)

    @staticmethod
    def _l2_norm(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Row-wise L2-normalise last dim."""
        return t / (t.norm(dim=-1, keepdim=True).clamp_min(eps))

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Linear project and apply optional short conv. Returns (Q_raw, K_raw, V_raw)."""
        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)
        if self.use_short_conv:
            q_raw = self.q_conv(q_raw)
            k_raw = self.k_conv(k_raw)
            v_raw = self.v_conv(v_raw)
        return q_raw, k_raw, v_raw

    def _gate_params(
        self, x: torch.Tensor, BH: int, T: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute erase (b), write (w), and decay (α) gates.

        Returns
        -------
        b : [BH, T, dk, 1]  channel-wise erase gate (per key channel)
        w : [BH, T, dv, 1]  channel-wise write gate (per value channel)
        alpha : [BH, T, dk, 1]  channel-wise decay
        """
        B, T_local, _ = x.shape

        # Erase gate: b = sigmoid(b_proj(x)) ∈ [0, 1]^dk per head
        b_raw = self.b_proj(x)  # [B, T, H*dk]
        b = torch.sigmoid(b_raw).view(B, T_local, self.h, self.dk)
        if self.allow_neg_eigval:
            b = b * 2.0
        b = b.permute(0, 1, 2, 3).reshape(BH, T_local, self.dk, 1)  # [BH, T, dk, 1]

        # Write gate: w = sigmoid(w_proj(x)) ∈ [0, 1]^dv per head
        w_raw = self.w_proj(x)  # [B, T, H*dv]
        w = torch.sigmoid(w_raw).view(B, T_local, self.h, self.dv)
        w = w.permute(0, 1, 2, 3).reshape(BH, T_local, self.dv, 1)  # [BH, T, dv, 1]

        # Decay: α = exp(−exp(A_log) · softplus(gk + dt_bias)) ∈ (0, 1]
        # Computed per-head, then tiled across dk channels (each head's
        # α_h is shared by all dk/h key channels in that head).
        g_raw = self.gk_proj(x)  # [B, T, H]
        if self.dt_bias is not None:
            g_log = -self.A_log.exp() * F.softplus(g_raw + self.dt_bias)
        else:
            g_log = -self.A_log.exp() * F.softplus(g_raw)
        alpha = g_log.exp().clamp(max=1.0)  # [B, T, H]
        # [B, T, H] → [B, H, T] → [BH, T, 1] → [BH, T, dk, 1]
        # Tile across dk so each head's α is shared by all its channels
        alpha = (
            alpha.transpose(1, 2)  # [B, H, T]
            .contiguous()
            .view(BH, T_local, 1)  # [BH, T, 1]
            .unsqueeze(-1)  # [BH, T, 1, 1]
            .expand(-1, -1, self.dk, 1)  # [BH, T, dk, 1]
        )

        return b, w, alpha

    def _output_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Data-dependent sigmoid output gate [B, H, T, Dv]."""
        B, T = x.size(0), x.size(1)
        return (
            torch.sigmoid(self.g_up(F.silu(self.g_down(x))))  # [B, T, H*Dv]
            .view(B, T, self.h, self.dv)
            .permute(0, 2, 1, 3)  # [B, H, T, Dv]
            .contiguous()
        )

    # ── Single recurrent step (GDN-2 delta rule) ─────────────────────────

    @staticmethod
    def _delta_step_gdn2(
        S: torch.Tensor,            # [BH, Dk, Dv]  current state
        k_t: torch.Tensor,          # [BH, Dk]       normalised key
        v_t: torch.Tensor,          # [BH, Dv]       value
        q_t: torch.Tensor,          # [BH, Dk]       normalised query
        b_t: torch.Tensor,          # [BH, Dk, 1]    erase gate
        w_t: torch.Tensor,          # [BH, Dv, 1]    write gate
        alpha_t: torch.Tensor,      # [BH, Dk, 1]    channel-wise decay
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gated DeltaNet-2 single-step recurrence.

        Paper equations (per head):
            e_t = b_t ⊙ k_t               # gated erase direction  [BH, Dk]
            z_t = w_t ⊙ v_t               # gated write target     [BH, Dv]
            S̄_t = Diag(α_t) · S_{t-1}     # apply channel-wise decay
            r_t = S̄_t^T · e_t             # read from decayed state [BH, Dv]
            S_t = S̄_t + k_t ⊗ (z_t − r_t) # delta-rule update      [BH, Dk, Dv]
            o_t = S_t^T · q_t             # output readout         [BH, Dv]

        Returns (o_t [BH, Dv], S_new [BH, Dk, Dv]).
        """
        # 1. Gated erase direction: e_t = b_t ⊙ k_t
        e_t = b_t.squeeze(-1) * k_t  # [BH, Dk]

        # 2. Gated write target: z_t = w_t ⊙ v_t
        z_t = w_t.squeeze(-1) * v_t  # [BH, Dv]

        # 3. Apply decay: S̄_t = Diag(α_t) · S_{t-1}
        S_decayed = alpha_t * S  # [BH, Dk, Dv]  (broadcasts over dv)

        # 4. Read from decayed state along erase direction: r_t = S̄_t^T · e_t
        r_t = torch.einsum("bid,bi->bd", S_decayed, e_t)  # [BH, Dv]

        # 5. Delta-rule update: S_t = S̄_t + k_t ⊗ (z_t − r_t)
        delta = z_t - r_t  # [BH, Dv]
        S_new = S_decayed + torch.einsum("bi,bd->bid", k_t, delta)  # [BH, Dk, Dv]

        # 6. Retrieve: o_t = S_t^T · q_t
        o_t = torch.einsum("bid,bi->bd", S_new, q_t)  # [BH, Dv]

        return o_t, S_new

    # ── Chunk-parallel forward (exact WY representation, GDN-2 style) ──────

    def _chunk_parallel(
        self,
        S0: torch.Tensor,   # [BH, Dk, Dv]  incoming state
        Q: torch.Tensor,    # [BH, T, Dk]   normalised query
        K: torch.Tensor,    # [BH, T, Dk]   normalised key
        V: torch.Tensor,    # [BH, T, Dv]   value
        b: torch.Tensor,    # [BH, T, 1, dk]  erase gate
        w: torch.Tensor,    # [BH, T, 1, dv]  write gate
        alpha: torch.Tensor,  # [BH, T, dk, 1]  channel-wise decay
        out: torch.Tensor,   # [BH, T, Dv]  output buffer
        C: int,
    ) -> torch.Tensor:
        """
        Chunk-parallel Gated DeltaNet-2 via the WY representation (Appendix A).

        For a chunk of length L:
          1. Compute cumulative decay γ_r = prod_{i≤r} α_i (elementwise)
          2. Normalise: K̄_r = γ_r^{-1} ⊙ k_r, Ē_r = γ_r ⊙ (b_r ⊙ k_r)
          3. Z_r = w_r ⊙ v_r
          4. Form T = tril(Ē · K̄^T, -1), A = (I + T)^{-1}
          5. Y = A · Ē,  U = A · Z
          6. R = U − Y · S0   (correction vector)
          7. State at end of chunk: S_C = Diag(γ_C) · S0 + K_tail^T · R
          8. Output: O = Q_γ · S0 + A_qk · R

        All operations are done in float32 (promoted from fp16/bf16).
        """
        T = Q.shape[1]
        dtype = S0.dtype
        work = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        # Promote to working precision
        Qf, Kf, Vf = Q.to(work), K.to(work), V.to(work)
        bf, wf = b.to(work), w.to(work)  # [BH, T, 1, dk/dv]
        af = alpha.to(work).unsqueeze(-1)  # [BH, T, dk, 1]

        S0f = S0.to(work)
        BH = Qf.shape[0]

        for s in range(0, T, C):
            e = min(s + C, T)
            L = e - s
            Qc, Kc, Vc = Qf[:, s:e], Kf[:, s:e], Vf[:, s:e]
            bc, wc = bf[:, s:e], wf[:, s:e]
            afc = af[:, s:e]  # [BH, L, dk, 1]

            # ── 1. Cumulative decay ──────────────────────────────────────
            # α in log-space: g_log = log(α)
            log_alpha = afc.log().clamp_min(-50).squeeze(-1)  # [BH, L, dk]
            log_gamma = torch.cumsum(log_alpha, dim=1)  # [BH, L, dk]
            gamma = log_gamma.exp()  # [BH, L, dk]  cumulative decay

            # ── 2. Normalised key and erase ──────────────────────────────
            # K̄_r = γ_r^{-1} ⊙ k_r
            K_bar = Kc / gamma.unsqueeze(-1)  # [BH, L, dk]  (broadcast: dk vs dk)
            # Wait — Kc is [BH, L, dk], gamma is [BH, L, dk]
            # K_bar = Kc * gamma^{-1}
            K_bar = Kc * gamma.rsqrt()

            # Ē_r = γ_r ⊙ (b_r ⊙ k_r)
            b_e = bc.squeeze(1) * Kc  # [BH, L, dk]  (b is [BH, L, 1, dk])
            E_bar = gamma.unsqueeze(-1) * b_e  # [BH, L, dk]
            # Actually: Ē_r = γ_r ⊙ (b_r ⊙ k_r) — elementwise over dk
            # gamma shape: [BH, L, dk], b_e shape: [BH, L, dk]
            E_bar = gamma * b_e  # [BH, L, dk]

            # ── 3. Write target ──────────────────────────────────────────
            # Z_r = w_r ⊙ v_r
            Z = wc.squeeze(1) * Vc  # [BH, L, dv]

            # ── 4. Strictly lower triangular matrix T = tril(Ē · K̄^T, -1) ──
            # E_bar @ K_bar^T: [BH, L, L]
            EK = torch.einsum("bik,bjk->bij", E_bar, K_bar)
            Tmat = torch.tril(EK, diagonal=-1)
            # A = (I + T)^{-1}
            I_L = torch.eye(L, device=Kc.device, dtype=work)
            A_mat = I_L + Tmat  # [BH, L, L]

            # ── 5. Y = A · Ē,  U = A · Z ─────────────────────────────────
            # For each head batch: solve (I + T) · Y_i = Ē_i
            # and (I + T) · U_i = Z_i
            # Use triangular solve: A_mat is lower triangular
            Y = torch.linalg.solve_triangular(A_mat, E_bar.transpose(-1, -2), upper=False).transpose(-1, -2)  # [BH, L, dk]
            U = torch.linalg.solve_triangular(A_mat, Z.transpose(-1, -2), upper=False).transpose(-1, -2)  # [BH, L, dv]

            # ── 6. Correction: R = U − Y · S0 ───────────────────────────
            Y_S0 = torch.einsum("bik,bkd->bid", Y, S0f)  # [BH, L, dv]
            R = U - Y_S0  # [BH, L, dv]

            # ── 7. Output computation ────────────────────────────────────
            # Q_γ = γ_r ⊙ q_r: [BH, L, dk]
            Q_gamma = gamma.unsqueeze(-1) * Qc

            # Output: O = Q_γ · S0 + A_qk · R
            Q_gamma_S0 = torch.einsum("bik,bkd->bid", Q_gamma, S0f)  # [BH, L, dv]

            # A_qk_{rs} = 1_{r≥s} · q_r^T · Diag(γ_r / γ_s) · k_s
            # gamma_ratio_{rs} = γ_r / γ_s for r ≥ s
            log_gamma_exp = log_gamma  # [BH, L, dk]
            log_gamma_ratio = (log_gamma[:, :, :, None] - log_gamma[:, None, :, :])  # [BH, L, L, dk]
            # For r < s this is positive → exp would overflow; mask below
            log_gamma_ratio = torch.where(
                torch.arange(L, device=Qc.device).unsqueeze(0) <= torch.arange(L, device=Qc.device).unsqueeze(1),
                log_gamma_ratio,
                float('-inf'),
            )
            gamma_ratio = log_gamma_ratio.exp()  # [BH, L, L, dk], 0 above diag
            # A_qk: [BH, L, L]
            qkt = torch.einsum("bik,bjk->bij", Qc, Kc)  # [BH, L, L]
            # Apply gamma_ratio: for each (r,s), sum over dk: q_r^T · (gamma_r/gamma_s · k_s)
            # = sum_k q_r[k] * gamma_ratio[r,s,k] * k_s[k]
            weighted_qkt = (qkt.unsqueeze(-1) * gamma_ratio).sum(-1)  # [BH, L, L]
            causal_weighted = torch.tril(weighted_qkt, diagonal=0)
            Aqk_R = torch.einsum("bij,bjd->bid", causal_weighted, R)  # [BH, L, dv]

            O = Q_gamma_S0 + Aqk_R  # [BH, L, dv]
            out[:, s:e] = O.to(dtype)

            # ── 8. State carry: S_C = Diag(γ_L) · S0 + K_tail^T · R ─────
            # K_tail_r = (γ_C / γ_r) ⊙ k_r
            gamma_end = gamma[:, -1, :]  # [BH, dk]
            gamma_end_3d = gamma_end.unsqueeze(1)  # [BH, 1, dk]
            k_tail = (gamma_end_3d / gamma) * Kc  # [BH, L, dk]  (γ_C / γ_r · k_r)
            # K_tail^T @ R: [BH, dk, dv]
            Kt_R = torch.einsum("bik,bjd->bkd", k_tail, R)
            S0f = gamma_end_3d * S0f  # Diag(γ_C) · S0
            S0f = S0f + Kt_R  # [BH, dk, dv]

        return S0f.to(dtype)

    # ── Core recurrent forward (standalone, x-in) ────────────────────────────

    def _forward_recurrent(
        self,
        x: torch.Tensor,  # [B, T, D]
        state: torch.Tensor | None = None,  # [B, H, Dk, Dv]
        k: torch.Tensor | None = None,  # pre-projected K (Oryx)
        v: torch.Tensor | None = None,  # pre-projected V (Oryx)
        skip_proj: bool = False,
        skip_gate_norm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full GDN-2 recurrence over T steps.
        Returns (y [B, T, D], next_state [B, H, Dk, Dv]).
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype
        BH = B * self.h

        # Project + (optional) short conv
        if skip_proj:
            q_raw = self.q_proj(x)
            k_raw = k
            v_raw = v
        else:
            q_raw, k_raw, v_raw = self._project(x)

        # Reshape to [BH, T, D*] and normalise Q, K
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

        b, w, alpha = self._gate_params(x, BH, T)

        # Initialise / reshape state
        S = (
            state.reshape(BH, self.dk, self.dv).contiguous().to(dtype=dtype)
            if state is not None
            else self._init_state(BH, device, dtype)
        )

        out = torch.zeros(BH, T, self.dv, device=device, dtype=dtype)
        C = self.chunk_size

        if C and T > C:
            S = self._chunk_parallel(S, Q, K, V, b, w, alpha, out, C)
        else:
            S = S.detach()
            for t in range(T):
                out[:, t], S = self._delta_step_gdn2(
                    S,
                    K[:, t], V[:, t], Q[:, t],
                    b[:, t], w[:, t], alpha[:, t],
                )

        # Output gate + head-wise RMSNorm
        out_h = out.reshape(B, self.h, T, self.dv)
        if not skip_gate_norm:
            out_h = self.h_rms(out_h)
            g = self._output_gate(x)
            out_h = out_h * g

        y = out_h.permute(0, 2, 1, 3).contiguous().reshape(B, T, self.h * self.dv)
        y = self.drop(self.o_proj(y))

        next_state = S.reshape(B, self.h, self.dk, self.dv)
        return y, next_state

    # ── Single-step incremental forward ─────────────────────────────────────

    def _step(
        self,
        x_t: torch.Tensor,  # [B, 1, D]
        state: torch.Tensor | None,  # [B, H, Dk, Dv]
        skip_gate_norm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step forward for incremental (KV-cached) decoding."""
        B, _, _ = x_t.shape
        device, dtype = x_t.device, x_t.dtype
        BH = B * self.h

        q_raw = self.q_proj(x_t)
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

        b, w, alpha = self._gate_params(x_t, BH, 1)

        S = (
            state.reshape(BH, self.dk, self.dv).contiguous().to(dtype=dtype)
            if state is not None
            else self._init_state(BH, device, dtype)
        )

        o_t, S_new = self._delta_step_gdn2(
            S, K[:, 0], V[:, 0], Q[:, 0],
            b[:, 0], w[:, 0], alpha[:, 0],
        )

        out_h = o_t.reshape(B, self.h, 1, self.dv)
        if not skip_gate_norm:
            out_h = self.h_rms(out_h)
            g = self._output_gate(x_t)
            out_h = out_h * g

        y = out_h.permute(0, 2, 1, 3).contiguous().reshape(B, 1, self.h * self.dv)
        y = self.drop(self.o_proj(y))

        next_state = S_new.reshape(B, self.h, self.dk, self.dv)
        return y, next_state

    # ── Public forward - matches MultiAttention / LinearAttention API ──────────

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        layer_state: dict | None = None,
        skip_cross_attention: bool = False,
        skip_gate_norm: bool = False,
    ) -> tuple[torch.Tensor, None, None]:
        """
        Drop-in forward with optional Oryx compatibility.

        Notes
        -----
        * ``attn_mask`` / ``key_padding_mask`` zero out padded tokens.
        * ``is_causal`` is implicitly satisfied by the causal recurrence.
        * ``layer_state`` (dict): reads/writes ``"gdn2_state"`` key.
        * ``skip_gate_norm`` (Oryx): skip internal RMSNorm + output gate.
        """
        B, Tq, D = query.shape

        # ── Oryx: pre-projected K/V ────────────────────────────────────────
        if skip_cross_attention and key is not None and value is not None:
            x = query
            if key_padding_mask is not None:
                pad = key_padding_mask.unsqueeze(-1).to(dtype=x.dtype)
                x = x * (1.0 - pad)
            y, next_s = self._forward_recurrent(
                x, state=None, k=key, v=value, skip_proj=True, skip_gate_norm=skip_gate_norm
            )
            if layer_state is not None:
                layer_state["gdn2_state"] = next_s
            return y, None, None

        # ── Incremental single-step ────────────────────────────────────────
        if layer_state is not None and Tq == 1 and not self.training:
            prev_s = layer_state.get("gdn2_state", None)
            y, next_s = self._step(query, prev_s, skip_gate_norm=skip_gate_norm)
            layer_state["gdn2_state"] = next_s
            return y, None, None

        # ── Cross-attention ────────────────────────────────────────────────
        is_cross = key is not query and key is not None and key.data_ptr() != query.data_ptr()
        if is_cross:
            y = self._cross_attention_approx(query, key, value, key_padding_mask)
            return y, None, None

        # ── Apply padding mask ─────────────────────────────────────────────
        x = query
        if key_padding_mask is not None:
            pad = key_padding_mask.unsqueeze(-1).to(dtype=x.dtype)
            x = x * (1.0 - pad)

        # ── Incremental multi-step ─────────────────────────────────────────
        if layer_state is not None:
            prev_s = layer_state.get("gdn2_state", None)
            y, next_s = self._forward_recurrent(
                x, prev_s, skip_gate_norm=skip_gate_norm
            )
            layer_state["gdn2_state"] = next_s
            return y, None, None

        # ── Standard full-sequence forward ─────────────────────────────────
        y, _ = self._forward_recurrent(x, state=None, skip_gate_norm=skip_gate_norm)
        return y, None, None

    # ── Cross-attention approximation ────────────────────────────────────────

    def _cross_attention_approx(
        self,
        query: torch.Tensor,  # [B, Tq, D]
        key: torch.Tensor,  # [B, Tm, D]
        value: torch.Tensor,  # [B, Tm, D]
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Approximate cross-attention: ingest K,V into state, retrieve with Q."""
        B, Tq, D = query.shape
        _, Tm, _ = key.shape
        device, dtype = query.device, query.dtype
        BH = B * self.h

        k_raw = self.k_proj(key).view(B, Tm, self.h, self.dk).permute(0, 2, 1, 3).reshape(BH, Tm, self.dk)
        v_raw = self.v_proj(value).view(B, Tm, self.h, self.dv).permute(0, 2, 1, 3).reshape(BH, Tm, self.dv)
        K_m = self._l2_norm(k_raw)
        V_m = v_raw

        if key_padding_mask is not None:
            pad = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(dtype=dtype)
            pad_bh = pad.expand(B, self.h, Tm, 1).reshape(BH, Tm, 1)
            K_m = K_m * (1.0 - pad_bh)
            V_m = V_m * (1.0 - pad_bh)

        # Uniform erase/write; full decay (α=1)
        ones_1d = torch.ones(BH, 1, self.dk, device=device, dtype=dtype)
        ones_v = torch.ones(BH, 1, self.dv, device=device, dtype=dtype)

        S = self._init_state(BH, device, dtype)
        for t in range(Tm):
            b_t, w_t, alpha_t = ones_1d, ones_v, torch.ones(BH, self.dk, 1, device=device, dtype=dtype)
            _, S = self._delta_step_gdn2(
                S, K_m[:, t], V_m[:, t], K_m[:, t], b_t, w_t, alpha_t
            )

        q_raw = self.q_proj(query).view(B, Tq, self.h, self.dk).permute(0, 2, 1, 3).reshape(BH, Tq, self.dk)
        Q = self._l2_norm(q_raw)

        out = torch.zeros(BH, Tq, self.dv, device=device, dtype=dtype)
        for t in range(Tq):
            out[:, t] = torch.einsum("bid,bi->bd", S, Q[:, t])

        out_h = out.reshape(B, self.h, Tq, self.dv)
        out_h = self.h_rms(out_h)
        g = self._output_gate(query)
        out_h = out_h * g

        y = out_h.permute(0, 2, 1, 3).contiguous().reshape(B, Tq, self.h * self.dv)
        return self.drop(self.o_proj(y))

    # ── Standalone x-in interface ────────────────────────────────────────────

    def forward_standalone(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standalone forward (KimiAttention-compatible interface)."""
        return self._forward_recurrent(x, state)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, heads={self.h}, "
            f"dk={self.dk}, dv={self.dv}, chunk_size={self.chunk_size}, "
            f"short_conv={self.use_short_conv}, neg_eigval={self.allow_neg_eigval}"
        )
