#!/usr/bin/env python3
"""foreblocks.modules.attention.implementations.linear_att.gated_deltanet2.

Gated DeltaNet-2: decoupled erase and write in linear attention (arXiv:2605.22791).

https://arxiv.org/abs/2605.22791

Decouples the erase and write operations that a single β gate forces in
GatedDeltaNet: per-channel erase gate b_t controls which key coordinates read
old state, per-channel write gate w_t controls which value coordinates are
committed, and per-channel decay α_t applies Mamba2-style forgetting. Three
forward modes — sequential (exact training), chunk-parallel (WY form), and
incremental (single-step decoding).

Core API:
- GatedDeltaNet2: GDN-2 with decoupled erase/write gates and chunk modes

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.modules.attention.implementations.linear_att.gated_common import (
    CausalDepthwiseConv,
    GatedDeltaExecutionMixin,
    HeadRMSNorm,
)
from foreblocks.ops.attention import (
    can_use_fla_gdn2,
    can_use_fla_gdn2_chunk,
    can_use_fused_rmsnorm_sigmoid_gate,
    fla_gdn2_chunk_forward,
    fla_gdn2_forward,
    fused_rmsnorm_sigmoid_gate,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


class _CausalConv(CausalDepthwiseConv):
    pass


class _HeadRMSNorm(HeadRMSNorm):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────


class GatedDeltaNet2(GatedDeltaExecutionMixin, nn.Module):
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
        if n_heads <= 0 or d_model % n_heads:
            raise ValueError("n_heads must be positive and divide d_model")

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


    def _gate_params(
        self, x: torch.Tensor, BH: int, T: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        g_log = (
            g_log.transpose(1, 2)  # [B, H, T]
            .contiguous()
            .view(BH, T_local, 1)  # [BH, T, 1]
            .unsqueeze(-1)  # [BH, T, 1, 1]
            .expand(-1, -1, self.dk, 1)  # [BH, T, dk, 1]
        )

        return b, w, alpha, g_log


    # ── Single recurrent step (GDN-2 delta rule) ─────────────────────────

    @staticmethod
    def _delta_step_gdn2(
        S: torch.Tensor,  # [BH, Dk, Dv]  current state
        k_t: torch.Tensor,  # [BH, Dk]       normalised key
        v_t: torch.Tensor,  # [BH, Dv]       value
        q_t: torch.Tensor,  # [BH, Dk]       normalised query
        b_t: torch.Tensor,  # [BH, Dk, 1]    erase gate
        w_t: torch.Tensor,  # [BH, Dv, 1]    write gate
        alpha_t: torch.Tensor,  # [BH, Dk, 1]    channel-wise decay
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        S0: torch.Tensor,  # [BH, Dk, Dv]  incoming state
        Q: torch.Tensor,  # [BH, T, Dk]   normalised query
        K: torch.Tensor,  # [BH, T, Dk]   normalised key
        V: torch.Tensor,  # [BH, T, Dv]   value
        b: torch.Tensor,  # [BH, T, dk, 1]  erase gate
        w: torch.Tensor,  # [BH, T, dv, 1]  write gate
        alpha: torch.Tensor,  # [BH, T, dk, 1]  channel-wise decay
        out: torch.Tensor,  # [BH, T, Dv]  output buffer
        C: int,
    ) -> torch.Tensor:
        T = Q.shape[1]
        dtype = S0.dtype
        work = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        # Promote to working precision and squeeze gate trailing dims.
        # Gates arrive as [BH, T, dk|dv, 1]; squeeze the trailing singleton.
        Qf, Kf, Vf = Q.to(work), K.to(work), V.to(work)
        bf = b.to(work).squeeze(-1)  # [BH, T, dk]  erase gate
        wf = w.to(work).squeeze(-1)  # [BH, T, dv]  write gate
        af = alpha.to(work).squeeze(-1)  # [BH, T, dk]  channel-wise decay

        S0f = S0.to(work)
        eye_cache = torch.eye(C, device=Qf.device, dtype=work)

        for s in range(0, T, C):
            e = min(s + C, T)
            L = e - s
            Qc, Kc, Vc = Qf[:, s:e], Kf[:, s:e], Vf[:, s:e]  # [BH, L, dk|dv]
            bc, wc = bf[:, s:e], wf[:, s:e]  # [BH, L, dk|dv]
            ac = af[:, s:e]  # [BH, L, dk]

            # ── 1. Cumulative within-chunk decay γ_r = ∏_{i≤r} α_i ─────────
            # log-space cumsum for stability; γ is the decay from chunk start
            # (exclusive of the token's own step is handled by indexing below).
            log_alpha = ac.clamp_min(1e-12).log()  # [BH, L, dk]
            log_gamma = torch.cumsum(log_alpha, dim=1)  # [BH, L, dk]  (inclusive)
            gamma = log_gamma.exp()  # [BH, L, dk]

            # Gated erase/write directions.
            e_dir = bc * Kc  # e_r = b_r ⊙ k_r   [BH, L, dk]
            z_tgt = wc * Vc  # z_r = w_r ⊙ v_r   [BH, L, dv]

            # ── 2. Decay-normalised vectors (Mamba2 / KDA trick) ───────────
            # Pull all decay onto a common frame so cross-token interactions
            # use bounded ratios γ_r/γ_j ∈ (0,1] for r ≥ j.
            #   k̄_j = γ_j^{-1} ⊙ k_j ,  ē_r = γ_r ⊙ e_r ,  q̄_r = γ_r ⊙ q_r
            inv_gamma = (-log_gamma).exp()  # γ^{-1}            [BH, L, dk]
            K_bar = Kc * inv_gamma  # [BH, L, dk]
            E_bar = e_dir * gamma  # [BH, L, dk]
            Q_bar = Qc * gamma  # [BH, L, dk]

            # ── 3. WY transform: A = (I + tril(Ē K̄ᵀ, -1))⁻¹ ───────────────
            # The within-chunk delta interactions (token r erases content
            # written by earlier tokens j<r along direction e_r).
            EK = torch.einsum("bik,bjk->bij", E_bar, K_bar)  # [BH, L, L]
            Tmat = torch.tril(EK, diagonal=-1)
            I_L = eye_cache[:L, :L]
            A_mat = I_L + Tmat  # lower-tri, unit diag

            # Pseudo-write U = A·Z and pseudo-erase Y = A·(Ē·S0-readout coeff).
            #   For each token r: w-update is z_r minus what e_r reads from S0
            #   and from earlier in-chunk writes. Solve the triangular systems.
            # Readout of S0 along the (decayed) erase direction:
            E_S0 = torch.einsum("bik,bkd->bid", E_bar, S0f)  # [BH, L, dv]
            rhs = z_tgt - E_S0  # [BH, L, dv]
            U = torch.linalg.solve_triangular(A_mat, rhs, upper=False)  # [BH, L, dv]

            # ── 4. Output O_r = q_r·S_r ───────────────────────────────────
            # Inter-chunk part: q̄_r · S0  (decayed readout of incoming state)
            O_inter = torch.einsum("bik,bkd->bid", Q_bar, S0f)  # [BH, L, dv]
            # Intra-chunk part: causal attention over this chunk's pseudo-writes,
            #   coeff_{rj} = 1_{r≥j} · (q̄_r · k̄_j)   (bounded decay ratios)
            QK = torch.einsum("bik,bjk->bij", Q_bar, K_bar)  # [BH, L, L]
            QK = torch.tril(QK, diagonal=0)
            O_intra = torch.einsum("bij,bjd->bid", QK, U)  # [BH, L, dv]
            out[:, s:e] = (O_inter + O_intra).to(dtype)

            # ── 5. State carry S_C = Diag(γ_L)·S0 + Σ_j (γ_L/γ_j ⊙ k_j) uⱼᵀ ─
            gamma_end = gamma[:, -1:, :]  # [BH, 1, dk]  γ at chunk end
            K_tail = Kc * (gamma_end * inv_gamma)  # (γ_L/γ_j) ⊙ k_j  [BH, L, dk]
            Kt_U = torch.einsum("bjk,bjd->bkd", K_tail, U)  # [BH, dk, dv]
            S0f = gamma_end.transpose(1, 2) * S0f + Kt_U  # [BH, dk, dv]

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

        b, w, alpha, g_log = self._gate_params(x, BH, T)

        # Initialise / reshape state
        S = (
            state.reshape(BH, self.dk, self.dv).contiguous().to(dtype=dtype)
            if state is not None
            else self._init_state(BH, device, dtype)
        )

        out = torch.zeros(BH, T, self.dv, device=device, dtype=dtype)
        C = self.chunk_size

        if C and T > C:
            Qh = Q.reshape(B, self.h, T, self.dk)
            Kh = K.reshape(B, self.h, T, self.dk)
            Vh = V.reshape(B, self.h, T, self.dv)
            gh = g_log.squeeze(-1).reshape(B, self.h, T, self.dk)
            bh = b.squeeze(-1).reshape(B, self.h, T, self.dk)
            wh = w.squeeze(-1).reshape(B, self.h, T, self.dv)
            Sh = S.reshape(B, self.h, self.dk, self.dv)
            if can_use_fla_gdn2_chunk(Qh, Kh, Vh, gh, bh, wh, Sh, C):
                out_h, S_h = fla_gdn2_chunk_forward(
                    Qh,
                    Kh,
                    Vh,
                    gh,
                    bh,
                    wh,
                    Sh,
                    scale=1.0,
                    chunk_size=C,
                )
                out = out_h.reshape(BH, T, self.dv).to(dtype=dtype)
                S = S_h.reshape(BH, self.dk, self.dv).to(dtype=dtype)
            else:
                S = self._chunk_parallel(S, Q, K, V, b, w, alpha, out, C)
        elif not torch.is_grad_enabled():
            Qh = Q.reshape(B, self.h, T, self.dk)
            Kh = K.reshape(B, self.h, T, self.dk)
            Vh = V.reshape(B, self.h, T, self.dv)
            gh = g_log.squeeze(-1).reshape(B, self.h, T, self.dk)
            bh = b.squeeze(-1).reshape(B, self.h, T, self.dk)
            wh = w.squeeze(-1).reshape(B, self.h, T, self.dv)
            Sh = S.reshape(B, self.h, self.dk, self.dv)
            if can_use_fla_gdn2(Qh, Kh, Vh, gh, bh, wh, Sh, 0, recurrent=True):
                out_h, S_h = fla_gdn2_forward(
                    Qh,
                    Kh,
                    Vh,
                    gh,
                    bh,
                    wh,
                    Sh,
                    scale=1.0,
                    chunk_size=0,
                    recurrent=True,
                )
                out = out_h.reshape(BH, T, self.dv).to(dtype=dtype)
                S = S_h.reshape(BH, self.dk, self.dv).to(dtype=dtype)
            else:
                S = S.detach()
                for t in range(T):
                    out[:, t], S = self._delta_step_gdn2(
                        S,
                        K[:, t],
                        V[:, t],
                        Q[:, t],
                        b[:, t],
                        w[:, t],
                        alpha[:, t],
                    )
        else:
            S = S.detach()
            for t in range(T):
                out[:, t], S = self._delta_step_gdn2(
                    S,
                    K[:, t],
                    V[:, t],
                    Q[:, t],
                    b[:, t],
                    w[:, t],
                    alpha[:, t],
                )

        # Output gate + head-wise RMSNorm
        out_h = out.reshape(B, self.h, T, self.dv)
        if not skip_gate_norm:
            g = self._output_gate(x)
            if not torch.is_grad_enabled() and can_use_fused_rmsnorm_sigmoid_gate(
                out_h, g, self.h_rms.weight
            ):
                out_h = fused_rmsnorm_sigmoid_gate(
                    out_h, torch.sigmoid(g), self.h_rms.weight, self.h_rms.eps
                )
            else:
                out_h = self.h_rms(out_h)
                out_h = out_h * torch.sigmoid(g)

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

        b, w, alpha, _ = self._gate_params(x_t, BH, 1)

        S = (
            state.reshape(BH, self.dk, self.dv).contiguous().to(dtype=dtype)
            if state is not None
            else self._init_state(BH, device, dtype)
        )

        o_t, S_new = self._delta_step_gdn2(
            S,
            K[:, 0],
            V[:, 0],
            Q[:, 0],
            b[:, 0],
            w[:, 0],
            alpha[:, 0],
        )

        out_h = o_t.reshape(B, self.h, 1, self.dv)
        if not skip_gate_norm:
            g = self._output_gate(x_t)
            if not torch.is_grad_enabled() and can_use_fused_rmsnorm_sigmoid_gate(
                out_h, g, self.h_rms.weight
            ):
                out_h = fused_rmsnorm_sigmoid_gate(
                    out_h, torch.sigmoid(g), self.h_rms.weight, self.h_rms.eps
                )
            else:
                out_h = self.h_rms(out_h)
                out_h = out_h * torch.sigmoid(g)

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
    ) -> tuple[torch.Tensor, None, dict | None]:
        B, Tq, D = query.shape

        # ── Oryx: pre-projected K/V ────────────────────────────────────────
        if skip_cross_attention and key is not None and value is not None:
            x = query
            if key_padding_mask is not None:
                pad = key_padding_mask.unsqueeze(-1).to(dtype=x.dtype)
                x = x * (1.0 - pad)
            y, next_s = self._forward_recurrent(
                x,
                state=None,
                k=key,
                v=value,
                skip_proj=True,
                skip_gate_norm=skip_gate_norm,
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
        is_cross = (
            key is not query and key is not None and key.data_ptr() != query.data_ptr()
        )
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
            return y, None, {"gdn2_state": next_s}

        # ── Standard full-sequence forward ─────────────────────────────────
        y, next_s = self._forward_recurrent(
            x, state=None, skip_gate_norm=skip_gate_norm
        )
        return y, None, {"gdn2_state": next_s}

    # ── Cross-attention approximation ────────────────────────────────────────

    def _cross_attention_approx(
        self,
        query: torch.Tensor,  # [B, Tq, D]
        key: torch.Tensor,  # [B, Tm, D]
        value: torch.Tensor,  # [B, Tm, D]
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Tq, D = query.shape
        _, Tm, _ = key.shape
        device, dtype = query.device, query.dtype
        BH = B * self.h

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
            pad = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(dtype=dtype)
            pad_bh = pad.expand(B, self.h, Tm, 1).reshape(BH, Tm, 1)
            K_m = K_m * (1.0 - pad_bh)
            V_m = V_m * (1.0 - pad_bh)

        # Uniform erase/write; full decay (α=1)
        ones_1d = torch.ones(BH, 1, self.dk, device=device, dtype=dtype)
        ones_v = torch.ones(BH, 1, self.dv, device=device, dtype=dtype)

        S = self._init_state(BH, device, dtype)
        for t in range(Tm):
            b_t, w_t, alpha_t = (
                ones_1d,
                ones_v,
                torch.ones(BH, self.dk, 1, device=device, dtype=dtype),
            )
            _, S = self._delta_step_gdn2(
                S, K_m[:, t], V_m[:, t], K_m[:, t], b_t, w_t, alpha_t
            )

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

        out_h = out.reshape(B, self.h, Tq, self.dv)
        out_h = self.h_rms(out_h)
        g = self._output_gate(query)
        out_h = out_h * g

        y = out_h.permute(0, 2, 1, 3).contiguous().reshape(B, Tq, self.h * self.dv)
        return self.drop(self.o_proj(y))

    # ── Standalone x-in interface ────────────────────────────────────────────


    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, heads={self.h}, "
            f"dk={self.dk}, dv={self.dv}, chunk_size={self.chunk_size}, "
            f"short_conv={self.use_short_conv}, neg_eigval={self.allow_neg_eigval}"
        )
