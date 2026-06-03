"""
gated_delta.py - Gated Delta Network Attention

Based on: "Gated Delta Networks: Improving Mamba2 with Delta Rule" (Yang et al., 2024)
https://arxiv.org/abs/2412.06464

Architecture
------------
The Gated Delta Network extends DeltaNet with:
 1. **Per-head matrix state** S in R^{Dk x Dv} maintained by the delta rule.
 2. **Scalar forget gate** a_t in (0, 1] - element-wise decay of each row of S.
 3. **Write strength** b_t in [0, 1] - controls how much the new (k,v) pair is written.
 4. **Sigmoid output gate g_t** - data-dependent gating of retrieved memories.
 5. **Head-wise RMSNorm** before the output gate (stabilises large S values).
 6. **Optional short (causal) depthwise conv** on Q, K, V for local context.

State update (per-head):
    err_t  = v_t - S_{t-1} @ k_t                      (prediction error)
    S_t    = α_t · S_{t-1} + β_t · outer(err_t, k_t)  (delta rule + decay)

Output (per-head):
    o_t = g_t ⊙ RMSNorm(S_t @ q_t)

Three forward modes
-------------------
 * **Sequential** (default, exact, training) - step-by-step O(T·Dk·Dv) per head.
 * **Chunk-parallel** (optional, approximate intra-chunk) - reduce Python loop
   overhead; intra-chunk uses pre-S state (standard chunk-mode approximation).
   Exact sequential state update maintains correct S across chunks.
 * **Incremental** (layer_state dict) - single-step decoding for KV-cached
   generation. Reads/writes "gdn_state" from/to the provided dict.

Interface matches LinearAttention / KimiAttention:
    forward(query, key, value, attn_mask, key_padding_mask, is_causal, layer_state)
    → Tuple[Tensor, None, None]

For standalone use (internal KimiAttention-style) pass `x` directly:
    _forward_standalone(x, state) → (y, next_state)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...kernels import can_use_fused_rmsnorm_sigmoid_gate, fused_rmsnorm_sigmoid_gate


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
    """Per-head RMSNorm - applied to [B, H, T, Dv] or [BH, T, Dv]."""

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, *, Dv]  or  [BH, *, Dv]  - norm over last dim
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = x * rms
        # broadcast weight: handle both (B,H,T,Dv) and (BH,T,Dv)
        w = self.weight.view(*self.weight.shape)  # (H, Dv)
        if x.dim() == 4:
            # (B, H, T, Dv) - unsqueeze for broadcast
            out = out * w.unsqueeze(0).unsqueeze(2)
        else:
            # (BH, T, Dv) - we don't know H here, return as-is (caller reshapes)
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
    beta_max : float
        Maximum value for the write gate β.
    eps : float
        Epsilon for head-wise RMSNorm.
    use_mamba_gate : bool
        Use Mamba2-style gating for α: ``exp(-exp(A_log) * softplus(g + dt))``.
        Otherwise falls back to legacy ``sigmoid(α_proj(x))``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        # Ignored kwargs - keep API-compatible with MultiAttention / LinearAttention
        attention_type: str = "standard",
        freq_modes: int = 16,
        cross_attention: bool = False,
        # GDN-specific
        d_key: int | None = None,
        d_val: int | None = None,
        chunk_size: int = 64,
        use_short_conv: bool = True,
        conv_kernel: int = 4,
        beta_max: float = 1.0,
        eps: float = 1e-6,
        use_mamba_gate: bool = True,
        # Positional encoding options
        pos_encoding_type: str = "sinusoidal",
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.h = n_heads
        self.dk = d_key or (d_model // n_heads)
        self.dv = d_val or (d_model // n_heads)
        self.chunk_size = int(chunk_size) if chunk_size and chunk_size > 1 else 0
        self.beta_max = float(beta_max)
        self.use_mamba_gate = bool(use_mamba_gate)

        # Legacy compatibility: alpha_min is ignored when use_mamba_gate=True
        # (Mamba2 gating is inherently stable via exp(-positive))
        self.pos_encoding_type = pos_encoding_type
        self._rotary_emb: nn.Module | None = None

        # ── Projections ────────────────────────────────────────────────────
        self.q_proj = nn.Linear(d_model, self.h * self.dk, bias=False)
        self.k_proj = nn.Linear(d_model, self.h * self.dk, bias=False)
        self.v_proj = nn.Linear(d_model, self.h * self.dv, bias=False)
        self.o_proj = nn.Linear(self.h * self.dv, d_model, bias=False)

        # ── Gates ──────────────────────────────────────────────────────────
        # α (forget/decay gate): per-head scalar, one value per head per step.
        #   Mamba2-style: α = exp(-exp(A_log) * softplus(gk + dt_bias))
        #     - A_log from uniform(1, 16) [logged], dt_bias ≈ N(0, 1)
        #   Legacy fallback: α = sigmoid(alpha_proj(x))
        self.gk_proj = nn.Linear(d_model, self.h, bias=True)  # raw projection for gating
        if self.use_mamba_gate:
            # Per-head decay parameter: A_log = log(A), A ~ uniform(1, 16)
            #     Matches NVlabs: uniform_(1, 16).log()
            self.A_log = nn.Parameter(torch.empty(self.h, dtype=torch.float32).uniform_(1, 16).log())
            self.A_log._no_weight_decay = True
            # Per-head dt bias: Mamba2 / NVlabs inverse-softplus init.
            #   dt ~ exp(uniform(log dt_min, log dt_max)) clamped to >= dt_init_floor,
            #   dt_bias = dt + log(-expm1(-dt))  (so softplus(dt_bias) == dt at init)
            dt_min, dt_max, dt_floor = 1e-3, 1e-1, 1e-4
            dt = torch.exp(
                torch.rand(self.h, dtype=torch.float32)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp_min(dt_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse of softplus
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True
        else:
            self.alpha_proj = nn.Linear(d_model, self.h, bias=True)  # legacy
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
        # gk_proj: Xaviers uniform for Mamba2 gating input
        nn.init.xavier_uniform_(self.gk_proj.weight, gain=0.5)
        if not self.use_mamba_gate:
            # legacy: α bias for sigmoid - start near 0.9 (slow forgetting)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forget (α) and write (β) gates.

        Returns
        -------
        alpha : [BH, T, 1]  scalar decay per head per step
        beta  : [BH, T, 1]  scalar write strength per head per step
        """
        if self.use_mamba_gate:
            # Mamba2-style gating: α = exp(-exp(A_log) * softplus(gk + dt_bias))
            # g_raw: [B, T, H] → g_log: [B, T, H] (log-space decay, always ≤ 0)
            g_raw = self.gk_proj(x)  # [B, T, H]
            if self.dt_bias is not None:
                g_log = -self.A_log.exp() * F.softplus(g_raw + self.dt_bias)  # [B, T, H]
            else:
                g_log = -self.A_log.exp() * F.softplus(g_raw)  # [B, T, H]
            alpha = g_log.exp().clamp(max=1.0)  # [B, T, H] → (0, 1]
        else:
            # Legacy: α = sigmoid(alpha_proj(x))
            alpha = torch.sigmoid(self.alpha_proj(x)).clamp(max=1.0)  # [B, T, H]
        alpha = (
            alpha.transpose(1, 2)  # [B, H, T]
            .contiguous()
            .reshape(BH, T, 1)  # [BH, T, 1]
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

    # ── Single recurrent step ─────────────────────────────────────────────────

    @staticmethod
    def _delta_step(
        S: torch.Tensor,  # [BH, Dk, Dv]  current state
        k_t: torch.Tensor,  # [BH, Dk]       normalised key
        v_t: torch.Tensor,  # [BH, Dv]       value
        q_t: torch.Tensor,  # [BH, Dk]       normalised query
        alpha_t: torch.Tensor,  # [BH, 1]        scalar decay
        beta_t: torch.Tensor,  # [BH, 1]        write strength
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Delta-rule update + retrieval, per the Gated DeltaNet paper.

        Paper formula:
            S_t = α_t (I - β_t k_t^T k_t) S_{t-1} + β_t k_t^T v_t
        which expands to:
            S_t = α_t S_{t-1} + β_t k_t ⊗ (v_t - α_t k_t S_{t-1})

        Key: the prediction error is computed from the *decayed* state
        α_t S_{t-1}, not the raw S_{t-1}. This matches the NVlabs
        official implementation and the FLA reference kernel.

        Returns (o_t [BH, Dv], S_new [BH, Dk, Dv]).
        """
        # 1 Apply decay: S' = α S
        S_decayed = alpha_t.unsqueeze(-1) * S  # [BH, Dk, Dv]

        # 2 Prediction error from DECAYED state: v_hat = k^T S_decayed
        # einsum "bid,bi->bd" contracts over Dk: S_decayed @ k_t
        v_hat = torch.einsum("bid,bi->bd", S_decayed, k_t)  # [BH, Dv]
        err = v_t - v_hat  # [BH, Dv]  - error of decayed state prediction

        # 3 Outer-product write: k ⊗ (β · err)
        write = torch.einsum("bi,bd->bid", k_t, err)  # [BH, Dk, Dv]

        # 4 State update: S_t = α S + β k ⊗ err
        S_new = S_decayed + beta_t.unsqueeze(-1) * write  # [BH, Dk, Dv]

        # 5 Retrieve from updated state: o_t = q^T S_new
        o_t = torch.einsum("bid,bi->bd", S_new, q_t)  # [BH, Dv]

        return o_t, S_new

    # ── Chunk-parallel forward (exact WY representation, FLA-style) ──────────

    def _chunk_parallel(
        self,
        S: torch.Tensor,  # [BH, Dk, Dv]  incoming state
        Q: torch.Tensor,  # [BH, T, Dk]   normalised query
        K: torch.Tensor,  # [BH, T, Dk]   normalised key
        V: torch.Tensor,  # [BH, T, Dv]   value
        alpha: torch.Tensor,  # [BH, T, 1] scalar decay per step
        beta: torch.Tensor,  # [BH, T, 1]  scalar write strength per step
        out: torch.Tensor,  # [BH, T, Dv] output buffer (written in place)
        C: int,
    ) -> torch.Tensor:
        """
        Chunk-parallel gated delta rule via the WY representation.

        Mathematically equivalent (to fp error) to repeated ``_delta_step``.
        Within a chunk of length L starting from state S_0:

            b_t   = prod_{j<=t} alpha_j          (cumulative decay, log-space)
            r_tj  = b_t / b_j   (t>=j, in (0,1])  (bounded relative decay)

            RHS_t = beta_t v_t - beta_t b_t (K_c S_0)_t
            L_tj  = beta_t r_tj (k_j . k_t)   for j<t   (strictly lower)
            U     = (I + L)^{-1} RHS           (triangular solve;  U_t = beta_t err_t)

            S_new = b_L S_0 + sum_j (b_L/b_j) k_j (x) U_j
            o_t   = b_t (Q_c S_0)_t + sum_{j<=t} r_tj (q_t . k_j) U_j

        All decay factors are bounded in (0,1], so no 1/b overflow over long
        chunks. Derivation matches the file's ``_delta_step`` recurrence:
        err computed from the *decayed* state.
        """
        T = Q.shape[1]
        dtype = S.dtype
        # Solve / accumulate in fp32 at minimum (promote fp16/bf16) for stability;
        # keep higher precision (fp64) when the caller supplies it so the chunk
        # path stays bit-for-bit comparable to the sequential recurrence.
        work = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        Qf, Kf, Vf = Q.to(work), K.to(work), V.to(work)
        af, bf = alpha.to(work).squeeze(-1), beta.to(work).squeeze(-1)  # [BH, T]

        S = S.to(work)
        for s in range(0, T, C):
            e = min(s + C, T)
            L = e - s
            Qc, Kc, Vc = Qf[:, s:e], Kf[:, s:e], Vf[:, s:e]
            ac, bc = af[:, s:e], bf[:, s:e]  # [BH, L]

            logb = torch.cumsum(ac.clamp_min(1e-12).log(), dim=1)  # [BH, L]
            b = logb.exp()  # decay start→t inclusive, applied to S_0 path
            # r_tj = b_t / b_j = exp(logb_t - logb_j); bounded (0,1] for t>=j.
            # The upper triangle (t<j) has a positive exponent and would overflow
            # to +inf; we discard it via tril below, but autograd then propagates
            # inf·0 = nan. Mask the exponent to <=0 *before* exp so it stays finite.
            dlog = (logb[:, :, None] - logb[:, None, :]).tril(diagonal=0)  # [BH, L, L]
            ratio = dlog.exp()  # bounded in (0,1] on/below diagonal, finite above

            Kt_S0 = torch.einsum("bik,bkd->bid", Kc, S)  # (K_c S_0)_t : [BH, L, Dv]
            rhs = bc[..., None] * Vc - (bc * b)[..., None] * Kt_S0  # [BH, L, Dv]

            kkt = torch.einsum("bik,bjk->bij", Kc, Kc)  # (k_j . k_t) : [BH, L, L]
            Lmat = torch.tril(bc[:, :, None] * ratio * kkt, diagonal=-1)
            A = torch.eye(L, device=S.device, dtype=S.dtype) + Lmat
            U = torch.linalg.solve_triangular(A, rhs, upper=False)  # [BH, L, Dv]

            # Output uses the *pre-chunk* state S_0 (= current S):
            #   o_t = b_t (Q_c S_0)_t + sum_{j<=t} r_tj (q_t . k_j) U_j
            q_S0 = torch.einsum("bik,bkd->bid", Qc, S)  # [BH, L, Dv]
            qkt = torch.einsum("bik,bjk->bij", Qc, Kc)  # (q_t . k_j) : [BH, L, L]
            coef = torch.tril(ratio * qkt, diagonal=0)  # [BH, L, L]
            out[:, s:e] = (b[..., None] * q_S0 + torch.einsum("bij,bjd->bid", coef, U)).to(dtype)

            # State carry: b_L/b_j bounded in (0,1]
            end_ratio = (logb[:, -1, None] - logb).exp()  # [BH, L]
            Gsum = torch.einsum("bik,bid->bkd", Kc * end_ratio[..., None], U)
            S = b[:, -1, None, None] * S + Gsum  # [BH, Dk, Dv]

        return S.to(dtype)

    # ── Positional encoding helpers ───────────────────────────────────────────

    def _apply_rope_to_x(
        self, x: torch.Tensor, seqlen: int
    ) -> torch.Tensor:
        """Apply rotary position embeddings to input tensor before recurrence.

        NOTE: GatedDeltaNet is a recurrent attention mechanism. RoPE requires
        Q and K projections to be rotated separately. For now, this is a no-op
        placeholder. Full RoPE support requires modifying _forward_recurrent
        to apply RoPE after q_proj and k_proj.

        Args:
            x: Input tensor [B, T, D]
            seqlen: Sequence length

        Returns:
            Input tensor unchanged (RoPE not yet implemented for GDN)
        """
        # TODO: Implement RoPE for GatedDeltaNet
        # RoPE requires rotating Q and K after projection, before recurrence
        return x

    # ── Core recurrent forward (standalone, x-in)  ────────────────────────────

    def _forward_recurrent(
        self,
        x: torch.Tensor,  # [B, T, D]
        state: torch.Tensor | None = None,  # [B, H, Dk, Dv]
        k: torch.Tensor | None = None,  # pre-projected K (Oryx)
        v: torch.Tensor | None = None,  # pre-projected V (Oryx)
        skip_proj: bool = False,  # skip internal Q/K/V projection (Oryx)
        skip_gate_norm: bool = False,  # skip internal gate+norm (Oryx)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full recurrence over T steps.
        Returns (y [B, T, D], next_state [B, H, Dk, Dv]).
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype
        BH = B * self.h

        # Project + (optional) short conv
        if skip_proj:
            # Oryx: use pre-projected K/V from shared projections.
            # Only project Q internally (mixer-specific, no short conv per paper).
            q_raw = self.q_proj(x)  # [B, T, H*Dk]
            k_raw = k  # pre-projected + pre-convolved K: [B, T, H*Dk]
            v_raw = v  # pre-projected + pre-convolved V: [B, T, H*Dv]
        else:
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

        if C and T > C:
            # ── Chunk-parallel mode (exact WY representation) ─────────────
            S = self._chunk_parallel(S, Q, K, V, alpha, beta, out, C)
        else:
            # ── Exact sequential mode ─────────────────────────────────────
            S = S.detach()
            for t in range(T):
                out[:, t], S = self._delta_step(
                    S, K[:, t], V[:, t], Q[:, t], alpha[:, t], beta[:, t]
                )

        # ── Output gate + head-wise RMSNorm (skip for Oryx) ───────────────
        out_h = out.reshape(B, self.h, T, self.dv)  # [B, H, T, Dv]
        if not skip_gate_norm:
            g = self._output_gate(x)  # [B, H, T, Dv]
            if (
                not torch.is_grad_enabled()
                and can_use_fused_rmsnorm_sigmoid_gate(out_h, g, self.h_rms.weight)
            ):
                out_h = fused_rmsnorm_sigmoid_gate(
                    out_h, g, self.h_rms.weight, self.h_rms.eps
                )
            else:
                out_h = self.h_rms(out_h)
                out_h = out_h * g

        y = out_h.permute(0, 2, 1, 3).contiguous().reshape(B, T, self.h * self.dv)
        y = self.drop(self.o_proj(y))

        next_state = S.reshape(B, self.h, self.dk, self.dv)
        return y, next_state

    # ── Single-step incremental forward ───────────────────────────────────────

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

        q_raw = self.q_proj(x_t)  # [B, 1, H*Dk] - no short conv during step
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

        o_t, S_new = self._delta_step(
            S, K[:, 0], V[:, 0], Q[:, 0], alpha[:, 0], beta[:, 0]
        )  # o_t: [BH, Dv]

        # gate & norm (skip for Oryx: external GatedRMSNorm applied)
        out_h = o_t.reshape(B, self.h, 1, self.dv)  # [B, H, 1, Dv]
        if not skip_gate_norm:
            g = self._output_gate(x_t)  # [B, H, 1, Dv]
            if (
                not torch.is_grad_enabled()
                and can_use_fused_rmsnorm_sigmoid_gate(out_h, g, self.h_rms.weight)
            ):
                out_h = fused_rmsnorm_sigmoid_gate(
                    out_h, g, self.h_rms.weight, self.h_rms.eps
                )
            else:
                out_h = self.h_rms(out_h)
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
        * ``attn_mask`` / ``key_padding_mask`` are handled by masking V:
          padded positions are zeroed before the recurrence so they contribute
          nothing to the state update.
        * ``is_causal`` is implicitly satisfied by the causal recurrence.
        * For cross-attention (``key != query``) the module uses ``query``
          as the input token stream and projects K, V from ``key`` / ``value``
          (standard cross-attention approximation for recurrent models).
        * ``layer_state`` (dict): reads/writes ``"gdn_state"`` key for
          single-step incremental decoding.
        * ``skip_cross_attention`` (Oryx): when True and key/value are
          provided, uses pre-projected K/V in recurrent self-attention mode
          instead of the cross-attention approximation.
        * ``skip_gate_norm`` (Oryx): when True, skips internal RMSNorm +
          output gate so external GatedRMSNorm can be applied (per Oryx
          paper: Y = GatedRMSNorm(O, G) W_O).
        """
        B, Tq, D = query.shape
        is_cross = key is not query and key.data_ptr() != query.data_ptr()

        # ── Oryx: use pre-projected K/V in recurrent self-attention mode ──
        if skip_cross_attention and key is not None and value is not None:
            x = query
            if key_padding_mask is not None:
                pad = key_padding_mask.unsqueeze(-1).to(dtype=x.dtype)
                x = x * (1.0 - pad)
            y, next_s = self._forward_recurrent(
                x, state=None, k=key, v=value, skip_proj=True, skip_gate_norm=skip_gate_norm
            )
            if layer_state is not None:
                layer_state["gdn_state"] = next_s
            return y, None, None

        # ── Incremental single-step (decoder KV-cache path) ───────────────
        if layer_state is not None and Tq == 1 and not self.training:
            prev_s = layer_state.get("gdn_state", None)
            y, next_s = self._step(query, prev_s, skip_gate_norm=skip_gate_norm)
            layer_state["gdn_state"] = next_s
            return y, None, None

        # ── Cross-attention: use query stream but project from KV tensors ──
        if is_cross:
            # In the recurrent view we treat the entire encoder memory as a
            # single "batch" retrieved via the query stream.
            # Simplest correct approach: run recurrence on query, but use
            # projected K/V from the encoder memory as key/value vectors at
            # each step - however for T_q != T_k this is ill-defined.
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

        # ── Apply RoPE to Q/K before recurrence ────────────────────────────
        if self.pos_encoding_type == "rope":
            x = self._apply_rope_to_x(x, Tq)

        # ── Incremental multi-step (carry state across calls) ─────────────
        if layer_state is not None:
            prev_s = layer_state.get("gdn_state", None)
            y, next_s = self._forward_recurrent(
                x, prev_s, skip_gate_norm=skip_gate_norm
            )
            layer_state["gdn_state"] = next_s
            return y, None, None

        # ── Standard full-sequence forward ────────────────────────────────
        y, _ = self._forward_recurrent(x, state=None, skip_gate_norm=skip_gate_norm)
        return y, None, None

    # ── Cross-attention approximation ────────────────────────────────────────

    def _cross_attention_approx(
        self,
        query: torch.Tensor,  # [B, Tq, D]
        key: torch.Tensor,  # [B, Tm, D]   (memory)
        value: torch.Tensor,  # [B, Tm, D]
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Approximate cross-attention using the delta rule:
         1. Build a state S from (K, V) memory pairs using the delta rule.
         2. Retrieve from S using query vectors.

        This is O(Tm·Dk·Dv + Tq·Dk·Dv) - linear in both memory and query length.
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
        for t in range(Tm):
            alpha_t = torch.ones(BH, 1, device=device, dtype=dtype)
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
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            f"mamba_gate={self.use_mamba_gate}, beta_max={self.beta_max}"
        )
