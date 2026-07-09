"""foreblocks.ops.mamba.fused_mamba2.

Single-kernel Fused Mamba2 forward: conv + dt + SSM + output projection.

Replaces the standard 4-kernel path (causal_conv1d → dt_prep → SSD scan →
fused_out) with one kernel launch, eliminating the conv_output intermediate
write (~2-4 GB/layer memory bandwidth savings). Entry states are pre-computed
via PyTorch's segment_sum_log. Use when you need maximum Mamba2 throughput
and all tensors are on CUDA.

Core API:
- fused_mamba2_forward: single-kernel Mamba2 forward, requires all CUDA tensors
- _compute_entry_states: helper to pre-compute chunk boundary states

"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    FUSED_MAMBA2_TRITON_AVAILABLE = True
except Exception:
    triton = tl = None
    FUSED_MAMBA2_TRITON_AVAILABLE = False


# ── fused forward Triton kernel ──────────────────────────────────────

if FUSED_MAMBA2_TRITON_AVAILABLE:

    @triton.jit
    def _fused_mamba2_fwd_kernel(
        residual_ptr,  # [B, T, d_inner]
        conv_in_ptr,  # [B, T, conv_dim]  — conv_input (before conv1d)
        conv_w_ptr,  # [K, conv_dim]     — conv weights transposed
        conv_b_ptr,  # [conv_dim]
        dt_ptr,  # [B, T, H]  — pre-computed dt (when HAS_DT)
        dt_w_ptr,  # [H, dt_rank]
        dt_b_ptr,  # [H]
        A_ptr,  # [H]
        D_ptr,  # [H, P]
        norm_w_ptr,  # [d_inner]
        out_w_ptr,  # [d_inner, d_model]
        out_b_ptr,  # [d_model]
        B_ptr,  # [B, T, H, N]
        C_ptr,  # [B, T, H, N]
        entry_ptr,  # [B, nc, H, P, N]
        out_ptr,  # [B, T, d_model]
        u_ptr,  # [B, T, H, P]  — u component
        z_ptr,  # [B, T, d_inner]  — z component (for gating)
        dt_hidden_ptr,  # [B, T, dt_rank]  — dt source (before proj)
        Bsz,
        T,
        H,
        P,
        N,
        nc,
        CHUNK_SIZE,
        d_model,
        d_inner,
        conv_dim,
        dt_rank,
        num_heads,
        norm_eps,
        dt_min,
        dt_max,
        K: tl.constexpr,
        HAS_DT_PROJ: tl.constexpr,
        HAS_DT: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DT: tl.constexpr,
        D_MODEL: tl.constexpr,
    ):
        """One program per (b, t, h). Inner loop: conv + dt + SSM + out."""
        pid = tl.program_id(0)
        h = pid % H
        tmp = pid // H
        t = tmp % T
        b = tmp // T
        nc_idx = t // CHUNK_SIZE

        p_offs = tl.arange(0, BLOCK_P)
        p_mask = p_offs < P
        n_offs = tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        pn_mask = p_mask[:, None] & n_mask[None, :]

        # ── Load entry state for this (b, nc, h) ─────────────────
        entry_base = ((b * nc + nc_idx) * H + h) * P * N
        state = tl.load(
            entry_ptr + entry_base + p_offs[:, None] * N + n_offs[None, :],
            mask=pn_mask,
            other=0.0,
        ).to(tl.float32)

        a_val = tl.load(A_ptr + h).to(tl.float32)
        d_vals = tl.load(D_ptr + h * P + p_offs, mask=p_mask, other=0.0).to(tl.float32)

        # Base pointers for this (b, t)
        dt_hidden_base = dt_hidden_ptr + b * T * dt_rank + t * dt_rank
        B_base = b * T * num_heads * N + t * num_heads * N
        C_base = B_base
        u_base = b * T * H * P + t * H * P + h * P
        z_base = z_ptr + b * T * d_inner + t * d_inner
        res_base = residual_ptr + b * T * d_model + t * d_model
        out_base = out_ptr + b * T * d_model + t * d_model

        for ti in range(CHUNK_SIZE):
            t_abs = nc_idx * CHUNK_SIZE + ti
            active = t_abs < T

            # ── causal conv1d (depthwise, K=4) ───────────────────
            # Load conv_input for current head: [head_dim] matching BLOCK_P
            head_offset = h * P
            conv_in_head_base = (
                conv_in_ptr + b * T * conv_dim + t * conv_dim + head_offset
            )
            conv_in = tl.load(
                conv_in_head_base + tl.arange(0, BLOCK_P), mask=p_mask, other=0.0
            ).to(tl.float32)

            conv_acc = tl.zeros([BLOCK_P], dtype=tl.float32)
            conv_acc += tl.load(conv_b_ptr + p_offs, mask=p_mask, other=0.0).to(
                tl.float32
            )
            for k in range(K):
                t_in = t_abs - (K - 1 - k)
                valid = active & (t_in >= 0)
                shift = t_in
                ci_shifted = tl.load(
                    conv_in_head_base + tl.arange(0, BLOCK_P) - shift,
                    mask=valid & p_mask,
                    other=0.0,
                ).to(tl.float32)
                w = tl.load(
                    conv_w_ptr + k * conv_dim + p_offs, mask=p_mask, other=0.0
                ).to(tl.float32)
                conv_acc += ci_shifted * w

            # ── dt_proj + softplus + clamp ────────────────────────
            dt = tl.zeros([BLOCK_P], dtype=tl.float32)
            if HAS_DT:
                # Pre-computed dt: [B, T, H], load per-head value
                dt = tl.load(
                    dt_ptr + b * T * H + t * H + h
                ).to(tl.float32)
                dt = tl.maximum(dt, dt_min)
                dt = tl.minimum(dt, dt_max)
            elif HAS_DT_PROJ:
                dh = tl.load(
                    dt_hidden_base + tl.arange(0, BLOCK_DT),
                    mask=tl.arange(0, BLOCK_DT) < dt_rank,
                    other=0.0,
                ).to(tl.float32)
                w = tl.load(
                    dt_w_ptr + h * dt_rank + tl.arange(0, BLOCK_DT),
                    mask=tl.arange(0, BLOCK_DT) < dt_rank,
                    other=0.0,
                ).to(tl.float32)
                dt += tl.sum(dh * w)
                dt += tl.load(dt_b_ptr + h).to(tl.float32)
                dt = tl.maximum(dt, dt_min)
                dt = tl.minimum(dt, dt_max)
            else:
                dh = tl.load(
                    dt_hidden_base + tl.arange(0, BLOCK_DT),
                    mask=tl.arange(0, BLOCK_DT) < dt_rank,
                    other=0.0,
                ).to(tl.float32)
                dh += tl.load(dt_b_ptr + h).to(tl.float32)
                dt = tl.where(dh > 20.0, dh, tl.log(1.0 + tl.exp(dh)))
                dt = tl.maximum(dt, dt_min)
                dt = tl.minimum(dt, dt_max)

            # ── SSM recurrence ────────────────────────────────────
            decay = tl.exp(dt * a_val)
            u = tl.load(
                u_ptr + t_abs * H * P + p_offs, mask=active & p_mask, other=0.0
            ).to(tl.float32)
            Bv = tl.load(
                B_ptr + t_abs * num_heads * N + h * N + n_offs,
                mask=active & n_mask,
                other=0.0,
            ).to(tl.float32)
            Cv = tl.load(
                C_ptr + t_abs * num_heads * N + h * N + n_offs,
                mask=active & n_mask,
                other=0.0,
            ).to(tl.float32)

            state = state * decay + dt * u[:, None] * Bv[None, :]

            # ── output ────────────────────────────────────────────
            y = tl.sum(state * Cv[None, :], axis=1) + d_vals * u

            # ── Gated output: y * silu(z) + out_proj + residual ──
            z = tl.load(z_ptr + p_offs, mask=p_mask, other=0.0).to(tl.float32)
            silu_z = z / (1.0 + tl.exp(-z))

            # out_proj (GEMV): y * silu(z) @ out_weight + residual
            w_row = tl.load(
                out_w_ptr + p_offs[:, None] * D_MODEL + tl.arange(0, D_MODEL),
                mask=p_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            o = tl.sum((y * silu_z)[:, None] * w_row, axis=1)

            # Add residual
            res = tl.load(res_base + p_offs, mask=p_mask, other=0.0).to(tl.float32)
            o = o + res

            tl.store(out_base + p_offs, o, mask=p_mask)


# ── helper: compute entry states ─────────────────────────────────────


def _compute_entry_states(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cumsum_dtA and entry_states for fused forward.

    Parameters
    ----------
    u : [B, T, H, P] — SSM u component
    dt : [B, T, H] — pre-computed (projected) dt values
    A : [H] — SSM A parameter
    B : [B, T, H, N] — expanded B tensor
    C : [B, T, H, N] — expanded C tensor
    """
    Bsz, T_orig, H, P = u.shape
    N = B.shape[-1]
    pad = (chunk_size - T_orig % chunk_size) % chunk_size
    if pad > 0:
        # Pad TIME dimension (dim=-3 for 4D, dim=-2 for 3D)
        u = F.pad(u, (0, 0, 0, 0, pad, 0))  # [B, T+pad, H, P]
        dt = F.pad(dt, (0, 0, pad, 0))       # [B, T+pad, H]
        B = F.pad(B, (0, 0, 0, 0, pad, 0))   # [B, T+pad, H, N]
        C = F.pad(C, (0, 0, 0, 0, pad, 0))   # [B, T+pad, H, N]
    T_pad = T_orig + pad
    nc = T_pad // chunk_size

    dtA = dt.view(Bsz, T_pad, H) * A.view(1, 1, H)  # [B, T_pad, H]
    dtA_c = dtA.view(Bsz, nc, chunk_size, H)
    dt_c = dt.view(Bsz, nc, chunk_size, H)
    u_c = u.view(Bsz, nc, chunk_size, H, P)
    B_c = B.view(Bsz, nc, chunk_size, H, N)

    cumsum_dtA = torch.cumsum(dtA_c, dim=2)

    cs_last = cumsum_dtA[:, :, -1:, None, :]
    L_last = torch.exp(cs_last - cumsum_dtA[:, :, None, :, :])
    Ldt_last = L_last.squeeze(2) * dt_c
    LB_last = Ldt_last.unsqueeze(-1) * B_c
    state_end = torch.einsum("bcjhn,bcjhp->bchpn", LB_last, u_c)

    zero_state = torch.zeros(Bsz, 1, H, P, N, device=u.device, dtype=torch.float32)
    state_summaries = torch.cat([zero_state, state_end], dim=1)
    chunk_log_decay = cumsum_dtA[:, :, -1, :].transpose(1, 2)
    chunk_log_decay = F.pad(chunk_log_decay, (1, 0))

    from foreblocks.ops.mamba.ssd import _segment_sum_log

    decay_prefix = torch.exp(_segment_sum_log(chunk_log_decay)).transpose(1, 3)
    boundary_all = (
        decay_prefix[..., None, None] * state_summaries[:, :, None, ...]
    ).sum(dim=1)
    entry_states = boundary_all[:, 0, :-1].contiguous()

    return cumsum_dtA, entry_states


# ── fused forward function ───────────────────────────────────────────


def fused_mamba2_forward(
    residual_inner: torch.Tensor,
    conv_input: torch.Tensor,  # [B, T, conv_dim] — input to conv1d
    conv_weight: torch.Tensor,  # [K, conv_dim] — transposed conv weights
    conv_bias: torch.Tensor | None,
    dt_proj_weight: torch.Tensor | None,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    norm_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    u: torch.Tensor,  # [B, T, H, P]
    z: torch.Tensor,  # [B, T, d_inner]
    dt: torch.Tensor,  # [B, T, H] — pre-computed projected dt
    dt_hidden: torch.Tensor | None = None,  # [B, T, dt_rank] — for fallback mode
    B: torch.Tensor | None = None,
    C: torch.Tensor | None = None,
    chunk_size: int = 256,
    dt_min: float = 1e-4,
    dt_max: float = 1.0,
    norm_eps: float = 1e-5,
    num_heads: int | None = None,
) -> torch.Tensor:
    """Fused Mamba2 forward: conv + dt + SSM + output projection in one kernel.

    All tensors must be CUDA. Falls back to standard path if conditions aren't met.

    Parameters
    ----------
    conv_input : [B, T, conv_dim]  — input to conv1d
    conv_weight : [K, conv_dim]  — depthwise conv weights (transposed for kernel efficiency)
    u : [B, T, H, P]  — SSM u component
    z : [B, T, d_inner]  — gating component
    dt : [B, T, H]  — pre-computed projected dt (from fused_dt or equivalent)
    dt_hidden : [B, T, dt_rank]  — dt source before projection (optional, used in fallback)
    B, C : [B, T, H, N]  — expanded B/C tensors
    """
    Bsz, T, d_inner = residual_inner.shape
    d_model = out_proj_weight.shape[-1]
    H = A.shape[0]
    P = D.shape[1] if D.ndim == 2 else 1
    N = B.shape[-1]
    K = conv_weight.shape[0]
    conv_dim = conv_weight.shape[1]
    num_heads = num_heads or H

    has_dt_proj = dt_proj_weight is not None
    dt_rank = dt_proj_weight.shape[1] if has_dt_proj else dt.shape[-1]

    can_fuse = (
        FUSED_MAMBA2_TRITON_AVAILABLE
        and residual_inner.is_cuda
        and conv_input.is_cuda
        and conv_weight.is_cuda
        and conv_bias is not None
        and conv_bias.is_cuda
        and dt.is_cuda
        and dt_bias.is_cuda
        and A.is_cuda
        and D.is_cuda
        and norm_weight.is_cuda
        and out_proj_weight.is_cuda
        and u.is_cuda
        and z.is_cuda
        and B is not None
        and B.is_cuda
        and C is not None
        and C.is_cuda
    )
    if not can_fuse:
        raise RuntimeError(
            f"fused_mamba2_forward: all inputs must be CUDA or Triton missing "
            f"(dt={dt.is_cuda}, conv_bias={conv_bias is not None}, "
            f"B={B is not None}, C={C is not None})"
        )

    # Narrow types: can_fuse ensures B and C are not None
    assert B is not None, "B must not be None"
    assert C is not None, "C must not be None"

    out = torch.empty(Bsz, T, d_model, device=u.device, dtype=u.dtype)

    cumsum_dtA, entry_states = _compute_entry_states(
        u,
        dt,
        A,
        B,
        C,
        chunk_size,
    )

    block_p = min(max(triton.next_power_of_2(P), 16), 128)
    block_n = max(8, triton.next_power_of_2(N))
    block_dt = min(256, max(dt_rank, 16))
    block_conv = max(triton.next_power_of_2(conv_dim), 16)

    grid = (Bsz * T * H,)
    _fused_mamba2_fwd_kernel[grid](
        residual_inner,
        conv_input,
        conv_weight,
        conv_bias,
        dt,  # pre-computed dt [B, T, H]
        dt_proj_weight,
        dt_bias,
        A,
        D,
        norm_weight,
        out_proj_weight,
        out_proj_bias,
        B,
        C,
        entry_states,
        out,
        u,
        z,
        dt_hidden if dt_hidden is not None else dt,
        Bsz,
        T,
        H,
        P,
        N,
        1,
        chunk_size,
        d_model,
        d_inner,
        conv_dim,
        dt_rank,
        num_heads,
        norm_eps,
        dt_min,
        dt_max,
        K,
        HAS_DT_PROJ=has_dt_proj,
        HAS_DT=True,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
        BLOCK_DT=block_dt,
        D_MODEL=d_model,
    )
    return out
