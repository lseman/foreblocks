"""foreblocks.ops.mamba.selective_state_update.

Fused Triton kernel for Mamba single-token inference (state update + projection).

Implements the core inference loop for S6/Mamba models: given a single new token,
update the SSM state (A_t, B_t, C_t via dt-dependent projections) and produce output.
Fuses 5-6 separate PyTorch ops into one kernel launch, reducing memory I/O and
kernel overhead during autoregressive decoding.

Ported from mamba-ssm (Tri Dao, Albert Gu). Handles grouped B/C tensors (for
MoE or multi-head designs), optional z-gating, and dt_bias injection.

Core API:
- selective_state_update: fused state update + output computation for inference
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    SELECTIVE_STATE_UPDATE_TRITON_AVAILABLE = True
except Exception:
    triton = tl = None
    SELECTIVE_STATE_UPDATE_TRITON_AVAILABLE = False

if SELECTIVE_STATE_UPDATE_TRITON_AVAILABLE:

    @triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias"] is not None})
    @triton.heuristics({"HAS_D": lambda args: args["D"] is not None})
    @triton.heuristics({"HAS_Z": lambda args: args["z"] is not None})
    @triton.jit
    def _selective_scan_update_kernel(
        # Pointers to matrices
        state_ptr,
        x_ptr,
        dt_ptr,
        dt_bias_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
        z_ptr,
        out_ptr,
        # Matrix dimensions
        batch,
        nheads,
        dim,
        dstate,
        ngroups,
        # Strides
        stride_state_batch,
        stride_state_head,
        stride_state_dim,
        stride_state_dstate,
        stride_x_batch,
        stride_x_head,
        stride_x_dim,
        stride_dt_batch,
        stride_dt_head,
        stride_dt_dim,
        stride_dt_bias_head,
        stride_A_head,
        stride_B_batch,
        stride_B_group,
        stride_B_dstate,
        stride_C_batch,
        stride_C_group,
        stride_C_dstate,
        stride_D_head,
        stride_D_dim,
        stride_z_batch,
        stride_z_head,
        stride_z_dim,
        stride_out_batch,
        stride_out_head,
        stride_out_dim,
        # Meta-parameters
        DT_SOFTPLUS: tl.constexpr,
        TIE_HDIM: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        HAS_DT_BIAS: tl.constexpr,
        HAS_D: tl.constexpr,
        HAS_Z: tl.constexpr,
        BLOCK_SIZE_DSTATE: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_b = tl.program_id(axis=1)
        pid_h = tl.program_id(axis=2)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
        out_ptrs = out_ptr + offs_m

        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
        x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
        dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
        A_ptr += pid_h * stride_A_head
        B_ptr += pid_b * stride_B_batch + (pid_h // ngroups) * stride_B_group
        C_ptr += pid_b * stride_C_batch + (pid_h // ngroups) * stride_C_group
        if z_ptr is not None:
            z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head

        state = tl.load(
            state_ptr
            + offs_m[:, None] * stride_state_dim
            + tl.arange(0, BLOCK_SIZE_DSTATE)[None, :] * stride_state_dstate,
            mask=(offs_m[:, None] < dim)
            & (tl.arange(0, BLOCK_SIZE_DSTATE)[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        x = tl.load(x_ptr + offs_m, mask=offs_m < dim, other=0.0).to(tl.float32)

        # Load dt and apply softplus + bias
        dt = tl.load(dt_ptr + offs_m, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt_bias = tl.load(dt_bias_ptr + tl.arange(0, 1), mask=True, other=0.0).to(
                tl.float32
            )
            dt += dt_bias
        if DT_SOFTPLUS:
            dt = tl.where(
                dt <= 20.0,
                tl.log(1.0 + tl.exp(-dt)) + dt if dt > 0 else tl.log(1.0 + tl.exp(dt)),
                dt,
            )

        # dA = exp(dt * A) — A is per-head scalar
        A_val = tl.load(A_ptr).to(tl.float32)
        dA = tl.exp(A_val * dt[:, None])  # [BLOCK_SIZE_M, 1]

        # Load B, C once (shared across heads in group)
        B = tl.load(
            B_ptr + tl.arange(0, BLOCK_SIZE_DSTATE),
            mask=tl.arange(0, BLOCK_SIZE_DSTATE) < dstate,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_SIZE_DSTATE]
        C = tl.load(
            C_ptr + tl.arange(0, BLOCK_SIZE_DSTATE),
            mask=tl.arange(0, BLOCK_SIZE_DSTATE) < dstate,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_SIZE_DSTATE]

        # Load D if present
        if HAS_D:
            D = tl.load(D_ptr + offs_m, mask=offs_m < dim, other=0.0).to(tl.float32)

        # Update state: state = state * dA + dt * B * x
        # dB = B[None, :] * dt[:, None]  — but we absorb dt into the multiplication
        dB = B[None, :] * dt[:, None]
        new_state = state * dA + dB * x[:, None]
        tl.store(
            state_ptr
            + offs_m[:, None] * stride_state_dim
            + tl.arange(0, BLOCK_SIZE_DSTATE)[None, :] * stride_state_dstate,
            new_state,
            mask=(offs_m[:, None] < dim)
            & (tl.arange(0, BLOCK_SIZE_DSTATE)[None, :] < dstate),
        )

        # Output: y = sum(state * C) + D * x
        out = tl.sum(new_state * C[None, :], axis=1)  # [BLOCK_SIZE_M]
        if HAS_D:
            out += x * D
        if HAS_Z:
            z = tl.load(z_ptr + offs_m, mask=offs_m < dim, other=0.0).to(tl.float32)
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs, out, mask=offs_m < dim)


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton selective state update for single-token autoregressive decoding.

    Updates the SSM state with a single new token and computes the output.

    Args:
        state: [B, H, P, N] — current SSM state (hidden, head_dim, dstate)
        x: [B, H, P] — current input x (u component reshaped)
        dt: [B, H] — delta per head (already softplus+clamped)
        A: [H] — log decay per head (negative)
        B: [B, G, N] — expanded B (batch, ngroups, dstate)
        C: [B, G, N] — expanded C (batch, ngroups, dstate)
        D: [H, P] or [H] — skip connection (optional)
        z: [B, H, P] — gate for RMSNormGated (optional)
        dt_bias: [H] — bias added to dt (optional)
        dt_softplus: if True, apply softplus to dt (optional)

    Returns:
        (out, new_state): ([B, H, P], [B, H, P, N]) — output and updated state
    """
    if not (
        SELECTIVE_STATE_UPDATE_TRITON_AVAILABLE
        and state.is_cuda
        and x.is_cuda
        and dt.is_cuda
        and A.is_cuda
        and B.is_cuda
        and C.is_cuda
        and (D is None or D.is_cuda)
        and (z is None or z.is_cuda)
        and (dt_bias is None or dt_bias.is_cuda)
    ):
        return _selective_state_update_pytorch(
            state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus
        )

    Bsz = state.shape[0]
    nheads = state.shape[1]
    P = state.shape[2]
    N = state.shape[3]

    # Normalize: always use (B, H, P) shape
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if dt.ndim == 1:
        dt = dt.unsqueeze(1)
    if z is not None and z.ndim == 2:
        z = z.unsqueeze(1)

    batch, nheads, dim = x.shape

    # Tuned block sizes per dstate
    if N <= 16:
        BLOCK_SIZE_M, num_warps = 32, 4
    elif N <= 32:
        BLOCK_SIZE_M, num_warps = 16, 4
    elif N <= 64:
        BLOCK_SIZE_M, num_warps = 8, 4
    elif N <= 128:
        BLOCK_SIZE_M, num_warps = 4, 4
    else:
        BLOCK_SIZE_M, num_warps = 4, 8

    BLOCK_SIZE_DSTATE = max(triton.next_power_of_2(N), 16)

    out = torch.empty_like(x)
    grid = (triton.cdiv(dim, BLOCK_SIZE_M), batch, nheads)

    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            batch,
            nheads,
            dim,
            N,
            nheads // B.shape[1] if B.shape[1] > 1 else nheads,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0) if D is not None else 0,
            D.stride(1) if D is not None else 0,
            z.stride(0) if z is not None else 0,
            z.stride(1) if z is not None else 0,
            z.stride(2) if z is not None else 0,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            DT_SOFTPLUS=dt_softplus,
            TIE_HDIM=(A.stride(-1) == 0),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            HAS_DT_BIAS=dt_bias is not None,
            HAS_D=D is not None,
            HAS_Z=z is not None,
            BLOCK_SIZE_DSTATE=BLOCK_SIZE_DSTATE,
            num_warps=num_warps,
        )

    return out, state


def _selective_state_update_pytorch(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation."""
    import torch.nn.functional as F

    Bsz, nheads, P, N = state.shape
    H = nheads

    dt = dt.float()
    A = A.float()

    if dt_bias is not None:
        dt = dt + dt_bias.view(1, H)
    if dt_softplus:
        dt = F.softplus(dt)

    # dA = exp(dt * A) — [B, H, 1, 1]
    dA = torch.exp(dt.unsqueeze(-1).unsqueeze(-1) * A.view(1, H, 1, 1))

    # Broadcast B, C: [B, G, N] → [B, H, 1, N]
    B_exp = B.unsqueeze(2).expand(-1, -1, H, -1)  # [B, G, H, N]
    B_exp = B_exp.view(Bsz, H, 1, N)  # [B, H, 1, N]
    C_exp = C.unsqueeze(2).expand(-1, -1, H, -1).view(Bsz, H, 1, N)

    # Update state: state = state * dA + dt * B * x
    dB = B_exp * dt.unsqueeze(-1).unsqueeze(-1)
    new_state = state * dA + dB * x.unsqueeze(-1)

    # Output: y = sum(state * C) + D * x
    y = (new_state * C_exp).sum(dim=-1)  # [B, H, P]
    if D is not None:
        D_exp = D.view(1, H, P) if D.ndim == 2 else D.view(1, H, 1)
        y = y + D_exp * x
    if z is not None:
        y = y * F.silu(z)

    return y, new_state
