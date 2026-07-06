"""foreblocks.ops.mamba.ssd_fused.

Three-kernel fully-fused SSD forward with no host-side L/G matrices.

Reduces 5+ PyTorch kernels to 3 Triton launches: (1) parallel cumsum of
dt*A, (2) parallel intra-chunk Y_intra + state_end computation, (3) serial
chunk scan for state passing and final output. Mathematically matches the
modular SSD path in ssd.py exactly. Use when you need maximum SSD throughput
and prefer fewer kernel launches over maximum parallelism.

Core API:
- fused_ssd_forward: three-kernel fused SSD forward (Triton)
- fused_ssd_forward_torch: pure-PyTorch reference path
- FUSED_SSD_TRITON_AVAILABLE: whether Triton path is usable

"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    FUSED_SSD_TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    FUSED_SSD_TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
#  Kernel 1: cumsum  — parallel over (B*nc*H)
# ---------------------------------------------------------------------------


@triton.jit
def _ssd_cumsum_kernel(
    dtA_ptr,  # [B, nc, CS, H]  pre-multiplied dt*A (or adt for Mamba3)
    cs_ptr,  # [B, nc, CS, H]  output cumsum
    CS: tl.constexpr,
    H: tl.constexpr,
):
    """Cumsum of dtA along chunk-time dimension."""
    pid = tl.program_id(0)  # flat B*nc*H
    pid_h = pid % H
    base = (pid // H) * CS * H + pid_h

    acc = tl.zeros([], dtype=tl.float32)
    for t in range(CS):
        dtA_t = tl.load(dtA_ptr + base + t * H).to(tl.float32)
        acc = acc + dtA_t
        tl.store(cs_ptr + base + t * H, acc)


# ---------------------------------------------------------------------------
#  Kernel 2: Y_intra + state_end  — parallel over (B*nc*H)
# ---------------------------------------------------------------------------


@triton.jit
def _ssd_intra_kernel(
    u_ptr,  # [B, nc, CS, H, P]
    dt_inj_ptr,  # [B, nc, CS, H]  raw dt used as injection weight in LdtG
    B_ptr,  # [B, nc, CS, H, N]
    C_ptr,  # [B, nc, CS, H, N]
    cs_ptr,  # [B, nc, CS, H]  cumsum(dtA) — controls decay
    Y_intra_ptr,  # [B, nc, CS, H, P]  output
    state_end_ptr,  # [B, nc, H, P, N]   output
    CS: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One threadblock per (b, c, h). Dynamic loops over CS.

    dt_inj is the raw dt used as injection weight (L*dt*G*u).
    cs is cumsum(dt*A) or cumsum(adt) — controls decay only.
    For Mamba2: dt_inj = dt (raw), cs = cumsum(dt*A).
    For Mamba3: dt_inj = dt (raw), cs = cumsum(adt).
    """
    pid = tl.program_id(0)
    pid_h = pid % H
    pid_bc = pid // H

    SHP = H * P
    SHN = H * N
    SH = H

    base_u = (pid_bc * CS * H + pid_h) * P
    base_dt = pid_bc * CS * H + pid_h
    base_B = (pid_bc * CS * H + pid_h) * N
    base_C = (pid_bc * CS * H + pid_h) * N
    base_cs = pid_bc * CS * H + pid_h
    base_y = (pid_bc * CS * H + pid_h) * P
    base_se = (pid_bc * H + pid_h) * P * N

    p_offs = tl.arange(0, BLOCK_P)
    n_offs = tl.arange(0, BLOCK_N)
    p_mask = p_offs < P
    n_mask = n_offs < N

    cs_last = tl.load(cs_ptr + base_cs + (CS - 1) * SH)

    # ── Y_intra: outer t, inner j ────────────────────────────────────
    for t in range(CS):
        cs_t = tl.load(cs_ptr + base_cs + t * SH)
        c_t = tl.load(C_ptr + base_C + t * SHN + n_offs, mask=n_mask).to(tl.float32)
        y_t = tl.zeros([BLOCK_P], dtype=tl.float32)
        for j in range(t + 1):
            cs_j = tl.load(cs_ptr + base_cs + j * SH)
            dt_j = tl.load(dt_inj_ptr + base_dt + j * SH).to(tl.float32)
            b_j = tl.load(B_ptr + base_B + j * SHN + n_offs, mask=n_mask).to(tl.float32)
            u_j = tl.load(u_ptr + base_u + j * SHP + p_offs, mask=p_mask).to(tl.float32)
            G_tj = tl.sum(c_t * b_j, axis=0)
            y_t = y_t + (tl.exp(cs_t - cs_j) * dt_j * G_tj) * u_j
        tl.store(
            Y_intra_ptr + base_y + t * SHP + p_offs,
            y_t.to(Y_intra_ptr.dtype.element_ty),
            mask=p_mask,
        )

    # ── state_end ────────────────────────────────────────────────────
    se_off = p_offs[:, None] * N + n_offs[None, :]
    se_acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)
    for j in range(CS):
        cs_j = tl.load(cs_ptr + base_cs + j * SH)
        dt_j = tl.load(dt_inj_ptr + base_dt + j * SH).to(tl.float32)
        b_j = tl.load(B_ptr + base_B + j * SHN + n_offs, mask=n_mask).to(tl.float32)
        u_j = tl.load(u_ptr + base_u + j * SHP + p_offs, mask=p_mask).to(tl.float32)
        w_j = tl.exp(cs_last - cs_j) * dt_j
        se_acc = se_acc + u_j[:, None] * (w_j * b_j)[None, :]
    tl.store(
        state_end_ptr + base_se + se_off,
        se_acc.to(state_end_ptr.dtype.element_ty),
        mask=(p_offs[:, None] < P) & (n_offs[None, :] < N),
    )


# ---------------------------------------------------------------------------
#  Kernel 3: serial scan  — parallel over (B*H)
# ---------------------------------------------------------------------------


@triton.jit
def _ssd_scan_kernel(
    Y_intra_ptr,  # [B, nc, CS, H, P]  read-only
    state_end_ptr,  # [B, nc, H, P, N]   read-only
    cs_ptr,  # [B, nc, CS, H]     read-only
    u_ptr,  # [B, nc, CS, H, P]  read-only
    C_ptr,  # [B, nc, CS, H, N]  read-only
    D_ptr,  # [H, P] or None
    s_in_ptr,  # [B, H, P, N]
    y_ptr,  # [B, nc, CS, H, P]  output
    nc: tl.constexpr,
    CS: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    """One threadblock per (b, h). Serially scans nc chunks."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    SHP = H * P
    SHN = H * N
    SH = H

    p_offs = tl.arange(0, BLOCK_P)
    n_offs = tl.arange(0, BLOCK_N)
    p_mask = p_offs < P
    n_mask = n_offs < N
    se_off = p_offs[:, None] * N + n_offs[None, :]
    se_mask = (p_offs[:, None] < P) & (n_offs[None, :] < N)

    if HAS_D:
        d_v = tl.load(D_ptr + pid_h * P + p_offs, mask=p_mask).to(tl.float32)
    else:
        d_v = tl.zeros([BLOCK_P], dtype=tl.float32)

    S = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)
    if HAS_INIT:
        S = tl.load(s_in_ptr + (pid_b * H + pid_h) * P * N + se_off, mask=se_mask).to(
            tl.float32
        )

    for c in range(nc):
        pid_bc = pid_b * nc + c
        base_u = (pid_bc * CS * H + pid_h) * P
        base_C = (pid_bc * CS * H + pid_h) * N
        base_yi = (pid_bc * CS * H + pid_h) * P
        base_y = (pid_bc * CS * H + pid_h) * P
        base_cs = pid_bc * CS * H + pid_h
        base_se = (pid_bc * H + pid_h) * P * N

        for t in range(CS):
            cs_t = tl.load(cs_ptr + base_cs + t * SH)
            c_t = tl.load(C_ptr + base_C + t * SHN + n_offs, mask=n_mask).to(tl.float32)
            u_t = tl.load(u_ptr + base_u + t * SHP + p_offs, mask=p_mask).to(tl.float32)
            yi_t = tl.load(Y_intra_ptr + base_yi + t * SHP + p_offs, mask=p_mask).to(
                tl.float32
            )
            y_t = yi_t + tl.exp(cs_t) * tl.sum(S * c_t[None, :], axis=1) + d_v * u_t
            tl.store(
                y_ptr + base_y + t * SHP + p_offs,
                y_t.to(y_ptr.dtype.element_ty),
                mask=p_mask,
            )

        cs_last = tl.load(cs_ptr + base_cs + (CS - 1) * SH)
        se_c = tl.load(state_end_ptr + base_se + se_off, mask=se_mask).to(tl.float32)
        S = tl.exp(cs_last) * S + se_c


# ---------------------------------------------------------------------------
#  Python wrapper
# ---------------------------------------------------------------------------


def fused_ssd_forward(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    chunk_size: int = 128,
    adt: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Three-kernel fused SSD forward.

    Args:
        u:   [B, T, H, P]
        dt:  [B, T, H]  raw dt (before *A)
        A:   [H]
        B:   [B, T, H, N]
        C:   [B, T, H, N]
        D:   [H, P] or None
        chunk_size: tokens per chunk (128 optimal for fused kernel)
        adt: [B, T, H]  pre-computed dt*A for Mamba3
        initial_states: [B, H, P, N]
        return_final_states: also return final state [B, H, P, N]
    """
    if not FUSED_SSD_TRITON_AVAILABLE:
        raise RuntimeError("fused_ssd_forward requires Triton")
    if not u.is_cuda:
        raise RuntimeError("fused_ssd_forward requires CUDA tensors")

    Bsz, T, H, P = u.shape
    N = B.shape[-1]

    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        u = F.pad(u, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
        if adt is not None:
            adt = F.pad(adt, (0, 0, 0, pad))
    T_pad = T + pad
    nc = T_pad // chunk_size

    u = u.contiguous()
    dt = dt.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    if D is not None:
        D = D.contiguous()

    # dtA for cumsum (decay): dt*A for Mamba2, adt for Mamba3
    # dt_inj for LdtG injection weight: always raw dt
    if adt is not None:
        dtA = adt.contiguous()  # [B, T, H]  — pre-computed dt*A
    else:
        dtA = dt * A[None, None, :]  # [B, T, H]
    dt_inj = dt  # raw dt — injection weight

    u_r = u.view(Bsz, nc, chunk_size, H, P)
    dtA_r = dtA.view(Bsz, nc, chunk_size, H)
    dt_inj_r = dt_inj.view(Bsz, nc, chunk_size, H)
    B_r = B.view(Bsz, nc, chunk_size, H, N)
    C_r = C.view(Bsz, nc, chunk_size, H, N)

    BLOCK_P = min(triton.next_power_of_2(P), 128)
    BLOCK_N = min(max(8, triton.next_power_of_2(N)), 128)

    f32 = torch.float32
    dev = u.device
    cs_buf = torch.empty((Bsz, nc, chunk_size, H), device=dev, dtype=f32)
    Y_intra = torch.empty((Bsz, nc, chunk_size, H, P), device=dev, dtype=f32)
    state_end = torch.empty((Bsz, nc, H, P, N), device=dev, dtype=f32)
    y_out = torch.empty((Bsz, nc, chunk_size, H, P), device=dev, dtype=u.dtype)

    _ssd_cumsum_kernel[(Bsz * nc * H,)](
        dtA_r,
        cs_buf,
        chunk_size,
        H,
    )

    _ssd_intra_kernel[(Bsz * nc * H,)](
        u_r,
        dt_inj_r,
        B_r,
        C_r,
        cs_buf,
        Y_intra,
        state_end,
        chunk_size,
        H,
        P,
        N,
        BLOCK_P,
        BLOCK_N,
    )

    has_init = initial_states is not None
    s_in = (
        initial_states.to(f32).contiguous()
        if has_init
        else torch.zeros((Bsz, H, P, N), device=dev, dtype=f32)
    )

    _ssd_scan_kernel[(Bsz, H)](
        Y_intra,
        state_end,
        cs_buf,
        u_r,
        C_r,
        D,
        s_in,
        y_out,
        nc,
        chunk_size,
        H,
        P,
        N,
        BLOCK_P,
        BLOCK_N,
        D is not None,
        has_init,
    )

    y = y_out.view(Bsz, T_pad, H, P)
    if pad > 0:
        y = y[:, :T]

    if return_final_states:
        cs_last_all = cs_buf[:, :, -1, :]  # [B, nc, H]
        final = s_in.clone()
        for c in range(nc):
            final = (
                torch.exp(cs_last_all[:, c, :])[:, :, None, None] * final
                + state_end[:, c]
            )
        return y, final

    return y


def fused_ssd_forward_torch(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    chunk_size: int = 128,
    adt: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference — delegates to _chunked_ssd_forward_modular."""
    from foreblocks.ops.mamba.ssd import _chunked_ssd_forward_modular

    D_eff = (
        D
        if D is not None
        else torch.zeros(A.shape[0], u.shape[-1], device=u.device, dtype=u.dtype)
    )
    A_eff = torch.ones_like(A) if adt is not None else A

    y, intermediates = _chunked_ssd_forward_modular(
        u,
        dt,
        A_eff,
        B,
        C,
        D_eff,
        chunk_size=chunk_size,
        adt=adt,
        seq_idx=seq_idx,
        initial_states=initial_states,
    )

    if return_final_states:
        return y, intermediates["final_states"].squeeze(1)
    return y
