"""Triton kernels for chunked SSD and their Python-side launchers.

The raw ``@triton.jit`` kernels and their launcher functions are one
coherent unit (launchers call the kernels by name with no indirection), so
they are kept together in this module rather than split further.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from foreblocks.kernels.mamba.segment_sum import _segment_sum_log

try:
    import triton
    import triton.language as tl

    CHUNKED_SSD_TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    CHUNKED_SSD_TRITON_AVAILABLE = False


if CHUNKED_SSD_TRITON_AVAILABLE:

    @triton.jit
    def _chunked_ssd_forward_kernel(
        u_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
        adt_ptr,
        entry_state_ptr,
        out_ptr,
        T,
        H,
        P,
        N,
        NC,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_ADT: tl.constexpr,
        D_PER_HEAD: tl.constexpr,
    ):
        # Decode program ID: grid = (B * NC * H, npb)
        pid_bhnc = tl.program_id(axis=0)
        pid_p = tl.program_id(axis=1)

        nc = pid_bhnc % NC
        tmp = pid_bhnc // NC
        b = tmp // H
        h = tmp % H

        t0 = nc * CHUNK_SIZE
        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        n_offs = tl.arange(0, BLOCK_N)
        p_mask = p_offs < P
        n_mask = n_offs < N

        # Load entry state for this chunk: [B, nc, H, P, N]
        entry_base = (
            (b * NC * H + nc * H + h) * (P * N) + p_offs[:, None] * N + n_offs[None, :]
        )
        entry_mask = p_mask[:, None] & n_mask[None, :]
        state = tl.load(entry_state_ptr + entry_base, mask=entry_mask, other=0.0).to(
            tl.float32
        )

        a_val = tl.load(A_ptr + h).to(tl.float32) if not USE_ADT else 0.0
        if D_PER_HEAD:
            d_vals = tl.load(D_ptr + h).to(tl.float32) + 0.0 * p_offs
        else:
            d_vals = tl.load(D_ptr + h * P + p_offs, mask=p_mask, other=0.0).to(
                tl.float32
            )

        for ti in tl.range(0, CHUNK_SIZE):
            t = t0 + ti
            active = t < T
            base_bth = (b * T + t) * H + h

            u_vals = tl.load(
                u_ptr + base_bth * P + p_offs,
                mask=active & p_mask,
                other=0.0,
            ).to(tl.float32)
            dt_val = tl.load(dt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
            b_vals = tl.load(
                B_ptr + base_bth * N + n_offs,
                mask=active & n_mask,
                other=0.0,
            ).to(tl.float32)
            c_vals = tl.load(
                C_ptr + base_bth * N + n_offs,
                mask=active & n_mask,
                other=0.0,
            ).to(tl.float32)

            if USE_ADT:
                log_decay = tl.load(adt_ptr + base_bth, mask=active, other=0.0).to(
                    tl.float32
                )
            else:
                log_decay = dt_val * a_val
            decay = tl.exp(log_decay)
            new_state = state * decay + dt_val * u_vals[:, None] * b_vals[None, :]
            state = tl.where(active, new_state, state)

            y_vals = tl.sum(state * c_vals[None, :], axis=1) + d_vals * u_vals
            tl.store(out_ptr + base_bth * P + p_offs, y_vals, mask=active & p_mask)

    @triton.jit
    def _chunked_ssd_forward_parallel_kernel(
        u_ptr,
        dt_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
        cumsum_ptr,
        entry_state_ptr,
        out_ptr,
        T,
        H,
        P,
        N,
        NC,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D_PER_HEAD: tl.constexpr,
    ):
        pid_bhnct = tl.program_id(axis=0)
        pid_p = tl.program_id(axis=1)

        ti = pid_bhnct % CHUNK_SIZE
        tmp = pid_bhnct // CHUNK_SIZE
        nc = tmp % NC
        tmp = tmp // NC
        h = tmp % H
        b = tmp // H

        t = nc * CHUNK_SIZE + ti
        active = t < T
        base_bth = (b * T + t) * H + h

        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        n_offs = tl.arange(0, BLOCK_N)
        p_mask = p_offs < P
        n_mask = n_offs < N

        c_vals = tl.load(
            C_ptr + base_bth * N + n_offs,
            mask=active & n_mask,
            other=0.0,
        ).to(tl.float32)
        u_t = tl.load(
            u_ptr + base_bth * P + p_offs,
            mask=active & p_mask,
            other=0.0,
        ).to(tl.float32)

        cumsum_base = ((b * NC + nc) * CHUNK_SIZE + ti) * H + h
        cumsum_t = tl.load(cumsum_ptr + cumsum_base, mask=active, other=0.0).to(
            tl.float32
        )

        entry_base = (
            (b * NC * H + nc * H + h) * (P * N) + p_offs[:, None] * N + n_offs[None, :]
        )
        entry_mask = p_mask[:, None] & n_mask[None, :]
        entry_state = tl.load(
            entry_state_ptr + entry_base, mask=entry_mask, other=0.0
        ).to(tl.float32)

        decay_from_start = tl.exp(cumsum_t)
        acc = tl.sum(entry_state * (decay_from_start * c_vals[None, :]), axis=1)

        for j in tl.range(0, CHUNK_SIZE):
            src_active = active & (j <= ti)
            tj = nc * CHUNK_SIZE + j
            src_base_bth = (b * T + tj) * H + h
            src_cumsum_base = ((b * NC + nc) * CHUNK_SIZE + j) * H + h

            cumsum_j = tl.load(
                cumsum_ptr + src_cumsum_base, mask=src_active, other=0.0
            ).to(tl.float32)
            dt_j = tl.load(dt_ptr + src_base_bth, mask=src_active, other=0.0).to(
                tl.float32
            )
            b_j = tl.load(
                B_ptr + src_base_bth * N + n_offs,
                mask=src_active & n_mask,
                other=0.0,
            ).to(tl.float32)
            u_j = tl.load(
                u_ptr + src_base_bth * P + p_offs,
                mask=src_active & p_mask,
                other=0.0,
            ).to(tl.float32)

            cb = tl.sum(c_vals * b_j, axis=0)
            coeff = tl.exp(cumsum_t - cumsum_j) * dt_j * cb
            acc += tl.where(src_active, coeff, 0.0) * u_j

        if D_PER_HEAD:
            d_vals = tl.load(D_ptr + h).to(tl.float32) + 0.0 * p_offs
        else:
            d_vals = tl.load(D_ptr + h * P + p_offs, mask=p_mask, other=0.0).to(
                tl.float32
            )
        acc += d_vals * u_t

        tl.store(out_ptr + base_bth * P + p_offs, acc, mask=active & p_mask)

    @triton.jit
    def _chunked_ssd_forward_tiled_kernel(
        u_ptr,
        dt_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
        cumsum_ptr,
        entry_state_ptr,
        out_ptr,
        T,
        H,
        P,
        N,
        NC,
        TILES_PER_CHUNK: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D_PER_HEAD: tl.constexpr,
    ):
        pid_tile = tl.program_id(axis=0)
        pid_p = tl.program_id(axis=1)

        tile_t = pid_tile % TILES_PER_CHUNK
        tmp = pid_tile // TILES_PER_CHUNK
        nc = tmp % NC
        tmp = tmp // NC
        h = tmp % H
        b = tmp // H

        t_offs = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        n_offs = tl.arange(0, BLOCK_N)
        j_offs = tl.arange(0, BLOCK_J)

        t_abs = nc * CHUNK_SIZE + t_offs
        t_active = t_abs < T
        p_mask = p_offs < P
        n_mask = n_offs < N

        cumsum_t = tl.load(
            cumsum_ptr + ((b * NC + nc) * CHUNK_SIZE + t_offs) * H + h,
            mask=t_active,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_T, BLOCK_P), dtype=tl.float32)

        # Inter-chunk entry-state contribution:
        #   decay_from_start[t] * sum_n C[t,n] * entry_state[p,n]
        for n0 in tl.range(0, N, BLOCK_N):
            n_idx = n0 + n_offs
            n_active = n_idx < N
            c_vals = tl.load(
                C_ptr + (((b * T + t_abs[:, None]) * H + h) * N + n_idx[None, :]),
                mask=t_active[:, None] & n_active[None, :],
                other=0.0,
            ).to(tl.float32)
            entry_vals = tl.load(
                entry_state_ptr
                + (b * NC * H + nc * H + h) * (P * N)
                + p_offs[:, None] * N
                + n_idx[None, :],
                mask=p_mask[:, None] & n_active[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += (
                tl.dot(c_vals, tl.trans(entry_vals), input_precision="ieee")
                * tl.exp(cumsum_t)[:, None]
            )

        # Intra-chunk contribution, tiled over source tokens j and state N.
        for j0 in tl.range(0, CHUNK_SIZE, BLOCK_J):
            j_idx = j0 + j_offs
            j_abs = nc * CHUNK_SIZE + j_idx
            j_active = j_abs < T
            cumsum_j = tl.load(
                cumsum_ptr + ((b * NC + nc) * CHUNK_SIZE + j_idx) * H + h,
                mask=j_active,
                other=0.0,
            ).to(tl.float32)
            dt_j = tl.load(
                dt_ptr + ((b * T + j_abs) * H + h),
                mask=j_active,
                other=0.0,
            ).to(tl.float32)

            g = tl.zeros((BLOCK_T, BLOCK_J), dtype=tl.float32)
            for n0 in tl.range(0, N, BLOCK_N):
                n_idx = n0 + n_offs
                n_active = n_idx < N
                c_vals = tl.load(
                    C_ptr + (((b * T + t_abs[:, None]) * H + h) * N + n_idx[None, :]),
                    mask=t_active[:, None] & n_active[None, :],
                    other=0.0,
                ).to(tl.float32)
                b_vals = tl.load(
                    B_ptr + (((b * T + j_abs[:, None]) * H + h) * N + n_idx[None, :]),
                    mask=j_active[:, None] & n_active[None, :],
                    other=0.0,
                ).to(tl.float32)
                g += tl.dot(c_vals, tl.trans(b_vals), input_precision="ieee")

            causal = j_idx[None, :] <= t_offs[:, None]
            valid = t_active[:, None] & j_active[None, :] & causal
            weights = tl.where(
                valid,
                tl.exp(cumsum_t[:, None] - cumsum_j[None, :]) * dt_j[None, :] * g,
                0.0,
            )
            u_vals = tl.load(
                u_ptr + (((b * T + j_abs[:, None]) * H + h) * P + p_offs[None, :]),
                mask=j_active[:, None] & p_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(weights, u_vals, input_precision="ieee")

        if D_PER_HEAD:
            d_vals = tl.load(D_ptr + h).to(tl.float32) + 0.0 * p_offs
        else:
            d_vals = tl.load(D_ptr + h * P + p_offs, mask=p_mask, other=0.0).to(
                tl.float32
            )
        u_t = tl.load(
            u_ptr + (((b * T + t_abs[:, None]) * H + h) * P + p_offs[None, :]),
            mask=t_active[:, None] & p_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += u_t * d_vals[None, :]

        tl.store(
            out_ptr + (((b * T + t_abs[:, None]) * H + h) * P + p_offs[None, :]),
            acc,
            mask=t_active[:, None] & p_mask[None, :],
        )

    @triton.jit
    def _chunked_ssd_boundary_kernel(
        u_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        adt_ptr,
        bstate_ptr,  # [B, H, nc+1, P, N] — entry state of each chunk (bstate[0]=0)
        T,
        H,
        P,
        N,
        NC,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_ADT: tl.constexpr,
    ):
        pid_bh = tl.program_id(axis=0)
        pid_p = tl.program_id(axis=1)
        b = pid_bh // H
        h = pid_bh % H

        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        n_offs = tl.arange(0, BLOCK_N)
        p_mask = p_offs < P
        n_mask = n_offs < N
        pn_mask = p_mask[:, None] & n_mask[None, :]

        a_val = tl.load(A_ptr + h).to(tl.float32) if not USE_ADT else 0.0
        state = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

        # bstate layout stride: ((b*H + h)*(NC+1) + c)*P*N + p*N + n
        bstride_c = P * N
        bbase = (
            ((b * H + h) * (NC + 1)) * bstride_c + p_offs[:, None] * N + n_offs[None, :]
        )
        # chunk 0 entry state is zero
        tl.store(bstate_ptr + bbase, state, mask=pn_mask)

        for c in tl.range(0, NC):
            t0 = c * CHUNK_SIZE
            for ti in tl.range(0, CHUNK_SIZE):
                t = t0 + ti
                active = t < T
                base_bth = (b * T + t) * H + h
                u_vals = tl.load(
                    u_ptr + base_bth * P + p_offs, mask=active & p_mask, other=0.0
                ).to(tl.float32)
                dt_val = tl.load(dt_ptr + base_bth, mask=active, other=0.0).to(
                    tl.float32
                )
                b_vals = tl.load(
                    B_ptr + base_bth * N + n_offs, mask=active & n_mask, other=0.0
                ).to(tl.float32)
                if USE_ADT:
                    log_decay = tl.load(adt_ptr + base_bth, mask=active, other=0.0).to(
                        tl.float32
                    )
                else:
                    log_decay = dt_val * a_val
                decay = tl.exp(log_decay)
                new_state = state * decay + dt_val * u_vals[:, None] * b_vals[None, :]
                state = tl.where(active, new_state, state)
            # store entry state of chunk c+1
            wbase = (
                ((b * H + h) * (NC + 1) + (c + 1)) * bstride_c
                + p_offs[:, None] * N
                + n_offs[None, :]
            )
            tl.store(bstate_ptr + wbase, state, mask=pn_mask)

    @triton.jit
    def _chunked_ssd_backward_kernel(
        gy_ptr,
        u_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
        adt_ptr,
        bstate_ptr,  # [B, H, nc+1, P, N]
        scratch_ptr,  # [B*H, BLOCK_P_GRID, CHUNK_SIZE, BLOCK_P, BLOCK_N] forward states
        du_ptr,
        ddt_ptr,
        dA_ptr,  # [H]            (atomic; unused when USE_ADT)
        dB_ptr,  # [B, T, H, N]   (atomic over p-blocks)
        dC_ptr,  # [B, T, H, N]   (atomic over p-blocks)
        dD_ptr,  # [H] or [H, P]  (atomic)
        dadt_ptr,  # [B, T, H]      (atomic over p-blocks; unused when not USE_ADT)
        gstate_ptr,  # [B, H, P, N] running grad-state carried across chunks (reverse)
        T,
        H,
        P,
        N,
        NC,
        NPB,  # number of p-blocks (grid dim 1)
        CHUNK_SIZE: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_ADT: tl.constexpr,
        D_PER_HEAD: tl.constexpr,
        CHUNK_INDEX: tl.constexpr,
    ):
        pid_bh = tl.program_id(axis=0)
        pid_p = tl.program_id(axis=1)
        b = pid_bh // H
        h = pid_bh % H

        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        n_offs = tl.arange(0, BLOCK_N)
        p_mask = p_offs < P
        n_mask = n_offs < N
        pn_mask = p_mask[:, None] & n_mask[None, :]

        a_val = tl.load(A_ptr + h).to(tl.float32) if not USE_ADT else 0.0
        if D_PER_HEAD:
            d_vals = tl.load(D_ptr + h).to(tl.float32) + 0.0 * p_offs
        else:
            d_vals = tl.load(D_ptr + h * P + p_offs, mask=p_mask, other=0.0).to(
                tl.float32
            )

        bstride_c = P * N
        # scratch base for this (pid_bh, pid_p): [CHUNK_SIZE, BLOCK_P, BLOCK_N]
        sc_block = CHUNK_SIZE * BLOCK_P * BLOCK_N
        sc0 = (pid_bh * NPB + pid_p) * sc_block
        pn_idx = (
            tl.arange(0, BLOCK_P)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        )

        t0 = CHUNK_INDEX * CHUNK_SIZE

        # ── forward sweep: recompute & store state_after[ti] for this chunk ──
        entry_base = (
            ((b * H + h) * (NC + 1) + CHUNK_INDEX) * bstride_c
            + p_offs[:, None] * N
            + n_offs[None, :]
        )
        state = tl.load(bstate_ptr + entry_base, mask=pn_mask, other=0.0).to(tl.float32)
        for ti in tl.range(0, CHUNK_SIZE):
            t = t0 + ti
            active = t < T
            base_bth = (b * T + t) * H + h
            u_vals = tl.load(
                u_ptr + base_bth * P + p_offs, mask=active & p_mask, other=0.0
            ).to(tl.float32)
            dt_val = tl.load(dt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
            b_vals = tl.load(
                B_ptr + base_bth * N + n_offs, mask=active & n_mask, other=0.0
            ).to(tl.float32)
            if USE_ADT:
                log_decay = tl.load(adt_ptr + base_bth, mask=active, other=0.0).to(
                    tl.float32
                )
            else:
                log_decay = dt_val * a_val
            decay = tl.exp(log_decay)
            new_state = state * decay + dt_val * u_vals[:, None] * b_vals[None, :]
            state = tl.where(active, new_state, state)
            tl.store(scratch_ptr + sc0 + ti * (BLOCK_P * BLOCK_N) + pn_idx, state)

        # ── reverse sweep ──
        gstate_base = ((b * H + h) * P + p_offs[:, None]) * N + n_offs[None, :]
        grad_state = tl.load(gstate_ptr + gstate_base, mask=pn_mask, other=0.0).to(
            tl.float32
        )

        dA_acc = 0.0
        for ti_rev in tl.range(0, CHUNK_SIZE):
            ti = CHUNK_SIZE - 1 - ti_rev
            t = t0 + ti
            active = t < T
            base_bth = (b * T + t) * H + h

            u_vals = tl.load(
                u_ptr + base_bth * P + p_offs, mask=active & p_mask, other=0.0
            ).to(tl.float32)
            dt_val = tl.load(dt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
            b_vals = tl.load(
                B_ptr + base_bth * N + n_offs, mask=active & n_mask, other=0.0
            ).to(tl.float32)
            c_vals = tl.load(
                C_ptr + base_bth * N + n_offs, mask=active & n_mask, other=0.0
            ).to(tl.float32)
            gy_vals = tl.load(
                gy_ptr + base_bth * P + p_offs, mask=active & p_mask, other=0.0
            ).to(tl.float32)
            if USE_ADT:
                log_decay = tl.load(adt_ptr + base_bth, mask=active, other=0.0).to(
                    tl.float32
                )
            else:
                log_decay = dt_val * a_val
            decay = tl.exp(log_decay)

            # state_t = state_after[ti]; state_prev = state_after[ti-1] (entry if ti==0)
            state_t = tl.load(scratch_ptr + sc0 + ti * (BLOCK_P * BLOCK_N) + pn_idx)
            if ti == 0:
                state_prev = tl.load(
                    bstate_ptr + entry_base, mask=pn_mask, other=0.0
                ).to(tl.float32)
            else:
                state_prev = tl.load(
                    scratch_ptr + sc0 + (ti - 1) * (BLOCK_P * BLOCK_N) + pn_idx
                )

            # ── output term: y[p] = sum_n state_t[p,n] c[n] + D u[p] ──
            dC_partial = tl.sum(gy_vals[:, None] * state_t, axis=0)  # [BLOCK_N]
            tl.atomic_add(
                dC_ptr + base_bth * N + n_offs, dC_partial, mask=active & n_mask
            )
            if D_PER_HEAD:
                tl.atomic_add(dD_ptr + h, tl.sum(gy_vals * u_vals, axis=0), mask=active)
            else:
                tl.atomic_add(
                    dD_ptr + h * P + p_offs, gy_vals * u_vals, mask=active & p_mask
                )
            du_acc = gy_vals * d_vals  # du from D-skip
            grad_state = grad_state + gy_vals[:, None] * c_vals[None, :]

            # ── state term: state_t = decay state_prev + dt u⊗B ──
            du_acc = du_acc + tl.sum(grad_state * (dt_val * b_vals[None, :]), axis=1)
            tl.atomic_add(du_ptr + base_bth * P + p_offs, du_acc, mask=active & p_mask)
            dB_partial = tl.sum(grad_state * (dt_val * u_vals[:, None]), axis=0)
            tl.atomic_add(
                dB_ptr + base_bth * N + n_offs, dB_partial, mask=active & n_mask
            )
            ddt_partial = tl.sum(grad_state * (u_vals[:, None] * b_vals[None, :]))
            g_logdecay = tl.sum(grad_state * decay * state_prev)
            if USE_ADT:
                tl.atomic_add(dadt_ptr + base_bth, g_logdecay, mask=active)
                tl.atomic_add(ddt_ptr + base_bth, ddt_partial, mask=active)
            else:
                tl.atomic_add(
                    ddt_ptr + base_bth, ddt_partial + g_logdecay * a_val, mask=active
                )
                dA_acc += tl.where(active, g_logdecay * dt_val, 0.0)
            grad_state = grad_state * decay

        tl.store(gstate_ptr + gstate_base, grad_state, mask=pn_mask)
        if not USE_ADT:
            tl.atomic_add(dA_ptr + h, dA_acc)


def chunked_ssd_forward_triton(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    adt: torch.Tensor | None = None,
) -> torch.Tensor:
    if not CHUNKED_SSD_TRITON_AVAILABLE:
        raise RuntimeError(
            "chunked_ssd_forward_triton called but Triton is unavailable"
        )
    use_adt = adt is not None
    tensors = [u, dt, A, B, C, D] + ([adt] if use_adt else [])
    if not all(t.is_cuda for t in tensors):
        raise ValueError("chunked_ssd_forward_triton expects CUDA tensors")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    Bsz, T, H, P = u.shape
    N = B.shape[-1]
    if dt.shape != (Bsz, T, H):
        raise ValueError("dt shape must match [B, T, H]")
    if not use_adt and A.shape != (H,):
        raise ValueError("A shape must match [H]")
    if use_adt and adt.shape != (Bsz, T, H):
        raise ValueError("adt shape must match [B, T, H]")
    if B.shape != (Bsz, T, H, N) or C.shape != B.shape:
        raise ValueError("B and C must have shape [B, T, H, N]")
    d_per_head = D.shape == (H,)
    if D.shape != (H, P) and not d_per_head:
        raise ValueError("D shape must match [H, P] or [H]")
    if N > 128:
        raise ValueError("chunked_ssd_forward_triton currently supports d_state <= 128")

    u_f = u.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    D_f = D.float()
    adt_f = adt.float() if use_adt else None

    # pad to chunk boundary
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        u_f = F.pad(u_f, (0, 0, 0, 0, 0, pad))
        dt_f = F.pad(dt_f, (0, 0, 0, pad))
        B_f = F.pad(B_f, (0, 0, 0, 0, 0, pad))
        C_f = F.pad(C_f, (0, 0, 0, 0, 0, pad))
        if use_adt:
            adt_f = F.pad(adt_f, (0, 0, 0, pad))
    T_pad = T + pad
    nc = T_pad // chunk_size

    # ── dtA = dt * A — shape [B, T, H] ─────────────────────────────
    if use_adt:
        dtA = adt_f
    else:
        dtA = dt_f * A_f

    # reshape to chunks
    dtA_c = dtA.view(Bsz, nc, chunk_size, H)  # [B, nc, C, H]
    dt_c = dt_f.view(Bsz, nc, chunk_size, H)
    u_c = u_f.view(Bsz, nc, chunk_size, H, P)
    B_c = B_f.view(Bsz, nc, chunk_size, H, N)
    C_c = C_f.view(Bsz, nc, chunk_size, H, N)

    # cumsum along chunk-time
    cumsum_dtA = torch.cumsum(dtA_c, dim=2)  # [B, nc, C, H]

    # ── state_end[c, h, p, n] — intra-chunk state at end ──────────
    cumsum_dtA_last = cumsum_dtA[:, :, -1:, None, :]  # [B, nc, 1, 1, H]
    L_last = torch.exp(
        cumsum_dtA_last - cumsum_dtA[:, :, None, :, :]
    )  # [B, nc, 1, C, H]
    Ldt_last = L_last.squeeze(2) * dt_c  # [B, nc, C, H]
    LB_last = Ldt_last.unsqueeze(-1) * B_c  # [B, nc, C, H, N]
    state_end = torch.einsum("bcjhn,bcjhp->bchpn", LB_last, u_c)  # [B, nc, H, P, N]

    # ── parallel inter-chunk boundary via segment_sum_log ──────────
    zero_state = torch.zeros(Bsz, 1, H, P, N, device=u_f.device, dtype=torch.float32)
    state_summaries = torch.cat([zero_state, state_end], dim=1)  # [B, nc+1, H, P, N]
    chunk_log_decay = cumsum_dtA[:, :, -1, :].transpose(1, 2)  # [B, H, nc]
    chunk_log_decay = F.pad(chunk_log_decay, (1, 0))  # [B, H, nc+1]
    decay_prefix = torch.exp(_segment_sum_log(chunk_log_decay)).transpose(1, 3)
    boundary_all = (
        decay_prefix[..., None, None] * state_summaries[:, :, None, ...]
    ).sum(dim=1)  # [B, nc+1, H, P, N]
    entry_states = boundary_all[:, :-1]  # [B, nc, H, P, N]

    # ── launch ONE kernel: all (b, nc, h) threads at once ─────────
    u_contig = u_c.contiguous()
    dt_contig = dt_c.contiguous()
    A_contig = A_f.contiguous()
    B_contig = B_c.contiguous()
    C_contig = C_c.contiguous()
    D_contig = D_f.contiguous()
    adt_contig = adt_f.contiguous() if use_adt else u_contig
    entry_states_contig = entry_states.contiguous()
    out = torch.empty(Bsz, T_pad, H, P, dtype=u.dtype, device=u.device)

    block_p = min(max(triton.next_power_of_2(P), 16), 128)
    block_n = max(8, triton.next_power_of_2(N))
    # num_warps: each program covers block_p head-dims × chunk_size time steps
    # 4 warps for small block_p, 8 for large; num_stages=2 hides memory latency
    num_warps = 8 if block_p >= 64 else 4
    num_stages = 2

    grid = (Bsz * nc * H, triton.cdiv(P, block_p))
    _chunked_ssd_forward_kernel[grid](
        u_contig,
        dt_contig,
        A_contig,
        B_contig,
        C_contig,
        D_contig,
        adt_contig,
        entry_states_contig,
        out,
        T_pad,
        H,
        P,
        N,
        nc,
        CHUNK_SIZE=chunk_size,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
        USE_ADT=use_adt,
        D_PER_HEAD=d_per_head,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if pad > 0:
        out = out[:, :T]
    return out


def chunked_ssd_forward_triton_parallel(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    adt: torch.Tensor | None = None,
) -> torch.Tensor:
    if not CHUNKED_SSD_TRITON_AVAILABLE:
        raise RuntimeError(
            "chunked_ssd_forward_triton_parallel called but Triton is unavailable"
        )
    use_adt = adt is not None
    tensors = [u, dt, A, B, C, D] + ([adt] if use_adt else [])
    if not all(t.is_cuda for t in tensors):
        raise ValueError("chunked_ssd_forward_triton_parallel expects CUDA tensors")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    Bsz, T, H, P = u.shape
    N = B.shape[-1]
    if dt.shape != (Bsz, T, H):
        raise ValueError("dt shape must match [B, T, H]")
    if not use_adt and A.shape != (H,):
        raise ValueError("A shape must match [H]")
    if use_adt and adt.shape != (Bsz, T, H):
        raise ValueError("adt shape must match [B, T, H]")
    if B.shape != (Bsz, T, H, N) or C.shape != B.shape:
        raise ValueError("B and C must have shape [B, T, H, N]")
    d_per_head = D.shape == (H,)
    if D.shape != (H, P) and not d_per_head:
        raise ValueError("D shape must match [H, P] or [H]")
    if N > 128:
        raise ValueError(
            "chunked_ssd_forward_triton_parallel currently supports d_state <= 128"
        )

    u_f = u.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    D_f = D.float()
    adt_f = adt.float() if use_adt else None

    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        u_f = F.pad(u_f, (0, 0, 0, 0, 0, pad))
        dt_f = F.pad(dt_f, (0, 0, 0, pad))
        B_f = F.pad(B_f, (0, 0, 0, 0, 0, pad))
        C_f = F.pad(C_f, (0, 0, 0, 0, 0, pad))
        if use_adt:
            adt_f = F.pad(adt_f, (0, 0, 0, pad))
    T_pad = T + pad
    nc = T_pad // chunk_size

    dtA = adt_f if use_adt else dt_f * A_f
    dtA_c = dtA.view(Bsz, nc, chunk_size, H)
    dt_c = dt_f.view(Bsz, nc, chunk_size, H)
    u_c = u_f.view(Bsz, nc, chunk_size, H, P)
    B_c = B_f.view(Bsz, nc, chunk_size, H, N)
    C_c = C_f.view(Bsz, nc, chunk_size, H, N)

    cumsum_dtA = torch.cumsum(dtA_c, dim=2).contiguous()

    cumsum_dtA_last = cumsum_dtA[:, :, -1:, None, :]
    L_last = torch.exp(cumsum_dtA_last - cumsum_dtA[:, :, None, :, :])
    Ldt_last = L_last.squeeze(2) * dt_c
    LB_last = Ldt_last.unsqueeze(-1) * B_c
    state_end = torch.einsum("bcjhn,bcjhp->bchpn", LB_last, u_c)

    zero_state = torch.zeros(Bsz, 1, H, P, N, device=u_f.device, dtype=torch.float32)
    state_summaries = torch.cat([zero_state, state_end], dim=1)
    chunk_log_decay = cumsum_dtA[:, :, -1, :].transpose(1, 2)
    chunk_log_decay = F.pad(chunk_log_decay, (1, 0))
    decay_prefix = torch.exp(_segment_sum_log(chunk_log_decay)).transpose(1, 3)
    boundary_all = (
        decay_prefix[..., None, None] * state_summaries[:, :, None, ...]
    ).sum(dim=1)
    entry_states = boundary_all[:, :-1].contiguous()

    u_contig = u_c.contiguous()
    dt_contig = dt_c.contiguous()
    B_contig = B_c.contiguous()
    C_contig = C_c.contiguous()
    D_contig = D_f.contiguous()
    out = torch.empty(Bsz, T_pad, H, P, dtype=u.dtype, device=u.device)

    block_p = min(max(triton.next_power_of_2(P), 16), 128)
    block_n = max(8, triton.next_power_of_2(N))
    num_warps = 8 if block_p >= 64 else 4
    grid = (Bsz * nc * H * chunk_size, triton.cdiv(P, block_p))
    _chunked_ssd_forward_parallel_kernel[grid](
        u_contig,
        dt_contig,
        B_contig,
        C_contig,
        D_contig,
        cumsum_dtA,
        entry_states,
        out,
        T_pad,
        H,
        P,
        N,
        nc,
        CHUNK_SIZE=chunk_size,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
        D_PER_HEAD=d_per_head,
        num_warps=num_warps,
        num_stages=2,
    )
    if pad > 0:
        out = out[:, :T]
    return out


def chunked_ssd_forward_triton_tiled(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    adt: torch.Tensor | None = None,
) -> torch.Tensor:
    if not CHUNKED_SSD_TRITON_AVAILABLE:
        raise RuntimeError(
            "chunked_ssd_forward_triton_tiled called but Triton is unavailable"
        )
    use_adt = adt is not None
    tensors = [u, dt, A, B, C, D] + ([adt] if use_adt else [])
    if not all(t.is_cuda for t in tensors):
        raise ValueError("chunked_ssd_forward_triton_tiled expects CUDA tensors")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    Bsz, T, H, P = u.shape
    N = B.shape[-1]
    if dt.shape != (Bsz, T, H):
        raise ValueError("dt shape must match [B, T, H]")
    if not use_adt and A.shape != (H,):
        raise ValueError("A shape must match [H]")
    if use_adt and adt.shape != (Bsz, T, H):
        raise ValueError("adt shape must match [B, T, H]")
    if B.shape != (Bsz, T, H, N) or C.shape != B.shape:
        raise ValueError("B and C must have shape [B, T, H, N]")
    d_per_head = D.shape == (H,)
    if D.shape != (H, P) and not d_per_head:
        raise ValueError("D shape must match [H, P] or [H]")
    if N > 128:
        raise ValueError(
            "chunked_ssd_forward_triton_tiled currently supports d_state <= 128"
        )

    u_f = u.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    D_f = D.float()
    adt_f = adt.float() if use_adt else None

    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        u_f = F.pad(u_f, (0, 0, 0, 0, 0, pad))
        dt_f = F.pad(dt_f, (0, 0, 0, pad))
        B_f = F.pad(B_f, (0, 0, 0, 0, 0, pad))
        C_f = F.pad(C_f, (0, 0, 0, 0, 0, pad))
        if use_adt:
            adt_f = F.pad(adt_f, (0, 0, 0, pad))
    T_pad = T + pad
    nc = T_pad // chunk_size

    dtA = adt_f if use_adt else dt_f * A_f
    dtA_c = dtA.view(Bsz, nc, chunk_size, H)
    dt_c = dt_f.view(Bsz, nc, chunk_size, H)
    u_c = u_f.view(Bsz, nc, chunk_size, H, P)
    B_c = B_f.view(Bsz, nc, chunk_size, H, N)
    C_c = C_f.view(Bsz, nc, chunk_size, H, N)

    cumsum_dtA = torch.cumsum(dtA_c, dim=2).contiguous()

    cumsum_dtA_last = cumsum_dtA[:, :, -1:, None, :]
    L_last = torch.exp(cumsum_dtA_last - cumsum_dtA[:, :, None, :, :])
    Ldt_last = L_last.squeeze(2) * dt_c
    LB_last = Ldt_last.unsqueeze(-1) * B_c
    state_end = torch.einsum("bcjhn,bcjhp->bchpn", LB_last, u_c)

    zero_state = torch.zeros(Bsz, 1, H, P, N, device=u_f.device, dtype=torch.float32)
    state_summaries = torch.cat([zero_state, state_end], dim=1)
    chunk_log_decay = cumsum_dtA[:, :, -1, :].transpose(1, 2)
    chunk_log_decay = F.pad(chunk_log_decay, (1, 0))
    decay_prefix = torch.exp(_segment_sum_log(chunk_log_decay)).transpose(1, 3)
    boundary_all = (
        decay_prefix[..., None, None] * state_summaries[:, :, None, ...]
    ).sum(dim=1)
    entry_states = boundary_all[:, :-1].contiguous()

    u_contig = u_c.contiguous()
    dt_contig = dt_c.contiguous()
    B_contig = B_c.contiguous()
    C_contig = C_c.contiguous()
    D_contig = D_f.contiguous()
    out = torch.empty(Bsz, T_pad, H, P, dtype=u.dtype, device=u.device)

    block_t = 16 if chunk_size >= 16 else triton.next_power_of_2(chunk_size)
    block_j = min(32, max(16, triton.next_power_of_2(chunk_size)))
    block_p = min(max(triton.next_power_of_2(P), 16), 64)
    block_n = min(max(triton.next_power_of_2(N), 16), 64)
    tiles_per_chunk = triton.cdiv(chunk_size, block_t)
    num_warps = 8 if block_p >= 32 else 4
    grid = (Bsz * nc * H * tiles_per_chunk, triton.cdiv(P, block_p))
    _chunked_ssd_forward_tiled_kernel[grid](
        u_contig,
        dt_contig,
        B_contig,
        C_contig,
        D_contig,
        cumsum_dtA,
        entry_states,
        out,
        T_pad,
        H,
        P,
        N,
        nc,
        tiles_per_chunk,
        CHUNK_SIZE=chunk_size,
        BLOCK_T=block_t,
        BLOCK_J=block_j,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
        D_PER_HEAD=d_per_head,
        num_warps=num_warps,
        num_stages=2,
    )
    if pad > 0:
        out = out[:, :T]
    return out


def chunked_ssd_backward_triton(
    grad_y: torch.Tensor,
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    needs_input_grad: tuple[bool, ...] | None = None,
    adt: torch.Tensor | None = None,
    needs_adt_grad: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    if not CHUNKED_SSD_TRITON_AVAILABLE:
        raise RuntimeError(
            "chunked_ssd_backward_triton called but Triton is unavailable"
        )
    use_adt = adt is not None
    B_, T, H, P = u.shape
    N = B.shape[-1]
    d_per_head = D.ndim == 1

    gy = grad_y.contiguous()
    u_c = u.contiguous()
    dt_c = dt.contiguous()
    A_c = A.contiguous()
    Bm = B.contiguous()
    Cm = C.contiguous()
    Dm = D.contiguous()
    adt_c = adt.contiguous() if use_adt else u_c
    nc = triton.cdiv(T, chunk_size)

    bstate = torch.empty(B_, H, nc + 1, P, N, device=u.device, dtype=torch.float32)
    # grads (fp32 accumulators for atomics)
    du = torch.zeros_like(u_c, dtype=torch.float32)
    ddt = torch.zeros(B_, T, H, device=u.device, dtype=torch.float32)
    dA = torch.zeros(H, device=u.device, dtype=torch.float32)
    dB = torch.zeros(B_, T, H, N, device=u.device, dtype=torch.float32)
    dC = torch.zeros(B_, T, H, N, device=u.device, dtype=torch.float32)
    dD = torch.zeros_like(Dm, dtype=torch.float32)
    dadt = torch.zeros(B_, T, H, device=u.device, dtype=torch.float32)
    gstate = torch.zeros(B_, H, P, N, device=u.device, dtype=torch.float32)

    block_p = min(max(triton.next_power_of_2(P), 16), 128)
    block_n = max(8, triton.next_power_of_2(N))
    npb = triton.cdiv(P, block_p)
    grid = (B_ * H, npb)

    _chunked_ssd_boundary_kernel[grid](
        u_c,
        dt_c,
        A_c,
        Bm,
        adt_c,
        bstate,
        T,
        H,
        P,
        N,
        nc,
        CHUNK_SIZE=chunk_size,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
        USE_ADT=use_adt,
    )
    # scratch for recomputed forward states of the chunk under processing
    scratch = torch.empty(
        B_ * H, npb, chunk_size, block_p, block_n, device=u.device, dtype=torch.float32
    )
    for chunk_idx in range(nc - 1, -1, -1):
        _chunked_ssd_backward_kernel[grid](
            gy,
            u_c,
            dt_c,
            A_c,
            Bm,
            Cm,
            Dm,
            adt_c,
            bstate,
            scratch,
            du,
            ddt,
            dA,
            dB,
            dC,
            dD,
            dadt,
            gstate,
            T,
            H,
            P,
            N,
            nc,
            npb,
            CHUNK_SIZE=chunk_size,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
            USE_ADT=use_adt,
            D_PER_HEAD=d_per_head,
            CHUNK_INDEX=chunk_idx,
        )

    od = grad_y.dtype
    out_du = du.to(od) if needs_input_grad[0] else None
    out_ddt = ddt.to(od) if needs_input_grad[1] else None
    out_dA = dA.to(od) if (needs_input_grad[2] and not use_adt) else None
    out_dB = dB.to(od) if needs_input_grad[3] else None
    out_dC = dC.to(od) if needs_input_grad[4] else None
    out_dD = dD.to(od) if needs_input_grad[5] else None
    out_dadt = dadt.to(od) if (use_adt and needs_adt_grad) else None
    return (out_du, out_ddt, out_dA, out_dB, out_dC, out_dD, out_dadt)


