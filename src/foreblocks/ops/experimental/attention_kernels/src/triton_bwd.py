"""custom_att.triton_bwd.

Triton-based flash-attention backward kernel with multiple scheduling strategies.

Implements backward passes for Q, K, V gradients using split-K accumulation
for unbalanced causal workloads, persistent kernels for large sequences, and a
combined kernel that fuses dK/dV and dQ computation. Includes Ada GPU config
selection and fused delta optimization.

Core API:
- triton_flash_bwd: backward pass via Triton with multiple kernel dispatch strategies
- can_use_triton_bwd: check whether Triton backward kernel can handle input tensor

"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_preprocess(
    o,
    do,
    delta,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    block_m: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    base = pid_bh * n_ctx * d_head

    o_desc = tl.make_tensor_descriptor(
        base=o + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    do_desc = tl.make_tensor_descriptor(
        base=do + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )

    o_tile = o_desc.load([pid_m * block_m, 0]).to(tl.float32)
    do_tile = do_desc.load([pid_m * block_m, 0]).to(tl.float32)
    d = tl.sum(o_tile * do_tile, axis=1)
    tl.store(delta + pid_bh * n_ctx + offs_m, d, mask=offs_m < n_ctx)


# stage: 0 = non-causal, 1 = causal off-diagonal (no mask), 2 = causal diagonal (mask)
@triton.jit
def _bwd_dkdv_inner(
    dk_acc,
    dv_acc,
    k_tile,
    v_tile,
    q_desc,
    o_desc,
    do_desc,
    lse_ptr,
    delta_ptr,
    base,
    lse_base,
    offs_n,
    start_m,
    end_m,
    scale: tl.constexpr,
    qk_scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    block_m: tl.constexpr,
    stage: tl.constexpr,
    needs_m_mask: tl.constexpr,
    needs_n_mask: tl.constexpr,
    fused_delta: tl.constexpr,
    prescale_k: tl.constexpr,
):
    offs_m = tl.arange(0, block_m)
    log2e = 1.4426950408889634
    for m in range(start_m, end_m, block_m):
        rows = m + offs_m
        if needs_m_mask:
            row_mask = rows < n_ctx
            q_tile = q_desc.load([m, 0])
            do_tile = do_desc.load([m, 0])
            lse_row = tl.load(lse_ptr + lse_base + rows, mask=row_mask, other=0.0)
        else:
            q_tile = q_desc.load([m, 0])
            do_tile = do_desc.load([m, 0])
            lse_row = tl.load(lse_ptr + lse_base + rows)
        if fused_delta:
            o_tile = o_desc.load([m, 0])
            d_row = tl.sum(o_tile.to(tl.float32) * do_tile.to(tl.float32), axis=1)
        elif needs_m_mask:
            d_row = tl.load(delta_ptr + lse_base + rows, mask=row_mask, other=0.0)
        else:
            d_row = tl.load(delta_ptr + lse_base + rows)

        if prescale_k:
            qk = tl.dot(q_tile, tl.trans(k_tile))
            p = tl.math.exp2(qk - lse_row[:, None] * log2e)
        else:
            qk = tl.dot(q_tile, tl.trans(k_tile)) * qk_scale
            p = tl.math.exp2(qk - lse_row[:, None] * log2e)

        if stage == 2 or needs_m_mask or needs_n_mask:
            if needs_m_mask and needs_n_mask:
                valid = row_mask[:, None] & (offs_n[None, :] < n_ctx)
            elif needs_m_mask:
                valid = tl.broadcast_to(row_mask[:, None], (block_m, offs_n.shape[0]))
            elif needs_n_mask:
                valid = tl.broadcast_to(
                    (offs_n < n_ctx)[None, :], (block_m, offs_n.shape[0])
                )
            else:
                valid = tl.full((block_m, offs_n.shape[0]), 1, tl.int1)
            if stage == 2:
                valid = valid & (offs_n[None, :] <= rows[:, None])
            p = tl.where(valid, p, 0.0)

        dv_acc += tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)
        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = (p * (dp - d_row[:, None])) * scale
        # ds inherits the masking via p (already zero where invalid)
        if prescale_k:
            dk_acc += tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * log2e
        else:
            dk_acc += tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile)
    return dk_acc, dv_acc


@triton.jit
def _bwd_dkdv_kernel(
    q,
    k,
    v,
    o,
    do,
    lse,
    delta,
    dk,
    dv,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    fused_delta: tl.constexpr,
    prescale_k: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_d = tl.arange(0, d_head)
    base = pid_bh * n_ctx * d_head
    lse_base = pid_bh * n_ctx
    qk_scale = scale * 1.4426950408889634
    q_desc = tl.make_tensor_descriptor(
        base=q + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    o_desc = tl.make_tensor_descriptor(
        base=o + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    do_desc = tl.make_tensor_descriptor(
        base=do + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )

    n_mask_needed = ((pid_n + 1) * block_n) > n_ctx
    k_desc = tl.make_tensor_descriptor(
        base=k + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )
    v_desc = tl.make_tensor_descriptor(
        base=v + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )
    k_tile = k_desc.load([pid_n * block_n, 0])
    v_tile = v_desc.load([pid_n * block_n, 0])

    dk_acc = tl.zeros((block_n, d_head), tl.float32)
    dv_acc = tl.zeros((block_n, d_head), tl.float32)

    if causal:
        # K-tile spans cols [pid_n*block_n, (pid_n+1)*block_n).
        # Q-rows below (pid_n+1)*block_n can interact (some columns are <= q_row).
        # Diagonal q-tiles: any that overlap the K-tile column range.
        # Off-diagonal q-tiles: rows where m_start >= (pid_n+1)*block_n (all k_cols <= q_row).
        k_col_start = pid_n * block_n
        k_col_end = (pid_n + 1) * block_n

        # Diagonal block: first q-tile is the one containing k_col_start.
        # In general k_col_start aligned to block_m boundary only if block_n | block_m or m_start coincides.
        # Process q-tiles overlapping [k_col_start, k_col_end) with stage=2.
        diag_start = (k_col_start // block_m) * block_m
        diag_end = ((k_col_end - 1) // block_m + 1) * block_m
        # Diagonal q-tiles need masks (causal + bounds).
        dk_acc, dv_acc = _bwd_dkdv_inner(
            dk_acc,
            dv_acc,
            k_tile,
            v_tile,
            q_desc,
            o_desc,
            do_desc,
            lse,
            delta,
            base,
            lse_base,
            offs_n,
            diag_start,
            tl.minimum(diag_end, n_ctx),
            scale,
            qk_scale,
            n_ctx,
            d_head,
            block_m,
            stage=2,
            needs_m_mask=True,
            needs_n_mask=n_mask_needed,
            fused_delta=fused_delta,
            prescale_k=prescale_k,
        )
        # Off-diagonal: interior q-tiles (unmasked) + tail q-tile (m-masked).
        off_interior_end = (n_ctx // block_m) * block_m
        off_start = tl.maximum(diag_end, 0)
        interior_lo = tl.maximum(off_start, 0)
        dk_acc, dv_acc = _bwd_dkdv_inner(
            dk_acc,
            dv_acc,
            k_tile,
            v_tile,
            q_desc,
            o_desc,
            do_desc,
            lse,
            delta,
            base,
            lse_base,
            offs_n,
            interior_lo,
            off_interior_end,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            block_m,
            stage=1,
            needs_m_mask=False,
            needs_n_mask=n_mask_needed,
            fused_delta=fused_delta,
            prescale_k=prescale_k,
        )
        dk_acc, dv_acc = _bwd_dkdv_inner(
            dk_acc,
            dv_acc,
            k_tile,
            v_tile,
            q_desc,
            o_desc,
            do_desc,
            lse,
            delta,
            base,
            lse_base,
            offs_n,
            tl.maximum(off_interior_end, off_start),
            n_ctx,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            block_m,
            stage=1,
            needs_m_mask=True,
            needs_n_mask=n_mask_needed,
            fused_delta=fused_delta,
            prescale_k=prescale_k,
        )
    else:
        interior_end = (n_ctx // block_m) * block_m
        dk_acc, dv_acc = _bwd_dkdv_inner(
            dk_acc,
            dv_acc,
            k_tile,
            v_tile,
            q_desc,
            o_desc,
            do_desc,
            lse,
            delta,
            base,
            lse_base,
            offs_n,
            0,
            interior_end,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            block_m,
            stage=0,
            needs_m_mask=False,
            needs_n_mask=n_mask_needed,
            fused_delta=fused_delta,
            prescale_k=prescale_k,
        )
        dk_acc, dv_acc = _bwd_dkdv_inner(
            dk_acc,
            dv_acc,
            k_tile,
            v_tile,
            q_desc,
            o_desc,
            do_desc,
            lse,
            delta,
            base,
            lse_base,
            offs_n,
            interior_end,
            n_ctx,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            block_m,
            stage=0,
            needs_m_mask=True,
            needs_n_mask=n_mask_needed,
            fused_delta=fused_delta,
            prescale_k=prescale_k,
        )

    out_mask = offs_n[:, None] < n_ctx
    if prescale_k:
        tl.store(
            dk + base + offs_n[:, None] * d_head + offs_d[None, :],
            (dk_acc * 1.4426950408889634).to(dk.dtype.element_ty),
            mask=out_mask,
        )
    else:
        tl.store(
            dk + base + offs_n[:, None] * d_head + offs_d[None, :],
            dk_acc.to(dk.dtype.element_ty),
            mask=out_mask,
        )
    tl.store(
        dv + base + offs_n[:, None] * d_head + offs_d[None, :],
        dv_acc.to(dv.dtype.element_ty),
        mask=out_mask,
    )


@triton.jit
def _bwd_dq_inner(
    dq_acc,
    q_tile,
    do_tile,
    lse_row,
    d_row,
    k_desc,
    v_desc,
    base,
    offs_m,
    start_n,
    end_n,
    scale: tl.constexpr,
    qk_scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    block_n: tl.constexpr,
    stage: tl.constexpr,
    needs_n_mask: tl.constexpr,
    row_mask_for_dot,
    needs_m_mask: tl.constexpr,
    prescale_k: tl.constexpr,
):
    offs_n = tl.arange(0, block_n)
    log2e = 1.4426950408889634
    for n in range(start_n, end_n, block_n):
        cols = n + offs_n
        if needs_n_mask:
            col_mask = cols < n_ctx
        k_tile = k_desc.load([n, 0])
        v_tile = v_desc.load([n, 0])

        if prescale_k:
            qk = tl.dot(q_tile, tl.trans(k_tile))
            p = tl.math.exp2(qk - lse_row[:, None] * log2e)
        else:
            qk = tl.dot(q_tile, tl.trans(k_tile)) * qk_scale
            p = tl.math.exp2(qk - lse_row[:, None] * log2e)

        if stage == 2 or needs_n_mask or needs_m_mask:
            if needs_m_mask and needs_n_mask:
                valid = row_mask_for_dot[:, None] & col_mask[None, :]
            elif needs_m_mask:
                valid = tl.broadcast_to(
                    row_mask_for_dot[:, None], (offs_m.shape[0], block_n)
                )
            elif needs_n_mask:
                valid = tl.broadcast_to(col_mask[None, :], (offs_m.shape[0], block_n))
            else:
                valid = tl.full((offs_m.shape[0], block_n), 1, tl.int1)
            if stage == 2:
                valid = valid & (cols[None, :] <= offs_m[:, None])
            p = tl.where(valid, p, 0.0)

        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = (p * (dp - d_row[:, None])) * scale
        if prescale_k:
            dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile) * log2e
        else:
            dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)
    return dq_acc


@triton.jit
def _bwd_dq_kernel(
    q,
    k,
    v,
    o,
    do,
    lse,
    delta,
    dq,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    fused_delta: tl.constexpr,
    prescale_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, d_head)
    base = pid_bh * n_ctx * d_head
    lse_base = pid_bh * n_ctx
    qk_scale = scale * 1.4426950408889634

    row_mask = offs_m < n_ctx
    needs_m_mask = (pid_m + 1) * block_m > n_ctx
    q_desc = tl.make_tensor_descriptor(
        base=q + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    o_desc = tl.make_tensor_descriptor(
        base=o + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    do_desc = tl.make_tensor_descriptor(
        base=do + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    k_desc = tl.make_tensor_descriptor(
        base=k + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )
    v_desc = tl.make_tensor_descriptor(
        base=v + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )
    q_tile = q_desc.load([pid_m * block_m, 0])
    do_tile = do_desc.load([pid_m * block_m, 0])
    lse_row = tl.load(lse + lse_base + offs_m, mask=row_mask, other=0.0)
    if fused_delta:
        o_tile = o_desc.load([pid_m * block_m, 0])
        d_row = tl.sum(o_tile.to(tl.float32) * do_tile.to(tl.float32), axis=1)
    else:
        d_row = tl.load(delta + lse_base + offs_m, mask=row_mask, other=0.0)

    dq_acc = tl.zeros((block_m, d_head), tl.float32)

    if causal:
        # Q-tile spans rows [pid_m*block_m, (pid_m+1)*block_m).
        # Off-diagonal K-tiles: cols < pid_m*block_m, aligned to block_n. No causal mask.
        # Diagonal range covers [off_end, (pid_m+1)*block_m), masked.
        q_row_start = pid_m * block_m
        off_end = (q_row_start // block_n) * block_n
        dq_acc = _bwd_dq_inner(
            dq_acc,
            q_tile,
            do_tile,
            lse_row,
            d_row,
            k_desc,
            v_desc,
            base,
            offs_m,
            0,
            off_end,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            block_n,
            stage=1,
            needs_n_mask=False,
            row_mask_for_dot=row_mask,
            needs_m_mask=needs_m_mask,
            prescale_k=prescale_k,
        )
        diag_end = tl.minimum(n_ctx, (pid_m + 1) * block_m)
        dq_acc = _bwd_dq_inner(
            dq_acc,
            q_tile,
            do_tile,
            lse_row,
            d_row,
            k_desc,
            v_desc,
            base,
            offs_m,
            off_end,
            diag_end,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            block_n,
            stage=2,
            needs_n_mask=True,
            row_mask_for_dot=row_mask,
            needs_m_mask=needs_m_mask,
            prescale_k=prescale_k,
        )
    else:
        interior_end = (n_ctx // block_n) * block_n
        dq_acc = _bwd_dq_inner(
            dq_acc,
            q_tile,
            do_tile,
            lse_row,
            d_row,
            k_desc,
            v_desc,
            base,
            offs_m,
            0,
            interior_end,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            block_n,
            stage=0,
            needs_n_mask=False,
            row_mask_for_dot=row_mask,
            needs_m_mask=needs_m_mask,
            prescale_k=prescale_k,
        )
        dq_acc = _bwd_dq_inner(
            dq_acc,
            q_tile,
            do_tile,
            lse_row,
            d_row,
            k_desc,
            v_desc,
            base,
            offs_m,
            interior_end,
            n_ctx,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            block_n,
            stage=0,
            needs_n_mask=True,
            row_mask_for_dot=row_mask,
            needs_m_mask=needs_m_mask,
            prescale_k=prescale_k,
        )

    tl.store(
        dq + base + offs_m[:, None] * d_head + offs_d[None, :],
        dq_acc.to(dq.dtype.element_ty),
        mask=row_mask[:, None],
    )


@triton.jit
def _bwd_dq_persistent_kernel(
    q,
    k,
    v,
    o,
    do,
    lse,
    delta,
    dq,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    fused_delta: tl.constexpr,
    n_tiles_m: tl.constexpr,
    tile_stride: tl.constexpr,
    prescale_k: tl.constexpr,
):
    tile_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_d = tl.arange(0, d_head)
    base = pid_bh * n_ctx * d_head
    lse_base = pid_bh * n_ctx
    qk_scale = scale * 1.4426950408889634

    q_desc = tl.make_tensor_descriptor(
        base=q + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    o_desc = tl.make_tensor_descriptor(
        base=o + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    do_desc = tl.make_tensor_descriptor(
        base=do + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    k_desc = tl.make_tensor_descriptor(
        base=k + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )
    v_desc = tl.make_tensor_descriptor(
        base=v + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )

    while tile_m < n_tiles_m:
        offs_m = tile_m * block_m + tl.arange(0, block_m)
        row_mask = offs_m < n_ctx
        needs_m_mask = (tile_m + 1) * block_m > n_ctx

        q_tile = q_desc.load([tile_m * block_m, 0])
        do_tile = do_desc.load([tile_m * block_m, 0])
        lse_row = tl.load(lse + lse_base + offs_m, mask=row_mask, other=0.0)
        if fused_delta:
            o_tile = o_desc.load([tile_m * block_m, 0])
            d_row = tl.sum(o_tile.to(tl.float32) * do_tile.to(tl.float32), axis=1)
        else:
            d_row = tl.load(delta + lse_base + offs_m, mask=row_mask, other=0.0)

        dq_acc = tl.zeros((block_m, d_head), tl.float32)

        if causal:
            q_row_start = tile_m * block_m
            off_end = (q_row_start // block_n) * block_n
            dq_acc = _bwd_dq_inner(
                dq_acc,
                q_tile,
                do_tile,
                lse_row,
                d_row,
                k_desc,
                v_desc,
                base,
                offs_m,
                0,
                off_end,
                scale,
                qk_scale,
                n_ctx,
                d_head,
                block_n,
                stage=1,
                needs_n_mask=False,
                row_mask_for_dot=row_mask,
                needs_m_mask=needs_m_mask,
                prescale_k=prescale_k,
            )
            diag_end = tl.minimum(n_ctx, (tile_m + 1) * block_m)
            dq_acc = _bwd_dq_inner(
                dq_acc,
                q_tile,
                do_tile,
                lse_row,
                d_row,
                k_desc,
                v_desc,
                base,
                offs_m,
                off_end,
                diag_end,
                scale,
                qk_scale,
                n_ctx,
                d_head,
                block_n,
                stage=2,
                needs_n_mask=True,
                row_mask_for_dot=row_mask,
                needs_m_mask=needs_m_mask,
                prescale_k=prescale_k,
            )
        else:
            interior_end = (n_ctx // block_n) * block_n
            dq_acc = _bwd_dq_inner(
                dq_acc,
                q_tile,
                do_tile,
                lse_row,
                d_row,
                k_desc,
                v_desc,
                base,
                offs_m,
                0,
                interior_end,
                scale,
                qk_scale,
                n_ctx,
                d_head,
                block_n,
                stage=0,
                needs_n_mask=False,
                row_mask_for_dot=row_mask,
                needs_m_mask=needs_m_mask,
                prescale_k=prescale_k,
            )
            dq_acc = _bwd_dq_inner(
                dq_acc,
                q_tile,
                do_tile,
                lse_row,
                d_row,
                k_desc,
                v_desc,
                base,
                offs_m,
                interior_end,
                n_ctx,
                scale,
                qk_scale,
                n_ctx,
                d_head,
                block_n,
                stage=0,
                needs_n_mask=True,
                row_mask_for_dot=row_mask,
                needs_m_mask=needs_m_mask,
                prescale_k=prescale_k,
            )

        tl.store(
            dq + base + offs_m[:, None] * d_head + offs_d[None, :],
            dq_acc.to(dq.dtype.element_ty),
            mask=row_mask[:, None],
        )
        tile_m += tile_stride


@triton.jit
def _bwd_combined_dkdv_inner(
    dk,
    dv,
    q,
    k_tile,
    v_tile,
    do,
    lse,
    delta,
    base,
    lse_base,
    start_n,
    start_m,
    num_steps,
    scale: tl.constexpr,
    qk_scale: tl.constexpr,
    d_head: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    mask: tl.constexpr,
    prescale_k: tl.constexpr,
):
    offs_n = start_n + tl.arange(0, block_n)
    offs_d = tl.arange(0, d_head)
    offs_m = start_m + tl.arange(0, block_m)
    q_t_ptrs = q + base + offs_d[:, None] + offs_m[None, :] * d_head
    do_ptrs = do + base + offs_m[:, None] * d_head + offs_d[None, :]
    for _ in range(num_steps):
        q_t = tl.load(q_t_ptrs)
        do_tile = tl.load(do_ptrs)
        lse_row = tl.load(lse + lse_base + offs_m)
        d_row = tl.load(delta + lse_base + offs_m)

        if prescale_k:
            k_scaled = (k_tile * qk_scale).to(k_tile.dtype)
            qk_t = tl.dot(k_scaled, q_t)
        else:
            qk_t = tl.dot(k_tile, q_t) * qk_scale
        p_t = tl.math.exp2(qk_t - lse_row[None, :] * 1.4426950408889634)
        if mask:
            p_t = tl.where(offs_m[None, :] >= offs_n[:, None], p_t, 0.0)

        dv += tl.dot(p_t.to(do_tile.dtype), do_tile)
        dp_t = tl.dot(v_tile, tl.trans(do_tile))
        ds_t = p_t * (dp_t - d_row[None, :])
        if not prescale_k:
            ds_t *= scale
        dk += tl.dot(ds_t.to(q_t.dtype), tl.trans(q_t))

        offs_m += block_m
        q_t_ptrs += block_m * d_head
        do_ptrs += block_m * d_head
    return dk, dv


@triton.jit
def _bwd_combined_dq_inner(
    dq,
    q_tile,
    do_tile,
    k,
    v,
    lse_row,
    d_row,
    base,
    start_m,
    start_n,
    num_steps,
    scale: tl.constexpr,
    qk_scale: tl.constexpr,
    d_head: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    mask: tl.constexpr,
    prescale_k: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, block_m)
    offs_n = start_n + tl.arange(0, block_n)
    offs_d = tl.arange(0, d_head)
    k_t_ptrs = k + base + offs_n[None, :] * d_head + offs_d[:, None]
    v_t_ptrs = v + base + offs_n[None, :] * d_head + offs_d[:, None]
    for _ in range(num_steps):
        k_t = tl.load(k_t_ptrs)
        v_t = tl.load(v_t_ptrs)
        if prescale_k:
            k_t_scaled = (k_t * qk_scale).to(k_t.dtype)
            qk = tl.dot(q_tile, k_t_scaled)
        else:
            qk = tl.dot(q_tile, k_t) * qk_scale
        p = tl.math.exp2(qk - lse_row[:, None] * 1.4426950408889634)
        if mask:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

        dp = tl.dot(do_tile, v_t)
        ds = p * (dp - d_row[:, None])
        if prescale_k:
            dq += tl.dot(ds.to(k_t_scaled.dtype), tl.trans(k_t_scaled))
        else:
            ds *= scale
            dq += tl.dot(ds.to(k_t.dtype), tl.trans(k_t))

        offs_n += block_n
        k_t_ptrs += block_n * d_head
        v_t_ptrs += block_n * d_head
    return dq


@triton.jit
def _bwd_combined_kernel(
    q,
    k,
    v,
    do,
    lse,
    delta,
    dq,
    dk,
    dv,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    causal: tl.constexpr,
    block_m1: tl.constexpr,
    block_n1: tl.constexpr,
    block_m2: tl.constexpr,
    block_n2: tl.constexpr,
    blk_slice_factor: tl.constexpr,
    prescale_k: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = pid_bh * n_ctx * d_head
    lse_base = pid_bh * n_ctx
    offs_d = tl.arange(0, d_head)
    qk_scale = scale * 1.4426950408889634

    start_n = pid * block_n1
    offs_n = start_n + tl.arange(0, block_n1)
    k_tile = tl.load(k + base + offs_n[:, None] * d_head + offs_d[None, :])
    v_tile = tl.load(v + base + offs_n[:, None] * d_head + offs_d[None, :])
    dk_acc = tl.zeros((block_n1, d_head), tl.float32)
    dv_acc = tl.zeros((block_n1, d_head), tl.float32)

    start_m = 0
    mask_block_m1: tl.constexpr = block_m1 // blk_slice_factor
    if causal:
        start_m = start_n
        num_steps = block_n1 // mask_block_m1
        dk_acc, dv_acc = _bwd_combined_dkdv_inner(
            dk_acc,
            dv_acc,
            q,
            k_tile,
            v_tile,
            do,
            lse,
            delta,
            base,
            lse_base,
            start_n,
            start_m,
            num_steps,
            scale,
            qk_scale,
            d_head,
            mask_block_m1,
            block_n1,
            mask=True,
            prescale_k=prescale_k,
        )
        start_m += num_steps * mask_block_m1

    num_steps = (n_ctx - start_m) // block_m1
    dk_acc, dv_acc = _bwd_combined_dkdv_inner(
        dk_acc,
        dv_acc,
        q,
        k_tile,
        v_tile,
        do,
        lse,
        delta,
        base,
        lse_base,
        start_n,
        start_m,
        num_steps,
        scale,
        qk_scale,
        d_head,
        block_m1,
        block_n1,
        mask=False,
        prescale_k=prescale_k,
    )

    if prescale_k:
        tl.store(
            dk + base + offs_n[:, None] * d_head + offs_d[None, :],
            (dk_acc * scale).to(dk.dtype.element_ty),
        )
    else:
        tl.store(
            dk + base + offs_n[:, None] * d_head + offs_d[None, :],
            dk_acc.to(dk.dtype.element_ty),
        )
    tl.store(
        dv + base + offs_n[:, None] * d_head + offs_d[None, :],
        dv_acc.to(dv.dtype.element_ty),
    )

    mask_block_n2: tl.constexpr = block_n2 // blk_slice_factor
    for dq_offset in range(0, block_n1, block_m2):
        start_m2 = pid * block_n1 + dq_offset
        offs_m2 = start_m2 + tl.arange(0, block_m2)
        q_tile = tl.load(q + base + offs_m2[:, None] * d_head + offs_d[None, :])
        do_tile = tl.load(do + base + offs_m2[:, None] * d_head + offs_d[None, :])
        lse_row = tl.load(lse + lse_base + offs_m2)
        d_row = tl.load(delta + lse_base + offs_m2)
        dq_acc = tl.zeros((block_m2, d_head), tl.float32)

        start_n2 = 0
        num_steps2 = n_ctx // block_n2
        if causal:
            end_n = start_m2 + block_m2
            num_mask_steps = block_m2 // mask_block_n2
            dq_acc = _bwd_combined_dq_inner(
                dq_acc,
                q_tile,
                do_tile,
                k,
                v,
                lse_row,
                d_row,
                base,
                start_m2,
                end_n - num_mask_steps * mask_block_n2,
                num_mask_steps,
                scale,
                qk_scale,
                d_head,
                block_m2,
                mask_block_n2,
                mask=True,
                prescale_k=prescale_k,
            )
            end_n -= num_mask_steps * mask_block_n2
            num_steps2 = end_n // block_n2
            start_n2 = end_n - num_steps2 * block_n2

        dq_acc = _bwd_combined_dq_inner(
            dq_acc,
            q_tile,
            do_tile,
            k,
            v,
            lse_row,
            d_row,
            base,
            start_m2,
            start_n2,
            num_steps2,
            scale,
            qk_scale,
            d_head,
            block_m2,
            block_n2,
            mask=False,
            prescale_k=prescale_k,
        )

        if prescale_k:
            tl.store(
                dq + base + offs_m2[:, None] * d_head + offs_d[None, :],
                (dq_acc * 0.6931471805599453).to(dq.dtype.element_ty),
            )
        else:
            tl.store(
                dq + base + offs_m2[:, None] * d_head + offs_d[None, :],
                dq_acc.to(dq.dtype.element_ty),
            )


# ---------------------------------------------------------------------------
# Split-K dK/dV: parallel K/V-tile accumulation for unbalanced causal workloads
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_dkdv_split_k_inner(
    dk_acc,
    dv_acc,
    k_tile,
    v_tile,
    q_desc,
    o_desc,
    do_desc,
    lse_ptr,
    delta_ptr,
    base,
    lse_base,
    offs_n,
    start_m,
    end_m,
    scale: tl.constexpr,
    qk_scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    block_m: tl.constexpr,
    stage: tl.constexpr,
    needs_m_mask: tl.constexpr,
    needs_n_mask: tl.constexpr,
    fused_delta: tl.constexpr,
    prescale_k: tl.constexpr,
    n_splits: tl.constexpr,
    split_id: tl.constexpr,
    split_block_m: tl.constexpr,
):
    offs_m = tl.arange(0, block_m)
    log2e = 1.4426950408889634
    for m in range(start_m, end_m, split_block_m):
        rows = m + tl.arange(0, split_block_m)
        row_mask = rows < n_ctx
        q_tile = q_desc.load([m, 0])
        do_tile = do_desc.load([m, 0])
        lse_row = tl.load(lse_ptr + lse_base + rows, mask=row_mask, other=0.0)
        if fused_delta:
            o_tile = o_desc.load([m, 0])
            d_row = tl.sum(o_tile.to(tl.float32) * do_tile.to(tl.float32), axis=1)
        else:
            d_row = tl.load(delta_ptr + lse_base + rows, mask=row_mask, other=0.0)

        if prescale_k:
            qk = tl.dot(q_tile, tl.trans(k_tile))
            p = tl.math.exp2(qk - lse_row[:, None] * log2e)
        else:
            qk = tl.dot(q_tile, tl.trans(k_tile)) * qk_scale
            p = tl.math.exp2(qk - lse_row[:, None] * log2e)

        if stage == 2 or needs_m_mask or needs_n_mask:
            if needs_m_mask and needs_n_mask:
                valid = row_mask[:, None] & (offs_n[None, :] < n_ctx)
            elif needs_m_mask:
                valid = tl.broadcast_to(
                    row_mask[:, None], (split_block_m, offs_n.shape[0])
                )
            elif needs_n_mask:
                valid = tl.broadcast_to(
                    (offs_n < n_ctx)[None, :], (split_block_m, offs_n.shape[0])
                )
            else:
                valid = tl.full((split_block_m, offs_n.shape[0]), 1, tl.int1)
            if stage == 2:
                valid = valid & (offs_n[None, :] <= rows[:, None])
            p = tl.where(valid, p, 0.0)

        dv_acc += tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)
        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = (p * (dp - d_row[:, None])) * scale
        if prescale_k:
            dk_acc += tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile) * log2e
        else:
            dk_acc += tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile)
    return dk_acc, dv_acc


@triton.jit
def _bwd_dkdv_split_k_kernel(
    q,
    k,
    v,
    o,
    do,
    lse,
    delta,
    dk,
    dv,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    causal: tl.constexpr,
    block_n: tl.constexpr,
    split_block_m: tl.constexpr,
    n_splits: tl.constexpr,
    fused_delta: tl.constexpr,
    prescale_k: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    split_id = tl.program_id(2)

    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_d = tl.arange(0, d_head)
    base = pid_bh * n_ctx * d_head
    lse_base = pid_bh * n_ctx
    qk_scale = scale * 1.4426950408889634

    q_desc = tl.make_tensor_descriptor(
        base=q + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[split_block_m, d_head],
    )
    o_desc = tl.make_tensor_descriptor(
        base=o + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[split_block_m, d_head],
    )
    do_desc = tl.make_tensor_descriptor(
        base=do + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[split_block_m, d_head],
    )

    n_mask_needed = ((pid_n + 1) * block_n) > n_ctx
    k_desc = tl.make_tensor_descriptor(
        base=k + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )
    v_desc = tl.make_tensor_descriptor(
        base=v + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )
    k_tile = k_desc.load([pid_n * block_n, 0])
    v_tile = v_desc.load([pid_n * block_n, 0])

    dk_acc = tl.zeros((block_n, d_head), tl.float32)
    dv_acc = tl.zeros((block_n, d_head), tl.float32)

    if causal:
        k_col_start = pid_n * block_n
        k_col_end = (pid_n + 1) * block_n
        diag_start = (k_col_start // split_block_m) * split_block_m
        diag_end = ((k_col_end - 1) // split_block_m + 1) * split_block_m

        dk_acc, dv_acc = _bwd_dkdv_split_k_inner(
            dk_acc,
            dv_acc,
            k_tile,
            v_tile,
            q_desc,
            o_desc,
            do_desc,
            lse,
            delta,
            base,
            lse_base,
            offs_n,
            diag_start,
            tl.minimum(diag_end, n_ctx),
            scale,
            qk_scale,
            n_ctx,
            d_head,
            split_block_m,
            stage=2,
            needs_m_mask=True,
            needs_n_mask=n_mask_needed,
            fused_delta=fused_delta,
            prescale_k=prescale_k,
            n_splits=n_splits,
            split_id=split_id,
            split_block_m=split_block_m,
        )
        off_interior_end = (n_ctx // split_block_m) * split_block_m
        off_start = tl.maximum(diag_end, 0)
        dk_acc, dv_acc = _bwd_dkdv_split_k_inner(
            dk_acc,
            dv_acc,
            k_tile,
            v_tile,
            q_desc,
            o_desc,
            do_desc,
            lse,
            delta,
            base,
            lse_base,
            offs_n,
            tl.maximum(off_interior_end, off_start),
            off_interior_end,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            split_block_m,
            stage=1,
            needs_m_mask=False,
            needs_n_mask=n_mask_needed,
            fused_delta=fused_delta,
            prescale_k=prescale_k,
            n_splits=n_splits,
            split_id=split_id,
            split_block_m=split_block_m,
        )
        dk_acc, dv_acc = _bwd_dkdv_split_k_inner(
            dk_acc,
            dv_acc,
            k_tile,
            v_tile,
            q_desc,
            o_desc,
            do_desc,
            lse,
            delta,
            base,
            lse_base,
            offs_n,
            off_interior_end,
            n_ctx,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            split_block_m,
            stage=1,
            needs_m_mask=True,
            needs_n_mask=n_mask_needed,
            fused_delta=fused_delta,
            prescale_k=prescale_k,
            n_splits=n_splits,
            split_id=split_id,
            split_block_m=split_block_m,
        )
    else:
        interior_end = (n_ctx // split_block_m) * split_block_m
        dk_acc, dv_acc = _bwd_dkdv_split_k_inner(
            dk_acc,
            dv_acc,
            k_tile,
            v_tile,
            q_desc,
            o_desc,
            do_desc,
            lse,
            delta,
            base,
            lse_base,
            offs_n,
            0,
            interior_end,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            split_block_m,
            stage=0,
            needs_m_mask=False,
            needs_n_mask=n_mask_needed,
            fused_delta=fused_delta,
            prescale_k=prescale_k,
            n_splits=n_splits,
            split_id=split_id,
            split_block_m=split_block_m,
        )
        dk_acc, dv_acc = _bwd_dkdv_split_k_inner(
            dk_acc,
            dv_acc,
            k_tile,
            v_tile,
            q_desc,
            o_desc,
            do_desc,
            lse,
            delta,
            base,
            lse_base,
            offs_n,
            interior_end,
            n_ctx,
            scale,
            qk_scale,
            n_ctx,
            d_head,
            split_block_m,
            stage=0,
            needs_m_mask=True,
            needs_n_mask=n_mask_needed,
            fused_delta=fused_delta,
            prescale_k=prescale_k,
            n_splits=n_splits,
            split_id=split_id,
            split_block_m=split_block_m,
        )

    out_mask = offs_n[:, None] < n_ctx
    if prescale_k:
        tl.store(
            dk + base + offs_n[:, None] * d_head + offs_d[None, :],
            (dk_acc * 1.4426950408889634).to(dk.dtype.element_ty),
            mask=out_mask,
        )
    else:
        tl.store(
            dk + base + offs_n[:, None] * d_head + offs_d[None, :],
            dk_acc.to(dk.dtype.element_ty),
            mask=out_mask,
        )
    tl.store(
        dv + base + offs_n[:, None] * d_head + offs_d[None, :],
        dv_acc.to(dv.dtype.element_ty),
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Persistent dK/dV kernel
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_dkdv_persistent_kernel(
    q,
    k,
    v,
    o,
    do,
    lse,
    delta,
    dk,
    dv,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    n_tiles_n: tl.constexpr,
    fused_delta: tl.constexpr,
    prescale_k: tl.constexpr,
):
    tile_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    log2e = 1.4426950408889634
    qk_scale = scale * log2e
    base = pid_bh * n_ctx * d_head
    lse_base = pid_bh * n_ctx
    offs_d = tl.arange(0, d_head)

    q_desc = tl.make_tensor_descriptor(
        base=q + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    o_desc = tl.make_tensor_descriptor(
        base=o + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    do_desc = tl.make_tensor_descriptor(
        base=do + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    k_desc = tl.make_tensor_descriptor(
        base=k + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )
    v_desc = tl.make_tensor_descriptor(
        base=v + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )

    while tile_n < n_tiles_n:
        offs_n = tile_n * block_n + tl.arange(0, block_n)
        n_mask_needed = ((tile_n + 1) * block_n) > n_ctx
        k_tile = k_desc.load([tile_n * block_n, 0])
        v_tile = v_desc.load([tile_n * block_n, 0])

        dk_acc = tl.zeros((block_n, d_head), tl.float32)
        dv_acc = tl.zeros((block_n, d_head), tl.float32)

        if causal:
            k_col_start = tile_n * block_n
            k_col_end = (tile_n + 1) * block_n
            diag_start = (k_col_start // block_m) * block_m
            diag_end = ((k_col_end - 1) // block_m + 1) * block_m

            dk_acc, dv_acc = _bwd_dkdv_inner(
                dk_acc,
                dv_acc,
                k_tile,
                v_tile,
                q_desc,
                o_desc,
                do_desc,
                lse,
                delta,
                base,
                lse_base,
                offs_n,
                diag_start,
                tl.minimum(diag_end, n_ctx),
                scale,
                qk_scale,
                n_ctx,
                d_head,
                block_m,
                stage=2,
                needs_m_mask=True,
                needs_n_mask=n_mask_needed,
                fused_delta=fused_delta,
                prescale_k=prescale_k,
            )
            off_interior_end = (n_ctx // block_m) * block_m
            off_start = tl.maximum(diag_end, 0)
            dk_acc, dv_acc = _bwd_dkdv_inner(
                dk_acc,
                dv_acc,
                k_tile,
                v_tile,
                q_desc,
                o_desc,
                do_desc,
                lse,
                delta,
                base,
                lse_base,
                offs_n,
                tl.maximum(off_interior_end, off_start),
                off_interior_end,
                scale,
                qk_scale,
                n_ctx,
                d_head,
                block_m,
                stage=1,
                needs_m_mask=False,
                needs_n_mask=n_mask_needed,
                fused_delta=fused_delta,
                prescale_k=prescale_k,
            )
            dk_acc, dv_acc = _bwd_dkdv_inner(
                dk_acc,
                dv_acc,
                k_tile,
                v_tile,
                q_desc,
                o_desc,
                do_desc,
                lse,
                delta,
                base,
                lse_base,
                offs_n,
                off_interior_end,
                n_ctx,
                scale,
                qk_scale,
                n_ctx,
                d_head,
                block_m,
                stage=1,
                needs_m_mask=True,
                needs_n_mask=n_mask_needed,
                fused_delta=fused_delta,
                prescale_k=prescale_k,
            )
        else:
            interior_end = (n_ctx // block_m) * block_m
            dk_acc, dv_acc = _bwd_dkdv_inner(
                dk_acc,
                dv_acc,
                k_tile,
                v_tile,
                q_desc,
                o_desc,
                do_desc,
                lse,
                delta,
                base,
                lse_base,
                offs_n,
                0,
                interior_end,
                scale,
                qk_scale,
                n_ctx,
                d_head,
                block_m,
                stage=0,
                needs_m_mask=False,
                needs_n_mask=n_mask_needed,
                fused_delta=fused_delta,
                prescale_k=prescale_k,
            )
            dk_acc, dv_acc = _bwd_dkdv_inner(
                dk_acc,
                dv_acc,
                k_tile,
                v_tile,
                q_desc,
                o_desc,
                do_desc,
                lse,
                delta,
                base,
                lse_base,
                offs_n,
                interior_end,
                n_ctx,
                scale,
                qk_scale,
                n_ctx,
                d_head,
                block_m,
                stage=0,
                needs_m_mask=True,
                needs_n_mask=n_mask_needed,
                fused_delta=fused_delta,
                prescale_k=prescale_k,
            )

        out_mask = offs_n[:, None] < n_ctx
        tl.store(
            dk + base + offs_n[:, None] * d_head + offs_d[None, :],
            dk_acc.to(dk.dtype.element_ty),
            mask=out_mask,
        )
        tl.store(
            dv + base + offs_n[:, None] * d_head + offs_d[None, :],
            dv_acc.to(dv.dtype.element_ty),
            mask=out_mask,
        )
        tile_n += tl.num_programs(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_ada_or_newer(q):
    major, minor = torch.cuda.get_device_capability(q.device)
    return major > 8 or (major == 8 and minor >= 9)


def _split_k_config(q, n_ctx, d_head, causal, fused_delta):
    if not _is_ada_or_newer(q):
        return None, None
    # Shared memory budget: max 80KB to leave room for other buffers
    max_sm = 80 * 1024
    block_n = 64 if d_head <= 64 else 128
    sm_per_tile = block_n * d_head * 8  # dk_acc + dv_acc float32
    if sm_per_tile > max_sm:
        return None, None

    n_tiles_n = triton.cdiv(n_ctx, block_n)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count

    # For large non-causal or large causal, split-K helps
    if n_ctx < 2048:
        return None, None

    # For causal, diagonal work dominates; for non-causal uniform
    if causal:
        # Only split for large sequences where diagonal bottleneck matters
        ideal_splits = min(4, sm_count // max(n_tiles_n, 1))
    else:
        ideal_splits = min(4, sm_count // max(n_tiles_n, 1))

    if ideal_splits >= 2:
        # Use conservative block_m for split-k to keep SM usage low
        split_block_m = 32  # small blocks to fit shared memory
        return ideal_splits, split_block_m
    return None, None


def _use_persistent_dkdv(q, n_ctx, d_head, causal):
    if not _is_ada_or_newer(q):
        return False
    if n_ctx < 2048:
        return False
    # Shared memory check: each K-tile uses block_n * d_head * 8 bytes
    block_n = 64 if d_head <= 64 else 64  # use smaller block_n to save SM
    sm_per_tile = block_n * d_head * 8
    if sm_per_tile > 80 * 1024:
        return False
    n_tiles_n = triton.cdiv(n_ctx, block_n)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    return n_tiles_n < sm_count * 2 and n_tiles_n >= 4


# NOTE: prescale_k (the trailing config field) is only correctly implemented in
# the combined backward kernel (``_bwd_combined_*_inner``), which prescales the
# K tile and compensates accordingly. The standalone dK/dV and dQ kernels load
# K/Q raw and the prescale_k branch in their inner loops omits the qk_scale
# multiply, which yields non-finite gradients. Until those kernels actually
# prescale their tiles, the standalone configs must request prescale_k=False.
def _select_bwd_dkdv_config(q, n_ctx, d_head, causal, fused_delta):
    if _is_ada_or_newer(q):
        if d_head <= 32:
            return 128, 128, 4, 2 if fused_delta else 4, False
        if d_head == 64:
            if causal:
                return 128, 64, 4, 2 if fused_delta else 3, False
            if n_ctx >= 1024:
                return 64, 128, 4, 2 if fused_delta else 3, False
            return 128, 64, 4, 2 if fused_delta else 3, False
        if d_head == 128:
            return 64, 64, 4, 2, False
    if d_head <= 64:
        return 128, 64, 4, 3, False
    return 64, 64, 4, 3, False


def _select_bwd_dq_config(q, n_ctx, d_head, causal, fused_delta):
    if _is_ada_or_newer(q):
        if d_head <= 32:
            return 128, 128, 4, 2 if fused_delta else 4, False
        if d_head == 64:
            if causal:
                return 128, 64, 4, 3, False
            if n_ctx >= 1024:
                return 128, 128, 4, 2, False
            return 64, 128, 4, 3, False
        if d_head == 128:
            if causal:
                return 32, 64, 4, 2, False
            return 128, 32, 4, 2, False
    if d_head <= 64:
        return 128, 64, 4, 3, False
    return 64, 64, 4, 3, False


def _use_fused_delta(n_ctx, d_head, causal):
    # Fusing D = sum(O * dO) removes one launch and global delta traffic. On Ada it
    # helps launch-bound cases, but large tiles prefer the precomputed delta path.
    return (d_head == 64 and n_ctx <= 512) or (
        d_head == 128 and n_ctx <= 512 and causal
    )


def _use_persistent_dq(q, n_ctx, d_head, causal):
    return (
        _is_ada_or_newer(q)
        and n_ctx >= 4096
        and d_head == 128
        and not causal
        and os.environ.get("CUSTOM_ATT_DISABLE_PERSISTENT_DQ") != "1"
        and torch.cuda.get_device_properties(q.device).multi_processor_count > 0
    )


def _combined_bwd_config(q, n_ctx, d_head, causal, fused_delta):
    if not (
        _is_ada_or_newer(q)
        and not fused_delta
        and d_head in (64, 128)
        and n_ctx >= 2048
        and os.environ.get("CUSTOM_ATT_DISABLE_COMBINED_BWD") != "1"
    ):
        return None
    if d_head == 64:
        return 32, 128, 64, 64, 2, 4, 3, True
    return 32, 128, 128, 32, 2, 8, 2, True


def _use_combined_bwd(q, n_ctx, d_head, causal, fused_delta):
    return _combined_bwd_config(q, n_ctx, d_head, causal, fused_delta) is not None


def triton_flash_bwd(grad_out, q, k, v, out, lse, causal=False, softmax_scale=None):
    assert grad_out.is_contiguous() and q.is_contiguous() and k.is_contiguous()
    assert v.is_contiguous() and out.is_contiguous() and lse.is_contiguous()
    n_ctx = q.shape[-2]
    d_head = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else d_head**-0.5

    bh = q.shape[0] * q.shape[1]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    fused_delta = _use_fused_delta(n_ctx, d_head, bool(causal))
    if fused_delta:
        delta = out
    else:
        delta = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        pre_block = 256 if d_head <= 128 else 128
        _bwd_preprocess[(triton.cdiv(n_ctx, pre_block), bh)](
            out,
            grad_out,
            delta,
            n_ctx,
            d_head,
            pre_block,
            num_warps=8 if pre_block == 256 else 4,
            num_stages=2,
        )

    combined_config = _combined_bwd_config(q, n_ctx, d_head, bool(causal), fused_delta)
    if combined_config is not None:
        (
            block_m1,
            block_n1,
            block_m2,
            block_n2,
            slice_factor,
            warps,
            stages,
            prescale_k,
        ) = combined_config
        _bwd_combined_kernel[(n_ctx // block_n1, bh)](
            q,
            k,
            v,
            grad_out,
            lse,
            delta,
            dq,
            dk,
            dv,
            scale,
            n_ctx,
            d_head,
            causal,
            block_m1,
            block_n1,
            block_m2,
            block_n2,
            slice_factor,
            prescale_k,
            num_warps=warps,
            num_stages=stages,
        )
        return dq, dk, dv

    dkdv_block_m, dkdv_block_n, dkdv_warps, dkdv_stages, dkdv_prescale_k = (
        _select_bwd_dkdv_config(q, n_ctx, d_head, bool(causal), fused_delta)
    )

    # Split-K dK/dV: unbalanced causal workloads benefit from parallel Q-tile splits
    n_splits, split_block_m = _split_k_config(
        q, n_ctx, d_head, bool(causal), fused_delta
    )
    if n_splits is not None and n_splits >= 2:
        grid = (triton.cdiv(n_ctx, dkdv_block_n), bh, n_splits)
        _bwd_dkdv_split_k_kernel[grid](
            q,
            k,
            v,
            out,
            grad_out,
            lse,
            delta,
            dk,
            dv,
            scale,
            n_ctx,
            d_head,
            causal,
            dkdv_block_n,
            split_block_m,
            n_splits,
            fused_delta,
            dkdv_prescale_k,
            num_warps=dkdv_warps,
            num_stages=dkdv_stages,
        )
    # Persistent dK/dV: fills SMs when tile count < 2x SMs
    elif _use_persistent_dkdv(q, n_ctx, d_head, bool(causal)):
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        n_tiles_n = triton.cdiv(n_ctx, dkdv_block_n)
        n_persistent = min(n_tiles_n, sm_count * 2)
        _bwd_dkdv_persistent_kernel[(n_persistent, bh)](
            q,
            k,
            v,
            out,
            grad_out,
            lse,
            delta,
            dk,
            dv,
            scale,
            n_ctx,
            d_head,
            causal,
            dkdv_block_m,
            dkdv_block_n,
            n_tiles_n,
            fused_delta,
            dkdv_prescale_k,
            num_warps=dkdv_warps,
            num_stages=dkdv_stages,
        )
    else:
        _bwd_dkdv_kernel[(triton.cdiv(n_ctx, dkdv_block_n), bh)](
            q,
            k,
            v,
            out,
            grad_out,
            lse,
            delta,
            dk,
            dv,
            scale,
            n_ctx,
            d_head,
            causal,
            dkdv_block_m,
            dkdv_block_n,
            fused_delta,
            dkdv_prescale_k,
            num_warps=dkdv_warps,
            num_stages=dkdv_stages,
        )

    dq_block_m, dq_block_n, dq_warps, dq_stages, dq_prescale_k = _select_bwd_dq_config(
        q, n_ctx, d_head, bool(causal), fused_delta
    )

    dq_tiles = triton.cdiv(n_ctx, dq_block_m)
    if _use_persistent_dq(q, n_ctx, d_head, bool(causal)):
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        persistent_tiles = min(dq_tiles, sm_count * 2)
        _bwd_dq_persistent_kernel[(persistent_tiles, bh)](
            q,
            k,
            v,
            out,
            grad_out,
            lse,
            delta,
            dq,
            scale,
            n_ctx,
            d_head,
            causal,
            dq_block_m,
            dq_block_n,
            fused_delta,
            dq_tiles,
            persistent_tiles,
            num_warps=dq_warps,
            num_stages=dq_stages,
            prescale_k=dq_prescale_k,
        )
    else:
        _bwd_dq_kernel[(dq_tiles, bh)](
            q,
            k,
            v,
            out,
            grad_out,
            lse,
            delta,
            dq,
            scale,
            n_ctx,
            d_head,
            causal,
            dq_block_m,
            dq_block_n,
            fused_delta,
            dq_prescale_k,
            num_warps=dq_warps,
            num_stages=dq_stages,
        )

    return dq, dk, dv


def can_use_triton_bwd(q):
    return (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (32, 64, 128)
    )
