import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_preprocess(
    o, do, delta,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    block_m: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, d_head)
    base = pid_bh * n_ctx * d_head

    o_tile = tl.load(
        o + base + offs_m[:, None] * d_head + offs_d[None, :],
        mask=offs_m[:, None] < n_ctx, other=0.0,
    ).to(tl.float32)
    do_tile = tl.load(
        do + base + offs_m[:, None] * d_head + offs_d[None, :],
        mask=offs_m[:, None] < n_ctx, other=0.0,
    ).to(tl.float32)
    d = tl.sum(o_tile * do_tile, axis=1)
    tl.store(delta + pid_bh * n_ctx + offs_m, d, mask=offs_m < n_ctx)


# stage: 0 = non-causal, 1 = causal off-diagonal (no mask), 2 = causal diagonal (mask)
@triton.jit
def _bwd_dkdv_inner(
    dk_acc, dv_acc,
    k_tile, v_tile,
    q_ptr, do_ptr, lse_ptr, delta_ptr,
    base, lse_base, offs_n, offs_d,
    start_m, end_m,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    block_m: tl.constexpr,
    stage: tl.constexpr,
    needs_m_mask: tl.constexpr,
    needs_n_mask: tl.constexpr,
):
    offs_m = tl.arange(0, block_m)
    for m in range(start_m, end_m, block_m):
        rows = m + offs_m
        if needs_m_mask:
            row_mask = rows < n_ctx
            q_tile = tl.load(
                q_ptr + base + rows[:, None] * d_head + offs_d[None, :],
                mask=row_mask[:, None], other=0.0,
            )
            do_tile = tl.load(
                do_ptr + base + rows[:, None] * d_head + offs_d[None, :],
                mask=row_mask[:, None], other=0.0,
            )
            lse_row = tl.load(lse_ptr + lse_base + rows, mask=row_mask, other=0.0)
            d_row = tl.load(delta_ptr + lse_base + rows, mask=row_mask, other=0.0)
        else:
            q_tile = tl.load(q_ptr + base + rows[:, None] * d_head + offs_d[None, :])
            do_tile = tl.load(do_ptr + base + rows[:, None] * d_head + offs_d[None, :])
            lse_row = tl.load(lse_ptr + lse_base + rows)
            d_row = tl.load(delta_ptr + lse_base + rows)

        qk = tl.dot(q_tile, tl.trans(k_tile)) * scale
        p = tl.exp(qk - lse_row[:, None])

        if stage == 2 or needs_m_mask or needs_n_mask:
            if needs_m_mask and needs_n_mask:
                valid = row_mask[:, None] & (offs_n[None, :] < n_ctx)
            elif needs_m_mask:
                valid = tl.broadcast_to(row_mask[:, None], (block_m, offs_n.shape[0]))
            elif needs_n_mask:
                valid = tl.broadcast_to((offs_n < n_ctx)[None, :], (block_m, offs_n.shape[0]))
            else:
                valid = tl.full((block_m, offs_n.shape[0]), 1, tl.int1)
            if stage == 2:
                valid = valid & (offs_n[None, :] <= rows[:, None])
            p = tl.where(valid, p, 0.0)

        dv_acc += tl.dot(tl.trans(p).to(do_tile.dtype), do_tile)
        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = (p * (dp - d_row[:, None])) * scale
        # ds inherits the masking via p (already zero where invalid)
        dk_acc += tl.dot(tl.trans(ds).to(q_tile.dtype), q_tile)
    return dk_acc, dv_acc


def _bwd_dkdv_configs():
    cfgs = []
    for bm, bn, w, s in [
        (64, 64, 4, 2),
        (64, 64, 4, 3),
        (128, 64, 4, 2),
        (128, 64, 4, 3),
        (128, 64, 8, 2),
        (128, 64, 8, 3),
        (64, 128, 4, 2),
        (64, 128, 4, 3),
        (64, 128, 8, 2),
        (128, 128, 4, 3),
        (128, 128, 8, 3),
        (32, 64, 4, 2),
        (64, 32, 4, 2),
    ]:
        cfgs.append(triton.Config({"block_m": bm, "block_n": bn}, num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_bwd_dkdv_configs(), key=["n_ctx", "d_head", "causal"])
@triton.jit
def _bwd_dkdv_kernel(
    q, k, v, do, lse, delta, dk, dv,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_d = tl.arange(0, d_head)
    base = pid_bh * n_ctx * d_head
    lse_base = pid_bh * n_ctx

    n_mask_needed = ((pid_n + 1) * block_n) > n_ctx
    if n_mask_needed:
        n_valid = offs_n[:, None] < n_ctx
        k_tile = tl.load(
            k + base + offs_n[:, None] * d_head + offs_d[None, :],
            mask=n_valid, other=0.0,
        )
        v_tile = tl.load(
            v + base + offs_n[:, None] * d_head + offs_d[None, :],
            mask=n_valid, other=0.0,
        )
    else:
        k_tile = tl.load(k + base + offs_n[:, None] * d_head + offs_d[None, :])
        v_tile = tl.load(v + base + offs_n[:, None] * d_head + offs_d[None, :])

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
            dk_acc, dv_acc, k_tile, v_tile,
            q, do, lse, delta,
            base, lse_base, offs_n, offs_d,
            diag_start, tl.minimum(diag_end, n_ctx),
            scale, n_ctx, d_head, block_m,
            stage=2, needs_m_mask=True, needs_n_mask=n_mask_needed,
        )
        # Off-diagonal: interior q-tiles (unmasked) + tail q-tile (m-masked).
        off_interior_end = (n_ctx // block_m) * block_m
        off_start = tl.maximum(diag_end, 0)
        interior_lo = tl.maximum(off_start, 0)
        dk_acc, dv_acc = _bwd_dkdv_inner(
            dk_acc, dv_acc, k_tile, v_tile,
            q, do, lse, delta,
            base, lse_base, offs_n, offs_d,
            interior_lo, off_interior_end,
            scale, n_ctx, d_head, block_m,
            stage=1, needs_m_mask=False, needs_n_mask=n_mask_needed,
        )
        dk_acc, dv_acc = _bwd_dkdv_inner(
            dk_acc, dv_acc, k_tile, v_tile,
            q, do, lse, delta,
            base, lse_base, offs_n, offs_d,
            tl.maximum(off_interior_end, off_start), n_ctx,
            scale, n_ctx, d_head, block_m,
            stage=1, needs_m_mask=True, needs_n_mask=n_mask_needed,
        )
    else:
        interior_end = (n_ctx // block_m) * block_m
        dk_acc, dv_acc = _bwd_dkdv_inner(
            dk_acc, dv_acc, k_tile, v_tile,
            q, do, lse, delta,
            base, lse_base, offs_n, offs_d,
            0, interior_end,
            scale, n_ctx, d_head, block_m,
            stage=0, needs_m_mask=False, needs_n_mask=n_mask_needed,
        )
        dk_acc, dv_acc = _bwd_dkdv_inner(
            dk_acc, dv_acc, k_tile, v_tile,
            q, do, lse, delta,
            base, lse_base, offs_n, offs_d,
            interior_end, n_ctx,
            scale, n_ctx, d_head, block_m,
            stage=0, needs_m_mask=True, needs_n_mask=n_mask_needed,
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


@triton.jit
def _bwd_dq_inner(
    dq_acc,
    q_tile, do_tile, lse_row, d_row,
    k_ptr, v_ptr,
    base, offs_m, offs_d,
    start_n, end_n,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    block_n: tl.constexpr,
    stage: tl.constexpr,
    needs_n_mask: tl.constexpr,
    row_mask_for_dot,
    needs_m_mask: tl.constexpr,
):
    offs_n = tl.arange(0, block_n)
    for n in range(start_n, end_n, block_n):
        cols = n + offs_n
        if needs_n_mask:
            col_mask = cols < n_ctx
            k_tile = tl.load(
                k_ptr + base + cols[:, None] * d_head + offs_d[None, :],
                mask=col_mask[:, None], other=0.0,
            )
            v_tile = tl.load(
                v_ptr + base + cols[:, None] * d_head + offs_d[None, :],
                mask=col_mask[:, None], other=0.0,
            )
        else:
            k_tile = tl.load(k_ptr + base + cols[:, None] * d_head + offs_d[None, :])
            v_tile = tl.load(v_ptr + base + cols[:, None] * d_head + offs_d[None, :])

        qk = tl.dot(q_tile, tl.trans(k_tile)) * scale
        p = tl.exp(qk - lse_row[:, None])

        if stage == 2 or needs_n_mask or needs_m_mask:
            if needs_m_mask and needs_n_mask:
                valid = row_mask_for_dot[:, None] & col_mask[None, :]
            elif needs_m_mask:
                valid = tl.broadcast_to(row_mask_for_dot[:, None], (offs_m.shape[0], block_n))
            elif needs_n_mask:
                valid = tl.broadcast_to(col_mask[None, :], (offs_m.shape[0], block_n))
            else:
                valid = tl.full((offs_m.shape[0], block_n), 1, tl.int1)
            if stage == 2:
                valid = valid & (cols[None, :] <= offs_m[:, None])
            p = tl.where(valid, p, 0.0)

        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = (p * (dp - d_row[:, None])) * scale
        dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)
    return dq_acc


def _bwd_dq_configs():
    cfgs = []
    for bm, bn, w, s in [
        (64, 64, 4, 2),
        (64, 64, 4, 3),
        (128, 64, 4, 2),
        (128, 64, 4, 3),
        (128, 64, 8, 2),
        (128, 64, 8, 3),
        (64, 128, 4, 2),
        (64, 128, 4, 3),
        (64, 128, 8, 2),
        (128, 128, 4, 3),
        (128, 128, 8, 3),
        (32, 64, 4, 2),
        (64, 32, 4, 2),
    ]:
        cfgs.append(triton.Config({"block_m": bm, "block_n": bn}, num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_bwd_dq_configs(), key=["n_ctx", "d_head", "causal"])
@triton.jit
def _bwd_dq_kernel(
    q, k, v, do, lse, delta, dq,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, d_head)
    base = pid_bh * n_ctx * d_head
    lse_base = pid_bh * n_ctx

    row_mask = offs_m < n_ctx
    needs_m_mask = (pid_m + 1) * block_m > n_ctx
    q_tile = tl.load(
        q + base + offs_m[:, None] * d_head + offs_d[None, :],
        mask=row_mask[:, None], other=0.0,
    )
    do_tile = tl.load(
        do + base + offs_m[:, None] * d_head + offs_d[None, :],
        mask=row_mask[:, None], other=0.0,
    )
    lse_row = tl.load(lse + lse_base + offs_m, mask=row_mask, other=0.0)
    d_row = tl.load(delta + lse_base + offs_m, mask=row_mask, other=0.0)

    dq_acc = tl.zeros((block_m, d_head), tl.float32)

    if causal:
        # Q-tile spans rows [pid_m*block_m, (pid_m+1)*block_m).
        # Off-diagonal K-tiles: cols < pid_m*block_m, aligned to block_n. No causal mask.
        # Diagonal range covers [off_end, (pid_m+1)*block_m), masked.
        q_row_start = pid_m * block_m
        off_end = (q_row_start // block_n) * block_n
        dq_acc = _bwd_dq_inner(
            dq_acc, q_tile, do_tile, lse_row, d_row,
            k, v, base, offs_m, offs_d,
            0, off_end,
            scale, n_ctx, d_head, block_n,
            stage=1, needs_n_mask=False,
            row_mask_for_dot=row_mask, needs_m_mask=needs_m_mask,
        )
        diag_end = tl.minimum(n_ctx, (pid_m + 1) * block_m)
        dq_acc = _bwd_dq_inner(
            dq_acc, q_tile, do_tile, lse_row, d_row,
            k, v, base, offs_m, offs_d,
            off_end, diag_end,
            scale, n_ctx, d_head, block_n,
            stage=2, needs_n_mask=True,
            row_mask_for_dot=row_mask, needs_m_mask=needs_m_mask,
        )
    else:
        interior_end = (n_ctx // block_n) * block_n
        dq_acc = _bwd_dq_inner(
            dq_acc, q_tile, do_tile, lse_row, d_row,
            k, v, base, offs_m, offs_d,
            0, interior_end,
            scale, n_ctx, d_head, block_n,
            stage=0, needs_n_mask=False,
            row_mask_for_dot=row_mask, needs_m_mask=needs_m_mask,
        )
        dq_acc = _bwd_dq_inner(
            dq_acc, q_tile, do_tile, lse_row, d_row,
            k, v, base, offs_m, offs_d,
            interior_end, n_ctx,
            scale, n_ctx, d_head, block_n,
            stage=0, needs_n_mask=True,
            row_mask_for_dot=row_mask, needs_m_mask=needs_m_mask,
        )

    tl.store(
        dq + base + offs_m[:, None] * d_head + offs_d[None, :],
        dq_acc.to(dq.dtype.element_ty),
        mask=row_mask[:, None],
    )


def triton_flash_bwd(grad_out, q, k, v, out, lse, causal=False, softmax_scale=None):
    assert grad_out.is_contiguous() and q.is_contiguous() and k.is_contiguous()
    assert v.is_contiguous() and out.is_contiguous() and lse.is_contiguous()
    n_ctx = q.shape[-2]
    d_head = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else d_head ** -0.5

    bh = q.shape[0] * q.shape[1]
    delta = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    pre_block = 128
    _bwd_preprocess[(triton.cdiv(n_ctx, pre_block), bh)](
        out, grad_out, delta,
        n_ctx, d_head, pre_block,
        num_warps=4, num_stages=2,
    )

    dkdv_grid = lambda meta: (triton.cdiv(n_ctx, meta["block_n"]), bh)
    _bwd_dkdv_kernel[dkdv_grid](
        q, k, v, grad_out, lse, delta, dk, dv,
        scale, n_ctx, d_head, causal,
    )

    dq_grid = lambda meta: (triton.cdiv(n_ctx, meta["block_m"]), bh)
    _bwd_dq_kernel[dq_grid](
        q, k, v, grad_out, lse, delta, dq,
        scale, n_ctx, d_head, causal,
    )

    return dq, dk, dv


def can_use_triton_bwd(q):
    return (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (32, 64, 128)
    )
