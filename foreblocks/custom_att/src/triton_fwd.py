import torch
import triton
import triton.language as tl


# stage: 0 = non-causal (no mask), 1 = causal off-diagonal, 2 = causal diagonal
@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q_tile,
    k_desc,
    v_desc,
    offs_m,
    start_n,
    end_n,
    n_ctx: tl.constexpr,
    block_n: tl.constexpr,
    stage: tl.constexpr,
    needs_n_mask: tl.constexpr,
):
    offs_n = tl.arange(0, block_n)
    for n in range(start_n, end_n, block_n):
        cols = n + offs_n
        k_tile = k_desc.load([n, 0])
        qk = tl.dot(q_tile, tl.trans(k_tile))
        if stage == 2:
            qk = tl.where(cols[None, :] <= offs_m[:, None], qk, -float("inf"))
        if needs_n_mask:
            qk = tl.where(cols[None, :] < n_ctx, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None]
        v_tile = v_desc.load([n, 0])
        acc += tl.dot(p.to(v_tile.dtype), v_tile)
        m_i = m_new
        l_i = l_new
    return acc, l_i, m_i


@triton.jit
def _flash_fwd_kernel(
    q,
    k,
    v,
    out,
    lse,
    scale: tl.constexpr,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    start_m = pid_m * block_m
    offs_m = start_m + tl.arange(0, block_m)
    base = pid_bh * n_ctx * d_head
    log2e = 1.4426950408889634
    qk_scale = scale * log2e

    q_desc = tl.make_tensor_descriptor(
        base=q + base,
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
    out_desc = tl.make_tensor_descriptor(
        base=out + base,
        shape=[n_ctx, d_head],
        strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )

    q_tile = q_desc.load([start_m, 0])
    q_tile = (q_tile * qk_scale).to(q_tile.dtype)

    m_i = tl.full((block_m,), -float("inf"), tl.float32)
    l_i = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, d_head), tl.float32)

    if causal:
        off_end = (start_m // block_n) * block_n
        if off_end > 0:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q_tile,
                k_desc,
                v_desc,
                offs_m,
                0,
                off_end,
                n_ctx,
                block_n,
                stage=1,
                needs_n_mask=False,
            )

        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q_tile,
            k_desc,
            v_desc,
            offs_m,
            off_end,
            tl.minimum(n_ctx, (pid_m + 1) * block_m),
            n_ctx,
            block_n,
            stage=2,
            needs_n_mask=True,
        )
    else:
        interior_end = (n_ctx // block_n) * block_n
        if interior_end > 0:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q_tile,
                k_desc,
                v_desc,
                offs_m,
                0,
                interior_end,
                n_ctx,
                block_n,
                stage=0,
                needs_n_mask=False,
            )
        if interior_end < n_ctx:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q_tile,
                k_desc,
                v_desc,
                offs_m,
                interior_end,
                n_ctx,
                n_ctx,
                block_n,
                stage=0,
                needs_n_mask=True,
            )

    acc = acc / l_i[:, None]
    out_desc.store([start_m, 0], acc.to(q_tile.dtype))
    tl.store(
        lse + pid_bh * n_ctx + offs_m,
        (m_i + tl.math.log2(l_i)) / log2e,
        mask=offs_m < n_ctx,
    )


def _is_ada_or_newer(q):
    major, minor = torch.cuda.get_device_capability(q.device)
    return major > 8 or (major == 8 and minor >= 9)


def _select_fwd_config(q, n_ctx, d_head, causal):
    if _is_ada_or_newer(q):
        if d_head <= 32:
            return 128, 128, 4, 4
        if d_head == 64:
            if causal:
                return 128, 64, 4, 3
            return (128, 128, 4, 3) if n_ctx >= 1024 else (64, 128, 4, 3)
        if d_head == 128:
            return (32, 64, 4, 2) if causal else (128, 32, 4, 2)
    if d_head <= 64:
        return 128, 64, 4, 3
    return 64, 64, 4, 3


def triton_flash_fwd(q, k, v, causal=False, softmax_scale=None):
    n_ctx = q.shape[-2]
    d_head = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else d_head**-0.5
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    bh = q.shape[0] * q.shape[1]
    block_m, block_n, num_warps, num_stages = _select_fwd_config(
        q, n_ctx, d_head, bool(causal)
    )
    grid = (triton.cdiv(n_ctx, block_m), bh)
    _flash_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        scale,
        n_ctx,
        d_head,
        causal,
        block_m,
        block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse


def can_use_triton_fwd(q):
    return (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (32, 64, 128)
    )
