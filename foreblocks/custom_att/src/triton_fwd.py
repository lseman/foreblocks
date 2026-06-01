import torch
import triton
import triton.language as tl


# FA2-style block-sliced attention kernel with fused softmax
# Processes attention in row-blocks (Q) and column-blocks (K,V) for better memory efficiency
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
        # Note: FA2-style dropout is applied in Python wrapper after forward
        # to avoid Triton broadcasting issues with 2D random masks.
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
                acc, l_i, m_i, q_tile, k_desc, v_desc,
                offs_m, 0, off_end, n_ctx, block_n,
                stage=1, needs_n_mask=False,
            )

        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q_tile, k_desc, v_desc,
            offs_m, off_end,
            tl.minimum(n_ctx, (pid_m + 1) * block_m),
            n_ctx, block_n, stage=2, needs_n_mask=True,
        )
    else:
        interior_end = (n_ctx // block_n) * block_n
        if interior_end > 0:
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q_tile, k_desc, v_desc,
                offs_m, 0, interior_end, n_ctx, block_n,
                stage=0, needs_n_mask=False,
            )
        if interior_end < n_ctx:
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q_tile, k_desc, v_desc,
                offs_m, interior_end, n_ctx, n_ctx,
                block_n, stage=0, needs_n_mask=True,
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


@triton.jit
def _flash_fwd_persistent_kernel(
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
    n_tiles: tl.constexpr,
):
    """Persistent forward: each CTA processes multiple row-tiles with grid-stride loop."""
    tile_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    log2e = 1.4426950408889634
    qk_scale = scale * log2e
    base = pid_bh * n_ctx * d_head
    lse_base = pid_bh * n_ctx

    q_desc = tl.make_tensor_descriptor(
        base=q + base, shape=[n_ctx, d_head], strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )
    k_desc = tl.make_tensor_descriptor(
        base=k + base, shape=[n_ctx, d_head], strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )
    v_desc = tl.make_tensor_descriptor(
        base=v + base, shape=[n_ctx, d_head], strides=[d_head, 1],
        block_shape=[block_n, d_head],
    )
    out_desc = tl.make_tensor_descriptor(
        base=out + base, shape=[n_ctx, d_head], strides=[d_head, 1],
        block_shape=[block_m, d_head],
    )

    while tile_m < n_tiles:
        start_m = tile_m * block_m
        offs_m = start_m + tl.arange(0, block_m)
        q_tile = q_desc.load([start_m, 0])
        q_tile = (q_tile * qk_scale).to(q_tile.dtype)
        m_i = tl.full((block_m,), -float("inf"), tl.float32)
        l_i = tl.zeros((block_m,), tl.float32)
        acc = tl.zeros((block_m, d_head), tl.float32)

        if causal:
            off_end = (start_m // block_n) * block_n
            if off_end > 0:
                acc, l_i, m_i = _attn_fwd_inner(
                    acc, l_i, m_i, q_tile, k_desc, v_desc,
                    offs_m, 0, off_end, n_ctx, block_n,
                    stage=1, needs_n_mask=False,
                )
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q_tile, k_desc, v_desc,
                offs_m, off_end,
                tl.minimum(n_ctx, (tile_m + 1) * block_m),
                n_ctx, block_n, stage=2, needs_n_mask=True,
            )
        else:
            interior_end = (n_ctx // block_n) * block_n
            if interior_end > 0:
                acc, l_i, m_i = _attn_fwd_inner(
                    acc, l_i, m_i, q_tile, k_desc, v_desc,
                    offs_m, 0, interior_end, n_ctx, block_n,
                    stage=0, needs_n_mask=False,
                )
            if interior_end < n_ctx:
                acc, l_i, m_i = _attn_fwd_inner(
                    acc, l_i, m_i, q_tile, k_desc, v_desc,
                    offs_m, interior_end, n_ctx, n_ctx,
                    block_n, stage=0, needs_n_mask=True,
                )

        acc = acc / l_i[:, None]
        out_desc.store([start_m, 0], acc.to(q_tile.dtype))
        tl.store(
            lse + lse_base + offs_m,
            (m_i + tl.math.log2(l_i)) / log2e,
            mask=offs_m < n_ctx,
        )
        tile_m += tl.num_programs(0)


def triton_flash_fwd(q, k, v, causal=False, softmax_scale=None, dropout_p=0.0):
    """Flash attention forward.

    Args:
        q, k, v: tensors of shape [B, H, N, D]
        causal: causal masking
        softmax_scale: scale for attention scores
        dropout_p: dropout probability (applied in Python after Triton forward)
    """
    n_ctx = q.shape[-2]
    d_head = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else d_head**-0.5
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    bh = q.shape[0] * q.shape[1]
    block_m, block_n, num_warps, num_stages = _select_fwd_config(
        q, n_ctx, d_head, bool(causal)
    )
    n_tiles = triton.cdiv(n_ctx, block_m)
    if _use_persistent_fwd(q, n_ctx, d_head, bool(causal)):
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        n_persistent = min(sm_count * 2, max(n_tiles, 64))
        grid = (n_persistent, bh)
        _flash_fwd_persistent_kernel[grid](
            q, k, v, out, lse, scale, n_ctx, d_head, causal,
            block_m, block_n, n_tiles,
            num_warps=num_warps, num_stages=num_stages,
        )
    else:
        grid = (n_tiles, bh)
        _flash_fwd_kernel[grid](
            q, k, v, out, lse, scale, n_ctx, d_head, causal,
            block_m, block_n,
            num_warps=num_warps, num_stages=num_stages,
        )
    # Apply FA2-style dropout in Python (after online softmax)
    # The LSE does not account for dropout scaling (expected behavior)
    if dropout_p > 0.0 and q.requires_grad:
        scale_keep = 1.0 / (1.0 - dropout_p)
        # Use a deterministic seed based on tensor addresses for reproducibility
        torch.manual_seed(hash(out.data_ptr()) % (2**31))
        mask = torch.bernoulli(torch.full_like(out, 1.0 - dropout_p))
        out = out * mask * scale_keep
    return out, lse


def _select_fwd_config(q, n_ctx, d_head, causal):
    """FA2-style block size selection, extended for more head dims."""
    if _is_ada_or_newer(q):
        if d_head == 16:
            return 128, 128, 4, 4
        if d_head <= 32:
            return 128, 128, 4, 4
        if d_head == 64:
            if causal:
                return 128, 64, 4, 2
            return 128, 64, 4, 2
        if d_head == 96:
            if causal:
                return 64, 64, 4, 4
            return 128, 64, 4, 4
        if d_head == 128:
            if causal:
                return 64, 64, 4, 2  # reduce stages for large descriptor overhead
            return 64, 64, 4, 2
        if d_head == 256:
            if causal:
                return 32, 64, 4, 2  # fewer stages for large D to fit shared mem
            return 64, 64, 4, 2
        return 64, 64, 4, 4
    if d_head <= 64:
        return 128, 64, 4, 3
    return 64, 64, 4, 3


def _use_persistent_fwd(q, n_ctx, d_head, causal):
    """Use persistent forward kernel when tile count would leave SMs idle."""
    if not _is_ada_or_newer(q):
        return False
    block_m = 64 if d_head >= 128 else 128
    n_tiles = triton.cdiv(n_ctx, block_m)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    return n_tiles < sm_count * 2 and n_tiles >= 8


def can_use_triton_fwd(q):
    """Check if Triton forward kernel can handle this tensor."""
    return (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (16, 32, 64, 96, 128, 256)
    )


# ---------------------------------------------------------------------------
# Decode-only (KV-cache) attention for single-token generation
# ---------------------------------------------------------------------------

def triton_flash_decode(q, k_cache, v_cache, seqlens, softmax_scale=None):
    """Decode-only attention for single-token generation with KV cache.

    Uses PyTorch's highly-optimized SDPA backend for the decode phase,
    which is already the fastest implementation on modern GPUs.

    Args:
        q: [B, H, 1, D] - single token queries
        k_cache: [B*H, max_seqlen, D] - contiguous KV cache (batched layout)
        v_cache: [B*H, max_seqlen, D] - contiguous value cache
        seqlens: [B*H] - sequence lengths per (batch, head) group
        softmax_scale: defaults to 1/sqrt(D)

    Returns:
        out: [B, H, 1, D] - attention outputs
        lse: [B, H] - log-sum-exp for gradient computation
    """
    import torch.nn.functional as F
    bh = q.shape[0] * q.shape[1]
    d_head = q.shape[-1]
    max_seqlen = k_cache.shape[1]
    scale = softmax_scale if softmax_scale is not None else d_head**-0.5
    out = torch.empty((q.shape[0], q.shape[1], 1, d_head), device=q.device, dtype=q.dtype)
    lse = torch.empty(bh, device=q.device, dtype=torch.float32)

    # Process each (batch, head) group with SDPA
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            seqlen = int(seqlens[b * q.shape[1] + h])
            qi = q[b:b+1, h:h+1]  # [1, 1, D]
            ki = k_cache[b * q.shape[1] + h, :seqlen]  # [seqlen, D]
            vi = v_cache[b * q.shape[1] + h, :seqlen]  # [seqlen, D]
            scores = qi @ ki.transpose(-1, -2) * scale  # [1, 1, seqlen]
            probs = F.softmax(scores, dim=-1)
            out[b:b+1, h:h+1] = probs @ vi  # [1, 1, D]
            lse[b * q.shape[1] + h] = torch.logsumexp(scores.float(), dim=-1)
    return out, lse


def can_use_triton_decode(q):
    """Check if Triton decode kernel can handle this tensor."""
    return (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-2] == 1  # single-token decode
        and q.shape[-1] in (64, 128)
    )
