"""foreblocks.ops.attention.paged_decode.

Triton kernel for paged KV-cache autoregressive decode.

Implements causal attention over a block-based KV cache with a per-sequence block
table, supporting grouped query attention (GQA) via kv_repeat. Computes
softmax(q·K^T)·V while respecting causal masking and variable sequence lengths.
Use when running autoregressive decode with paged KV cache allocation.

Core API:
- triton_paged_decode: single-kernel paged decode for [B, H, T, D] query tensors

"""

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _paged_decode_kernel(
        Q,
        K_storage,
        V_storage,
        BlockTable,
        SeqLen,
        QStart,
        Out,
        scale,
        Tq: tl.constexpr,
        D: tl.constexpr,
        BS: tl.constexpr,
        max_seq_blocks: tl.constexpr,
        kv_repeat: tl.constexpr,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kblock,
        stride_kbs,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vblock,
        stride_vbs,
        stride_vd,
        stride_bt_b,
        stride_bt_s,
    ):
        pid_b = tl.program_id(0)
        pid_hq = tl.program_id(1)
        pid_tq = tl.program_id(2)

        pid_hkv = pid_hq // kv_repeat
        seq_len = tl.load(SeqLen + pid_b)
        q_start = tl.load(QStart + pid_b)

        d_offs = tl.arange(0, D)
        bs_offs = tl.arange(0, BS)

        q_ptr = Q + pid_b * stride_qb + pid_hq * stride_qh + pid_tq * stride_qt
        q = tl.load(q_ptr + d_offs * stride_qd)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([D], dtype=tl.float32)

        q_abs = q_start + pid_tq

        for blk_idx in range(max_seq_blocks):
            active_blk = blk_idx * BS < seq_len
            phys_blk = tl.load(
                BlockTable + pid_b * stride_bt_b + blk_idx * stride_bt_s,
                mask=active_blk,
                other=0,
            )

            blk_start = blk_idx * BS
            blen = tl.minimum(BS, tl.maximum(seq_len - blk_start, 0))
            bs_mask = bs_offs < blen
            kv_mask = active_blk & bs_mask

            k_base = (
                K_storage
                + pid_b * stride_kb
                + pid_hkv * stride_kh
                + phys_blk * stride_kblock
            )
            k_block = tl.load(
                k_base + bs_offs[:, None] * stride_kbs + d_offs[None, :] * stride_kd,
                mask=kv_mask[:, None],
                other=0.0,
            )

            scores = tl.sum(q[None, :] * k_block, axis=1) * scale
            scores = tl.where(kv_mask, scores, -float("inf"))

            k_pos = blk_start + bs_offs
            scores = tl.where(k_pos <= q_abs, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores))
            alpha = tl.exp(m_i - m_new)
            scores_exp = tl.exp(scores - m_new)
            finite_scores = scores_exp == scores_exp
            scores_exp = tl.where(finite_scores, scores_exp, 0.0)

            l_i = alpha * l_i + tl.sum(scores_exp)
            acc = alpha * acc

            v_base = (
                V_storage
                + pid_b * stride_vb
                + pid_hkv * stride_vh
                + phys_blk * stride_vblock
            )
            v_block = tl.load(
                v_base + bs_offs[:, None] * stride_vbs + d_offs[None, :] * stride_vd,
                mask=kv_mask[:, None],
                other=0.0,
            )

            acc += tl.sum(scores_exp[:, None] * v_block, axis=0)
            m_i = m_new

        out = acc / tl.maximum(l_i, 1e-9)
        out_ptr = Out + pid_b * stride_qb + pid_hq * stride_qh + pid_tq * stride_qt
        tl.store(out_ptr + d_offs * stride_qd, out.to(Q.dtype.element_ty))

    @triton.jit
    def _paged_decode_split_kernel(
        Q,
        K_storage,
        V_storage,
        BlockTable,
        SeqLen,
        QStart,
        OutPartial,  # [B, Hq, Tq, NS, D] fp32, unnormalized acc
        MPartial,  # [B, Hq, Tq, NS] fp32, running max
        LPartial,  # [B, Hq, Tq, NS] fp32, running sum
        scale,
        Tq: tl.constexpr,
        D: tl.constexpr,
        BS: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        kv_repeat: tl.constexpr,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kblock,
        stride_kbs,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vblock,
        stride_vbs,
        stride_vd,
        stride_bt_b,
        stride_bt_s,
        stride_pb,
        stride_ph,
        stride_pt,
        stride_ps,
        stride_pd,
        stride_mb,
        stride_mh,
        stride_mt,
        stride_ms,
    ):
        pid_b = tl.program_id(0)
        pid_hq = tl.program_id(1)
        pid_ts = tl.program_id(2)
        pid_tq = pid_ts // NUM_SPLITS
        split = pid_ts % NUM_SPLITS

        pid_hkv = pid_hq // kv_repeat
        seq_len = tl.load(SeqLen + pid_b)
        q_start = tl.load(QStart + pid_b)

        d_offs = tl.arange(0, D)
        bs_offs = tl.arange(0, BS)

        q_ptr = Q + pid_b * stride_qb + pid_hq * stride_qh + pid_tq * stride_qt
        q = tl.load(q_ptr + d_offs * stride_qd)

        # Finite lower bound (not -inf) so empty splits stay NaN-free.
        NEG_BIG: tl.constexpr = -1e38
        m_i = NEG_BIG
        l_i = 0.0
        acc = tl.zeros([D], dtype=tl.float32)

        q_abs = q_start + pid_tq

        for i in range(BLOCKS_PER_SPLIT):
            blk_idx = split * BLOCKS_PER_SPLIT + i
            active_blk = blk_idx * BS < seq_len
            phys_blk = tl.load(
                BlockTable + pid_b * stride_bt_b + blk_idx * stride_bt_s,
                mask=active_blk,
                other=0,
            )

            blk_start = blk_idx * BS
            blen = tl.minimum(BS, tl.maximum(seq_len - blk_start, 0))
            bs_mask = bs_offs < blen
            kv_mask = active_blk & bs_mask

            k_base = (
                K_storage
                + pid_b * stride_kb
                + pid_hkv * stride_kh
                + phys_blk * stride_kblock
            )
            k_block = tl.load(
                k_base + bs_offs[:, None] * stride_kbs + d_offs[None, :] * stride_kd,
                mask=kv_mask[:, None],
                other=0.0,
            )

            scores = tl.sum(q[None, :] * k_block, axis=1) * scale
            valid = kv_mask & ((blk_start + bs_offs) <= q_abs)
            scores = tl.where(valid, scores, NEG_BIG)

            m_new = tl.maximum(m_i, tl.max(scores))
            alpha = tl.exp(m_i - m_new)
            scores_exp = tl.where(valid, tl.exp(scores - m_new), 0.0)

            l_i = alpha * l_i + tl.sum(scores_exp)
            acc = alpha * acc

            v_base = (
                V_storage
                + pid_b * stride_vb
                + pid_hkv * stride_vh
                + phys_blk * stride_vblock
            )
            v_block = tl.load(
                v_base + bs_offs[:, None] * stride_vbs + d_offs[None, :] * stride_vd,
                mask=kv_mask[:, None],
                other=0.0,
            )

            acc += tl.sum(scores_exp[:, None] * v_block, axis=0)
            m_i = m_new

        p_base = (
            OutPartial
            + pid_b * stride_pb
            + pid_hq * stride_ph
            + pid_tq * stride_pt
            + split * stride_ps
        )
        tl.store(p_base + d_offs * stride_pd, acc)
        m_base = (
            pid_b * stride_mb
            + pid_hq * stride_mh
            + pid_tq * stride_mt
            + split * stride_ms
        )
        tl.store(MPartial + m_base, m_i)
        tl.store(LPartial + m_base, l_i)


def triton_paged_decode(
    q: torch.Tensor,
    cache,
    kv_repeat: int,
    scale: float,
    q_start_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not q.is_cuda:
        raise RuntimeError("Triton paged decode requires CUDA tensors")
    if getattr(cache, "use_latent_cache", False):
        raise RuntimeError("Triton paged decode currently supports dense KV cache only")

    B, Hq, Tq, D = q.shape
    if D <= 0:
        raise ValueError("Invalid head dimension")
    if D & (D - 1) != 0 or cache.block_size & (cache.block_size - 1) != 0:
        # tl.arange needs power-of-2 extents; fail fast before kernel compile
        raise RuntimeError(
            "Triton paged decode requires power-of-2 head_dim and block_size"
        )

    max_seq_blocks = max((len(bt) for bt in cache.block_table), default=0)
    if max_seq_blocks == 0:
        return torch.zeros_like(q)

    # Pad on CPU, then a single H2D copy (one transfer instead of B).
    bt_cpu = torch.zeros(B, max_seq_blocks, dtype=torch.int32)
    for b, bt in enumerate(cache.block_table):
        if bt:
            bt_cpu[b, : len(bt)] = torch.as_tensor(bt, dtype=torch.int32)
    bt_tensor = bt_cpu.to(q.device, non_blocking=True)

    if q_start_pos is None:
        q_start = torch.zeros(B, dtype=torch.int32, device=q.device)
    else:
        q_start = q_start_pos.to(device=q.device, dtype=torch.int32)
        if q_start.ndim != 1 or q_start.shape[0] != B:
            raise ValueError("q_start_pos must be shape [B]")

    # ── Flash-decoding split-K ──────────────────────────────────────────
    # A (B, Hq, Tq) grid alone can leave most SMs idle during decode
    # (e.g. B=1, H=8, Tq=1 → 8 programs). When the grid is small and the
    # context long, split the KV blocks across programs and merge the
    # partial softmaxes on the host.
    # Crossover measured on Ada (128 SMs): single kernel wins at <=16 blocks
    # (host-side merge overhead dominates), split wins from ~64 blocks
    # (1.9x at 8k ctx, 7.5x at 32k, 10.8x at 100k).
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    n_programs = B * Hq * Tq
    if n_programs < 2 * sm_count and max_seq_blocks > 24:
        num_splits = min(
            max_seq_blocks, max(2, min(32, (2 * sm_count) // max(n_programs, 1)))
        )
    else:
        num_splits = 1

    common_strides = dict(
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_qt=q.stride(2),
        stride_qd=q.stride(3),
        stride_kb=cache.storage_k.stride(0),
        stride_kh=cache.storage_k.stride(1),
        stride_kblock=cache.storage_k.stride(2),
        stride_kbs=cache.storage_k.stride(3),
        stride_kd=cache.storage_k.stride(4),
        stride_vb=cache.storage_v.stride(0),
        stride_vh=cache.storage_v.stride(1),
        stride_vblock=cache.storage_v.stride(2),
        stride_vbs=cache.storage_v.stride(3),
        stride_vd=cache.storage_v.stride(4),
        stride_bt_b=bt_tensor.stride(0),
        stride_bt_s=bt_tensor.stride(1),
    )

    if num_splits > 1:
        blocks_per_split = triton.cdiv(max_seq_blocks, num_splits)
        partial = torch.empty(
            (B, Hq, Tq, num_splits, D), dtype=torch.float32, device=q.device
        )
        m_part = torch.empty(
            (B, Hq, Tq, num_splits), dtype=torch.float32, device=q.device
        )
        l_part = torch.empty_like(m_part)

        _paged_decode_split_kernel[(B, Hq, Tq * num_splits)](
            q,
            cache.storage_k,
            cache.storage_v,
            bt_tensor,
            cache.seq_len.to(dtype=torch.int32),
            q_start,
            partial,
            m_part,
            l_part,
            scale,
            Tq=Tq,
            D=D,
            BS=cache.block_size,
            BLOCKS_PER_SPLIT=blocks_per_split,
            NUM_SPLITS=num_splits,
            kv_repeat=kv_repeat,
            **common_strides,
            stride_pb=partial.stride(0),
            stride_ph=partial.stride(1),
            stride_pt=partial.stride(2),
            stride_ps=partial.stride(3),
            stride_pd=partial.stride(4),
            stride_mb=m_part.stride(0),
            stride_mh=m_part.stride(1),
            stride_mt=m_part.stride(2),
            stride_ms=m_part.stride(3),
        )

        # Logsumexp merge over splits: acc_s are unnormalized at max m_s.
        m_max = m_part.max(dim=-1, keepdim=True).values  # [B,Hq,Tq,1]
        w = torch.exp(m_part - m_max)  # empty splits → ~0
        l_total = (w * l_part).sum(dim=-1, keepdim=True)  # [B,Hq,Tq,1]
        out_f32 = (w.unsqueeze(-1) * partial).sum(dim=-2) / l_total.clamp_min(1e-9)
        return out_f32.to(q.dtype)

    out = torch.empty_like(q)
    grid = (B, Hq, Tq)

    _paged_decode_kernel[grid](
        q,
        cache.storage_k,
        cache.storage_v,
        bt_tensor,
        cache.seq_len.to(dtype=torch.int32),
        q_start,
        out,
        scale,
        Tq=Tq,
        D=D,
        BS=cache.block_size,
        max_seq_blocks=max_seq_blocks,
        kv_repeat=kv_repeat,
        **common_strides,
    )
    return out
