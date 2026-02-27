from typing import Optional

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


def triton_paged_decode(
    q: torch.Tensor,
    cache,
    kv_repeat: int,
    scale: float,
    q_start_pos: Optional[torch.Tensor] = None,
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

    max_seq_blocks = max((len(bt) for bt in cache.block_table), default=0)
    if max_seq_blocks == 0:
        return torch.zeros_like(q)

    bt_tensor = torch.zeros(B, max_seq_blocks, dtype=torch.int32, device=q.device)
    for b, bt in enumerate(cache.block_table):
        if bt:
            bt_tensor[b, : len(bt)] = torch.tensor(
                bt, dtype=torch.int32, device=q.device
            )

    if q_start_pos is None:
        q_start = torch.zeros(B, dtype=torch.int32, device=q.device)
    else:
        q_start = q_start_pos.to(device=q.device, dtype=torch.int32)
        if q_start.ndim != 1 or q_start.shape[0] != B:
            raise ValueError("q_start_pos must be shape [B]")

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
    return out
