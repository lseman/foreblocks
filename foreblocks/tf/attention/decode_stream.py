from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .paged import PagedKVCache


def _dense_kv_block(
    cache: PagedKVCache,
    b: int,
    blk: int,
    blen: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k_blk = cache.storage_k[b, :, blk, :blen, :]
    v_blk = cache.storage_v[b, :, blk, :blen, :]
    return k_blk, v_blk


def _latent_kv_block(
    cache: PagedKVCache,
    b: int,
    blk: int,
    blen: int,
    d_head: int,
    mla_k_up_proj: Optional[nn.Module],
    mla_v_up_proj: Optional[nn.Module],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mla_k_up_proj is None or mla_v_up_proj is None:
        raise RuntimeError(
            "MLA latent paged decode requires k_up/v_up projection modules."
        )
    latent_blk = cache.storage_latent[b, blk, :blen, :]
    k_blk = (
        mla_k_up_proj(latent_blk)
        .view(blen, cache.Hkv, d_head)
        .permute(1, 0, 2)
        .contiguous()
    )
    v_blk = (
        mla_v_up_proj(latent_blk)
        .view(blen, cache.Hkv, d_head)
        .permute(1, 0, 2)
        .contiguous()
    )
    return k_blk, v_blk


def paged_stream_decode_standard(
    q_bhtd: torch.Tensor,
    cache: PagedKVCache,
    kv_repeat: int,
    scale: float,
    dropout_p: float,
    training: bool,
    is_causal: bool,
    q_start_pos: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    mla_k_up_proj: Optional[nn.Module] = None,
    mla_v_up_proj: Optional[nn.Module] = None,
) -> torch.Tensor:
    B, Hq, Tq, D = q_bhtd.shape
    BS = cache.block_size

    o_num = torch.zeros(B, Hq, Tq, D, device=q_bhtd.device, dtype=q_bhtd.dtype)
    l_den = torch.zeros(B, Hq, Tq, device=q_bhtd.device, dtype=q_bhtd.dtype)
    m_max = torch.full(
        (B, Hq, Tq),
        -float("inf"),
        device=q_bhtd.device,
        dtype=q_bhtd.dtype,
    )

    if is_causal:
        if q_start_pos is None:
            raise ValueError("q_start_pos is required when is_causal=True")
        q_abs = q_start_pos.view(B, 1, 1).to(torch.long) + torch.arange(
            Tq, device=q_bhtd.device
        ).view(1, 1, Tq)

    for b in range(B):
        blocks = cache.block_table[b]
        if not blocks:
            continue

        last_idx_in_table, last_off = cache.write_pos[b]
        q_blk = q_bhtd[b]

        for bi, blk in enumerate(blocks):
            if bi < last_idx_in_table:
                blen = BS
            elif bi == last_idx_in_table:
                blen = last_off
            else:
                blen = 0

            if blen == 0:
                continue

            k_start = bi * BS

            if getattr(cache, "use_latent_cache", False):
                k_blk, v_blk = _latent_kv_block(
                    cache,
                    b,
                    blk,
                    blen,
                    D,
                    mla_k_up_proj,
                    mla_v_up_proj,
                )
            else:
                k_blk, v_blk = _dense_kv_block(cache, b, blk, blen)

            if kv_repeat > 1:
                k_blk = k_blk.repeat_interleave(kv_repeat, dim=0)
                v_blk = v_blk.repeat_interleave(kv_repeat, dim=0)

            scores = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale

            if is_causal:
                k_pos = k_start + torch.arange(blen, device=q_bhtd.device).view(
                    1, 1, blen
                )
                q_pos = q_abs[b].view(1, Tq, 1)
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

            if key_padding_mask is not None:
                kp_blk = key_padding_mask[b, k_start : k_start + blen]
                scores = scores.masked_fill(kp_blk.view(1, 1, blen), float("-inf"))

            if attn_mask is not None:
                attn_blk = attn_mask[b, :, :, k_start : k_start + blen]
                scores = scores.masked_fill(attn_blk, float("-inf"))

            m_old = m_max[b]
            m_block = torch.amax(scores, dim=-1)
            m_new = torch.maximum(m_old, m_block)
            alpha = torch.exp(m_old - m_new)
            alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))

            scores_shift = scores - m_new.unsqueeze(-1)
            scores_exp = torch.exp(scores_shift)
            scores_exp = torch.where(
                torch.isfinite(scores_shift), scores_exp, torch.zeros_like(scores_exp)
            )

            if training and dropout_p > 0.0:
                scores_exp = F.dropout(scores_exp, p=dropout_p, training=True)

            l_new = alpha * l_den[b] + scores_exp.sum(dim=-1)
            o_new = alpha.unsqueeze(-1) * o_num[b] + torch.matmul(scores_exp, v_blk)

            m_max[b] = m_new
            l_den[b] = l_new
            o_num[b] = o_new

    return o_num / l_den.clamp_min(1e-9).unsqueeze(-1)
