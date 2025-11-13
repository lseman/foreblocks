# -*- coding: utf-8 -*-
import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from paged import PagedKVCache
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_available_backends():
    backends = {"flash": False, "xformers": False, "sdp": False, "softpick": False}
    try:
        from flash_attn import flash_attn_func  # noqa: F401
        backends["flash"] = True
    except ImportError:
        pass
    try:
        import xformers.ops  # noqa: F401
        backends["xformers"] = True
    except ImportError:
        pass
    backends["sdp"] = hasattr(F, "scaled_dot_product_attention")
    try:
        from ..third_party.flash_softpick_attn import (  # noqa: F401
            parallel_softpick_attn,
        )
        backends["softpick"] = True
    except ImportError:
        pass
    return backends

# ─────────────────────────────────────────────────────────────────────────────
# vLLM-like streaming over paged cache (no gather) for your layout
# ─────────────────────────────────────────────────────────────────────────────
def _paged_stream_decode_standard(
    q_bhtd: torch.Tensor,              # [B, Hq, Tq, D]
    cache: PagedKVCache,
    kv_repeat: int,                    # n_heads // n_kv_heads
    scale: float,
    dropout_p: float,
    training: bool,
    is_causal: bool,
) -> torch.Tensor:
    """
    Numerically-stable streaming softmax across blocks in cache.
    Works directly with PagedKVCache storage: [B,Hkv,max_blocks,BS,D].
    """
    B, Hq, Tq, D = q_bhtd.shape
    BS = cache.block_size

    # running accumulators
    o_num = torch.zeros(B, Hq, Tq, D, device=q_bhtd.device, dtype=q_bhtd.dtype)
    l_den = torch.zeros(B, Hq, Tq, device=q_bhtd.device, dtype=q_bhtd.dtype)
    m_max = torch.full((B, Hq, Tq), -float("inf"), device=q_bhtd.device, dtype=q_bhtd.dtype)

    # absolute q positions (for causal) = seq_len_before_append + arange(Tq)
    if is_causal:
        q_abs = cache.seq_len.view(B, 1, 1).to(torch.long) + torch.arange(Tq, device=q_bhtd.device).view(1, 1, Tq)

    for b in range(B):
        blocks = cache.block_table[b]
        if not blocks:
            continue

        # length of last block
        last_idx_in_table, last_off = cache.write_pos[b]
        for bi, blk in enumerate(blocks):
            # block valid length
            if bi < last_idx_in_table:
                blen = BS
            elif bi == last_idx_in_table:
                blen = last_off
            else:
                blen = 0
            if blen == 0:
                continue

            # K/V block: [Hkv, blen, D]
            k_blk = cache.storage_k[b, :, blk, :blen, :]  # [Hkv, blen, D]
            v_blk = cache.storage_v[b, :, blk, :blen, :]

            # GQA repeat per block if needed
            if kv_repeat > 1:
                k_blk = k_blk.repeat_interleave(kv_repeat, dim=0)  # [Hq, blen, D]
                v_blk = v_blk.repeat_interleave(kv_repeat, dim=0)  # [Hq, blen, D]

            # scores: [Hq, Tq, blen]
            q_blk = q_bhtd[b]  # [Hq,Tq,D]
            scores = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale  # [Hq,Tq,blen]

            # causal mask inside this block: k_pos > q_pos
            if is_causal:
                k_pos = (bi * BS) + torch.arange(blen, device=q_bhtd.device).view(1, 1, blen)
                q_pos = q_abs[b].view(1, Tq, 1)
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

            # streaming softmax merge: compare with previous m_max
            m_old = m_max[b]                    # [Hq,Tq]
            m_block = torch.amax(scores, dim=-1)  # [Hq,Tq]
            m_new = torch.maximum(m_old, m_block)
            alpha = torch.exp(m_old - m_new)

            scores_exp = torch.exp(scores - m_new.unsqueeze(-1))
            if training and dropout_p > 0.0:
                scores_exp = F.dropout(scores_exp, p=dropout_p, training=True)

            l_new = alpha * l_den[b] + scores_exp.sum(dim=-1)  # [Hq,Tq]
            o_new = alpha.unsqueeze(-1) * o_num[b] + torch.matmul(scores_exp, v_blk)  # [Hq,Tq,D]

            # write back
            m_max[b] = m_new
            l_den[b] = l_new
            o_num[b] = o_new

    return o_num / l_den.clamp_min(1e-9).unsqueeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# MultiAttention (with paged cache support)
# ─────────────────────────────────────────────────────────────────────────────
class MultiAttention(nn.Module):
    """
    Multi-backend attention with GQA/MQA support + optional paged KV cache.

    Features:
    - GQA / MQA
    - RoPE for self-attention
    - Attention types: standard, prob_sparse, softpick, sliding_window, frequency/dwt/autocor
    - Backends: PyTorch SDPA, FlashAttn, xFormers (if present)
    - Boolean mask semantics (True = masked/disallowed)
    - Paged KV cache for causal self-attention decode (no growing torch.cat; no gather)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        attention_type: str = "standard",
        prob_sparse_factor: float = 0.4,
        freq_modes: int = 32,
        use_rotary: bool = True,
        max_seq_len: int = 4096,
        cross_attention: bool = False,
        softpick_chunk_size: int = 128,
        window_size: int = 64,
        global_attention_ratio: float = 0.1,
        chunk_size: int = 1024,
        use_flash_sliding: bool = True,
        use_swiglu: bool = True,
        verbose_init: bool = False,
        # NEW:
        use_paged_cache: bool = True,
        cache_block_size: int = 128,
        max_cache_blocks: int = 2048,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_type = attention_type
        self.cross_attention = cross_attention

        # GQA/MQA
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim

        # Attn params
        self.dropout_p = dropout
        self.scale = self.head_dim ** -0.5
        self.prob_sparse_factor = prob_sparse_factor
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.use_flash_sliding = use_flash_sliding
        self.softpick_chunk_size = softpick_chunk_size

        # Setup projections / type-specific
        self._setup_attention_modules(attention_type, freq_modes, use_swiglu, use_rotary)

        # Backends
        self.backends = (
            _get_available_backends()
            if attention_type in ["standard", "prob_sparse", "softpick", "sliding_window"]
            else {}
        )

        # NEW: paged cache options
        self.use_paged_cache = use_paged_cache
        self.cache_block_size = cache_block_size
        self.max_cache_blocks = max_cache_blocks

        if verbose_init:
            gqa_info = f"GQA({n_heads}q/{self.n_kv_heads}kv)" if self.n_rep > 1 else "MHA"
            print(
                f"[MultiAttention] {gqa_info}, type={attention_type}, "
                f"backends={self.backends}, rotary={self.use_rotary}, paged_cache={self.use_paged_cache}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────
    def _setup_attention_modules(
        self, attention_type: str, freq_modes: int, use_swiglu: bool, use_rotary: bool
    ):
        if attention_type in ["standard", "prob_sparse", "softpick", "sliding_window"]:
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.k_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)

            self.out_proj = (
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 4, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.d_model * 4, self.d_model, bias=False),
                )
                if use_swiglu
                else nn.Linear(self.d_model, self.d_model)
            )
            self.dropout = nn.Dropout(self.dropout_p)

            self.use_rotary = use_rotary and not self.cross_attention
            if self.use_rotary:
                from .rotary import RotaryEmbedding  # local import to avoid hard dep at import-time
                self.rotary_emb = RotaryEmbedding(self.head_dim)
            else:
                self.rotary_emb = None
        else:
            self.use_rotary = False
            self.rotary_emb = None
            if attention_type == "frequency":
                from .multi_att_helper import FrequencyAttention
                self.freq_attention = FrequencyAttention(self.d_model, self.n_heads, self.dropout_p, modes=freq_modes)
            elif attention_type == "dwt":
                from .multi_att_helper import DWTAttention
                self.dwt_attention = DWTAttention(self.d_model, self.n_heads, self.dropout_p, modes=freq_modes)
            elif attention_type == "autocor":
                from .multi_att_helper import AutoCorrelation, AutoCorrelationLayer
                autocorr = AutoCorrelation(mask_flag=True, factor=1, attention_dropout=0.1, output_attention=False)
                self.freq_attention = AutoCorrelationLayer(
                    correlation=autocorr, d_model=self.d_model, n_heads=self.n_heads
                )

    # Public
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
        layer_state: Optional[Dict[str, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        key = key if key is not None else query
        value = value if value is not None else key

        attention_map = {
            "standard": self._standard_path,
            "prob_sparse": self._prob_sparse_path,
            "frequency": self._frequency_path,
            "dwt": self._dwt_path,
            "autocor": self._autocor_path,
            "softpick": self._softpick_path,
            "sliding_window": self._sliding_window_path,
        }
        fn = attention_map.get(self.attention_type)
        if fn is None:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")

        return fn(
            query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, cu_seqlens
        )

    def reset_cache(self):
        """Reset any non-paged cached patterns in frequency modules."""
        for attr_name in ["freq_attention", "dwt_attention"]:
            attn = getattr(self, attr_name, None)
            if attn is not None and hasattr(attn, "cache"):
                attn.cache.clear()

    def reset_paged_cache(self, layer_state: Optional[Dict[str, torch.Tensor]]):
        """Clear paged cache for all batch items (if present in layer_state)."""
        if layer_state and "paged_cache" in layer_state:
            cache: PagedKVCache = layer_state["paged_cache"]
            for b in range(cache.B):
                cache.reset_seq(b)

    # ─────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        return x.repeat_interleave(self.n_rep, dim=1)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, seqlen_offset=0) -> Tuple[torch.Tensor, torch.Tensor]:
        from foreblocks.tf.rotary import apply_rotary_emb
        B, H, T, D = q.shape
        max_len = T + (seqlen_offset if isinstance(seqlen_offset, int) else int(seqlen_offset.max().item()))
        self.rotary_emb._update_cos_sin_cache(max_len, device=q.device, dtype=q.dtype)

        q_bt_hd = q.transpose(1, 2).contiguous()
        k_bt_hd = k.transpose(1, 2).contiguous()

        cos_q, sin_q = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        if getattr(self.rotary_emb, "scale", None) is None:
            cos_k, sin_k = cos_q, sin_q
        else:
            cos_k, sin_k = self.rotary_emb._cos_k_cached, self.rotary_emb._sin_k_cached

        q_bt_hd = apply_rotary_emb(
            q_bt_hd, cos_q, sin_q, interleaved=self.rotary_emb.interleaved, seqlen_offsets=seqlen_offset, inplace=False
        )
        k_bt_hd = apply_rotary_emb(
            k_bt_hd, cos_k, sin_k, interleaved=self.rotary_emb.interleaved, seqlen_offsets=seqlen_offset, inplace=False
        )
        return q_bt_hd.transpose(1, 2).contiguous(), k_bt_hd.transpose(1, 2).contiguous()

    def _create_sliding_window_mask(
        self, T_q: int, T_k: int, device: torch.device, is_causal: bool = True
    ) -> torch.Tensor:
        i = torch.arange(T_q, device=device).unsqueeze(1)
        j = torch.arange(T_k, device=device).unsqueeze(0)
        if is_causal:
            return (j > i) | (j < (i - self.window_size + 1))
        else:
            half = self.window_size // 2
            return (j < (i - half)) | (j > (i + half))

    def _apply_masks(
        self,
        scores: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, H, T_q, T_k = scores.shape
        if attn_mask is not None:
            mask = attn_mask.bool()
            if mask.dim() == 2:
                mask = mask.view(1, 1, T_q, T_k)
            scores = scores.masked_fill(mask, float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, T_k).bool(), float("-inf"))
        return scores

    # ─────────────────────────────────────────────────────────────────────
    # Core attention dispatcher
    # ─────────────────────────────────────────────────────────────────────
    def _standard_path(
        self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, *_  # cu_seqlens
    ):
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # If we're in self-attn decode with paged cache: stream like vLLM
        use_paged_decode = (
            self.use_paged_cache and (layer_state is not None) and (not self.cross_attention)
        )

        if use_paged_decode:
            # 1) Project to heads
            q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)  # [B,Hq,Tq,D]
            k = self.k_proj(key).view(B, T_k, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(value).view(B, T_k, self.n_kv_heads, self.head_dim).transpose(1, 2)

            # 2) RoPE with offsets BEFORE append
            if self.use_rotary and self.rotary_emb is not None:
                cache = PagedKVCache.ensure(
                    layer_state,
                    batch_size=B,
                    n_kv_heads=self.n_kv_heads,
                    head_dim=self.head_dim,
                    block_size=self.cache_block_size,
                    device=q.device,
                    dtype=q.dtype,
                    max_blocks=self.max_cache_blocks,
                )
                seqlen_offsets = cache.seq_len.clone()
                # apply RoPE to q and the new k
                q, k = self._apply_rope(q, k, seqlen_offset=seqlen_offsets)

            # 3) Append ONLY new keys/values into cache
            if T_k > 0:
                cache = PagedKVCache.ensure(
                    layer_state,
                    batch_size=B,
                    n_kv_heads=self.n_kv_heads,
                    head_dim=self.head_dim,
                    block_size=self.cache_block_size,
                    device=q.device,
                    dtype=q.dtype,
                    max_blocks=self.max_cache_blocks,
                )
                for b in range(B):
                    cache.append_step(k[b], v[b], b)

            # 4) Streaming block-wise softmax over cache (no gather)
            out_bhqd = _paged_stream_decode_standard(
                q_bhtd=q,
                cache=layer_state["paged_cache"],
                kv_repeat=self.n_rep,
                scale=self.scale,
                dropout_p=self.dropout_p,
                training=self.training,
                is_causal=is_causal and not self.cross_attention,
            )
            out = out_bhqd.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
            return self.out_proj(self.dropout(out)), None, layer_state

        # Else: use your existing full attention path (prefill or cross-attn)
        q, k, v = self._process_qkv(query, key, value, layer_state)
        out, weights = self._compute_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(self.dropout(out)), weights, layer_state

    # Original concat-growing (and optional paged-gather) QKV pipeline
    def _process_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_state: Optional[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        use_paged_now = (
            self.use_paged_cache and (layer_state is not None) and (not self.cross_attention)
        )
        if use_paged_now:
            return self._process_qkv_paged(query, key, value, layer_state)

        B, T_q, _ = query.shape
        T_k = key.shape[1]
        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.n_kv_heads, self.head_dim).transpose(1, 2)

        seqlen_offset = 0
        if self.use_rotary and (layer_state is not None) and (not self.cross_attention):
            if "k" in layer_state and isinstance(layer_state["k"], torch.Tensor):
                seqlen_offset = layer_state["k"].size(2)

        if self.use_rotary and self.rotary_emb is not None:
            q, k = self._apply_rope(q, k, seqlen_offset=seqlen_offset)

        if not self.cross_attention:
            if layer_state is None:
                layer_state = {}
            if "k" in layer_state:
                k = torch.cat([layer_state["k"], k], dim=2)
                v = torch.cat([layer_state["v"], v], dim=2)
            layer_state["k"] = k
            layer_state["v"] = v

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        return q, k, v

    def _process_qkv_paged(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_state: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Kept for compatibility with your previous code paths that expect
        gathered KV; the decode path above no longer calls the gather.
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.n_kv_heads, self.head_dim).transpose(1, 2)

        seqlen_offsets = None
        if self.use_rotary and not self.cross_attention:
            cache = PagedKVCache.ensure(
                layer_state,
                batch_size=B,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                block_size=self.cache_block_size,
                device=q.device,
                dtype=q.dtype,
                max_blocks=self.max_cache_blocks,
            )
            seqlen_offsets = cache.seq_len.clone()

        if self.use_rotary and self.rotary_emb is not None:
            q, k = self._apply_rope(q, k, seqlen_offset=(seqlen_offsets if seqlen_offsets is not None else 0))

        # Append, then (if some caller really needs) gather the full history
        if not self.cross_attention:
            cache = PagedKVCache.ensure(
                layer_state,
                batch_size=B,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                block_size=self.cache_block_size,
                device=q.device,
                dtype=q.dtype,
                max_blocks=self.max_cache_blocks,
            )
            for b in range(B):
                cache.append_step(k[b], v[b], b)

            k_full, v_full = cache.gather_kv_batched()  # [B, Hkv, T_total, D]
            k = k_full
            v = v_full

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        return q, k, v

    # Plain full attention compute (prefill / cross)
    def _compute_attention(
        self,
        q: torch.Tensor,  # [B, H, T_q, D]
        k: torch.Tensor,  # [B, H, T_k, D]
        v: torch.Tensor,  # [B, H, T_k, D]
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if is_causal and not self.cross_attention:
            causal_mask = torch.triu(torch.ones(T_q, T_k, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.view(1, 1, T_q, T_k), float("-inf"))

        scores = self._apply_masks(scores, attn_mask, key_padding_mask)

        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p, training=True)

        out = torch.matmul(weights, v)
        return out, (weights if need_weights else None)

    # ─────────────────────────────────────────────────────────────────────
    # Prob-sparse
    # ─────────────────────────────────────────────────────────────────────
    def _prob_sparse_path(
        self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, *_  # cu_seqlens
    ):
        B, T_q, _ = query.shape
        q, k, v = self._process_qkv(query, key, value, layer_state)
        out, weights = self._prob_sparse_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(self.dropout(out)), weights, layer_state

    def _prob_sparse_attention(
        self,
        q: torch.Tensor,  # [B, H, T_q, D]
        k: torch.Tensor,  # [B, H, T_k, D]
        v: torch.Tensor,  # [B, H, T_k, D]
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        u = max(1, min(T_q, int(5 * math.log(max(T_q, 2)))))
        sample_k = max(1, min(T_k, int(math.ceil(5 * math.log(max(T_k, 2))))))

        if u >= T_q or sample_k >= T_k:
            return self._compute_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)

        k_sample = k[:, :, :: max(1, T_k // sample_k), :][:, :, :sample_k, :]
        scores_sample = torch.matmul(q, k_sample.transpose(-2, -1)) * self.scale
        sparsity = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)

        _, top_idx = torch.topk(sparsity, k=u, dim=-1)  # [B, H, u]
        top_q = torch.gather(q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D))

        scores = torch.matmul(top_q, k.transpose(-2, -1)) * self.scale  # [B, H, u, T_k]

        if is_causal and not self.cross_attention:
            q_pos = top_idx.unsqueeze(-1)
            k_pos = torch.arange(T_k, device=q.device).view(1, 1, 1, T_k)
            scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

        if attn_mask is not None:
            mask = attn_mask.bool()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, T_q, T_k)
            mask_top = torch.gather(mask, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, T_k))
            scores = scores.masked_fill(mask_top, float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, T_k), float("-inf"))

        weights = F.softmax(scores - scores.max(dim=-1, keepdim=True)[0], dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p, training=True)
        top_out = torch.matmul(weights, v)  # [B, H, u, D]

        output = v.mean(dim=2, keepdim=True).expand(B, H, T_q, D).clone()
        output.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D), top_out)

        full_weights = None
        if need_weights:
            full_weights = torch.zeros(B, H, T_q, T_k, device=q.device, dtype=weights.dtype)
            full_weights.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, T_k), weights)
        return output, full_weights

    # ─────────────────────────────────────────────────────────────────────
    # Sliding window
    # ─────────────────────────────────────────────────────────────────────
    def _sliding_window_path(
        self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, *_  # cu_seqlens
    ):
        B, T_q, _ = query.shape
        q, k, v = self._process_qkv(query, key, value, layer_state)

        # Try SDPA with window mask
        if self.use_flash_sliding and self.backends.get("sdp") and not need_weights:
            try:
                window_mask = self._create_sliding_window_mask(q.size(2), k.size(2), q.device, is_causal)
                combined = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T_q, T_k]
                if attn_mask is not None:
                    mask = attn_mask.bool()
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0).unsqueeze(0)
                    combined = combined | mask
                if key_padding_mask is not None:
                    combined = combined | key_padding_mask.view(B, 1, 1, k.size(2)).bool()

                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=combined,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,
                )
                out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
                return self.out_proj(self.dropout(out)), None, layer_state
            except Exception:
                pass

        out, weights = self._sliding_window_manual(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(self.dropout(out)), weights, layer_state

    def _sliding_window_manual(
        self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        if T_q <= self.chunk_size:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            window_mask = self._create_sliding_window_mask(T_q, T_k, q.device, is_causal)
            scores = scores.masked_fill(window_mask.view(1, 1, T_q, T_k), float("-inf"))
            scores = self._apply_masks(scores, attn_mask, key_padding_mask)

            weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                weights = F.dropout(weights, p=self.dropout_p, training=True)
            return torch.matmul(weights, v), (weights if need_weights else None)

        output = torch.zeros_like(q)
        for i in range(0, T_q, self.chunk_size):
            end_i = min(i + self.chunk_size, T_q)
            if is_causal:
                start_k = max(0, i - self.window_size + 1)
                end_k = end_i
            else:
                center = (i + end_i) // 2
                half = self.window_size // 2
                start_k, end_k = max(0, center - half), min(T_k, center + half + 1)

            q_chunk = q[:, :, i:end_i]
            k_chunk = k[:, :, start_k:end_k]
            v_chunk = v[:, :, start_k:end_k]

            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale

            q_pos = torch.arange(i, end_i, device=q.device).unsqueeze(1)
            k_pos = torch.arange(start_k, end_k, device=q.device).unsqueeze(0)
            if is_causal:
                local_mask = (k_pos > q_pos) | (k_pos < (q_pos - self.window_size + 1))
            else:
                half = self.window_size // 2
                local_mask = (k_pos < (q_pos - half)) | (k_pos > (q_pos + half))
            scores = scores.masked_fill(local_mask.view(1, 1, end_i - i, end_k - start_k), float("-inf"))

            if key_padding_mask is not None:
                chunk_pad = key_padding_mask[:, start_k:end_k]
                scores = scores.masked_fill(chunk_pad.view(B, 1, 1, -1), float("-inf"))

            weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                weights = F.dropout(weights, p=self.dropout_p, training=True)
            output[:, :, i:end_i] = torch.matmul(weights, v_chunk)

        return output, None

    # ─────────────────────────────────────────────────────────────────────
    # SoftPick (third-party optional)
    # ─────────────────────────────────────────────────────────────────────
    def _softpick_path(
        self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, cu_seqlens
    ):
        if not self.backends.get("softpick"):
            warnings.warn("[MultiAttention] SoftPick unavailable, falling back to standard.")
            return self._standard_path(
                query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, cu_seqlens
            )
        try:
            from ..third_party.flash_softpick_attn import parallel_softpick_attn
            B, T_q, _ = query.shape
            q, k, v = self._process_qkv(query, key, value, layer_state)

            if cu_seqlens is None:
                out = parallel_softpick_attn(q, k, v, scale=self.scale, cu_seqlens=None, head_first=False)
                out = out.contiguous().view(B, T_q, self.d_model)
            else:
                T_k = k.size(2)
                q_flat = q.reshape(B * T_q, self.n_heads, self.head_dim)
                k_flat = k.reshape(B * T_k, self.n_heads, self.head_dim)
                v_flat = v.reshape(B * T_k, self.n_heads, self.head_dim)
                out = parallel_softpick_attn(
                    q_flat, k_flat, v_flat, scale=self.scale, cu_seqlens=cu_seqlens, head_first=True
                )
                out = out.view(B, T_q, self.n_heads, self.head_dim).contiguous().view(B, T_q, self.d_model)

            return self.out_proj(self.dropout(out)), None, layer_state
        except Exception as e:
            warnings.warn(f"[MultiAttention] SoftPick failed ({e}), falling back.")
            return self._standard_path(
                query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, cu_seqlens
            )

    # ─────────────────────────────────────────────────────────────────────
    # Frequency / DWT / AutoCorrelation
    # ─────────────────────────────────────────────────────────────────────
    def _frequency_path(self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, *_):
        out, weights = self.freq_attention(query, key, value, attn_mask, key_padding_mask, is_causal, need_weights)
        return out, weights, None

    def _dwt_path(self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, *_):
        out, weights = self.dwt_attention(query, key, value, attn_mask, key_padding_mask, is_causal, need_weights)
        return out, weights, None

    def _autocor_path(self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, *_):
        out, weights = self.freq_attention(query, key, value, attn_mask, key_padding_mask, is_causal, need_weights)
        return out, weights, None
