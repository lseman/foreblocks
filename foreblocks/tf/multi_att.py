import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_available_backends():
    """Check which attention backends are available."""
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


class MultiAttention(nn.Module):
    """
    Multi-backend attention with GQA/MQA support.

    Features:
    - Grouped Query Attention (GQA) and Multi-Query Attention (MQA)
    - RoPE positional embeddings for self-attention
    - Multiple attention types: standard, prob_sparse, softpick, sliding_window, frequency
    - Multiple backends: PyTorch SDPA, Flash Attention, xFormers
    - Boolean mask semantics (True = masked/disallowed)
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
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_type = attention_type
        self.cross_attention = cross_attention

        # GQA/MQA setup
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.kv_dim = self.n_kv_heads * self.head_dim

        # Attention parameters
        self.dropout_p = dropout
        self.scale = self.head_dim ** -0.5
        self.prob_sparse_factor = prob_sparse_factor
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.use_flash_sliding = use_flash_sliding
        self.softpick_chunk_size = softpick_chunk_size

        # Setup attention-specific modules
        self._setup_attention_modules(attention_type, freq_modes, use_swiglu, use_rotary)

        # Backend availability
        self.backends = (
            _get_available_backends()
            if attention_type in ["standard", "prob_sparse", "softpick", "sliding_window"]
            else {}
        )

        if verbose_init:
            gqa_info = f"GQA({n_heads}q/{self.n_kv_heads}kv)" if self.n_rep > 1 else "MHA"
            print(
                f"[MultiAttention] {gqa_info}, type={attention_type}, "
                f"backends={self.backends}, rotary={self.use_rotary}"
            )

    def _setup_attention_modules(
        self, attention_type: str, freq_modes: int, use_swiglu: bool, use_rotary: bool
    ):
        """Setup projections and type-specific modules."""
        if attention_type in ["standard", "prob_sparse", "softpick", "sliding_window"]:
            # QKV projections
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.k_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)

            # Output projection with optional SwiGLU
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

            # RoPE for self-attention only
            self.use_rotary = use_rotary and not self.cross_attention
            if self.use_rotary:
                from .rotary import RotaryEmbedding

                self.rotary_emb = RotaryEmbedding(self.head_dim)
            else:
                self.rotary_emb = None
        else:
            # Frequency-domain attention types
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

    # ========================================================================
    # Public API
    # ========================================================================

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
        """Forward pass with automatic attention type routing."""
        key = key if key is not None else query
        value = value if value is not None else key

        # Route to appropriate attention implementation
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
        """Reset any cached attention patterns."""
        for attr_name in ["freq_attention", "dwt_attention"]:
            attn = getattr(self, attr_name, None)
            if attn is not None and hasattr(attn, "cache"):
                attn.cache.clear()

    # ========================================================================
    # Core QKV Processing Pipeline
    # ========================================================================

    def _process_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_state: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unified QKV processing pipeline: project → transpose → RoPE → cache → repeat.
        Returns Q, K, V in [B, H, T, D] format ready for attention.

        NOTE:
        - For self-attention we maintain a per-layer KV cache in `layer_state`.
        - RoPE is applied with an offset equal to the cached KV length to keep positions consistent.
        - KV is cached BEFORE GQA repetition; repetition happens after caching.
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # 1) Project to heads: [B, T, d_model] → [B, T, H, D]
        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(B, T_k, self.n_kv_heads, self.head_dim)
        v = self.v_proj(value).view(B, T_k, self.n_kv_heads, self.head_dim)

        # 2) Transpose to [B, H, T, D] for attention
        q = q.transpose(1, 2)  # [B, H, T_q, D]
        k = k.transpose(1, 2)  # [B, H_kv, T_k, D]
        v = v.transpose(1, 2)  # [B, H_kv, T_k, D]

        # 3) Apply RoPE (before caching, on unrepeated KV) with correct offset
        seqlen_offset = 0
        if self.use_rotary and (layer_state is not None) and (not self.cross_attention):
            if "k" in layer_state and isinstance(layer_state["k"], torch.Tensor):
                seqlen_offset = layer_state["k"].size(2)  # cached T_prev

        if self.use_rotary and self.rotary_emb is not None:
            q, k = self._apply_rope(q, k, seqlen_offset=seqlen_offset)

        # 4) Cache KV for autoregressive self-attention (before repeat)
        #    For cross-attention, we do NOT maintain a target-time-growing KV cache.
        if not self.cross_attention:
            if layer_state is None:
                layer_state = {}
            if "k" in layer_state:
                k = torch.cat([layer_state["k"], k], dim=2)
                v = torch.cat([layer_state["v"], v], dim=2)
            layer_state["k"] = k
            layer_state["v"] = v

        # 5) Repeat KV heads for GQA (after caching)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        return q, k, v

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads for GQA. [B, n_kv_heads, T, D] → [B, n_heads, T, D]"""
        if self.n_rep == 1:
            return x
        B, n_kv, T, D = x.shape
        return x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, D).reshape(B, n_kv * self.n_rep, T, D)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, seqlen_offset=0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to Q and K in [B, H, T, D] format.
        """
        from foreblocks.tf.rotary import apply_rotary_emb

        B, H, T, D = q.shape

        # Update cos/sin cache to cover positions up to T + offset
        max_len = T + (seqlen_offset if isinstance(seqlen_offset, int) else int(seqlen_offset.max().item()))
        self.rotary_emb._update_cos_sin_cache(max_len, device=q.device, dtype=q.dtype)

        # RoPE expects [B, T, H, D]
        q_bt_hd = q.transpose(1, 2).contiguous()
        k_bt_hd = k.transpose(1, 2).contiguous()

        cos_q, sin_q = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        if self.rotary_emb.scale is None:
            cos_k, sin_k = cos_q, sin_q
        else:
            cos_k, sin_k = self.rotary_emb._cos_k_cached, self.rotary_emb._sin_k_cached

        q_bt_hd = apply_rotary_emb(
            q_bt_hd, cos_q, sin_q, interleaved=self.rotary_emb.interleaved, seqlen_offsets=seqlen_offset, inplace=False
        )
        k_bt_hd = apply_rotary_emb(
            k_bt_hd, cos_k, sin_k, interleaved=self.rotary_emb.interleaved, seqlen_offsets=seqlen_offset, inplace=False
        )

        # Back to [B, H, T, D]
        return q_bt_hd.transpose(1, 2).contiguous(), k_bt_hd.transpose(1, 2).contiguous()

    # ========================================================================
    # Mask Utilities
    # ========================================================================

    def _create_causal_mask(self, T_q: int, T_k: int, device: torch.device) -> torch.Tensor:
        """Create causal mask: [T_q, T_k], True = masked."""
        i = torch.arange(T_q, device=device).unsqueeze(1)
        j = torch.arange(T_k, device=device).unsqueeze(0)
        return j > i

    def _create_sliding_window_mask(
        self, T_q: int, T_k: int, device: torch.device, is_causal: bool = True
    ) -> torch.Tensor:
        """Create sliding window mask: [T_q, T_k], True = masked."""
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
        """Apply attention and key padding masks. scores: [B, H, T_q, T_k]"""
        B, H, T_q, T_k = scores.shape

        if attn_mask is not None:
            mask = attn_mask.bool()
            if mask.dim() == 2:
                mask = mask.view(1, 1, T_q, T_k)
            scores = scores.masked_fill(mask, float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, T_k).bool(), float("-inf"))

        return scores

    # ========================================================================
    # Standard Attention
    # ========================================================================

    def _standard_path(
        self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, *_  # cu_seqlens
    ):
        """Standard attention path."""
        B, T_q, _ = query.shape

        q, k, v = self._process_qkv(query, key, value, layer_state)
        out, weights = self._compute_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(self.dropout(out)), weights, layer_state

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
        """Core attention computation."""
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask for self-attention
        if is_causal and not self.cross_attention:
            causal_mask = torch.triu(torch.ones(T_q, T_k, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.view(1, 1, T_q, T_k), float("-inf"))

        scores = self._apply_masks(scores, attn_mask, key_padding_mask)

        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p, training=True)

        out = torch.matmul(weights, v)
        return out, (weights if need_weights else None)

    # ========================================================================
    # ProbSparse Attention (Informer)
    # ========================================================================

    def _prob_sparse_path(
        self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, *_  # cu_seqlens
    ):
        """ProbSparse attention path."""
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
        """
        ProbSparse attention following Informer paper.
        Selects top-u queries based on sparsity measurement M = max - mean.
        """
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        # Calculate sample sizes (Informer paper: c * ln(L))
        u = max(1, min(T_q, int(5 * math.log(max(T_q, 2)))))
        sample_k = max(1, min(T_k, int(math.ceil(5 * math.log(max(T_k, 2))))))

        # Degenerate case: fall back to standard
        if u >= T_q or sample_k >= T_k:
            return self._compute_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)

        # Sample keys uniformly
        k_sample = k[:, :, :: max(1, T_k // sample_k), :][:, :, :sample_k, :]  # [B, H, sample_k, D]

        # Compute sparsity measure: M(q_i) = max - mean
        scores_sample = torch.matmul(q, k_sample.transpose(-2, -1)) * self.scale  # [B, H, T_q, sample_k]
        sparsity = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)  # [B, H, T_q]

        # Select top-u queries
        _, top_idx = torch.topk(sparsity, k=u, dim=-1)  # [B, H, u]
        top_q = torch.gather(q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D))  # [B, H, u, D]

        # Full attention for top queries
        scores = torch.matmul(top_q, k.transpose(-2, -1)) * self.scale  # [B, H, u, T_k]

        # Apply causal mask
        if is_causal and not self.cross_attention:
            q_pos = top_idx.unsqueeze(-1)  # [B, H, u, 1]
            k_pos = torch.arange(T_k, device=q.device).view(1, 1, 1, T_k)
            scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

        # Apply attention masks
        if attn_mask is not None:
            mask = attn_mask.bool()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, T_q, T_k)
            mask_top = torch.gather(mask, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, T_k))
            scores = scores.masked_fill(mask_top, float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, T_k), float("-inf"))

        # Compute attention output for top queries
        weights = F.softmax(scores - scores.max(dim=-1, keepdim=True)[0], dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p, training=True)
        top_out = torch.matmul(weights, v)  # [B, H, u, D]

        # Scatter top outputs back, fill rest with V-mean
        output = v.mean(dim=2, keepdim=True).expand(B, H, T_q, D).clone()
        output.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D), top_out)

        # Optionally reconstruct full weights
        full_weights = None
        if need_weights:
            full_weights = torch.zeros(B, H, T_q, T_k, device=q.device, dtype=weights.dtype)
            full_weights.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, T_k), weights)

        return output, full_weights

    # ========================================================================
    # Sliding Window Attention
    # ========================================================================

    def _sliding_window_path(
        self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, *_  # cu_seqlens
    ):
        """Sliding window attention path."""
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
                    q,
                    k,
                    v,
                    attn_mask=combined,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,
                )
                out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
                return self.out_proj(self.dropout(out)), None, layer_state
            except Exception:
                pass  # Fall through

        # Manual sliding window (with chunking for long sequences)
        out, weights = self._sliding_window_manual(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(self.dropout(out)), weights, layer_state

    def _sliding_window_manual(
        self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Manual sliding window with optional chunking."""
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        # Small sequences: compute directly
        if T_q <= self.chunk_size:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            window_mask = self._create_sliding_window_mask(T_q, T_k, q.device, is_causal)
            scores = scores.masked_fill(window_mask.view(1, 1, T_q, T_k), float("-inf"))
            scores = self._apply_masks(scores, attn_mask, key_padding_mask)

            weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                weights = F.dropout(weights, p=self.dropout_p, training=True)

            return torch.matmul(weights, v), (weights if need_weights else None)

        # Large sequences: chunk processing
        output = torch.zeros_like(q)
        for i in range(0, T_q, self.chunk_size):
            end_i = min(i + self.chunk_size, T_q)

            # Determine KV range for this chunk
            if is_causal:
                start_k = max(0, i - self.window_size + 1)
                end_k = end_i
            else:
                center = (i + end_i) // 2
                half = self.window_size // 2
                start_k, end_k = max(0, center - half), min(T_k, center + half + 1)

            # Slice QKV for chunk
            q_chunk = q[:, :, i:end_i]
            k_chunk = k[:, :, start_k:end_k]
            v_chunk = v[:, :, start_k:end_k]

            # Compute chunk attention
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale

            # Create local window mask
            q_pos = torch.arange(i, end_i, device=q.device).unsqueeze(1)
            k_pos = torch.arange(start_k, end_k, device=q.device).unsqueeze(0)
            if is_causal:
                local_mask = (k_pos > q_pos) | (k_pos < (q_pos - self.window_size + 1))
            else:
                half = self.window_size // 2
                local_mask = (k_pos < (q_pos - half)) | (k_pos > (q_pos + half))

            scores = scores.masked_fill(local_mask.view(1, 1, end_i - i, end_k - start_k), float("-inf"))

            # Apply padding mask if present
            if key_padding_mask is not None:
                chunk_pad = key_padding_mask[:, start_k:end_k]
                scores = scores.masked_fill(chunk_pad.view(B, 1, 1, -1), float("-inf"))

            weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                weights = F.dropout(weights, p=self.dropout_p, training=True)

            output[:, :, i:end_i] = torch.matmul(weights, v_chunk)

        return output, None  # Don't reconstruct full weights for chunked

    # ========================================================================
    # SoftPick Attention
    # ========================================================================

    def _softpick_path(
        self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, cu_seqlens
    ):
        """SoftPick attention path with fallback."""
        if not self.backends.get("softpick"):
            warnings.warn("[MultiAttention] SoftPick unavailable, falling back to standard.")
            return self._standard_path(
                query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, layer_state, cu_seqlens
            )

        try:
            from ..third_party.flash_softpick_attn import parallel_softpick_attn

            B, T_q, _ = query.shape
            q, k, v = self._process_qkv(query, key, value, layer_state)

            # SoftPick expects specific format
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

    # ========================================================================
    # Frequency-Domain Attention
    # ========================================================================

    def _frequency_path(self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, *_):
        """Frequency attention path."""
        out, weights = self.freq_attention(query, key, value, attn_mask, key_padding_mask, is_causal, need_weights)
        return out, weights, None

    def _dwt_path(self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, *_):
        """DWT attention path."""
        out, weights = self.dwt_attention(query, key, value, attn_mask, key_padding_mask, is_causal, need_weights)
        return out, weights, None

    def _autocor_path(self, query, key, value, attn_mask, key_padding_mask, is_causal, need_weights, *_):
        """AutoCorrelation attention path."""
        out, weights = self.freq_attention(query, key, value, attn_mask, key_padding_mask, is_causal, need_weights)
        return out, weights, None
