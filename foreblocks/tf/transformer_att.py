# transformer_att.py (fixed)

import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MultiAttention(nn.Module):
    """
    Multi-backend attention for time series.

    Notes:
    - Mask semantics are boolean with True = masked (disallowed) everywhere.
    - Rotary embeddings are applied only for self-attention (never cross-attn).
    - SDPA fast path uses boolean masks (PyTorch ≥ 2.0).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
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
        verbose_init: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_type = attention_type
        self.dropout_p = dropout
        self.cross_attention = cross_attention
        self.softpick_chunk_size = softpick_chunk_size
        self.prob_sparse_factor = prob_sparse_factor
        self.scale = self.head_dim ** -0.5

        # Sliding window parameters
        self.window_size = window_size
        self.global_attention_ratio = global_attention_ratio
        self.chunk_size = chunk_size
        self.use_flash_sliding = use_flash_sliding

        if attention_type in ["standard", "prob_sparse", "softpick", "sliding_window"]:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        # RoPE for self-attn only
        self.use_rotary = (
            use_rotary
            and attention_type in ["standard", "prob_sparse", "softpick", "sliding_window"]
            and not cross_attention
        )
        if self.use_rotary:
            from .embeddings import RotaryEmbedding
            self.rotary_emb = RotaryEmbedding(self.head_dim)

        # Frequency-space variants
        if attention_type == "frequency":
            from .fed import FrequencyAttention
            self.freq_attention = FrequencyAttention(d_model, n_heads, dropout, modes=freq_modes)
        elif attention_type == "dwt":
            from .fed import DWTAttention
            self.dwt_attention = DWTAttention(d_model, n_heads, dropout, modes=freq_modes)
        elif attention_type == "autocor":
            from .fed import AutoCorrelation, AutoCorrelationLayer
            autocorr_mech = AutoCorrelation(mask_flag=True, factor=1, attention_dropout=0.1, output_attention=False)
            self.freq_attention = AutoCorrelationLayer(correlation=autocorr_mech, d_model=d_model, n_heads=n_heads)

        self.backends = (
            _get_available_backends()
            if attention_type in ["standard", "prob_sparse", "softpick", "sliding_window"]
            else {}
        )
        self.attention_map = {
            "standard": self._internal_attention,
            "prob_sparse": self._internal_attention,
            "frequency": self._forward_frequency,
            "dwt": self._forward_dwt,
            "autocor": self._forward_autocor,
            "softpick": self._softpick_attention,
            "sliding_window": self._sliding_window_attention,
        }
        if verbose_init:
            print(f"[MultiAttention] type={self.attention_type}, backends={self.backends}, rotary={self.use_rotary}")

    # --------------------------------------------------------------------- #
    # Public forward
    # --------------------------------------------------------------------- #
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

        fn = self.attention_map.get(self.attention_type, None)
        if fn is None:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")

        return fn(
            query, key, value, attn_mask, key_padding_mask,
            is_causal, need_weights, layer_state, cu_seqlens
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _project_qkv(self, query, key, value):
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(B, T_k, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(B, T_k, self.n_heads, self.head_dim)
        return q, k, v

    def _apply_masks(self, scores, attn_mask, key_padding_mask, B, T_q, T_k):
        """
        scores: [B,H,T_q,T_k], float
        attn_mask/key_padding_mask: boolean, True = masked
        """
        if attn_mask is not None:
            am = attn_mask
            if am.dim() == 2:
                am = am.view(1, 1, T_q, T_k)
            scores = scores.masked_fill(am.bool(), float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, T_k).bool(), float("-inf"))
        return scores

    def _create_sliding_window_mask(self, T_q: int, T_k: int, device: torch.device, is_causal: bool = True) -> torch.Tensor:
        """
        Create sliding window mask for arbitrary Q/K lengths.
        Returns (T_q, T_k) boolean mask, True = masked.
        """
        i = torch.arange(T_q, device=device).unsqueeze(1)  # (T_q, 1)
        j = torch.arange(T_k, device=device).unsqueeze(0)  # (1, T_k)
        
        if is_causal:
            # Mask if: j > i (future) OR j < i - (window_size - 1) (too far in past)
            return (j > i) | (j < (i - (self.window_size - 1)))
        else:
            # Symmetric window
            half = self.window_size // 2
            return (j < (i - half)) | (j > (i + half))

    # --------------------------------------------------------------------- #
    # Sliding-window attention (efficient and chunked)
    # --------------------------------------------------------------------- #
    def _sliding_window_attention_efficient(
        self,
        q: torch.Tensor,  # [B,H,T,D]
        k: torch.Tensor,  # [B,H,T,D]
        v: torch.Tensor,  # [B,H,T,D]
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        if T_q > self.chunk_size:
            return self._chunked_sliding_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T_q,T_k]

        window_mask = self._create_sliding_window_mask(T_q, T_k, q.device, is_causal)
        scores = scores.masked_fill(window_mask.view(1, 1, T_q, T_k), float("-inf"))
        scores = self._apply_masks(scores, attn_mask, key_padding_mask, B, T_q, T_k)

        attn_weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=True)

        out = torch.matmul(attn_weights, v)  # [B,H,T_q,D]
        return out, (attn_weights if need_weights else None)

    def _chunked_sliding_attention(
        self,
        q: torch.Tensor,  # [B,H,T,D]
        k: torch.Tensor,  # [B,H,T,D]
        v: torch.Tensor,  # [B,H,T,D]
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)
        output = torch.zeros_like(q)
        all_weights = [] if need_weights else None

        for i in range(0, T_q, self.chunk_size):
            end_i = min(i + self.chunk_size, T_q)
            
            if is_causal:
                # Most restrictive query at position i needs keys in [max(0, i-w+1), i]
                # Most permissive query at end_i-1 needs keys in [max(0, end_i-1-w+1), end_i-1]
                start_k = max(0, i - self.window_size + 1)
                end_k = end_i
            else:
                center = (i + end_i) // 2
                half = self.window_size // 2
                start_k = max(0, center - half)
                end_k = min(T_k, center + half + 1)

            q_chunk = q[:, :, i:end_i]
            k_chunk = k[:, :, start_k:end_k]
            v_chunk = v[:, :, start_k:end_k]

            chunk_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale

            Qi = end_i - i
            Kc = end_k - start_k

            if is_causal:
                # Create mask based on absolute positions
                q_pos = torch.arange(i, end_i, device=q.device).unsqueeze(1)      # (Qi, 1)
                k_pos = torch.arange(start_k, end_k, device=q.device).unsqueeze(0)  # (1, Kc)
                # Mask future OR outside window
                local_mask = (k_pos > q_pos) | (k_pos < (q_pos - self.window_size + 1))
            else:
                local_mask = torch.zeros(Qi, Kc, device=q.device, dtype=torch.bool)

            chunk_scores = chunk_scores.masked_fill(local_mask.view(1, 1, Qi, Kc), float('-inf'))

            if key_padding_mask is not None:
                chunk_key_mask = key_padding_mask[:, start_k:end_k]
                chunk_scores = chunk_scores.masked_fill(chunk_key_mask.view(B, 1, 1, Kc), float('-inf'))

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    am = attn_mask.bool()[i:end_i, start_k:end_k]
                    chunk_scores = chunk_scores.masked_fill(am.view(1, 1, Qi, Kc), float('-inf'))
                else:
                    am = attn_mask.bool()
                    am = am[..., i:end_i, start_k:end_k]
                    chunk_scores = chunk_scores.masked_fill(am, float('-inf'))

            chunk_weights = F.softmax(chunk_scores, dim=-1)
            if self.training and self.dropout_p > 0:
                chunk_weights = F.dropout(chunk_weights, p=self.dropout_p, training=True)

            chunk_output = torch.matmul(chunk_weights, v_chunk)
            output[:, :, i:end_i] = chunk_output

            if need_weights:
                all_weights.append((i, end_i, start_k, end_k, chunk_weights))

        combined_weights = None
        if need_weights:
            combined_weights = torch.zeros(B, H, T_q, T_k, device=q.device, dtype=output.dtype)
            for i, end_i, start_k, end_k, weights in all_weights:
                combined_weights[:, :, i:end_i, start_k:end_k] = weights

        return output, combined_weights

    # --------------------------------------------------------------------- #
    # Sliding-window attention (public entry)
    # --------------------------------------------------------------------- #
    def _sliding_window_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
        layer_state: Optional[Dict[str, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        q, k, v = self._project_qkv(query, key, value)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply RoPE BEFORE caching (critical fix)
        if self.use_rotary:
            q, k = self.rotary_emb(q, k)

        # Cache for self-attention only (after RoPE)
        if layer_state is not None and not self.cross_attention:
            if "k" in layer_state:
                k = torch.cat([layer_state["k"], k], dim=2)
                v = torch.cat([layer_state["v"], v], dim=2)
            layer_state["k"], layer_state["v"] = k, v

        # SDPA fast-path with proper mask shape handling
        if self.use_flash_sliding and self.backends.get("sdp", False) and hasattr(F, "scaled_dot_product_attention"):
            try:
                # Create mask with correct shape from the start
                base_mask = self._create_sliding_window_mask(q.size(2), k.size(2), q.device, is_causal)
                combined = base_mask.bool()

                if attn_mask is not None:
                    if attn_mask.dim() == 2:
                        combined = combined | attn_mask.bool()
                        combined = combined.unsqueeze(0).unsqueeze(0)
                    else:
                        combined = combined.unsqueeze(0).unsqueeze(0) | attn_mask.bool()
                else:
                    combined = combined.unsqueeze(0).unsqueeze(0)

                if key_padding_mask is not None:
                    combined = combined | key_padding_mask.view(B, 1, 1, k.size(2)).bool()

                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=combined,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,
                )
                weights = None
            except Exception as e:
                warnings.warn(f"[MultiAttention] SDPA sliding-window failed: {e}. Falling back.")
                out, weights = self._sliding_window_attention_efficient(
                    q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
                )
        else:
            out, weights = self._sliding_window_attention_efficient(
                q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
            )

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.out_proj(self.dropout(out))
        return out, weights, layer_state

    # --------------------------------------------------------------------- #
    # Frequency/DWT/AutoCorrelation wrappers
    # --------------------------------------------------------------------- #
    def _forward_frequency(
        self, query, key, value, attn_mask, key_padding_mask,
        is_causal, need_weights, layer_state, cu_seqlens
    ):
        out, weights = self.freq_attention(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights
        )
        return out, weights, layer_state

    def _forward_dwt(
        self, query, key, value, attn_mask, key_padding_mask,
        is_causal, need_weights, layer_state, cu_seqlens
    ):
        out, weights = self.dwt_attention(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights
        )
        return out, weights, layer_state

    def _forward_autocor(
        self, query, key, value, attn_mask, key_padding_mask,
        is_causal, need_weights, layer_state, cu_seqlens
    ):
        out, weights = self.freq_attention(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights
        )
        return out, weights, layer_state

    # --------------------------------------------------------------------- #
    # SoftPick
    # --------------------------------------------------------------------- #
    def _softpick_attention(
        self, query, key, value, attn_mask, key_padding_mask,
        is_causal, need_weights, layer_state, cu_seqlens
    ):
        if not self.backends.get("softpick", False):
            warnings.warn("[MultiAttention] SoftPick not available; falling back to standard attention.")
            return self._internal_attention(
                query, key, value, attn_mask, key_padding_mask,
                is_causal, need_weights, layer_state, cu_seqlens
            )

        from ..third_party.flash_softpick_attn import parallel_softpick_attn
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        q, k, v = self._project_qkv(query, key, value)

        # Apply RoPE BEFORE caching
        if self.use_rotary:
            q_r, k_r = q.transpose(1, 2), k.transpose(1, 2)
            q_r, k_r = self.rotary_emb(q_r, k_r)
            q, k = q_r.transpose(1, 2), k_r.transpose(1, 2)

        # Cache for self-attn (after RoPE)
        if layer_state is not None and not self.cross_attention:
            if "k" in layer_state:
                k = torch.cat([layer_state["k"], k], dim=1)
                v = torch.cat([layer_state["v"], v], dim=1)
            layer_state["k"] = k
            layer_state["v"] = v

        try:
            if cu_seqlens is None:
                out = parallel_softpick_attn(q=q, k=k, v=v, scale=self.scale, cu_seqlens=None, head_first=False)
                out = out.contiguous().view(B, T_q, self.d_model)
            else:
                qf = q.view(B * T_q, self.n_heads, self.head_dim)
                kf = k.view(B * T_k, self.n_heads, self.head_dim)
                vf = v.view(B * T_k, self.n_heads, self.head_dim)
                out = parallel_softpick_attn(q=qf, k=kf, v=vf, scale=self.scale, cu_seqlens=cu_seqlens, head_first=True)
                out = out.view(B, T_q, self.n_heads, self.head_dim).contiguous().view(B, T_q, self.d_model)

            return self.out_proj(self.dropout(out)), None, layer_state
        except Exception as e:
            warnings.warn(f"[MultiAttention] SoftPick failed: {e}. Falling back to standard attention.")
            return self._internal_attention(
                query, key, value, attn_mask, key_padding_mask,
                is_causal, need_weights, layer_state, cu_seqlens
            )

    # --------------------------------------------------------------------- #
    # Standard / ProbSparse internal
    # --------------------------------------------------------------------- #
    def _internal_attention(
        self, query, key, value, attn_mask, key_padding_mask,
        is_causal, need_weights, layer_state, *_,
    ):
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        q, k, v = self._project_qkv(query, key, value)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply RoPE BEFORE caching
        if self.use_rotary:
            q, k = self.rotary_emb(q, k)

        # Cache for self-attn only (after RoPE)
        if layer_state is not None and not self.cross_attention:
            if "k" in layer_state:
                k = torch.cat([layer_state["k"], k], dim=2)
                v = torch.cat([layer_state["v"], v], dim=2)
            layer_state["k"], layer_state["v"] = k, v

        if self.attention_type == "standard":
            out, weights = self._standard_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)
        else:
            out, weights = self._prob_sparse_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(self.dropout(out)), weights, layer_state

    def _standard_attention(self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights):
        B, H, T_q, T_k = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal for self-attn
        if is_causal and not self.cross_attention:
            causal_mask = torch.triu(torch.ones(T_q, T_k, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.view(1, 1, T_q, T_k), float("-inf"))

        scores = self._apply_masks(scores, attn_mask, key_padding_mask, B, T_q, T_k)

        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p, training=True)
        out = torch.matmul(weights, v)
        return out, (weights if need_weights else None)

    def _prob_sparse_attention(self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights):
        """
        Stable prob-sparse attention with proper handling of small sequences.
        """
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        # Calculate sampling parameters
        u = max(8, int(self.prob_sparse_factor * math.sqrt(max(T_q, 2))))
        u = min(u, T_q)
        sample_k = max(8, int(self.prob_sparse_factor * T_k))
        sample_k = min(sample_k, T_k)

        if sample_k < 1 or u >= T_q:
            # Degenerate case: fall back to standard attention
            return self._standard_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)

        # Sample keys
        idx = torch.randperm(T_k, device=q.device)[:sample_k]
        k_sample = k[:, :, idx, :]

        # Compute sparsity scores
        scores_sample = torch.matmul(q, k_sample.transpose(-2, -1)) * self.scale
        sparsity = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)
        _, top_idx = torch.topk(sparsity, k=u, dim=-1)

        # Gather top queries
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        top_q = torch.gather(q, 2, top_idx_exp)

        # Compute full attention for top queries
        scores = torch.matmul(top_q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask for self-attn
        if is_causal and not self.cross_attention:
            q_pos = top_idx.unsqueeze(-1)
            k_pos = torch.arange(T_k, device=q.device).view(1, 1, 1, T_k)
            causal_mask = k_pos > q_pos
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                am = attn_mask.bool().view(1, 1, T_q, T_k)
                am = torch.gather(am.expand(B, H, T_q, T_k), 2, 
                                 top_idx.unsqueeze(-1).expand(B, H, u, T_k))
            else:
                am = attn_mask.bool()
                am = torch.gather(am, 2, top_idx.unsqueeze(-1).expand(B, H, u, T_k))
            scores = scores.masked_fill(am, float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, T_k), float("-inf"))

        # Compute attention weights
        scores = scores - scores.max(dim=-1, keepdim=True)[0]
        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p, training=True)
        top_out = torch.matmul(weights, v)

        # Scatter back to full length
        output = torch.zeros_like(q)
        output.scatter_(2, top_idx_exp, top_out)

        # Fill non-selected positions with mean
        if u < T_q:
            mask = torch.zeros(B, H, T_q, device=q.device, dtype=torch.bool)
            mask.scatter_(2, top_idx, True)
            mean_v = v.mean(dim=2, keepdim=True).expand(B, H, T_q, D)
            output = torch.where(mask.unsqueeze(-1), output, mean_v)

        if need_weights:
            full_weights = torch.zeros(B, H, T_q, T_k, device=q.device, dtype=weights.dtype)
            full_weights.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, T_k), weights)
            return output, full_weights
        return output, None

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    def reset_cache(self):
        """Reset any cached attention patterns."""
        for name in ["freq_attention", "dwt_attention"]:
            attn = getattr(self, name, None)
            if attn is not None and hasattr(attn, "cache"):
                attn.cache.clear()
