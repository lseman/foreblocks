import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_available_backends():
    """Check what optimized attention backends are available"""
    backends = {"flash": False, "xformers": False, "sdp": False, "softpick": False}

    try:
        from flash_attn import flash_attn_func

        backends["flash"] = True
    except ImportError:
        pass

    try:
        import xformers.ops

        backends["xformers"] = True
    except ImportError:
        pass

    backends["sdp"] = hasattr(F, "scaled_dot_product_attention")

    # Check for SoftPick availability
    try:
        # Assuming the softpick code is in a module called 'softpick_attention'
        from ..third_party.flash_softpick_attn import parallel_softpick_attn

        backends["softpick"] = True
    except ImportError:
        pass

    return backends


class MultiAttention(nn.Module):
    """
    Unified attention module supporting multiple strategies:
    - standard: Regular scaled dot-product attention
    - prob_sparse: ProbSparse attention from Informer
    - frequency: Frequency domain attention from FEDformer (uses external FrequencyAttention)
    - dwt: DWT attention (uses external DWTAttention)
    - softpick: SoftPick attention with Triton kernels

    Automatically uses optimized backends (Flash/xFormers/SDP/SoftPick) when possible.
    Maintains layer_state for efficient KV caching during inference.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attention_type: str = "standard",
        prob_sparse_factor: float = 0.4,
        freq_modes: int = 32,
        use_rotary: bool = False,
        max_seq_len: int = 4096,
        cross_attention: bool = False,
        softpick_chunk_size: int = 128,  # New parameter for SoftPick
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_type = attention_type
        self.prob_sparse_factor = prob_sparse_factor
        self.dropout_p = dropout
        self.cross_attention = cross_attention
        self.softpick_chunk_size = softpick_chunk_size

        # For standard/prob_sparse/softpick attention, we need our own projections
        if attention_type in ["standard", "prob_sparse", "softpick"]:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        # Rotary embeddings (for standard/prob_sparse/softpick, not typically used in cross-attention)
        self.use_rotary = (
            use_rotary
            and attention_type in ["standard", "prob_sparse", "softpick"]
            and not cross_attention
        )
        if self.use_rotary:
            from .embeddings import RotaryEmbedding

            self.rotary_emb = RotaryEmbedding(self.head_dim)

        # For frequency/dwt attention, use external classes
        if attention_type == "frequency":
            from .fed import FrequencyAttention

            self.freq_attention = FrequencyAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                modes=freq_modes,
            )
        elif attention_type == "dwt":
            from .fed import DWTAttention

            self.dwt_attention = DWTAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                modes=freq_modes,
            )
        elif attention_type == "autocor":
            from .fed import AutoCorrelation, AutoCorrelationLayer

            autocorr_mechanism = AutoCorrelation(
                mask_flag=True,
                factor=1,
                attention_dropout=0.1,
                output_attention=False,
            )
            self.freq_attention = AutoCorrelationLayer(
                correlation=autocorr_mechanism,
                d_model=d_model,
                n_heads=n_heads,
            )

        # Check available backends for attention types that support them
        if attention_type in ["standard", "prob_sparse", "softpick"]:
            self.backends = _get_available_backends()
            print(f"[MultiAttention] Type: {attention_type}, Backends: {self.backends}")
        else:
            print(f"[MultiAttention] Type: {attention_type} (external implementation)")

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
        cu_seqlens: Optional[torch.LongTensor] = None,  # New parameter for SoftPick
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        # Handle self-attention
        if key is None:
            key = query
        if value is None:
            value = key

        # Route to appropriate attention implementation
        if self.attention_type == "frequency":
            out, weights = self.freq_attention(
                query, key, value, attn_mask, key_padding_mask, is_causal, need_weights
            )
            return out, weights, layer_state

        elif self.attention_type == "dwt":
            out, weights = self.dwt_attention(
                query, key, value, attn_mask, key_padding_mask, is_causal, need_weights
            )
            return out, weights, layer_state

        elif self.attention_type == "autocor":
            out, weights = self.freq_attention(query, key, value, attn_mask)
            return out, weights, layer_state

        elif self.attention_type == "softpick":
            return self._softpick_attention(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
                layer_state,
                cu_seqlens,
            )

        else:
            # Handle standard and prob_sparse with KV caching
            return self._internal_attention(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
                layer_state,
            )

    def _softpick_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        need_weights: bool,
        layer_state: Optional[Dict[str, torch.Tensor]],
        cu_seqlens: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """SoftPick attention implementation (supports packed and non-packed modes)"""

        if not self.backends.get("softpick", False):
            print(
                "[MultiAttention] SoftPick not available, falling back to standard attention"
            )
            return self._internal_attention(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
                layer_state,
            )

        from ..third_party.flash_softpick_attn import parallel_softpick_attn

        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(B, T_k, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(B, T_k, self.n_heads, self.head_dim)

        if layer_state is not None and not self.cross_attention:
            cached_k = layer_state.get("k")
            cached_v = layer_state.get("v")

            if cached_k is not None and cached_v is not None:
                k = torch.cat([cached_k, k], dim=1)
                v = torch.cat([cached_v, v], dim=1)

            layer_state["k"] = k
            layer_state["v"] = v

        if self.use_rotary:
            q_rot = q.transpose(1, 2)  # [B, H, T, D]
            k_rot = k.transpose(1, 2)
            q_rot, k_rot = self.rotary_emb(q_rot, k_rot)
            q = q_rot.transpose(1, 2)
            k = k_rot.transpose(1, 2)

        if attn_mask is not None or key_padding_mask is not None:
            print(
                "[MultiAttention] Warning: SoftPick may not fully support custom masks"
            )

        scale = 1.0 / math.sqrt(self.head_dim)

        try:
            if cu_seqlens is None:
                # NON-PACKED MODE (standard [B, T, H, D])
                out = parallel_softpick_attn(
                    q=q, k=k, v=v, scale=scale, cu_seqlens=None, head_first=False
                )
                B_, T_real, H, D = out.shape
                assert B_ == B and H == self.n_heads and D == self.head_dim
                out = out.contiguous().view(B, T_real, self.d_model)

            else:
                # PACKED MODE (requires flattening to [B*T, H, D])
                q = q.reshape(B * T_q, self.n_heads, self.head_dim)
                k = k.reshape(B * T_k, self.n_heads, self.head_dim)
                v = v.reshape(B * T_k, self.n_heads, self.head_dim)

                out = parallel_softpick_attn(
                    q=q, k=k, v=v, scale=scale, cu_seqlens=cu_seqlens, head_first=True
                )
                # Output will be [B*T_q, H, D] → reshape to [B, T_q, H, D]
                out = out.view(B, T_q, self.n_heads, self.head_dim)
                out = out.contiguous().view(B, T_q, self.d_model)

            out = self.out_proj(out)
            out = self.dropout(out)

            weights = None
            if need_weights:
                print(
                    "[MultiAttention] Warning: SoftPick doesn't support returning attention weights"
                )

            return out, weights, layer_state

        except Exception as e:
            print(
                f"[MultiAttention] SoftPick failed with error: {e}, falling back to standard attention"
            )
            return self._internal_attention(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
                layer_state,
            )

    def _internal_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        need_weights: bool,
        layer_state: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        # Project Q, K, V
        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        # Handle KV caching for efficient inference (typically not used in cross-attention)
        if layer_state is not None and not self.cross_attention:
            # Retrieve cached K, V
            cached_k = layer_state.get("k")  # [B, H, T_prev, D]
            cached_v = layer_state.get("v")  # [B, H, T_prev, D]

            if cached_k is not None and cached_v is not None:
                # Concatenate new K, V with cached ones
                k = torch.cat([cached_k, k], dim=2)  # [B, H, T_prev + T_k, D]
                v = torch.cat([cached_v, v], dim=2)  # [B, H, T_prev + T_k, D]

            # Update cache with new K, V
            layer_state["k"] = k
            layer_state["v"] = v

        # Apply rotary embeddings
        if self.use_rotary:
            q, k = self.rotary_emb(q, k)

        # Choose attention strategy
        if self.attention_type == "standard":
            out, weights = self._standard_attention(
                q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
            )
        elif self.attention_type == "prob_sparse":
            out, weights = self._prob_sparse_attention(
                q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
            )
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out, weights, layer_state

    def _try_optimized_attention(
        self, q, k, v, is_causal, need_weights, attn_mask, key_padding_mask
    ):
        """Try to use optimized backends if possible"""
        if need_weights or attn_mask is not None or key_padding_mask is not None:
            return None

        # Try Flash Attention
        if self.backends["flash"]:
            try:
                from flash_attn import flash_attn_func

                if q.dtype in (torch.float16, torch.bfloat16):
                    return flash_attn_func(
                        q,
                        k,
                        v,
                        dropout_p=self.dropout_p if self.training else 0.0,
                        causal=is_causal and not self.cross_attention,
                    )
            except Exception:
                pass

        # Try PyTorch SDP
        if self.backends["sdp"]:
            try:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=is_causal and not self.cross_attention,
                )
            except Exception:
                pass

        # Try xFormers
        if self.backends["xformers"]:
            try:
                import xformers.ops as xops

                bias = (
                    None
                    if not is_causal or self.cross_attention
                    else xops.LowerTriangularMask()
                )
                return xops.memory_efficient_attention(
                    q, k, v, attn_bias=bias, p=self.dropout_p if self.training else 0.0
                )
            except Exception:
                pass

        return None

    def _standard_attention(
        self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
    ):
        """Standard scaled dot-product attention with optimized backend fallback"""

        # Try optimized backends first
        optimized_out = self._try_optimized_attention(
            q, k, v, is_causal, need_weights, attn_mask, key_padding_mask
        )
        if optimized_out is not None:
            return optimized_out, None

        # Manual implementation
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply masks
        B, H, T_q, T_k = scores.shape

        if is_causal and not self.cross_attention:
            causal_mask = torch.tril(
                torch.ones(T_q, T_k, device=scores.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(1, 1, T_q, T_k)
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.view(B, 1, 1, T_k), float("-inf")
            )

        # Softmax and dropout
        weights = F.softmax(scores, dim=-1)
        if self.dropout_p > 0 and self.training:
            weights = F.dropout(weights, p=self.dropout_p)

        out = torch.matmul(weights, v)
        return out, weights if need_weights else None

    def _prob_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
        prob_sparse_factor: float = 5.0,
        dropout_p: float = 0.0,
        need_weights: bool = False,
        cross_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Optimized ProbSparse attention (Informer-style) with vectorized logic.

        Args:
            q: [B, H, T_q, D]
            k: [B, H, T_k, D]
            v: [B, H, T_k, D]
            attn_mask: optional attention mask [T_q, T_k] or [B, H, T_q, T_k]
            key_padding_mask: [B, T_k] mask
            is_causal: whether to apply causal masking
            prob_sparse_factor: controls number of top queries sampled
            dropout_p: attention dropout
            need_weights: whether to return attention weights
            cross_attention: disable causal masking on cross-attention

        Returns:
            output: [B, H, T_q, D]
            attn_weights: [B, H, T_q, T_k] or None
        """
        B, H, T_q, D = q.shape
        T_k = k.size(2)
        scale = 1.0 / math.sqrt(D)

        # Compute number of top queries u
        u = min(T_q, max(int(prob_sparse_factor * math.log(T_k)), int(0.3 * T_q)))

        if u >= T_q * 0.9:
            # Fallback to full attention (basic scaled dot-product)
            scores = torch.matmul(q * scale, k.transpose(-2, -1))  # [B, H, T_q, T_k]

            if is_causal and not cross_attention:
                causal_mask = torch.tril(torch.ones(T_q, T_k, device=q.device)).bool()
                scores = scores.masked_fill(~causal_mask, float("-inf"))

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    scores = scores.masked_fill(attn_mask == 0, float("-inf"))
                else:
                    scores = scores.masked_fill(attn_mask == 0, float("-inf"))

            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )

            attn_weights = F.softmax(scores, dim=-1)
            if dropout_p > 0.0 and q.requires_grad:
                attn_weights = F.dropout(attn_weights, p=dropout_p)
            output = torch.matmul(attn_weights, v)
            return output, attn_weights if need_weights else None

        # --- ProbSparse attention ---
        sample_k = max(1, min(int(self.prob_sparse_factor * T_k), T_k))  # Fix 1
        sample_indices = torch.randperm(T_k, device=k.device)[:sample_k]
        k_sample = k[:, :, sample_indices, :]  # [B, H, sample_k, D]

        scores_sample = torch.matmul(
            q * scale, k_sample.transpose(-2, -1)
        )  # [B, H, T_q, sample_k]
        sparsity_score = scores_sample.max(dim=-1)[0] - scores_sample.mean(
            dim=-1
        )  # [B, H, T_q]

        u = max(1, min(int(self.prob_sparse_factor * math.log(T_k)), T_q))  # Fix 2

        # Select top-u queries with highest sparsity
        _, top_indices = torch.topk(sparsity_score, k=u, dim=-1)  # [B, H, u]
        top_q = torch.gather(
            q, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        )  # [B, H, u, D]

        # Compute attention for selected top queries
        attn_scores = torch.matmul(top_q * scale, k.transpose(-2, -1))  # [B, H, u, T_k]

        if is_causal and not cross_attention:
            q_pos = top_indices.unsqueeze(-1)  # [B, H, u, 1]
            k_pos = torch.arange(T_k, device=q.device).view(1, 1, 1, T_k)
            causal_mask = q_pos >= k_pos  # [B, H, u, T_k]
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        if attn_mask is not None and attn_mask.dim() == 2:
            selected_attn_mask = (
                attn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, T_q, T_k)
            )
            selected_attn_mask = torch.gather(
                selected_attn_mask, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, T_k)
            )
            attn_scores = attn_scores.masked_fill(
                selected_attn_mask == 0, float("-inf")
            )

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Stable softmax
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]
        attn_weights = F.softmax(attn_scores, dim=-1)
        if dropout_p > 0.0 and q.requires_grad:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        top_out = torch.matmul(attn_weights, v)  # [B, H, u, D]

        # Scatter result back to full sequence
        output = torch.zeros_like(q)
        output.scatter_(2, top_indices.unsqueeze(-1).expand(-1, -1, -1, D), top_out)

        # Fill non-selected queries with mean of all values
        if u < T_q:
            mask = torch.zeros(B, H, T_q, dtype=torch.bool, device=q.device)
            mask.scatter_(2, top_indices, True)
            mean_v = v.mean(dim=2, keepdim=True).expand(B, H, T_q, D)
            output = torch.where(mask.unsqueeze(-1), output, mean_v)

        return output, attn_weights if need_weights else None

    def reset_cache(self):
        """Reset any internal caches (useful for inference)"""
        if hasattr(self, "freq_attention"):
            if hasattr(self.freq_attention, "cache"):
                self.freq_attention.cache.clear()
        if hasattr(self, "dwt_attention"):
            if hasattr(self.dwt_attention, "cache"):
                self.dwt_attention.cache.clear()
