import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


def _get_available_backends():
    """Check what optimized attention backends are available"""
    backends = {"flash": False, "xformers": False, "sdp": False}

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
    return backends


class MultiAttention(nn.Module):
    """
    Unified attention module supporting multiple strategies:
    - standard: Regular scaled dot-product attention
    - prob_sparse: ProbSparse attention from Informer
    - frequency: Frequency domain attention from FEDformer (uses external FrequencyAttention)
    - dwt: DWT attention (uses external DWTAttention)

    Automatically uses optimized backends (Flash/xFormers/SDP) when possible.
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

        # For standard/prob_sparse attention, we need our own projections
        if attention_type in ["standard", "prob_sparse"]:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        # Rotary embeddings (for standard/prob_sparse, not typically used in cross-attention)
        self.use_rotary = (
            use_rotary
            and attention_type in ["standard", "prob_sparse"]
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
            from .fed import AutoCorrelationLayer, AutoCorrelation

            # 1. Create the AutoCorrelation mechanism
            autocorr_mechanism = AutoCorrelation(
                mask_flag=True,
                factor=1,  # Controls top-k selection
                attention_dropout=0.1,
                output_attention=False,
            )
            self.freq_attention = AutoCorrelationLayer(
                correlation=autocorr_mechanism,
                d_model=d_model,
                n_heads=n_heads,
            )

        # Check available backends for standard attention
        if attention_type in ["standard", "prob_sparse"]:
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:

        # Handle self-attention
        if key is None:
            key = query
        if value is None:
            value = key

        # Route to appropriate attention implementation
        if self.attention_type == "frequency":
            # Use external FrequencyAttention class
            out, weights = self.freq_attention(
                query, key, value, attn_mask, key_padding_mask, is_causal, need_weights
            )
            return out, weights, layer_state

        elif self.attention_type == "dwt":
            # Use external DWTAttention class
            out, weights = self.dwt_attention(
                query, key, value, attn_mask, key_padding_mask, is_causal, need_weights
            )
            return out, weights, layer_state
        elif self.attention_type == "autocor":
            # Use external AutocorAttention class
            out, weights = self.freq_attention(query, key, value, attn_mask)
            return out, weights, layer_state

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
        self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
    ):
        """ProbSparse attention from Informer paper"""
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        # Calculate number of top queries to keep
        u = max(int(self.prob_sparse_factor * math.log(T_k)), int(T_q * 0.3))
        u = min(u, T_q)

        # Compute measurement M(qi, K)
        scale = 1.0 / math.sqrt(D)
        scores_sample = torch.matmul(q * scale, k.transpose(-2, -1))  # [B, H, T_q, T_k]

        # Compute M(qi, K) = ln(Σ exp(qi·kj/√d)) - 1/Lk * Σ(qi·kj/√d)
        max_scores = scores_sample.max(dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(scores_sample - max_scores)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)
        mean_scores = scores_sample.mean(dim=-1, keepdim=True)

        M = torch.log(sum_exp) + max_scores - mean_scores  # [B, H, T_q, 1]
        M = M.squeeze(-1)  # [B, H, T_q]

        # Select top-u queries
        _, top_indices = torch.topk(M, u, dim=-1)  # [B, H, u]

        # Gather top queries
        top_q = torch.gather(
            q, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        )  # [B, H, u, D]

        # Compute attention for selected queries
        scores = torch.matmul(top_q * scale, k.transpose(-2, -1))  # [B, H, u, T_k]

        # Apply masks to selected scores
        if is_causal and not self.cross_attention:
            # Create causal mask for selected positions
            causal_mask = torch.tril(
                torch.ones(u, T_k, device=scores.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.view(B, 1, 1, T_k), float("-inf")
            )

        weights = F.softmax(scores, dim=-1)
        if self.dropout_p > 0 and self.training:
            weights = F.dropout(weights, p=self.dropout_p)

        # Compute output for selected queries
        top_out = torch.matmul(weights, v)  # [B, H, u, D]

        # Scatter back to full size
        out = torch.zeros_like(q)
        out.scatter_(2, top_indices.unsqueeze(-1).expand(-1, -1, -1, D), top_out)

        return out, weights if need_weights else None

    def reset_cache(self):
        """Reset any internal caches (useful for inference)"""
        if hasattr(self, "freq_attention"):
            if hasattr(self.freq_attention, "cache"):
                self.freq_attention.cache.clear()
        if hasattr(self, "dwt_attention"):
            if hasattr(self.dwt_attention, "cache"):
                self.dwt_attention.cache.clear()
