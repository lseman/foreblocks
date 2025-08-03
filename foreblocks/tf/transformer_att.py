import math
import warnings
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
        softpick_chunk_size: int = 128,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_type = attention_type
        self.dropout_p = dropout
        self.cross_attention = cross_attention
        self.softpick_chunk_size = softpick_chunk_size
        self.prob_sparse_factor = prob_sparse_factor
        self.scale = self.head_dim ** -0.5

        if attention_type in ["standard", "prob_sparse", "softpick"]:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        self.use_rotary = (
            use_rotary
            and attention_type in ["standard", "prob_sparse", "softpick"]
            and not cross_attention
        )
        if self.use_rotary:
            from .embeddings import RotaryEmbedding
            self.rotary_emb = RotaryEmbedding(self.head_dim)

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

        self.backends = _get_available_backends() if attention_type in ["standard", "prob_sparse", "softpick"] else {}
        self.attention_map = {
            "standard": self._internal_attention,
            "prob_sparse": self._internal_attention,
            "frequency": self._forward_frequency,
            "dwt": self._forward_dwt,
            "autocor": self._forward_autocor,
            "softpick": self._softpick_attention,
        }
        print(f"[MultiAttention] Initialized with attention type: {self.attention_type}, "
              f"available backends: {self.backends}")

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

        if self.attention_type not in self.attention_map:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")
        
        return self.attention_map[self.attention_type](
            query, key, value, attn_mask, key_padding_mask,
            is_causal, need_weights, layer_state, cu_seqlens
        )

    def _project_qkv(self, query, key, value):
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(B, T_k, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(B, T_k, self.n_heads, self.head_dim)
        return q, k, v

    def _apply_masks(self, scores, attn_mask, key_padding_mask, B, T_q, T_k):
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(1, 1, T_q, T_k)
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, T_k), float("-inf"))
        return scores

    def _forward_frequency(self, q, k, v, *args, **kwargs):
        out, weights = self.freq_attention(q, k, v, *args, **kwargs[:4])
        return out, weights, kwargs[4]  # return layer_state

    def _forward_dwt(self, q, k, v, *args, **kwargs):
        out, weights = self.dwt_attention(q, k, v, *args, **kwargs[:4])
        return out, weights, kwargs[4]

    def _forward_autocor(self, q, k, v, *args, **kwargs):
        out, weights = self.freq_attention(q, k, v, *args[:1])  # attn_mask
        return out, weights, kwargs[4]

    def _softpick_attention(self, query, key, value, attn_mask, key_padding_mask,
                            is_causal, need_weights, layer_state, cu_seqlens):
        if not self.backends.get("softpick", False):
            warnings.warn("[MultiAttention] SoftPick not available, falling back to standard attention.")
            return self._internal_attention(query, key, value, attn_mask, key_padding_mask,
                                            is_causal, need_weights, layer_state, cu_seqlens)

        from ..third_party.flash_softpick_attn import parallel_softpick_attn
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape
        q, k, v = self._project_qkv(query, key, value)

        if layer_state and not self.cross_attention:
            cached_k = layer_state.get("k")
            cached_v = layer_state.get("v")
            if cached_k is not None and cached_v is not None:
                k = torch.cat([cached_k, k], dim=1)
                v = torch.cat([cached_v, v], dim=1)
            layer_state["k"] = k
            layer_state["v"] = v

        if self.use_rotary:
            q, k = self.rotary_emb(q.transpose(1, 2), k.transpose(1, 2))
            q, k = q.transpose(1, 2), k.transpose(1, 2)

        try:
            if cu_seqlens is None:
                out = parallel_softpick_attn(q=q, k=k, v=v, scale=self.scale, cu_seqlens=None, head_first=False)
                out = out.contiguous().view(B, T_q, self.d_model)
            else:
                q = q.view(B * T_q, self.n_heads, self.head_dim)
                k = k.view(B * T_k, self.n_heads, self.head_dim)
                v = v.view(B * T_k, self.n_heads, self.head_dim)
                out = parallel_softpick_attn(q=q, k=k, v=v, scale=self.scale, cu_seqlens=cu_seqlens, head_first=True)
                out = out.view(B, T_q, self.n_heads, self.head_dim).contiguous().view(B, T_q, self.d_model)

            return self.out_proj(self.dropout(out)), None, layer_state
        except Exception as e:
            warnings.warn(f"[MultiAttention] SoftPick failed: {e}. Falling back to standard attention.")
            return self._internal_attention(query, key, value, attn_mask, key_padding_mask,
                                            is_causal, need_weights, layer_state)

    def _internal_attention(self, query, key, value, attn_mask, key_padding_mask,
                            is_causal, need_weights, layer_state, *_):
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape
        q, k, v = self._project_qkv(query, key, value)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if layer_state and not self.cross_attention:
            cached_k = layer_state.get("k")
            cached_v = layer_state.get("v")
            if cached_k is not None and cached_v is not None:
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            layer_state["k"] = k
            layer_state["v"] = v

        if self.use_rotary:
            q, k = self.rotary_emb(q, k)

        if self.attention_type == "standard":
            out, weights = self._standard_attention(q, k, v, attn_mask, key_padding_mask,
                                                    is_causal, need_weights)
        elif self.attention_type == "prob_sparse":
            out, weights = self._prob_sparse_attention(q, k, v, attn_mask, key_padding_mask,
                                                       is_causal, need_weights)
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(self.dropout(out)), weights, layer_state

    def _standard_attention(self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights):
        B, H, T_q, T_k = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if is_causal and not self.cross_attention:
            causal_mask = torch.tril(torch.ones(T_q, T_k, device=scores.device)).bool()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        scores = self._apply_masks(scores, attn_mask, key_padding_mask, B, T_q, T_k)
        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p)
        out = torch.matmul(weights, v)
        return out, weights if need_weights else None

    def _prob_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)
        u = max(1, min(int(self.prob_sparse_factor * math.log(T_k)), T_q))
        sample_k = max(1, int(self.prob_sparse_factor * T_k))

        # Step 1: Randomly sample keys for estimating sparsity
        sample_indices = torch.randperm(T_k, device=q.device)[:sample_k]
        k_sample = k[:, :, sample_indices, :]  # [B, H, sample_k, D]
        scores_sample = torch.matmul(q, k_sample.transpose(-2, -1)) * self.scale  # [B, H, T_q, sample_k]
        sparsity_score = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)  # [B, H, T_q]

        # Step 2: Select top-u queries per head
        _, top_indices = torch.topk(sparsity_score, k=u, dim=-1)  # [B, H, u]
        D_k = k.shape[-1]

        # Gather top queries
        top_q = torch.gather(
            q, 2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, D_k)
        )  # [B, H, u, D]

        # Step 3: Compute attention for selected queries
        attn_scores = torch.matmul(top_q, k.transpose(-2, -1)) * self.scale  # [B, H, u, T_k]

        if is_causal and not self.cross_attention:
            q_pos = top_indices.unsqueeze(-1)  # [B, H, u, 1]
            k_pos = torch.arange(T_k, device=q.device).view(1, 1, 1, T_k)
            causal_mask = q_pos >= k_pos
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        if attn_mask is not None:
            expanded_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
            selected_mask = torch.gather(expanded_mask, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, T_k))
            attn_scores = attn_scores.masked_fill(selected_mask == 0, float("-inf"))

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.view(B, 1, 1, T_k), float("-inf")
            )

        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]
        attn_weights = F.softmax(attn_scores, dim=-1)
        if self.dropout_p > 0 and q.requires_grad:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        top_output = torch.matmul(attn_weights, v)  # [B, H, u, D]

        # Step 4: Scatter results back into full output shape
        output = torch.zeros_like(q)  # [B, H, T_q, D]
        output.scatter_(
            2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, D),
            top_output
        )

        # Fill remaining tokens with mean of v
        if u < T_q:
            mask = torch.zeros(B, H, T_q, dtype=torch.bool, device=q.device)
            mask.scatter_(2, top_indices, True)
            mean_v = v.mean(dim=2, keepdim=True).expand(B, H, T_q, D)
            output = torch.where(mask.unsqueeze(-1), output, mean_v)

        if need_weights:
            full_weights = torch.zeros(B, H, T_q, T_k, device=q.device, dtype=attn_weights.dtype)
            full_weights.scatter_(
                2, top_indices.unsqueeze(-1).expand(-1, -1, -1, T_k), attn_weights
            )
            return output, full_weights
        return output, None

    def reset_cache(self):
        for name in ["freq_attention", "dwt_attention"]:
            attn = getattr(self, name, None)
            if attn and hasattr(attn, "cache"):
                attn.cache.clear()
