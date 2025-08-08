import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_available_backends():
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
    try:
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
        # New sliding window parameters
        window_size: int = 64,
        global_attention_ratio: float = 0.1,
        chunk_size: int = 1024,
        use_flash_sliding: bool = True,
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

        self.use_rotary = (
            use_rotary
            and attention_type in ["standard", "prob_sparse", "softpick", "sliding_window"]
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

        self.backends = _get_available_backends() if attention_type in ["standard", "prob_sparse", "softpick", "sliding_window"] else {}
        self.attention_map = {
            "standard": self._internal_attention,
            "prob_sparse": self._internal_attention,
            "frequency": self._forward_frequency,
            "dwt": self._forward_dwt,
            "autocor": self._forward_autocor,
            "softpick": self._softpick_attention,
            "sliding_window": self._sliding_window_attention,
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

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device, is_causal: bool = True) -> torch.Tensor:
        """Create efficient sliding window mask."""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        for i in range(seq_len):
            # Define window boundaries
            start_pos = max(0, i - self.window_size + 1) if is_causal else max(0, i - self.window_size // 2)
            end_pos = min(seq_len, i + 1) if is_causal else min(seq_len, i + self.window_size // 2 + 1)
            
            # Mark positions outside window as True (to be masked)
            mask[i, :start_pos] = True
            if not is_causal:
                mask[i, end_pos:] = True
            else:
                mask[i, i + 1:] = True  # Causal mask
                
        return mask

    def _add_global_tokens(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Add global attention tokens for long-range dependencies."""
        if self.global_attention_ratio <= 0:
            return q, k, 0
            
        num_global = max(1, int(seq_len * self.global_attention_ratio))
        
        # Select global positions (recent + evenly spaced)
        recent_positions = list(range(max(0, seq_len - num_global // 2), seq_len))
        remaining = num_global - len(recent_positions)
        
        if remaining > 0:
            step = max(1, seq_len // (remaining + 1))
            spaced_positions = list(range(0, seq_len - len(recent_positions), step))[:remaining]
            global_positions = spaced_positions + recent_positions
        else:
            global_positions = recent_positions
            
        global_positions = sorted(list(set(global_positions))[:num_global])
        
        return q, k, len(global_positions)

    def _sliding_window_attention_efficient(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Memory-efficient sliding window attention."""
        B, H, T, D = q.shape
        
        # For very long sequences, use chunked computation
        if T > self.chunk_size:
            return self._chunked_sliding_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)
        
        # Standard sliding window for moderate sequences
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create and apply sliding window mask
        window_mask = self._create_sliding_window_mask(T, q.device, is_causal)
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply additional masks
        scores = self._apply_masks(scores, attn_mask, key_padding_mask, B, T, T)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)
        
        out = torch.matmul(attn_weights, v)
        
        return out, attn_weights if need_weights else None

    def _chunked_sliding_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process very long sequences in chunks to save memory."""
        B, H, T, D = q.shape
        output = torch.zeros_like(q)
        all_weights = [] if need_weights else None
        
        for i in range(0, T, self.chunk_size):
            end_i = min(i + self.chunk_size, T)
            
            # Define context window for this chunk
            if is_causal:
                start_k = max(0, i - self.window_size)
                end_k = end_i
            else:
                center = (i + end_i) // 2
                start_k = max(0, center - self.window_size // 2)
                end_k = min(T, center + self.window_size // 2)
            
            # Extract chunks
            q_chunk = q[:, :, i:end_i]  # [B, H, chunk_len, D]
            k_chunk = k[:, :, start_k:end_k]  # [B, H, context_len, D]
            v_chunk = v[:, :, start_k:end_k]  # [B, H, context_len, D]
            
            # Compute attention for this chunk
            chunk_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale
            
            # Create local mask for this chunk
            chunk_len = end_i - i
            context_len = end_k - start_k
            
            if is_causal:
                # Causal mask for the chunk
                q_positions = torch.arange(i, end_i, device=q.device).unsqueeze(1)
                k_positions = torch.arange(start_k, end_k, device=q.device).unsqueeze(0)
                local_mask = q_positions < k_positions
            else:
                # Symmetric window mask
                local_mask = torch.zeros(chunk_len, context_len, device=q.device, dtype=torch.bool)
                for j in range(chunk_len):
                    global_pos = i + j
                    local_start = max(0, global_pos - self.window_size // 2 - start_k)
                    local_end = min(context_len, global_pos + self.window_size // 2 + 1 - start_k)
                    local_mask[j, :local_start] = True
                    local_mask[j, local_end:] = True
            
            chunk_scores = chunk_scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Apply additional masks if provided
            if key_padding_mask is not None:
                chunk_key_mask = key_padding_mask[:, start_k:end_k]
                chunk_scores = chunk_scores.masked_fill(
                    chunk_key_mask.view(B, 1, 1, context_len), float('-inf')
                )
            
            chunk_weights = F.softmax(chunk_scores, dim=-1)
            if self.training and self.dropout_p > 0:
                chunk_weights = F.dropout(chunk_weights, p=self.dropout_p)
            
            chunk_output = torch.matmul(chunk_weights, v_chunk)
            output[:, :, i:end_i] = chunk_output
            
            if need_weights:
                all_weights.append(chunk_weights)
        
        # Combine attention weights for visualization/debugging
        combined_weights = None
        if need_weights:
            combined_weights = torch.zeros(B, H, T, T, device=q.device)
            chunk_idx = 0
            for i in range(0, T, self.chunk_size):
                end_i = min(i + self.chunk_size, T)
                if is_causal:
                    start_k = max(0, i - self.window_size)
                    end_k = end_i
                else:
                    center = (i + end_i) // 2
                    start_k = max(0, center - self.window_size // 2)
                    end_k = min(T, center + self.window_size // 2)
                    
                combined_weights[:, :, i:end_i, start_k:end_k] = all_weights[chunk_idx]
                chunk_idx += 1
                
        return output, combined_weights

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
        """Main sliding window attention method."""
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape
        
        # Project to Q, K, V
        q, k, v = self._project_qkv(query, key, value)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, H, T, D]
        
        # Handle layer state for incremental decoding
        if layer_state and not self.cross_attention:
            if "k" in layer_state:
                k = torch.cat([layer_state["k"], k], dim=2)
                v = torch.cat([layer_state["v"], v], dim=2)
            layer_state["k"], layer_state["v"] = k, v
        
        # Apply rotary embeddings if enabled
        if self.use_rotary:
            q, k = self.rotary_emb(q, k)
        
        # Use Flash Attention with sliding window if available
        if self.use_flash_sliding and self.backends.get("sdp", False) and hasattr(F, 'scaled_dot_product_attention'):
            try:
                # PyTorch 2.0+ scaled_dot_product_attention
                # Note: This doesn't have native sliding window support, so we still need to create masks
                window_mask = self._create_sliding_window_mask(T_q, q.device, is_causal)
                combined_mask = window_mask
                
                if attn_mask is not None:
                    if attn_mask.dim() == 2:
                        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                    combined_mask = combined_mask.unsqueeze(0).unsqueeze(0) | (attn_mask == 0)
                else:
                    combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)
                
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=~combined_mask if combined_mask is not None else None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,  # We handle causality in our mask
                )
                weights = None  # Flash attention doesn't return weights
                
            except Exception as e:
                warnings.warn(f"[MultiAttention] Flash sliding window failed: {e}. Using standard implementation.")
                out, weights = self._sliding_window_attention_efficient(
                    q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
                )
        else:
            # Standard sliding window implementation
            out, weights = self._sliding_window_attention_efficient(
                q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
            )
        
        # Reshape output
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.out_proj(self.dropout(out))
        
        return out, weights, layer_state

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
            if "k" in layer_state:
                k = torch.cat([layer_state["k"], k], dim=2)
                v = torch.cat([layer_state["v"], v], dim=2)
            layer_state["k"], layer_state["v"] = k, v

        if self.use_rotary:
            q, k = self.rotary_emb(q, k)

        if self.attention_type == "standard":
            out, weights = self._standard_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)
        else:
            out, weights = self._prob_sparse_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(self.dropout(out)), weights, layer_state

    def _standard_attention(self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights):
        B, H, T_q, T_k = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if is_causal and not self.cross_attention:
            mask = torch.tril(torch.ones(T_q, T_k, device=q.device)).bool()
            scores = scores.masked_fill(~mask, float("-inf"))
        scores = self._apply_masks(scores, attn_mask, key_padding_mask, B, T_q, T_k)
        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0:
            weights = F.dropout(weights, p=self.dropout_p)
        out = torch.matmul(weights, v)
        return out, weights if need_weights else None

    def _prob_sparse_attention(self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights):
            B, H, T_q, D = q.shape
            T_k = k.size(2)
            u = max(1, min(int(self.prob_sparse_factor * math.log(T_k)), T_q))
            sample_k = max(1, int(self.prob_sparse_factor * T_k))
            idx = torch.randperm(T_k, device=q.device)[:sample_k]
            k_sample = k[:, :, idx, :]
            scores = torch.matmul(q, k_sample.transpose(-2, -1)) * self.scale
            sparsity = scores.max(dim=-1)[0] - scores.mean(dim=-1)
            _, top_idx = torch.topk(sparsity, k=u, dim=-1)
            top_q = torch.gather(q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D))
            scores = torch.matmul(top_q, k.transpose(-2, -1)) * self.scale

            if is_causal and not self.cross_attention:
                q_pos = top_idx.unsqueeze(-1)
                k_pos = torch.arange(T_k, device=q.device).view(1, 1, 1, T_k)
                mask = q_pos >= k_pos
                scores = scores.masked_fill(~mask, float("-inf"))

            if attn_mask is not None:
                mask = torch.gather(attn_mask.unsqueeze(0).unsqueeze(0), 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, T_k))
                scores = scores.masked_fill(mask == 0, float("-inf"))
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, T_k), float("-inf"))

            scores -= scores.max(dim=-1, keepdim=True)[0]
            weights = F.softmax(scores, dim=-1)
            if self.dropout_p > 0:
                weights = F.dropout(weights, p=self.dropout_p)
            top_out = torch.matmul(weights, v)

            output = torch.zeros_like(q)
            output.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D), top_out)
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

    def reset_cache(self):
        for name in ["freq_attention", "dwt_attention"]:
            attn = getattr(self, name, None)
            if attn and hasattr(attn, "cache"):
                attn.cache.clear()

