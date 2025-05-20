import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class PositionalEncoding(nn.Module):
    """
    Enhanced positional encoding with support for:
    - Standard sinusoidal encoding
    - Relative position encoding
    - Extended context length
    - More stable initialization
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        scale: float = 1.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        self.d_model = d_model

        # Create more stable positional encoding with scaling factor
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # Register buffer (not a parameter but still part of the module)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + (self.pe[:, :seq_len] * self.scale)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_len=5000):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(
            d_model=hidden_size, max_len=max_len
        )

    def forward(self, x):
        # x: [B, T, input_size]
        x = self.input_projection(x)  # [B, T, hidden_size]
        x = self.positional_encoding(x)  # [B, T, hidden_size]
        return x


class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings for transformer models.
    This implementation uses complex numbers to represent the embeddings.
    """

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

    def _build_cache(self, seq_len, device):
        """Build a cache for fast rotary embedding computation."""
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        seq_idx = torch.arange(seq_len, device=device).float()

        # Shape: [seq_len, dim/2]
        freqs = torch.outer(seq_idx, theta)

        # Shape: [seq_len, dim/2, 2]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis

    def forward(self, q, k, q_pos=None, k_pos=None):
        """Apply rotary embeddings to query and key tensors."""
        batch, heads, q_len, head_dim = q.shape
        _, _, k_len, _ = k.shape

        if head_dim != self.dim:
            rotary_dim = min(self.dim, head_dim)
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        else:
            q_rot, k_rot = q, k
            q_pass, k_pass = None, None

        # Get or build position indices
        max_seq_len = max(q_len, k_len)
        freqs_cis = self._build_cache(max_seq_len, q.device)

        # Handle custom position indices
        if q_pos is not None:
            q_freqs = freqs_cis[q_pos]
            k_freqs = freqs_cis[k_pos if k_pos is not None else q_pos]
        else:
            q_freqs = freqs_cis[:q_len]
            k_freqs = freqs_cis[:k_len]

        # Reshape for broadcasting
        q_freqs = q_freqs.view(1, 1, q_len, -1, 2)
        k_freqs = k_freqs.view(1, 1, k_len, -1, 2)

        # Reshape input for rotation
        q_rot = q_rot.view(batch, heads, q_len, -1, 2)
        k_rot = k_rot.view(batch, heads, k_len, -1, 2)

        # Perform complex multiplication
        q_rot = torch.view_as_complex(q_rot)
        k_rot = torch.view_as_complex(k_rot)
        q_freqs = torch.view_as_complex(q_freqs)
        k_freqs = torch.view_as_complex(k_freqs)

        q_rot = q_rot * q_freqs
        k_rot = k_rot * k_freqs

        # Convert back to real and reshape
        q_rot = torch.view_as_real(q_rot).flatten(3)
        k_rot = torch.view_as_real(k_rot).flatten(3)

        # Concatenate passed-through parts if needed
        if q_pass is not None:
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        else:
            q, k = q_rot, k_rot

        return q, k


# Helper functions for feature detection
def _is_flash_attn_available():
    try:
        # First try the modern import
        from flash_attn import flash_attn_func

        return True
    except ImportError:
        try:
            # Fallback to older versions
            from flash_attn.flash_attn_interface import flash_attn_func

            return True
        except ImportError:
            return False


def _is_xformers_available():
    try:
        import xformers.ops as xops

        return True
    except ImportError:
        return False


def _is_torch_sdp_available():
    return hasattr(F, "scaled_dot_product_attention")


class XFormerAttention(nn.Module):
    """
    Enhanced multi-head attention with support for various optimized attention mechanisms:
    - FlashAttention
    - Memory-efficient xFormers attention
    - PyTorch's scaled_dot_product_attention
    - ProbSparse attention with improved sampling
    - Optional rotary position embeddings
    """

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        batch_first=True,
        cross_attention=False,
        attention_type="default",
        prob_sparse_factor=0.4,
        use_rotary=False,
        rotary_dim=None,
        flash_attention=True,
        use_flash_attn=None,  # For backward compatibility
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.cross_attention = cross_attention

        # Unified QKV projection for efficiency (can be split in forward)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Attention type configuration
        self.attention_type = attention_type
        self.prob_sparse_factor = prob_sparse_factor

        # FlashAttention and other optimizations
        # Handle backward compatibility with use_flash_attn parameter
        if use_flash_attn is not None:
            flash_attention = use_flash_attn

        self.flash_attention = flash_attention and _is_flash_attn_available()
        self.has_sdp = _is_torch_sdp_available()
        self.has_xformers = _is_xformers_available()

        # Optional rotary position embeddings
        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary_dim = rotary_dim or self.head_dim
            self.rotary_emb = RotaryEmbedding(self.rotary_dim) if use_rotary else None

        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        query,
        key=None,
        value=None,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
        need_weights=False,
        position_ids=None,
    ):
        """
        Unified forward pass supporting different attention mechanisms.
        For encoder self-attention: only provide query
        For decoder self-attention or cross-attention: provide query, key, value
        """
        # Handle self-attention case
        if key is None and value is None:
            key = value = query

        # Handle sequence ordering
        if not self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        # Get batch size and sequence lengths
        batch_size, tgt_len = query.size(0), query.size(1)
        src_len = key.size(1)

        # Efficient QKV projection
        if key is value and key is query:  # Self-attention optimization
            q, k, v = self.qkv_proj(query).chunk(3, dim=-1)
        else:  # Cross-attention
            q = self.qkv_proj(query)[:, :, : self.d_model]
            if key is value:  # Common in many architectures
                k, v = self.qkv_proj(key)[:, :, self.d_model :].chunk(2, dim=-1)
            else:
                k = self.qkv_proj(key)[:, :, self.d_model : 2 * self.d_model]
                v = self.qkv_proj(value)[:, :, 2 * self.d_model :]

        # Reshape to multi-head format
        q = q.view(batch_size, tgt_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.nhead, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings if enabled
        if self.use_rotary:
            if position_ids is None:
                q_pos = k_pos = torch.arange(max(tgt_len, src_len), device=query.device)
                q_pos = q_pos[:tgt_len]
                k_pos = k_pos[:src_len]
            else:
                q_pos, k_pos = position_ids

            q, k = self.rotary_emb(q, k, q_pos=q_pos, k_pos=k_pos)

        # Dispatch to appropriate attention implementation
        if self.attention_type == "prob_sparse":
            return self._prob_sparse_attention(
                q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
            )

        # Try optimized attention implementations first
        output = self._try_optimized_attention(
            q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
        )
        if output is not None:
            return output

        # Fallback to standard attention
        return self._standard_attention(
            q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
        )

    def _try_optimized_attention(
        self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
    ):
        """Try to use the most efficient attention implementation available."""
        # Skip optimizations if weights are needed or masks are present
        if need_weights or attn_mask is not None or key_padding_mask is not None:
            return None

        batch_size, _, tgt_len, _ = q.shape

        # FlashAttention - fastest when available
        if self.flash_attention and not need_weights:
            try:
                q, k, v = [x.contiguous() for x in (q, k, v)]

                # Handle different flash attention versions
                if hasattr(self, "_flash_attn_func_import"):
                    flash_func = self._flash_attn_func_import
                else:
                    try:
                        from flash_attn import flash_attn_func

                        self._flash_attn_func_import = flash_attn_func
                        flash_func = flash_attn_func
                    except ImportError:
                        try:
                            from flash_attn.flash_attn_interface import flash_attn_func

                            self._flash_attn_func_import = flash_attn_func
                            flash_func = flash_attn_func
                        except ImportError:
                            return None

                out = flash_func(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    causal=is_causal and not self.cross_attention,
                )
                out = out.transpose(1, 2).reshape(batch_size, tgt_len, self.d_model)
                return self.out_proj(out), None
            except Exception:
                # Fallback if flash attention fails for any reason
                pass

        # PyTorch's scaled_dot_product_attention
        if self.has_sdp and not need_weights:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal and not self.cross_attention,
            )
            out = out.transpose(1, 2).reshape(batch_size, tgt_len, self.d_model)
            return self.out_proj(out), None

        # xFormers memory-efficient attention
        if self.has_xformers and not need_weights:
            bias = (
                None
                if not is_causal or self.cross_attention
                else xops.LowerTriangularMask()
            )
            out = xops.memory_efficient_attention(
                q, k, v, attn_bias=bias, p=self.dropout_p if self.training else 0.0
            )
            out = out.transpose(1, 2).reshape(batch_size, tgt_len, self.d_model)
            return self.out_proj(out), None

        return None

    def _standard_attention(
        self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
    ):
        """Standard attention implementation with proper masking."""
        batch_size, _, tgt_len, _ = q.shape
        src_len = k.size(2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply causal mask if needed
        if is_causal and not self.cross_attention:
            causal_mask = torch.ones(tgt_len, src_len, device=q.device).tril()
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None, :, :]
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        # Handle numerical stability
        attn_scores = torch.clamp(attn_scores, min=-1e4, max=1e4)

        # Compute attention weights and apply dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        if self.training and self.dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, tgt_len, self.d_model)
        out = self.out_proj(out)

        return (out, attn_weights) if need_weights else (out, None)

    def _prob_sparse_attention(
        self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights
    ):
        """
        Improved ProbSparse attention with KNN optimization.
        """
        batch_size, num_heads, tgt_len, dim = q.shape
        src_len = k.size(2)

        # Calculate u: number of queries to retain based on sequence length
        u = max(
            int(min(self.prob_sparse_factor * math.log(src_len), tgt_len * 0.5)),
            int(tgt_len * 0.3),  # Minimum coverage of 30%
        )

        # Use approximate KL score for faster selection
        # Scale down for better numerical stability
        q_scaled = q * self.scaling
        attn_approx = torch.matmul(q_scaled, k.transpose(-2, -1))
        attn_approx_sm = F.softmax(attn_approx, dim=-1)

        # Calculate sampling criterion (KL-divergence from uniform)
        kl_score = torch.sum(
            attn_approx_sm * (torch.log(attn_approx_sm + 1e-10) + math.log(src_len)),
            dim=-1,
        )

        # Select top-u queries
        topk_indices = torch.topk(kl_score, u, dim=-1).indices

        # Gather selected queries
        q_reduced = torch.gather(
            q, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
        )

        # Try to use optimized attention for the reduced set if possible
        output = None
        if attn_mask is None and key_padding_mask is None and not need_weights:
            # Try Flash Attention or other optimized implementations
            if self.flash_attention:
                try:
                    q_reduced = q_reduced.contiguous()
                    k_cont = k.contiguous()
                    v_cont = v.contiguous()

                    # Handle different flash attention versions
                    if hasattr(self, "_flash_attn_func_import"):
                        flash_func = self._flash_attn_func_import
                    else:
                        try:
                            from flash_attn import flash_attn_func

                            self._flash_attn_func_import = flash_attn_func
                            flash_func = flash_attn_func
                        except ImportError:
                            try:
                                from flash_attn.flash_attn_interface import (
                                    flash_attn_func,
                                )

                                self._flash_attn_func_import = flash_attn_func
                                flash_func = flash_attn_func
                            except ImportError:
                                flash_func = None

                    if flash_func is not None:
                        out_reduced = flash_func(
                            q_reduced,
                            k_cont,
                            v_cont,
                            dropout_p=self.dropout_p if self.training else 0.0,
                            causal=is_causal and not self.cross_attention,
                        )
                        output = out_reduced
                except Exception:
                    # Fallback if flash attention fails for any reason
                    pass

            elif self.has_xformers:
                try:
                    import xformers.ops as xops

                    bias = (
                        None
                        if not is_causal or self.cross_attention
                        else xops.LowerTriangularMask()
                    )
                    out_reduced = xops.memory_efficient_attention(
                        q_reduced,
                        k,
                        v,
                        attn_bias=bias,
                        p=self.dropout_p if self.training else 0.0,
                    )
                    output = out_reduced
                except Exception:
                    # Fallback if xformers fails
                    pass

        # Fallback to standard attention for the reduced set
        if output is None:
            # Compute scores for reduced set
            attn_scores = torch.matmul(q_reduced, k.transpose(-2, -1)) * self.scaling

            # Apply masks if needed
            if is_causal and not self.cross_attention:
                causal_mask = torch.ones(u, src_len, device=q.device).tril()
                attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    key_padding_mask[:, None, None, :], float("-inf")
                )

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask[None, None, :, :]
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask[:, None, :, :]

                if attn_mask.size(2) == tgt_len:
                    # Gather relevant parts of the attention mask
                    attn_mask_reduced = torch.gather(
                        attn_mask,
                        2,
                        topk_indices.unsqueeze(-1).expand(-1, -1, -1, src_len),
                    )
                    attn_scores = attn_scores.masked_fill(
                        attn_mask_reduced == 0, float("-inf")
                    )
                else:
                    attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

            # Compute weights and apply attention
            attn_scores = torch.clamp(attn_scores, min=-1e4, max=1e4)
            attn_weights = F.softmax(attn_scores, dim=-1)

            if self.training and self.dropout_p > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout_p)

            output = torch.matmul(attn_weights, v)

        # Create full output by scattering back the reduced output
        out = torch.zeros(
            batch_size, num_heads, tgt_len, dim, device=q.device, dtype=q.dtype
        )
        out.scatter_(2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, dim), output)

        # Final projection
        out = out.transpose(1, 2).reshape(batch_size, tgt_len, self.d_model)
        out = self.out_proj(out)

        return out, None

from .blocks.fed import FrequencyAttention

class TransformerEncoderLayer(nn.Module):
    """
    Optimized Transformer Encoder Layer with modern architecture improvements:
    - Optional SwiGLU activations for better performance
    - Pre-norm architecture for better training stability
    - Optimized forward pass with fused operations
    - Adaptive layer normalization strategy
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        att_type="prob_sparse",
        use_swiglu=True,
        use_flash_attn=True,
        layer_norm_eps=1e-5,
        norm_strategy="pre_norm",  # 'pre_norm' or 'post_norm'
        freq_att=True, freq_modes=16, seq_len=None
    ):
        super().__init__()

        if freq_att:
            print("Using Frequency Attention -- FEDFormer style")
            self.self_attn = FrequencyAttention(
                d_model=d_model,
                n_heads=nhead,
                modes=freq_modes,
                seq_len_q=seq_len,
                seq_len_kv=seq_len,
            )
        else:
            self.self_attn = XFormerAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=True,
                use_flash_attn=use_flash_attn,
                attention_type=att_type,
            )

        # Choose whether to use pre-norm or post-norm
        self.norm_strategy = norm_strategy

        # SwiGLU FFN (if enabled) or standard MLP
        self.use_swiglu = use_swiglu
        if use_swiglu:
            # SwiGLU uses 2/3 * 3/2 = 1.0x param count but better performance
            self.w1 = nn.Linear(d_model, int(dim_feedforward * 4 / 3))
            self.w2 = nn.Linear(d_model, int(dim_feedforward * 4 / 3))
            self.w3 = nn.Linear(int(dim_feedforward * 4 / 3), d_model)
        else:
            # Standard MLP
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.activation = F.relu if activation == "relu" else F.gelu

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _swiglu_ffn(self, x):
        """SwiGLU feed-forward block (better than GELU)"""
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2  # SwiGLU activation
        return self.w3(hidden)

    def _standard_ffn(self, x):
        """Standard feed-forward block with GELU/ReLU"""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-norm architecture (more stable for training)
        if self.norm_strategy == "pre_norm":
            # Attention block
            src_norm = self.norm1(src)
            src2, _ = self.self_attn(
                src_norm,
                src_norm,
                src_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + self.dropout1(src2)

            # Feed-forward block
            src_norm = self.norm2(src)
            if self.use_swiglu:
                src2 = self._swiglu_ffn(src_norm)
            else:
                src2 = self._standard_ffn(src_norm)
            src = src + self.dropout2(src2)

        # Post-norm architecture (original transformer)
        else:
            # Attention block
            src2, _ = self.self_attn(
                src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
            )
            src = self.norm1(src + self.dropout1(src2))

            # Feed-forward block
            if self.use_swiglu:
                src2 = self._swiglu_ffn(src)
            else:
                src2 = self._standard_ffn(src)
            src = self.norm2(src + self.dropout2(src2))

        return src


class TransformerDecoderLayer(nn.Module):
    """
    Optimized Transformer Decoder Layer with modern architecture improvements:
    - Optional SwiGLU activations for better performance
    - Pre-norm architecture for better training stability
    - Support for Informer-style non-autoregressive decoding
    - Optimized attention patterns
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        informer_like=False,
        att_type="prob_sparse",
        use_swiglu=True,
        use_flash_attn=True,
        layer_norm_eps=1e-5,
        norm_strategy="pre_norm",  # 'pre_norm' or 'post_norm'
    ):
        super().__init__()

        # Attention blocks
        self.self_attn = XFormerAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            use_flash_attn=use_flash_attn,
            attention_type=att_type,
        )

        self.multihead_attn = XFormerAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            cross_attention=True,
            use_flash_attn=use_flash_attn,
            attention_type=att_type,
        )

        # Choose whether to use pre-norm or post-norm
        self.norm_strategy = norm_strategy

        # SwiGLU FFN (if enabled) or standard MLP
        self.use_swiglu = use_swiglu
        if use_swiglu:
            # SwiGLU uses 2/3 * 3/2 = 1.0x param count but better performance
            self.w1 = nn.Linear(d_model, int(dim_feedforward * 4 / 3))
            self.w2 = nn.Linear(d_model, int(dim_feedforward * 4 / 3))
            self.w3 = nn.Linear(int(dim_feedforward * 4 / 3), d_model)
        else:
            # Standard MLP
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.activation = F.relu if activation == "relu" else F.gelu

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Informer-style setting (non-autoregressive)
        self.informer_like = informer_like

    def _swiglu_ffn(self, x):
        """SwiGLU feed-forward block (better than GELU)"""
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2  # SwiGLU activation
        return self.w3(hidden)

    def _standard_ffn(self, x):
        """Standard feed-forward block with GELU/ReLU"""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Pre-norm architecture (more stable for training)
        if self.norm_strategy == "pre_norm":
            # Self-attention block
            tgt_norm = self.norm1(tgt)

            # Decide whether to use causal masking
            is_causal = not self.informer_like

            tgt2, _ = self.self_attn(
                tgt_norm,
                tgt_norm,
                tgt_norm,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=is_causal,
            )
            tgt = tgt + self.dropout1(tgt2)

            # Cross-attention block
            tgt_norm = self.norm2(tgt)
            tgt2, _ = self.multihead_attn(
                tgt_norm,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
            tgt = tgt + self.dropout2(tgt2)

            # Feed-forward block
            tgt_norm = self.norm3(tgt)
            if self.use_swiglu:
                tgt2 = self._swiglu_ffn(tgt_norm)
            else:
                tgt2 = self._standard_ffn(tgt_norm)
            tgt = tgt + self.dropout3(tgt2)

        # Post-norm architecture (original transformer)
        else:
            # Self-attention block
            is_causal = not self.informer_like
            tgt2, _ = self.self_attn(
                tgt,
                tgt,
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=is_causal,
            )
            tgt = self.norm1(tgt + self.dropout1(tgt2))

            # Cross-attention block
            tgt2, _ = self.multihead_attn(
                tgt,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
            tgt = self.norm2(tgt + self.dropout2(tgt2))

            # Feed-forward block
            if self.use_swiglu:
                tgt2 = self._swiglu_ffn(tgt)
            else:
                tgt2 = self._standard_ffn(tgt)
            tgt = self.norm3(tgt + self.dropout3(tgt2))

        return tgt


class TransformerEncoder(nn.Module):
    """
    Enhanced Transformer Encoder with modern improvements:
    - Optimized layer stacking
    - Optional embedding sharing
    - Configurable normalization strategy
    - Memory-efficient attention mechanisms
    - Gradient checkpointing support for training larger models
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        hidden_size: Optional[int] = None,
        att_type: str = "prob_sparse",
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        use_gradient_checkpointing: bool = False,
        pos_encoding_scale: float = 1.0,
        max_seq_len: int = 5000,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        # Set up model dimensions
        self.hidden_size = hidden_size if hidden_size is not None else d_model
        self.d_model = d_model
        self.input_size = input_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.is_transformer = True

        # Input projection and embedding
        self.input_projection = nn.Linear(self.hidden_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model, dropout=dropout, max_len=max_seq_len, scale=pos_encoding_scale
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    att_type=att_type,
                    use_swiglu=use_swiglu,
                    use_flash_attn=use_flash_attn,
                    layer_norm_eps=layer_norm_eps,
                    norm_strategy=norm_strategy,
                )
                for _ in range(num_layers)
            ]
        )

        # Output normalization (for pre-norm architecture)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer encoder.

        Args:
            src: Source sequence [batch_size, seq_len, hidden_size]
            src_mask: Optional attention mask [seq_len, seq_len]
            src_key_padding_mask: Optional padding mask [batch_size, seq_len]

        Returns:
            Encoded output [batch_size, seq_len, d_model]
        """
        # Project input to model dimension
        src = self.input_projection(src)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Apply dropout
        src = self.dropout(src)

        # Apply transformer layers with optional gradient checkpointing
        for layer in self.layers:
            if self.training and self.use_gradient_checkpointing:
                src = torch.utils.checkpoint.checkpoint(
                    layer, src, src_mask, src_key_padding_mask
                )
            else:
                src = layer(
                    src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
                )

        # Apply final layer normalization
        return self.norm(src)


class TransformerDecoder(nn.Module):
    """
    Enhanced Transformer Decoder with modern improvements:
    - Optimized layer stacking
    - Optional embedding sharing
    - Support for incremental/autoregressive decoding
    - Configurable normalization strategy
    - Memory-efficient attention mechanisms
    - Gradient checkpointing support for training larger models
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        hidden_size: Optional[int] = None,
        informer_like: bool = False,
        att_type: str = "prob_sparse",
        layer_norm_eps: float = 1e-5,
        norm_strategy: str = "pre_norm",
        use_gradient_checkpointing: bool = False,
        pos_encoding_scale: float = 1.0,
        max_seq_len: int = 5000,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        # Set up model dimensions
        self.hidden_size = hidden_size if hidden_size is not None else d_model
        self.d_model = d_model
        self.output_size = output_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.is_transformer = True

        # Input projection
        self.input_projection = nn.Linear(self.hidden_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model, dropout=dropout, max_len=max_seq_len, scale=pos_encoding_scale
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Transformer decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    informer_like=informer_like,
                    att_type=att_type,
                    use_swiglu=use_swiglu,
                    use_flash_attn=use_flash_attn,
                    layer_norm_eps=layer_norm_eps,
                    norm_strategy=norm_strategy,
                )
                for _ in range(num_layers)
            ]
        )

        # Output normalization and projection
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Improved output projection
        self.output_projection = (
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, output_size),
            )
            if output_size > 1
            else nn.Linear(d_model, output_size)
        )

        # Cache for incremental decoding
        self.incremental_state = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer decoder.

        Args:
            tgt: Target sequence [batch_size, tgt_len, hidden_size]
            memory: Memory from encoder [batch_size, src_len, d_model]
            tgt_mask: Optional attention mask [tgt_len, tgt_len]
            memory_mask: Optional cross-attention mask [tgt_len, src_len]
            tgt_key_padding_mask: Optional padding mask [batch_size, tgt_len]
            memory_key_padding_mask: Optional padding mask [batch_size, src_len]
            incremental_state: Optional state for incremental decoding

        Returns:
            Decoded output [batch_size, tgt_len, output_size]
        """
        # Project input to model dimension
        tgt = self.input_projection(tgt)

        # Add positional encoding
        tgt = self.pos_encoder(tgt)

        # Apply dropout
        tgt = self.dropout(tgt)

        # Initialize incremental state if not provided
        if incremental_state is None:
            incremental_state = [None] * len(self.layers)

        # Apply transformer layers with optional gradient checkpointing
        for idx, layer in enumerate(self.layers):
            if self.training and self.use_gradient_checkpointing:
                tgt = torch.utils.checkpoint.checkpoint(
                    layer,
                    tgt,
                    memory,
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                )
            else:
                tgt = layer(
                    tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

        # Apply final layer normalization
        tgt = self.norm(tgt)

        # Project to output size
        return self.output_projection(tgt)

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: Optional[List[dict]] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        Optimized single-step forward pass for autoregressive generation.

        Args:
            tgt: The current input token(s) [batch_size, 1, hidden_size]
            memory: Encoder memory [batch_size, src_len, d_model]
            incremental_state: Cached state for faster decoding
            memory_key_padding_mask: Optional padding mask [batch_size, src_len]

        Returns:
            Tuple of (output, updated_state)
        """
        # Initialize incremental state if None
        if incremental_state is None:
            incremental_state = [None] * len(self.layers)

        # Project input
        tgt = self.input_projection(tgt)

        # For the last position only
        if tgt.size(1) == 1:
            pos = incremental_state[0]["position"] if incremental_state[0] else 0
            pos_enc = self.pos_encoder.pe[:, pos : pos + 1]
            tgt = tgt + pos_enc

            # Update position
            for i in range(len(self.layers)):
                if incremental_state[i] is None:
                    incremental_state[i] = {"position": pos + 1}
                else:
                    incremental_state[i]["position"] = pos + 1
        else:
            # Standard positional encoding
            tgt = self.pos_encoder(tgt)

        # Apply dropout
        tgt = self.dropout(tgt)

        # Apply layers with incremental state
        for i, layer in enumerate(self.layers):
            tgt = layer(
                tgt,
                memory,
                tgt_mask=None,  # Not needed for single step
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # Apply final normalization and projection
        tgt = self.norm(tgt)
        output = self.output_projection(tgt)

        return output, incremental_state

