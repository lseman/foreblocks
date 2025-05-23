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

from .embeddings import PositionalEncoding, RotaryEmbedding
from .blocks.fed import FrequencyAttention


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
    Optimized multi-head attention with support for:
    - FlashAttention
    - PyTorch's SDP
    - xFormers memory-efficient attention
    - ProbSparse attention
    - Rotary embeddings
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
        use_rotary=True,
        rotary_dim=None,
        flash_attention=True,
        use_flash_attn=None,
        freq_modes=16,
        debug=False,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.cross_attention = cross_attention
        self.scaling = self.head_dim**-0.5
        self.debug = debug

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if use_flash_attn is not None:
            flash_attention = use_flash_attn

        self.flash_attention = flash_attention and _is_flash_attn_available()
        self.has_sdp = _is_torch_sdp_available()
        self.has_xformers = _is_xformers_available()

        self.use_rotary = use_rotary
        self.rotary_dim = rotary_dim or self.head_dim
        self.rotary_emb = RotaryEmbedding(self.rotary_dim) if use_rotary else None

        self.attention_type = attention_type
        self.attention_strategy_factory = {
            "default": lambda: StandardAttentionStrategy(),
            "prob_sparse": lambda: ProbSparseAttentionStrategy(prob_sparse_factor),
            "frequency": lambda: FrequencyAttention(
                d_model=d_model,
                n_heads=nhead,
                dropout=dropout,
                modes=freq_modes,
                seq_len_q=None,
                seq_len_kv=None,
            ),
        }

        self._attention_strategy = self.attention_strategy_factory.get(
            self.attention_type, self.attention_strategy_factory["default"]
        )()
        print("Using attention strategy:", self.attention_type)

        self.optimized_strategy = OptimizedAttentionStrategy()

    def _prepare_qkv(self, query, key=None, value=None):
        if key is None and value is None:
            key = value = query

        if not self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        bsz, tgt_len = query.size(0), query.size(1)
        src_len = key.size(1)

        # Check for NaN in inputs
        if (
            torch.isnan(query).any()
            or torch.isnan(key).any()
            or torch.isnan(value).any()
        ):
            raise RuntimeError("NaN detected in query/key/value inputs")

        if key is value and key is query:
            q, k, v = self.qkv_proj(query).chunk(3, dim=-1)
        else:
            q = self.qkv_proj(query)[..., : self.d_model]
            k = self.qkv_proj(key)[..., self.d_model : 2 * self.d_model]
            v = self.qkv_proj(value)[..., 2 * self.d_model :]

        q = q.view(bsz, tgt_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.nhead, self.head_dim).transpose(1, 2)

        if self.debug:
            print(
                f"[DEBUG] Q stats: mean={q.mean().item():.5f}, std={q.std().item():.5f}"
            )
            print(
                f"[DEBUG] K stats: mean={k.mean().item():.5f}, std={k.std().item():.5f}"
            )
            print(
                f"[DEBUG] V stats: mean={v.mean().item():.5f}, std={v.std().item():.5f}"
            )

        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            raise RuntimeError("NaN detected in Q/K/V after projection")

        return q, k, v, bsz, tgt_len, src_len

    def _apply_rotary(self, q, k, position_ids, query):
        if not self.use_rotary or self.rotary_emb is None:
            return q, k

        bsz, tgt_len = query.size(0), query.size(1)
        src_len = k.size(2)

        if position_ids is None:
            q_pos = torch.arange(tgt_len, device=query.device)
            k_pos = torch.arange(src_len, device=query.device)
        else:
            q_pos, k_pos = position_ids

        return self.rotary_emb(q, k, q_pos=q_pos, k_pos=k_pos)

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
        q, k, v, bsz, tgt_len, src_len = self._prepare_qkv(query, key, value)

        if self.use_rotary:
            q, k = self._apply_rotary(q, k, position_ids, query)

        if attn_mask is None and key_padding_mask is None and not need_weights:
            fast_out = self.optimized_strategy.compute_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p,
                training=self.training,
                cross_attention=self.cross_attention,
                d_model=self.d_model,
                scaling=self.scaling,
                is_causal=is_causal,
                flash_attention=self.flash_attention,
                has_sdp=self.has_sdp,
                has_xformers=self.has_xformers,
            )
            if fast_out is not None:
                out, weights = fast_out
                if torch.isnan(out).any():
                    print("[Warning] NaN in optimized attention output — fallback.")
                else:
                    out = out.transpose(1, 2).reshape(bsz, tgt_len, self.d_model)
                    return self.out_proj(out), weights

        strategy = self._attention_strategy
        if isinstance(strategy, FrequencyAttention):
            return strategy(
                query, key, value, attn_mask, key_padding_mask, is_causal, need_weights
            )

        out, weights = strategy.compute_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights,
            dropout_p=self.dropout_p,
            training=self.training,
            cross_attention=self.cross_attention,
            d_model=self.d_model,
            scaling=self.scaling,
        )

        if torch.isnan(out).any():
            raise RuntimeError("NaN detected in fallback attention output")

        out = out.transpose(1, 2).reshape(bsz, tgt_len, self.d_model)
        return self.out_proj(out), weights


class AttentionStrategy:
    """Base class for all attention strategies"""

    def compute_attention(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
        need_weights=False,
        dropout_p=0.0,
        training=False,
        cross_attention=False,
        d_model=None,
        scaling=None,
    ):
        raise NotImplementedError("Subclasses must implement compute_attention")


class StandardAttentionStrategy(AttentionStrategy):
    """Optimized standard scaled dot-product attention implementation."""

    def __init__(self):
        print("[Attention] StandardAttentionStrategy")

    def compute_attention(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
        need_weights=False,
        dropout_p=0.0,
        training=False,
        cross_attention=False,
        d_model=None,
        scaling=None,
    ):
        batch_size, _, tgt_len, _ = q.shape
        src_len = k.size(2)

        # === Fast path with PyTorch fused attention ===
        if (
            not need_weights
            and attn_mask is None
            and key_padding_mask is None
            and hasattr(F, "scaled_dot_product_attention")
        ):
            print("[Attention] Using PyTorch fused attention")
            return (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p if training else 0.0,
                    is_causal=is_causal and not cross_attention,
                ),
                None,
            )

        print("[Attention] Using manual attention computation")

        # === Manual attention computation ===
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        if scaling is not None:
            attn_scores *= scaling

        # Causal mask (only for self-attention)
        if is_causal and not cross_attention:
            causal_mask = torch.ones(
                tgt_len, src_len, device=q.device, dtype=torch.bool
            ).tril()
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        # Attention mask
        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.dim() == 3
                else attn_mask[None, None]
            )
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        # Key padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool), float("-inf")
            )

        # [NEW] Clamp for sanity (optional, protects against inf)
        attn_scores = attn_scores.clamp(min=-50, max=50)

        # [NEW] Max-subtraction for numerical stability
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values

        attn_weights = F.softmax(attn_scores, dim=-1)

        if torch.isnan(attn_scores).any():
            raise RuntimeError("NaN in attention scores before softmax")

        if torch.isnan(attn_weights).any():
            raise RuntimeError("NaN in attention weights after softmax")

        if dropout_p > 0.0 and training:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        attn_weights = attn_weights.clamp(min=1e-9, max=1.0)

        # Weighted sum
        out = torch.matmul(attn_weights, v)

        return (out, attn_weights if need_weights else None)


class ProbSparseAttentionStrategy(AttentionStrategy):
    """ProbSparse attention with KLD-based top-u selection for sparse querying."""

    def __init__(self, prob_sparse_factor=0.4, debug=False):
        print("[Attention] ProbSparseAttentionStrategy")
        self.prob_sparse_factor = prob_sparse_factor
        self.debug = debug

    def _safe_check(self, tensor, name="tensor"):
        if torch.isnan(tensor).any():
            print(f"[NaN] Detected in {name}")
            return False
        if torch.isinf(tensor).any():
            print(f"[Inf] Detected in {name}")
            return False
        if tensor.abs().max() > 1e5:
            print(
                f"[Exploding] Detected in {name}, max={tensor.abs().max().item():.2f}"
            )
            return False
        return True

    def compute_attention(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
        need_weights=False,
        dropout_p=0.0,
        training=False,
        cross_attention=False,
        d_model=None,
        scaling=None,
    ):
        batch_size, num_heads, tgt_len, dim = q.shape
        src_len = k.size(2)

        # === Compute u ===
        try:
            u = max(
                int(min(self.prob_sparse_factor * math.log(src_len), tgt_len * 0.5)),
                int(tgt_len * 0.3),
            )
        except Exception as e:
            raise RuntimeError(f"Invalid src_len or tgt_len: {e}")

        if u <= 0:
            raise ValueError(
                f"[ProbSparse] Invalid number of top queries selected: u={u}"
            )

        # === Compute KL divergence ===
        q_scaled = q * scaling
        attn_approx = torch.matmul(q_scaled, k.transpose(-2, -1))  # [B, H, T_q, T_k]

        if not self._safe_check(attn_approx, "KL attention approximation"):
            raise RuntimeError("NaN in approximate attention scores")

        attn_approx = attn_approx.clamp(min=-50, max=50)
        attn_approx = attn_approx - attn_approx.max(dim=-1, keepdim=True).values
        attn_approx_sm = F.softmax(attn_approx, dim=-1)

        log_uniform = math.log(src_len)
        kl_score = torch.sum(
            attn_approx_sm * (torch.log(attn_approx_sm + 1e-10) + log_uniform),
            dim=-1,  # [B, H, T_q]
        )

        if torch.isnan(kl_score).any():
            raise RuntimeError("NaN in KL divergence scores")

        # === Top-u query selection ===
        topk_indices = torch.topk(kl_score, u, dim=-1).indices  # [B, H, u]
        if topk_indices.max() >= tgt_len:
            raise IndexError("Top-k indices out of range")

        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, dim)

        # === Reduce queries ===
        q_reduced = torch.gather(q, 2, topk_indices_expanded)  # [B, H, u, D]

        # === Compute attention scores ===
        attn_scores = (
            torch.matmul(q_reduced, k.transpose(-2, -1)) * scaling
        )  # [B, H, u, T_k]
        attn_scores = attn_scores.clamp(min=-50, max=50)
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values

        # === Causal mask ===
        if is_causal and not cross_attention:
            causal_mask = torch.tril(
                torch.ones(u, src_len, device=q.device, dtype=torch.bool)
            )
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        # === Key padding mask ===
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool), float("-inf")
            )

        # === Attention mask ===
        if attn_mask is not None:
            attn_mask = (
                attn_mask[None, None]
                if attn_mask.dim() == 2
                else attn_mask[:, None] if attn_mask.dim() == 3 else attn_mask
            )

            if attn_mask.size(2) == tgt_len:
                attn_mask_reduced = torch.gather(
                    attn_mask, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, src_len)
                )
                attn_scores = attn_scores.masked_fill(
                    attn_mask_reduced == 0, float("-inf")
                )
            else:
                attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        if torch.isnan(attn_weights).any():
            raise RuntimeError("NaN in attention weights (ProbSparse)")

        if dropout_p > 0.0 and training:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

        attn_weights = attn_weights.clamp(min=1e-9, max=1.0)

        # === Compute output ===
        output_reduced = torch.matmul(attn_weights, v)  # [B, H, u, D]

        if not self._safe_check(output_reduced, "output_reduced"):
            raise RuntimeError("NaN in output_reduced")

        # === Scatter back into full Q-sized tensor ===
        output_full = torch.zeros_like(q)  # [B, H, T_q, D]
        output_full.scatter_(2, topk_indices_expanded, output_reduced)

        return output_full, attn_weights if need_weights else None


class OptimizedAttentionStrategy(AttentionStrategy):
    """Optimized attention strategy using FlashAttention, PyTorch SDP, or xFormers."""

    def __init__(self, debug=False):
        self.flash_attn_func_import = None
        self.xformers_ops = None
        self.debug = debug
        print("[Attention] OptimizedAttentionStrategy")

    def _safe_check(self, tensor, name="tensor"):
        if torch.isnan(tensor).any():
            print(f"[NaN] Found in {name}")
            return False
        if torch.isinf(tensor).any():
            print(f"[Inf] Found in {name}")
            return False
        if tensor.abs().max() > 1e5:
            print(f"[Exploding] Values in {name}: max={tensor.abs().max().item():.2f}")
            return False
        return True

    def _try_flash_attn(self, q, k, v, dropout_p, is_causal, cross_attention, training):
        try:
            if self.flash_attn_func_import is None:
                try:
                    from flash_attn import flash_attn_func
                except ImportError:
                    from flash_attn.flash_attn_interface import flash_attn_func
                self.flash_attn_func_import = flash_attn_func

            q, k, v = [x.contiguous() for x in (q, k, v)]
            out = self.flash_attn_func_import(
                q,
                k,
                v,
                dropout_p=dropout_p if training else 0.0,
                causal=is_causal and not cross_attention,
            )
            if not self._safe_check(out, "FlashAttention output"):
                return None
            return out
        except Exception as e:
            if self.debug:
                print(f"[FlashAttention] Error: {type(e).__name__}: {e}")
            return None

    def _try_torch_sdp(self, q, k, v, dropout_p, is_causal, cross_attention, training):
        try:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p if training else 0.0,
                is_causal=is_causal and not cross_attention,
            )
            if not self._safe_check(out, "SDP output"):
                return None
            return out
        except Exception as e:
            if self.debug:
                print(f"[SDP] Error: {type(e).__name__}: {e}")
            return None

    def _try_xformers(self, q, k, v, dropout_p, is_causal, cross_attention, training):
        try:
            if self.xformers_ops is None:
                import xformers.ops as xops

                self.xformers_ops = xops

            bias = (
                None
                if not is_causal or cross_attention
                else self.xformers_ops.LowerTriangularMask()
            )
            out = self.xformers_ops.memory_efficient_attention(
                q, k, v, attn_bias=bias, p=dropout_p if training else 0.0
            )
            if not self._safe_check(out, "xFormers output"):
                return None
            return out
        except Exception as e:
            if self.debug:
                print(f"[xFormers] Error: {type(e).__name__}: {e}")
            return None

    def compute_attention(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
        need_weights=False,
        dropout_p=0.0,
        training=False,
        cross_attention=False,
        d_model=None,
        scaling=None,
        flash_attention=False,
        has_sdp=False,
        has_xformers=False,
    ):
        # Guard: fallback if masks or weights are needed
        if need_weights or attn_mask is not None or key_padding_mask is not None:
            if self.debug:
                print(
                    "[OptimizedAttention] Fallback triggered due to masks or weight request."
                )
            return None

        # Validate inputs
        if not all(
            [
                self._safe_check(q, "q"),
                self._safe_check(k, "k"),
                self._safe_check(v, "v"),
            ]
        ):
            print("[OptimizedAttention] Skipping optimized due to invalid input.")
            return None

        # Try backends in order of preference
        if flash_attention:
            out = self._try_flash_attn(
                q, k, v, dropout_p, is_causal, cross_attention, training
            )
            if out is not None:
                return out, None

        if has_sdp:
            out = self._try_torch_sdp(
                q, k, v, dropout_p, is_causal, cross_attention, training
            )
            if out is not None:
                return out, None

        if has_xformers:
            out = self._try_xformers(
                q, k, v, dropout_p, is_causal, cross_attention, training
            )
            if out is not None:
                return out, None

        # Final fallback
        if self.debug:
            print("[OptimizedAttention] No optimized backend succeeded. Falling back.")
        return None
