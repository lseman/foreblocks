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
from .fed import FrequencyAttention, DWTAttention


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
        use_flash_attn=None,
        freq_modes=16,
        debug=False,
    ):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.batch_first = batch_first
        self.cross_attention = cross_attention
        self.dropout_p = dropout
        self.debug = debug

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.flash_available = _is_flash_attn_available()
        self.sdp_available = _is_torch_sdp_available()
        self.xformers_available = _is_xformers_available()
        self.use_flash_attn = (
            use_flash_attn if use_flash_attn is not None else self.flash_available
        )

        self.use_rotary = use_rotary
        self.rotary_dim = rotary_dim or self.head_dim
        self.rotary_emb = RotaryEmbedding(self.rotary_dim) if use_rotary else None

        # Strategy selector
        if attention_type == "default":
            self.attention_strategy = StandardAttentionStrategy()
        elif attention_type == "prob_sparse":
            self.attention_strategy = ProbSparseAttentionStrategy(prob_sparse_factor)
        elif attention_type == "frequency":
            self.attention_strategy = FrequencyAttention(
                d_model=d_model,
                n_heads=nhead,
                dropout=dropout,
                modes=freq_modes,
                seq_len_q=None,
                seq_len_kv=None,
            )
        elif attention_type == "dwt":
            self.attention_strategy = DWTAttention(
                d_model=d_model,
                n_heads=nhead,
                dropout=dropout,
                modes=freq_modes,
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.optimized_strategy = OptimizedAttentionStrategy(debug=debug)

    def _prepare_qkv(self, query, key=None, value=None):
        key = query if key is None else key
        value = query if value is None else value

        if not self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        B, T_q = query.shape[:2]
        T_kv = key.shape[1]

        if key is value and key is query:
            q, k, v = self.qkv_proj(query).chunk(3, dim=-1)
        else:
            qkv = self.qkv_proj(query)
            q = qkv[..., : self.d_model]
            kv = self.qkv_proj(key)
            k = kv[..., self.d_model : 2 * self.d_model]
            v = kv[..., 2 * self.d_model :]

        q = q.view(B, T_q, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.nhead, self.head_dim).transpose(1, 2)

        return q, k, v, B, T_q

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
        layer_state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:

        q, k, v, B, T_q = self._prepare_qkv(query, key, value)

        # Handle key/value caching for incremental decoding
        if layer_state is not None:
            # Use cached K/V if available
            past_k = layer_state.get("k")  # [B, nhead, T_prev, head_dim]
            past_v = layer_state.get("v")

            if past_k is not None and past_v is not None:
                # Append along time dimension
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            # Update cache
            layer_state["k"] = k
            layer_state["v"] = v

        if self.use_rotary:
            q, k = self.rotary_emb(q, k)

        # Fast path: optimized attention (Flash/xFormers/SDP)
        if not any([attn_mask, key_padding_mask, need_weights]):
            out = self.optimized_strategy.compute_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p,
                training=self.training,
                is_causal=is_causal,
                cross_attention=self.cross_attention,
                flash_attention=self.use_flash_attn,
                has_sdp=self.sdp_available,
                has_xformers=self.xformers_available,
            )
            if out is not None:
                out, weights = out
                return (
                    self.out_proj(out.transpose(1, 2).reshape(B, T_q, self.d_model)),
                    weights,
                    layer_state,
                )

        # Fallback to manual strategy (Standard/ProbSparse/Frequency)
        out, weights = self.attention_strategy.compute_attention(
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
            scaling=self.head_dim**-0.5,
            layer_state=layer_state,
        )
        out = out.transpose(1, 2).reshape(B, T_q, self.d_model)
        return self.out_proj(out), weights, layer_state


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
        layer_state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        raise NotImplementedError("Subclasses must implement compute_attention")


class StandardAttentionStrategy(AttentionStrategy):
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

        if self._can_use_sdp(attn_mask, key_padding_mask, need_weights):
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

        scores = torch.matmul(q, k.transpose(-2, -1))
        if scaling:
            scores *= scaling

        scores = self._apply_masks(
            scores, attn_mask, key_padding_mask, is_causal, cross_attention
        )

        scores = scores.clamp(min=-50, max=50)
        scores = scores - scores.max(dim=-1, keepdim=True).values  # Stabilize

        weights = F.softmax(scores, dim=-1)
        if dropout_p > 0 and training:
            weights = F.dropout(weights, p=dropout_p)
        weights = weights.clamp(min=1e-9, max=1.0)

        out = torch.matmul(weights, v)
        return (out, weights if need_weights else None)

    def _can_use_sdp(self, attn_mask, key_padding_mask, need_weights):
        return (
            not need_weights
            and attn_mask is None
            and key_padding_mask is None
            and hasattr(F, "scaled_dot_product_attention")
        )

    def _apply_masks(
        self, scores, attn_mask, key_padding_mask, is_causal, cross_attention
    ):
        B, H, T_q, T_k = scores.shape

        if is_causal and not cross_attention:
            causal_mask = torch.ones(
                T_q, T_k, device=scores.device, dtype=torch.bool
            ).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None]
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool), float("-inf")
            )

        return scores


class ProbSparseAttentionStrategy(AttentionStrategy):
    def __init__(self, prob_sparse_factor=0.4, debug=False):
        self.prob_sparse_factor = prob_sparse_factor
        self.debug = debug

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

        B, H, T_q, D = q.shape
        T_k = k.size(2)

        u = self._compute_top_u(T_q, T_k)
        kl_score = self._compute_kl_score(q, k, scaling, T_k)
        topk_indices = torch.topk(kl_score, u, dim=-1).indices  # [B, H, u]

        q_reduced = self._reduce_queries(q, topk_indices, D)  # [B, H, u, D]
        scores = torch.matmul(q_reduced, k.transpose(-2, -1)) * scaling
        scores = scores.clamp(min=-50, max=50)
        scores -= scores.max(dim=-1, keepdim=True).values

        scores = self._apply_masks(
            scores,
            topk_indices,
            attn_mask,
            key_padding_mask,
            is_causal,
            cross_attention,
            T_k,
        )

        weights = F.softmax(scores, dim=-1)
        if dropout_p > 0 and training:
            weights = F.dropout(weights, p=dropout_p)
        weights = weights.clamp(min=1e-9, max=1.0)

        output_reduced = torch.matmul(weights, v)
        output_full = self._scatter_back(
            output_reduced, topk_indices, full_shape=q.shape
        )

        return output_full, weights if need_weights else None

    def _compute_top_u(self, T_q, T_k):
        try:
            u = max(
                int(min(self.prob_sparse_factor * math.log(T_k), T_q * 0.5)),
                int(T_q * 0.3),
            )
            if u <= 0:
                raise ValueError
            return u
        except:
            raise RuntimeError(
                f"[ProbSparse] Invalid top-u computation (T_q={T_q}, T_k={T_k})"
            )

    def _compute_kl_score(self, q, k, scaling, T_k):
        q_scaled = q * scaling
        approx = torch.matmul(q_scaled, k.transpose(-2, -1))
        approx = approx.clamp(min=-50, max=50)
        approx -= approx.max(dim=-1, keepdim=True).values
        sm = F.softmax(approx, dim=-1)

        log_uniform = math.log(T_k)
        return torch.sum(sm * (torch.log(sm + 1e-10) + log_uniform), dim=-1)

    def _reduce_queries(self, q, topk_indices, D):
        return torch.gather(q, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, D))

    def _scatter_back(self, output_reduced, topk_indices, full_shape):
        output_full = torch.zeros(full_shape, device=output_reduced.device)
        output_full.scatter_(
            2,
            topk_indices.unsqueeze(-1).expand(-1, -1, -1, full_shape[-1]),
            output_reduced,
        )

        return output_full

    def _apply_masks(
        self,
        scores,
        topk_indices,
        attn_mask,
        key_padding_mask,
        is_causal,
        cross_attention,
        T_k,
    ):
        B, H, u, _ = scores.shape

        if is_causal and not cross_attention:
            causal_mask = torch.tril(
                torch.ones(u, T_k, device=scores.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool), float("-inf")
            )

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

            if attn_mask.size(2) == scores.size(2):
                attn_mask_reduced = torch.gather(
                    attn_mask, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, T_k)
                )
                scores = scores.masked_fill(attn_mask_reduced == 0, float("-inf"))
            else:
                scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        return scores


class OptimizedAttentionStrategy(AttentionStrategy):
    """
    Optimized attention strategy using FlashAttention, PyTorch SDP, or xFormers.
    Automatically detects backend availability and imports only once.
    """

    def __init__(self, debug=False):
        self.debug = debug

        # FlashAttention
        try:
            from flash_attn import flash_attn_func

            self.flash_attn_func_import = flash_attn_func
            self.flash_available = True
        except ImportError:
            try:
                from flash_attn.flash_attn_interface import flash_attn_func

                self.flash_attn_func_import = flash_attn_func
                self.flash_available = True
            except ImportError:
                self.flash_attn_func_import = None
                self.flash_available = False

        # xFormers
        try:
            import xformers.ops as xops

            self.xformers_ops = xops
            self.xformers_available = True
        except ImportError:
            self.xformers_ops = None
            self.xformers_available = False

        # PyTorch Scaled Dot Product Attention
        self.sdp_available = hasattr(F, "scaled_dot_product_attention")

        if self.debug:
            print(
                f"[Attention Init] flash-attn: {self.flash_available}, "
                f"xformers: {self.xformers_available}, SDP: {self.sdp_available}"
            )

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
                return None

            # Ensure dtype is fp16 or bf16
            if q.dtype not in (torch.float16, torch.bfloat16):
                if self.debug:
                    print(
                        f"[FlashAttention] Converting dtype from {q.dtype} to float16"
                    )
                q = q.to(dtype=torch.float16)
                k = k.to(dtype=torch.float16)
                v = v.to(dtype=torch.float16)

            q, k, v = [x.contiguous() for x in (q, k, v)]
            out = self.flash_attn_func_import(
                q,
                k,
                v,
                dropout_p=dropout_p if training else 0.0,
                causal=is_causal and not cross_attention,
            )
            return out if self._safe_check(out, "FlashAttention output") else None
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
            return out if self._safe_check(out, "SDP output") else None
        except Exception as e:
            if self.debug:
                print(f"[SDP] Error: {type(e).__name__}: {e}")
            return None

    def _try_xformers(self, q, k, v, dropout_p, is_causal, cross_attention, training):
        try:
            if self.xformers_ops is None:
                return None
            bias = (
                None
                if not is_causal or cross_attention
                else self.xformers_ops.LowerTriangularMask()
            )
            out = self.xformers_ops.memory_efficient_attention(
                q, k, v, attn_bias=bias, p=dropout_p if training else 0.0
            )
            return out if self._safe_check(out, "xFormers output") else None
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
        flash_attention=None,
        has_sdp=None,
        has_xformers=None,
    ):
        if need_weights or attn_mask is not None or key_padding_mask is not None:
            if self.debug:
                print(
                    "[OptimizedAttention] Fallback triggered due to masks or weight request."
                )
            return None

        if not all(
            [
                self._safe_check(q, "q"),
                self._safe_check(k, "k"),
                self._safe_check(v, "v"),
            ]
        ):
            if self.debug:
                print("[OptimizedAttention] Invalid input detected. Fallback.")
            return None

        flash_attention = (
            self.flash_available if flash_attention is None else flash_attention
        )
        has_sdp = self.sdp_available if has_sdp is None else has_sdp
        has_xformers = self.xformers_available if has_xformers is None else has_xformers

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

        if self.debug:
            print("[OptimizedAttention] No optimized backend succeeded. Falling back.")
        return None
