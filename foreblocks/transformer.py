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

##################################################
# TRANSFORMER
##################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [D/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:x.size(1)].unsqueeze(0)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_len=5000):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(d_model=hidden_size, max_len=max_len)

    def forward(self, x):
        # x: [B, T, input_size]
        x = self.input_projection(x)         # [B, T, hidden_size]
        x = self.positional_encoding(x)      # [B, T, hidden_size]
        return x


try:
    from torch.nn.functional import scaled_dot_product_attention as torch_sdp_attention
    HAS_TORCH_SDP = True
except ImportError:
    HAS_TORCH_SDP = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import xformers.ops as xops
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False
from xformers.ops import memory_efficient_attention

class XFormerAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True, cross_attention=False,
                 use_flash_attn=False, attention_type="standard", prob_sparse_factor=0.4):
        
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.batch_first = batch_first
        self.cross_attention = cross_attention
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout_f = nn.Dropout(dropout)

        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN
        self.attention_type = attention_type
        self.prob_sparse_factor = prob_sparse_factor
   
    def _prob_sparse_attention(self, q, k, v, attn_mask=None, key_padding_mask=None, is_causal=False, need_weights=False):
        """
        Optimized ProbSparse attention with optional xFormers support.
        """
        B, H, T_q, D = q.shape
        _, _, T_k, _ = k.shape
        scale = math.sqrt(D)

        # === KL divergence for sparsity scores
        attn_scores_all = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_probs = F.softmax(attn_scores_all, dim=-1)
        uniform = torch.full_like(attn_probs, 1.0 / T_k)
        kl_div = torch.sum(attn_probs * (torch.log(attn_probs + 1e-10) - torch.log(uniform + 1e-10)), dim=-1)

        # === Select top-u queries
        u = int(min(self.prob_sparse_factor * math.log(T_k), T_q))
        topk = torch.topk(kl_div, u, dim=-1).indices  # [B, H, u]
        q_reduced = torch.gather(q, 2, topk.unsqueeze(-1).expand(-1, -1, -1, D))  # [B, H, u, D]

        # === Optionally use xFormers if possible
        use_xformers = (
            not need_weights and attn_mask is None and key_padding_mask is None and
            hasattr(self, "use_flash_attn") and self.use_flash_attn and
            q_reduced.shape[2] == k.shape[2]  # Mq == Mk required for xformers
        )

        if use_xformers:
            from xformers.ops import memory_efficient_attention
            out_reduced = memory_efficient_attention(q_reduced, k, v)  # [B, H, u, D]
            attn_weights = None
        else:
            # Compute scores
            attn_scores = torch.matmul(q_reduced, k.transpose(-2, -1)) / scale  # [B, H, u, T_k]

            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask[None, None, :, :]
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask[:, None, :, :]
                if attn_mask.size(2) == T_q:
                    attn_mask_reduced = torch.gather(attn_mask, 2, topk.unsqueeze(-1).expand(-1, -1, -1, T_k))
                    attn_scores = attn_scores.masked_fill(attn_mask_reduced == 0, float('-inf'))
                else:
                    attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

            attn_scores = torch.nan_to_num(attn_scores, nan=-1e4)
            attn_scores = torch.clamp(attn_scores, min=-1e4, max=1e4)
            attn_weights = F.softmax(attn_scores, dim=-1)
            if self.training and self.dropout > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            out_reduced = torch.matmul(attn_weights, v)

        # === Scatter back to full output
        out = torch.zeros(B, H, T_q, D, device=q.device, dtype=q.dtype)
        out.scatter_(2, topk.unsqueeze(-1).expand(-1, -1, -1, D), out_reduced)

        # === Final projection
        out = out.transpose(1, 2).reshape(B, T_q, self.d_model)
        out = self.out_proj(out)

        return (out, attn_weights if need_weights else None)


    def forward(
        self, query, key, value,
        attn_mask=None, key_padding_mask=None,
        is_causal=False, need_weights=False,
    ):
        if not self.batch_first:
            query, key, value = map(lambda x: x.transpose(0, 1), (query, key, value))

        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        # Project to QKV and reshape
        q = self.q_proj(query).reshape(B, T_q, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, T_q, D]
        k = self.k_proj(key).reshape(B, T_k, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, T_k, self.nhead, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0

        if self.attention_type == "prob_sparse":
            return self._prob_sparse_attention(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights)

        # Case 1: Use FlashAttention
        if HAS_FLASH_ATTN and attn_mask is None and key_padding_mask is None and not need_weights and self.use_flash_attn:
            q = q.contiguous().to(torch.float16)
            k = k.contiguous().to(torch.float16)
            v = v.contiguous().to(torch.float16)

            out = flash_attn_func(q, k, v, dropout_p=self.dropout_f.p, causal=False)  # [B, nhead, T, head_dim]
            out = out.transpose(1, 2)  # [B, T, nhead, head_dim]
            out = out.contiguous().view(B, -1, self.nhead * self.head_dim)  # [B, T, nhead * head_dim]
            out = out.to(torch.float32)  # convert back to float32
            return self.out_proj(out), None
        if HAS_TORCH_SDP and attn_mask is None and key_padding_mask is None and not need_weights:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=is_causal and not self.cross_attention
            )
            out = out.transpose(1, 2).reshape(B, T_q, self.d_model)
            return self.out_proj(out), None

        # Case 2: Use xFormers memory-efficient attention
        if HAS_XFORMERS and attn_mask is None and key_padding_mask is None and not need_weights:
            bias = None if not is_causal or self.cross_attention else xops.LowerTriangularMask()
            out = xops.memory_efficient_attention(q, k, v, attn_bias=bias, p=dropout_p)
            out = out.transpose(1, 2).reshape(B, T_q, self.d_model)
            return self.out_proj(out), None

        # Case 3: Fallback to manual attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, T_q, T_k]

        if is_causal and not self.cross_attention:
            mask = torch.ones(T_q, T_k, device=query.device).tril()
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None, :, :]
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :].expand(B, self.nhead, T_q, T_k)
            attn_scores = attn_scores.masked_fill(kpm, float('-inf'))

        attn_scores = torch.nan_to_num(attn_scores, nan=-1e4)
        attn_scores = torch.clamp(attn_scores, min=-1e4, max=1e4)

        attn_weights = F.softmax(attn_scores, dim=-1)
        if dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        out = torch.matmul(attn_weights, v)  # [B, H, T_q, D]
        out = out.transpose(1, 2).reshape(B, T_q, self.d_model)
        out = self.out_proj(out)

        return (out, attn_weights) if need_weights else (out, None)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = XFormerAttention(d_model, nhead, dropout=dropout, batch_first=True, use_flash_attn=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(
            self.norm1(src), self.norm1(src), self.norm1(src),
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu", informer_like=False):
        super().__init__()
        self.self_attn = XFormerAttention(d_model, nhead, dropout=dropout, batch_first=True, use_flash_attn=True)
        self.multihead_attn = XFormerAttention(d_model, nhead, dropout=dropout, batch_first=True, cross_attention=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

        self.informer_like = informer_like

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        # Decoder self-attention (causal!)
        if not self.informer_like:
            tgt2, _ = self.self_attn(
                self.norm1(tgt), self.norm1(tgt), self.norm1(tgt),
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=True  # <-- This is important
            )
        else:
            # Informer-style — remove causal masking
            tgt2, _ = self.self_attn(
                self.norm1(tgt), self.norm1(tgt), self.norm1(tgt),
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=False  # Informer-style: NOT autoregressive
            )

        tgt = tgt + self.dropout1(tgt2)

        # Cross-attention (NOT causal)
        tgt2, _ = self.multihead_attn(
            self.norm2(tgt), self.norm2(memory), self.norm2(memory),
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
            # No is_causal here
        )
        tgt = tgt + self.dropout2(tgt2)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation='gelu', hidden_size=None):
        super().__init__()
        self.hidden_size = hidden_size if hidden_size is not None else d_model
        self.input_projection = nn.Linear(self.hidden_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.input_size = input_size
        self.is_transformer = True

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.input_projection(src)
        src = self.dropout(self.pos_encoder(src))

        for layer in self.layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return self.norm(src)

class TransformerDecoder(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation='gelu', hidden_size=None, informer_like=False):
        super().__init__()
        self.hidden_size = hidden_size if hidden_size is not None else d_model
        self.input_projection = nn.Linear(self.hidden_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, informer_like)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_size)
        self.output_size = output_size
        self.is_transformer = True

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.input_projection(tgt)
        tgt = self.dropout(self.pos_encoder(tgt))

        for layer in self.layers:
            tgt = layer(
                tgt, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        tgt = self.norm(tgt)
        return self.output_projection(tgt)