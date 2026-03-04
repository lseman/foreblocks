"""Transformer encoder and decoder blocks."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_attention import LinearSelfAttention
from .bb_primitives import RMSNorm, SwiGLUFFN

__all__ = [
    "LightweightTransformerEncoder",
    "PatchTSTEncoder",
    "LightweightTransformerDecoder",
]


class LightweightTransformerEncoder(nn.Module):
    """Improved transformer encoder with better RNN compatibility"""

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_layers=2,
        dropout=0.1,
        nhead=4,
        max_seq_len=512,
        causal=False,
        rope_base: float = 500000.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.causal = causal

        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": LinearSelfAttention(
                            latent_dim,
                            heads=nhead,
                            dropout=dropout,
                            causal=causal,
                            rope_base=rope_base,
                            rope_max_seq_len=max_seq_len,
                        ),
                        "ffn": SwiGLUFFN(latent_dim, expand=4),
                        "norm1": RMSNorm(latent_dim),
                        "norm2": RMSNorm(latent_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(latent_dim)
        self.dropout_p = dropout
        self.state_proj = nn.Linear(latent_dim, latent_dim * 2, bias=False)

    def forward(self, x, hidden_state=None):
        B, T, _ = x.shape
        x = self.input_proj(x)

        for layer in self.layers:
            attn_out = layer["self_attn"](layer["norm1"](x))
            if self.training and self.dropout_p > 0:
                attn_out = F.dropout(attn_out, p=self.dropout_p)
            x = x + attn_out

            ffn_out = layer["ffn"](layer["norm2"](x))
            if self.training and self.dropout_p > 0:
                ffn_out = F.dropout(ffn_out, p=self.dropout_p)
            x = x + ffn_out

        x = self.final_norm(x)

        context = x[:, -1:, :]
        pooled = x.mean(dim=1)
        state_proj = self.state_proj(pooled)
        h_state, c_state = state_proj.chunk(2, dim=-1)
        h_state = h_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_state = c_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

        return x, context, (h_state, c_state)


class PatchTSTEncoder(nn.Module):
    """PatchTST-style patch-based encoder option for MixedEncoder.

    Slices the sequence into overlapping patches (channel-independent),
    projects each patch to *latent_dim*, adds learnable positional
    embeddings, processes with *num_layers* transformer blocks, then
    interpolates the patch tokens back to the original sequence length
    so the output shape is identical to LSTM/GRU/Transformer encoders.
    Returns ``(output [B,L,D], context [B,1,D], (h [nl,B,D], c [nl,B,D]))``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        patch_size: int = 16,
        stride: Optional[int] = None,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.patch_size = max(2, int(patch_size))
        self.stride = max(
            1, int(stride if stride is not None else self.patch_size // 2)
        )

        padded_len = max(seq_len, self.patch_size)
        self.num_patches = (padded_len - self.patch_size) // self.stride + 1

        patch_dim = input_dim * self.patch_size
        self.patch_proj = nn.Linear(patch_dim, latent_dim, bias=False)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": LinearSelfAttention(
                            latent_dim,
                            heads=nhead,
                            dropout=dropout,
                            causal=False,
                        ),
                        "ffn": SwiGLUFFN(latent_dim, expand=4),
                        "norm1": RMSNorm(latent_dim),
                        "norm2": RMSNorm(latent_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(latent_dim)
        self.dropout_p = dropout
        self.state_proj = nn.Linear(latent_dim, latent_dim * 2, bias=False)

    def forward(
        self, x: torch.Tensor, hidden_state=None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, C = x.shape

        x_t = x.transpose(1, 2)
        if L < self.patch_size:
            x_t = F.pad(x_t, (0, self.patch_size - L))

        patches = x_t.unfold(2, self.patch_size, self.stride)
        N = patches.size(2)
        patches = (
            patches.permute(0, 2, 1, 3).contiguous().view(B, N, C * self.patch_size)
        )

        tokens = self.patch_proj(patches)
        if N != self.pos_embed.size(1):
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=N,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.pos_embed
        tokens = tokens + pos

        for layer in self.layers:
            attn_out = layer["self_attn"](layer["norm1"](tokens))
            if self.training and self.dropout_p > 0:
                attn_out = F.dropout(attn_out, p=self.dropout_p)
            tokens = tokens + attn_out

            ffn_out = layer["ffn"](layer["norm2"](tokens))
            if self.training and self.dropout_p > 0:
                ffn_out = F.dropout(ffn_out, p=self.dropout_p)
            tokens = tokens + ffn_out

        tokens = self.final_norm(tokens)

        output = F.interpolate(
            tokens.transpose(1, 2), size=L, mode="linear", align_corners=False
        ).transpose(1, 2)

        context = tokens[:, -1:, :]
        pooled = tokens.mean(dim=1)
        h_state, c_state = self.state_proj(pooled).chunk(2, dim=-1)
        h_state = h_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_state = c_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

        return output, context, (h_state, c_state)


class LightweightTransformerDecoder(nn.Module):
    """Improved transformer decoder with better compatibility"""

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_layers=2,
        dropout=0.1,
        nhead=4,
        max_seq_len=512,
        causal=True,
        rope_base: float = 500000.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.causal = causal

        self.input_proj = nn.Linear(input_dim, latent_dim, bias=False)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": LinearSelfAttention(
                            latent_dim,
                            heads=nhead,
                            dropout=dropout,
                            causal=causal,
                            rope_base=rope_base,
                            rope_max_seq_len=max_seq_len,
                        ),
                        "cross_attn": nn.MultiheadAttention(
                            latent_dim,
                            nhead,
                            dropout=dropout,
                            batch_first=True,
                            bias=False,
                        ),
                        "ffn": SwiGLUFFN(latent_dim, expand=4),
                        "norm1": RMSNorm(latent_dim),
                        "norm2": RMSNorm(latent_dim),
                        "norm3": RMSNorm(latent_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(latent_dim)
        self.dropout_p = dropout
        self.state_proj = nn.Linear(latent_dim, latent_dim * 2, bias=False)

    def _prepare_memory(self, memory_or_hidden):
        if memory_or_hidden is None:
            return None

        if isinstance(memory_or_hidden, tuple):
            if len(memory_or_hidden) == 2:
                h, c = memory_or_hidden
                if h.dim() == 3:
                    if h.size(0) == self.num_layers:
                        memory = h.transpose(0, 1)
                    else:
                        memory = h
                else:
                    memory = h.unsqueeze(1)
            else:
                memory = memory_or_hidden[0]
        else:
            if memory_or_hidden.dim() == 3:
                if memory_or_hidden.size(0) == self.num_layers:
                    memory = memory_or_hidden.transpose(0, 1)
                else:
                    memory = memory_or_hidden
            else:
                memory = memory_or_hidden.unsqueeze(1)

        return memory

    def forward(self, tgt, memory_or_hidden, hidden_state=None):
        tgt = self.input_proj(tgt)
        memory = self._prepare_memory(memory_or_hidden)

        for layer in self.layers:
            self_attn_out = layer["self_attn"](layer["norm1"](tgt))
            if self.training and self.dropout_p > 0:
                self_attn_out = F.dropout(self_attn_out, p=self.dropout_p)
            tgt = tgt + self_attn_out

            if memory is not None:
                try:
                    cross_out, _ = layer["cross_attn"](
                        layer["norm2"](tgt), memory, memory
                    )
                    if self.training and self.dropout_p > 0:
                        cross_out = F.dropout(cross_out, p=self.dropout_p)
                    tgt = tgt + cross_out
                except (RuntimeError, ValueError):
                    pass

            ffn_out = layer["ffn"](layer["norm3"](tgt))
            if self.training and self.dropout_p > 0:
                ffn_out = F.dropout(ffn_out, p=self.dropout_p)
            tgt = tgt + ffn_out

        tgt = self.final_norm(tgt)

        last_token = tgt[:, -1]
        state_proj = self.state_proj(last_token)
        h_state, c_state = state_proj.chunk(2, dim=-1)
        h_state = h_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_state = c_state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

        return tgt, (h_state, c_state)
