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

class EncoderBase(nn.Module):
    def __init__(self):
        super(EncoderBase, self).__init__()

    def forward(self, x, hidden=None):
        raise NotImplementedError("Subclasses must implement forward method")

class DecoderBase(nn.Module):
    def __init__(self):
        super(DecoderBase, self).__init__()

    def forward(self, x, hidden=None):
        raise NotImplementedError("Subclasses must implement forward method")

################################################
# LSTM
################################################

class LSTMEncoder(EncoderBase):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, x, hidden=None):
        outputs, hidden = self.lstm(x, hidden)
        return outputs, hidden

class LSTMDecoder(DecoderBase):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.output_layer(lstm_out.squeeze(1))
        return output, hidden

###################################################
# GRU
###################################################

class GRUEncoder(EncoderBase):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        super(GRUEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, x, hidden=None):
        outputs, hidden = self.gru(x, hidden)
        return outputs, hidden


class GRUDecoder(DecoderBase):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        gru_out, hidden = self.gru(x, hidden)
        output = self.output_layer(gru_out.squeeze(1))
        return output, hidden

##################################################
# TRANSFORMER
##################################################

try:
    from xformers.ops import memory_efficient_attention, LowerTriangularMask
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

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

class XFormerAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, batch_first=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, is_causal=False, need_weights=False):
        if not self.batch_first:
            query, key, value = map(lambda x: x.transpose(0, 1), (query, key, value))

        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        q = self.q_proj(query).reshape(B, T_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, T_k, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, T_k, self.nhead, self.head_dim).transpose(1, 2)

        if not HAS_XFORMERS or need_weights:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if is_causal:
                causal_mask = torch.ones(T_q, T_k, device=q.device).tril()
                attn_scores = attn_scores.masked_fill(~causal_mask.bool()[None, None, :, :], float('-inf'))

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask[None, None, :, :]
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask[:, None, :, :]
                attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

            if key_padding_mask is not None:
                key_mask = key_padding_mask[:, None, None, :].expand(-1, self.nhead, T_q, -1)
                attn_scores = attn_scores.masked_fill(key_mask, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            if self.training and self.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)

            output = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, T_q, self.d_model)
            output = self.out_proj(output)

            if not self.batch_first:
                output = output.transpose(0, 1)

            return output, attn_weights if need_weights else None

        # Use xFormers
        attn_bias = LowerTriangularMask() if is_causal else None
        output = memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.dropout)
        output = output.transpose(1, 2).reshape(B, T_q, self.d_model)
        output = self.out_proj(output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, None
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = XFormerAttention(d_model, nhead, dropout=dropout, batch_first=True)

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
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = XFormerAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

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

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, _ = self.self_attn(
            self.norm1(tgt), self.norm1(tgt), self.norm1(tgt),
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt2, _ = self.multihead_attn(
            self.norm2(tgt), self.norm2(memory), self.norm2(memory),
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)

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

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.input_projection(src)
        src = self.dropout(self.pos_encoder(src))

        for layer in self.layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return self.norm(src)

class TransformerDecoder(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation='gelu', hidden_size=None):
        super().__init__()
        self.hidden_size = hidden_size if hidden_size is not None else d_model
        self.input_projection = nn.Linear(self.hidden_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_size)
        self.output_size = output_size

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

    
###############

# === Variational Encoder Wrapper ===
class VariationalEncoderWrapper(nn.Module):
    def __init__(self, base_encoder, latent_dim):
        super().__init__()
        self.base_encoder = base_encoder
        self.latent_dim = latent_dim
        self.hidden_to_mu = None
        self.hidden_to_logvar = None
        self.hidden_size = base_encoder.hidden_size

    def forward(self, x):
        encoder_outputs, encoder_hidden = self.base_encoder(x)
        if isinstance(encoder_hidden, tuple):
            h = encoder_hidden[0][-1]
        else:
            h = encoder_hidden[-1]

        # Lazy init hidden size based on actual tensor
        if self.hidden_to_mu is None:
            self.hidden_to_mu = nn.Linear(h.size(-1), self.latent_dim).to(h.device)
            self.hidden_to_logvar = nn.Linear(h.size(-1), self.latent_dim).to(h.device)

        mu = self.hidden_to_mu(h)
        logvar = self.hidden_to_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return encoder_outputs, (z, mu, logvar)
    
# === Latent-aware Decoder Wrapper ===
class LatentConditionedDecoder(nn.Module):
    def __init__(self, base_decoder, latent_dim, hidden_size, num_layers=1):
        super().__init__()
        self.base_decoder = base_decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size * num_layers)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = base_decoder.output_size

    def forward(self, x, hidden):
        z = hidden[0]
        if z.dim() == 3 and z.size(0) == self.num_layers:
            z = z[-1]  # ✅ extract last layer: [batch_size, latent_dim]
        batch_size = z.size(0)
        projected = self.latent_to_hidden(z)  # [batch, hidden_size * num_layers]

        h0 = projected.view(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros_like(h0)

        return self.base_decoder(x, (h0, c0))

# === VAE Loss Function ===
def vae_loss_function(recon_x, target, kl):
    recon_loss = F.mse_loss(recon_x, target, reduction='mean')
    return recon_loss + (kl if kl is not None else 0.0), recon_loss, kl