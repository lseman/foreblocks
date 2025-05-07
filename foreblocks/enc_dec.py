

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.amp import autocast, GradScaler
import time
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class EncoderBase(nn.Module):
    def __init__(self):
        super(EncoderBase, self).__init__()

    def forward(self, x, hidden=None):
        raise NotImplementedError("Subclasses must implement forward method")


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


class DecoderBase(nn.Module):
    def __init__(self):
        super(DecoderBase, self).__init__()

    def forward(self, x, hidden=None):
        raise NotImplementedError("Subclasses must implement forward method")


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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import math
import math
import torch
import torch.nn as nn


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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

HAS_XFORMERS = True
from xformers.ops import memory_efficient_attention, LowerTriangularMask

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

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


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



#######################################################
# Modern transformers
#######################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, n_heads=8):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.n_heads = n_heads

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_Q, E = Q.shape
        _, _, L_K, _ = K.shape

        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        
        # Ensure sample_k is not larger than L_K
        sample_k = min(sample_k, L_K)
        
        # Make sure L_Q is not 0
        if L_Q <= 0:
            return torch.zeros((B, H, 0, L_K), device=Q.device), torch.zeros((B, H, 0), dtype=torch.long, device=Q.device)
        
        # Make sure n_top is not larger than L_Q
        n_top = min(n_top, L_Q)
        
        # Handle edge case where sample_k is 0
        if sample_k <= 0:
            sample_k = 1
            
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        
        # Matmul and squeeze
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1))
        
        # Make sure the dimension is correct - handle edge case with a single query
        if Q_K_sample.dim() == 5:  # [B, H, L_Q, 1, sample_k]
            Q_K_sample = Q_K_sample.squeeze(-2)
        
        # Find the Top-k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], 
                     torch.arange(H)[None, :, None], 
                     M_top, :]  # factor*ln(L_Q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_Q)*L_K
        
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert(L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape
        
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)
        
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V], device=V.device) / L_V).type_as(attn)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask=None):
        # Handle different input shapes
        if queries.dim() == 3:  # [B, L, D]
            # Reshape for multi-head attention [B, L, H, D/H]
            B, L, D = queries.shape
            H = self.n_heads
            
            # Make sure D is divisible by H
            assert D % H == 0, f"Hidden dimension {D} must be divisible by number of heads {H}"
            
            # Reshape to [B, L, H, D/H]
            queries = queries.view(B, L, H, -1)
            keys = keys.view(B, keys.size(1), H, -1)
            values = values.view(B, values.size(1), H, -1)
            
        # Transform from [B, L, H, D/H] to [B, H, L, D/H]
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Handle edge cases with small sequences
        if L_Q == 0 or L_K == 0:
            # Return empty tensors of correct shape
            output = torch.zeros(B, H, L_Q, D, device=queries.device)
            return output.transpose(1, 2).contiguous(), None
        
        # Compute the sparse attention
        U_part = self.factor * np.ceil(np.log(max(L_K, 1))).astype('int').item()  # c*ln(L_K)
        u = self.factor * np.ceil(np.log(max(L_Q, 1))).astype('int').item()  # c*ln(L_Q)
        
        # Ensure at least 1 sample
        U_part = max(1, min(U_part, L_K))
        u = max(1, min(u, L_Q))
        
        scores_top, index = self._prob_QK(queries, keys, U_part, u)
        
        # Add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        
        # Get the context
        context = self._get_initial_context(values, L_Q)
        
        # Update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)
        
        return context.transpose(1, 2).contiguous(), attn


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device)
        _mask[index] = False
        self.mask = _mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, scores.shape[-1])




class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, d_ff=None, dropout=0.1, activation="relu"):
        super(InformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.n_heads = n_heads
        
        self.attention = ProbAttention(
            mask_flag=False, 
            attention_dropout=dropout, 
            output_attention=False,
            n_heads=n_heads
        )
        
        self.attn_norm = nn.LayerNorm(d_model)
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x):
        # Reshape input if not in proper format
        if x.dim() == 3:  # [B, L, D]
            batch_size, seq_len, d_model = x.shape
            x = x.reshape(batch_size, seq_len, self.n_heads, -1)
        
        # Multi-head attention
        new_x, attn = self.attention(x, x, x)
        
        # Reshape for feed-forward
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], -1)
        new_x = new_x.reshape(x_shape[0], x_shape[1], -1)
        
        # Add & norm
        x = self.attn_norm(x + self.dropout(new_x))
        
        # Feed-forward
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, -2))))
        y = self.dropout(self.conv2(y)).transpose(-1, -2)
        
        # Add & norm
        x = self.ffn_norm(x + y)
        
        # Reshape back for next layer
        x = x.reshape(x_shape[0], x_shape[1], self.n_heads, -1)
        
        return x


class DistilledAttention(nn.Module):
    def __init__(self, factor=5, distil_dropout=0.1):
        super(DistilledAttention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(distil_dropout)
        
    def forward(self, x):
        B, L, H, D = x.shape
        
        # Handle edge case where L is 0
        if L <= 0:
            return x
            
        # Handle edge case where factor is greater than L
        if self.factor >= L:
            x_pooled = x.mean(dim=1, keepdim=True)
            return self.dropout(x_pooled)
        
        # Take the mean of every 'factor' elements
        remaining = L % self.factor
        if remaining > 0:
            # Pad the sequence to be divisible by factor
            padding = self.factor - remaining
            x_padded = torch.cat([x, torch.zeros(B, padding, H, D, device=x.device)], dim=1)
            x_reshaped = x_padded.reshape(B, -1, self.factor, H, D)
        else:
            x_reshaped = x.reshape(B, -1, self.factor, H, D)
            
        x_pooled = x_reshaped.mean(dim=2)
        
        return self.dropout(x_pooled)


class InformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, nhead=8, num_layers=3, dropout=0.1, distil=True):
        super(InformerEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.distil = distil
        self.nhead = nhead
        
        # Embedding
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_embedding = PositionalEncoding(hidden_size)
        
        # Encoder layers
        self.layer_stack = nn.ModuleList([
            InformerEncoderLayer(
                hidden_size, 
                n_heads=nhead, 
                d_ff=hidden_size*4, 
                dropout=dropout, 
                activation="gelu"
            ) for _ in range(num_layers)
        ])
        
        # Distilling
        if self.distil:
            self.distil_conv = nn.ModuleList([
                DistilledAttention(factor=2, distil_dropout=dropout) for _ in range(num_layers-1)
            ])
        
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        # x: [B, L, D]
        batch_size, seq_len, _ = x.shape
        
        # Embedding
        x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        
        # Reshape for multi-head: [B, L, H, D//H]
        head_dim = self.hidden_size // self.nhead
        x = x.reshape(batch_size, seq_len, self.nhead, head_dim)
        
        # Encoding
        attns = []
        for i, layer in enumerate(self.layer_stack):
            x = layer(x)
            
            if self.distil and i < len(self.layer_stack) - 1:
                x = self.distil_conv[i](x)
        
        # Reshape back for output
        x = x.reshape(batch_size, x.shape[1], -1)
        output = self.output_layer(x)
        
        return output, output


class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, d_ff=None, dropout=0.1, activation="relu"):
        super(InformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.n_heads = n_heads
        
        # Self-attention
        self.self_attention = ProbAttention(
            mask_flag=True, 
            attention_dropout=dropout, 
            output_attention=False,
            n_heads=n_heads
        )
        
        # Cross-attention
        self.cross_attention = ProbAttention(
            mask_flag=False, 
            attention_dropout=dropout, 
            output_attention=False,
            n_heads=n_heads
        )
        
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Input shape check and reshape if needed
        if x.dim() == 3:  # [B, L, D]
            batch_size, seq_len, d_model = x.shape
            x = x.reshape(batch_size, seq_len, self.n_heads, -1)
            
        if cross.dim() == 3:  # [B, L, D]
            batch_size, seq_len, d_model = cross.shape
            cross = cross.reshape(batch_size, seq_len, self.n_heads, -1)
        
        # Self-attention
        x_attn, _ = self.self_attention(x, x, x, x_mask)
        
        # Reshape for residual connection
        x_shape = x.shape
        x_flat = x.reshape(x_shape[0], x_shape[1], -1)
        x_attn_flat = x_attn.reshape(x_shape[0], x_shape[1], -1)
        
        # Add & norm
        x_flat = self.self_attn_norm(x_flat + self.dropout(x_attn_flat))
        
        # Reshape back for cross-attention
        x = x_flat.reshape(x_shape)
        
        # Cross-attention
        cross_attn, _ = self.cross_attention(x, cross, cross, cross_mask)
        
        # Reshape for residual connection
        cross_attn_flat = cross_attn.reshape(x_shape[0], x_shape[1], -1)
        
        # Add & norm
        x_flat = self.cross_attn_norm(x_flat + self.dropout(cross_attn_flat))
        
        # Feed-forward
        y = self.dropout(self.activation(self.conv1(x_flat.transpose(-1, -2))))
        y = self.dropout(self.conv2(y)).transpose(-1, -2)
        
        # Add & norm
        x_flat = self.ffn_norm(x_flat + y)
        
        # Reshape back for next layer
        x = x_flat.reshape(x_shape)
        
        return x


class InformerDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=8, num_layers=3, dropout=0.1):
        super(InformerDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nhead = nhead
        
        # Embedding
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_embedding = PositionalEncoding(hidden_size)
        
        # Decoder layers
        self.layer_stack = nn.ModuleList([
            InformerDecoderLayer(
                hidden_size, 
                n_heads=nhead, 
                d_ff=hidden_size*4, 
                dropout=dropout, 
                activation="gelu"
            ) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def _generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        return mask > 0.5  # Convert to boolean mask
    
    def forward(self, x, encoder_output):
        # x: [B, L, D]
        batch_size, seq_len, _ = x.shape
        
        # Embedding
        x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        
        # Generate masks for self-attention (subsequent positions)
        device = x.device
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device) if seq_len > 0 else None
        
        # Reshape for multi-head attention
        head_dim = self.hidden_size // self.nhead
        x = x.reshape(batch_size, seq_len, self.nhead, head_dim)
        
        # Reshape encoder output for cross-attention if needed
        if encoder_output.dim() == 3:  # [B, L, D]
            encoder_output = encoder_output.reshape(
                encoder_output.shape[0], encoder_output.shape[1], self.nhead, head_dim
            )
        
        # Decoding
        for layer in self.layer_stack:
            x = layer(x, encoder_output, tgt_mask, None)
        
        # Reshape back for output
        x = x.reshape(batch_size, x.shape[1], -1)
        
        # Take the last step for output
        if x.shape[1] > 0:
            output = self.output_layer(x[:, -1, :])
        else:
            output = torch.zeros(batch_size, self.output_size, device=device)
        
        return output, encoder_output

#####################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple



class DecompositionLayer(nn.Module):
    """
    Decomposition layer for Autoformer.
    Decomposes the input into trend and seasonal components.
    """
    def __init__(self, kernel_size=25):
        super(DecompositionLayer, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
    def forward(self, x):
        # x: [B, L, D]
        # Extract trend component with moving average
        batch_size, seq_len, dim = x.shape
        
        # Padding for handling edge cases with sequence length
        if seq_len < self.kernel_size:
            # If sequence is shorter than kernel, adjust kernel size
            temp_kernel = seq_len if seq_len % 2 == 1 else seq_len - 1
            if temp_kernel < 3:
                temp_kernel = 3  # Minimum kernel size
            temp_avg = nn.AvgPool1d(kernel_size=temp_kernel, stride=1, padding=temp_kernel//2)
            x_permuted = x.permute(0, 2, 1)  # [B, D, L]
            trend = temp_avg(x_permuted).permute(0, 2, 1)  # [B, L, D]
        else:
            x_permuted = x.permute(0, 2, 1)  # [B, D, L]
            trend = self.avg(x_permuted).permute(0, 2, 1)  # [B, L, D]
        
        # Ensure trend has the same dimension as the input
        if trend.size(1) != x.size(1):
            # Adjust trend size to match x
            trend = F.interpolate(trend.permute(0, 2, 1), size=x.size(1), mode='linear', align_corners=False)
            trend = trend.permute(0, 2, 1)
        
        # Extract seasonal component (residual)
        seasonal = x - trend
        
        return seasonal, trend