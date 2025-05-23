import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyAttention(nn.Module):
    """
    Frequency-domain attention from FEDformer, AMP-safe and corrected.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        modes=64,
        seq_len_q=None,
        seq_len_kv=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modes = modes
        self.scaling = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.frequencies_q = nn.Parameter(torch.randn(n_heads, modes, self.head_dim))
        self.frequencies_k = nn.Parameter(torch.randn(n_heads, modes, self.head_dim))
        self.frequencies_v = nn.Parameter(torch.randn(n_heads, modes, self.head_dim))

        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.fft_base_q = None
        self.fft_base_kv = None

        if seq_len_q is not None:
            self._init_fft_bases(seq_len_q, seq_len_kv or seq_len_q)

        print("[Attention] FrequencyAttention initialized with modes:", modes)
    def _init_fft_bases(self, len_q, len_kv):
        device = self.frequencies_q.device
        n_fft_q = 2 ** math.ceil(math.log2(len_q))
        fft_base_q = torch.fft.fft(torch.eye(n_fft_q, dtype=torch.float32, device=device))
        self.fft_base_q = fft_base_q[:, :len_q].to(torch.complex64)

        if len_kv != len_q:
            n_fft_kv = 2 ** math.ceil(math.log2(len_kv))
            fft_base_kv = torch.fft.fft(torch.eye(n_fft_kv, dtype=torch.float32, device=device))
            self.fft_base_kv = fft_base_kv[:, :len_kv].to(torch.complex64)
        else:
            self.fft_base_kv = self.fft_base_q

    def _to_frequency_domain(self, x, dim, fft_base=None):
        bsz, n_head, seq_len, head_dim = x.shape
        x = x.float()  # AMP safety

        if fft_base is not None and fft_base.size(1) == seq_len:
            x_complex = x.transpose(2, 3).to(torch.complex64)
            return torch.matmul(x_complex, fft_base.conj().t()).transpose(2, 3)

        return torch.fft.fft(x, dim=dim)

    def _from_frequency_domain(self, x, seq_len, dim, fft_base=None):
        if fft_base is not None and x.size(dim) == fft_base.size(0):
            x_ifft = torch.matmul(x.transpose(2, 3), fft_base).transpose(2, 3)
            return x_ifft.real

        x_ifft = torch.fft.ifft(x, dim=dim)
        return x_ifft.real[:, :, :seq_len]

    def _apply_frequency_filter(self, x_fft, freq_weights):
        bsz, n_head, seq_len, head_dim = x_fft.shape
        x_fft_truncated = x_fft[:, :, :self.modes]
        x_fft_filtered = torch.einsum('bhfm,hfm->bhfm', x_fft_truncated, freq_weights)

        if self.modes < seq_len:
            x_fft_full = torch.zeros_like(x_fft)
            x_fft_full[:, :, :self.modes] = x_fft_filtered
            return x_fft_full
        else:
            return x_fft_filtered

    def forward(self, query, key=None, value=None, attn_mask=None,
                key_padding_mask=None, is_causal=False, need_weights=False):
        if key is None and value is None:
            key = value = query

        bsz, seq_len_q = query.size(0), query.size(1)
        seq_len_kv = key.size(1)

        if (self.seq_len_q != seq_len_q or self.seq_len_kv != seq_len_kv) and \
           (self.fft_base_q is None or self.fft_base_kv is None):
            self._init_fft_bases(seq_len_q, seq_len_kv)
            self.seq_len_q, self.seq_len_kv = seq_len_q, seq_len_kv

        if key is value and key is query:
            q, k, v = self.qkv_proj(query).chunk(3, dim=-1)
        else:
            q = self.qkv_proj(query)[:, :, :self.d_model]
            if key is value:
                k, v = self.qkv_proj(key)[:, :, self.d_model:].chunk(2, dim=-1)
            else:
                k = self.qkv_proj(key)[:, :, self.d_model:2*self.d_model]
                v = self.qkv_proj(value)[:, :, 2*self.d_model:]

        q = q.view(bsz, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)

        q_fft = self._to_frequency_domain(q, dim=2, fft_base=self.fft_base_q)
        k_fft = self._to_frequency_domain(k, dim=2, fft_base=self.fft_base_kv)
        v_fft = self._to_frequency_domain(v, dim=2, fft_base=self.fft_base_kv)

        q_fft = self._apply_frequency_filter(q_fft, self.frequencies_q)
        k_fft = self._apply_frequency_filter(k_fft, self.frequencies_k)
        v_fft = self._apply_frequency_filter(v_fft, self.frequencies_v)

        # Frequency attention: Q·Kᵀ -> softmax -> V
        attn_scores = (q_fft * k_fft.conj()).real.sum(-1, keepdim=True)
        attn_weights = F.softmax(attn_scores, dim=2)
        output_fft = attn_weights * v_fft

        output = self._from_frequency_domain(output_fft, seq_len_q, dim=2,
                                             fft_base=self.fft_base_q)

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len_q, self.d_model)
        output = self.out_proj(output)

        if self.training:
            output = self.dropout(output)

        return output, None
