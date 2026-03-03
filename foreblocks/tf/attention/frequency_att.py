import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class FrequencyAttention(nn.Module):
    """Frequency-domain attention using rFFT mode mixing."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        modes: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modes = int(modes)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.freq_weight = nn.Parameter(
            torch.randn(n_heads, max(1, self.modes), self.head_dim) * 0.02
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key is None:
            key = query
        if value is None:
            value = key

        B, L_q, _ = query.shape
        L_k = key.size(1)

        q = self.q_proj(query).view(B, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, self.n_heads, self.head_dim).transpose(1, 2)

        qf = torch.fft.rfft(q, dim=2)
        kf = torch.fft.rfft(k, dim=2)
        vf = torch.fft.rfft(v, dim=2)

        n_freq = qf.size(2)
        modes = min(self.modes, n_freq)

        q_modes = qf[:, :, :modes, :]
        k_modes = kf[:, :, :modes, :]
        v_modes = vf[:, :, :modes, :]

        scores = (q_modes * torch.conj(k_modes)).real
        attn = torch.softmax(scores / math.sqrt(self.head_dim), dim=2)

        weight = self.freq_weight[:, :modes, :].unsqueeze(0)
        mixed_modes = (attn * v_modes) * weight

        out_f = torch.zeros_like(qf)
        out_f[:, :, :modes, :] = mixed_modes

        out = torch.fft.irfft(out_f, n=L_q, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, None
