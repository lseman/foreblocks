import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTAttention(nn.Module):
    """
    Discrete Wavelet Transform attention.
    Alternative to frequency attention using wavelets instead of FFT.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        modes: int = 32,
        wavelet: str = "db4",
    ):
        super().__init__()
        print("[Attention] Using DWT attention")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modes = modes
        self.wavelet = wavelet

        try:
            import pywt

            self.pywt = pywt
            self.has_pywt = True
        except ImportError:
            print(
                "Warning: PyWavelets not available. DWT attention will use simple approximation."
            )
            self.has_pywt = False

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.wavelet_weight = nn.Parameter(
            torch.randn(n_heads, modes, self.head_dim) * 0.02
        )

    def _simple_dwt(self, x):
        """Simple DWT approximation using average pooling and differences"""
        B, H, L, D = x.shape

        if L % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode="reflect")
            L += 1

        approx = (x[:, :, ::2, :] + x[:, :, 1::2, :]) / 2
        detail = (x[:, :, ::2, :] - x[:, :, 1::2, :]) / 2

        return torch.cat([approx, detail], dim=2)

    def _simple_idwt(self, coeffs, target_len):
        """Simple inverse DWT approximation"""
        B, H, L, D = coeffs.shape
        half_L = L // 2

        approx = coeffs[:, :, :half_L, :]
        detail = coeffs[:, :, half_L:, :]

        even = approx + detail
        odd = approx - detail

        result = torch.zeros(
            B, H, half_L * 2, D, device=coeffs.device, dtype=coeffs.dtype
        )
        result[:, :, ::2, :] = even
        result[:, :, 1::2, :] = odd

        return result[:, :, :target_len, :]

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

        q = self.q_proj(query).view(B, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        v = (
            self.v_proj(value)
            .view(B, value.size(1), self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(B, key.size(1), self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        q_dwt = self._simple_dwt(q)
        k_dwt = self._simple_dwt(k)
        v_dwt = self._simple_dwt(v)

        modes = min(self.modes, q_dwt.size(2))
        q_modes = q_dwt[:, :, :modes, :]
        k_modes = k_dwt[:, :, :modes, :]
        v_modes = v_dwt[:, :, :modes, :]

        scores = q_modes * k_modes
        attn = torch.softmax(scores / math.sqrt(self.head_dim), dim=2)
        mixed = torch.einsum(
            "bhmd,hmd->bhmd", attn * v_modes, self.wavelet_weight[:, :modes, :]
        )

        out_dwt = torch.zeros_like(q_dwt)
        out_dwt[:, :, :modes, :] = mixed

        out_time = self._simple_idwt(out_dwt, L_q)

        out = out_time.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out, None
