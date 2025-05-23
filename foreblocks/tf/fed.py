import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyAttention(nn.Module):
    """
    Frequency-domain attention as proposed in FEDformer (Zhou et al. 2022).
    This module directly mixes Q and V in the frequency domain using learned filters.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        modes: int = 64,
        seq_len_q: int = None,
        seq_len_kv: int = None,
    ):
        super().__init__()
        print("[Attention] Using frequency attention")
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modes = modes
        self.dropout = nn.Dropout(dropout)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable frequency filters (real only, like FEDformer)
        self.freq_filter_q = nn.Parameter(torch.randn(n_heads, modes, self.head_dim))
        self.freq_filter_v = nn.Parameter(torch.randn(n_heads, modes, self.head_dim))

        # Optional fixed FFT bases
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.fft_base_q = None
        self.fft_base_kv = None
        if seq_len_q is not None:
            self._init_fft_bases(seq_len_q, seq_len_kv or seq_len_q)

    def _init_fft_bases(self, len_q: int, len_kv: int):
        def fft_eye(n, length):
            base = torch.fft.fft(torch.eye(n))
            return base[:, :length].to(torch.complex64)

        device = self.freq_filter_q.device
        n_fft_q = 2 ** math.ceil(math.log2(len_q))
        n_fft_kv = 2 ** math.ceil(math.log2(len_kv))

        self.fft_base_q = fft_eye(n_fft_q, len_q).to(device)
        self.fft_base_kv = fft_eye(n_fft_kv, len_kv).to(device)

    def _to_fft(self, x, fft_base):
        # x: [B, H, L, D] → [B, H, modes, D]
        x = x.float()
        if fft_base is not None and fft_base.size(1) == x.size(2):
            x = x.to(torch.complex64)
            return torch.matmul(x.transpose(2, 3), fft_base.conj().T).transpose(2, 3)
        return torch.fft.fft(x, dim=2)

    def _ifft(self, x, fft_base, seq_len):
        if fft_base is not None and x.size(2) <= fft_base.size(0):
            x_ifft = torch.matmul(x.transpose(2, 3), fft_base).transpose(2, 3)
            return x_ifft.real[..., :seq_len]
        return torch.fft.ifft(x, dim=2).real[..., :seq_len]

    def _filter(self, x_fft, filt):
        # x_fft: [B, H, L_fft, D], filt: [H, modes, D]
        x_fft_trunc = x_fft[:, :, : self.modes]  # [B, H, modes, D]
        filtered = torch.einsum("bhfd,hfd->bhfd", x_fft_trunc, filt)
        if self.modes < x_fft.size(2):
            pad = torch.zeros_like(x_fft)
            pad[:, :, : self.modes] = filtered
            return pad
        return filtered

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
        need_weights=False,
    ):
        if key is None and value is None:
            key = value = query

        B, L_q, _ = query.size()
        L_kv = key.size(1)

        # Initialize FFT bases if needed
        if (self.seq_len_q != L_q or self.seq_len_kv != L_kv) and (
            self.fft_base_q is None or self.fft_base_kv is None
        ):
            self._init_fft_bases(L_q, L_kv)
            self.seq_len_q, self.seq_len_kv = L_q, L_kv

        # Project Q, K, V
        if key is query and value is query:
            q, k, v = self.qkv_proj(query).chunk(3, dim=-1)
        else:
            q = self.qkv_proj(query)[..., : self.d_model]
            kv = self.qkv_proj(key)
            k = kv[..., self.d_model : 2 * self.d_model]
            v = kv[..., 2 * self.d_model :]

        q = q.view(B, L_q, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, L_q, D]
        v = v.view(B, L_kv, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, L_kv, D]

        # FFT
        q_fft = self._to_fft(q, self.fft_base_q)  # [B, H, L_q_fft, D]
        v_fft = self._to_fft(v, self.fft_base_kv)  # [B, H, L_kv_fft, D]

        # Apply learnable frequency filters
        q_fft = self._filter(q_fft, self.freq_filter_q)
        v_fft = self._filter(v_fft, self.freq_filter_v)

        # Frequency attention: elementwise complex multiplication
        out_fft = q_fft * v_fft

        # Inverse FFT back to time domain
        out_time = self._ifft(out_fft, self.fft_base_q, L_q)  # [B, H, L_q, D]
        out = (
            out_time.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        )  # [B, L_q, D]

        return self.dropout(self.out_proj(out)), None


import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class DWTAttention(nn.Module):
    """
    Wavelet-domain attention (DWT-based) for temporal localization.
    Inspired by FEDformer, replaces FFT with DWT for better time-scale sensitivity.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        modes: int = 3,
        wavelet: str = "db1",
    ):
        super().__init__()
        print("[Attention] Using DWT attention")
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modes = modes  # Number of DWT decomposition levels to keep
        self.wavelet = wavelet

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable weights per head, per level, per dimension
        self.weight_q = nn.Parameter(torch.randn(n_heads, modes, self.head_dim))
        self.weight_v = nn.Parameter(torch.randn(n_heads, modes, self.head_dim))

    def _apply_dwt(self, x, level):
        """
        x: [B, H, L, D]
        Returns list of coeffs: [approx, detail1, detail2, ..., detailL]
        """
        B, H, L, D = x.shape
        coeffs = []
        for b in range(B):
            for h in range(H):
                per_head = x[b, h]  # [L, D]
                ch = [
                    torch.tensor(c, dtype=torch.float32, device=x.device)
                    for c in zip(
                        *[
                            pywt.wavedec(
                                per_head[:, d].cpu().numpy(), self.wavelet, level=level
                            )
                            for d in range(D)
                        ]
                    )
                ]
                coeffs.append(ch)
        # Reshape: [B*H, levels+1, D]
        return coeffs

    def _apply_idwt(self, coeffs, length):
        """
        Reconstruct [B, H, L, D] from wavelet coefficients
        """
        BxH = len(coeffs)
        levels = len(coeffs[0])
        D = coeffs[0][0].shape[1]
        recon = []
        for c in coeffs:
            # List of [approx, d1, d2, ..., dL] → per dim
            series = []
            for d in range(D):
                per_channel = [c[j][:, d].cpu().numpy() for j in range(levels)]
                rec = pywt.waverec(per_channel, self.wavelet)[:length]
                series.append(torch.tensor(rec, dtype=torch.float32))
            recon.append(torch.stack(series, dim=1))  # [L, D]
        return torch.stack(recon).view(-1, length, D)

    def _weight_coeffs(self, coeffs, weights):
        """
        Multiply wavelet coeffs by weights: shape [H, levels, D]
        """
        weighted = []
        for i, c in enumerate(coeffs):
            h = i % self.n_heads
            w = weights[h]  # [levels, D]
            weighted.append([c[j] * w[j] for j in range(self.modes)])
        return weighted

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
        need_weights=False,
    ):
        if key is None:
            key = query
        if value is None:
            value = query

        B, L, _ = query.shape
        q, k, v = self.qkv_proj(query).chunk(3, dim=-1)

        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply DWT: [B*H, levels, D]
        q_dwt = self._apply_dwt(q, level=self.modes)
        v_dwt = self._apply_dwt(v, level=self.modes)

        # Weight in wavelet domain
        q_w = self._weight_coeffs(q_dwt, self.weight_q)
        v_w = self._weight_coeffs(v_dwt, self.weight_v)

        # Combine: q * v (element-wise attention in wavelet domain)
        out_coeffs = [
            [q_w[i][j] * v_w[i][j] for j in range(self.modes)] for i in range(len(q_w))
        ]

        # Inverse DWT
        out_recon = self._apply_idwt(out_coeffs, length=L)  # [B*H, L, D]
        out = (
            out_recon.view(B, self.n_heads, L, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        out = out.view(B, L, self.d_model)

        return self.dropout(self.out_proj(out)), None
