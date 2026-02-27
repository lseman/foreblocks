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

        # Check if PyWavelets is available
        try:
            import pywt

            self.pywt = pywt
            self.has_pywt = True
        except ImportError:
            print(
                "Warning: PyWavelets not available. DWT attention will use simple approximation."
            )
            self.has_pywt = False

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable wavelet mixing weights
        self.wavelet_weight = nn.Parameter(
            torch.randn(n_heads, modes, self.head_dim) * 0.02
        )

    def _simple_dwt(self, x):
        """Simple DWT approximation using average pooling and differences"""
        # Approximate DWT using pooling operations
        # This is a simplified version when PyWavelets is not available
        B, H, L, D = x.shape

        if L % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode="reflect")
            L += 1

        # Approximation coefficients (low-pass)
        approx = (x[:, :, ::2, :] + x[:, :, 1::2, :]) / 2

        # Detail coefficients (high-pass)
        detail = (x[:, :, ::2, :] - x[:, :, 1::2, :]) / 2

        return torch.cat([approx, detail], dim=2)

    def _simple_idwt(self, coeffs, target_len):
        """Simple inverse DWT approximation"""
        B, H, L, D = coeffs.shape
        half_L = L // 2

        approx = coeffs[:, :, :half_L, :]
        detail = coeffs[:, :, half_L:, :]

        # Reconstruct
        even = approx + detail
        odd = approx - detail

        # Interleave
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

        # Project Q, K, V
        q = self.q_proj(query).view(B, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        v = (
            self.v_proj(value)
            .view(B, value.size(1), self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Project K and reshape
        k = (
            self.k_proj(key)
            .view(B, key.size(1), self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply DWT
        q_dwt = self._simple_dwt(q)
        k_dwt = self._simple_dwt(k)
        v_dwt = self._simple_dwt(v)

        # Apply wavelet domain mixing (simplified)
        modes = min(self.modes, q_dwt.size(2))
        q_modes = q_dwt[:, :, :modes, :]
        k_modes = k_dwt[:, :, :modes, :]
        v_modes = v_dwt[:, :, :modes, :]

        # Element-wise mixing with learnable weights
        # Q acts as a query filter against K, then weighted aggregation over V
        scores = q_modes * k_modes  # [B, H, modes, D] — cross-correlation proxy
        attn = torch.softmax(scores / math.sqrt(self.head_dim), dim=2)
        mixed = torch.einsum(
            "bhmd,hmd->bhmd", attn * v_modes, self.wavelet_weight[:, :modes, :]
        )

        # Reconstruct
        out_dwt = torch.zeros_like(q_dwt)
        out_dwt[:, :, :modes, :] = mixed

        # Inverse DWT
        out_time = self._simple_idwt(out_dwt, L_q)

        # Reshape and project
        out = out_time.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out, None


class AutoCorrelation(nn.Module):
    """
    Fixed AutoCorrelation Mechanism - correcting critical bugs in the current implementation.
    """

    def __init__(
        self,
        mask_flag=True,
        factor=1,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention

    def time_delay_agg(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Vectorized time delay aggregation using batched gather.
        values: [B, H, D, L]
        corr:   [B, L, H, D]
        """
        B, H, D, L = values.shape
        device = values.device
        top_k = max(1, min(int(self.factor * math.log(L)), L))

        # Average correlation over head and feature dimensions
        mean_corr = corr.mean(dim=2).mean(dim=2)  # [B, L]
        global_mean = mean_corr.mean(dim=0)  # [L]
        topk = torch.topk(global_mean, top_k, dim=0).indices  # [top_k]

        # [B, top_k]
        weights = mean_corr[:, topk]
        soft_weights = torch.softmax(weights, dim=-1)  # [B, top_k]

        # Create index tensor for gathering
        base = torch.arange(L, device=device)  # [L]
        shifts = topk.view(-1, 1)  # [top_k, 1]
        indices = (base[None, :] - shifts) % L  # [top_k, L]

        # Expand to shape [B, H, D, top_k, L] for batched gather
        values_exp = values.unsqueeze(3).expand(
            B, H, D, top_k, L
        )  # [B, H, D, top_k, L]
        gather_idx = indices.view(1, 1, 1, top_k, L).expand(
            B, H, D, top_k, L
        )  # [B, H, D, top_k, L]

        # Gather values with shifted indices
        rolled = torch.gather(
            values_exp, dim=-1, index=gather_idx
        )  # [B, H, D, top_k, L]

        # Apply weights
        soft_weights = soft_weights.transpose(0, 1).view(
            top_k, B, 1, 1, 1
        )  # [top_k, B, 1, 1, 1]
        weighted = rolled.permute(3, 0, 1, 2, 4) * soft_weights  # [top_k, B, H, D, L]

        return weighted.sum(dim=0)  # [B, H, D, L]

    def _safe_fft_operations(self, queries, keys):
        """
        Fixed FFT operations with proper length specification and tensor handling.
        """
        B, L, H, E = queries.shape
        original_dtype = queries.dtype
        device = queries.device

        # Handle half precision
        needs_conversion = (
            original_dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
        )

        if needs_conversion:
            queries = queries.float()
            keys = keys.float()

        # Permute for FFT: [B, L, H, E] -> [B, H, E, L]
        q_perm = queries.permute(0, 2, 3, 1).contiguous()
        k_perm = keys.permute(0, 2, 3, 1).contiguous()

        # FFT operations
        q_fft = torch.fft.rfft(q_perm, dim=-1)
        k_fft = torch.fft.rfft(k_perm, dim=-1)

        # Cross-correlation in frequency domain
        res = q_fft * torch.conj(k_fft)

        # Fixed: Specify the length parameter for irfft
        corr = torch.fft.irfft(res, n=L, dim=-1)  # [B, H, E, L]

        # Convert back to original dtype
        if needs_conversion:
            corr = corr.to(original_dtype)

        # Fixed: Convert to expected shape [B, L, H, E]
        corr = corr.permute(0, 3, 1, 2).contiguous()

        return corr

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # Handle sequence length mismatch
        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :, :])
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Period-based dependencies discovery
        corr = self._safe_fft_operations(queries, keys)  # [B, L, H, E]

        # Time delay aggregation
        # Convert values to [B, H, D, L] for aggregation
        values_perm = values.permute(0, 2, 3, 1).contiguous()

        V = self.time_delay_agg(values_perm, corr)

        # Convert back to [B, L, H, D]
        V = V.permute(0, 3, 1, 2).contiguous()

        if self.output_attention:
            return V, corr
        else:
            return V, None


class AutoCorrelationLayer(nn.Module):
    """AutoCorrelation layer with projections."""

    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project and reshape
        queries = self.query_projection(queries).view(B, L, H, self.d_keys)
        keys = self.key_projection(keys).view(B, S, H, self.d_keys)
        values = self.value_projection(values).view(B, S, H, self.d_values)

        # Apply autocorrelation
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)

        # Reshape and project
        out = out.view(B, L, H * self.d_values)
        return self.out_projection(out), attn
