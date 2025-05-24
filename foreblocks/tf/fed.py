import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class FrequencyAttention(nn.Module):
    """
    Frequency-domain attention as proposed in FEDformer (Zhou et al. 2022).

    Fixed to handle cuFFT limitations with half precision:
    - Automatically pads sequences to power-of-2 lengths for half precision
    - Falls back to float32 for FFT operations when needed
    - Handles arbitrary sequence lengths robustly
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        modes: int = 64,
        activation: str = "tanh",
        force_fp32_fft: bool = False,  # Force float32 for FFT operations
    ):
        super().__init__()
        print("[Attention] Using frequency attention with cuFFT compatibility")
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modes = modes
        self.force_fp32_fft = force_fp32_fft
        self.dropout = nn.Dropout(dropout)

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable frequency mixing weights (complex-valued)
        self.freq_weight_real = nn.Parameter(
            torch.randn(n_heads, modes, self.head_dim, dtype=torch.float32) * 0.02
        )
        self.freq_weight_imag = nn.Parameter(
            torch.randn(n_heads, modes, self.head_dim, dtype=torch.float32) * 0.02
        )

        # Activation function for frequency domain
        if activation == "tanh":
            self.freq_activation = torch.tanh
        elif activation == "gelu":
            self.freq_activation = F.gelu
        elif activation == "relu":
            self.freq_activation = F.relu
        else:
            self.freq_activation = lambda x: x

    def _is_power_of_2(self, n):
        """Check if n is a power of 2"""
        return n > 0 and (n & (n - 1)) == 0

    def _next_power_of_2(self, n):
        """Find the next power of 2 >= n"""
        if n <= 1:
            return 1
        return 2 ** math.ceil(math.log2(n))

    def _safe_rfft(self, x, target_len=None):
        """
        Perform real FFT with cuFFT compatibility.
        Handles half precision limitations by padding to power-of-2 or using float32.
        """
        original_dtype = x.dtype
        B, H, L, D = x.shape

        # Determine if we need special handling for half precision
        needs_power_of_2 = (
            original_dtype in (torch.float16, torch.bfloat16)
            and x.is_cuda
            and not self._is_power_of_2(L)
        )

        if self.force_fp32_fft or needs_power_of_2:
            if needs_power_of_2:
                # Pad to next power of 2 for cuFFT compatibility
                padded_len = self._next_power_of_2(L)
                if padded_len != L:
                    padding = (0, 0, 0, padded_len - L)  # Pad along sequence dimension
                    x_padded = F.pad(x, padding, mode="constant", value=0)
                else:
                    x_padded = x
            else:
                x_padded = x

            # Convert to float32 for FFT, then back to original dtype
            x_fft = torch.fft.rfft(x_padded.float(), dim=2)

            # Truncate back to original frequency length if we padded
            if target_len is not None:
                freq_len = target_len // 2 + 1
                x_fft = x_fft[:, :, :freq_len, :]
            elif L != x_padded.size(2):
                freq_len = L // 2 + 1
                x_fft = x_fft[:, :, :freq_len, :]

            return x_fft
        else:
            # Direct FFT for float32 or when padding isn't needed
            return torch.fft.rfft(x, dim=2)

    def _safe_irfft(self, x_fft, target_len):
        """
        Perform inverse real FFT with proper length handling.
        """
        original_dtype = x_fft.dtype

        if self.force_fp32_fft or x_fft.is_cuda:
            # Use float32 for IFFT operations
            x_fft_float = (
                x_fft if x_fft.dtype == torch.complex64 else x_fft.to(torch.complex64)
            )
            result = torch.fft.irfft(x_fft_float, n=target_len, dim=2)
            return result[:, :, :target_len, :]  # Ensure exact length
        else:
            return torch.fft.irfft(x_fft, n=target_len, dim=2)[:, :, :target_len, :]

    def _get_freq_modes(self, seq_len: int) -> int:
        """Get the number of frequency modes to use based on sequence length"""
        return min(self.modes, seq_len // 2 + 1)

    def _frequency_mixing(
        self, q_fft: torch.Tensor, v_fft: torch.Tensor
    ) -> torch.Tensor:
        """
        Core frequency domain mixing operation following FEDformer
        """
        B, H, L_freq, D = q_fft.shape
        modes = self._get_freq_modes(L_freq)

        # Create complex weights from real and imaginary parts
        freq_weight = torch.complex(
            self.freq_weight_real[:, :modes, :].to(q_fft.device),
            self.freq_weight_imag[:, :modes, :].to(q_fft.device),
        )

        # Take only the modes we want to process
        q_modes = q_fft[:, :, :modes, :]  # [B, H, modes, D]
        v_modes = v_fft[:, :, :modes, :]  # [B, H, modes, D]

        # Efficient mixing using einsum: Q * W * V
        mixed_modes = torch.einsum("bhmd,hmd,bhmd->bhmd", q_modes, freq_weight, v_modes)

        # Reconstruct full frequency tensor
        mixed_fft = torch.zeros_like(q_fft)
        mixed_fft[:, :, :modes, :] = mixed_modes

        # Apply activation in frequency domain (on real part to maintain structure)
        mixed_fft.real = self.freq_activation(mixed_fft.real)

        return mixed_fft

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

        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = key

        B, L_q, _ = query.shape
        _, L_k, _ = key.shape
        _, L_v, _ = value.shape

        assert L_k == L_v, "Key and value sequence lengths must match"

        # Project Q, K, V
        q = self.q_proj(query)  # [B, L_q, d_model]
        k = self.k_proj(key)  # [B, L_k, d_model]
        v = self.v_proj(value)  # [B, L_v, d_model]

        # Reshape for multi-head attention
        q = q.view(B, L_q, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, L_q, D]
        k = k.view(B, L_k, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, L_k, D]
        v = v.view(B, L_v, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, L_v, D]

        # For frequency attention, we typically use Q and V (not K explicitly)
        # Handle different sequence lengths by using the longer sequence for FFT
        max_len = max(L_q, L_v)

        # Pad shorter sequences if needed
        if L_q < max_len:
            q_padded = F.pad(q, (0, 0, 0, max_len - L_q), mode="constant", value=0)
        else:
            q_padded = q

        if L_v < max_len:
            v_padded = F.pad(v, (0, 0, 0, max_len - L_v), mode="constant", value=0)
        else:
            v_padded = v

        # Convert to frequency domain using safe FFT
        q_fft = self._safe_rfft(q_padded, max_len)  # [B, H, max_len//2+1, D]
        v_fft = self._safe_rfft(v_padded, max_len)  # [B, H, max_len//2+1, D]

        # Apply frequency domain mixing
        out_fft = self._frequency_mixing(q_fft, v_fft)

        # Convert back to time domain
        out_time = self._safe_irfft(out_fft, L_q)  # [B, H, L_q, D]

        # Reshape back to [B, L_q, d_model]
        out = out_time.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

        # Final output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        # Return None for attention weights since they don't exist in frequency attention
        return out, None


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

        # Apply DWT
        if self.has_pywt:
            # Use proper DWT if available (this would need more complex implementation)
            q_dwt = self._simple_dwt(q)
            v_dwt = self._simple_dwt(v)
        else:
            q_dwt = self._simple_dwt(q)
            v_dwt = self._simple_dwt(v)

        # Apply wavelet domain mixing (simplified)
        modes = min(self.modes, q_dwt.size(2))
        q_modes = q_dwt[:, :, :modes, :]
        v_modes = v_dwt[:, :, :modes, :]

        # Element-wise mixing with learnable weights
        mixed = torch.einsum(
            "bhmd,hmd->bhmd", q_modes * v_modes, self.wavelet_weight[:, :modes, :]
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
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
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
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .to(values.device)
        )
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def _safe_fft_operations(self, queries, keys):
        """Safe FFT operations handling cuFFT half-precision limitations"""
        original_dtype = queries.dtype
        device = queries.device

        # Check if we need to handle half precision on CUDA
        needs_conversion = (
            original_dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
        )

        if needs_conversion:
            # Convert to float32 for FFT operations
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
        corr = torch.fft.irfft(res, dim=-1)

        # Convert back to original dtype if needed
        if needs_conversion:
            corr = corr.to(original_dtype)

        return corr

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies with safe FFT
        corr = self._safe_fft_operations(queries, keys)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
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

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
