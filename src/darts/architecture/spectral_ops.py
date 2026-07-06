import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import ChannelRMSNorm, RMSNorm


try:
    from foreblocks.ops.norms_triton import (
        TRITON_AVAILABLE,
        RMSNormTritonFunction,
        _should_use_triton,
    )
except Exception:  # pragma: no cover - foreblocks namespace may exclude transformer
    TRITON_AVAILABLE = False
    RMSNormTritonFunction = None

    def _should_use_triton(x, min_numel: int = 2048) -> bool:
        return False



class FourierOp(nn.Module):
    """Fourier operation with learnable frequency weighting"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        num_frequencies: int | None = None,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.num_frequencies = (
            min(seq_length // 2 + 1, 32) if num_frequencies is None else num_frequencies
        )

        self.low_freq_proj = nn.Sequential(
            nn.Linear(input_dim * 2, latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )
        self.high_freq_proj = nn.Sequential(
            nn.Linear(input_dim * 2, latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )

        self.low_cutoff_logit = nn.Parameter(torch.tensor(0.0))
        self.high_cutoff_logit = nn.Parameter(torch.tensor(0.0))
        self.filter_sharpness = nn.Parameter(torch.tensor(6.0))
        self.post_mix_logit = nn.Parameter(torch.tensor(0.0))

        self.freq_weights = nn.Parameter(torch.randn(self.num_frequencies) * 0.02)

        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False), nn.Sigmoid()
        )

        self.output_proj = nn.Linear(input_dim + latent_dim, latent_dim, bias=False)
        self.norm = RMSNorm(latent_dim)

    def _build_low_high_masks(
        self, n_freq: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.linspace(0.0, 1.0, steps=n_freq, device=device, dtype=dtype)
        sharpness = torch.clamp(self.filter_sharpness, min=1.0).to(dtype=dtype)

        low_cut = torch.sigmoid(self.low_cutoff_logit).to(dtype=dtype)
        high_cut = torch.sigmoid(self.high_cutoff_logit).to(dtype=dtype)

        low_mask = torch.sigmoid((low_cut - pos) * sharpness)
        high_mask = torch.sigmoid((pos - high_cut) * sharpness)
        return low_mask.reshape(1, -1, 1), high_mask.reshape(1, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        # Build a fixed-length FFT reference window for stable frequency bins.
        if L < self.seq_length:
            x_padded = F.pad(x, (0, 0, 0, self.seq_length - L))
        else:
            x_padded = x[:, : self.seq_length]

        # cuFFT only supports power-of-two signal lengths for fp16/bf16 rFFT.
        # Keep AMP enabled around the op, but run the FFT itself in fp32.
        if x_padded.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                fft_input = x_padded if x_padded.dtype == torch.float64 else x_padded.float()
                x_fft = torch.fft.rfft(fft_input, dim=1, norm="ortho")
        else:
            x_fft = torch.fft.rfft(x_padded, dim=1, norm="ortho")
        x_fft = x_fft[:, : self.num_frequencies]
        n_freq = x_fft.size(1)
        low_mask, high_mask = self._build_low_high_masks(
            n_freq, device=x_fft.device, dtype=x_fft.real.dtype
        )
        low_fft = x_fft * low_mask
        high_fft = x_fft * high_mask

        # Frequency tokenization + learnable low/high filtering before projection.
        low_feat = torch.cat([low_fft.real, low_fft.imag], dim=-1)
        high_feat = torch.cat([high_fft.real, high_fft.imag], dim=-1)
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            low_feat = low_feat.to(dtype=x.dtype)
            high_feat = high_feat.to(dtype=x.dtype)

        low_feat = self.low_freq_proj(low_feat)
        high_feat = self.high_freq_proj(high_feat)

        post_mix = torch.sigmoid(self.post_mix_logit)
        freq_feat = post_mix * low_feat + (1.0 - post_mix) * high_feat

        # Global spectral summary (weighted) + local spectral context (interpolated).
        weights = F.softmax(self.freq_weights[:n_freq], dim=0).reshape(1, -1, 1)
        global_feat = (freq_feat * weights).sum(dim=1, keepdim=True).expand(-1, L, -1)

        local_feat = F.interpolate(
            freq_feat.transpose(1, 2),
            size=L,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        spectral_context = 0.5 * global_feat + 0.5 * local_feat
        gated = self.gate(spectral_context)
        if spectral_context.dtype != x.dtype:
            spectral_context = spectral_context.to(dtype=x.dtype)
            gated = gated.to(dtype=x.dtype)

        combined = torch.cat([x, gated * spectral_context], dim=-1)
        return self.norm(self.output_proj(combined))


class WaveletOp(nn.Module):
    """Multi-scale wavelet-style operation using dilated convolutions"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_scales: int = 3,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        candidate_dilations = dilations or [1, 2, 4, 8]
        self.dilations = candidate_dilations[
            : max(1, min(num_scales, len(candidate_dilations)))
        ]
        self.num_scales = len(self.dilations)

        self.scale_layers = nn.ModuleList()
        for dilation in self.dilations:
            layer = nn.Sequential(
                nn.Conv1d(
                    input_dim,
                    input_dim,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=input_dim,
                    bias=False,
                ),
                nn.Conv1d(input_dim, input_dim, kernel_size=1, bias=False),
                ChannelRMSNorm(input_dim),
                nn.GELU(),
                nn.Dropout(0.05),
            )
            self.scale_layers.append(layer)

        self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))

        self.fusion = nn.Conv1d(
            input_dim * self.num_scales, latent_dim, kernel_size=1, bias=False
        )
        self.norm = RMSNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.shape
        x_t = x.transpose(1, 2)

        scale_weights = F.softmax(self.scale_logits, dim=0)
        features = []
        for idx, layer in enumerate(self.scale_layers):
            feat = layer(x_t)
            if feat.shape[-1] != L:
                feat = F.adaptive_avg_pool1d(feat, L)
            features.append(feat * scale_weights[idx])

        fused = torch.cat(features, dim=1)
        output = self.fusion(fused).transpose(1, 2)
        return self.norm(output)


