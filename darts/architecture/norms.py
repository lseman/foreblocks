import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from foreblocks.transformer.norms.triton_backend import (
        TRITON_AVAILABLE,
        RMSNormTritonFunction,
        _should_use_triton,
    )
except Exception:  # pragma: no cover - foreblocks namespace may exclude transformer
    TRITON_AVAILABLE = False
    RMSNormTritonFunction = None

    def _should_use_triton(x, min_numel: int = 2048) -> bool:
        return False


class RMSNorm(nn.Module):
    """Simple RMS normalization with Triton acceleration when available."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            _should_use_triton(x, min_numel=2048)
            and TRITON_AVAILABLE
            and RMSNormTritonFunction is not None
        ):
            return RMSNormTritonFunction.apply(x, self.scale, self.eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.scale


class ChannelRMSNorm(nn.Module):
    """RMSNorm over channel dimension for [B, C, L] tensors."""

    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.norm = RMSNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


