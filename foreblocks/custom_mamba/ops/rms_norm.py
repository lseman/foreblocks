from __future__ import annotations

import torch

try:
    from foreblocks.transformer.kernels.rms_norm import (
        RMSNormTritonFunction,
        TRITON_AVAILABLE as RMS_NORM_TRITON_AVAILABLE,
        _should_use_triton,
    )
except Exception:
    RMSNormTritonFunction = None  # type: ignore[assignment]
    RMS_NORM_TRITON_AVAILABLE = False

    def _should_use_triton(*_, **__) -> bool:  # type: ignore[misc]
        return False


def rms_norm_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms * weight


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm wrapper shared with transformer Triton kernels.

    ``custom_mamba`` historically carried its own RMSNorm Triton kernel whose
    backward used per-row atomics for ``grad_weight``. Reusing the transformer
    kernel keeps the public Mamba API intact while picking up the block-reduced
    backward path.
    """
    if (
        RMSNormTritonFunction is not None
        and RMS_NORM_TRITON_AVAILABLE
        and x.is_cuda
        and weight.is_cuda
        and _should_use_triton(x)
    ):
        return RMSNormTritonFunction.apply(x, weight, eps)
    return rms_norm_fallback(x, weight, eps)
