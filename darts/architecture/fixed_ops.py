import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import RMSNorm


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



class FixedOp(nn.Module):
    """Simple wrapper for fixed operations"""

    def __init__(self, selected_op: nn.Module):
        super().__init__()
        self.op = selected_op

    def forward(self, x):
        return self.op(x)


__all__ = [
    "RMSNorm",
    "ChannelRMSNorm",
    "CausalConv1d",
    "IdentityOp",
    "TimeConvOp",
    "ResidualMLPOp",
    "TCNOp",
    "FourierOp",
    "WaveletOp",
    "ConvMixerOp",
    "GRNOp",
    "MultiScaleConvOp",
    "PyramidConvOp",
    "PatchEmbedOp",
    "InvertedAttentionOp",
    "MLPMixerOp",
    "DLinearOp",
    "NBeatsOp",
    "TimesNetOp",
    "FixedOp",
]
