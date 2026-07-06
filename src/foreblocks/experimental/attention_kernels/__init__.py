"""foreblocks.experimental.attention_kernels.

Package initializer that exposes the public symbols for this namespace.
It belongs to the experimental attention kernel implementations and benchmarks area of Foreblocks.

"""

from foreblocks.experimental.attention_kernels.src import (
    FlashAttnRMSNorm,
    FlashDecodeModule,
    flash_attn_backward_backend,
    flash_attn_decode,
    flash_attn_dropout_func,
    flash_attn_forward,
    flash_attn_forward_backend,
    flash_attn_func,
    flash_attn_uses_cuda_backward,
)

__all__ = [
    "flash_attn_backward_backend",
    "flash_attn_forward",
    "flash_attn_forward_backend",
    "flash_attn_func",
    "flash_attn_uses_cuda_backward",
    "flash_attn_dropout_func",
    "flash_attn_decode",
    "FlashAttnRMSNorm",
    "FlashDecodeModule",
]
