"""foreblocks.ops.mamba.ssd.

Chunked state-space diffusion (SSD) forward and backward — Mamba2 scan operator.

Implements the Mamba2-style structured state-space forward with chunked
parallel scan, supporting both diagonal-A (Mamba2) and adt-based (Mamba3)
discretisation. Provides vectorised PyTorch, Triton parallel, Triton tiled,
and Triton direct (recurrent-in-chunk) paths. Includes trapezoidal
discretisation support (Mamba3) and variable-length sequence handling via
seq_idx masking.

Split into submodules by algorithm stage:
- segment_sum.py: L-matrix cumulative-sum helpers
- torch_backward.py: pure-PyTorch vectorized forward/backward (production backward)
- modular.py: 3-stage modular kernels + their autograd-wrapped public entry
- triton_kernels.py: raw Triton kernels + their launchers
- api.py: main public entry point (chunked_ssd_forward), auto-selects Triton/PyTorch
- reference.py: simplest reference implementation, for correctness testing

Core API:
- chunked_ssd_forward: main entry, auto-selects Triton/pytorch path
- chunked_ssd_forward_triton: parallel Triton chunk kernel
- chunked_ssd_forward_triton_parallel: token-parallel Triton kernel
- chunked_ssd_forward_triton_tiled: tiled token-parallel kernel
- chunked_ssd_backward_triton: reverse-time Triton backward
- chunked_ssd_forward_modular: modular forward with intermediate saving
- segment_sum: lower-triangular cumulative sum (L-matrix)

"""

from __future__ import annotations

from .api import chunked_ssd_forward
from .modular import chunked_ssd_forward_modular
from .reference import chunked_ssd_backward_reference, chunked_ssd_forward_reference
from .segment_sum import segment_sum
from .torch_backward import _chunked_ssd_backward_torch, _chunked_ssd_forward_torch
from .triton_kernels import (
    CHUNKED_SSD_TRITON_AVAILABLE,
    chunked_ssd_backward_triton,
    chunked_ssd_forward_triton,
    chunked_ssd_forward_triton_parallel,
    chunked_ssd_forward_triton_tiled,
)

__all__ = [
    "CHUNKED_SSD_TRITON_AVAILABLE",
    "chunked_ssd_backward_reference",
    "chunked_ssd_forward",
    "chunked_ssd_forward_reference",
    "chunked_ssd_forward_triton",
    "chunked_ssd_forward_triton_parallel",
    "chunked_ssd_forward_triton_tiled",
    "segment_sum",
]
