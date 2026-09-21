"""Main public entry point: ``chunked_ssd_forward`` auto-selects Triton vs.
PyTorch-modular, and its autograd ``Function`` wrapper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .modular import _chunked_ssd_backward_modular, _chunked_ssd_forward_modular
from .torch_backward import _chunked_ssd_backward_torch
from foreblocks.kernels.mamba.ssd import (
    CHUNKED_SSD_TRITON_AVAILABLE,
    chunked_ssd_forward_triton,
    chunked_ssd_forward_triton_parallel,
    chunked_ssd_forward_triton_tiled,
)

# Backward selection (when the Triton forward ran): the fused Triton backward is
# memory-flat and on par with the vectorised backward at long T, but ~15-20%
# slower at short T (per-chunk launch + boundary recompute). The vectorised
# backward is faster at short T but its memory grows with T (materialises the
# [B, nc, C, C, H] intra-chunk matrices) and eventually OOMs. So we pick by
# sequence length: vectorised below the threshold, Triton at/above it.
SSD_TRITON_BACKWARD_MIN_SEQLEN = 0  # vectorized torch backward is always faster; Triton bwd is O(chunks) sequential (#1 bottleneck per analysis)  # Always use torch backward (triton bwd is O(chunks) sequential)


class _ChunkedSSDFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u,
        dt,
        A,
        B,
        C,
        D,
        chunk_size: int,
        use_triton: bool,
        use_parallel: bool | str,
        adt=None,
        seq_idx=None,
        initial_states=None,
        dfinal_states=None,
    ):
        ctx.chunk_size = chunk_size
        ctx.seq_idx = seq_idx
        if use_triton:
            if use_parallel == "direct":
                fwd = chunked_ssd_forward_triton_parallel
            elif use_parallel == "tiled" or use_parallel is True:
                fwd = chunked_ssd_forward_triton_tiled
            else:
                fwd = chunked_ssd_forward_triton
            y = fwd(u, dt, A, B, C, D, chunk_size=chunk_size, adt=adt)
            ctx.use_triton = use_triton
            ctx.save_for_backward(u, dt, A, B, C, D, adt)
            return y
        else:
            # Modular forward: saves intermediates for efficient backward
            y, intermediates = _chunked_ssd_forward_modular(
                u,
                dt,
                A,
                B,
                C,
                D,
                chunk_size=chunk_size,
                adt=adt,
                seq_idx=seq_idx,
                initial_states=initial_states,
            )
            ctx.use_triton = False
            ctx.save_for_backward(u, dt, A, B, C, D, adt)
            ctx._intermediates = intermediates
            # Store dfinal_states for backward gradient
            if dfinal_states is not None:
                intermediates["dfinal_states"] = dfinal_states
            return y

    @staticmethod
    def backward(ctx, grad_y):
        u, dt, A, B, C, D, adt = ctx.saved_tensors
        if hasattr(ctx, "_intermediates") and ctx._intermediates is not None:
            # Modular backward with intermediate reuse (Mamba2-style)
            grads = _chunked_ssd_backward_modular(
                grad_y,
                ctx._intermediates,
                A,
                D,
                adt,
                needs_input_grad=ctx.needs_input_grad[:6],
                needs_adt_grad=ctx.needs_input_grad[12],
            )
            du, ddt, dA, dB, dC, dD, dadt = grads
            return (du, ddt, dA, dB, dC, dD, None, None, None, dadt, None, None, None)
        # always use vectorized torch backward — Triton bwd is O(chunks) sequential
        bwd = _chunked_ssd_backward_torch
        grads = bwd(
            grad_y,
            u,
            dt,
            A,
            B,
            C,
            D,
            chunk_size=ctx.chunk_size,
            needs_input_grad=ctx.needs_input_grad[:6],
            adt=adt,
            needs_adt_grad=ctx.needs_input_grad[12],
        )
        du, ddt, dA, dB, dC, dD, dadt = grads
        return (du, ddt, dA, dB, dC, dD, None, None, None, dadt, None, None, None)


def chunked_ssd_forward(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    use_triton: bool = False,
    adt: torch.Tensor | None = None,
    trap: torch.Tensor | None = None,
    parallel: bool | str = "tiled",
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    dfinal_states: torch.Tensor | None = None,
) -> torch.Tensor:
    can_use_triton = (
        use_triton
        and CHUNKED_SSD_TRITON_AVAILABLE
        and u.is_cuda
        and dt.is_cuda
        and A.is_cuda
        and B.is_cuda
        and C.is_cuda
        and D.is_cuda
        and B.shape[-1] <= 128
    )
    if trap is not None:
        # Trapezoidal discretisation (Mamba3)
        if adt is None:
            raise ValueError("trapezoidal (trap) discretisation requires adt (Mamba3)")
        T = u.shape[1]
        trap_h = trap  # [B, T, H], broadcasts against dt [B, T, H]
        y_cur = _ChunkedSSDFn.apply(
            u,
            dt * trap_h,
            A,
            B,
            C,
            D,
            chunk_size,
            can_use_triton,
            parallel,
            adt,
            seq_idx,
            initial_states,
            dfinal_states,
        )
        B_prev = F.pad(B, (0, 0, 0, 0, 1, 0))[:, :T]
        u_prev = F.pad(u, (0, 0, 0, 0, 1, 0))[:, :T]
        y_prev = _ChunkedSSDFn.apply(
            u_prev,
            dt * (1.0 - trap_h),
            A,
            B_prev,
            C,
            torch.zeros_like(D),
            chunk_size,
            can_use_triton,
            parallel,
            adt,
            seq_idx,
            initial_states,
            dfinal_states,
        )
        return y_cur + y_prev
    return _ChunkedSSDFn.apply(
        u,
        dt,
        A,
        B,
        C,
        D,
        chunk_size,
        can_use_triton,
        parallel,
        adt,
        seq_idx,
        initial_states,
        dfinal_states,
    )


