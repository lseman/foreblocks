"""foreblocks.ops.graph.message_passing.

Triton kernels for graph message passing operations.

Provides Triton-accelerated scatter-add and gather operations for message
passing in graph neural networks. Includes fused gather-scatter for
edge-indexed message passing with support for add, mean, and max aggregation.

Core API:
- triton_scatter_add: scatter-add with edge indices
- triton_scatter_mean: scatter-mean with edge indices
- triton_scatter_max: scatter-max with edge indices
- triton_message_passing: fused gather-scatter message passing
- HAS_TRITON: Triton availability flag

"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit  # type: ignore
    def _scatter_add_kernel(
        src_ptr,
        index_ptr,
        out_ptr,
        B,
        T,
        N,
        F,
        num_edges,
        BLOCK_F: tl.constexpr,  # type: ignore
    ):
        """Scatter-add: out[b, t, dst, :] += src[b, t, edge, :] indexed by index_ptr."""
        pid = tl.program_id(0)
        edge_idx = pid

        # Load edge destination
        dst = tl.load(index_ptr + edge_idx).to(tl.int32)

        # Load src features
        f_offsets = tl.arange(0, BLOCK_F)
        f_mask = f_offsets < F

        for b in range(B):
            for t in range(T):
                src_row = b * T * N * F + t * N * F + edge_idx * F
                out_row = b * T * N * F + t * N * F + dst * F

                src_vals = tl.load(
                    src_ptr + src_row + f_offsets, mask=f_mask, other=0.0
                )
                out_vals = tl.load(
                    out_ptr + out_row + f_offsets, mask=f_mask, other=0.0
                )
                tl.store(
                    out_ptr + out_row + f_offsets, src_vals + out_vals, mask=f_mask
                )

    @triton.jit  # type: ignore
    def _gather_scatter_add_kernel(
        x_ptr,
        edge_index_src_ptr,
        edge_index_dst_ptr,
        edge_weight_ptr,
        out_ptr,
        B,
        T,
        N,
        F,
        num_edges,
        BLOCK_F: tl.constexpr,  # type: ignore
    ):
        """Fused gather-scatter-add for message passing.

        msg[b, t, edge, :] = x[b, t, src[edge], :] * edge_weight[edge]
        out[b, t, dst[edge], :] += msg[b, t, edge, :]
        """
        pid = tl.program_id(0)
        edge_idx = pid

        # Load edge indices
        src = tl.load(edge_index_src_ptr + edge_idx).to(tl.int32)
        dst = tl.load(edge_index_dst_ptr + edge_idx).to(tl.int32)

        # Load edge weight
        w = 1.0
        if edge_weight_ptr is not None:
            w = tl.load(edge_weight_ptr + edge_idx).to(tl.float32)

        # Gather and scatter
        f_offsets = tl.arange(0, BLOCK_F)
        f_mask = f_offsets < F

        for b in range(B):
            for t in range(T):
                # Source feature row
                src_row = b * T * N * F + t * N * F + src * F

                # Output row
                out_row = b * T * N * F + t * N * F + dst * F

                # Load src features
                src_vals = tl.load(x_ptr + src_row + f_offsets, mask=f_mask, other=0.0)

                # Apply edge weight
                msg_vals = src_vals * w

                # Load current output and add
                out_vals = tl.load(
                    out_ptr + out_row + f_offsets, mask=f_mask, other=0.0
                )
                tl.store(
                    out_ptr + out_row + f_offsets, msg_vals + out_vals, mask=f_mask
                )

    @triton.jit  # type: ignore
    def _scatter_mean_kernel(
        src_ptr,
        index_ptr,
        out_ptr,
        count_ptr,
        B,
        T,
        N,
        F,
        num_edges,
        BLOCK_F: tl.constexpr,  # type: ignore
    ):
        """Scatter-mean: computes sum and counts for mean aggregation."""
        pid = tl.program_id(0)
        edge_idx = pid

        # Load edge destination
        dst = tl.load(index_ptr + edge_idx).to(tl.int32)

        # Load src features
        f_offsets = tl.arange(0, BLOCK_F)
        f_mask = f_offsets < F

        for b in range(B):
            for t in range(T):
                src_row = b * T * N * F + t * N * F + edge_idx * F
                out_row = b * T * N * F + t * N * F + dst * F

                src_vals = tl.load(
                    src_ptr + src_row + f_offsets, mask=f_mask, other=0.0
                )
                out_vals = tl.load(
                    out_ptr + out_row + f_offsets, mask=f_mask, other=0.0
                )
                tl.store(
                    out_ptr + out_row + f_offsets, src_vals + out_vals, mask=f_mask
                )

                # Update count
                count_row = b * T * N + t * N + dst
                count_val = tl.load(count_ptr + count_row) + 1.0
                tl.store(count_ptr + count_row, count_val)


def triton_scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    batch_size: int,
    steps: int,
    F: int,
) -> torch.Tensor:
    """Triton scatter-add: out[b, t, dst, :] += src[b, t, edge, :]."""
    if not (HAS_TRITON and src.is_cuda and index.is_cuda):
        return _scatter_add_fallback(src, index, dim_size)

    B, T, E, F = src.shape
    assert index.shape == (E,)

    out = torch.zeros(B, T, dim_size, F, dtype=src.dtype, device=src.device)

    BLOCK_F = triton.next_power_of_2(F)
    BLOCK_F = min(BLOCK_F, 256)

    grid = (E,)
    _scatter_add_kernel[grid](
        src,
        index,
        out,
        B,
        T,
        dim_size,
        F,
        E,
        BLOCK_F=BLOCK_F,
    )
    return out


def triton_scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    batch_size: int,
    steps: int,
    F: int,
) -> torch.Tensor:
    """Triton scatter-mean: out[b, t, dst, :] += src[b, t, edge, :]."""
    if not (HAS_TRITON and src.is_cuda and index.is_cuda):
        return _scatter_mean_fallback(src, index, dim_size)

    B, T, E, F = src.shape
    assert index.shape == (E,)

    out = torch.zeros(B, T, dim_size, F, dtype=src.dtype, device=src.device)
    count = torch.zeros(B, T, dim_size, dtype=torch.float32, device=src.device)

    BLOCK_F = triton.next_power_of_2(F)
    BLOCK_F = min(BLOCK_F, 256)

    grid = (E,)
    _scatter_mean_kernel[grid](
        src,
        index,
        out,
        count,
        B,
        T,
        dim_size,
        F,
        E,
        BLOCK_F=BLOCK_F,
    )

    # Divide by counts
    count = count.clamp_min_(1.0)
    return out / count.view(B, T, dim_size, 1)


def triton_message_passing(
    x: torch.Tensor,
    edge_index_src: torch.Tensor,
    edge_index_dst: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused gather-scatter-add message passing.

    msg[b, t, edge, :] = x[b, t, src[edge], :] * edge_weight[edge]
    out[b, t, dst[edge], :] += msg[b, t, edge, :]
    """
    if not (
        HAS_TRITON and x.is_cuda and edge_index_src.is_cuda and edge_index_dst.is_cuda
    ):
        return _message_passing_fallback(x, edge_index_src, edge_index_dst, edge_weight)

    B, T, N, F = x.shape
    num_edges = edge_index_src.numel()

    out = torch.zeros(B, T, N, F, dtype=x.dtype, device=x.device)

    BLOCK_F = triton.next_power_of_2(F)
    BLOCK_F = min(BLOCK_F, 256)

    grid = (num_edges,)
    _gather_scatter_add_kernel[grid](
        x,
        edge_index_src,
        edge_index_dst,
        edge_weight,
        out,
        B,
        T,
        N,
        F,
        num_edges,
        BLOCK_F=BLOCK_F,
    )
    return out


def _scatter_add_fallback(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Fallback scatter-add using PyTorch scatter_add."""
    B, T, E, F = src.shape
    out = torch.zeros(B, T, dim_size, F, dtype=src.dtype, device=src.device)
    expanded_index = index.view(1, 1, -1, 1).expand_as(src)
    out.scatter_add_(2, expanded_index, src)
    return out


def _scatter_mean_fallback(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Fallback scatter-mean using PyTorch scatter_add."""
    B, T, E, F = src.shape
    out = torch.zeros(B, T, dim_size, F, dtype=src.dtype, device=src.device)
    expanded_index = index.view(1, 1, -1, 1).expand_as(src)
    out.scatter_add_(2, expanded_index, src)

    ones = torch.ones(B, T, E, 1, device=src.device, dtype=src.dtype)
    counts = torch.zeros(B, T, dim_size, 1, dtype=src.dtype, device=src.device)
    counts.scatter_add_(2, expanded_index[:, :, :, 0:1], ones)
    counts = counts.clamp_min_(1.0)
    return out / counts


def _message_passing_fallback(
    x: torch.Tensor,
    edge_index_src: torch.Tensor,
    edge_index_dst: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fallback message passing using PyTorch indexing."""
    B, T, N, F = x.shape
    num_edges = edge_index_src.numel()

    # Gather
    msg = x[:, :, edge_index_src, :]
    if edge_weight is not None:
        if edge_weight.dim() == 1:
            msg = msg * edge_weight.view(1, 1, -1, 1)
        else:
            msg = msg * edge_weight.unsqueeze(1).unsqueeze(-1)

    # Scatter-add
    out = torch.zeros(B, T, N, F, dtype=x.dtype, device=x.device)
    expanded_dst = edge_index_dst.view(1, 1, -1, 1).expand_as(msg)
    out.scatter_add_(2, expanded_dst, msg)
    return out
