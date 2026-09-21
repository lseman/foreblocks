"""Triton residual addition; eligibility and dropout decisions belong to the caller."""

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:

    @triton.jit
    def _add_kernel(
        residual_ptr,
        update_ptr,
        out_ptr,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        a = tl.load(residual_ptr + offs, mask=mask, other=0.0)
        b = tl.load(update_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a + b, mask=mask)


def triton_add(residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(residual)
    r = residual.contiguous().view(-1)
    u = update.contiguous().view(-1)
    o = out.view(-1)
    n = o.numel()
    block = 1024
    grid = (triton.cdiv(n, block),)
    _add_kernel[grid](r, u, o, n, BLOCK=block)
    return out


