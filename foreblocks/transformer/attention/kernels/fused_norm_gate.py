import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_sigmoid_gate_kernel(
        X,
        G,
        W,
        O,
        N: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = (row // T) % H
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N

        x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + h * N + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        y = x * tl.rsqrt(var + eps) * w * g
        tl.store(O + row * N + offs, y, mask=mask)


def can_use_fused_rmsnorm_sigmoid_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    if not HAS_TRITON:
        return False
    if os.environ.get("FOREBLOCKS_DISABLE_TRITON_NORM_GATE", "") == "1":
        return False
    if not (x.is_cuda and gate.is_cuda and weight.is_cuda):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if gate.dtype != x.dtype:
        return False
    if x.shape != gate.shape or x.ndim != 4:
        return False
    if weight.shape != (x.shape[1], x.shape[-1]):
        return False
    return x.shape[-1] <= 8192 and x.stride(-1) == gate.stride(-1) == 1


@torch.compiler.disable
def fused_rmsnorm_sigmoid_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMSNorm(x) * weight * gate for [B, H, T, D] tensors."""
    if not can_use_fused_rmsnorm_sigmoid_gate(x, gate, weight):
        raise RuntimeError("fused_rmsnorm_sigmoid_gate is not available")
    x = x.contiguous()
    gate = gate.contiguous()
    weight = weight.contiguous()
    B, H, T, D = x.shape
    out = torch.empty_like(x)
    block_n = min(65536 // x.element_size(), triton.next_power_of_2(D))
    num_warps = min(max(block_n // 256, 1), 8)
    grid = (B * H * T,)
    _rmsnorm_sigmoid_gate_kernel[grid](
        x,
        gate,
        weight,
        out,
        D,
        T,
        H,
        float(eps),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


__all__ = [
    "can_use_fused_rmsnorm_sigmoid_gate",
    "fused_rmsnorm_sigmoid_gate",
]
