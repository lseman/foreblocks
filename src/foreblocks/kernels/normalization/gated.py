"""foreblocks.kernels.normalization.gated.

Fused RMSNorm + sigmoid gate kernel for Foreblocks attention tensors.

Applies RMSNorm, element-wise sigmoid gating, and per-head weight scaling in a single
Triton kernel launch. Designed for `[B, H, T, D]` tensors common in attention
backends. Use when you need the combined normalization + gating step (e.g., Mamba2-style
RMSNormGated) with maximum throughput.

Core API:
- can_use_fused_rmsnorm_sigmoid_gate: runtime capability check (Triton, CUDA, shape/dtype valid)
- fused_rmsnorm_sigmoid_gate: RMSNorm(x) * weight * activated_gate, with autograd

The default accepts a preactivated gate for backward compatibility. Select
gate_activation="sigmoid" or "silu" to fuse activation of raw gate logits.

"""

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
        OUT,
        N: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = (row // T) % H
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N

        x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "sigmoid":
            g = tl.sigmoid(g)
        elif ACTIVATION == "silu":
            g = g * tl.sigmoid(g)
        w = tl.load(W + h * N + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        y = x * tl.rsqrt(var + eps) * w * g
        tl.store(OUT + row * N + offs, y, mask=mask)

    @triton.jit
    def _rmsnorm_gate_bwd_kernel(
        X,
        G,
        W,
        DY,
        DX,
        DG,
        DW,
        B: tl.constexpr,
        H: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        EPS: tl.constexpr,
        GROUPS: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        head = tl.program_id(0)
        group = tl.program_id(1)
        cols = tl.arange(0, BLOCK)
        rows = tl.arange(0, ROWS)
        weight = tl.load(W + head * D + cols, cols < D, other=0).to(tl.float32)
        dw = tl.zeros((BLOCK,), tl.float32)
        for start in range(group * ROWS, B * T, GROUPS * ROWS):
            bt = start + rows
            offsets = (
                ((bt // T) * H + head)[:, None] * T * D
                + (bt % T)[:, None] * D
                + cols[None, :]
            )
            mask = (bt[:, None] < B * T) & (cols[None, :] < D)
            x = tl.load(X + offsets, mask, other=0).to(tl.float32)
            raw = tl.load(G + offsets, mask, other=0).to(tl.float32)
            dy = tl.load(DY + offsets, mask, other=0).to(tl.float32)
            if ACTIVATION == "identity":
                gate = raw
                derivative = tl.full((ROWS, BLOCK), 1.0, tl.float32)
            else:
                sig = tl.sigmoid(raw)
                if ACTIVATION == "sigmoid":
                    gate = sig
                    derivative = sig * (1 - sig)
                else:
                    gate = raw * sig
                    derivative = sig * (1 + raw * (1 - sig))
            rms = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
            scaled = dy * weight[None, :] * gate
            projection = tl.sum(scaled * x, 1) / D
            dx = rms[:, None] * (scaled - x * (rms * rms * projection)[:, None])
            dg = dy * x * rms[:, None] * weight[None, :] * derivative
            tl.store(DX + offsets, dx, mask)
            tl.store(DG + offsets, dg, mask)
            dw += tl.sum(tl.where(mask, dy * x * rms[:, None] * gate, 0.0), 0)
        tl.store(DW + (head * GROUPS + group) * D + cols, dw, cols < D)


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
    return (
        x.device == gate.device == weight.device
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() > 0
        and x.shape[-1] <= 8192
        and x.stride(-1) == gate.stride(-1) == 1
    )


def _gate_forward(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    gate_activation: str = "identity",
) -> torch.Tensor:
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
        ACTIVATION=gate_activation,
        num_warps=num_warps,
    )
    return out


class _RMSNormGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, weight, eps, gate_activation):
        x, gate, weight = x.contiguous(), gate.contiguous(), weight.contiguous()
        ctx.save_for_backward(x, gate, weight)
        ctx.eps = eps
        ctx.gate_activation = gate_activation
        return _gate_forward(x, gate, weight, eps, gate_activation)

    @staticmethod
    def backward(ctx, grad_out):
        x, gate, weight = ctx.saved_tensors
        B, H, T, D = x.shape
        rows = 4 if D <= 256 else 1
        groups = min(
            triton.cdiv(B * T, rows),
            2 * torch.cuda.get_device_properties(x.device).multi_processor_count,
        )
        partial = torch.empty(H, groups, D, device=x.device, dtype=torch.float32)
        dx, dg = torch.empty_like(x), torch.empty_like(gate)
        _rmsnorm_gate_bwd_kernel[(H, groups)](
            x,
            gate,
            weight,
            grad_out.contiguous(),
            dx,
            dg,
            partial,
            B,
            H,
            T,
            D,
            ctx.eps,
            groups,
            rows,
            triton.next_power_of_2(D),
            ctx.gate_activation,
        )
        return dx, dg, partial.sum(1).to(weight.dtype), None, None


@torch.compiler.disable
def fused_rmsnorm_sigmoid_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    gate_activation: str = "identity",
) -> torch.Tensor:
    """RMSNorm(x) * weight * gate, with optional fused gate activation.

    The default preserves the legacy preactivated-gate API. Pass ``sigmoid``
    or ``silu`` for raw gate logits. All three tensor inputs support autograd.
    """
    if gate_activation not in {"identity", "sigmoid", "silu"}:
        raise ValueError("gate_activation must be identity, sigmoid, or silu")
    if not can_use_fused_rmsnorm_sigmoid_gate(x, gate, weight):
        raise RuntimeError("fused_rmsnorm_sigmoid_gate is not available")
    return _RMSNormGateFunction.apply(x, gate, weight, eps, gate_activation)


__all__ = [
    "can_use_fused_rmsnorm_sigmoid_gate",
    "fused_rmsnorm_sigmoid_gate",
]
