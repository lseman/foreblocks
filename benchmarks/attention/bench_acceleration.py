#!/usr/bin/env python3
"""Measure the five attention/MoE accelerations against equivalent eager work.

Run with PYTHONPATH=src. Compilation/autotuning is warmed before measurement.
Outputs and state are checked before timing; gate training also checks gradients.
Use --output to save hardware/version metadata and measurements as JSON.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from triton import __version__ as triton_version
from triton.testing import do_bench

from foreblocks.kernels.attention.linear import fused_recurrent_causal_linear_attn
from foreblocks.kernels.attention.short_conv import streaming_short_conv
from foreblocks.kernels.linear.grouped_gemm import grouped_mm_varM
from foreblocks.kernels.normalization.gated import fused_rmsnorm_sigmoid_gate
from foreblocks.nn.attention.execution.window import SparseWindowAttention


def measure(name, baseline, accelerated, *, training=False, rep=100):
    with torch.set_grad_enabled(training):
        torch.testing.assert_close(accelerated(), baseline(), atol=0.08, rtol=0.04)
        eager_ms = do_bench(baseline, warmup=25, rep=rep)
        accelerated_ms = do_bench(accelerated, warmup=25, rep=rep)
        memory = {}
        for label, operation in (("eager", baseline), ("accelerated", accelerated)):
            torch.cuda.synchronize()
            start = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output = operation()
            torch.cuda.synchronize()
            memory[label] = torch.cuda.max_memory_allocated() - start
            del output
    result = dict(
        name=name,
        eager_ms=eager_ms,
        accelerated_ms=accelerated_ms,
        speedup=eager_ms / accelerated_ms,
        peak_temporary_bytes=memory,
    )
    print(
        f"{name}: {eager_ms:.4f} -> {accelerated_ms:.4f} ms ({result['speedup']:.2f}x)",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.rep <= 0:
        parser.error("requires CUDA and a positive --rep")
    torch.manual_seed(67)
    device, dtype = "cuda", torch.bfloat16
    results = []

    q = torch.randn(2, 8, 1024, 32, device=device, dtype=dtype)
    k, v = [torch.randn(2, 2, 1024, 32, device=device, dtype=dtype) for _ in range(2)]
    sparse = SparseWindowAttention()

    def dense_window():
        positions = torch.arange(1024, device=device)
        distance = positions[:, None] - positions[None, :]
        keep = (distance >= 0) & (distance < 64)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=keep, enable_gqa=True)

    def sparse_window():
        return sparse(q, k, v, window=64, causal=True, scale=32**-0.5)

    results.append(
        measure(
            "sliding_window_B2_H8_KV2_T1024_D32",
            dense_window,
            sparse_window,
            rep=args.rep,
        )
    )

    x, gate = [
        torch.randn(4, 8, 128, 64, device=device, dtype=dtype).requires_grad_()
        for _ in range(2)
    ]
    weight = torch.randn(8, 64, device=device, requires_grad=True)
    gradient = torch.randn_like(x)

    def eager_gate():
        return (
            x.float()
            * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            * weight[None, :, None]
            * gate.float().sigmoid()
        ).to(dtype)

    def fused_gate():
        return fused_rmsnorm_sigmoid_gate(x, gate, weight, gate_activation="sigmoid")

    def eager_gate_train():
        return torch.autograd.grad(eager_gate(), (x, gate, weight), gradient)

    def fused_gate_train():
        return torch.autograd.grad(fused_gate(), (x, gate, weight), gradient)

    results.append(
        measure(
            "norm_gate_forward_B4_H8_T128_D64", eager_gate, fused_gate, rep=args.rep
        )
    )
    results.append(
        measure(
            "norm_gate_forward_backward_B4_H8_T128_D64",
            eager_gate_train,
            fused_gate_train,
            training=True,
            rep=args.rep,
        )
    )

    lq, lk = [torch.rand(4, 8, 1, 64, device=device, dtype=dtype) for _ in range(2)]
    lv = torch.randn_like(lq)
    z = torch.rand(4, 8, 64, device=device)
    state = torch.randn(4, 8, 64, 64, device=device)

    def eager_decode():
        next_z = z + lk[:, :, 0].float()
        next_state = state + lk[:, :, 0].float().unsqueeze(-1) * lv[
            :, :, 0
        ].float().unsqueeze(-2)
        output = (lq[:, :, 0].float().unsqueeze(-1) * next_state).sum(-2) / (
            (lq[:, :, 0].float() * next_z).sum(-1, keepdim=True) + 1e-6
        )
        return output.unsqueeze(2).to(dtype), (next_z, next_state)

    def fused_decode():
        return fused_recurrent_causal_linear_attn(
            lq, lk, lv, initial_state=(z, state), output_final_state=True
        )

    results.append(
        measure(
            "stateful_linear_decode_B4_H8_D64", eager_decode, fused_decode, rep=args.rep
        )
    )

    for count, rows in ((32, 64), (32, 512)):
        a = torch.randn(count * rows, 128, device=device, dtype=dtype)
        b = torch.randn(count, 128, 128, device=device, dtype=dtype)
        offsets = torch.arange(count + 1, device=device) * rows

        def legacy_gemm():
            return grouped_mm_varM(a, offsets, b, backend="legacy")

        def persistent_gemm():
            return grouped_mm_varM(a, offsets, b, backend="persistent")

        results.append(
            measure(
                f"grouped_gemm_E{count}_M{rows}_K128_N128",
                legacy_gemm,
                persistent_gemm,
                rep=args.rep,
            )
        )

    cx = torch.randn(4, 1, 512, device=device, dtype=dtype)
    cw = torch.randn(512, 4, device=device, dtype=dtype)
    cb = torch.randn(512, device=device, dtype=dtype)
    history = torch.randn(4, 512, 3, device=device, dtype=dtype)

    def eager_conv():
        joined = torch.cat((history, cx.transpose(1, 2)), -1)
        out = F.silu(F.conv1d(joined, cw[:, None], cb, groups=512))
        return out.transpose(1, 2).contiguous(), joined[..., -3:].clone()

    def fused_conv():
        return streaming_short_conv(cx, cw, cb, history, activation="silu")

    results.append(
        measure("short_conv_update_B4_D512_K4", eager_conv, fused_conv, rep=args.rep)
    )
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton_version,
        dtype=str(dtype),
        measurements=results,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
