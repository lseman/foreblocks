#!/usr/bin/env python3
"""Benchmark Mamba2 Triton kernels vs PyTorch implementations."""
from __future__ import annotations

import argparse
import json
import time

import torch


def _sync() -> None:
    torch.cuda.synchronize()


def measure(f, *args, warmup_iters=10, iters=50, **kwargs):
    for _ in range(warmup_iters):
        f(*args, **kwargs)
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        f(*args, **kwargs)
    _sync()
    return (time.perf_counter() - start) / iters * 1000


def speedup(torch_ms: float | None, triton_ms: float | None) -> float | None:
    if torch_ms is None or triton_ms is None or triton_ms <= 0:
        return None
    return torch_ms / triton_ms


def speedup_str(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}x"


def ms_str(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}ms"


def max_abs_err(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs - rhs).abs().max().item()


def make_ssd_args(B, T, H, P, N, chunk_size=64, device="cuda"):
    u = torch.randn(B, T, H, P, device=device)
    dt = torch.abs(torch.randn(B, T, H, device=device)) * 0.1 + 1e-4
    A = -torch.exp(torch.randn(H, device=device) * 0.3)
    Bp = torch.randn(B, T, H, N, device=device)
    Cp = torch.randn(B, T, H, N, device=device)
    D = torch.randn(H, P, device=device)
    return u, dt, A, Bp, Cp, D


# ── Benchmarks ──────────────────────────────────────────────────

def bench_dt_prep(use_triton=True, iters=50):
    from foreblocks.ops.mamba.triton_ops import dt_prep_fallback, dt_prep_triton
    configs = [(2, 256, 512), (2, 1024, 2048), (2, 4096, 4096)]
    results = {}
    for B, T, D in configs:
        dt_raw = torch.randn(B, T, D, device="cuda")
        bias = torch.randn(D, device="cuda")
        fn = dt_prep_triton if use_triton else dt_prep_fallback
        t = measure(fn, dt_raw, bias, iters=iters)
        results[f"B{B}_T{T}_D{D}"] = round(t, 4)
        print(f"  dt_prep{'_triton' if use_triton else '_fallback'} B={B} T={T} D={D}: {t:.4f} ms")
    return results


def bench_fused_dt(use_triton=True, iters=50):
    from foreblocks.ops.mamba.fused_dt import fused_dt_fallback, fused_dt_triton
    configs = [(2, 1024, 16, 8), (2, 4096, 64, 8), (2, 4096, 64, 32)]
    results = {}
    for B, T, R, H in configs:
        dt_hidden = torch.randn(B, T, R, device="cuda")
        weight = torch.randn(H, R, device="cuda")
        bias = torch.randn(H, device="cuda")
        fn = fused_dt_triton if use_triton else fused_dt_fallback
        t = measure(fn, dt_hidden, weight, bias, iters=iters)
        results[f"B{B}_T{T}_R{R}_H{H}"] = round(t, 4)
        print(
            f"  fused_dt{'_triton' if use_triton else '_fallback'} "
            f"B={B} T={T} R={R} H={H}: {t:.4f} ms"
        )
    return results


def bench_fused_out(use_triton=True, iters=50):
    from foreblocks.ops.mamba.triton_ops import fused_out_fallback, fused_out_triton
    configs = [(2, 256, 512), (2, 1024, 2048), (2, 4096, 4096)]
    results = {}
    for B, T, D in configs:
        y = torch.randn(B, T, D, device="cuda")
        z = torch.randn(B, T, D, device="cuda")
        res = torch.randn(B, T, D, device="cuda")
        w = torch.randn(D, device="cuda")
        fn = fused_out_triton if use_triton else fused_out_fallback
        t = measure(fn, y, z, res, w, iters=iters)
        results[f"B{B}_T{T}_D{D}"] = round(t, 4)
        print(f"  fused_out{'_triton' if use_triton else '_fallback'} B={B} T={T} D={D}: {t:.4f} ms")
    return results


def bench_ssd_forward_sweep(Ts, B=2, H=8, P=16, N=8, chunk_size=64, iters=50):
    from foreblocks.ops.mamba.ssd import (
        _chunked_ssd_forward_torch as torch_fwd,
        chunked_ssd_forward_triton as triton_fwd,
        chunked_ssd_forward_triton_parallel as triton_parallel_fwd,
        chunked_ssd_forward_triton_tiled as triton_tiled_fwd,
    )

    results = {}
    for T in Ts:
        u, dt, A, Bp, Cp, D = make_ssd_args(B, T, H, P, N, chunk_size)
        n_chunks = (T + chunk_size - 1) // chunk_size

        try:
            t_torch = measure(torch_fwd, u, dt, A, Bp, Cp, D, chunk_size, iters=iters)
        except Exception as e:
            t_torch = None
            print(f"  SSD fwd B={B} T={T} (torch): ERROR {e}")
            continue

        try:
            t_triton = measure(triton_fwd, u, dt, A, Bp, Cp, D, chunk_size, iters=iters)
        except Exception as e:
            t_triton = None
            print(f"  SSD fwd B={B} T={T} (triton): ERROR {e}")

        try:
            t_parallel = measure(
                triton_parallel_fwd,
                u,
                dt,
                A,
                Bp,
                Cp,
                D,
                chunk_size,
                iters=iters,
            )
        except Exception as e:
            t_parallel = None
            print(f"  SSD fwd B={B} T={T} (triton_parallel): ERROR {e}")

        try:
            t_tiled = measure(
                triton_tiled_fwd,
                u,
                dt,
                A,
                Bp,
                Cp,
                D,
                chunk_size,
                iters=iters,
            )
        except Exception as e:
            t_tiled = None
            print(f"  SSD fwd B={B} T={T} (triton_tiled): ERROR {e}")

        ratio = speedup(t_torch, t_triton)
        parallel_ratio = speedup(t_torch, t_parallel)
        tiled_ratio = speedup(t_torch, t_tiled)
        print(
            f"  SSD fwd B={B} T={T} chunks={n_chunks}: "
            f"torch={ms_str(t_torch)} triton={ms_str(t_triton)} "
            f"parallel_direct={ms_str(t_parallel)} parallel_tiled={ms_str(t_tiled)} "
            f"speedup={speedup_str(ratio)} direct_speedup={speedup_str(parallel_ratio)} "
            f"tiled_speedup={speedup_str(tiled_ratio)}"
        )

        results[str(T)] = {
            "chunks": n_chunks,
            "torch_ms": round(t_torch, 4) if t_torch else None,
            "triton_ms": round(t_triton, 4) if t_triton else None,
            "triton_parallel_ms": round(t_parallel, 4) if t_parallel else None,
            "triton_tiled_ms": round(t_tiled, 4) if t_tiled else None,
            "speedup": round(ratio, 4) if ratio else None,
            "parallel_speedup": round(parallel_ratio, 4) if parallel_ratio else None,
            "tiled_speedup": round(tiled_ratio, 4) if tiled_ratio else None,
        }
    return results


def bench_ssd_backward_sweep(Ts, B=2, H=8, P=16, N=8, chunk_size=64, iters=20):
    from foreblocks.ops.mamba.ssd import (
        _chunked_ssd_backward_torch as torch_bwd,
        _chunked_ssd_forward_torch as torch_fwd,
        chunked_ssd_backward_triton as triton_bwd,
    )

    results = {}
    for T in Ts:
        u, dt, A, Bp, Cp, D = make_ssd_args(B, T, H, P, N, chunk_size)
        out = torch_fwd(u, dt, A, Bp, Cp, D, chunk_size)
        gy = torch.randn_like(out)
        n_chunks = (T + chunk_size - 1) // chunk_size

        try:
            t_torch = measure(
                torch_bwd, gy, u, dt, A, Bp, Cp, D, chunk_size,
                needs_input_grad=(True, True, False, True, True, True),
                iters=iters,
            )
        except Exception as e:
            t_torch = None
            print(f"  SSD bwd B={B} T={T} (torch): ERROR {e}")
            continue

        try:
            t_triton = measure(
                triton_bwd, gy, u, dt, A, Bp, Cp, D, chunk_size,
                needs_input_grad=(True, True, False, True, True, True),
                iters=iters,
            )
        except Exception as e:
            t_triton = None
            print(f"  SSD bwd B={B} T={T} (triton): ERROR {e}")

        ratio = speedup(t_torch, t_triton)
        print(
            f"  SSD raw bwd B={B} T={T} chunks={n_chunks}: "
            f"torch={t_torch:.3f}ms triton={t_triton:.3f}ms speedup={speedup_str(ratio)}"
        )

        results[str(T)] = {
            "chunks": n_chunks,
            "torch_ms": round(t_torch, 4) if t_torch else None,
            "triton_ms": round(t_triton, 4) if t_triton else None,
            "speedup": round(ratio, 4) if ratio else None,
        }
    return results


def bench_full_mamba2(B, T, num_heads=8, head_dim=512,
                      n_groups=4, d_state=16, dt_rank=64, chunk_size=64,
                      iters=20):
    from foreblocks.ops.mamba.mamba2_combined import mamba2_split_conv1d_scan_combined

    device = "cuda"
    d_inner = num_heads * head_dim  # total capacity
    conv_dim = d_inner + 2 * n_groups * d_state  # conv output = u + B + C
    projected = torch.randn(B, T, d_inner + conv_dim + dt_rank, device=device)
    residual_inner = torch.randn(B, T, d_inner, device=device)
    conv_weight = torch.randn(conv_dim, 3, device=device)
    conv_bias = torch.randn(conv_dim, device=device)
    dt_proj_weight = torch.randn(num_heads, dt_rank, device=device)
    dt_bias = torch.randn(num_heads, device=device)
    A_log = torch.randn(num_heads, device=device)
    Dskip = torch.randn(num_heads, head_dim, device=device)
    norm_weight = torch.randn(d_inner, device=device)
    out_proj_weight = torch.randn(d_inner, d_inner, device=device)
    out_proj_bias = torch.randn(d_inner, device=device)

    args = dict(
        conv_weight=conv_weight, conv_bias=conv_bias,
        dt_proj_weight=dt_proj_weight, dt_bias=dt_bias,
        A_log=A_log, Dskip=Dskip, norm_weight=norm_weight,
        out_proj_weight=out_proj_weight, out_proj_bias=out_proj_bias,
        d_inner=d_inner, conv_dim=conv_dim, dt_rank=dt_rank,
        num_heads=num_heads, head_dim=head_dim, n_groups=n_groups,
        d_state=d_state, chunk_size=chunk_size,
        dt_limit=(1e-4, 1.0), norm_eps=1e-6,
    )

    t_triton = measure(
        mamba2_split_conv1d_scan_combined, projected, residual_inner,
        use_triton_ssd=True, iters=iters,
        **args,
    )
    t_torch = measure(
        mamba2_split_conv1d_scan_combined, projected, residual_inner,
        use_triton_ssd=False, iters=iters,
        **args,
    )
    ratio = speedup(t_torch, t_triton)
    print(
        f"  Full Mamba2 B={B} T={T}: triton_ssd={t_triton:.3f}ms "
        f"torch_ssd={t_torch:.3f}ms speedup={speedup_str(ratio)}"
    )
    return {
        "triton_ssd_ms": round(t_triton, 4),
        "torch_ssd_ms": round(t_torch, 4),
        "speedup": round(ratio, 4) if ratio else None,
    }


def check_correctness():
    from foreblocks.ops.mamba.fused_dt import fused_dt_fallback, fused_dt_triton
    from foreblocks.ops.mamba.mamba2_combined import mamba2_split_conv1d_scan_combined
    from foreblocks.ops.mamba.ssd import (
        _chunked_ssd_forward_torch as torch_fwd,
        chunked_ssd_forward,
        chunked_ssd_forward_triton as triton_fwd,
        chunked_ssd_forward_triton_parallel as triton_parallel_fwd,
        chunked_ssd_forward_triton_tiled as triton_tiled_fwd,
    )
    from foreblocks.ops.mamba.triton_ops import (
        dt_prep_fallback,
        dt_prep_triton,
        fused_out_fallback,
        fused_out_triton,
    )

    torch.manual_seed(123)
    B, T, H, P, N = 2, 128, 8, 16, 8
    u, dt, A, Bp, Cp, D = make_ssd_args(B, T, H, P, N)

    y_t = triton_fwd(u, dt, A, Bp, Cp, D, 64)
    y_parallel = triton_parallel_fwd(u, dt, A, Bp, Cp, D, 64)
    y_tiled = triton_tiled_fwd(u, dt, A, Bp, Cp, D, 64)
    y_p = torch_fwd(u, dt, A, Bp, Cp, D, 64)
    max_err = (y_t - y_p).abs().max().item()
    rel_err = max_err / y_p.abs().max().item()
    parallel_max_err = max_abs_err(y_parallel, y_p)
    tiled_max_err = max_abs_err(y_tiled, y_p)

    public_args_t = [x.detach().clone().requires_grad_(True) for x in (u, dt, A, Bp, Cp, D)]
    public_args_p = [x.detach().clone().requires_grad_(True) for x in (u, dt, A, Bp, Cp, D)]
    y_public_t = chunked_ssd_forward(*public_args_t, chunk_size=64, use_triton=True)
    y_public_p = chunked_ssd_forward(*public_args_p, chunk_size=64, use_triton=False)
    public_max_err = max_abs_err(y_public_t, y_public_p)
    grad = torch.randn_like(y_public_t)
    y_public_t.backward(grad)
    y_public_p.backward(grad)
    public_grad_errs = {
        name: max_abs_err(t_arg.grad, p_arg.grad)
        for name, t_arg, p_arg in zip(
            ("u", "dt", "A", "B", "C", "D"), public_args_t, public_args_p
        )
    }

    dt_raw = torch.randn(B, T, H, device="cuda")
    bias = torch.randn(H, device="cuda")
    dt_err = max_abs_err(dt_prep_triton(dt_raw, bias), dt_prep_fallback(dt_raw, bias))

    dt_hidden = torch.randn(B, T, 16, device="cuda")
    dt_weight = torch.randn(H, 16, device="cuda")
    fused_dt_err = max_abs_err(
        fused_dt_triton(dt_hidden, dt_weight, bias),
        fused_dt_fallback(dt_hidden, dt_weight, bias),
    )

    y = torch.randn(B, T, H, device="cuda")
    z = torch.randn(B, T, H, device="cuda")
    res = torch.randn(B, T, H, device="cuda")
    w = torch.randn(H, device="cuda")
    fo_err = max_abs_err(fused_out_triton(y, z, res, w), fused_out_fallback(y, z, res, w))

    Bf, Tf, heads, head_dim, groups, d_state, dt_rank = 2, 256, 8, 64, 4, 16, 16
    d_inner = heads * head_dim
    conv_dim = d_inner + 2 * groups * d_state
    projected = torch.randn(Bf, Tf, d_inner + conv_dim + dt_rank, device="cuda")
    residual = torch.randn(Bf, Tf, d_inner, device="cuda")
    full_args = dict(
        conv_weight=torch.randn(conv_dim, 3, device="cuda"),
        conv_bias=torch.randn(conv_dim, device="cuda"),
        dt_proj_weight=torch.randn(heads, dt_rank, device="cuda"),
        dt_bias=torch.randn(heads, device="cuda"),
        A_log=torch.randn(heads, device="cuda"),
        Dskip=torch.randn(heads, head_dim, device="cuda"),
        norm_weight=torch.randn(d_inner, device="cuda"),
        out_proj_weight=torch.randn(d_inner, d_inner, device="cuda"),
        out_proj_bias=torch.randn(d_inner, device="cuda"),
        d_inner=d_inner,
        conv_dim=conv_dim,
        dt_rank=dt_rank,
        num_heads=heads,
        head_dim=head_dim,
        n_groups=groups,
        d_state=d_state,
        chunk_size=64,
        dt_limit=(1e-4, 1.0),
        norm_eps=1e-6,
    )
    full_t = mamba2_split_conv1d_scan_combined(
        projected, residual, use_triton_ssd=True, **full_args
    )
    full_p = mamba2_split_conv1d_scan_combined(
        projected, residual, use_triton_ssd=False, **full_args
    )
    full_err = max_abs_err(full_t, full_p)

    print(f"  SSD forward  max_err={max_err:.2e} rel_err={rel_err:.2e}")
    print(f"  SSD parallel max_err={parallel_max_err:.2e}")
    print(f"  SSD tiled    max_err={tiled_max_err:.2e}")
    print(f"  SSD public   max_err={public_max_err:.2e} grad_max={max(public_grad_errs.values()):.2e}")
    print(f"  dt_prep      max_err={dt_err:.2e}")
    print(f"  fused_dt     max_err={fused_dt_err:.2e}")
    print(f"  fused_out    max_err={fo_err:.2e}")
    print(f"  full_block   max_err={full_err:.2e}")
    return {
        "ssd_max_err": max_err,
        "ssd_rel_err": rel_err,
        "ssd_parallel_max_err": parallel_max_err,
        "ssd_tiled_max_err": tiled_max_err,
        "ssd_public_max_err": public_max_err,
        "ssd_public_grad_errs": public_grad_errs,
        "dt_prep_err": dt_err,
        "fused_dt_err": fused_dt_err,
        "fused_out_err": fo_err,
        "full_block_err": full_err,
    }


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full", "correctness"], default="quick")
    parser.add_argument("--Ts", type=int, nargs="+", default=None)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}, Triton: {__import__('triton').__version__}\n")

    results = {}

    if args.mode == "quick":
        print("=== dt_prep ===")
        results["dt_prep_triton"] = bench_dt_prep(use_triton=True, iters=args.iters)
        results["dt_prep_fallback"] = bench_dt_prep(use_triton=False, iters=args.iters)

        print("\n=== fused_dt ===")
        results["fused_dt_triton"] = bench_fused_dt(use_triton=True, iters=args.iters)
        results["fused_dt_fallback"] = bench_fused_dt(use_triton=False, iters=args.iters)

        print("\n=== fused_out ===")
        results["fused_out_triton"] = bench_fused_out(use_triton=True, iters=args.iters)
        results["fused_out_fallback"] = bench_fused_out(use_triton=False, iters=args.iters)

        Ts = args.Ts or [128, 256, 512, 1024, 2048, 4096]
        print(f"\n=== SSD forward sweep T={Ts} ===")
        results["ssd_fwd_sweep"] = bench_ssd_forward_sweep(Ts, iters=args.iters)

    elif args.mode == "full":
        Ts = args.Ts or [128, 256, 512, 1024, 2048, 4096]
        print("=== dt_prep ===")
        results["dt_prep_triton"] = bench_dt_prep(use_triton=True, iters=args.iters)
        results["dt_prep_fallback"] = bench_dt_prep(use_triton=False, iters=args.iters)

        print("\n=== fused_dt ===")
        results["fused_dt_triton"] = bench_fused_dt(use_triton=True, iters=args.iters)
        results["fused_dt_fallback"] = bench_fused_dt(use_triton=False, iters=args.iters)

        print("\n=== fused_out ===")
        results["fused_out_triton"] = bench_fused_out(use_triton=True, iters=args.iters)
        results["fused_out_fallback"] = bench_fused_out(use_triton=False, iters=args.iters)

        print(f"\n=== SSD forward sweep T={Ts} ===")
        results["ssd_fwd_sweep"] = bench_ssd_forward_sweep(Ts, iters=args.iters)

        print(f"\n=== SSD raw backward sweep T={Ts} ===")
        results["ssd_raw_bwd_sweep"] = bench_ssd_backward_sweep(Ts, iters=max(10, args.iters // 2))

        print("\n=== Full Mamba2 block ===")
        results["full_mamba2"] = bench_full_mamba2(2, 4096, iters=max(10, args.iters // 2))
        results["full_mamba2_small"] = bench_full_mamba2(2, 1024, iters=max(10, args.iters // 2))

    elif args.mode == "correctness":
        print("=== Correctness ===")
        results["correctness"] = check_correctness()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
