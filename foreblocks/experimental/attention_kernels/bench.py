"""foreblocks.experimental.attention_kernels.bench.

Benchmark entry points and measurement helpers.
It belongs to the experimental attention kernel implementations and benchmarks area of Foreblocks.
It exposes functions such as bench_case, main.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from custom_att import flash_attn_backward_backend, flash_attn_func


PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))



def _time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _tol(dtype):
    if dtype is torch.float32:
        return 2e-4, 2e-4
    if dtype is torch.bfloat16:
        return 5e-2, 5e-2
    return 3e-2, 3e-2


def _flops_fwd(B, H, N, D, causal):
    # 2 GEMMs of (N,D)x(D,N) and (N,N)x(N,D). Causal halves both.
    f = 4.0 * B * H * N * N * D
    return f * 0.5 if causal else f


def _flops_bwd(B, H, N, D, causal):
    # FA-2 backward is ~2.5x forward (recompute + 3 GEMMs).
    return 2.5 * _flops_fwd(B, H, N, D, causal)


def _check_correctness(q, k, v, causal, dtype):
    out = flash_attn_func(q, k, v, causal=causal)
    ref = _sdpa(q, k, v, causal)
    atol, rtol = _tol(dtype)
    try:
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
        return "ok"
    except AssertionError:
        max_abs = (out - ref).abs().max().item()
        return f"FAIL (max_abs={max_abs:.2e})"


def bench_case(B, H, N, D, dtype, causal, backward, warmup, iters, check):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=backward)
    k = torch.randn_like(q, requires_grad=backward)
    v = torch.randn_like(q, requires_grad=backward)
    grad = torch.randn_like(q) if backward else None

    status = (
        _check_correctness(q.detach(), k.detach(), v.detach(), causal, dtype)
        if check
        else "-"
    )

    def custom():
        out = flash_attn_func(q, k, v, causal=causal)
        if backward:
            out.backward(grad, retain_graph=True)
            q.grad = k.grad = v.grad = None

    def sdpa():
        out = _sdpa(q, k, v, causal)
        if backward:
            out.backward(grad, retain_graph=True)
            q.grad = k.grad = v.grad = None

    custom_ms = _time_ms(custom, warmup, iters)
    sdpa_ms = _time_ms(sdpa, warmup, iters)
    speed = sdpa_ms / custom_ms if custom_ms > 0 else float("inf")
    flops = (
        _flops_bwd(B, H, N, D, causal) if backward else _flops_fwd(B, H, N, D, causal)
    )
    custom_tflops = flops / (custom_ms * 1e-3) / 1e12
    sdpa_tflops = flops / (sdpa_ms * 1e-3) / 1e12
    mode = "fwd+bwd" if backward else "fwd"
    bwd_backend = flash_attn_backward_backend(q) if backward else "-"
    print(
        f"{mode:7s} N={N:5d} D={D:3d} {str(dtype).split('.')[-1]:9s} "
        f"causal={int(causal)} bwd={bwd_backend:6s} "
        f"custom={custom_ms:7.3f}ms ({custom_tflops:6.1f} TF/s) "
        f"sdpa={sdpa_ms:7.3f}ms ({sdpa_tflops:6.1f} TF/s) "
        f"speedup={speed:5.2f}x  {status}"
    )


_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _sweep_cases(args):
    seqs = args.seqs if args.seqs else [512, 1024, 2048, 4096, 8192]
    dims = args.dims if args.dims else [64, 128]
    dtypes = [_DTYPES[d] for d in (args.dtypes if args.dtypes else ["fp16", "bf16"])]
    causals = [False, True] if args.both_causal else [args.causal]
    for dtype in dtypes:
        for D in dims:
            for N in seqs:
                for causal in causals:
                    yield (args.batch, args.heads, N, D, dtype, causal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--seqs", type=int, nargs="+", default=None)
    parser.add_argument("--dtype", choices=list(_DTYPES), default="fp16")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=None,
        help="Sweep over multiple dtypes, e.g. --dtypes fp16 bf16",
    )
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--both-causal", action="store_true", help="Sweep both causal and non-causal"
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--sweep", action="store_true", help="Run the full (dtype, D, N, causal) matrix"
    )
    parser.add_argument(
        "--no-check", action="store_true", help="Skip correctness check vs SDPA"
    )
    args = parser.parse_args()

    if args.sweep:
        cases = list(_sweep_cases(args))
    else:
        dtype = _DTYPES[args.dtype]
        dims = args.dims if args.dims else [32, 64, 128]
        cases = [
            (args.batch, args.heads, args.seq, D, dtype, args.causal) for D in dims
        ]

    for B, H, N, D, dtype, causal in cases:
        bench_case(
            B,
            H,
            N,
            D,
            dtype,
            causal,
            args.backward,
            args.warmup,
            args.iters,
            check=not args.no_check,
        )


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
