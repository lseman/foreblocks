from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

from .cuda import load_selective_scan_extension
from .cuda import precompile_selective_scan_extension
from .layers import HybridMambaBlock
from .layers import TinyHybridMambaLM
from .ops import CAUSAL_CONV1D_TRITON_AVAILABLE
from .ops import TRITON_AVAILABLE
from .ops import causal_depthwise_conv1d
from .ops import causal_depthwise_conv1d_reference
from .ops import dt_prep
from .ops import selective_scan
from .ops import selective_scan_reference


def _sync_if_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def check_causal_conv_close(
    B: int = 2,
    T: int = 32,
    D: int = 64,
    K: int = 4,
    dtype: torch.dtype = torch.float16,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not CAUSAL_CONV1D_TRITON_AVAILABLE:
        print("Triton not available; skipping causal-conv compare.")
        return

    x = torch.randn(B, D, T, device=device, dtype=dtype)
    weight = torch.randn(D, K, device=device, dtype=dtype) * 0.1
    bias = torch.randn(D, device=device, dtype=dtype) * 0.1

    y_ref = causal_depthwise_conv1d_reference(x, weight, bias)
    y_fast = causal_depthwise_conv1d(x, weight, bias)

    max_abs = (y_ref - y_fast).abs().max().item()
    mean_abs = (y_ref - y_fast).abs().mean().item()
    print(f"[causal conv close] max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}")


def check_causal_conv_backward(
    B: int = 2,
    T: int = 32,
    D: int = 64,
    K: int = 4,
    dtype: torch.dtype = torch.float32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not CAUSAL_CONV1D_TRITON_AVAILABLE:
        print("Triton not available; skipping causal-conv backward compare.")
        return

    forward_tol, grad_tol = _default_compare_tolerances(dtype)
    torch.manual_seed(0)

    def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(B, D, T, device=device, dtype=dtype, requires_grad=True)
        weight = (
            (torch.randn(D, K, device=device, dtype=dtype) * 0.1)
            .detach()
            .requires_grad_(True)
        )
        bias = (
            (torch.randn(D, device=device, dtype=dtype) * 0.1)
            .detach()
            .requires_grad_(True)
        )
        return x, weight, bias

    x_fast, w_fast, b_fast = _make_inputs()
    x_ref = x_fast.detach().clone().requires_grad_(True)
    w_ref = w_fast.detach().clone().requires_grad_(True)
    b_ref = b_fast.detach().clone().requires_grad_(True)

    y_fast = causal_depthwise_conv1d(x_fast, w_fast, b_fast)
    y_ref = causal_depthwise_conv1d_reference(x_ref, w_ref, b_ref)

    fast_loss = y_fast.square().mean()
    ref_loss = y_ref.square().mean()
    fast_loss.backward()
    ref_loss.backward()

    failures: list[str] = []
    if (
        _print_compare_metric(
            "causal_conv forward", y_fast.detach(), y_ref.detach(), forward_tol
        )
        > forward_tol
    ):
        failures.append("forward")
    if (
        _print_compare_metric("causal_conv grad[x]", x_fast.grad, x_ref.grad, grad_tol)
        > grad_tol
    ):
        failures.append("grad[x]")
    if (
        _print_compare_metric(
            "causal_conv grad[weight]", w_fast.grad, w_ref.grad, grad_tol
        )
        > grad_tol
    ):
        failures.append("grad[weight]")
    if (
        _print_compare_metric(
            "causal_conv grad[bias]", b_fast.grad, b_ref.grad, grad_tol
        )
        > grad_tol
    ):
        failures.append("grad[bias]")

    if failures:
        raise AssertionError(
            "causal conv backward comparison failed for: " + ", ".join(failures)
        )

    print("[causal conv backward] ok")


@torch.no_grad()
def check_forward_close(
    B: int = 2,
    T: int = 32,
    D: int = 64,
    N: int = 8,
    dtype: torch.dtype = torch.float16,
):
    if not torch.cuda.is_available():
        print("CUDA not available; skipping forward-close test.")
        return

    device = "cuda"
    u = torch.randn(B, T, D, device=device, dtype=dtype)
    dt = F.softplus(torch.randn(B, T, D, device=device, dtype=dtype)) * 0.1
    A = -torch.exp(torch.randn(D, N, device=device, dtype=dtype))
    Bpar = torch.randn(B, T, D, N, device=device, dtype=dtype) * 0.1
    Cpar = torch.randn(B, T, D, N, device=device, dtype=dtype) * 0.1
    Dskip = torch.randn(D, device=device, dtype=dtype)

    y_ref = selective_scan_reference(u, dt, A, Bpar, Cpar, Dskip)
    y_cuda = selective_scan(u, dt, A, Bpar, Cpar, Dskip, use_cuda_kernel=True)

    max_abs = (y_ref - y_cuda).abs().max().item()
    mean_abs = (y_ref - y_cuda).abs().mean().item()
    print(f"[forward close] max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}")


def check_backward(
    B: int = 2,
    T: int = 16,
    D: int = 32,
    N: int = 8,
    dtype: torch.dtype = torch.float32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    u = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    dt_raw = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    dt_bias = torch.randn(D, device=device, dtype=dtype, requires_grad=True)
    dt = dt_prep(dt_raw, dt_bias)

    A_log = torch.randn(D, N, device=device, dtype=dtype, requires_grad=True)
    A = -torch.exp(A_log)
    Bpar = torch.randn(B, T, D, N, device=device, dtype=dtype, requires_grad=True) * 0.1
    Cpar = torch.randn(B, T, D, N, device=device, dtype=dtype, requires_grad=True) * 0.1
    Dskip = torch.randn(D, device=device, dtype=dtype, requires_grad=True)

    y = selective_scan(u, dt, A, Bpar, Cpar, Dskip, use_cuda_kernel=True)
    loss = y.square().mean()
    loss.backward()

    print("[backward] ok")
    print("u.grad:", u.grad.abs().mean().item())
    print("dt_raw.grad:", dt_raw.grad.abs().mean().item())
    print("A_log.grad:", A_log.grad.abs().mean().item())
    print("Dskip.grad:", Dskip.grad.abs().mean().item())


def _default_compare_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in {torch.float16, torch.bfloat16}:
        return 2e-3, 2e-2
    return 1e-5, 1e-4


def _error_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs - rhs).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _print_compare_metric(
    label: str, lhs: torch.Tensor, rhs: torch.Tensor, tol: float
) -> float:
    max_abs, mean_abs = _error_stats(lhs, rhs)
    status = "ok" if max_abs <= tol else "FAIL"
    print(
        f"[compare] {label}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, tol={tol:.6e} [{status}]"
    )
    return max_abs


def _clone_input_dict(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone().requires_grad_(tensor.requires_grad)
        for name, tensor in inputs.items()
    }


def _run_impl(
    run_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    local_inputs = _clone_input_dict(inputs)
    out = run_fn(local_inputs)
    loss = out.square().mean()
    loss.backward()
    grads = {
        name: (tensor.grad.detach().clone() if tensor.grad is not None else None)
        for name, tensor in local_inputs.items()
    }
    return out.detach(), grads


def compare_against_official(
    B: int = 2,
    T: int = 16,
    D: int = 32,
    N: int = 8,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    assert_close: bool = True,
) -> None:
    try:
        from mamba_ssm.ops.selective_scan_interface import (
            selective_scan_fn as official_selective_scan_fn,
        )
        from mamba_ssm.ops.selective_scan_interface import (
            selective_scan_ref as official_selective_scan_ref,
        )
    except Exception as exc:
        print(f"official compare skipped: could not import mamba_ssm ({exc})")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    forward_tol, grad_tol = _default_compare_tolerances(dtype)
    torch.manual_seed(seed)

    inputs = {
        "u": torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True),
        "dt": (torch.rand(B, T, D, device=device, dtype=dtype) * 0.1 + 1e-3)
        .detach()
        .requires_grad_(True),
        "A": (-torch.exp(torch.randn(D, N, device=device, dtype=dtype)))
        .detach()
        .requires_grad_(True),
        "Bpar": (torch.randn(B, T, D, N, device=device, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True),
        "Cpar": (torch.randn(B, T, D, N, device=device, dtype=dtype) * 0.1)
        .detach()
        .requires_grad_(True),
        "Dskip": torch.randn(D, device=device, dtype=dtype, requires_grad=True),
    }

    def _to_official(local_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return (
            local_inputs["u"].permute(0, 2, 1).contiguous(),
            local_inputs["dt"].permute(0, 2, 1).contiguous(),
            local_inputs["A"].contiguous(),
            local_inputs["Bpar"].permute(0, 2, 3, 1).contiguous(),
            local_inputs["Cpar"].permute(0, 2, 3, 1).contiguous(),
            local_inputs["Dskip"].contiguous(),
        )

    def _from_official(y: torch.Tensor) -> torch.Tensor:
        return y.permute(0, 2, 1).contiguous()

    def run_custom_reference(local_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return selective_scan_reference(**local_inputs)

    def run_custom_scan(local_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return selective_scan(**local_inputs, use_cuda_kernel=True)

    def run_official_reference(local_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        out = official_selective_scan_ref(*_to_official(local_inputs))
        return _from_official(out)

    def run_official_fast(local_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        out = official_selective_scan_fn(*_to_official(local_inputs))
        return _from_official(out)

    print(f"[compare] device={device} dtype={dtype} seed={seed}")

    custom_ref_out, custom_ref_grads = _run_impl(run_custom_reference, inputs)
    official_ref_out, official_ref_grads = _run_impl(run_official_reference, inputs)

    failures: list[str] = []

    if (
        _print_compare_metric(
            "custom_ref vs official_ref forward",
            custom_ref_out,
            official_ref_out,
            forward_tol,
        )
        > forward_tol
    ):
        failures.append("custom_ref forward")
    for name in inputs:
        grad = custom_ref_grads[name]
        ref_grad = official_ref_grads[name]
        if grad is None or ref_grad is None:
            continue
        if (
            _print_compare_metric(
                f"custom_ref vs official_ref grad[{name}]", grad, ref_grad, grad_tol
            )
            > grad_tol
        ):
            failures.append(f"custom_ref grad[{name}]")

    custom_scan_out, custom_scan_grads = _run_impl(run_custom_scan, inputs)
    if (
        _print_compare_metric(
            "custom_scan vs official_ref forward",
            custom_scan_out,
            official_ref_out,
            forward_tol,
        )
        > forward_tol
    ):
        failures.append("custom_scan forward")
    for name in inputs:
        grad = custom_scan_grads[name]
        ref_grad = official_ref_grads[name]
        if grad is None or ref_grad is None:
            continue
        if (
            _print_compare_metric(
                f"custom_scan vs official_ref grad[{name}]", grad, ref_grad, grad_tol
            )
            > grad_tol
        ):
            failures.append(f"custom_scan grad[{name}]")

    if device == "cuda":
        try:
            official_fast_out, official_fast_grads = _run_impl(
                run_official_fast, inputs
            )
            if (
                _print_compare_metric(
                    "official_fast vs official_ref forward",
                    official_fast_out,
                    official_ref_out,
                    forward_tol,
                )
                > forward_tol
            ):
                failures.append("official_fast forward")
            if (
                _print_compare_metric(
                    "custom_scan vs official_fast forward",
                    custom_scan_out,
                    official_fast_out,
                    forward_tol,
                )
                > forward_tol
            ):
                failures.append("custom_scan vs official_fast forward")
            for name in inputs:
                grad = official_fast_grads[name]
                ref_grad = official_ref_grads[name]
                if grad is None or ref_grad is None:
                    continue
                if (
                    _print_compare_metric(
                        f"official_fast vs official_ref grad[{name}]",
                        grad,
                        ref_grad,
                        grad_tol,
                    )
                    > grad_tol
                ):
                    failures.append(f"official_fast grad[{name}]")
                custom_grad = custom_scan_grads[name]
                if custom_grad is None:
                    continue
                if (
                    _print_compare_metric(
                        f"custom_scan vs official_fast grad[{name}]",
                        custom_grad,
                        grad,
                        grad_tol,
                    )
                    > grad_tol
                ):
                    failures.append(f"custom_scan vs official_fast grad[{name}]")
        except Exception as exc:
            print(f"[compare] official fast CUDA path skipped: {exc}")
    else:
        print("[compare] official fast CUDA path skipped: CUDA not available")

    if failures and assert_close:
        raise AssertionError("official comparison failed for: " + ", ".join(failures))

    if not failures:
        print("[compare] all parity checks passed")


def benchmark_block(
    B: int = 8,
    T: int = 256,
    d_model: int = 128,
    d_state: int = 16,
    dtype: torch.dtype = torch.float16,
    iters: int = 50,
    warmup: int = 10,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HybridMambaBlock(
        d_model=d_model,
        d_inner=2 * d_model,
        d_state=d_state,
        d_conv=4,
        use_cuda_scan=True,
    ).to(device=device, dtype=dtype)

    x = torch.randn(B, T, d_model, device=device, dtype=dtype)

    for _ in range(warmup):
        _ = model(x)
    _sync_if_cuda(device)

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    _sync_if_cuda(device)
    t1 = time.time()

    ms = (t1 - t0) * 1000.0 / iters
    print(f"[benchmark] {ms:.3f} ms/iter")


@torch.no_grad()
def benchmark_causal_conv(
    B: int = 8,
    T: int = 512,
    D: int = 256,
    K: int = 4,
    dtype: torch.dtype = torch.float16,
    iters: int = 100,
    warmup: int = 20,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(B, D, T, device=device, dtype=dtype)
    weight = torch.randn(D, K, device=device, dtype=dtype) * 0.1
    bias = torch.randn(D, device=device, dtype=dtype) * 0.1

    def _run(
        fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        label: str,
    ) -> float:
        for _ in range(warmup):
            _ = fn(x, weight, bias)
        _sync_if_cuda(device)
        t0 = time.time()
        for _ in range(iters):
            _ = fn(x, weight, bias)
        _sync_if_cuda(device)
        t1 = time.time()
        ms = (t1 - t0) * 1000.0 / iters
        print(f"[benchmark causal] {label}: {ms:.3f} ms/iter")
        return ms

    ref_ms = _run(causal_depthwise_conv1d_reference, "reference")
    fast_ms = _run(causal_depthwise_conv1d, "fast")
    if fast_ms > 0:
        print(f"[benchmark causal] speedup: {ref_ms / fast_ms:.2f}x")


def example_train_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = TinyHybridMambaLM(
        vocab_size=5000,
        d_model=128,
        n_layers=2,
        d_state=16,
        d_conv=4,
    ).to(device=device, dtype=dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    input_ids = torch.randint(0, 5000, (4, 128), device=device)
    target = torch.randint(0, 5000, (4, 128), device=device)

    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), target.view(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    print("train step ok, loss =", float(loss.item()))


def run_default_diagnostics() -> None:
    print("Triton available:", TRITON_AVAILABLE)
    print("Causal conv Triton available:", CAUSAL_CONV1D_TRITON_AVAILABLE)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Building CUDA extension...")
        load_selective_scan_extension(verbose=True)

    check_causal_conv_close()
    check_causal_conv_backward()
    check_forward_close()
    check_backward()
    benchmark_causal_conv()
    benchmark_block()
    example_train_step()
    compare_against_official(assert_close=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hybrid Mamba utilities")
    subparsers = parser.add_subparsers(dest="command")

    precompile_parser = subparsers.add_parser(
        "precompile", help="Precompile the CUDA extension"
    )
    precompile_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose extension build logs"
    )
    precompile_parser.add_argument(
        "--force", action="store_true", help="Force a fresh extension load"
    )
    precompile_parser.add_argument(
        "--build-dir",
        type=str,
        default=None,
        help="Optional custom build cache directory",
    )

    subparsers.add_parser(
        "diagnostics", help="Run the original smoke tests and benchmark"
    )
    compare_parser = subparsers.add_parser(
        "compare-official", help="Compare selective scan against mamba_ssm"
    )
    compare_parser.add_argument("--batch", type=int, default=2)
    compare_parser.add_argument("--seqlen", type=int, default=16)
    compare_parser.add_argument("--d-model", type=int, default=32)
    compare_parser.add_argument("--d-state", type=int, default=8)
    compare_parser.add_argument("--seed", type=int, default=0)
    compare_parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    compare_parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Print metrics without failing on tolerance",
    )
    causal_backward_parser = subparsers.add_parser(
        "check-causal-backward", help="Compare causal-conv forward and gradients"
    )
    causal_backward_parser.add_argument("--batch", type=int, default=2)
    causal_backward_parser.add_argument("--seqlen", type=int, default=32)
    causal_backward_parser.add_argument("--d-model", type=int, default=64)
    causal_backward_parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    bench_causal_parser = subparsers.add_parser(
        "bench-causal", help="Benchmark causal conv reference vs fast path"
    )
    bench_causal_parser.add_argument("--batch", type=int, default=8)
    bench_causal_parser.add_argument("--seqlen", type=int, default=512)
    bench_causal_parser.add_argument("--d-model", type=int, default=256)
    bench_causal_parser.add_argument("--iters", type=int, default=100)
    bench_causal_parser.add_argument("--warmup", type=int, default=20)
    bench_block_parser = subparsers.add_parser(
        "bench-block", help="Benchmark the HybridMamba block"
    )
    bench_block_parser.add_argument("--batch", type=int, default=8)
    bench_block_parser.add_argument("--seqlen", type=int, default=256)
    bench_block_parser.add_argument("--d-model", type=int, default=128)
    bench_block_parser.add_argument("--d-state", type=int, default=16)
    bench_block_parser.add_argument("--iters", type=int, default=50)
    bench_block_parser.add_argument("--warmup", type=int, default=10)

    args = parser.parse_args(argv)

    if args.command == "precompile":
        precompile_selective_scan_extension(
            verbose=args.verbose,
            force=args.force,
            build_directory=args.build_dir,
        )
        return

    if args.command == "compare-official":
        compare_against_official(
            B=args.batch,
            T=args.seqlen,
            D=args.d_model,
            N=args.d_state,
            dtype=getattr(torch, args.dtype),
            seed=args.seed,
            assert_close=not args.no_assert,
        )
        return

    if args.command == "check-causal-backward":
        check_causal_conv_backward(
            B=args.batch,
            T=args.seqlen,
            D=args.d_model,
            dtype=getattr(torch, args.dtype),
        )
        return

    if args.command == "bench-causal":
        benchmark_causal_conv(
            B=args.batch,
            T=args.seqlen,
            D=args.d_model,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            iters=args.iters,
            warmup=args.warmup,
        )
        return

    if args.command == "bench-block":
        benchmark_block(
            B=args.batch,
            T=args.seqlen,
            d_model=args.d_model,
            d_state=args.d_state,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            iters=args.iters,
            warmup=args.warmup,
        )
        return

    run_default_diagnostics()
