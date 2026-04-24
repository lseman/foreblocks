from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hybrid_mamba import HybridMambaBlock  # noqa: E402
from hybrid_mamba import (
    TinyHybridMambaLM,
    benchmark_block,
    benchmark_causal_conv,
    check_backward,
    check_causal_conv_backward,
    check_causal_conv_close,
    check_forward_close,
    compare_against_official,
    precompile_selective_scan_extension,
)


def _ensure_import_path() -> None:
    if __package__:
        return
    package_dir = Path(__file__).resolve().parent
    parent = package_dir.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


_ensure_import_path()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def run_import_test() -> None:
    print("import ok")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())


def run_block_forward(
    *,
    device: str,
    batch: int,
    seqlen: int,
    d_model: int,
    d_state: int,
) -> None:
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = HybridMambaBlock(
        d_model=d_model,
        d_inner=2 * d_model,
        d_state=d_state,
        d_conv=4,
        use_cuda_scan=device == "cuda",
    ).to(device=device, dtype=dtype)
    x = torch.randn(batch, seqlen, d_model, device=device, dtype=dtype)
    y = model(x)
    print("block forward ok")
    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("dtype:", y.dtype)
    print("device:", y.device)


def run_lm_forward(
    *,
    device: str,
    batch: int,
    seqlen: int,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    d_state: int,
) -> None:
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = TinyHybridMambaLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=4,
    ).to(device=device, dtype=dtype)
    input_ids = torch.randint(0, vocab_size, (batch, seqlen), device=device)
    logits = model(input_ids)
    print("lm forward ok")
    print("input shape:", tuple(input_ids.shape))
    print("logits shape:", tuple(logits.shape))
    print("dtype:", logits.dtype)
    print("device:", logits.device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quick smoke tests for hybrid_mamba")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=[
            "all",
            "import",
            "precompile",
            "forward",
            "lm",
            "close",
            "close-causal",
            "close-causal-backward",
            "backward",
            "compare-official",
            "bench-causal",
            "bench-block",
        ],
        help="Which smoke test to run",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")

    args = parser.parse_args(argv)
    device = _resolve_device(args.device)

    if args.mode in {"all", "import"}:
        run_import_test()

    if args.mode in {"all", "precompile"}:
        if not torch.cuda.is_available():
            print("precompile skipped: CUDA is not available")
        else:
            precompile_selective_scan_extension(
                verbose=args.verbose_build,
                force=args.force_build,
            )
            print("precompile ok")

    if args.mode in {"all", "forward"}:
        run_block_forward(
            device=device,
            batch=args.batch,
            seqlen=args.seqlen,
            d_model=args.d_model,
            d_state=args.d_state,
        )

    if args.mode in {"all", "lm"}:
        run_lm_forward(
            device=device,
            batch=args.batch,
            seqlen=args.seqlen,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=args.d_state,
        )

    if args.mode in {"all", "close-causal"}:
        check_causal_conv_close(
            B=args.batch,
            T=args.seqlen,
            D=args.d_model,
            K=4,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    if args.mode in {"all", "close-causal-backward"}:
        check_causal_conv_backward(
            B=args.batch,
            T=args.seqlen,
            D=args.d_model,
            K=4,
            dtype=getattr(torch, args.dtype),
        )

    if args.mode in {"all", "close"}:
        check_forward_close(
            B=args.batch,
            T=args.seqlen,
            D=args.d_model,
            N=args.d_state,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    if args.mode in {"all", "backward"}:
        check_backward(
            B=args.batch,
            T=min(args.seqlen, 16),
            D=args.d_model,
            N=args.d_state,
            dtype=torch.float32,
        )

    if args.mode in {"all", "compare-official"}:
        compare_against_official(
            B=args.batch,
            T=min(args.seqlen, 16),
            D=args.d_model,
            N=args.d_state,
            dtype=getattr(torch, args.dtype),
        )

    if args.mode == "bench-causal":
        benchmark_causal_conv(
            B=args.batch,
            T=args.seqlen,
            D=args.d_model,
            K=4,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            iters=args.iters,
            warmup=args.warmup,
        )

    if args.mode == "bench-block":
        benchmark_block(
            B=args.batch,
            T=args.seqlen,
            d_model=args.d_model,
            d_state=args.d_state,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            iters=args.iters,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
