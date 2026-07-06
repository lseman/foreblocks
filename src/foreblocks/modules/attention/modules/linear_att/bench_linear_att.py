#!/usr/bin/env python3
"""foreblocks.modules.attention.modules.linear_att.bench_linear_att.

Benchmark harness comparing custom PyTorch linear attention backends against FLA (flash-linear-attn) kernels.

Runs forward and backward timing passes for all backends (RDA, GLA, DeltaNet,
GatedDeltaNet, GatedDeltaNet2, KimiAttention) against their corresponding FLA
Triton/TileLang kernels, with optional shape correctness checks. CLI-driven;
runs as a script, not importable as a library.

Core API (benchmark harness):
- bench_case: run one benchmark case for one backend
- main: CLI entry point for running all benchmarks

"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from foreblocks.modules.attention.modules.linear_att.deltanet import DeltaNetBackend
from foreblocks.modules.attention.modules.linear_att.gated_delta import GatedDeltaNet
from foreblocks.modules.attention.modules.linear_att.gated_deltanet2 import (
    GatedDeltaNet2,
)
from foreblocks.modules.attention.modules.linear_att.gla import GLABackend
from foreblocks.modules.attention.modules.linear_att.kimi import KimiAttention

# ── Custom backends ──────────────────────────────────────────────────────────
from foreblocks.modules.attention.modules.linear_att.rda import RDABackend

# ── FLA availability ─────────────────────────────────────────────────────────
from foreblocks.ops.attention.fla_backend import (
    fla_path,
    has_fla_checkout,
)
from foreblocks.ops.attention.fla_delta_rule import (
    can_use_fla_delta_rule,
    can_use_fla_recurrent_delta_rule,
    fla_delta_rule_forward,
)
from foreblocks.ops.attention.fla_gated_delta_rule import (
    can_use_fla_gated_delta_rule,
    fla_gated_delta_rule_forward,
)
from foreblocks.ops.attention.fla_gdn2 import (
    can_use_fla_gdn2,
    can_use_fla_gdn2_chunk,
    fla_gdn2_chunk_forward,
    fla_gdn2_forward,
)
from foreblocks.ops.attention.fla_gla import can_use_fla_gla, fla_gla_forward
from foreblocks.ops.attention.fla_kda import can_use_fla_kda, fla_kda_forward
from foreblocks.ops.attention.fla_linear_attention import (
    can_use_fla_linear_attn,
    fla_recurrent_linear_attn_forward,
)

# ── Ensure repo root is on path ──────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_HAS_FLA = has_fla_checkout()


# ── Timing helper ────────────────────────────────────────────────────────────


def _time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


# ── Backend builders ─────────────────────────────────────────────────────────
# All builders return a model placed on CUDA in the requested dtype.


def _build_rda(B, T, H, dk, dv, dtype, chunk_size=64):
    return RDABackend(d_model=H * dk, n_heads=H, dropout=0.0).to("cuda", dtype)


def _build_gla(B, T, H, dk, dv, dtype, chunk_size=64):
    return GLABackend(
        d_model=H * dk, n_heads=H, dropout=0.0, mode="chunk", chunk_size=chunk_size
    ).to("cuda", dtype)


def _build_deltanet(B, T, H, dk, dv, dtype, chunk_size=64):
    return DeltaNetBackend(
        d_model=H * dk, n_heads=H, dropout=0.0, mode="chunk", chunk_size=chunk_size
    ).to("cuda", dtype)


def _build_gated_delta(B, T, H, dk, dv, dtype, chunk_size=64):
    return GatedDeltaNet(
        d_model=H * dk,
        n_heads=H,
        d_key=dk,
        d_val=dv,
        chunk_size=chunk_size if chunk_size and T > chunk_size else 0,
        use_short_conv=False,
        use_mamba_gate=True,
    ).to("cuda", dtype)


def _build_gdn2(B, T, H, dk, dv, dtype, chunk_size=64):
    return GatedDeltaNet2(
        d_model=H * dk,
        n_heads=H,
        d_key=dk,
        d_val=dv,
        chunk_size=chunk_size if chunk_size and T > chunk_size else 0,
        use_short_conv=False,
        eps=1e-6,
        allow_neg_eigval=False,
    ).to("cuda", dtype)


def _build_kimi(B, T, H, dk, dv, dtype, chunk_size=64):
    return KimiAttention(
        d_model=H * dk,
        n_heads=H,
        d_key=dk,
        d_val=dv,
        expand_v=1.0,
        chunk_size=chunk_size if chunk_size and T > chunk_size else 0,
        shortconv_mode="off",
    ).to("cuda", dtype)


# ── Custom forward callables ─────────────────────────────────────────────────

# Backends that expose forward_standalone(x) -> (y, state).
_STANDALONE = ("gated_delta", "gated_deltanet2")


def _make_custom_fwd(model, backend_name, x, inputs):
    """Return a forward-only callable for the custom model."""
    if backend_name in _STANDALONE:

        def fn():
            with torch.no_grad():
                model.forward_standalone(x)

    else:
        # rda / gla / deltanet / kimi: forward(query, key, value) -> (out, _, state)
        def fn():
            with torch.no_grad():
                model(**inputs)

    return fn


def _make_custom_bwd(model, backend_name, x, inputs):
    """Return a forward+backward callable for the custom model."""
    if backend_name in _STANDALONE:

        def fn():
            x2 = x.detach().clone().requires_grad_(True)
            y, _ = model.forward_standalone(x2)
            y.backward(torch.randn_like(y), retain_graph=True)
            x2.grad = None

    else:

        def fn():
            x2 = x.detach().clone().requires_grad_(True)
            out = model(query=x2, key=x2, value=x2)[0]
            out.backward(torch.randn_like(out), retain_graph=True)
            x2.grad = None

    return fn


# ── FLA input extractors ─────────────────────────────────────────────────────
# Every FLA wrapper expects Foreblocks layout [B, H, T, *] and transposes to
# FLA's [B, T, H, *] internally.  Extractors below build random tensors directly
# in [B, H, T, *] — the bench measures kernel throughput, not parity with the
# custom path, so independent random inputs are fine.


def _extract_gla_fla(x, H, dk, dv):
    B, T = x.shape[0], x.shape[1]
    dev, dt = x.device, x.dtype
    q = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    k = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    v = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    # GLA decay gate g must match q exactly (log-domain, <= 0)
    g = F.logsigmoid(torch.randn(B, H, T, dk, device=dev, dtype=dt))
    S0 = torch.zeros(B, H, dk, dk, device=dev, dtype=torch.float32)
    return (q, k, v, g, S0)


def _extract_delta_fla(x, H, dk, dv):
    B, T = x.shape[0], x.shape[1]
    dev, dt = x.device, x.dtype
    q = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    k = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    v = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    beta = torch.sigmoid(torch.randn(B, H, T, 1, device=dev, dtype=dt))
    S0 = torch.zeros(B, H, dk, dk, device=dev, dtype=torch.float32)
    return (q, k, v, beta, S0)


def _extract_gated_delta_fla(x, H, dk, dv):
    B, T = x.shape[0], x.shape[1]
    dev, dt = x.device, x.dtype
    q = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    k = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    v = torch.randn(B, H, T, dv, device=dev, dtype=dt)
    # g and beta are per-head scalars: [B, H, T]
    g = F.logsigmoid(torch.randn(B, H, T, device=dev, dtype=dt))
    beta = torch.sigmoid(torch.randn(B, H, T, device=dev, dtype=dt))
    S0 = torch.zeros(B, H, dk, dv, device=dev, dtype=torch.float32)
    return (q, k, v, g, beta, S0)


def _extract_gdn2_fla(x, H, dk, dv):
    B, T = x.shape[0], x.shape[1]
    dev, dt = x.device, x.dtype
    q = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    k = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    v = torch.randn(B, H, T, dv, device=dev, dtype=dt)
    g = F.logsigmoid(torch.randn(B, H, T, dk, device=dev, dtype=dt))  # == q.shape
    b = torch.sigmoid(torch.randn(B, H, T, dk, device=dev, dtype=dt))  # == q.shape
    w = torch.sigmoid(torch.randn(B, H, T, dv, device=dev, dtype=dt))  # == v.shape
    S0 = torch.zeros(B, H, dk, dv, device=dev, dtype=torch.float32)
    return (q, k, v, g, b, w, S0)


def _extract_kda_fla(x, H, dk, dv):
    B, T = x.shape[0], x.shape[1]
    dev, dt = x.device, x.dtype
    q = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    k = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    v = torch.randn(B, H, T, dv, device=dev, dtype=dt)
    g = F.logsigmoid(torch.randn(B, H, T, dk, device=dev, dtype=dt))  # == q.shape
    beta = torch.sigmoid(torch.randn(B, H, T, device=dev, dtype=dt))  # per-head scalar
    S0 = torch.zeros(B, H, dk, dv, device=dev, dtype=torch.float32)
    return (q, k, v, g, beta, S0)


def _extract_rda_fla(x, H, dk, dv):
    B, T = x.shape[0], x.shape[1]
    dev, dt = x.device, x.dtype
    scale = dk**-0.5
    q = F.elu(torch.randn(B, H, T, dk, device=dev, dtype=dt) * scale) + 1.0
    k = F.elu(torch.randn(B, H, T, dk, device=dev, dtype=dt) * scale) + 1.0
    v = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    return (q, k, v)


# ── FLA backward helper ──────────────────────────────────────────────────────


def _grad_tensors(fla_tensors):
    """Clone the FLA input tuple with requires_grad on the floating q/k/v inputs.

    Integer/state tensors and the fp32 initial_state are passed through detached;
    enabling grad on q/k/v is enough to give the kernel output a grad path.
    """
    out = []
    for i, t in enumerate(fla_tensors):
        if i < 3 and t.is_floating_point():  # q, k, v
            out.append(t.detach().clone().requires_grad_(True))
        else:
            out.append(t)
    return tuple(out)


# ── FLA forward callables ────────────────────────────────────────────────────


def _fla_gla_fwd_chunk(fl_tensors, chunk_size=64):
    return fla_gla_forward(*fl_tensors, 1.0, mode="chunk")


def _fla_gla_fwd_rccr(fl_tensors):
    return fla_gla_forward(*fl_tensors, 1.0, mode="recurrent")


def _fla_delta_fwd_chunk(fl_tensors, chunk_size=64):
    return fla_delta_rule_forward(*fl_tensors, 1.0, recurrent=False)


def _fla_delta_fwd_rccr(fl_tensors):
    return fla_delta_rule_forward(*fl_tensors, 1.0, recurrent=True)


def _fla_gated_delta_fwd_chunk(fl_tensors, chunk_size=64):
    return fla_gated_delta_rule_forward(*fl_tensors, 1.0, chunk_size, recurrent=False)


def _fla_gated_delta_fwd_rccr(fl_tensors):
    return fla_gated_delta_rule_forward(*fl_tensors, 1.0, 0, recurrent=True)


def _fla_gdn2_fwd_chunk(fl_tensors, chunk_size=64):
    return fla_gdn2_chunk_forward(*fl_tensors, 1.0, chunk_size)


def _fla_gdn2_fwd_rccr(fl_tensors):
    return fla_gdn2_forward(*fl_tensors, 1.0, 0, recurrent=True)


def _fla_kda_fwd_chunk(fl_tensors, chunk_size=64):
    return fla_kda_forward(*fl_tensors, 1.0, chunk_size, recurrent=False)


def _fla_kda_fwd_rccr(fl_tensors):
    return fla_kda_forward(*fl_tensors, 1.0, 0, recurrent=True)


def _fla_rda_fwd(fl_tensors, chunk_size=None):
    # fused_recurrent_linear_attn has no chunk variant; returns a single tensor.
    out = fla_recurrent_linear_attn_forward(*fl_tensors, eps=1e-6)
    return out, None


# ── FLA availability check helpers ───────────────────────────────────────────


def _avail_gla(tensors):
    return can_use_fla_gla(*tensors)


def _avail_delta(tensors):
    return can_use_fla_delta_rule(*tensors)


def _avail_delta_rccr(tensors):
    return can_use_fla_recurrent_delta_rule(*tensors)


def _avail_gated_delta_chunk(tensors):
    # chunk kernel requires chunk_size == 64
    return can_use_fla_gated_delta_rule(*tensors, 64, recurrent=False)


def _avail_gated_delta_rccr(tensors):
    return can_use_fla_gated_delta_rule(*tensors, 0, recurrent=True)


def _avail_gdn2_chunk(tensors):
    # chunk kernel requires chunk_size == 64
    return can_use_fla_gdn2_chunk(*tensors, 64)


def _avail_gdn2_rccr(tensors):
    return can_use_fla_gdn2(*tensors, 0, recurrent=True)


def _avail_kda(tensors):
    # chunk kernel requires chunk_size in (32, 64)
    return can_use_fla_kda(*tensors, 64)


def _avail_rda(tensors):
    return can_use_fla_linear_attn(*tensors)


# ── Benchmark case ───────────────────────────────────────────────────────────


def bench_case(
    bname,
    build_fn,
    extract_fn,
    custom_fwd_fn,
    custom_bwd_fn,
    fla_fwd_fn,
    fla_avail_fn,
    fla_rccr_fn,
    fla_rccr_avail_fn,
    B,
    T,
    dk,
    dv,
    dtype,
    chunk_size,
    backward,
    warmup,
    iters,
    check,
    no_fla,
):
    """Run one benchmark case for one backend."""
    H = 8  # default; overridden from config
    device = "cuda"

    torch.manual_seed(42)

    # ── Build model ──────────────────────────────────────────────────────
    model = build_fn(B, T, 8, dk, dv, dtype, chunk_size)
    model.eval()

    # ── Generate inputs ──────────────────────────────────────────────────
    D = 8 * dk
    if bname == "rda":
        x = torch.randn(B, T, D, device=device, dtype=dtype)
        inputs = {"query": x, "key": x, "value": x}
    elif bname == "gla":
        x = torch.randn(B, T, D, device=device, dtype=dtype)
        inputs = {"query": x, "key": x, "value": x}
    elif bname == "deltanet":
        x = torch.randn(B, T, D, device=device, dtype=dtype)
        inputs = {"query": x, "key": x, "value": x}
    else:  # gated_delta, gated_deltanet2, kimi
        x = torch.randn(B, T, D, device=device, dtype=dtype)
        inputs = {"x": x}

    # ── FLA tensors ──────────────────────────────────────────────────────
    fla_tensors = None
    fla_avail = fla_avail_rccr = False
    if not no_fla and _HAS_FLA:
        fla_tensors = extract_fn(x, 8, dk, dv)
        if fla_avail_fn:
            fla_avail = fla_avail_fn(fla_tensors)
        if fla_rccr_avail_fn:
            fla_avail_rccr = fla_rccr_avail_fn(fla_tensors)

    # ── Time custom ──────────────────────────────────────────────────────
    if backward:
        custom_ms = _time_ms(custom_bwd_fn, warmup, iters)
    else:
        custom_ms = _time_ms(custom_fwd_fn, warmup, iters)

    # ── Time FLA chunk ───────────────────────────────────────────────────
    fla_chunk_ms = fla_rccr_ms = None
    if not no_fla and _HAS_FLA:
        if fla_avail and fla_tensors and fla_fwd_fn:

            def call_chunk():
                fla_fwd_fn(fla_tensors, chunk_size)

            if backward:

                def call_chunk_bwd():
                    out, _ = fla_fwd_fn(fla_tensors, chunk_size)
                    out.backward(torch.randn_like(out), retain_graph=True)
                    out.grad = None

                fla_chunk_ms = _time_ms(call_chunk_bwd, warmup, iters)
            else:
                fla_chunk_ms = _time_ms(call_chunk, warmup, iters)

        if fla_avail_rccr and fla_tensors and fla_rccr_fn:

            def call_rccr():
                fla_rccr_fn(fla_tensors)

            if backward:

                def call_rccr_bwd():
                    out, _ = fla_rccr_fn(fla_tensors)
                    out.backward(torch.randn_like(out), retain_graph=True)
                    out.grad = None

                fla_rccr_ms = _time_ms(call_rccr_bwd, warmup, iters)
            else:
                fla_rccr_ms = _time_ms(call_rccr, warmup, iters)

    # ── Shape check ──────────────────────────────────────────────────────
    status = ""
    if check and not no_fla and fla_tensors:
        try:
            Bf, Hf, Tf, Df = fla_tensors[0].shape
            if fla_avail and fla_fwd_fn:
                y_f, _ = fla_fwd_fn(fla_tensors, chunk_size)
                ok = y_f.shape == (Bf, Hf, Tf, Df)
                status += f"chunk={'ok' if ok else 'FAIL'}"
            if fla_avail_rccr and fla_rccr_fn:
                y_f2, _ = fla_rccr_fn(fla_tensors)
                ok = y_f2.shape == (Bf, Hf, Tf, Df)
                sep = " | " if fla_avail else ""
                status += f"{sep}rccr={'ok' if ok else 'FAIL'}"
        except Exception as e:
            status += f"err={e}"

    # ── Print ────────────────────────────────────────────────────────────
    mode = "fwd+bwd" if backward else "fwd"
    tokens = B * T
    custom_tps = tokens / (custom_ms * 1e-3) if custom_ms > 0 else 0

    def _fmt(ms, label):
        if ms is None:
            return f"{label:12s} {'N/A':>8s}"
        return f"{label:12s} {ms:7.3f}ms"

    line = (
        f"{mode:7s} {bname:16s} "
        f"B={B} H=8 T={T:5d} dk={dk:3d} dv={dv:3d} {str(dtype).split('.')[-1]:5s}  "
        f"{custom_ms:7.3f}ms ({custom_tps:8.0f} tok/s)  "
        f"{_fmt(fla_chunk_ms, 'fla_chunk=')[-11:]:>11s}  "
        f"{_fmt(fla_rccr_ms, 'fla_rccr=')[-11:]:>11s}"
    )
    if status:
        line += f"  [{status}]"
    print(line)


# ── Backend registry ─────────────────────────────────────────────────────────

_BACKENDS = {
    "rda": {
        # fused_recurrent_linear_attn is itself the recurrent kernel; report it
        # once under fla_chunk and leave fla_rccr empty.
        "build": _build_rda,
        "extract": _extract_rda_fla,
        "fwd": _fla_rda_fwd,
        "avail": _avail_rda,
        "rccr_fn": None,
        "rccr_avail": None,
    },
    "gla": {
        "build": _build_gla,
        "extract": _extract_gla_fla,
        "fwd": _fla_gla_fwd_chunk,
        "avail": _avail_gla,
        "rccr_fn": _fla_gla_fwd_rccr,
        "rccr_avail": _avail_gla,
    },
    "deltanet": {
        "build": _build_deltanet,
        "extract": _extract_delta_fla,
        "fwd": _fla_delta_fwd_chunk,
        "avail": _avail_delta,
        "rccr_fn": _fla_delta_fwd_rccr,
        "rccr_avail": _avail_delta_rccr,
    },
    "gated_delta": {
        "build": _build_gated_delta,
        "extract": _extract_gated_delta_fla,
        "fwd": _fla_gated_delta_fwd_chunk,
        "avail": _avail_gated_delta_chunk,
        "rccr_fn": _fla_gated_delta_fwd_rccr,
        "rccr_avail": _avail_gated_delta_rccr,
    },
    "gated_deltanet2": {
        "build": _build_gdn2,
        "extract": _extract_gdn2_fla,
        "fwd": _fla_gdn2_fwd_chunk,
        "avail": _avail_gdn2_chunk,
        "rccr_fn": _fla_gdn2_fwd_rccr,
        "rccr_avail": _avail_gdn2_rccr,
    },
    "kimi": {
        "build": _build_kimi,
        "extract": _extract_kda_fla,
        "fwd": _fla_kda_fwd_chunk,
        "avail": _avail_kda,
        "rccr_fn": _fla_kda_fwd_rccr,
        "rccr_avail": _avail_kda,
    },
}


# ── Default configs ──────────────────────────────────────────────────────────


def _default_configs():
    configs = []
    for B in [1, 4]:
        for T in [128, 512, 2048]:
            for dk in [64, 128]:
                configs.append(
                    dict(B=B, T=T, dk=dk, dv=dk, dtype=torch.bfloat16, chunk_size=64)
                )
    return configs


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all linear attention backends vs FLA kernels"
    )
    parser.add_argument(
        "--backend", type=str, default=None, help="Backend name (default: all)"
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--dk", type=int, default=None)
    parser.add_argument("--dv", type=int, default=None)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-fla", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--cases", type=int, default=32)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    # Determine backends
    backend_names = [args.backend] if args.backend else sorted(_BACKENDS.keys())

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dt = dtype_map.get(args.dtype, torch.bfloat16)

    # Determine configs
    if args.sweep:
        all_configs = _default_configs()[: args.cases]
    else:
        H = args.heads or 8
        dk = args.dk or 128
        all_configs = [
            {
                "B": args.batch or 4,
                "T": args.seq or 512,
                "dk": dk,
                "dv": args.dv or dk,
                "dtype": dt,
                "chunk_size": args.chunk_size,
                "H": H,
            }
        ]

    print("=" * 100)
    print("Linear Attention Benchmark: Custom vs FLA Kernels")
    print("=" * 100)
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"FLA available: {_HAS_FLA} ({fla_path() if _HAS_FLA else 'N/A'})")
    print(f"Backends: {', '.join(backend_names)}")
    print(
        f"Backward: {args.backward} | Check: {not args.no_check} | Iters: {args.iters}"
    )
    print("-" * 100)

    total = len(backend_names) * len(all_configs)
    idx = 0

    for bname in backend_names:
        bi = _BACKENDS[bname]
        print(f"\n{'=' * 100}\n  Backend: {bname}\n{'=' * 100}")
        print(
            f"{'mode':7s} {'backend':16s} {'B':>3s} {'H':>2s} {'T':>5s} {'dk':>3s} {'dv':>3s} {'dtype':>5s}  "
            f"{'custom':>10s} {'tok/s':>7s} {'fla_chunk':>11s} {'fla_rccr':>11s}"
        )
        print("-" * 100)

        for c in all_configs:
            idx += 1
            H = c.get("H", 8)
            dk, dv = c["dk"], c.get("dv", c["dk"])
            dtype = c["dtype"]
            cs = c["chunk_size"]

            print(
                f"\n--- Case {idx}/{total}: B={c['B']} H={H} T={c['T']} dk={dk} dv={dv} {dtype} ---"
            )

            # Rebuild forward/bwd callables per case (model is rebuilt each time)
            # We'll create them inside bench_case by rebuilding the model
            # To avoid passing model around, we'll modify bench_case to accept build_fn directly

            model = bi["build"](c["B"], c["T"], H, dk, dv, dtype, cs)
            model.eval()

            D = H * dk
            x = torch.randn(c["B"], c["T"], D, device="cuda", dtype=dtype)
            if bname in _STANDALONE:
                inputs = {"x": x}
            else:
                # rda / gla / deltanet / kimi take query/key/value
                inputs = {"query": x, "key": x, "value": x}

            custom_fwd = _make_custom_fwd(model, bname, x, inputs)
            custom_bwd = _make_custom_bwd(model, bname, x, inputs)

            # Extract FLA tensors
            fla_tensors = None
            fla_avail = fla_avail_rccr = False
            if not args.no_fla and _HAS_FLA:
                fla_tensors = bi["extract"](x, H, dk, dv)
                if bi["avail"]:
                    fla_avail = bi["avail"](fla_tensors)
                if bi["rccr_avail"]:
                    fla_avail_rccr = bi["rccr_avail"](fla_tensors)

            # Time custom
            if args.backward:
                custom_ms = _time_ms(custom_bwd, args.warmup, args.iters)
            else:
                custom_ms = _time_ms(custom_fwd, args.warmup, args.iters)

            # Time FLA
            fla_chunk_ms = fla_rccr_ms = None
            fla_notes = []

            def _safe_time(fn, label):
                # Some FLA kernels (e.g. fused_recurrent backward) fail to compile
                # on this Triton/GPU. Report N/A instead of aborting the sweep.
                try:
                    return _time_ms(fn, args.warmup, args.iters)
                except Exception as e:
                    fla_notes.append(f"{label}: {type(e).__name__}")
                    return None

            if not args.no_fla and _HAS_FLA and fla_tensors:
                if fla_avail and bi["fwd"]:

                    def cfn(fla_tensors=fla_tensors, chunk_size=cs, fwd_fn=bi["fwd"]):
                        fwd_fn(fla_tensors, chunk_size)

                    if args.backward:

                        def cfn_bwd(
                            fla_tensors=fla_tensors, chunk_size=cs, fwd_fn=bi["fwd"]
                        ):
                            out, _ = fwd_fn(_grad_tensors(fla_tensors), chunk_size)
                            out.backward(torch.randn_like(out))

                        fla_chunk_ms = _safe_time(cfn_bwd, "chunk_bwd")
                    else:
                        fla_chunk_ms = _safe_time(cfn, "chunk")

                if fla_avail_rccr and bi["rccr_fn"]:

                    def rfn(fla_tensors=fla_tensors, fwd_fn=bi["rccr_fn"]):
                        fwd_fn(fla_tensors)

                    if args.backward:

                        def rfn_bwd(fla_tensors=fla_tensors, fwd_fn=bi["rccr_fn"]):
                            out, _ = fwd_fn(_grad_tensors(fla_tensors))
                            out.backward(torch.randn_like(out))

                        fla_rccr_ms = _safe_time(rfn_bwd, "rccr_bwd")
                    else:
                        fla_rccr_ms = _safe_time(rfn, "rccr")

            # Shape check
            status = ""
            if args.no_check is False and not args.no_fla and fla_tensors:
                try:
                    Bf, Hf, Tf, _ = fla_tensors[0].shape
                    Vf = fla_tensors[2].shape[-1]  # output last dim == v's head dim
                    expected = (Bf, Hf, Tf, Vf)
                    if fla_avail and bi["fwd"]:
                        y_f, _ = bi["fwd"](fla_tensors, cs)
                        status += f"chunk={'ok' if y_f.shape == expected else 'FAIL'}"
                    if fla_avail_rccr and bi["rccr_fn"]:
                        y_f2, _ = bi["rccr_fn"](fla_tensors)
                        sep = " | " if fla_avail else ""
                        status += (
                            f"{sep}rccr={'ok' if y_f2.shape == expected else 'FAIL'}"
                        )
                except Exception as e:
                    status += f"err={e}"

            if fla_notes:
                status += (" | " if status else "") + "; ".join(fla_notes)

            mode = "fwd+bwd" if args.backward else "fwd"
            tokens = c["B"] * c["T"]
            custom_tps = tokens / (custom_ms * 1e-3) if custom_ms > 0 else 0

            def _fmt(ms, label):
                if ms is None:
                    return f"{label:12s} {'N/A':>8s}"
                return f"{label:12s} {ms:7.3f}ms"

            line = (
                f"{mode:7s} {bname:16s} "
                f"B={c['B']} H={H} T={c['T']:5d} dk={dk:3d} dv={dv:3d} {str(dtype).split('.')[-1]:5s}  "
                f"{custom_ms:7.3f}ms ({custom_tps:8.0f} tok/s)  "
                f"{_fmt(fla_chunk_ms, 'fla_chunk=')[-11:]:>11s}  "
                f"{_fmt(fla_rccr_ms, 'fla_rccr=')[-11:]:>11s}"
            )
            if status:
                line += f"  [{status}]"
            print(line)

    print(f"\n{'=' * 100}\nDone.")


if __name__ == "__main__":
    main()
