#!/usr/bin/env python3
"""Benchmark module execution separately from equivalent attention kernels.

Kernel rows use identical Q/K/V, gates, initial state, output gradients, and
precision for the eager reference and FLA. Correctness checks include values,
final state, and Q/K/V gradients when --backward is selected. Module timings
include projections, convolutions, normalization, and output gating; they are
reported separately and are never used to calculate kernel speedups.

Run with PYTHONPATH=src from the repository root.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

# ── FLA availability ─────────────────────────────────────────────────────────
from foreblocks.integrations.fla.delta_rule import (
    can_use_fla_delta_rule,
    can_use_fla_recurrent_delta_rule,
    fla_delta_rule_forward,
)
from foreblocks.integrations.fla.gated_delta_rule import (
    can_use_fla_gated_delta_rule,
    fla_gated_delta_rule_forward,
)
from foreblocks.integrations.fla.gated_deltanet2 import (
    can_use_fla_gdn2,
    can_use_fla_gdn2_chunk,
    fla_gdn2_chunk_forward,
    fla_gdn2_forward,
)
from foreblocks.integrations.fla.gated_linear import can_use_fla_gla, fla_gla_forward
from foreblocks.integrations.fla.kimi_delta import can_use_fla_kda, fla_kda_forward
from foreblocks.integrations.fla.linear import (
    can_use_fla_linear_attn,
    fla_recurrent_linear_attn_forward,
)
from foreblocks.nn.attention.algorithms.linear.deltanet import (
    DeltaNetBackend,
)
from foreblocks.nn.attention.algorithms.linear.gated_delta import (
    GatedDeltaNetBackend,
)
from foreblocks.nn.attention.algorithms.linear.gated_deltanet2 import (
    GatedDeltaNet2Backend,
)
from foreblocks.nn.attention.algorithms.linear.gla import GLABackend
from foreblocks.nn.attention.algorithms.linear.kimi import KimiAttentionBackend

# ── Custom backends ──────────────────────────────────────────────────────────
from foreblocks.nn.attention.algorithms.linear.rda import RDABackend
from foreblocks.ops.attention.chunked_causal_linear_attention import (
    can_use_fused_recurrent_linear_attn,
    chunked_causal_linear_attn,
    fused_recurrent_causal_linear_attn,
)

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
    return GatedDeltaNetBackend(
        d_model=H * dk,
        n_heads=H,
        d_key=dk,
        d_val=dv,
        chunk_size=chunk_size if chunk_size and chunk_size < T else 0,
        use_short_conv=False,
        use_mamba_gate=True,
    ).to("cuda", dtype)


def _build_gdn2(B, T, H, dk, dv, dtype, chunk_size=64):
    return GatedDeltaNet2Backend(
        d_model=H * dk,
        n_heads=H,
        d_key=dk,
        d_val=dv,
        chunk_size=chunk_size if chunk_size and chunk_size < T else 0,
        use_short_conv=False,
        eps=1e-6,
        allow_neg_eigval=False,
    ).to("cuda", dtype)


def _build_kimi(B, T, H, dk, dv, dtype, chunk_size=64):
    return KimiAttentionBackend(
        d_model=H * dk,
        n_heads=H,
        d_key=dk,
        d_val=dv,
        expand_v=1.0,
        chunk_size=chunk_size if chunk_size and chunk_size < T else 0,
        shortconv_mode="off",
    ).to("cuda", dtype)


# ── Custom forward callables ─────────────────────────────────────────────────

# Backends that expose forward_standalone(x) -> (y, state).
_STANDALONE = ("gated_delta", "gated_deltanet2")


def _make_custom_fwd(model, backend_name, x, inputs):
    if backend_name in _STANDALONE:

        def fn():
            with torch.no_grad():
                model.forward_standalone(x)

    else:
        # rda / gla / deltanet / kimi: forward(query, key, value) -> (out, _, state)
        def fn():
            with torch.no_grad():
                model(**inputs, is_causal=True)

    return fn


def _make_custom_bwd(model, backend_name, x, inputs):
    if backend_name in _STANDALONE:

        def fn():
            model.zero_grad(set_to_none=True)
            x2 = x.detach().clone().requires_grad_(True)
            y, _ = model.forward_standalone(x2)
            y.backward(torch.randn_like(y))
            x2.grad = None

    else:

        def fn():
            model.zero_grad(set_to_none=True)
            x2 = x.detach().clone().requires_grad_(True)
            out = model(query=x2, key=x2, value=x2, is_causal=True)[0]
            out.backward(torch.randn_like(out))
            x2.grad = None

    return fn


# ── FLA input extractors ─────────────────────────────────────────────────────
# Every FLA wrapper expects Foreblocks layout [B, H, T, *] and transposes to
# FLA's [B, T, H, *] internally.  Extractors below build random tensors directly
# in [B, H, T, *]. These same tensors feed both the reference and FLA paths.


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
    q = F.normalize(torch.randn(B, H, T, dk, device=dev, dtype=dt), dim=-1)
    k = F.normalize(torch.randn(B, H, T, dk, device=dev, dtype=dt), dim=-1)
    v = torch.randn(B, H, T, dk, device=dev, dtype=dt)
    beta = torch.sigmoid(torch.randn(B, H, T, 1, device=dev, dtype=dt))
    S0 = torch.zeros(B, H, dk, dk, device=dev, dtype=torch.float32)
    return (q, k, v, beta, S0)


def _extract_gated_delta_fla(x, H, dk, dv):
    B, T = x.shape[0], x.shape[1]
    dev, dt = x.device, x.dtype
    q = F.normalize(torch.randn(B, H, T, dk, device=dev, dtype=dt), dim=-1)
    k = F.normalize(torch.randn(B, H, T, dk, device=dev, dtype=dt), dim=-1)
    v = torch.randn(B, H, T, dv, device=dev, dtype=dt)
    # g and beta are per-head scalars: [B, H, T]
    g = F.logsigmoid(torch.randn(B, H, T, device=dev, dtype=dt))
    beta = torch.sigmoid(torch.randn(B, H, T, device=dev, dtype=dt))
    S0 = torch.zeros(B, H, dk, dv, device=dev, dtype=torch.float32)
    return (q, k, v, g, beta, S0)


def _extract_gdn2_fla(x, H, dk, dv):
    B, T = x.shape[0], x.shape[1]
    dev, dt = x.device, x.dtype
    q = F.normalize(torch.randn(B, H, T, dk, device=dev, dtype=dt), dim=-1)
    k = F.normalize(torch.randn(B, H, T, dk, device=dev, dtype=dt), dim=-1)
    v = torch.randn(B, H, T, dv, device=dev, dtype=dt)
    g = F.logsigmoid(torch.randn(B, H, T, dk, device=dev, dtype=dt))  # == q.shape
    b = torch.sigmoid(torch.randn(B, H, T, dk, device=dev, dtype=dt))  # == q.shape
    w = torch.sigmoid(torch.randn(B, H, T, dv, device=dev, dtype=dt))  # == v.shape
    S0 = torch.zeros(B, H, dk, dv, device=dev, dtype=torch.float32)
    return (q, k, v, g, b, w, S0)


def _extract_kda_fla(x, H, dk, dv):
    B, T = x.shape[0], x.shape[1]
    dev, dt = x.device, x.dtype
    q = F.normalize(torch.randn(B, H, T, dk, device=dev, dtype=dt), dim=-1)
    k = F.normalize(torch.randn(B, H, T, dk, device=dev, dtype=dt), dim=-1)
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


def _reference_kernel(name, tensors):
    """Independent fp32 recurrent equations, with public adapter state layouts."""
    q, k, v = (t.float() for t in tensors[:3])
    batch, heads, length, features = q.shape
    if name == "rda":
        state = q.new_zeros(batch, heads, features, v.size(-1))
        normalizer = q.new_zeros(batch, heads, features)
    else:
        state = tensors[-1].float()
        if name == "deltanet":
            state = state.transpose(-1, -2)
    outputs = []
    for t in range(length):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        if name in {"rda", "gla"}:
            if name == "gla":
                state = tensors[3][:, :, t].float().exp().unsqueeze(-1) * state
            state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        else:
            if name in {"gated_delta", "gated_deltanet2", "kimi"}:
                decay = tensors[3][:, :, t].float().exp()
                if name == "gated_delta":
                    decay = decay.unsqueeze(-1)
                state = decay.unsqueeze(-1) * state
            if name == "gated_deltanet2":
                erase = tensors[4][:, :, t].float() * kt
                target = tensors[5][:, :, t].float() * vt
                delta = target - (erase.unsqueeze(-1) * state).sum(-2)
            else:
                beta = tensors[3 if name == "deltanet" else 4][:, :, t].float()
                if beta.ndim == 2:
                    beta = beta.unsqueeze(-1)
                delta = beta * (vt - (kt.unsqueeze(-1) * state).sum(-2))
            state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        output = (qt.unsqueeze(-1) * state).sum(-2)
        if name == "rda":
            normalizer = normalizer + kt
            output = output / ((qt * normalizer).sum(-1, keepdim=True) + 1e-6)
        outputs.append(output)
    final = None if name == "rda" else state
    if name == "deltanet":
        final = final.transpose(-1, -2)
    return torch.stack(outputs, dim=2).to(tensors[0].dtype), final


def _kernel_runner(fn, tensors, backward, gradient):
    if not backward:

        def forward():
            with torch.no_grad():
                return fn(tensors)

        return forward
    inputs = _grad_tensors(tensors)

    def forward_backward():
        output, _ = fn(inputs)
        return torch.autograd.grad(output, inputs[:3], gradient)

    return forward_backward


def _check_kernel(reference, candidate, tensors, backward, gradient):
    with torch.no_grad():
        expected, expected_state = reference(tensors)
        actual, actual_state = candidate(tensors)
    tolerance = 3e-2 if tensors[0].dtype in (torch.float16, torch.bfloat16) else 2e-4
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    if expected_state is not None:
        torch.testing.assert_close(
            actual_state.float(), expected_state, atol=tolerance, rtol=tolerance
        )
    if backward:
        actual_grad = _kernel_runner(candidate, tensors, True, gradient)()
        expected_grad = _kernel_runner(reference, tensors, True, gradient)()
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(_BACKENDS))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--dk", type=int, default=128)
    parser.add_argument("--dv", type=int)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
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
        parser.error("CUDA is required")
    if (
        min(args.batch, args.heads, args.seq, args.dk, args.iters, args.cases) <= 0
        or args.warmup < 0
        or args.chunk_size < 0
    ):
        parser.error(
            "sizes/iterations must be positive; warmup/chunk-size must be nonnegative"
        )
    dv = args.dv if args.dv is not None else args.dk
    if dv <= 0:
        parser.error("dv must be positive")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.dtype
    ]
    configs = [(args.batch, args.seq, args.dk, dv)]
    if args.sweep:
        configs = [
            (b, t, d, d)
            for b in (1, 4)
            for t in (32, 64, 128, 512, 2048)
            for d in (64, 128)
        ][: args.cases]
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch={torch.__version__}")
    print(
        "Module auto-dispatch timings include projections and gates. Kernel rows compare identical inputs."
    )
    print(
        "Reference kernel is an eager sequential baseline, not an optimized performance target."
    )
    failures = []
    for name in [args.backend] if args.backend else sorted(_BACKENDS):
        backend = _BACKENDS[name]
        for batch, length, dk, value_dim in configs:
            if name in {"rda", "gla", "deltanet"} and dk != value_dim:
                parser.error(f"{name} module requires dv == dk")
            torch.manual_seed(42)
            model = backend["build"](
                batch, length, args.heads, dk, value_dim, dtype, args.chunk_size
            ).eval()
            x = torch.randn(batch, length, args.heads * dk, device="cuda", dtype=dtype)
            inputs = {"query": x, "key": x, "value": x}
            module_fn = (
                _make_custom_bwd(model, name, x, inputs)
                if args.backward
                else _make_custom_fwd(model, name, x, inputs)
            )
            module_ms = _time_ms(module_fn, args.warmup, args.iters)
            print(
                f"{name} B={batch} H={args.heads} T={length} dk={dk} dv={value_dim} {args.dtype}: module_auto={module_ms:.4f}ms"
            )
            tensors = backend["extract"](x, args.heads, dk, value_dim)
            gradient = torch.randn_like(tensors[2])

            def reference(ts):
                return _reference_kernel(name, ts)

            ref_ms = _time_ms(
                _kernel_runner(reference, tensors, args.backward, gradient),
                args.warmup,
                args.iters,
            )
            print(f"  kernel_reference={ref_ms:.4f}ms")
            candidates = []
            if name == "rda":

                def local_chunked(ts):
                    return chunked_causal_linear_attn(
                        *ts, chunk_size=max(1, args.chunk_size)
                    ), None

                candidates.append(("local_chunked", local_chunked))
                if not args.backward and can_use_fused_recurrent_linear_attn(*tensors):

                    def local_recurrent(ts):
                        return fused_recurrent_causal_linear_attn(*ts), None

                    candidates.append(("local_triton_recurrent", local_recurrent))
            fla_modes = (
                (
                    "recurrent" if name == "rda" else "chunk",
                    backend["fwd"],
                    backend["avail"],
                ),
            )
            if name != "rda":
                fla_modes += (("recurrent", backend["rccr_fn"], backend["rccr_avail"]),)
            for label, operation, available in fla_modes:
                if (
                    args.no_fla
                    or operation is None
                    or available is None
                    or not available(tensors)
                ):
                    print(f"  fla_{label}=N/A (disabled or unavailable)")
                    continue

                def candidate(ts, operation=operation, label=label):
                    return (
                        operation(ts, args.chunk_size)
                        if label == "chunk"
                        else operation(ts)
                    )

                candidates.append((f"fla_{label}", candidate))

            for label, candidate in candidates:
                try:
                    if not args.no_check:
                        _check_kernel(
                            reference, candidate, tensors, args.backward, gradient
                        )
                    elapsed = _time_ms(
                        _kernel_runner(candidate, tensors, args.backward, gradient),
                        args.warmup,
                        args.iters,
                    )
                    print(
                        f"  {label}={elapsed:.4f}ms reference_speedup={ref_ms / elapsed:.2f}x check={'disabled' if args.no_check else 'passed'}"
                    )
                except Exception as exc:
                    failures.append(f"{name}/{label}: {type(exc).__name__}: {exc}")
                    print(f"  {label}=FAILED: {exc}")
    if failures:
        raise SystemExit("Failed kernel checks/runs:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
