"""The benchmark must compare actual values, state, and backward results."""

import runpy
from pathlib import Path

import pytest
import torch

from foreblocks.nn.attention.algorithms.linear import ModernLinearAttention
from foreblocks.nn.attention.algorithms.linear.kimi import _kda_chunk_parallel
from foreblocks.ops.attention.chunked_causal_linear_attention import (
    chunked_causal_linear_attn,
)


@pytest.fixture(scope="module")
def harness():
    path = (
        Path(__file__).resolve().parents[3]
        / "benchmarks/attention/bench_linear_attention.py"
    )
    return runpy.run_path(str(path))


@pytest.mark.parametrize(
    "name", ["rda", "gla", "deltanet", "gated_delta", "gated_deltanet2", "kimi"]
)
def test_reference_equations_match_native_kernels(harness, name):
    torch.manual_seed(9)
    batch, heads, length, dim = 2, 4, 7, 8
    x = torch.randn(batch, length, heads * dim)
    tensors = harness["_BACKENDS"][name]["extract"](x, heads, dim, dim)
    expected, state = harness["_reference_kernel"](name, tensors)
    module = (
        ModernLinearAttention(heads * dim, heads, backend=name, dropout=0).eval().impl
    )
    q, k, v = tensors[:3]
    if name == "rda":
        actual = chunked_causal_linear_attn(q, k, v, chunk_size=4)
        actual_state = None
    elif name == "gla":
        module.scale = 1.0
        actual, actual_state = module._chunk_forward(*tensors)
    elif name == "deltanet":
        beta, initial = tensors[3:]
        actual, actual_state = module._chunk_forward(q * dim**0.5, k, v, initial, beta)
    else:

        def flat(t):
            return t.reshape(batch * heads, length, -1)

        output = torch.empty(batch * heads, length, dim)
        if name in {"gated_delta", "kimi"}:
            g, beta, initial = tensors[3:]
            args = (flat(g).exp(), flat(beta))
        else:
            g, b, w, initial = tensors[3:]
            args = (
                flat(b).unsqueeze(-1),
                flat(w).unsqueeze(-1),
                flat(g).exp().unsqueeze(-1),
            )
        if name == "kimi":
            actual_state = _kda_chunk_parallel(
                initial.reshape(batch * heads, dim, dim),
                q,
                k,
                flat(v),
                *args,
                output,
                4,
            )
        else:
            actual_state = module._chunk_parallel(
                initial.reshape(batch * heads, dim, dim),
                flat(q),
                flat(k),
                flat(v),
                *args,
                output,
                4,
            )
        actual = output.reshape(batch, heads, length, dim)
        actual_state = actual_state.reshape(batch, heads, dim, dim)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    if state is not None:
        torch.testing.assert_close(actual_state, state, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("wrong_gradient", [False, True])
def test_checker_rejects_wrong_values_or_gradients(harness, wrong_gradient):
    x = torch.randn(1, 5, 16)
    tensors = harness["_extract_rda_fla"](x, 2, 8, 8)

    def reference(ts):
        return harness["_reference_kernel"]("rda", ts)

    def broken(ts):
        output, state = reference(ts)
        if wrong_gradient:
            output = output.detach() + sum(t.sum() * 0 for t in ts[:3])
        else:
            output = output + 1
        return output, state

    with pytest.raises(AssertionError):
        harness["_check_kernel"](
            reference, broken, tensors, wrong_gradient, torch.ones_like(tensors[2])
        )
