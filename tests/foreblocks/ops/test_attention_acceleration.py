"""CUDA parity for fused gate training, stateful scans, and short convolutions."""

import pytest
import torch
import torch.nn.functional as F

from foreblocks.kernels.attention.linear import fused_recurrent_causal_linear_attn
from foreblocks.kernels.attention.short_conv import streaming_short_conv
from foreblocks.kernels.normalization.gated import fused_rmsnorm_sigmoid_gate

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["identity", "sigmoid", "silu"])
def test_norm_gate_outputs_and_all_gradients(dtype, activation):
    torch.manual_seed(15)
    inputs = [
        torch.randn(2, 3, 17, 23, device="cuda", dtype=dtype).requires_grad_()
        for _ in range(2)
    ]
    inputs.append(
        torch.randn(3, 23, device="cuda", dtype=torch.float32, requires_grad=True)
    )
    refs = [t.detach().clone().requires_grad_() for t in inputs]
    x, raw, w = refs
    gate = raw.float()
    if activation == "sigmoid":
        gate = gate.sigmoid()
    elif activation == "silu":
        gate = F.silu(gate)
    expected = (
        x.float()
        * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        * w[None, :, None]
        * gate
    ).to(dtype)
    actual = fused_rmsnorm_sigmoid_gate(*inputs, gate_activation=activation)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    tol = 3e-2 if dtype == torch.bfloat16 else 3e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    for a, b in zip(inputs, refs):
        torch.testing.assert_close(a.grad, b.grad, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_recurrent_scan_carries_nonzero_state_without_mutating_it(dtype):
    torch.manual_seed(20)
    q, k = [torch.rand(2, 3, 9, 23, device="cuda", dtype=dtype) for _ in range(2)]
    v = torch.randn(2, 3, 9, 17, device="cuda", dtype=dtype)
    initial = (
        torch.rand(2, 3, 23, device="cuda"),
        torch.randn(2, 3, 23, 17, device="cuda"),
    )
    saved = tuple(t.clone() for t in initial)
    z, s = initial
    expected = []
    for t in range(9):
        z = z + k[:, :, t].float()
        s = s + k[:, :, t].float().unsqueeze(-1) * v[:, :, t].float().unsqueeze(-2)
        expected.append(
            (q[:, :, t].float().unsqueeze(-1) * s).sum(-2)
            / ((q[:, :, t].float() * z).sum(-1, keepdim=True) + 1e-6)
        )
    state = initial
    outputs = []
    with torch.no_grad():
        for start, end in [(0, 1), (1, 4), (4, 9)]:
            out, state = fused_recurrent_causal_linear_attn(
                q[:, :, start:end],
                k[:, :, start:end],
                v[:, :, start:end],
                initial_state=state,
                output_final_state=True,
            )
            outputs.append(out)
    tol = 2e-2 if dtype != torch.float32 else 2e-5
    torch.testing.assert_close(
        torch.cat(outputs, 2), torch.stack(expected, 2).to(dtype), atol=tol, rtol=tol
    )
    for a, b in zip(state, (z, s)):
        assert a.dtype == torch.float32
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    for a, b in zip(initial, saved):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("kernel", [1, 4])
@pytest.mark.parametrize("length", [1, 7])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_short_conv_cache_activation_and_backward(kernel, length, training, dtype):
    torch.manual_seed(23)
    inputs = [
        torch.randn(2, length, 23, device="cuda", dtype=dtype),
        torch.randn(23, kernel, device="cuda", dtype=dtype),
        torch.randn(23, device="cuda", dtype=dtype),
        torch.randn(2, 23, kernel - 1, device="cuda", dtype=dtype),
    ]
    inputs = [t.requires_grad_(training) for t in inputs]
    refs = [t.detach().clone().requires_grad_(training) for t in inputs]
    x, w, bias, history = refs
    joined = torch.cat((history, x.transpose(1, 2)), -1)
    expected = F.silu(F.conv1d(joined, w.unsqueeze(1), bias, groups=23).transpose(1, 2))
    expected_history = joined[..., -(kernel - 1) :] if kernel > 1 else joined[..., :0]
    with torch.set_grad_enabled(training):
        actual, next_history = streaming_short_conv(
            *inputs[:3], history=inputs[3], activation="silu"
        )
    tol = 8e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    torch.testing.assert_close(next_history, expected_history, atol=0, rtol=0)
    if training:
        gradient = torch.randn_like(actual)
        (actual * gradient).sum().add(next_history.sum()).backward()
        (expected * gradient).sum().add(expected_history.sum()).backward()
        for a, b in zip(inputs, refs):
            torch.testing.assert_close(a.grad, b.grad, atol=tol, rtol=tol)


@pytest.mark.parametrize("backend", ["gated_delta", "gated_deltanet2", "kimi"])
def test_linear_training_integrates_gate_and_convolution_backward(backend, monkeypatch):
    from foreblocks.nn.attention.algorithms.linear import ModernLinearAttention

    torch.manual_seed(59)
    module = (
        ModernLinearAttention(64, 4, backend=backend, dropout=0, chunk_size=4)
        .cuda()
        .eval()
    )
    value = torch.randn(2, 9, 64, device="cuda", requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    actual = module(value, value, value, is_causal=True)[0]
    params = tuple(module.parameters())
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (value, *params), grad)
    monkeypatch.setenv("FOREBLOCKS_DISABLE_TRITON_NORM_GATE", "1")
    monkeypatch.setenv("FOREBLOCKS_DISABLE_TRITON_SHORT_CONV", "1")
    expected = module(reference, reference, reference, is_causal=True)[0]
    expected_grads = torch.autograd.grad(expected, (reference, *params), grad)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=3e-4, rtol=3e-4)


def test_rda_bf16_mixed_prefill_and_decode_preserves_state():
    from foreblocks.nn.attention.algorithms.linear import ModernLinearAttention

    torch.manual_seed(61)
    module = (
        ModernLinearAttention(64, 4, backend="rda", dropout=0).cuda().bfloat16().eval()
    )
    x = torch.randn(2, 9, 64, device="cuda", dtype=torch.bfloat16)
    state, outputs = {}, []
    with torch.no_grad():
        expected = module(x, x, x, is_causal=True)[0]
        for chunk in x.split([1, 3, 1, 4], dim=1):
            output, _, state = module(
                chunk, chunk, chunk, is_causal=True, layer_state=state
            )
            outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=0.02, rtol=0.02)
    assert state["kv_sum"].dtype == torch.float32


@pytest.mark.parametrize("shape", [(2, 2, 1025, 33), (1, 2, 3, 1024)])
def test_norm_gate_backward_handles_strides_and_multiple_row_blocks(shape):
    torch.manual_seed(71)
    b, h, t, d = shape
    x = torch.randn(b, h, t * 2, d, device="cuda")[:, :, ::2].requires_grad_()
    gate = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(d, h, device="cuda").t().requires_grad_()
    refs = [a.detach().clone().requires_grad_() for a in (x, gate, weight)]
    rx, rg, rw = refs
    expected = (
        rx
        * (rx.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
        * rw[None, :, None]
        * rg.sigmoid()
    )
    actual = fused_rmsnorm_sigmoid_gate(
        x, gate, weight, 1e-5, gate_activation="sigmoid"
    )
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    for a, b in zip((x, gate, weight), refs):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-4, rtol=3e-4)
