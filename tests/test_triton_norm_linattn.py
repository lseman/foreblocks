"""GPU correctness tests for the transformer norm + linear-attention kernels.

Complements the existing kernel suites (swiglu, grouped_gemm, fused_add_rmsnorm)
by covering, against autograd/eager references:
  - LayerNormTritonFunction: forward + dX/dW/dB
  - RMSNormTritonFunction: forward + dX/dW
  - fused_add_rmsnorm: forward + dResidual/dUpdate/dW
  - chunked_causal_linear_attn: forward vs the naive inclusive-cumsum scan

Skipped when CUDA/Triton is unavailable.
"""
import pytest
import torch

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

from foreblocks.transformer.kernels.layer_norm import LayerNormTritonFunction
from foreblocks.transformer.kernels.rms_norm import (
    RMSNormTritonFunction,
    fused_add_rmsnorm,
)
from foreblocks.transformer.attention.kernels.chunked_causal_linear_attention import (
    chunked_causal_linear_attn,
    fused_recurrent_causal_linear_attn,
)
from foreblocks.transformer.attention.kernels.delta_rule import (
    can_use_fla_recurrent_delta_rule,
    fla_recurrent_delta_rule,
    fused_recurrent_delta_rule,
)
from foreblocks.transformer.attention.kernels.fla_backend import (
    fla_path,
    has_fla_checkout,
    is_fla_available,
)
from foreblocks.transformer.attention.kernels.fused_norm_gate import (
    fused_rmsnorm_sigmoid_gate,
)

DEV = "cuda"
DT = torch.float32


def _rms_ref(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def test_fla_submodule_adapter_discovers_checkout():
    path = fla_path()
    assert path.name == "flash-linear-attention"
    assert has_fla_checkout()
    assert (path / "fla").is_dir()


def test_layernorm_fwd_bwd_matches_torch():
    torch.manual_seed(0)
    M, N = 64, 256
    x = torch.randn(M, N, device=DEV, dtype=DT, requires_grad=True)
    w = torch.randn(N, device=DEV, dtype=DT, requires_grad=True)
    b = torch.randn(N, device=DEV, dtype=DT, requires_grad=True)
    y = LayerNormTritonFunction.apply(x, w, b, 1e-5)

    xr, wr, br = (t.detach().clone().requires_grad_(True) for t in (x, w, b))
    yr = torch.nn.functional.layer_norm(xr, (N,), wr, br, 1e-5)
    torch.testing.assert_close(y, yr, atol=1e-4, rtol=0)

    g = torch.randn_like(y)
    y.backward(g)
    yr.backward(g)
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-3, rtol=0)
    torch.testing.assert_close(w.grad, wr.grad, atol=1e-3, rtol=0)
    torch.testing.assert_close(b.grad, br.grad, atol=1e-3, rtol=0)


def test_rmsnorm_fwd_bwd_matches_reference():
    torch.manual_seed(1)
    M, N = 64, 256
    x = torch.randn(M, N, device=DEV, dtype=DT, requires_grad=True)
    w = torch.randn(N, device=DEV, dtype=DT, requires_grad=True)
    y = RMSNormTritonFunction.apply(x, w, 1e-5)

    xr, wr = (t.detach().clone().requires_grad_(True) for t in (x, w))
    yr = _rms_ref(xr, wr, 1e-5)
    torch.testing.assert_close(y, yr, atol=1e-4, rtol=0)

    g = torch.randn_like(y)
    y.backward(g)
    yr.backward(g)
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-3, rtol=0)
    torch.testing.assert_close(w.grad, wr.grad, atol=1e-3, rtol=0)


def test_fused_add_rmsnorm_fwd_bwd_matches_reference():
    torch.manual_seed(2)
    M, N = 64, 256
    r = torch.randn(M, N, device=DEV, dtype=DT, requires_grad=True)
    u = torch.randn(M, N, device=DEV, dtype=DT, requires_grad=True)
    w = torch.randn(N, device=DEV, dtype=DT, requires_grad=True)
    y = fused_add_rmsnorm(r, u, w, 1e-5)

    rr, ur, wr = (t.detach().clone().requires_grad_(True) for t in (r, u, w))
    yr = _rms_ref(rr + ur, wr, 1e-5)
    torch.testing.assert_close(y, yr, atol=1e-4, rtol=0)

    g = torch.randn_like(y)
    y.backward(g)
    yr.backward(g)
    torch.testing.assert_close(r.grad, rr.grad, atol=1e-3, rtol=0)
    torch.testing.assert_close(u.grad, ur.grad, atol=1e-3, rtol=0)
    torch.testing.assert_close(w.grad, wr.grad, atol=1e-3, rtol=0)


def test_causal_linear_attention_matches_naive_scan():
    torch.manual_seed(3)
    B, H, T, Fd, Dh = 2, 3, 9, 8, 16
    q = torch.rand(B, H, T, Fd, device=DEV) + 0.1  # feature-mapped (positive)
    k = torch.rand(B, H, T, Fd, device=DEV) + 0.1
    v = torch.randn(B, H, T, Dh, device=DEV)
    out = chunked_causal_linear_attn(q, k, v, eps=1e-6)

    KV = torch.zeros(B, H, Fd, Dh, device=DEV)
    ks = torch.zeros(B, H, Fd, device=DEV)
    ref = torch.empty(B, H, T, Dh, device=DEV)
    for t in range(T):  # inclusive cumsum: output[t] uses k[0..t], v[0..t]
        KV = KV + k[:, :, t, :, None] * v[:, :, t, None, :]
        ks = ks + k[:, :, t, :]
        numer = (KV * q[:, :, t, :, None]).sum(2)
        denom = (q[:, :, t, :] * ks).sum(-1, keepdim=True) + 1e-6
        ref[:, :, t, :] = numer / denom
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)


def test_fused_recurrent_causal_linear_attention_matches_chunked():
    torch.manual_seed(4)
    B, H, T, Fd, Dh = 2, 3, 17, 16, 32
    q = torch.rand(B, H, T, Fd, device=DEV) + 0.1
    k = torch.rand(B, H, T, Fd, device=DEV) + 0.1
    v = torch.randn(B, H, T, Dh, device=DEV)
    with torch.enable_grad():
        ref = chunked_causal_linear_attn(q, k, v, chunk_size=8, eps=1e-6)
    with torch.no_grad():
        out = fused_recurrent_causal_linear_attn(q, k, v, eps=1e-6)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)


def test_fused_rmsnorm_sigmoid_gate_matches_reference():
    torch.manual_seed(5)
    B, H, T, D = 2, 3, 7, 32
    x = torch.randn(B, H, T, D, device=DEV, dtype=DT)
    gate = torch.sigmoid(torch.randn(B, H, T, D, device=DEV, dtype=DT))
    weight = torch.randn(H, D, device=DEV, dtype=DT)
    out = fused_rmsnorm_sigmoid_gate(x, gate, weight, eps=1e-6)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = ref * weight.view(1, H, 1, D) * gate
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)


def test_fused_recurrent_delta_rule_matches_reference():
    torch.manual_seed(6)
    B, H, T, D = 2, 3, 9, 16
    q = torch.randn(B, H, T, D, device=DEV, dtype=DT)
    k = torch.randn(B, H, T, D, device=DEV, dtype=DT)
    v = torch.randn(B, H, T, D, device=DEV, dtype=DT)
    beta = torch.sigmoid(torch.randn(B, H, T, 1, device=DEV, dtype=DT))
    state = torch.randn(B, H, D, D, device=DEV, dtype=DT) * 0.01

    out, final_state = fused_recurrent_delta_rule(q, k, v, beta, state, D ** -0.5)

    ref_state = state.clone()
    ref_out = torch.empty_like(v)
    for t in range(T):
        pred = torch.einsum("bhvk,bhk->bhv", ref_state, k[:, :, t])
        delta = beta[:, :, t] * (v[:, :, t] - pred)
        ref_state = ref_state + torch.einsum("bhv,bhk->bhvk", delta, k[:, :, t])
        ref_out[:, :, t] = torch.einsum(
            "bhvk,bhk->bhv", ref_state, q[:, :, t] * (D ** -0.5)
        )

    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=0)
    torch.testing.assert_close(final_state, ref_state, atol=2e-4, rtol=0)


def test_fla_recurrent_delta_rule_matches_reference_when_available():
    if not is_fla_available("fla.ops.delta_rule"):
        pytest.skip("flash-linear-attention delta_rule ops are unavailable")

    torch.manual_seed(7)
    B, H, T, D = 2, 3, 9, 16
    q = torch.randn(B, H, T, D, device=DEV, dtype=DT)
    k = torch.randn(B, H, T, D, device=DEV, dtype=DT)
    v = torch.randn(B, H, T, D, device=DEV, dtype=DT)
    beta = torch.sigmoid(torch.randn(B, H, T, 1, device=DEV, dtype=DT))
    state = torch.randn(B, H, D, D, device=DEV, dtype=DT) * 0.01

    assert can_use_fla_recurrent_delta_rule(q, k, v, beta, state)
    out, final_state = fla_recurrent_delta_rule(q, k, v, beta, state, D ** -0.5)

    ref_state = state.clone()
    ref_out = torch.empty_like(v)
    for t in range(T):
        pred = torch.einsum("bhvk,bhk->bhv", ref_state, k[:, :, t])
        delta = beta[:, :, t] * (v[:, :, t] - pred)
        ref_state = ref_state + torch.einsum("bhv,bhk->bhvk", delta, k[:, :, t])
        ref_out[:, :, t] = torch.einsum(
            "bhvk,bhk->bhv", ref_state, q[:, :, t] * (D ** -0.5)
        )

    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=0)
    torch.testing.assert_close(final_state, ref_state, atol=2e-4, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
