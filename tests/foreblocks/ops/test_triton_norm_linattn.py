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

from foreblocks.ops.attention.chunked_causal_linear_attention import (
    chunked_causal_linear_attn,
    fused_recurrent_causal_linear_attn,
)
from foreblocks.ops.attention.fla_backend import (
    fla_path,
    has_fla_checkout,
    is_fla_available,
)
from foreblocks.ops.attention.fla_delta_rule import (
    can_use_fla_recurrent_delta_rule,
    fla_recurrent_delta_rule,
)
from foreblocks.ops.attention.fla_gated_delta_rule import (
    can_use_fla_gated_delta_rule,
    fla_gated_delta_rule_forward,
)
from foreblocks.ops.attention.fla_gdn2 import (
    can_use_fla_gdn2_chunk,
    fla_gdn2_chunk_forward,
)
from foreblocks.ops.attention.fla_gla import fla_gla_forward
from foreblocks.ops.attention.fla_kda import (
    can_use_fla_kda,
    fla_kda_forward,
)
from foreblocks.ops.attention.fla_linear_attention import (
    fla_recurrent_linear_attn_forward,
)
from foreblocks.ops.attention.fused_norm_gate import (
    fused_rmsnorm_sigmoid_gate,
)
from foreblocks.ops.kernels.layer_norm import LayerNormTritonFunction
from foreblocks.ops.kernels.rms_norm import (
    RMSNormTritonFunction,
    fused_add_rmsnorm,
)

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
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


def test_layernorm_wide_hidden_matches_torch():
    torch.manual_seed(10)
    M, N = 8, 4096
    x = torch.randn(M, N, device=DEV, dtype=torch.float16, requires_grad=True)
    w = torch.randn(N, device=DEV, dtype=torch.float16, requires_grad=True)
    b = torch.randn(N, device=DEV, dtype=torch.float16, requires_grad=True)
    y = LayerNormTritonFunction.apply(x, w, b, 1e-5)

    xr, wr, br = (t.detach().clone().requires_grad_(True) for t in (x, w, b))
    yr = torch.nn.functional.layer_norm(xr, (N,), wr, br, 1e-5)
    torch.testing.assert_close(y, yr, atol=2e-3, rtol=1e-3)

    g = torch.randn_like(y)
    y.backward(g)
    yr.backward(g)
    torch.testing.assert_close(x.grad, xr.grad, atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(w.grad, wr.grad, atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(b.grad, br.grad, atol=2e-3, rtol=1e-3)


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


def test_rmsnorm_wide_hidden_matches_reference():
    torch.manual_seed(11)
    M, N = 8, 4096
    x = torch.randn(M, N, device=DEV, dtype=torch.float16, requires_grad=True)
    w = torch.randn(N, device=DEV, dtype=torch.float16, requires_grad=True)
    y = RMSNormTritonFunction.apply(x, w, 1e-5)

    xr, wr = (t.detach().clone().requires_grad_(True) for t in (x, w))
    yr = _rms_ref(xr, wr, 1e-5)
    torch.testing.assert_close(y, yr, atol=2e-3, rtol=1e-3)

    g = torch.randn_like(y)
    y.backward(g)
    yr.backward(g)
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-2, rtol=2e-3)
    torch.testing.assert_close(w.grad, wr.grad, atol=1e-2, rtol=2e-3)


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


def test_fla_gla_forward_matches_reference_when_available():
    if not is_fla_available("fla.ops.gla"):
        pytest.skip("flash-linear-attention GLA ops are unavailable")

    torch.manual_seed(8)
    B, H, T, D = 2, 3, 9, 16
    q = torch.randn(B, H, T, D, device=DEV, dtype=DT)
    k = torch.randn(B, H, T, D, device=DEV, dtype=DT)
    v = torch.randn(B, H, T, D, device=DEV, dtype=DT)
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, H, T, D, device=DEV, dtype=DT)
    ) / 16.0
    state = torch.randn(B, H, D, D, device=DEV, dtype=DT) * 0.01
    scale = D ** -0.5

    out, final_state = fla_gla_forward(q, k, v, g, state, scale, mode="recurrent")

    ref_state = state.clone()
    ref_out = torch.empty_like(v)
    for t in range(T):
        ref_state = torch.exp(g[:, :, t]).unsqueeze(-1) * ref_state
        ref_state = ref_state + torch.einsum("bhk,bhv->bhkv", k[:, :, t], v[:, :, t])
        ref_out[:, :, t] = torch.einsum(
            "bhk,bhkv->bhv", q[:, :, t] * scale, ref_state
        )

    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=0)
    torch.testing.assert_close(final_state, ref_state, atol=2e-4, rtol=0)


def test_fla_recurrent_linear_attention_matches_reference_when_available():
    if not is_fla_available("fla.ops.linear_attn"):
        pytest.skip("flash-linear-attention linear_attn ops are unavailable")

    torch.manual_seed(9)
    B, H, T, Fd, Dh = 2, 3, 17, 16, 32
    q = torch.rand(B, H, T, Fd, device=DEV, dtype=DT) + 0.1
    k = torch.rand(B, H, T, Fd, device=DEV, dtype=DT) + 0.1
    v = torch.randn(B, H, T, Dh, device=DEV, dtype=DT)
    out = fla_recurrent_linear_attn_forward(q, k, v, eps=1e-6)

    KV = torch.zeros(B, H, Fd, Dh, device=DEV, dtype=DT)
    ks = torch.zeros(B, H, Fd, device=DEV, dtype=DT)
    ref = torch.empty(B, H, T, Dh, device=DEV, dtype=DT)
    for t in range(T):
        KV = KV + k[:, :, t, :, None] * v[:, :, t, None, :]
        ks = ks + k[:, :, t, :]
        numer = (KV * q[:, :, t, :, None]).sum(2)
        denom = (q[:, :, t, :] * ks).sum(-1, keepdim=True) + 1e-10
        ref[:, :, t, :] = numer / denom

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)


def test_fla_gated_delta_rule_matches_reference_when_available():
    if not is_fla_available("fla.ops.gated_delta_rule"):
        pytest.skip("flash-linear-attention gated_delta_rule ops are unavailable")

    torch.manual_seed(10)
    B, H, T, Dk, Dv = 2, 3, 9, 16, 24
    q = torch.randn(B, H, T, Dk, device=DEV, dtype=DT)
    k = torch.randn(B, H, T, Dk, device=DEV, dtype=DT)
    v = torch.randn(B, H, T, Dv, device=DEV, dtype=DT)
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, H, T, device=DEV, dtype=DT)
    )
    beta = torch.sigmoid(torch.randn(B, H, T, device=DEV, dtype=DT))
    state = torch.randn(B, H, Dk, Dv, device=DEV, dtype=DT) * 0.01

    assert can_use_fla_gated_delta_rule(q, k, v, g, beta, state)
    out, final_state = fla_gated_delta_rule_forward(
        q, k, v, g, beta, state, scale=1.0
    )

    ref_state = state.clone()
    ref_out = torch.empty_like(v)
    for t in range(T):
        decayed = g[:, :, t].exp().view(B, H, 1, 1) * ref_state
        pred = torch.einsum("bhkv,bhk->bhv", decayed, k[:, :, t])
        err = v[:, :, t] - pred
        write = torch.einsum("bhk,bhv->bhkv", k[:, :, t], err)
        ref_state = decayed + beta[:, :, t].view(B, H, 1, 1) * write
        ref_out[:, :, t] = torch.einsum("bhkv,bhk->bhv", ref_state, q[:, :, t])

    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=0)
    torch.testing.assert_close(final_state, ref_state, atol=2e-4, rtol=0)


def test_fla_gdn2_chunk_matches_reference_when_available():
    if not is_fla_available("fla.ops.gdn2"):
        pytest.skip("flash-linear-attention GDN-2 ops are unavailable")

    torch.manual_seed(11)
    B, H, T, Dk, Dv = 2, 3, 64, 16, 24
    q = torch.randn(B, H, T, Dk, device=DEV, dtype=DT)
    k = torch.randn(B, H, T, Dk, device=DEV, dtype=DT)
    v = torch.randn(B, H, T, Dv, device=DEV, dtype=DT)
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, H, T, Dk, device=DEV, dtype=DT)
    )
    b = torch.sigmoid(torch.randn(B, H, T, Dk, device=DEV, dtype=DT))
    w = torch.sigmoid(torch.randn(B, H, T, Dv, device=DEV, dtype=DT))
    state = torch.randn(B, H, Dk, Dv, device=DEV, dtype=torch.float32) * 0.01
    q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(DT)
    k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(DT)

    assert can_use_fla_gdn2_chunk(q, k, v, g, b, w, state, chunk_size=64)
    out, final_state = fla_gdn2_chunk_forward(
        q, k, v, g, b, w, state, scale=1.0, chunk_size=64
    )

    ref_state = state.clone()
    ref_out = torch.empty_like(v)
    for t in range(T):
        decayed = g[:, :, t].exp().unsqueeze(-1) * ref_state
        erase = b[:, :, t] * k[:, :, t]
        read = torch.einsum("bhkv,bhk->bhv", decayed, erase)
        write = w[:, :, t] * v[:, :, t] - read
        ref_state = decayed + torch.einsum("bhk,bhv->bhkv", k[:, :, t], write)
        ref_out[:, :, t] = torch.einsum("bhkv,bhk->bhv", ref_state, q[:, :, t])

    torch.testing.assert_close(out, ref_out, atol=5e-3, rtol=0)
    torch.testing.assert_close(final_state, ref_state, atol=5e-3, rtol=0)


def test_fla_kda_chunk_matches_reference_when_available():
    if not is_fla_available("fla.ops.kda"):
        pytest.skip("flash-linear-attention KDA ops are unavailable")

    torch.manual_seed(12)
    B, H, T, Dk, Dv = 2, 3, 64, 16, 24
    q = torch.randn(B, H, T, Dk, device=DEV, dtype=DT)
    k = torch.randn(B, H, T, Dk, device=DEV, dtype=DT)
    v = torch.randn(B, H, T, Dv, device=DEV, dtype=DT)
    q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(DT)
    k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(DT)
    g = torch.empty(B, H, T, Dk, device=DEV, dtype=DT).uniform_(-5.0, -0.1)
    beta = torch.sigmoid(torch.randn(B, H, T, device=DEV, dtype=DT))
    state = torch.randn(B, H, Dk, Dv, device=DEV, dtype=torch.float32) * 0.01

    assert can_use_fla_kda(q, k, v, g, beta, state, chunk_size=64)
    out, final_state = fla_kda_forward(
        q, k, v, g, beta, state, scale=1.0, chunk_size=64
    )

    ref_state = state.clone()
    ref_out = torch.empty_like(v)
    for t in range(T):
        ref_state = g[:, :, t].exp().unsqueeze(-1) * ref_state
        pred = torch.einsum("bhkv,bhk->bhv", ref_state, k[:, :, t])
        delta = beta[:, :, t].unsqueeze(-1) * (v[:, :, t] - pred)
        ref_state = ref_state + torch.einsum("bhk,bhv->bhkv", k[:, :, t], delta)
        ref_out[:, :, t] = torch.einsum("bhkv,bhk->bhv", ref_state, q[:, :, t])

    torch.testing.assert_close(out, ref_out, atol=5e-3, rtol=0)
    torch.testing.assert_close(final_state, ref_state, atol=5e-3, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
