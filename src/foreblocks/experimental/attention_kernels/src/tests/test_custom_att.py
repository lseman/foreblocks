"""custom_att.tests.test_custom_att.

Smoke and correctness tests for custom attention kernels.
It belongs to the experimental attention kernel implementations and benchmarks area of Foreblocks.
It exposes functions such as ref, run.

"""

import math

import torch
from foreblocks.experimental.attention_kernels.custom_att import (
    flash_attn_backward_backend,
    flash_attn_func,
)


def ref(q, k, v, causal=False, scale=None):
    scale = scale or (1.0 / math.sqrt(q.shape[-1]))
    scores = q @ k.transpose(-1, -2) * scale
    if causal:
        n = q.shape[-2]
        mask = torch.ones(n, n, device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _tol(dtype):
    if dtype is torch.float32:
        return 2e-4, 2e-4
    if dtype is torch.bfloat16:
        return 5e-2, 5e-2
    return 3e-2, 3e-2


def run(dtype=torch.float16, D=64, causal=False, N=128):
    torch.manual_seed(0)
    q = torch.randn(2, 3, N, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = flash_attn_func(q, k, v, causal=causal)
    out_ref = ref(q_ref, k_ref, v_ref, causal=causal)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)

    backend = flash_attn_backward_backend(q)
    print(dtype, D, "causal=", causal, backend, "ok")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    for D in [16, 32, 64, 96, 128, 256]:
        run(torch.float16, D, False)
        run(torch.float16, D, True)
    for D in [32, 64, 128]:
        run(torch.float32, D, False, N=64)
        run(torch.float32, D, True, N=64)
    if torch.cuda.is_bf16_supported():
        run(torch.bfloat16, 64, False)
