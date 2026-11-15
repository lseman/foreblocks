"""Sparse window kernels must preserve dense semantics and compact KV heads."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from foreblocks.nn.attention import AttentionConfig, MultiAttention
from foreblocks.nn.attention.execution.window import SparseWindowAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("span", [None, 12])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_sparse_windows_match_dense_outputs_and_gradients(span, causal, dynamic):
    torch.manual_seed(24)
    inputs = [
        torch.randn(2, h, t, 16, device="cuda", requires_grad=True)
        for h, t in [(4, 7), (2, 17), (2, 17)]
    ]
    refs = [t.detach().clone().requires_grad_() for t in inputs]
    starts = torch.tensor([3, 8], device="cuda") if dynamic else None
    positions = torch.arange(7, device="cuda")[None, :, None]
    if dynamic:
        positions = positions + starts[:, None, None]
    distance = positions - torch.arange(17, device="cuda")[None, None, :]
    if span is None:
        keep = (distance >= 0) & (distance < 3) if causal else distance.abs() <= 1
    else:
        keep = (distance.abs() < 3) | (
            (distance.abs() < span) & (distance.abs() % 3 == 0)
        )
        if causal:
            keep &= distance >= 0
    padding = None
    blocked = None
    if dynamic:
        padding = torch.zeros(2, 17, device="cuda", dtype=torch.bool)
        padding[0] = True  # fully masked rows must have finite gradients
        padding[1, 5] = True
        blocked = torch.zeros(7, 17, device="cuda", dtype=torch.bool)
        blocked[:, 9] = True
        keep = keep & ~padding[:, None, :] & ~blocked[None]
    sparse = SparseWindowAttention()
    actual = sparse(
        *inputs,
        window=3,
        causal=causal,
        scale=0.25,
        q_start_pos=starts,
        attention_mask=blocked,
        padding_mask=padding,
        dilation=3,
        span=span,
    )
    expected = F.scaled_dot_product_attention(
        *refs, attn_mask=keep[:, None], scale=0.25, enable_gqa=True
    )
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for a, b in zip(inputs, refs):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("name", ["sliding_window", "dilated_sliding_window"])
def test_public_window_avoids_dense_mask_and_kv_repetition(name):
    cfg = AttentionConfig.from_legacy_kwargs(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        dropout=0,
        attention_type=name,
        attn_implementation="flex_attention",
        use_mla=False,
        pos_encoding_type="none",
        use_swiglu=False,
        window_size=3,
        attention_dilation=3,
        dilated_window_size=12,
    )
    module = MultiAttention(cfg).cuda().eval()
    x = torch.randn(2, 17, 64, device="cuda")
    with (
        patch.object(module, "_repeat_kv", side_effect=AssertionError("expanded KV")),
        patch.object(
            module,
            "_create_sliding_window_mask",
            side_effect=AssertionError("dense window"),
        ),
    ):
        actual = module(x, is_causal=True, need_weights=False)[0]
    assert actual.shape == x.shape


@pytest.mark.parametrize("name", ["sliding_window", "dilated_sliding_window"])
def test_public_sparse_outputs_and_parameter_gradients_match_manual(name):
    torch.manual_seed(73)
    cfg = AttentionConfig.from_legacy_kwargs(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        dropout=0,
        attention_type=name,
        attn_implementation="flex_attention",
        use_mla=False,
        pos_encoding_type="none",
        use_swiglu=False,
        window_size=3,
        attention_dilation=3,
        dilated_window_size=12,
    )
    module = MultiAttention(cfg).cuda().eval()
    x = torch.randn(2, 17, 64, device="cuda", requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    actual = module(x, is_causal=True, need_weights=False)[0]
    expected = module(ref, is_causal=True, need_weights=True)[0]
    gradient = torch.randn_like(actual)
    params = tuple(module.parameters())
    actual_grads = torch.autograd.grad(actual, (x, *params), gradient)
    expected_grads = torch.autograd.grad(expected, (ref, *params), gradient)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
