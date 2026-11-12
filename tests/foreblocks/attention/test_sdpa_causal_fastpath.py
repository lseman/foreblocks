"""Native SDPA causality must retain eager masks, outputs, and gradients."""

from unittest.mock import patch

import pytest
import torch

from foreblocks.nn.attention.config import (
    AttentionConfig,
    AttentionShapeConfig,
    AttentionVariantConfig,
)
from foreblocks.nn.attention.multihead import MultiAttention


@pytest.mark.parametrize("query_length,key_length", [(8, 8), (3, 8), (8, 3)])
@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("mask_kind", [None, "padding", "attention"])
@pytest.mark.parametrize("causal", [True, False])
def test_sdpa_outputs_and_gradients_match_eager(
    query_length, key_length, kv_heads, mask_kind, causal
):
    torch.manual_seed(4)
    attention = (
        MultiAttention(
            AttentionConfig(
                shape=AttentionShapeConfig(
                    d_model=16, n_heads=4, n_kv_heads=kv_heads, dropout=0
                ),
                variant=AttentionVariantConfig(backend="sdpa"),
            )
        )
        .double()
        .eval()
    )
    tensors = [
        torch.randn(2, heads, length, 4, dtype=torch.float64)
        for heads, length in [
            (4, query_length),
            (kv_heads, key_length),
            (kv_heads, key_length),
        ]
    ]
    mask = torch.zeros(query_length, key_length, dtype=torch.bool)
    mask[:, -1] = True
    padding = torch.zeros(2, key_length, dtype=torch.bool)
    padding[0, -1] = True
    kwargs = dict(
        attn_mask=mask if mask_kind == "attention" else None,
        key_padding_mask=padding if mask_kind == "padding" else None,
        is_causal=causal,
        need_weights=False,
    )
    results = []
    for backend in ["sdpa", "eager"]:
        attention.attn_implementation = backend
        inputs = [t.clone().requires_grad_() for t in tensors]
        output, _ = attention._compute_attention(*inputs, **kwargs)
        gradients = torch.autograd.grad(output.square().sum(), inputs)
        results.append((output, gradients))
    torch.testing.assert_close(results[0][0], results[1][0])
    for actual, expected in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(actual, expected)


def test_unmasked_causal_sdpa_does_not_construct_dense_mask():
    attention = MultiAttention(
        AttentionConfig(
            shape=AttentionShapeConfig(d_model=16, n_heads=4, dropout=0),
            variant=AttentionVariantConfig(backend="sdpa"),
        )
    ).eval()
    query = torch.randn(2, 4, 8, 4)
    with patch.object(
        attention.kernel_dispatcher,
        "_sdpa_attention_mask",
        side_effect=AssertionError("dense mask allocated"),
    ):
        output, weights = attention._compute_attention(
            query, query, query, None, None, True, False
        )
    assert output.shape == query.shape
    assert weights is None


def test_uncached_full_forward_reaches_native_sdpa_and_matches_eager():
    attention = (
        MultiAttention(
            AttentionConfig(
                shape=AttentionShapeConfig(d_model=16, n_heads=4, dropout=0),
                variant=AttentionVariantConfig(backend="sdpa"),
            )
        )
        .double()
        .eval()
    )
    value = torch.randn(2, 8, 16, dtype=torch.float64, requires_grad=True)
    with patch.object(
        attention.kernel_dispatcher,
        "_sdpa_attention_mask",
        side_effect=AssertionError("dense mask allocated"),
    ):
        actual, _, _ = attention(value, is_causal=True)
    actual_grad = torch.autograd.grad(actual.square().sum(), value)[0]
    attention.attn_implementation = "eager"
    expected, _, _ = attention(value, is_causal=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), value)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
