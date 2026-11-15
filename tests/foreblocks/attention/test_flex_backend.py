"""Exercise the registered FlexAttention runner against SDPA on CUDA."""

import pytest
import torch
import torch.nn.functional as F

from foreblocks.nn.attention.execution.backends import _flex_attention_runner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_flex_outputs_and_gradients_match_sdpa(causal, masked):
    pytest.importorskip("torch.nn.attention.flex_attention")
    torch.manual_seed(5)
    tensors = [
        torch.randn(2, h, 13, 16, device="cuda", requires_grad=True) for h in (4, 2, 2)
    ]
    reference = [x.detach().clone().requires_grad_() for x in tensors]
    blocked = None
    if masked:
        blocked = torch.zeros(2, 4, 13, 13, device="cuda", dtype=torch.bool)
        blocked[..., 3] = True
        if causal:
            blocked |= torch.ones(13, 13, device="cuda", dtype=torch.bool).triu(1)
    actual = _flex_attention_runner(*tensors, attention_mask=blocked, causal=causal)
    expected = F.scaled_dot_product_attention(
        *reference,
        attn_mask=~blocked if blocked is not None else None,
        is_causal=causal and blocked is None,
        enable_gqa=True,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    for x, ref in zip(tensors, reference):
        torch.testing.assert_close(x.grad, ref.grad, atol=3e-5, rtol=3e-5)


def test_flex_training_dropout_preserves_sdpa_semantics():
    q, k, v = [torch.randn(1, 2, 7, 16, device="cuda") for _ in range(3)]
    torch.manual_seed(10)
    actual = _flex_attention_runner(q, k, v, causal=True, dropout_p=0.25)
    torch.manual_seed(10)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.25)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("d_model", [32, 64])
def test_public_flex_backend_matches_sdpa(d_model):
    from foreblocks.nn.attention import (
        AttentionCacheConfig,
        AttentionConfig,
        AttentionPositionConfig,
        AttentionShapeConfig,
        AttentionVariantConfig,
        MultiAttention,
    )

    def build(backend):
        return (
            MultiAttention(
                AttentionConfig(
                    shape=AttentionShapeConfig(d_model=d_model, n_heads=4, dropout=0),
                    cache=AttentionCacheConfig(use_mla=False),
                    position=AttentionPositionConfig(encoding="none"),
                    variant=AttentionVariantConfig(backend=backend, use_swiglu=False),
                )
            )
            .cuda()
            .eval()
        )

    torch.manual_seed(11)
    flex = build("flex_attention")
    sdpa = build("sdpa")
    sdpa.load_state_dict(flex.state_dict())
    x = torch.randn(2, 13, d_model, device="cuda")
    padding = torch.zeros(2, 13, device="cuda", dtype=torch.bool)
    padding[:, -2:] = True
    actual = flex(x, key_padding_mask=padding, is_causal=True, need_weights=False)[0]
    expected = sdpa(x, key_padding_mask=padding, is_causal=True, need_weights=False)[0]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
