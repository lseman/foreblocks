"""Public mask, checkpoint, and streaming contracts for linear attention."""

import pytest
import torch

from foreblocks.nn.attention.algorithms.linear import ModernLinearAttention

BACKENDS = ("rda", "gla", "deltanet", "gated_delta", "gated_deltanet2", "kimi")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("splits", [(1,) * 13, (5, 3, 1, 4)])
def test_streamed_outputs_and_reset_match_full(backend, splits, device):
    torch.manual_seed(42)
    module = (
        ModernLinearAttention(32, 4, backend=backend, dropout=0, chunk_size=4)
        .to(device)
        .eval()
    )
    x = torch.randn(2, 13, 32, device=device)
    with torch.no_grad():
        expected = module(x, x, x, is_causal=True)[0]
        state = {"unrelated": "preserve"}
        outputs = []
        for chunk in x.split(splits, dim=1):
            output, _, returned = module(
                chunk, chunk, chunk, is_causal=True, layer_state=state
            )
            state = returned if returned is not None else state
            outputs.append(output)
        torch.testing.assert_close(
            torch.cat(outputs, dim=1), expected, atol=2e-5, rtol=2e-5
        )
        module.reset_state(state)
        assert state == {"unrelated": "preserve"}
        actual = module(x, x, x, is_causal=True, layer_state=state)[0]
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("backend", ["rda", "gla"])
def test_streaming_rope_preserves_absolute_positions(backend):
    torch.manual_seed(13)
    module = ModernLinearAttention(
        32, 4, backend=backend, dropout=0, pos_encoding_type="rope"
    ).eval()
    x = torch.randn(2, 11, 32)
    with torch.no_grad():
        expected = module(x, x, x, is_causal=True)[0]
        state = {}
        outputs = []
        for chunk in x.split([5, 1, 5], dim=1):
            output, _, state = module(
                chunk, chunk, chunk, is_causal=True, layer_state=state
            )
            outputs.append(output)
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), expected, atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_streaming_preserves_gradients_across_chunk_boundaries(backend):
    torch.manual_seed(7)
    module = ModernLinearAttention(
        32, 4, backend=backend, dropout=0, mode="recurrent"
    ).eval()
    x = torch.randn(2, 9, 32, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    expected = module(reference, reference, reference, is_causal=True)[0]
    state = {}
    outputs = []
    for chunk in x.split([2, 3, 4], dim=1):
        output, _, returned = module(
            chunk, chunk, chunk, is_causal=True, layer_state=state
        )
        state = returned if returned is not None else state
        outputs.append(output)
    # Only the final token contributes to the loss: earlier-token gradients
    # must travel through both recurrent memory and convolution history.
    actual = torch.cat(outputs, dim=1)
    actual[:, -1].square().sum().backward()
    expected[:, -1].square().sum().backward()
    torch.testing.assert_close(x.grad, reference.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_arbitrary_mask_is_rejected_and_causal_mask_is_supported(backend):
    module = ModernLinearAttention(32, 4, backend=backend, dropout=0).eval()
    x = torch.randn(2, 7, 32)
    with pytest.raises(ValueError, match="arbitrary attn_mask"):
        module(x, x, x, attn_mask=torch.ones(7, 7, dtype=torch.bool))
    causal = torch.ones(7, 7, dtype=torch.bool).triu(1)
    with torch.no_grad():
        expected = module(x, x, x, is_causal=True)[0]
        actual = module(x, x, x, is_causal=True, attn_mask=causal)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["deltanet", "kimi"])
def test_unsupported_padding_is_explicit(backend):
    module = ModernLinearAttention(32, 4, backend=backend)
    x = torch.randn(2, 7, 32)
    with pytest.raises(ValueError, match="key_padding_mask"):
        module(x, x, x, key_padding_mask=torch.zeros(2, 7, dtype=torch.bool))


def test_gla_padding_excludes_masked_keys():
    torch.manual_seed(4)
    module = ModernLinearAttention(32, 4, backend="gla", dropout=0).eval()
    q, k, v = (torch.randn(2, 7, 32) for _ in range(3))
    padding = torch.zeros(2, 7, dtype=torch.bool)
    padding[:, 2:4] = True
    with torch.no_grad():
        expected = module(q, k, v, key_padding_mask=padding)[0]
        k[:, 2:4] += 100
        v[:, 2:4] -= 100
        actual = module(q, k, v, key_padding_mask=padding)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_rff_projection_moves_and_roundtrips_in_checkpoint(device):
    torch.manual_seed(19)
    module = (
        ModernLinearAttention(32, 4, dropout=0, state="rff", num_features=12)
        .to(device=device, dtype=torch.float64)
        .eval()
    )
    replica = (
        ModernLinearAttention(32, 4, dropout=0, state="rff", num_features=12)
        .to(device=device, dtype=torch.float64)
        .eval()
    )
    replica.load_state_dict(module.state_dict())
    x = torch.randn(2, 7, 32, device=device, dtype=torch.float64, requires_grad=True)
    expected = module(x, x, x, is_causal=True)[0]
    actual = replica(x, x, x, is_causal=True)[0]
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert replica.impl.feature_fn.omega.device == x.device
    assert replica.impl.feature_fn.omega.dtype == x.dtype
