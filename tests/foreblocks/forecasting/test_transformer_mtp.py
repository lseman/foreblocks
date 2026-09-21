"""Decoder MTP target alignment and boundary validation."""

import pytest
import torch
from torch import nn

from foreblocks.nn.transformer.decoder import TransformerDecoder
from foreblocks.nn.transformer.runtime.mtp import build_decoder_mtp_targets


def build_targets(targets, **overrides):
    options = dict(
        batch_size=2,
        sequence_length=3,
        d_model=4,
        num_heads=4,
        input_adapter=nn.Linear(2, 4),
    )
    options.update(overrides)
    return build_decoder_mtp_targets(targets, **options)


def test_shifted_targets_preserve_values_and_zero_pad_missing_horizons():
    base = torch.arange(24.0).reshape(2, 3, 4)
    result = build_targets(base)
    assert result.shape == (2, 3, 4, 4)
    torch.testing.assert_close(result[:, :2, 0], base[:, 1:])
    torch.testing.assert_close(result[:, :1, 1], base[:, 2:])
    assert torch.count_nonzero(result[:, 2:, 0]) == 0
    assert torch.count_nonzero(result[:, 1:, 1]) == 0
    assert torch.count_nonzero(result[:, :, 2:]) == 0


def test_input_width_projection_preserves_gradient_flow():
    adapter = nn.Linear(2, 4, bias=False)
    base = torch.randn(2, 3, 2, requires_grad=True)
    result = build_targets(base, input_adapter=adapter)
    torch.testing.assert_close(result[:, :2, 0], adapter(base[:, 1:]))
    result.sum().backward()
    assert base.grad is not None
    assert adapter.weight.grad is not None
    assert torch.count_nonzero(base.grad[:, 0]) == 0
    assert torch.count_nonzero(base.grad[:, 1:]) > 0


def test_aligned_targets_allow_extra_heads_and_preserve_identity():
    targets = torch.randn(2, 3, 5, 4)
    assert build_targets(targets) is targets


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((2, 3), "must be"),
        ((1, 3, 4), "batch and length"),
        ((2, 2, 4), "batch and length"),
        ((2, 3, 7), "last dim"),
        ((1, 3, 4, 4), "batch and length"),
        ((2, 3, 4, 7), "feature dim"),
        ((2, 3, 3, 4), "horizon"),
    ],
)
def test_invalid_targets_fail_at_preparation_boundary(shape, message):
    with pytest.raises(ValueError, match=message):
        build_targets(torch.zeros(shape))


def test_single_token_has_no_future_targets():
    result = build_targets(torch.ones(2, 1, 4), sequence_length=1)
    assert result.shape == (2, 1, 4, 4)
    assert torch.count_nonzero(result) == 0


def test_decoder_validates_targets_only_when_mtp_is_active(monkeypatch):
    decoder = TransformerDecoder(
        input_size=2,
        output_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
        dropout=0.0,
    )
    monkeypatch.setattr(decoder, "_infer_mtp_num_heads", lambda: 2)
    inputs = torch.randn(2, 3, 2)
    memory = torch.randn(2, 4, 8)
    targets = torch.randn(1, 3, 2)
    with pytest.raises(ValueError, match="batch and length"):
        decoder(inputs, memory, mtp_targets=targets)
    decoder.eval()
    decoder(inputs, memory, mtp_targets=targets)
