"""Execution contracts across the private HeadComposer helpers."""

import pytest
import torch
from torch import nn

from foreblocks.nn.heads.engine.composer import HeadComposer, HeadSpec


@pytest.mark.parametrize(
    "fusion",
    [
        "concat",
        "sum",
        "mean",
        "weighted_sum",
        "gated_fusion",
        "lora_mix",
        "moe_routing",
        "hypernetwork_mix",
        "attention_fusion",
        "multi_scale_attn",
    ],
)
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_parallel_fusion_checkpoint_reload_and_gradients(fusion, dropout):
    def build():
        return HeadComposer(
            parallel_specs=[
                HeadSpec(nn.Linear(4, 4), name="first", alpha_mode="soft"),
                HeadSpec(nn.Linear(4, 6), name="second", alpha_mode="off"),
            ],
            composer_mode="parallel",
            parallel_combine=fusion,
            parallel_align_mode="project",
            parallel_project=True,
            parallel_project_dim=4,
            parallel_fusion_dropout=dropout,
            enable_nas=True,
        ).eval()

    torch.manual_seed(23)
    value = torch.randn(2, 5, 4, requires_grad=True)
    composer = build()
    output, state = composer(value)
    checkpoint = composer.state_dict()
    parameter_ids = {name: id(p) for name, p in composer.named_parameters()}
    repeated, _ = composer(value)
    torch.testing.assert_close(repeated, output)
    assert parameter_ids == {name: id(p) for name, p in composer.named_parameters()}
    assert composer.inverse_post(output, state) is output

    restored = build()
    # Warm up lazy projections before loading, as required by the existing interface.
    restored(value.detach())
    restored.load_state_dict(checkpoint, strict=True)
    actual, actual_state = restored(value.detach())
    torch.testing.assert_close(actual, output)
    assert [(s.kind, s.name, s.stage) for s in actual_state] == [
        (s.kind, s.name, s.stage) for s in state
    ]
    assert list(restored.state_dict()) == list(checkpoint)

    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    for head in composer.parallel_heads:
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
    assert any(p.grad is not None for p in composer.arch_parameters())


class AffineInvert(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.scale = scale
        self.offset = offset

    def forward(self, x):
        return x * self.scale + self.offset, None

    def invert(self, y, state):
        return (y - self.offset) / self.scale


class SplitCarry(nn.Module):
    def forward(self, x):
        return x * 0.75, x * 0.25


@pytest.mark.parametrize("mode", ["serial", "hybrid"])
def test_inverse_reverses_order_restores_carry_and_ignores_disabled_heads(mode):
    serial_specs = [
        HeadSpec(AffineInvert(2, 3), name="first", combine="invert"),
        HeadSpec(SplitCarry(), name="carry", combine="add"),
        HeadSpec(AffineInvert(3, -2), name="last", combine="invert"),
        HeadSpec(AffineInvert(5, 8), name="disabled", combine="invert"),
    ]
    stage_specs = (
        {"specs": serial_specs}
        if mode == "serial"
        else {
            "parallel_specs": [HeadSpec(nn.Identity(), name="parallel")],
            "serial_specs": serial_specs,
        }
    )
    composer = HeadComposer(
        **stage_specs, composer_mode=mode, use_spectral_norm_invert=False
    )
    composer.disable_head("disabled")
    value = torch.randn(2, 5, 4, requires_grad=True)
    if mode == "hybrid":
        expected, _ = composer._forward_parallel_stage_impl(value)
    else:
        expected = value
    output, state = composer(value)
    recovered = composer.inverse_post(output, state)
    torch.testing.assert_close(recovered, expected)
    recovered.sum().backward()
    assert torch.isfinite(value.grad).all()
    assert state[-1].kind == "disabled"


def test_parallel_fusion_preserves_internal_key_errors(monkeypatch):
    composer = HeadComposer(composer_mode="parallel", parallel_combine="sum")

    def fail(branches, value):
        raise KeyError("missing internal state")

    monkeypatch.setattr(composer, "_combine_parallel_sum", fail)
    with pytest.raises(KeyError, match="missing internal state"):
        composer._combine_parallel_branches([], torch.zeros(1, 2, 3))


def test_parallel_fusion_reports_unknown_mode():
    composer = HeadComposer(composer_mode="parallel")
    composer.parallel_combine = "unknown"
    with pytest.raises(ValueError, match="Unknown parallel_combine: unknown"):
        composer._combine_parallel_branches([], torch.zeros(1, 2, 3))
