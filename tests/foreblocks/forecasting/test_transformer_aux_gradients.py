"""Training losses must retain gradients across routing, reuse, and checkpointing."""

import pytest
import torch

from foreblocks.nn.transformer import (
    TransformerConfig,
    TransformerDecoder,
    TransformerEncoder,
)


def make_model(constructor, **overrides):
    return constructor(
        TransformerConfig(
            input_size=2,
            output_size=2,
            d_model=8,
            n_heads=2,
            num_layers=2,
            ff_dim=16,
            dropout=0.0,
            patching="none",
            **overrides,
        )
    ).train()


def inputs_for(constructor):
    inputs = [torch.randn(2, 4, 2)]
    if constructor is TransformerDecoder:
        inputs.append(torch.randn(2, 5, 8))
    return inputs


@pytest.mark.parametrize("constructor", [TransformerEncoder, TransformerDecoder])
@pytest.mark.parametrize(
    "feature",
    [
        {"residual": "gateskip"},
        {"moe_experts": 2, "moe_top_k": 1},
        {"residual": "mod"},
    ],
    ids=["gateskip", "moe", "mod"],
)
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_auxiliary_loss_alone_trains_parameters(
    constructor, feature, checkpoint, shared
):
    torch.manual_seed(123)
    model = make_model(
        constructor,
        **feature,
        share_layers=shared,
        gradient_checkpointing=checkpoint,
    )
    inputs = inputs_for(constructor)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        output = model(*inputs)
        assert output.aux_loss.requires_grad
        assert torch.isfinite(output.aux_loss)
        output.aux_loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


@pytest.mark.parametrize("constructor", [TransformerEncoder, TransformerDecoder])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_shared_layer_loss_matches_each_invocation(constructor, checkpoint):
    torch.manual_seed(321)
    model = make_model(
        constructor,
        residual="gateskip",
        share_layers=True,
        gradient_checkpointing=checkpoint,
        moe_aux_weight=0.7,
    )
    losses = []
    handle = model.shared_layer.register_forward_hook(
        lambda layer, args, out: losses.append(layer.aux_loss)
    )
    try:
        output = model(*inputs_for(constructor))
    finally:
        handle.remove()
    assert len(losses) == model.config.num_layers
    expected = torch.stack(losses).mean() * model.config.moe_aux_weight
    torch.testing.assert_close(output.aux_loss, expected)
    parameter = (
        model.shared_layer.gate_attn
        if constructor is TransformerEncoder
        else model.shared_layer.gate_self
    )
    parameter = next(parameter.parameters())
    actual_grad = torch.autograd.grad(output.aux_loss, parameter, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, parameter)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("constructor", [TransformerEncoder, TransformerDecoder])
def test_checkpointed_shared_hybrid_matches_ordinary_gradients(constructor):
    torch.manual_seed(456)
    ordinary = make_model(
        constructor,
        share_layers=True,
        residual="gateskip",
    )
    ordinary = constructor(
        ordinary.config, attention="linear", attention_pattern="hybrid"
    ).train()
    checkpointed = constructor(ordinary.config, gradient_checkpointing=True).train()
    checkpointed.load_state_dict(ordinary.state_dict())
    inputs = inputs_for(constructor)
    outputs = []
    for model in (ordinary, checkpointed):
        output = model(*inputs)
        outputs.append(output)
        (output.last_hidden_state.square().sum() + output.aux_loss).backward()
    torch.testing.assert_close(
        outputs[0].last_hidden_state, outputs[1].last_hidden_state
    )
    torch.testing.assert_close(outputs[0].aux_loss, outputs[1].aux_loss)
    for (name, parameter), (_, checkpoint_parameter) in zip(
        ordinary.named_parameters(), checkpointed.named_parameters()
    ):
        assert (parameter.grad is None) == (checkpoint_parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, checkpoint_parameter.grad, msg=name
            )
