from __future__ import annotations

import pytest
import torch

from foreblocks.models.transformer import (
    MixingTransformer,
    StackedMixingTransformer,
    TransformerConfig,
)
from foreblocks.models.transformer.core.encoder import TransformerEncoder
from foreblocks.models.transformer.runtime.outputs import TransformerEncoderOutput
from foreblocks.attention import (
    AttentionCacheConfig,
    AttentionConfig,
    AttentionPositionConfig,
    AttentionShapeConfig,
    AttentionVariantConfig,
    PositionEncoding,
)


def _config(*, num_layers: int = 1) -> TransformerConfig:
    return TransformerConfig(
        d_model=8,
        nhead=2,
        num_layers=num_layers,
        dim_feedforward=16,
        dropout=0.0,
        patch_encoder=False,
        custom_norm="rms",
        attention=AttentionConfig(
            shape=AttentionShapeConfig(
                d_model=8,
                n_heads=2,
                dropout=0.0,
                max_seq_len=32,
            ),
            cache=AttentionCacheConfig(use_paged_cache=False, use_mla=False),
            position=AttentionPositionConfig(encoding=PositionEncoding.ROPE),
            variant=AttentionVariantConfig(name="standard", use_swiglu=False),
        ),
    )


def test_mixing_transformer_shape_weights_and_gradients() -> None:
    layer = MixingTransformer(_config())
    inputs = torch.randn(2, 3, 5, 8, requires_grad=True)

    output, sequence_weights, variate_weights = layer(inputs, need_weights=True)

    assert output.shape == inputs.shape
    assert sequence_weights is not None
    assert sequence_weights.shape == (6, 2, 5, 5)
    assert variate_weights is not None
    assert variate_weights.shape == (10, 2, 3, 3)
    output.square().mean().backward()
    assert inputs.grad is not None
    assert layer.seq_attn.q_proj.weight.grad is not None
    assert layer.var_attn is not None
    assert layer.var_attn.q_proj.weight.grad is not None


def test_patch_mask_is_applied_on_both_attention_axes() -> None:
    layer = MixingTransformer(_config())
    inputs = torch.randn(1, 2, 3, 8)
    patch_mask = torch.tensor([[[False, False, True], [False, False, True]]])

    output, sequence_weights, variate_weights = layer(
        inputs, patch_mask, need_weights=True
    )

    assert torch.isfinite(output).all()
    assert sequence_weights is not None
    torch.testing.assert_close(
        sequence_weights[..., 2], torch.zeros_like(sequence_weights[..., 2])
    )
    assert variate_weights is not None
    assert torch.isfinite(variate_weights).all()
    # At the final time step both variates are padding, so its update is empty.
    torch.testing.assert_close(variate_weights[2], torch.zeros_like(variate_weights[2]))


def test_sequence_attention_can_be_causal_without_causal_variate_attention() -> None:
    layer = MixingTransformer(_config())
    inputs = torch.randn(1, 3, 4, 8)

    _, sequence_weights, variate_weights = layer(
        inputs, is_causal=True, need_weights=True
    )

    assert sequence_weights is not None
    assert torch.count_nonzero(torch.triu(sequence_weights, diagonal=1)) == 0
    assert variate_weights is not None
    assert torch.count_nonzero(torch.triu(variate_weights, diagonal=1)) > 0


def test_per_example_masks_are_repeated_across_each_mixing_axis() -> None:
    layer = MixingTransformer(_config())
    inputs = torch.randn(2, 3, 4, 8)
    sequence_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    sequence_mask[0, :, 3] = True
    variate_mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    variate_mask[1, :, 2] = True

    _, sequence_weights, variate_weights = layer(
        inputs,
        sequence_mask=sequence_mask,
        variate_mask=variate_mask,
        need_weights=True,
    )

    assert sequence_weights is not None
    assert torch.count_nonzero(sequence_weights[:3, ..., 3]) == 0
    assert torch.count_nonzero(sequence_weights[3:, ..., 3]) > 0
    assert variate_weights is not None
    assert torch.count_nonzero(variate_weights[:4, ..., 2]) > 0
    assert torch.count_nonzero(variate_weights[4:, ..., 2]) == 0


def test_variate_attention_can_be_disabled() -> None:
    layer = MixingTransformer(_config(), use_variate_attention=False)
    inputs = torch.randn(2, 3, 4, 8)

    output, _, variate_weights = layer(inputs, need_weights=True)

    assert output.shape == inputs.shape
    assert variate_weights is None
    assert layer.var_attn is None


def test_stacked_mixing_transformer_runs_every_layer() -> None:
    model = StackedMixingTransformer(_config(num_layers=3))
    inputs = torch.randn(2, 2, 4, 8)

    output, sequence_weights, variate_weights = model(inputs)

    assert output.shape == inputs.shape
    assert len(sequence_weights) == 3
    assert len(variate_weights) == 3
    assert all(weight is None for weight in sequence_weights)
    assert all(weight is None for weight in variate_weights)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((2, 4, 8), "input_embeddings must have shape"),
        ((2, 3, 4, 7), "does not match d_model"),
    ],
)
def test_mixing_transformer_validates_embedding_shape(
    shape: tuple[int, ...], message: str
) -> None:
    layer = MixingTransformer(_config())

    with pytest.raises(ValueError, match=message):
        layer(torch.randn(shape))


def test_mixing_transformer_validates_patch_mask_shape() -> None:
    layer = MixingTransformer(_config())

    with pytest.raises(ValueError, match="patch_mask must have shape"):
        layer(torch.randn(2, 3, 4, 8), torch.zeros(2, 4, dtype=torch.bool))


def test_transformer_encoder_enables_variate_attention_as_an_option() -> None:
    config = _config(num_layers=2).with_overrides(
        input_size=3,
        patch_encoder=False,
        use_variate_attention=True,
        return_dict=True,
    )
    encoder = TransformerEncoder(config)
    inputs = torch.randn(2, 5, 3, requires_grad=True)
    padding_mask = torch.zeros(2, 3, 5, dtype=torch.bool)
    padding_mask[:, 2, -1] = True

    result = encoder(
        inputs,
        src_key_padding_mask=padding_mask,
        output_hidden_states=True,
        output_attentions=True,
    )

    assert isinstance(result, TransformerEncoderOutput)
    assert result.last_hidden_state.shape == (2, 5, 8)
    assert result.padding_mask is not None
    assert result.padding_mask.shape == (2, 5)
    assert result.hidden_states is not None
    assert all(state.shape == (2, 5, 8) for state in result.hidden_states)
    assert result.attentions is not None
    assert result.attentions[0].shape == (6, 2, 5, 5)
    assert result.variate_attentions is not None
    assert result.variate_attentions[0].shape == (10, 2, 3, 3)
    assert isinstance(encoder._get_layer(0), MixingTransformer)
    result.last_hidden_state.mean().backward()
    assert inputs.grad is not None


def test_transformer_encoder_can_keep_the_variate_axis() -> None:
    config = _config().with_overrides(
        input_size=3,
        patch_encoder=False,
        use_variate_attention=True,
        variate_fuse="none",
        return_dict=False,
    )
    encoder = TransformerEncoder(config)

    output = encoder(torch.randn(2, 5, 3))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 3, 5, 8)


def test_transformer_encoder_patches_each_variate_independently() -> None:
    config = _config().with_overrides(
        input_size=3,
        patch_encoder=True,
        patch_len=3,
        patch_stride=2,
        use_variate_attention=True,
        return_dict=True,
    )
    encoder = TransformerEncoder(config)

    result = encoder(
        torch.randn(2, 7, 3),
        src_key_padding_mask=torch.zeros(2, 7, dtype=torch.bool),
    )

    assert isinstance(result, TransformerEncoderOutput)
    assert result.last_hidden_state.shape == (2, 3, 8)
    assert result.padding_mask is not None
    assert result.padding_mask.shape == (2, 3)


def test_variate_encoder_supports_gradient_checkpointing() -> None:
    config = _config(num_layers=2).with_overrides(
        input_size=3,
        patch_encoder=False,
        use_variate_attention=True,
        use_gradient_checkpointing=True,
        return_dict=False,
    )
    encoder = TransformerEncoder(config).train()
    inputs = torch.randn(2, 5, 3, requires_grad=True)

    output = encoder(inputs)
    assert isinstance(output, torch.Tensor)
    output.square().mean().backward()

    assert inputs.grad is not None
    for layer in encoder.layers:
        assert isinstance(layer, MixingTransformer)
        assert layer.seq_attn.q_proj.weight.grad is not None
        assert layer.var_attn is not None
        assert layer.var_attn.q_proj.weight.grad is not None


@pytest.mark.parametrize(
    "feature", ["use_mhc", "use_mod", "use_gateskip", "use_moe", "ct_patchtst"]
)
def test_variate_encoder_rejects_incompatible_three_dimensional_features(
    feature: str,
) -> None:
    with pytest.raises(ValueError, match="use_variate_attention is incompatible"):
        TransformerEncoder(
            _config().with_overrides(
                input_size=3,
                use_variate_attention=True,
                **{feature: True},
            )
        )


def _contiguous_encoder(*, num_layers: int = 2) -> TransformerEncoder:
    config = _config(num_layers=num_layers).with_overrides(
        input_size=3,
        patch_encoder=True,
        patch_len=2,
        patch_stride=2,
        use_variate_attention=True,
        use_contiguous_patch_decoding=True,
        return_dict=True,
    )
    return TransformerEncoder(config)


def test_contiguous_decode_predicts_all_horizon_patches_in_one_pass() -> None:
    encoder = _contiguous_encoder(num_layers=2).eval()
    target = torch.randn(2, 5, 1)
    past_only = torch.randn(2, 5, 1)
    known_future = torch.randn(2, 8, 1)
    layer_calls = [0, 0]
    hooks = []
    for index, layer in enumerate(encoder.layers):

        def count_call(_module, _args, _output, layer_index=index):
            layer_calls[layer_index] += 1

        hooks.append(layer.register_forward_hook(count_call))

    forecasts = encoder.decode_contiguous(
        target,
        past_only_covariates=past_only,
        past_future_covariates=known_future,
    )
    for hook in hooks:
        hook.remove()

    assert forecasts.shape == (2, 3, 1, 9)
    assert torch.isfinite(forecasts).all()
    assert layer_calls == [1, 1]
    assert forecasts.requires_grad is False


def test_contiguous_forecast_keeps_known_future_covariates_visible() -> None:
    encoder = _contiguous_encoder(num_layers=1).eval()
    target = torch.randn(1, 4, 1)
    past_only = torch.randn(1, 4, 1)
    known_future = torch.randn(1, 8, 1)

    baseline = encoder.forecast_contiguous(
        target,
        horizon=4,
        past_only_covariates=past_only,
        past_future_covariates=known_future,
    )
    changed = known_future.clone()
    changed[:, 4:] += 10.0
    with_signal = encoder.forecast_contiguous(
        target,
        horizon=4,
        past_only_covariates=past_only,
        past_future_covariates=changed,
    )

    assert not torch.allclose(baseline, with_signal)


def test_contiguous_forecast_supports_training_with_pinball_loss() -> None:
    encoder = _contiguous_encoder(num_layers=1).train()
    context = torch.randn(2, 4, 1)
    future = torch.randn(2, 3, 1)
    known = torch.randn(2, 7, 2)

    predictions = encoder.forecast_contiguous(
        context,
        horizon=3,
        past_future_covariates=known,
    )
    loss = encoder.contiguous_quantile_loss(predictions, future)
    loss.backward()

    assert predictions.shape == (2, 3, 1, 9)
    assert loss.ndim == 0
    assert encoder.contiguous_patch_head is not None
    assert encoder.contiguous_patch_head.weight.grad is not None
    layer = encoder._get_layer(0)
    assert isinstance(layer, MixingTransformer)
    assert layer.seq_attn.q_proj.weight.grad is not None
    assert layer.var_attn is not None
    assert layer.var_attn.q_proj.weight.grad is not None


def test_contiguous_decode_masks_unknown_horizon_but_not_its_tokens(
    monkeypatch,
) -> None:
    encoder = _contiguous_encoder(num_layers=1).eval()
    original_forward = encoder.forward
    captured = {}

    def capture_forward(src, *args, **kwargs):
        captured["src"] = src
        captured["padding"] = kwargs["src_key_padding_mask"]
        captured["values"] = kwargs["value_mask"]
        return original_forward(src, *args, **kwargs)

    monkeypatch.setattr(encoder, "forward", capture_forward)
    encoder.decode_contiguous(
        torch.randn(1, 3, 1),
        horizon=3,
        past_only_covariates=torch.randn(1, 3, 1),
        past_future_covariates=torch.randn(1, 6, 1),
    )

    # One left-padding step, followed by four context/horizon patch pairs.
    assert captured["src"].shape == (1, 8, 3)
    value_mask = captured["values"]
    attention_padding = captured["padding"]
    assert value_mask[:, 4:, :2].all()  # target and past-only are unknown
    assert not value_mask[:, 4:7, 2].any()  # known future signal stays visible
    assert not attention_padding[:, 4:].any()  # placeholders remain attention keys


def test_contiguous_quantile_loss_respects_missing_target_mask() -> None:
    encoder = _contiguous_encoder(num_layers=1)
    predictions = torch.zeros(1, 2, 1, 9)
    targets = torch.tensor([[[1.0], [100.0]]])
    mask = torch.tensor([[[False], [True]]])

    loss = encoder.contiguous_quantile_loss(predictions, targets, mask)

    assert loss.item() == pytest.approx(0.5)


def test_contiguous_decode_requires_compatible_patch_configuration() -> None:
    with pytest.raises(ValueError, match="requires use_variate_attention"):
        TransformerEncoder(_config().with_overrides(use_contiguous_patch_decoding=True))
    with pytest.raises(ValueError, match="patch_stride == patch_len"):
        TransformerEncoder(
            _config().with_overrides(
                input_size=2,
                use_variate_attention=True,
                use_contiguous_patch_decoding=True,
                patch_encoder=True,
                patch_len=4,
                patch_stride=2,
            )
        )
