"""Public transformer configuration and construction contracts."""

import json
import pickle
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from foreblocks.nn.transformer import (
    TransformerConfig,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


def small_config(**overrides):
    settings = dict(
        input_size=2,
        output_size=2,
        d_model=8,
        n_heads=2,
        num_layers=1,
        ff_dim=16,
        dropout=0.0,
        patching="none",
    )
    return TransformerConfig(**{**settings, **overrides})


STACKS = [TransformerEncoder, TransformerDecoder]
LAYERS = [TransformerEncoderLayer, TransformerDecoderLayer]


@pytest.mark.parametrize("constructor", STACKS + LAYERS)
def test_keywords_config_and_both_build_the_same_settings(constructor):
    config = small_config()
    assert constructor(config).config is config
    from_keywords = constructor(**config.to_dict()).config
    assert from_keywords == config
    overridden = constructor(config, d_model=12, n_heads=3).config
    assert (overridden.d_model, overridden.n_heads) == (12, 3)
    assert (config.d_model, config.n_heads) == (8, 2)


def test_one_config_serves_encoder_and_decoder():
    config = small_config(d_model=16, n_heads=4, residual="gateskip")
    encoder = TransformerEncoder(config, input_size=3)
    decoder = TransformerDecoder(config, output_size=5)
    memory = encoder(torch.randn(2, 6, 3)).last_hidden_state
    output = decoder(torch.randn(2, 4, 2), memory).last_hidden_state
    assert output.shape == (2, 4, 5)


def test_unknown_settings_and_values_fail_fast():
    with pytest.raises(TypeError, match="nhead"):
        TransformerConfig(nhead=4)
    with pytest.raises(ValueError, match="norm_placement must be one of"):
        TransformerConfig(norm_placement="pre_norm")
    with pytest.raises(ValueError, match="d_model must be divisible"):
        TransformerConfig(d_model=10, n_heads=4)
    with pytest.raises(ValueError, match="only supported by the encoder"):
        TransformerDecoder(small_config(variate_attention=True))


def test_config_is_plain_serializable_data():
    config = small_config(
        attention="gla",
        attention_pattern="3to1",
        attention_options={"qk_norm": True},
        quantiles=[0.1, 0.5, 0.9],
    )
    with pytest.raises(FrozenInstanceError):
        config.d_model = 32
    restored = TransformerConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert restored.quantiles == (0.1, 0.5, 0.9)
    assert pickle.loads(pickle.dumps(config)) == config


def test_attention_config_is_derived_from_flat_settings():
    config = small_config(
        n_kv_heads=1,
        attention="sliding_window",
        attention_kernel="sdpa",
        position="alibi",
        kv_cache="dynamic",
        attention_options={"window_size": 16, "qk_norm": True, "block_size": 32},
    )
    attention = config.attention_config()
    assert attention.shape.n_kv_heads == 1
    assert attention.shape.d_model == config.d_model
    assert attention.variant.name == "sliding_window"
    assert attention.variant.backend == "sdpa"
    assert attention.variant.window_size == 16
    assert attention.features.qk_norm
    assert attention.cache.block_size == 32
    assert not attention.cache.use_paged_cache
    assert attention.position.encoding == "alibi"
    assert config.attention_config(cross=True).shape.cross_attention
    assert small_config(kv_cache="auto").attention_config().cache.use_paged_cache

    matching = small_config(attention_options={"attention_matching": True})
    assert not matching.attention_config().cache.use_mla
    with pytest.raises(ValueError, match="unknown attention_options: windw_size"):
        small_config(attention_options={"windw_size": 16})


@pytest.mark.parametrize("constructor", STACKS)
@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("uniform", ["gla"] * 4),
        ("hybrid", ["gla", "gla", "gla", "standard"]),
        ("3to1", ["gla", "gla", "gla", "standard"]),
    ],
)
def test_attention_pattern_selects_backend_per_depth(constructor, pattern, expected):
    model = constructor(
        small_config(num_layers=4, attention="gla", attention_pattern=pattern)
    )
    assert [layer.layer_attention_type for layer in model.layers] == expected
    assert [layer.config.attention for layer in model.layers] == expected
    assert all(layer.config.attention_pattern == "uniform" for layer in model.layers)


@pytest.mark.parametrize("constructor", LAYERS)
def test_layers_build_their_configured_backend(constructor):
    from foreblocks.nn.attention.algorithms import ModernLinearAttention

    layer = constructor(small_config(attention="linear"))
    assert isinstance(layer._self_attn(), ModernLinearAttention)
    with pytest.raises(ValueError, match="for_layer"):
        constructor(small_config(attention_pattern="hybrid"))
    with pytest.raises(ValueError, match="unknown attention 'typo'"):
        constructor(small_config(attention="typo"))


@pytest.mark.parametrize("constructor", LAYERS)
@pytest.mark.parametrize(
    "residual,module",
    [
        ("standard", None),
        ("gateskip", "gate_ff"),
        ("mhc", "mhc_conn_ff"),
        ("attention", "ff_input_residual"),
    ],
)
def test_layers_only_build_the_selected_residual_modules(constructor, residual, module):
    layer = constructor(small_config(residual=residual))
    for name in ("gate_ff", "mhc_conn_ff", "ff_input_residual"):
        assert (getattr(layer, name) is not None) == (name == module)


@pytest.mark.parametrize("constructor", STACKS)
def test_live_collaborators_are_constructor_arguments(constructor):
    from foreblocks.nn.routing.mod import LayerDropoutSchedule

    position = nn.Identity()
    schedule = LayerDropoutSchedule(2, 0.1, 0.3, "deeper_more")
    model = constructor(
        small_config(num_layers=2), pos_encoder=position, dropout_schedule=schedule
    )
    assert model.pos_encoder is position
    for index, layer in enumerate(model.layers):
        assert layer.config.dropout == pytest.approx(schedule.get_dropout(index))
        assert layer.ff_norm.dropout.p == pytest.approx(schedule.get_dropout(index))
    assert model.config.dropout == 0.0


@pytest.mark.parametrize("constructor", STACKS)
def test_shared_layers_require_a_constant_dropout_schedule(constructor):
    from foreblocks.nn.routing.mod import LayerDropoutSchedule

    config = small_config(num_layers=2, share_layers=True)
    varying = LayerDropoutSchedule(2, 0.1, 0.3, "deeper_more")
    with pytest.raises(ValueError, match="constant layer dropout"):
        constructor(config, dropout_schedule=varying)
    constant = LayerDropoutSchedule(2, 0.2, 0.2, "deeper_more")
    model = constructor(config, dropout_schedule=constant)
    assert model.shared_layer.config.dropout == pytest.approx(0.2)


@pytest.mark.parametrize("encoding", ["sinusoidal", "learnable"])
def test_input_position_encoding_selection_and_trainability(encoding):
    from foreblocks.nn.embeddings import LearnablePositionalEncoding, PositionalEncoding

    model = TransformerEncoder(small_config(position=encoding))
    expected = (
        LearnablePositionalEncoding if encoding == "learnable" else PositionalEncoding
    )
    assert isinstance(model.pos_encoder, expected)
    assert TransformerEncoder(small_config(position="rope")).pos_encoder is None
    if encoding == "learnable":
        model(torch.randn(2, 3, 2)).last_hidden_state.square().sum().backward()
        assert model.pos_encoder.pe.grad.abs().sum() > 0


def test_informer_decoder_is_non_causal_and_masks_the_horizon():
    decoder = TransformerDecoder(small_config(informer=True, label_len=2))
    assert not decoder.layers[0].is_causal
    mask = decoder._informer_padding_mask(1, 5, torch.device("cpu"))
    assert mask.tolist() == [[False, False, True, True, True]]
    assert TransformerDecoder(small_config()).layers[0].is_causal


def test_new_backend_preserves_existing_layer_dtype_and_eval_mode():
    layer = TransformerEncoderLayer(small_config()).double().eval()
    backend = layer.materialize_attention_type("linear")
    assert next(backend.parameters()).dtype == torch.float64
    assert not backend.training


def test_moe_options_reach_the_feed_forward_block():
    layer = TransformerEncoderLayer(
        small_config(moe_experts=4, moe_options={"num_shared": 1})
    )
    assert layer.feed_forward.block.num_shared == 1
    with pytest.raises(ValueError, match="unknown moe_options: num_experts"):
        small_config(moe_options={"num_experts": 4})


@pytest.mark.parametrize("variate_position", [False, True])
def test_variate_attention_position_encoding_is_opt_in(variate_position):
    encoder = TransformerEncoder(
        small_config(
            input_size=3, variate_attention=True, variate_position=variate_position
        )
    )
    variate = encoder.layers[0].var_attn
    expected = "rope" if variate_position else "none"
    assert variate.pos_encoding_type == expected
    assert encoder(torch.randn(2, 5, 3)).last_hidden_state.shape == (2, 5, 8)


@pytest.mark.parametrize("constructor", STACKS)
def test_stacks_expose_the_sequence_module_sizes(constructor):
    model = constructor(small_config(input_size=3, output_size=5, d_model=12, n_heads=3))
    assert (model.input_size, model.output_size, model.d_model) == (3, 5, 12)
    with pytest.raises(AttributeError):
        model.d_model = 16
