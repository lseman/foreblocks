"""Public transformer construction and configuration contracts."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from foreblocks.nn.attention.config import AttentionPositionConfig
from foreblocks.nn.transformer import TransformerConfig
from foreblocks.nn.transformer.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from foreblocks.nn.transformer.encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


def small_config(**overrides):
    return TransformerConfig(
        input_size=2,
        output_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        patch_encoder=False,
        **overrides,
    )


@pytest.mark.parametrize(
    "constructor",
    [
        TransformerEncoder,
        TransformerDecoder,
        TransformerEncoderLayer,
        TransformerDecoderLayer,
    ],
)
def test_unmodified_config_keeps_identity_and_rejects_duplicate_supply(constructor):
    config = small_config()
    assert constructor(config).config is config
    assert constructor(config=config).config is config
    with pytest.raises(ValueError, match="once"):
        constructor(config, config=config)


@pytest.mark.parametrize("constructor", [TransformerEncoder, TransformerDecoder])
def test_explicit_model_dimensions_override_config_without_mutating_it(constructor):
    config = small_config()
    model = constructor(3, config=config, d_model=12, nhead=3)
    assert model.config.input_size == 3
    assert model.input_adapter.in_features == 3
    assert model.d_model == 12
    assert model.config.attention.shape.n_heads == 3
    assert config.input_size == 2
    assert config.d_model == config.attention.shape.d_model == 8


def test_named_decoder_overrides_include_false_and_zero():
    config = small_config(label_len=2, use_time_encoding=True)
    model = TransformerDecoder(
        config,
        output_size=3,
        label_len=0,
        informer_like=False,
        use_time_encoding=False,
        cache_implementation="static",
    )
    assert model.output_projection.out_features == 3
    assert model.label_len == 0
    assert model.layers[0].is_causal
    assert model.time_encoder is None
    assert model.cache_implementation == "static"
    assert config.output_size == 2
    assert config.label_len == 2
    assert config.use_time_encoding


def test_encoder_patch_preset_is_resolved_before_construction():
    config = small_config().with_overrides(patch_encoder=True)
    model = TransformerEncoder(config, ct_patchtst=True, ct_patch_len=4)
    assert model.ct_patch_embed.in_features == 4
    assert not model.config.patch_encoder
    assert config.patch_encoder
    assert not config.ct_patchtst


@pytest.mark.parametrize("constructor", [TransformerEncoder, TransformerDecoder])
def test_informer_preset_updates_effective_config(constructor):
    config = small_config(informer_like=False)
    model = constructor(config, model_type="informer-like")
    assert model.config.use_time_encoding
    assert model.time_encoder is not None
    if constructor is TransformerDecoder:
        assert model.informer_like
        assert not model.layers[0].is_causal
    assert not config.use_time_encoding


@pytest.mark.parametrize(
    "constructor", [TransformerEncoderLayer, TransformerDecoderLayer]
)
def test_standalone_layer_defaults_and_explicit_shape_overrides(constructor):
    legacy = constructor(8, 2)
    assert legacy.config.dim_feedforward == 2048
    if constructor is TransformerEncoderLayer:
        assert legacy.config.attention.variant.frequency_modes == 16
    else:
        assert legacy.is_causal
        assert not legacy.config.informer_like
    config = small_config()
    layer = constructor(12, 3, config=config, dropout=0.25)
    assert layer.d_model == 12
    assert layer._attention_config.shape.n_heads == 3
    assert layer._attention_config.shape.dropout == 0.25
    assert config.attention.shape.dropout == 0.0
    with pytest.raises(TypeError, match="required"):
        constructor()


def test_shared_shape_overrides_preserve_attention_backend_settings():
    config = small_config()
    config = config.with_overrides(
        attention=replace(
            config.attention,
            position=AttentionPositionConfig(encoding="sinusoidal"),
        )
    )
    updated = config.with_overrides(d_model=12, nhead=3, dropout=0.2, max_seq_len=64)
    assert updated.attention.shape.d_model == updated.d_model == 12
    assert updated.attention.shape.n_heads == updated.nhead == 3
    assert updated.attention.shape.dropout == updated.dropout == 0.2
    assert updated.attention.shape.max_seq_len == updated.max_seq_len == 64
    assert updated.attention.position is config.attention.position
    assert config.attention.shape.d_model == 8
    assert TransformerConfig.from_dict(updated.to_dict()).to_dict() == updated.to_dict()
    with pytest.raises(ValueError, match="must match d_model"):
        config.with_overrides(d_model=12, attention=config.attention)


@pytest.mark.parametrize("serialized", [False, True])
def test_legacy_construction_preserves_explicit_grouped_attention(serialized):
    config = small_config()
    attention = replace(
        config.attention,
        position=AttentionPositionConfig(encoding="sinusoidal"),
    )
    config = config.with_overrides(attention=attention)
    value = config.to_dict()["attention"] if serialized else attention
    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        attention=value,
    )
    assert model.config.attention == attention
    assert model.pos_encoding_type == "sinusoidal"


@pytest.mark.parametrize("constructor", [TransformerEncoder, TransformerDecoder])
def test_legacy_and_config_construction_have_matching_weights_and_outputs(constructor):
    arguments = dict(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        patch_encoder=False,
    )
    config = TransformerConfig.from_legacy_dict(arguments)
    torch.manual_seed(42)
    legacy = constructor(**arguments).eval()
    torch.manual_seed(42)
    configured = constructor(config).eval()
    torch.testing.assert_close(legacy.state_dict(), configured.state_dict())
    inputs = [torch.randn(2, 3, 2)]
    if constructor is TransformerDecoder:
        inputs.append(torch.randn(2, 4, 8))
    with torch.no_grad():
        torch.testing.assert_close(
            legacy(*inputs).last_hidden_state,
            configured(*inputs).last_hidden_state,
        )


@pytest.mark.parametrize("constructor", [TransformerEncoder, TransformerDecoder])
def test_injected_position_module_is_owned_by_base(constructor):
    position = nn.Identity()
    model = constructor(small_config(options={"pos_encoder": position}))
    assert model.pos_encoder is position


def test_role_specific_validation_runs_after_explicit_overrides():
    with pytest.raises(ValueError, match="only supported by the encoder"):
        TransformerDecoder(small_config(), use_variate_attention=True)
    with pytest.raises(ValueError, match="ct_patch_fuse"):
        small_config(ct_patch_fuse="unknown")
