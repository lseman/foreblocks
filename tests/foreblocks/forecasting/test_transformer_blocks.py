import json

import pytest
import torch

from foreblocks.nn import transformer as tf


def test_declarative_stack_round_trip_and_gradients():
    local = tf.Block(tf.GatedDeltaNet(heads=2), tf.SwiGLU(hidden_dim=17))
    global_block = tf.Block(
        tf.GQA(heads=4, kv_heads=2, position=tf.RoPE(), qk_norm=True),
        tf.MoE(tf.SwiGLU(hidden_dim=19), num_experts=3, top_k=2),
        tf.LayerNorm(),
    )
    encoder = tf.TransformerEncoder(
        input_size=3,
        d_model=32,
        patching="none",
        dropout=0.0,
        blocks=[local, local, global_block],
    )
    layers = encoder.layers
    assert len(layers) == 3
    assert layers[0].feed_forward.block.w1.out_features == 17
    assert layers[2].config.ff_dim == 19
    assert layers[2].config.moe_experts == 3
    assert (
        layers[0].feed_forward.block.w1.weight
        is not layers[1].feed_forward.block.w1.weight
    )
    x = torch.randn(2, 8, 3)
    output = encoder(x).last_hidden_state
    output.square().mean().backward()
    for layer in layers:
        assert any(p.grad is not None for p in layer._self_attn().parameters())
    assert any(
        p.grad is not None for n, p in layers[2].named_parameters() if "router" in n
    )
    restored = tf.TransformerEncoder(
        tf.TransformerConfig.from_dict(json.loads(json.dumps(encoder.config.to_dict())))
    )
    restored.load_state_dict(encoder.state_dict(), strict=True)
    torch.testing.assert_close(
        encoder.eval()(x).last_hidden_state, restored.eval()(x).last_hidden_state
    )


def test_block_validation():
    with pytest.raises(ValueError, match="kv_heads"):
        tf.Block(tf.GQA(3, 2), tf.SwiGLU(16))
    with pytest.raises(ValueError, match="nonempty"):
        tf.TransformerEncoder(blocks=[])


@pytest.mark.parametrize("residual", [tf.MHC(streams=2), tf.AttnRes()])
def test_stack_residual_layout(residual):
    block = tf.Block(tf.GQA(2, 1), tf.SwiGLU(17), residual=residual)
    encoder = tf.TransformerEncoder(
        input_size=2, d_model=16, patching="none", blocks=[block, block]
    )
    assert encoder(torch.randn(2, 5, 2)).last_hidden_state.shape == (2, 5, 16)
    assert tf.TransformerConfig.from_dict(encoder.config.to_dict()) == encoder.config


@pytest.mark.parametrize("mixer", [tf.GQA(2, 1), tf.GatedDeltaNet(2)])
def test_connected_encoder_decoder(mixer):
    enc = tf.EncoderBlock(tf.GQA(2, 1), tf.SwiGLU(24))
    dec = tf.DecoderBlock(mixer=mixer, cross_attention=tf.GQA(4, 2), ffn=tf.SwiGLU(25))
    encoder = tf.TransformerEncoder(
        input_size=3, d_model=16, blocks=[enc], patching="none", dropout=0.0
    )
    decoder = tf.TransformerDecoder(
        input_size=2, output_size=1, d_model=32, blocks=[dec], dropout=0.0
    )
    model = tf.EncoderDecoder(encoder, decoder, target_len=3, teacher_forcing_ratio=1.0)
    history = torch.randn(2, 8, 3)
    targets = torch.randn(2, 3, 1)
    pred = model(history, targets)
    assert pred.shape == targets.shape
    pred.square().mean().backward()
    assert encoder.input_adapter.weight.grad.abs().sum() > 0
    cross = decoder.layers[0].cross_attn
    assert (cross.n_heads, cross.n_kv_heads) == (4, 2)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in cross.parameters()
    )
    restored_decoder = tf.TransformerDecoder(
        tf.TransformerConfig.from_dict(json.loads(json.dumps(decoder.config.to_dict())))
    )
    clone = tf.EncoderDecoder(
        tf.TransformerEncoder(encoder.config), restored_decoder, target_len=3
    )
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone.eval()(history), model.eval()(history))
    with torch.no_grad():
        memory = model._encode_memory(history)
        prefix = torch.randn(2, 5, decoder.input_size)
        altered = prefix.clone()
        altered[:, 3:] += 10
        original = decoder(prefix, memory).last_hidden_state
        changed = decoder(altered, memory).last_hidden_state
        torch.testing.assert_close(original[:, :3], changed[:, :3])
    with pytest.raises(ValueError, match="EncoderBlock"):
        tf.TransformerEncoder(decoder.config)
    with pytest.raises(ValueError, match="DecoderBlock"):
        tf.TransformerDecoder(encoder.config)
