import warnings

import torch
import torch.nn.functional as F

from foreblocks.darts.architecture.base_blocks import (
    ArchitectureConverter,
    BaseMixedSequenceBlock,
    MixedEncoder,
)


def test_arch_weights_eval_mode_uses_softmax():
    block = BaseMixedSequenceBlock(
        input_dim=4,
        latent_dim=8,
        seq_len=6,
        dropout=0.0,
        temperature=0.7,
        num_layers=2,
        num_options=3,
        single_path_search=True,
    )
    block.eval()

    with torch.no_grad():
        weights = block._get_arch_weights()

    logits = block._get_layer_arch_logits()
    expected = F.softmax(logits / 0.7, dim=-1)
    assert torch.allclose(weights, expected, atol=1e-6, rtol=1e-6)


def test_train_no_grad_falls_back_to_deterministic_softmax():
    block = BaseMixedSequenceBlock(
        input_dim=4,
        latent_dim=8,
        seq_len=6,
        dropout=0.0,
        temperature=0.9,
        num_layers=2,
        num_options=3,
        single_path_search=True,
    )
    block.train()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            weights = block._get_arch_weights()

    logits = block._get_layer_arch_logits()
    expected = F.softmax(logits / 0.9, dim=-1)
    assert torch.allclose(weights, expected, atol=1e-6, rtol=1e-6)
    assert any("Falling back to deterministic softmax" in str(w.message) for w in caught)


def test_fixed_encoder_transfer_preserves_mixed_postprocessing():
    torch.manual_seed(7)

    mixed = MixedEncoder(
        input_dim=4,
        latent_dim=8,
        seq_len=6,
        dropout=0.0,
        temperature=0.01,
        single_path_search=False,
    )
    with torch.no_grad():
        mixed.alphas.copy_(torch.tensor([10.0, -10.0, -10.0]))
        mixed.layer_alpha_offsets.zero_()
    mixed.eval()

    fixed = ArchitectureConverter.create_fixed_encoder(
        mixed,
        num_layers=2,
        dropout=0.0,
    )
    fixed.eval()

    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        mixed_out, mixed_ctx, mixed_state = mixed(x)
        fixed_out, fixed_ctx, fixed_state = fixed(x)

    assert fixed.normalizer is not None
    assert fixed.context_proj is not None
    assert torch.allclose(mixed_out, fixed_out, atol=1e-4, rtol=1e-4)
    assert torch.allclose(mixed_ctx, fixed_ctx, atol=1e-4, rtol=1e-4)
    assert torch.allclose(mixed_state[0], fixed_state[0], atol=1e-4, rtol=1e-4)
    assert torch.allclose(mixed_state[1], fixed_state[1], atol=1e-4, rtol=1e-4)
