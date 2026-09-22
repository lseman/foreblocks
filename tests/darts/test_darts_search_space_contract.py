"""Search choices must execute and survive conversion to fixed models."""

import pytest
import torch
import torch.nn.functional as F

from darts.architecture.blocks.attention import SelfAttention
from darts.architecture.blocks.bridges import AttentionBridge
from darts.architecture.darts.converter import ArchitectureConverter
from darts.architecture.darts.mixed_encoder_decoder import MixedDecoder, MixedEncoder
from darts.architecture.darts.time_series_darts import TimeSeriesDARTS


@pytest.mark.parametrize("mode", SelfAttention.CAUSAL_MODES)
def test_causal_attention_choice_has_forward_and_fixed_equivalent(mode: str) -> None:
    torch.manual_seed(2)
    mixed = SelfAttention(8, heads=2, causal=True, attention_type="auto",
                          position_mode="none", dropout=0.0).eval()
    assert mixed.MODES == SelfAttention.CAUSAL_MODES
    assert mixed.attn_alphas.numel() == len(mixed.MODES)
    with torch.no_grad():
        mixed.attn_alphas.fill_(-30.0)
        mixed.attn_alphas[mixed.MODES.index(mode)] = 30.0
    fixed = SelfAttention(8, heads=2, causal=True, attention_type=mode,
                          position_mode="none", dropout=0.0).eval()
    fixed.load_state_dict(mixed.state_dict(), strict=False)
    x = torch.randn(2, 12, 8)
    later = x.clone()
    later[:, 6:] += 10.0
    with torch.no_grad():
        searched = mixed(x)
        deployed = fixed(x)
        causal_output = fixed(later)
    assert torch.allclose(searched, deployed, atol=1e-5)
    assert torch.allclose(deployed[:, :6], causal_output[:, :6], atol=1e-5)


def test_three_choice_causal_attention_logits_expand_on_load() -> None:
    attention = SelfAttention(8, heads=2, causal=True, attention_type="auto")
    old_state = attention.state_dict()
    old_state["attn_alphas"] = torch.tensor([1., 4., 5.])
    attention.load_state_dict(old_state, strict=True)
    assert torch.equal(
        attention.attn_alphas.detach(), torch.tensor([1., -30., -30., 4., 5.])
    )


@pytest.mark.parametrize("with_bias", [False, True])
def test_causal_linear_matches_explicit_feature_kernel(with_bias: bool) -> None:
    torch.manual_seed(10)
    attention = SelfAttention(8, heads=2, causal=True,
                              attention_type="linear", position_mode="none").eval()
    q = torch.randn(2, 2, 9, 4, requires_grad=True)
    k = torch.randn(2, 2, 9, 4, requires_grad=True)
    v = torch.randn(2, 2, 9, 4, requires_grad=True)
    bias = torch.randn(1, 2, 9, 9) if with_bias else None
    qf = F.elu(q * attention.scale) + 1.0
    kf = F.elu(k) + 1.0
    scores = qf @ kf.transpose(-2, -1)
    if bias is not None:
        scores = scores * bias.exp()
    mask = torch.ones(9, 9, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, 0.0)
    expected = (scores @ v) / scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    actual = attention._linear_kernel(q, k, v, bias, 9)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    actual.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all()
               for t in (q, k, v))


def test_causal_linear_half_precision_prefix_is_finite() -> None:
    attention = SelfAttention(8, heads=2, causal=True,
                              attention_type="linear", position_mode="none")
    q = torch.full((1, 2, 256, 4), 8.0, dtype=torch.bfloat16)
    k = torch.full_like(q, 8.0)
    v = torch.full_like(q, 2.0)
    output = attention._linear_kernel(q, k, v, None, 256)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert torch.allclose(output.float(), torch.full_like(output.float(), 2.0),
                          atol=1e-2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("position", ["alibi", "seasonal", "relative"])
def test_linear_lag_bias_matches_pairwise_reference(
    causal: bool, position: str,
) -> None:
    torch.manual_seed(13)
    attention = SelfAttention(8, heads=2, causal=causal,
                              attention_type="linear", position_mode=position).eval()
    with torch.no_grad():
        attention.relative_pos_bias.normal_(std=0.2)
    q_raw = torch.randn(2, 2, 13, 4, requires_grad=True)
    k_raw = torch.randn(2, 2, 13, 4, requires_grad=True)
    v = torch.randn(2, 2, 13, 4, requires_grad=True)
    q, k, bias = attention._apply_position_mode(q_raw, k_raw, position)
    fast = attention._linear_kernel(q, k, v, None, 13, position_mode=position)
    qf = F.elu(q * attention.scale) + 1.0
    kf = F.elu(k) + 1.0
    scores = (qf @ kf.transpose(-2, -1)) * bias.exp()
    if causal:
        scores = scores.masked_fill(
            ~torch.ones(13, 13, dtype=torch.bool).tril(), 0.0
        )
    reference = (scores @ v) / scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    assert torch.allclose(fast, reference, atol=2e-5, rtol=2e-5)
    inputs = (q_raw, k_raw, v, attention.relative_pos_bias)
    fast_grads = torch.autograd.grad(fast.sum(), inputs, allow_unused=True,
                                     retain_graph=True)
    ref_grads = torch.autograd.grad(reference.sum(), inputs, allow_unused=True)
    for fast_grad, ref_grad in zip(fast_grads, ref_grads):
        if ref_grad is None:
            assert fast_grad is None
        else:
            assert torch.allclose(fast_grad, ref_grad, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("position", ["alibi", "seasonal", "relative"])
def test_fixed_linear_attention_does_not_build_pairwise_bias(
    position: str, monkeypatch,
) -> None:
    attention = SelfAttention(
        8, heads=2, causal=True, attention_type="linear", position_mode=position
    ).eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("pairwise position bias was materialized")

    monkeypatch.setattr(attention, "_build_relative_bias", forbidden)
    monkeypatch.setattr(attention, "_apply_relative_pos", forbidden)
    with torch.no_grad():
        output = attention(torch.randn(2, 64, 8))
    assert output.shape == (2, 64, 8)
    assert torch.isfinite(output).all()


def test_causal_relative_lag_normalization_ignores_future_bias() -> None:
    torch.manual_seed(14)
    attention = SelfAttention(
        8, heads=2, causal=True, attention_type="linear",
        position_mode="relative",
    ).eval()
    length = 16
    with torch.no_grad():
        attention.relative_pos_bias.zero_()
        # Positive key-minus-query offsets are future positions.
        attention.relative_pos_bias[length : 2 * length - 1] = 1000.0
    x = torch.randn(1, length, 8)
    with torch.no_grad():
        output = attention(x)
    assert torch.isfinite(output).all()
    weights = attention._linear_lag_weights(
        "relative", length, torch.float32, x.device
    )
    assert torch.all(weights[:, :length - 1] == 0)
    assert torch.all(weights[:, length - 1 :] == 1)


@pytest.mark.parametrize("position", ["none", "alibi", "relative"])
def test_causal_probsparse_is_sparse_distinct_and_prefix_invariant(position: str) -> None:
    torch.manual_seed(11)
    sparse = SelfAttention(8, heads=2, causal=True, attention_type="probsparse",
                           position_mode=position, dropout=0.0)
    sparse.PROBSPARSE_C = 1
    sparse.eval()
    dense = SelfAttention(8, heads=2, causal=True, attention_type="sdp",
                          position_mode=position, dropout=0.0).eval()
    dense.load_state_dict(sparse.state_dict(), strict=False)
    x = torch.randn(2, 80, 8)
    altered = x.clone()
    altered[:, 45:] += 20.0
    with torch.no_grad():
        full = sparse(x)
        changed = sparse(altered)
        prefix = sparse(x[:, :45])
        dense_out = dense(x)
    assert torch.allclose(full[:, :45], changed[:, :45], atol=1e-5)
    assert torch.allclose(full[:, :45], prefix, atol=1e-5)
    assert not torch.allclose(full, dense_out, atol=1e-4)
    sparse.train()
    output = sparse(x.requires_grad_())
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("causal", [False, True])
def test_long_probsparse_search_matches_fixed_model(causal: bool) -> None:
    torch.manual_seed(12)
    mixed = SelfAttention(8, heads=2, causal=causal, attention_type="auto",
                          position_mode="none", dropout=0.0).eval()
    mixed.PROBSPARSE_C = 1
    with torch.no_grad():
        mixed.attn_alphas.fill_(-30.0)
        mixed.attn_alphas[mixed.MODES.index("probsparse")] = 30.0
    fixed = SelfAttention(8, heads=2, causal=causal,
                          attention_type="probsparse", position_mode="none",
                          dropout=0.0).eval()
    fixed.PROBSPARSE_C = 1
    fixed.load_state_dict(mixed.state_dict(), strict=False)
    x = torch.randn(2, 80, 8)
    with torch.no_grad():
        assert torch.allclose(mixed(x), fixed(x), atol=1e-5)
        assert torch.allclose(fixed(x), fixed(x), atol=1e-6)


@pytest.mark.parametrize("position", SelfAttention.POSITION_MODES)
def test_causal_position_choice_has_fixed_equivalent(position: str) -> None:
    torch.manual_seed(3)
    mixed = SelfAttention(8, heads=2, causal=True, attention_type="sdp",
                          position_mode="auto", dropout=0.0).eval()
    with torch.no_grad():
        mixed.position_alphas.fill_(-30.0)
        mixed.position_alphas[mixed.POSITION_MODES.index(position)] = 30.0
    fixed = SelfAttention(8, heads=2, causal=True, attention_type="sdp",
                          position_mode=position, dropout=0.0).eval()
    fixed.load_state_dict(mixed.state_dict(), strict=False)
    x = torch.randn(2, 10, 8)
    with torch.no_grad():
        assert torch.allclose(mixed(x), fixed(x), atol=1e-5)


@pytest.mark.parametrize("mode", SelfAttention.MODES)
def test_encoder_attention_choice_has_fixed_equivalent(mode: str) -> None:
    torch.manual_seed(7)
    mixed = SelfAttention(8, heads=2, attention_type="auto",
                          position_mode="none", dropout=0.0).eval()
    with torch.no_grad():
        mixed.attn_alphas.fill_(-30.0)
        mixed.attn_alphas[mixed.MODES.index(mode)] = 30.0
    fixed = SelfAttention(8, heads=2, attention_type=mode,
                          position_mode="none", dropout=0.0).eval()
    fixed.load_state_dict(mixed.state_dict(), strict=False)
    x = torch.randn(2, 12, 8)
    with torch.no_grad():
        assert torch.allclose(mixed(x), fixed(x), atol=1e-5)


@pytest.mark.parametrize("patch", [
    "direct", "patch_8", "patch_16", "patch_32", "multi_scale_patch",
    "hierarchical", "variate_tokens",
])
def test_encoder_patch_choice_survives_fixed_transfer(patch: str) -> None:
    torch.manual_seed(4)
    mixed = MixedEncoder(4, 8, seq_len=32, dropout=0.0,
                         transformer_self_attention_type="sdp",
                         transformer_ffn_variant="swiglu").eval()
    transformer = mixed.transformer
    with torch.no_grad():
        transformer.patch_alpha_logits.fill_(-30.0)
        transformer.patch_alpha_logits[transformer.patch_mode_names.index(patch)] = 30.0
        for layer in transformer.layers:
            attention = layer["self_attn"]
            attention.position_alphas.fill_(-30.0)
            attention.position_alphas[attention.POSITION_MODES.index("rope")] = 30.0
    fixed = ArchitectureConverter.create_fixed_encoder(mixed, dropout=0.0).eval()
    x = torch.randn(2, 32, 4)
    with torch.no_grad():
        searched = mixed(x)[0]
        deployed = fixed(x)[0]
    assert fixed.rnn.patching_mode == patch
    assert torch.allclose(searched, deployed, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mode", SelfAttention.CAUSAL_MODES)
def test_decoder_self_attention_choice_survives_fixed_transfer(mode: str) -> None:
    torch.manual_seed(6)
    mixed = MixedDecoder(
        4, 8, seq_len=12, dropout=0.0, use_learned_memory_pooling=False,
        transformer_self_attention_type="auto",
        transformer_cross_attention_type="sdp",
        transformer_ffn_variant="swiglu",
    ).eval()
    with torch.no_grad():
        for layer in mixed.transformer.layers:
            attention = layer["self_attn"]
            attention.attn_alphas.fill_(-30.0)
            attention.attn_alphas[attention.MODES.index(mode)] = 30.0
            attention.position_alphas.fill_(-30.0)
            attention.position_alphas[attention.POSITION_MODES.index("rope")] = 30.0
            cross = layer["cross_attn"]
            cross.position_alphas.fill_(-30.0)
            cross.position_alphas[cross.POSITION_MODES.index("rope")] = 30.0
    fixed = ArchitectureConverter.create_fixed_decoder(mixed, dropout=0.0).eval()
    tgt = torch.randn(2, 5, 4)
    memory = torch.randn(2, 12, 8)
    with torch.no_grad():
        searched = mixed(tgt, memory, encoder_output=memory)[0]
        deployed = fixed(tgt, memory)[0]
    assert fixed.rnn.layers[0]["self_attn"].attention_type == mode
    assert torch.allclose(searched, deployed, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mode", AttentionBridge.MODES)
def test_decoder_cross_attention_choice_survives_fixed_transfer(mode: str) -> None:
    torch.manual_seed(8)
    mixed = MixedDecoder(
        4, 8, seq_len=12, dropout=0.0, use_learned_memory_pooling=False,
        transformer_self_attention_type="sdp",
        transformer_cross_attention_type="auto",
        transformer_ffn_variant="swiglu",
    ).eval()
    with torch.no_grad():
        for layer in mixed.transformer.layers:
            attention = layer["self_attn"]
            attention.position_alphas.fill_(-30.0)
            attention.position_alphas[attention.POSITION_MODES.index("rope")] = 30.0
            cross = layer["cross_attn"]
            cross.attn_alphas.fill_(-30.0)
            cross.attn_alphas[cross.MODES.index(mode)] = 30.0
            cross.position_alphas.fill_(-30.0)
            cross.position_alphas[cross.POSITION_MODES.index("rope")] = 30.0
    fixed = ArchitectureConverter.create_fixed_decoder(mixed, dropout=0.0).eval()
    tgt = torch.randn(2, 5, 4)
    memory = torch.randn(2, 12, 8)
    with torch.no_grad():
        searched = mixed(tgt, memory, encoder_output=memory)[0]
        deployed = fixed(tgt, memory)[0]
    assert fixed.rnn.layers[0]["cross_attn"].attention_type == mode
    assert torch.allclose(searched, deployed, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("position", AttentionBridge.POSITION_MODES)
def test_decoder_cross_position_survives_fixed_transfer(position: str) -> None:
    torch.manual_seed(9)
    mixed = MixedDecoder(
        4, 8, seq_len=12, dropout=0.0, use_learned_memory_pooling=False,
        transformer_self_attention_type="sdp",
        transformer_cross_attention_type="sdp",
        transformer_ffn_variant="swiglu",
    ).eval()
    with torch.no_grad():
        for layer in mixed.transformer.layers:
            attention = layer["self_attn"]
            attention.position_alphas.fill_(-30.0)
            attention.position_alphas[attention.POSITION_MODES.index("rope")] = 30.0
            cross = layer["cross_attn"]
            cross.position_alphas.fill_(-30.0)
            cross.position_alphas[cross.POSITION_MODES.index(position)] = 30.0
    fixed = ArchitectureConverter.create_fixed_decoder(mixed, dropout=0.0).eval()
    tgt = torch.randn(2, 5, 4)
    memory = torch.randn(2, 12, 8)
    with torch.no_grad():
        searched = mixed(tgt, memory, encoder_output=memory)[0]
        deployed = fixed(tgt, memory)[0]
    assert fixed.rnn.layers[0]["cross_attn"].position_mode == position
    assert torch.allclose(searched, deployed, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("style,expected", [
    ("autoregressive", 1.0), ("informer", 2.0), ("autoformer", 3.0),
])
def test_decoder_style_choice_has_fixed_routing(style: str, expected: float) -> None:
    class SearchDecoder(torch.nn.Module):
        def __init__(self, selected: str):
            super().__init__()
            names = ("autoregressive", "informer", "autoformer")
            self.decode_style_alphas = torch.nn.Parameter(
                torch.tensor([30.0 if name == selected else -30.0 for name in names])
            )

        def get_decode_style_weights(self):
            return torch.softmax(self.decode_style_alphas, dim=0)

    class FixedStyleDecoder(torch.nn.Module):
        def __init__(self, selected: str):
            super().__init__()
            self.decode_style = selected
            self.anchor = torch.nn.Parameter(torch.zeros(()))

    class Stub(TimeSeriesDARTS):
        def __init__(self, decoder):
            torch.nn.Module.__init__(self)
            self.forecast_decoder = decoder

        def _decode_autoregressive_path(self, *args):
            return torch.tensor(1.0)

        def _decode_parallel_informer_path(self, *args):
            return torch.tensor(2.0)

        def _decode_autoformer_path(self, *args):
            return torch.tensor(3.0)

    inputs = dict(
        x_seq=torch.ones(1, 2, 1), x_future=None, decoder_targets=None,
        teacher_forcing_ratio=0.0, memory=torch.ones(1, 2, 1),
        encoder_output=torch.ones(1, 2, 1), decoder_hidden=None,
    )
    searched = Stub(SearchDecoder(style))._decode_with_style(**inputs)
    fixed = Stub(FixedStyleDecoder(style))._decode_with_style(**inputs)
    assert searched.item() == expected
    assert fixed.item() == expected
