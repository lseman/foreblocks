import pytest
import torch

from foreblocks.models.transformer import GenerationConfig, TransformerConfig
from foreblocks.models.transformer.runtime.outputs import (
    TransformerDecoderOutput,
    TransformerEncoderOutput,
)
from foreblocks.models.transformer.runtime.state import DecoderState
from foreblocks.models.transformer.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from foreblocks.modules.skip.gateskip import BudgetScheduler
from foreblocks.modules.skip.mod import MoDBudgetScheduler
from foreblocks.modules.attention.multi_att import MultiAttention
from foreblocks.modules.attention.cache.kv import StaticKVCache
from foreblocks.modules.attention.cache import KVCacheProtocol
from foreblocks.modules.attention.backends import (
    ATTENTION_BACKENDS,
    register_attention_backend,
)
from foreblocks.modules.attention.masking import build_attention_mask


def _optimizer_param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }


def test_transformer_schedulers_do_not_step_during_eval():
    gate_scheduler = BudgetScheduler(b_start=1.0, b_end=0.5, total_steps=10)

    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        use_gateskip=True,
        gate_budget=0.8,
        use_mod=False,
    )
    model.set_budget_scheduler(gate_scheduler)

    x = torch.randn(2, 6, 2)

    model.eval()
    with torch.no_grad():
        _ = model(x)

    assert gate_scheduler._step == 0


def test_transformer_gate_scheduler_steps_during_training():
    gate_scheduler = BudgetScheduler(b_start=1.0, b_end=0.5, total_steps=10)

    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        use_gateskip=False,
        use_mod=False,
    )
    model.set_budget_scheduler(gate_scheduler)

    x = torch.randn(2, 6, 2)

    model.train()
    _ = model(x)

    assert gate_scheduler._step == 1


def test_transformer_mod_scheduler_steps_only_during_training():
    mod_scheduler = MoDBudgetScheduler(num_layers=1, total_steps=10)

    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        use_gateskip=False,
        use_mod=True,
        mod_budget_scheduler=mod_scheduler,
    )

    x = torch.randn(2, 6, 2)

    model.eval()
    with torch.no_grad():
        _ = model(x)
    assert mod_scheduler._step == 0

    model.train()
    _ = model(x)
    assert mod_scheduler._step == 1


def test_transformer_attention_params_are_optimizer_visible_before_forward():
    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before_param_count = len(list(model.parameters()))

    _ = model(torch.randn(2, 6, 2))

    assert len(list(model.parameters())) == before_param_count
    assert {id(p) for p in model.parameters()} <= _optimizer_param_ids(optimizer)


def test_shared_hybrid_attention_materializes_configured_backends():
    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        attention_mode="hybrid",
        share_layers=True,
        patch_encoder=False,
    )
    layer = model.shared_layer

    assert layer is not None
    assert layer._attn_backends.get("linear") is not None
    assert layer._attn_backends.get("standard") is not None

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert {id(p) for p in model.parameters()} <= _optimizer_param_ids(optimizer)


def test_standalone_encoder_layer_materializes_selected_attention():
    layer = TransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        layer_attention_type="linear",
    )

    assert layer._attn_backends.get("linear") is not None
    assert layer._attn_backends.get("standard") is None


def test_sdpa_backend_matches_eager_with_gqa_and_padding():
    common = dict(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
        attention_type="standard",
        pos_encoding_type="sinusoidal",
        use_mla=False,
        use_paged_cache=False,
    )
    eager = MultiAttention(**common, attn_implementation="eager").eval()
    sdpa = MultiAttention(**common, attn_implementation="sdpa").eval()
    sdpa.load_state_dict(eager.state_dict())
    x = torch.randn(2, 6, 16)
    padding = torch.tensor(
        [[False, False, False, False, True, True], [False] * 6]
    )

    eager_out, _, _ = eager(
        x, key_padding_mask=padding, is_causal=True
    )
    sdpa_out, _, _ = sdpa(
        x, key_padding_mask=padding, is_causal=True
    )

    torch.testing.assert_close(sdpa_out, eager_out, rtol=1e-5, atol=1e-6)


def test_encoder_structured_output_and_hidden_states():
    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        patch_encoder=False,
    ).eval()
    result = model(
        torch.randn(2, 5, 2),
        output_hidden_states=True,
        return_dict=True,
    )
    assert isinstance(result, TransformerEncoderOutput)
    assert result.last_hidden_state.shape == (2, 5, 8)
    assert result.hidden_states is not None
    assert len(result.hidden_states) == 3


def test_encoder_propagates_padding_mask_to_moe_router():
    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
        use_moe=True,
        num_experts=4,
        top_k=2,
    ).eval()
    padding = torch.tensor(
        [[False, False, True, True], [False, True, False, True]]
    )
    result = model(
        torch.randn(2, 4, 2),
        src_key_padding_mask=padding,
        return_dict=True,
    )
    layer = model.layers[0]
    state = layer.feed_forward.block.last_routing_state
    assert isinstance(result, TransformerEncoderOutput)
    assert state is not None
    assert state.logits.size(0) == int((~padding).sum())


def test_decoder_structured_output_carries_incremental_state():
    decoder = TransformerDecoder(
        input_size=2,
        output_size=3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
        informer_like=False,
    ).eval()
    result = decoder(
        torch.randn(2, 3, 2),
        torch.randn(2, 4, 8),
        incremental_state={},
        return_incremental_state=True,
        output_hidden_states=True,
        return_dict=True,
    )
    assert isinstance(result, TransformerDecoderOutput)
    assert result.last_hidden_state.shape == (2, 3, 3)
    assert result.hidden_states is not None
    assert len(result.hidden_states) == 2
    assert result.past_key_values is not None
    assert len(result.past_key_values["layers"]) == 1


def test_depth_scaled_initialization_targets_residual_outputs():
    model = TransformerEncoder(
        input_size=16,
        d_model=64,
        nhead=4,
        num_layers=4,
        dim_feedforward=128,
        patch_encoder=False,
        initializer_range=0.04,
        depth_scaled_init=True,
    )
    layer = model.layers[0]
    attention = layer._attn_backends.get("standard")
    expected = 0.04 / (2.0 * 4) ** 0.5
    assert attention is not None
    assert attention.q_proj.weight.std().item() == pytest.approx(0.04, rel=0.15)
    assert attention.out_proj[2].weight.std().item() == pytest.approx(
        expected, rel=0.15
    )


def test_decoder_specific_modules_use_transformer_initialization():
    decoder = TransformerDecoder(
        input_size=2,
        output_size=3,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        patch_encoder=False,
        use_time_encoding=True,
        initializer_range=0.03,
    )

    assert decoder.output_projection.weight.std().item() == pytest.approx(
        0.03, rel=0.2
    )
    assert torch.count_nonzero(decoder.output_projection.bias) == 0


def test_encoder_ct_patch_modules_use_transformer_initialization():
    encoder = TransformerEncoder(
        input_size=3,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        ct_patchtst=True,
        ct_patch_len=8,
        initializer_range=0.03,
    )

    assert encoder.ct_patch_embed.weight.std().item() == pytest.approx(0.03, rel=0.2)
    assert encoder.ct_channel_fuse.weight.std().item() == pytest.approx(0.03, rel=0.2)
    assert torch.count_nonzero(encoder.ct_patch_embed.bias) == 0
    assert torch.count_nonzero(encoder.ct_channel_fuse.bias) == 0


def test_transformer_config_rejects_incompatible_residual_policies_early():
    with pytest.raises(ValueError, match="incompatible"):
        TransformerConfig(
            use_gateskip=True,
            options={"use_attention_residual": True},
        )


def test_transformer_config_normalizes_aliases_and_rejects_option_typos():
    config = TransformerConfig(attention_mode="hybrid_linear")
    assert config.attention.architecture == "hybrid"
    assert config.residual.policy == "standard"
    assert config.cache.implementation == "auto"

    with pytest.raises(ValueError, match="unsupported Transformer options"):
        TransformerConfig(options={"use_gateksip": True})


def test_runtime_mhc_arguments_cannot_mutate_layer_configuration():
    layer = TransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        use_mhc=False,
    )
    with pytest.raises(ValueError, match="runtime mHC overrides"):
        layer(torch.randn(1, 3, 8), use_mhc=True)
    assert layer.use_mhc is False


def test_incremental_decoder_returns_typed_mapping_compatible_state():
    decoder = TransformerDecoder(
        input_size=2,
        output_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
        informer_like=False,
        dropout=0.0,
    ).eval()
    result = decoder(
        torch.randn(1, 2, 2),
        torch.randn(1, 3, 8),
        incremental_state={},
        return_incremental_state=True,
        return_dict=True,
    )
    assert isinstance(result.past_key_values, DecoderState)
    assert result.past_key_values.layers[0].self_attention is not None


def test_static_cache_incremental_decode_matches_full_decode_last_token():
    torch.manual_seed(7)
    decoder = TransformerDecoder(
        input_size=2,
        output_size=3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
        informer_like=False,
        cache_implementation="static",
        max_seq_len=8,
        dropout=0.0,
    ).eval()
    memory = torch.randn(1, 4, 8)
    prefix = torch.randn(1, 3, 2)
    next_token = torch.randn(1, 1, 2)

    full = decoder(torch.cat([prefix, next_token], dim=1), memory)
    first = decoder(
        prefix,
        memory,
        return_incremental_state=True,
        return_dict=True,
    )
    second = decoder(
        next_token,
        memory,
        incremental_state=first.past_key_values,
        return_incremental_state=True,
        return_dict=True,
    )

    cache = second.past_key_values["layers"][0]["self_attn"]["static_cache"]
    assert cache.is_compileable
    assert cache.get_seq_length() == 4
    torch.testing.assert_close(
        second.last_hidden_state[:, -1],
        full.last_hidden_state[:, -1],
        rtol=1e-4,
        atol=1e-5,
    )


def test_structured_output_collects_router_states():
    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        patch_encoder=False,
        use_moe=True,
        num_experts=4,
        top_k=2,
    ).eval()
    result = model(torch.randn(1, 4, 2), return_dict=True)
    assert result.router_states is not None
    assert len(result.router_states) == 2


def test_static_cache_update_is_fullgraph_compileable():
    cache = StaticKVCache(
        1, 2, 8, 4, device="cpu", dtype=torch.float32
    )

    def update_cache(k, v):
        cached_k, cached_v = cache.update(k, v)
        return cached_k + cached_v

    compiled = torch.compile(update_cache, backend="eager", fullgraph=True)
    x = torch.randn(1, 2, 1, 4)
    result = compiled(x, x)
    assert result.shape == (1, 2, 8, 4)
    assert cache.get_seq_length() == 1


def test_standard_attention_dispatches_prefill_and_decode(monkeypatch):
    attention = MultiAttention(
        d_model=8,
        n_heads=2,
        dropout=0.0,
        use_mla=False,
        use_paged_cache=True,
    ).eval()
    calls = {"prefill": 0, "decode": 0}
    original_prefill = attention.impl.prefill
    original_decode = attention.impl.decode

    def prefill(*args, **kwargs):
        calls["prefill"] += 1
        return original_prefill(*args, **kwargs)

    def decode(*args, **kwargs):
        calls["decode"] += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(attention.impl, "prefill", prefill)
    monkeypatch.setattr(attention.impl, "decode", decode)
    x = torch.randn(1, 2, 8)
    attention(x, is_causal=True)
    static = StaticKVCache(1, 2, 4, 4, device="cpu", dtype=x.dtype)
    attention(x[:, :1], is_causal=True, layer_state={"static_cache": static})
    assert calls == {"prefill": 1, "decode": 1}


def test_structured_outputs_capture_attention_weights_on_request():
    encoder = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
        dropout=0.0,
    ).eval()
    result = encoder(
        torch.randn(1, 4, 2),
        output_attentions=True,
        return_dict=True,
    )
    assert result.attentions is not None
    assert result.attentions[0].shape == (1, 2, 4, 4)


def test_static_cache_supports_per_sequence_cache_positions():
    cache = StaticKVCache(2, 1, 6, 2, device="cpu", dtype=torch.float32)
    values = torch.arange(8, dtype=torch.float32).view(2, 1, 2, 2)
    cache.update(values, values, torch.tensor([[0, 1], [2, 3]]))
    assert cache.get_seq_lengths().tolist() == [2, 4]
    torch.testing.assert_close(cache.keys[1, :, 2:4], values[1])


def test_cache_aware_mask_uses_explicit_query_positions():
    q = torch.zeros(2, 1, 1, 4)
    mask = build_attention_mask(
        query=q,
        key_length=5,
        is_causal=True,
        cache_position=torch.tensor([[1], [3]]),
        key_lengths=torch.tensor([2, 4]),
    )
    assert mask[:, 0, 0].tolist() == [
        [False, False, True, True, True],
        [False, False, False, False, True],
    ]


def test_custom_attention_backend_registry_executes_runner():
    calls = []

    def runner(q, k, v, **kwargs):
        calls.append(kwargs["attention_mask"])
        return torch.zeros_like(q)

    register_attention_backend(
        "test_backend", runner, mask_builder=build_attention_mask
    )
    attention = MultiAttention(
        d_model=8, n_heads=2, dropout=0.0, attn_implementation="test_backend"
    ).eval()
    output, _, _ = attention(torch.randn(1, 2, 8), is_causal=True)
    assert output.shape == (1, 2, 8)
    assert calls[0].shape == (1, 2, 2, 2)


def test_decoder_generate_reuses_static_cache():
    decoder = TransformerDecoder(
        input_size=2,
        output_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
        informer_like=False,
        cache_implementation="static",
        max_seq_len=8,
        dropout=0.0,
    ).eval()
    generated = decoder.generate(
        torch.randn(1, 2, 2), torch.randn(1, 3, 8), 3, return_dict=True
    )
    assert generated.sequences.shape == (1, 3, 2)
    cache = generated.past_key_values["layers"][0]["self_attn"]["static_cache"]
    assert cache.get_seq_length() == 4


def test_static_cache_selectively_advances_batch_rows():
    cache = StaticKVCache(2, 1, 5, 2, device="cpu", dtype=torch.float32)
    first = torch.ones(2, 1, 1, 2)
    cache.update(first, first)
    before = cache.keys[1].clone()
    second = torch.full_like(first, 2.0)
    cache.update(second, second, update_mask=torch.tensor([True, False]))
    assert cache.get_seq_lengths().tolist() == [2, 1]
    torch.testing.assert_close(cache.keys[1], before)


def test_static_cache_conforms_to_public_protocol():
    cache = StaticKVCache(1, 1, 4, 2, device="cpu", dtype=torch.float32)
    assert isinstance(cache, KVCacheProtocol)


def test_attention_backend_exposes_capabilities():
    sdpa = ATTENTION_BACKENDS.get("sdpa")
    assert sdpa.supports("gqa")
    assert not sdpa.supports("attention_weights")
    assert sdpa.supports_compile


def test_decoder_prefill_and_decode_entry_points():
    decoder = TransformerDecoder(
        input_size=2,
        output_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
        informer_like=False,
        cache_implementation="static",
        max_seq_len=8,
        dropout=0.0,
    ).eval()
    memory = torch.randn(2, 3, 8)
    prefix = torch.randn(2, 2, 2)
    _, state = decoder.prefill(prefix, memory)
    output, state = decoder.decode(
        torch.randn(2, 1, 2),
        memory,
        state,
        cache_update_mask=torch.tensor([True, False]),
    )
    cache = state["layers"][0]["self_attn"]["static_cache"]
    assert output.shape == (2, 1, 2)
    assert cache.get_seq_lengths().tolist() == [3, 2]
    assert callable(decoder.compile_prefill(backend="eager"))
    assert callable(decoder.compile_decode(backend="eager"))


def _small_static_decoder():
    return TransformerDecoder(
        input_size=2, output_size=2, d_model=8, nhead=2, num_layers=1,
        dim_feedforward=16, patch_encoder=False, informer_like=False,
        cache_implementation="static", max_seq_len=12, dropout=0.0,
    ).eval()


def test_decoder_cache_snapshot_and_resume():
    decoder = _small_static_decoder()
    memory = torch.randn(1, 3, 8)
    _, state = decoder.prefill(torch.randn(1, 2, 2), memory)
    restored = decoder.load_cache_state_dict(decoder.offload_cache(state))
    cache = restored["layers"][0]["self_attn"]["static_cache"]
    assert cache.get_seq_length() == 2
    output, restored = decoder.decode(torch.randn(1, 1, 2), memory, restored)
    assert output.shape == (1, 1, 2)
    assert cache.get_seq_length() == 3


def test_speculative_decode_rolls_back_rejected_suffix():
    decoder = _small_static_decoder()
    memory = torch.randn(1, 3, 8)
    _, state = decoder.prefill(torch.randn(1, 2, 2), memory)
    output, state, accepted = decoder.speculative_decode(
        torch.randn(1, 3, 2), memory, state,
        verifier_fn=lambda output, draft: 1,
    )
    cache = state["layers"][0]["self_attn"]["static_cache"]
    assert accepted == 1
    assert output.shape[1] == 1
    assert cache.get_seq_length() == 3


def test_generic_beam_search_reorders_static_cache():
    decoder = _small_static_decoder()

    def proposals(prediction, step):
        base = prediction[:, 0]
        values = torch.stack([base, base + 0.1], dim=1)
        scores = prediction.new_tensor([0.0, -0.5]).expand(base.size(0), -1)
        return values, scores

    sequences, scores, state = decoder.beam_search(
        torch.randn(1, 2, 2), torch.randn(1, 3, 8), 2, 2, proposals
    )
    assert sequences.shape == (1, 2, 2)
    assert scores.shape == (1,)
    cache = state["layers"][0]["self_attn"]["static_cache"]
    assert cache.keys.size(0) == 2


def test_transformer_config_drives_structured_output_defaults():
    config = TransformerConfig(
        input_size=2,
        output_size=3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
    )
    encoder = TransformerEncoder(config).eval()
    decoder = TransformerDecoder(config).eval()

    encoded = encoder(torch.randn(1, 4, 2))
    decoded = decoder(torch.randn(1, 2, 2), encoded.last_hidden_state)

    assert isinstance(encoded, TransformerEncoderOutput)
    assert isinstance(decoded, TransformerDecoderOutput)
    assert encoder.config is config
    assert decoder.config is config


def test_transformer_config_promotes_stable_layer_options_to_fields():
    config = TransformerConfig(
        d_model=8,
        nhead=2,
        num_layers=1,
        layer_norm_eps=1e-6,
        use_swiglu=False,
        freq_modes=7,
        gate_lambda=0.25,
        mhc_n_streams=3,
        initializer_range=0.01,
    )

    model = TransformerEncoder(config)
    layer = model._get_layer(0)

    assert model.config is config
    assert layer.config is config
    assert model.gate_lambda == 0.25
    assert model.mhc_n_streams == 3
    assert model.initializer_range == 0.01
    assert layer._attn_backend_cfg["freq_modes"] == 7
    assert type(layer.feed_forward.block).__name__ == "_StandardFeedForwardBlock"


def test_transformer_config_loads_legacy_promoted_options():
    config = TransformerConfig.from_dict(
        {
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "options": {"gate_lambda": 0.4, "use_final_norm": False},
        }
    )

    assert config.gate_lambda == 0.4
    assert config.use_final_norm is False
    assert config.options == {}


@pytest.mark.parametrize("feature", ["use_mhc", "use_attention_residual"])
def test_checkpointing_rejects_stateful_residual_policies(feature):
    with pytest.raises(ValueError, match="use_gradient_checkpointing is incompatible"):
        TransformerConfig(
            d_model=8,
            nhead=2,
            num_layers=1,
            use_gradient_checkpointing=True,
            **{feature: True},
        )


def test_generation_config_is_separate_from_decoder_config():
    config = TransformerConfig(
        input_size=2,
        output_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
    )
    decoder = TransformerDecoder(config).eval()
    result = decoder.generate(
        torch.randn(1, 1, 2),
        torch.randn(1, 3, 8),
        generation_config=GenerationConfig(max_new_tokens=2),
    )

    assert result.sequences.shape == (1, 2, 2)
