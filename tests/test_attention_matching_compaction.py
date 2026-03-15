import torch

from foreblocks.tf.attention.multi_att import MultiAttention
from foreblocks.tf.transformer import TransformerEncoderLayer


def test_multiattention_attention_matching_compacts_cache():
    attn = MultiAttention(
        d_model=32,
        n_heads=4,
        dropout=0.0,
        attention_type="standard",
        use_mla=False,
        use_attention_matching_compaction=True,
        attention_matching_trigger_len=8,
        attention_matching_min_keep=4,
        attention_matching_keep_ratio=0.5,
    )
    attn.eval()

    layer_state = {}
    attn(torch.randn(1, 8, 32), torch.randn(1, 8, 32), torch.randn(1, 8, 32), is_causal=True, layer_state=layer_state)
    out, _, layer_state = attn(
        torch.randn(1, 1, 32),
        torch.randn(1, 1, 32),
        torch.randn(1, 1, 32),
        is_causal=True,
        layer_state=layer_state,
    )

    cache = layer_state["paged_cache"]
    assert out.shape == (1, 1, 32)
    assert cache.get_seq_len(0) < cache.get_logical_seq_len(0)
    assert cache.get_logical_seq_len(0) == 9
    positions = cache.gather_positions_for_seq(0)
    beta = cache.gather_beta_for_seq(0)
    assert positions.numel() == cache.get_seq_len(0)
    assert beta.shape == (cache.Hkv, cache.get_seq_len(0))
    assert torch.all(positions[:-1] <= positions[1:])


def test_transformer_encoder_attention_matching_runs():
    layer = TransformerEncoderLayer(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        use_attention_matching_compaction=True,
        attention_matching_trigger_len=8,
        attention_matching_min_keep=4,
        attention_matching_keep_ratio=0.5,
    )
    layer.eval()

    out, _ = layer(torch.randn(1, 10, 32))
    assert out.shape == (1, 10, 32)
    assert layer.self_attn_std.use_mla is False
