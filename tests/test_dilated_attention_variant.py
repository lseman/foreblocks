import torch

from foreblocks.modules.attention.multi_att import MultiAttention


def test_dilated_sliding_window_output_shape_and_weights():
    torch.manual_seed(0)
    attn = MultiAttention(
        d_model=32,
        n_heads=4,
        dropout=0.0,
        attention_type="dilated_sliding_window",
        use_mla=False,
        window_size=3,
        attention_dilation=2,
        dilated_window_size=8,
    ).eval()
    x = torch.randn(2, 10, 32)

    out, weights, _ = attn(x, x, x, is_causal=True, need_weights=True)

    assert out.shape == (2, 10, 32)
    assert weights is not None
    assert weights.shape == (2, 4, 10, 10)


def test_dilated_sliding_window_causal_output_ignores_future_tokens():
    torch.manual_seed(1)
    attn = MultiAttention(
        d_model=32,
        n_heads=4,
        dropout=0.0,
        attention_type="dilated_window",
        use_mla=False,
        window_size=3,
        attention_dilation=3,
        dilated_window_size=9,
    ).eval()
    x1 = torch.randn(2, 12, 32)
    x2 = x1.clone()
    x2[:, 8:] = torch.randn_like(x2[:, 8:])

    with torch.no_grad():
        out1, _, _ = attn(x1, x1, x1, is_causal=True)
        out2, _, _ = attn(x2, x2, x2, is_causal=True)

    assert (out1[:, :8] - out2[:, :8]).abs().max().item() < 1e-5
