import math
import unittest

import torch

from foreblocks.modules.attention.implementations.autocor_att import (
    AutoCorrelation,
    AutoCorrelationLayer,
)


def _official_time_delay_agg(values, corr, factor=1):
    """Transcription of Autoformer's time_delay_agg_training (thuml/Autoformer).

    values, corr: [B, H, E, L]  -> [B, H, E, L]
    """
    B, head, channel, length = values.shape
    top_k = int(factor * math.log(length))
    mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
    index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]  # [top_k]
    weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
    tmp_corr = torch.softmax(weights, dim=-1)  # [B, top_k]
    agg = torch.zeros_like(values)
    for i in range(top_k):
        pattern = torch.roll(values, -int(index[i]), -1)
        agg = agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return agg


class TestAutoCorrelation(unittest.TestCase):
    def test_time_delay_agg_matches_autoformer(self):
        torch.manual_seed(0)
        torch.set_default_dtype(torch.float64)
        ac = AutoCorrelation(factor=1).double()

        B, H, E, L = 2, 3, 4, 16
        values = torch.randn(B, H, E, L)  # [B, H, D, L] expected by method
        corr = torch.randn(B, H, E, L)
        corr_blhd = corr.permute(0, 3, 1, 2).contiguous()  # [B, L, H, D]

        out = ac.time_delay_agg(values, corr_blhd)  # [B, H, D, L]
        ref = _official_time_delay_agg(values, corr)
        self.assertLess((out - ref).abs().max().item(), 1e-12)
        torch.set_default_dtype(torch.float32)

    def test_forward_shapes(self):
        torch.manual_seed(0)
        ac = AutoCorrelation(factor=1, output_attention=True)
        B, L, H, E = 2, 24, 4, 8
        q = torch.randn(B, L, H, E)
        k = torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E)
        out, attn = ac(q, k, v, None)
        self.assertEqual(out.shape, (B, L, H, E))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_layer_forward_and_grad(self):
        torch.manual_seed(0)
        layer = AutoCorrelationLayer(
            AutoCorrelation(factor=1), d_model=32, n_heads=4
        )
        x = torch.randn(2, 24, 32, requires_grad=True)
        out, _ = layer(x, x, x, None)
        self.assertEqual(out.shape, (2, 24, 32))
        out.pow(2).mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_layer_accepts_spectral_impl_signature(self):
        # SpectralAttentionImpl calls with 8 positional args.
        layer = AutoCorrelationLayer(
            AutoCorrelation(factor=1), d_model=32, n_heads=4
        )
        x = torch.randn(2, 24, 32)
        out, _ = layer(x, x, x, None, None, False, False)
        self.assertEqual(out.shape, (2, 24, 32))

    def test_through_multiattention(self):
        from foreblocks.modules.attention.config import AttentionConfig
        from foreblocks.modules.attention.multi_att import MultiAttention

        m = MultiAttention(AttentionConfig.from_legacy_kwargs(
            d_model=32, n_heads=4, dropout=0.0, attention_type="autocor", freq_modes=8
        ))
        x = torch.randn(2, 24, 32)
        o = m(x, x, x)
        o = o[0] if isinstance(o, tuple) else o
        self.assertEqual(o.shape, (2, 24, 32))
        self.assertTrue(torch.isfinite(o).all().item())


if __name__ == "__main__":
    unittest.main()
