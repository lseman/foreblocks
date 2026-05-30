import unittest

import torch

from foreblocks.transformer.attention.modules.dwt_att import DWTAttention


class TestDWTAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = DWTAttention(d_model=16, n_heads=2, dropout=0.0, modes=8)

    def test_haar_roundtrip_even_and_odd(self):
        x = torch.randn(1, 2, 8, 8)
        r = self.m._haar_idwt(self.m._haar_dwt(x), 8)
        self.assertLess((r - x).abs().max().item(), 1e-5)
        x2 = torch.randn(1, 2, 7, 8)
        r2 = self.m._haar_idwt(self.m._haar_dwt(x2), 7)
        self.assertLess((r2 - x2).abs().max().item(), 1e-5)

    def test_self_attention_shape_and_grad(self):
        x = torch.randn(2, 20, 16, requires_grad=True)
        out, w = self.m(x)
        self.assertEqual(out.shape, (2, 20, 16))
        self.assertIsNone(w)
        out.pow(2).mean().backward()
        self.assertIsNotNone(self.m.wavelet_weight.grad)
        self.assertTrue(torch.isfinite(self.m.wavelet_weight.grad).all().item())
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_need_weights_returns_attention(self):
        out, w = self.m(torch.randn(2, 20, 16), need_weights=True)
        self.assertIsNotNone(w)
        # attention is over the m kept wavelet coefficients
        self.assertEqual(w.shape[-1], w.shape[-2])

    def test_cross_attention_different_lengths(self):
        out, _ = self.m(torch.randn(2, 15, 16), torch.randn(2, 23, 16), torch.randn(2, 23, 16))
        self.assertEqual(out.shape, (2, 15, 16))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_modes_exceed_coeffs(self):
        out, _ = self.m(torch.randn(2, 6, 16))  # few coefficients
        self.assertEqual(out.shape, (2, 6, 16))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_through_multiattention(self):
        from foreblocks.transformer.attention.multi_att import MultiAttention

        m = MultiAttention(
            d_model=32, n_heads=4, dropout=0.0, attention_type="dwt", freq_modes=8
        )
        x = torch.randn(2, 24, 32)
        o = m(x, x, x)
        o = o[0] if isinstance(o, tuple) else o
        self.assertEqual(o.shape, (2, 24, 32))
        self.assertTrue(torch.isfinite(o).all().item())


if __name__ == "__main__":
    unittest.main()
