import unittest

import torch

from foreblocks.transformer.attention.modules.frequency_att import FrequencyAttention


class TestFrequencyAttention(unittest.TestCase):
    def test_self_attention_shape_and_grad(self):
        torch.manual_seed(0)
        m = FrequencyAttention(d_model=32, n_heads=4, dropout=0.0, modes=8)
        x = torch.randn(2, 20, 32, requires_grad=True)
        out, w = m(x)
        self.assertEqual(out.shape, (2, 20, 32))
        self.assertIsNone(w)
        self.assertTrue(torch.isfinite(out).all().item())
        out.pow(2).mean().backward()
        # complex spectral weights must receive gradient
        self.assertIsNotNone(m.weights1.grad)
        self.assertTrue(torch.isfinite(m.weights1.grad).all().item())
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_cross_attention_different_lengths(self):
        m = FrequencyAttention(d_model=32, n_heads=4, dropout=0.0, modes=8)
        q = torch.randn(2, 15, 32)
        kv = torch.randn(2, 23, 32)
        out, _ = m(q, kv, kv)
        self.assertEqual(out.shape, (2, 15, 32))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_softmax_activation(self):
        m = FrequencyAttention(
            d_model=32, n_heads=4, dropout=0.0, modes=8, activation="softmax"
        )
        out, _ = m(torch.randn(2, 20, 32))
        self.assertEqual(out.shape, (2, 20, 32))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_random_mode_select(self):
        m = FrequencyAttention(
            d_model=32, n_heads=4, dropout=0.0, modes=8, mode_select="random"
        )
        out, _ = m(torch.randn(2, 30, 32))
        self.assertEqual(out.shape, (2, 30, 32))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_modes_exceed_frequency_count(self):
        # short sequence -> fewer rfft bins than requested modes
        m = FrequencyAttention(d_model=32, n_heads=4, dropout=0.0, modes=100)
        out, _ = m(torch.randn(2, 6, 32))
        self.assertEqual(out.shape, (2, 6, 32))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            FrequencyAttention(32, 4, activation="bad")
        with self.assertRaises(ValueError):
            FrequencyAttention(32, 4, mode_select="bad")


if __name__ == "__main__":
    unittest.main()
