import unittest

import torch

from foreblocks.transformer.attention.modules.linear_att.kimi import _KDA_Fast


class TestKDAChunkParallel(unittest.TestCase):
    """Chunk-parallel WY path must match the exact sequential KDA recurrence."""

    def _run_both(self, T, chunk_size, **kw):
        torch.manual_seed(0)
        m = _KDA_Fast(
            d_model=32,
            num_heads=4,
            chunk_size=chunk_size,
            shortconv_mode="off",
            dropout=0.0,
            **kw,
        ).double()
        m.eval()
        x = torch.randn(2, T, 32, dtype=torch.float64)
        with torch.no_grad():
            y_chunk, s_chunk = m(x)
            m.chunk_size = 0  # force exact sequential
            y_seq, s_seq = m(x)
        return y_chunk, s_chunk, y_seq, s_seq

    def test_chunk_matches_sequential(self):
        y_c, s_c, y_s, s_s = self._run_both(T=48, chunk_size=16)
        self.assertLess((y_c - y_s).abs().max().item(), 1e-9)
        self.assertLess((s_c - s_s).abs().max().item(), 1e-9)

    def test_ragged_last_chunk(self):
        y_c, s_c, y_s, s_s = self._run_both(T=50, chunk_size=16)
        self.assertLess((y_c - y_s).abs().max().item(), 1e-9)
        self.assertLess((s_c - s_s).abs().max().item(), 1e-9)

    def test_no_safe_updates(self):
        # without alpha clamping, decay can be tiny -> stresses ratio stability
        y_c, s_c, y_s, s_s = self._run_both(T=48, chunk_size=16, safe_updates=False)
        self.assertLess((y_c - y_s).abs().max().item(), 1e-9)
        self.assertLess((s_c - s_s).abs().max().item(), 1e-9)

    def test_carried_state(self):
        torch.manual_seed(1)
        m = _KDA_Fast(
            d_model=32, num_heads=4, chunk_size=16,
            shortconv_mode="off", dropout=0.0,
        ).double()
        m.eval()
        x = torch.randn(2, 40, 32, dtype=torch.float64)
        state = torch.randn(2, m.h, m.dk, m.dv, dtype=torch.float64) * 0.1
        with torch.no_grad():
            y_c, s_c = m(x, state=state.clone())
            m.chunk_size = 0
            y_s, s_s = m(x, state=state.clone())
        self.assertLess((y_c - y_s).abs().max().item(), 1e-9)
        self.assertLess((s_c - s_s).abs().max().item(), 1e-9)

    def test_gradients_finite_fp32(self):
        torch.manual_seed(0)
        m = _KDA_Fast(d_model=64, num_heads=8, chunk_size=16, shortconv_mode="off")
        x = torch.randn(2, 50, 64, requires_grad=True)
        y, _ = m(x)
        y.float().pow(2).mean().backward()
        g = sum(
            p.grad.float().pow(2).sum()
            for p in m.parameters()
            if p.grad is not None
        ).sqrt()
        self.assertTrue(torch.isfinite(g).item())
        self.assertTrue(torch.isfinite(x.grad).all().item())


if __name__ == "__main__":
    unittest.main()
