import unittest

import torch

from foreblocks.transformer.attention.modules.gated_delta import GatedDeltaNet


class TestGatedDeltaChunkParallel(unittest.TestCase):
    """Chunk-parallel WY path must match the exact sequential recurrence."""

    def _run_both(self, T, chunk_size, **kw):
        torch.manual_seed(0)
        m = GatedDeltaNet(
            d_model=32,
            n_heads=4,
            dropout=0.0,
            chunk_size=chunk_size,
            use_short_conv=False,
            **kw,
        ).double()
        m.eval()
        x = torch.randn(2, T, 32, dtype=torch.float64)

        with torch.no_grad():
            # chunk-parallel (T > chunk_size triggers _chunk_parallel)
            y_chunk, s_chunk = m._forward_recurrent(x, state=None)
            # force pure-sequential by disabling chunking
            saved = m.chunk_size
            m.chunk_size = 0
            y_seq, s_seq = m._forward_recurrent(x, state=None)
            m.chunk_size = saved
        return y_chunk, s_chunk, y_seq, s_seq

    def test_chunk_matches_sequential_mamba_gate(self):
        y_c, s_c, y_s, s_s = self._run_both(T=48, chunk_size=16, use_mamba_gate=True)
        self.assertLess((y_c - y_s).abs().max().item(), 1e-9)
        self.assertLess((s_c - s_s).abs().max().item(), 1e-9)

    def test_chunk_matches_sequential_legacy_gate(self):
        y_c, s_c, y_s, s_s = self._run_both(T=48, chunk_size=16, use_mamba_gate=False)
        self.assertLess((y_c - y_s).abs().max().item(), 1e-9)
        self.assertLess((s_c - s_s).abs().max().item(), 1e-9)

    def test_ragged_last_chunk(self):
        # T not a multiple of chunk_size -> last chunk shorter
        y_c, s_c, y_s, s_s = self._run_both(T=50, chunk_size=16, use_mamba_gate=True)
        self.assertLess((y_c - y_s).abs().max().item(), 1e-9)
        self.assertLess((s_c - s_s).abs().max().item(), 1e-9)

    def test_carried_state(self):
        # feed an incoming non-zero state and check both paths agree
        torch.manual_seed(1)
        m = GatedDeltaNet(
            d_model=32, n_heads=4, dropout=0.0, chunk_size=16,
            use_short_conv=False, use_mamba_gate=True,
        ).double()
        m.eval()
        x = torch.randn(2, 40, 32, dtype=torch.float64)
        state = torch.randn(2, m.h, m.dk, m.dv, dtype=torch.float64) * 0.1
        with torch.no_grad():
            y_c, s_c = m._forward_recurrent(x, state=state.clone())
            m.chunk_size = 0
            y_s, s_s = m._forward_recurrent(x, state=state.clone())
        self.assertLess((y_c - y_s).abs().max().item(), 1e-9)
        self.assertLess((s_c - s_s).abs().max().item(), 1e-9)


if __name__ == "__main__":
    unittest.main()
