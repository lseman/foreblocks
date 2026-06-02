"""Regression tests for the ModernLinearAttention chunk-parallel backends.

Each backend's chunk-parallel mode must match its exact sequential recurrent
mode (same weights). This pins the property that previously regressed:
  - DeltaNet chunk path used the wrong intra-chunk term + left-padding (fixed).
  - GLA was already correct; this guards against future drift.

Run in float64 for a tight (1e-9) tolerance. CPU-only is fine.
"""
import unittest

import torch

from foreblocks.transformer.attention.modules.modern_linear_attn import (
    GLABackend,
    ModernLinearAttention,
    RDABackend,
)
from foreblocks.transformer.embeddings.rotary import apply_rotary_emb


def _assert_backend_rope_matches_reference(backend, Lq=7, Lk=5):
    torch.manual_seed(123)
    backend = backend.double().eval()
    q = torch.randn(2, backend.n_heads, Lq, backend.d_head, dtype=torch.float64)
    k = torch.randn(2, backend.n_heads, Lk, backend.d_head, dtype=torch.float64)

    with torch.no_grad():
        q_rot, k_rot = backend._apply_rope(q, k)
        rotary = backend._rotary_emb
        q_ref = apply_rotary_emb(
            q.transpose(1, 2),
            rotary._cos_cached[:Lq],
            rotary._sin_cached[:Lq],
        ).transpose(1, 2)
        k_ref = apply_rotary_emb(
            k.transpose(1, 2),
            rotary._cos_cached[:Lk],
            rotary._sin_cached[:Lk],
        ).transpose(1, 2)

    torch.testing.assert_close(q_rot, q_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(k_rot, k_ref, atol=0.0, rtol=0.0)


def _chunk_vs_recurrent(backend, T, chunk_size, d_model=32, n_heads=4):
    """Build a backend, run chunk vs recurrent on identical weights, return outputs."""
    torch.manual_seed(0)
    chunk = ModernLinearAttention(
        d_model, n_heads, dropout=0.0, backend=backend,
        mode="chunk", chunk_size=chunk_size,
    ).double().eval()
    recur = ModernLinearAttention(
        d_model, n_heads, dropout=0.0, backend=backend,
        mode="recurrent", chunk_size=chunk_size,
    ).double().eval()
    recur.load_state_dict(chunk.state_dict())

    x = torch.randn(2, T, d_model, dtype=torch.float64)
    with torch.no_grad():
        y_chunk, _, _ = chunk(x, x, x, is_causal=True)
        y_recur, _, _ = recur(x, x, x, is_causal=True)
    return y_chunk, y_recur


class TestDeltaNetChunkMatchesRecurrent(unittest.TestCase):
    # DeltaNet's WY solve (torch.linalg.solve_triangular) runs in float32
    # internally, so the achievable chunk-vs-recurrent agreement is ~1e-7, not
    # full float64. 1e-6 still catches the ~O(1) bugs this test guards against.
    TOL = 1e-6

    def test_divisible_length(self):
        yc, yr = _chunk_vs_recurrent("deltanet", T=48, chunk_size=16)
        self.assertLess((yc - yr).abs().max().item(), self.TOL)

    def test_ragged_last_chunk(self):
        # T not a multiple of chunk_size -> padding path (previously broken)
        yc, yr = _chunk_vs_recurrent("deltanet", T=50, chunk_size=16)
        self.assertLess((yc - yr).abs().max().item(), self.TOL)

    def test_single_chunk(self):
        # chunk_size == 1: WY matrix is trivial, isolates the readout convention
        yc, yr = _chunk_vs_recurrent("deltanet", T=8, chunk_size=1)
        self.assertLess((yc - yr).abs().max().item(), self.TOL)

    def test_multi_chunk_padded(self):
        yc, yr = _chunk_vs_recurrent("deltanet", T=200, chunk_size=64)
        self.assertLess((yc - yr).abs().max().item(), self.TOL)

    def test_no_future_leak(self):
        torch.manual_seed(0)
        m = ModernLinearAttention(
            32, 4, dropout=0.0, backend="deltanet", mode="chunk", chunk_size=16
        ).double().eval()
        x = torch.randn(2, 50, 32, dtype=torch.float64)
        x2 = x.clone()
        x2[:, -1] += 10.0  # perturb only the last token
        with torch.no_grad():
            o1, _, _ = m(x, x, x, is_causal=True)
            o2, _, _ = m(x2, x2, x2, is_causal=True)
        self.assertLess((o1[:, :-1] - o2[:, :-1]).abs().max().item(), 1e-12)


class TestGLAChunkMatchesRecurrent(unittest.TestCase):
    def test_divisible_length(self):
        yc, yr = _chunk_vs_recurrent("gla", T=48, chunk_size=16)
        self.assertLess((yc - yr).abs().max().item(), 1e-9)

    def test_ragged_last_chunk(self):
        yc, yr = _chunk_vs_recurrent("gla", T=50, chunk_size=16)
        self.assertLess((yc - yr).abs().max().item(), 1e-9)

    def test_multi_chunk(self):
        yc, yr = _chunk_vs_recurrent("gla", T=128, chunk_size=32)
        self.assertLess((yc - yr).abs().max().item(), 1e-9)


class TestModernLinearAttentionRoPE(unittest.TestCase):
    def test_rda_rope_matches_rotary_reference(self):
        _assert_backend_rope_matches_reference(
            RDABackend(32, 4, dropout=0.0, pos_encoding_type="rope")
        )

    def test_gla_rope_matches_rotary_reference(self):
        _assert_backend_rope_matches_reference(
            GLABackend(32, 4, dropout=0.0, pos_encoding_type="rope")
        )

    def test_router_threads_pos_encoding_type_to_backend(self):
        for backend_name in ("rda", "gla", "deltanet"):
            attn = ModernLinearAttention(
                32, 4, dropout=0.0, backend=backend_name, pos_encoding_type="rope"
            )
            self.assertEqual(attn.impl.pos_encoding_type, "rope")


if __name__ == "__main__":
    unittest.main()
