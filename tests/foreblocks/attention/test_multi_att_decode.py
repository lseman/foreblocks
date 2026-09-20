import unittest

import torch

from foreblocks.attention.config import AttentionConfig
from foreblocks.attention.multi_att import MultiAttention


class TestMultiAttentionIncrementalDecode(unittest.TestCase):
    """Incremental (KV-cached) decode must match the full-sequence forward.

    Covers both cache paths (paged + dense fallback) and MLA/RoPE combos.
    The dense fallback (use_paged_cache=False, used by Oryx) previously
    ignored q_start_pos and produced a wrong causal mask over cached history.
    """

    def _decode_matches_full(self, *, paged, mla, rope, T=16, prefill=0):
        torch.manual_seed(0)
        m = MultiAttention(AttentionConfig.from_legacy_kwargs(
            d_model=64,
            n_heads=8,
            dropout=0.0,
            use_mla=mla,
            pos_encoding_type="rope" if rope else "sinusoidal",
            use_paged_cache=paged,
        )).eval()
        x = torch.randn(2, T, 64)
        with torch.no_grad():
            full, _, _ = m(x, x, x, is_causal=True)
            ls: dict = {}
            outs = []
            start = 0
            if prefill > 0:
                o, _, ls = m(
                    x[:, :prefill], x[:, :prefill], x[:, :prefill],
                    is_causal=True, layer_state=ls,
                )
                outs.append(o)
                start = prefill
            for t in range(start, T):
                xt = x[:, t : t + 1]
                o, _, ls = m(xt, xt, xt, is_causal=True, layer_state=ls)
                outs.append(o)
            inc = torch.cat(outs, dim=1)
        return (inc - full).abs().max().item()

    def test_paged_all_combos(self):
        for mla in (False, True):
            for rope in (False, True):
                err = self._decode_matches_full(paged=True, mla=mla, rope=rope)
                self.assertLess(err, 1e-5, f"paged mla={mla} rope={rope}: {err}")

    def test_dense_fallback_all_combos(self):
        for mla in (False, True):
            for rope in (False, True):
                err = self._decode_matches_full(paged=False, mla=mla, rope=rope)
                self.assertLess(err, 1e-5, f"dense mla={mla} rope={rope}: {err}")

    def test_dense_chunked_prefill_then_steps(self):
        err = self._decode_matches_full(
            paged=False, mla=True, rope=True, T=16, prefill=5
        )
        self.assertLess(err, 1e-5)

    def test_oryx_attention_decode(self):
        from foreblocks.forecasting.popular.oryx import OryxMixerBlock

        torch.manual_seed(0)
        b = OryxMixerBlock(
            d_model=32, n_heads=4, dropout=0.0, attention_type="standard",
            linear_mode="gdn", use_short_conv=False, gate=True, norm_type="rms",
        ).eval()
        x = torch.randn(2, 12, 32)
        with torch.no_grad():
            full, _ = b(x, mode="attention")
            ls: dict = {}
            outs = []
            for t in range(12):
                o, ls = b(x[:, t : t + 1], mode="attention", layer_state=ls)
                outs.append(o)
            inc = torch.cat(outs, dim=1)
        self.assertLess((inc - full).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()


class TestCausalMaskNonZeroQStartPos(unittest.TestCase):
    """Verify causal mask is correctly applied when q_start_pos > 0.

    These tests ensure that during incremental decode:
    - The query only attends to positions ≤ its own position (causal)
    - Padded sequences are handled correctly with q_start_pos
    - Chunked prefill + decode preserves correct attention
    """

    def test_causal_mask_decode_matches_full(self):
        """Each incremental decode step must produce identical output to full
        sequence forward, verifying the causal mask is correct."""
        for paged in (False, True):
            for mla in (False, True):
                torch.manual_seed(0)
                m = MultiAttention(AttentionConfig.from_legacy_kwargs(
                    d_model=64, n_heads=8, dropout=0.0,
                    use_mla=mla,
                    pos_encoding_type="rope",
                    use_paged_cache=paged,
                )).eval()
                x = torch.randn(3, 20, 64)
                with torch.no_grad():
                    full, _, _ = m(x, x, x, is_causal=True)
                    ls: dict = {}
                    outs = []
                    for t in range(20):
                        xt = x[:, t : t + 1]
                        o, _, ls = m(xt, xt, xt, is_causal=True, layer_state=ls)
                        outs.append(o)
                    inc = torch.cat(outs, dim=1)
                err = (inc - full).abs().max().item()
                self.assertLess(
                    err, 1e-5,
                    f"causal mask fail paged={paged} mla={mla}: {err}"
                )

    def test_causal_mask_with_padded_sequences(self):
        """With key_padding_mask, queries should not attend to padded positions
        even when q_start_pos > 0."""
        torch.manual_seed(42)
        m = MultiAttention(AttentionConfig.from_legacy_kwargs(
            d_model=64, n_heads=8, dropout=0.0,
            use_mla=False,
            pos_encoding_type="sinusoidal",
            use_paged_cache=False,
        )).eval()

        # Batch with different lengths: seq0=5 tokens, seq1=3 tokens (padded to 5)
        seq_len_0, seq_len_1 = 5, 3
        max_len = max(seq_len_0, seq_len_1)

        x = torch.randn(2, max_len, 64)
        key_padding_mask = torch.zeros(2, max_len, dtype=torch.bool)
        key_padding_mask[0, seq_len_0:] = True
        key_padding_mask[1, seq_len_1:] = True

        with torch.no_grad():
            full, _, _ = m(x, x, x, is_causal=True, key_padding_mask=key_padding_mask)

            ls: dict = {}
            outs = []
            for t in range(max_len):
                xt = x[:, t : t + 1]
                kpm = key_padding_mask[:, : t + 1] if t > 0 else key_padding_mask[:, :1]
                o, _, ls = m(
                    xt, xt, xt,
                    is_causal=True,
                    layer_state=ls,
                    key_padding_mask=kpm,
                )
                outs.append(o)
            inc = torch.cat(outs, dim=1)

        err = (inc - full).abs().max().item()
        self.assertLess(err, 1e-5, f"padded seq causal mask: {err}")

    def test_chunked_prefill_decode_consistency(self):
        """Chunked prefill + single-token decode must match full forward.

        This specifically tests q_start_pos > 0 from the very first decode step,
        which was the scenario that exposed the MLA get_current_length bug.
        """
        for chunk_size in (2, 4, 7):
            for mla in (False, True):
                for rope in (False, True):
                    torch.manual_seed(0)
                    m = MultiAttention(AttentionConfig.from_legacy_kwargs(
                        d_model=64, n_heads=8, dropout=0.0,
                        use_mla=mla,
                        pos_encoding_type="rope" if rope else "sinusoidal",
                        use_paged_cache=False,
                    )).eval()
                    x = torch.randn(2, 16, 64)
                    with torch.no_grad():
                        full, _, _ = m(x, x, x, is_causal=True)
                        ls: dict = {}
                        outs = []

                        # Prefill in chunks
                        for start in range(0, 16, chunk_size):
                            end = min(start + chunk_size, 16)
                            chunk = x[:, start:end]
                            o, _, ls = m(
                                chunk, chunk, chunk,
                                is_causal=True, layer_state=ls,
                            )
                            outs.append(o)

                        inc = torch.cat(outs, dim=1)
                    err = (inc - full).abs().max().item()
                    self.assertLess(
                        err, 1e-5,
                        f"chunked prefill {chunk_size} mla={mla} rope={rope}: {err}"
                    )

    def test_q_start_pos_values_correct(self):
        """Verify q_start_pos advances correctly during decode.

        Each decode step's q_start_pos should equal the number of tokens
        already in the cache, ensuring the causal mask and RoPE use correct
        absolute positions.
        """
        torch.manual_seed(0)
        m = MultiAttention(AttentionConfig.from_legacy_kwargs(
            d_model=64, n_heads=8, dropout=0.0,
            use_mla=True,
            pos_encoding_type="sinusoidal",
            use_paged_cache=False,
        )).eval()

        captured_q_start_pos = []
        orig_compute = m._compute_attention
        def traced_compute(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights, q_start_pos):
            # Only capture decode steps (when q.shape[2] == 1)
            if q.shape[2] == 1:
                captured_q_start_pos.append(q_start_pos.clone() if q_start_pos is not None else None)
            return orig_compute(q, k, v, attn_mask, key_padding_mask, is_causal, need_weights, q_start_pos)
        m._compute_attention = traced_compute

        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            ls = {}
            for t in range(8):
                xt = x[:, t:t+1]
                m(xt, xt, xt, is_causal=True, layer_state=ls)

        # Check q_start_pos values
        expected = [torch.tensor([0, 0]), torch.tensor([1, 1]), torch.tensor([2, 2]),
                     torch.tensor([3, 3]), torch.tensor([4, 4]), torch.tensor([5, 5]),
                     torch.tensor([6, 6]), torch.tensor([7, 7])]
        for i, (actual, exp) in enumerate(zip(captured_q_start_pos, expected)):
            self.assertTrue(
                torch.equal(actual, exp),
                f"Step {i}: q_start_pos={actual}, expected={exp}"
            )
