import unittest

import torch

from foreblocks.modules.attention.multi_att import MultiAttention


class TestMultiAttentionIncrementalDecode(unittest.TestCase):
    """Incremental (KV-cached) decode must match the full-sequence forward.

    Covers both cache paths (paged + dense fallback) and MLA/RoPE combos.
    The dense fallback (use_paged_cache=False, used by Oryx) previously
    ignored q_start_pos and produced a wrong causal mask over cached history.
    """

    def _decode_matches_full(self, *, paged, mla, rope, T=16, prefill=0):
        torch.manual_seed(0)
        m = MultiAttention(
            d_model=64,
            n_heads=8,
            dropout=0.0,
            use_mla=mla,
            pos_encoding_type="rope" if rope else "sinusoidal",
            use_paged_cache=paged,
        ).eval()
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
        from foreblocks.models.popular.oryx import OryxMixerBlock

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
