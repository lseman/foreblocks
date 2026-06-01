import unittest

import torch

from foreblocks.transformer.attention.multi_att import MultiAttention
from foreblocks.transformer.attention.variants import moba as moba_module


class TestMoBAAttention(unittest.TestCase):
    def test_single_block_matches_standard_attention(self):
        torch.manual_seed(0)
        base = MultiAttention(
            d_model=32,
            n_heads=4,
            dropout=0.0,
            attention_type="standard",
            use_mla=False,
            chunk_size=64,
        ).eval()
        moba = MultiAttention(
            d_model=32,
            n_heads=4,
            dropout=0.0,
            attention_type="moba",
            use_mla=False,
            chunk_size=64,
            moba_topk=4,
        ).eval()
        moba.load_state_dict(base.state_dict(), strict=False)

        x = torch.randn(2, 16, 32)
        with torch.no_grad():
            out_std, _, _ = base(x, x, x, is_causal=True)
            out_moba, _, _ = moba(x, x, x, is_causal=True)

        self.assertLess((out_std - out_moba).abs().max().item(), 1e-5)

    def test_causal_output_ignores_future_tokens(self):
        torch.manual_seed(0)
        moba = MultiAttention(
            d_model=32,
            n_heads=4,
            dropout=0.0,
            attention_type="moba",
            use_mla=False,
            chunk_size=4,
            moba_topk=2,
        ).eval()

        x1 = torch.randn(2, 12, 32)
        x2 = x1.clone()
        x2[:, 8:] = torch.randn_like(x2[:, 8:])

        with torch.no_grad():
            out1, _, _ = moba(x1, x1, x1, is_causal=True)
            out2, _, _ = moba(x2, x2, x2, is_causal=True)

        self.assertLess((out1[:, :8] - out2[:, :8]).abs().max().item(), 1e-5)

    def test_flash_backend_path_uses_varlen_ops(self):
        torch.manual_seed(0)
        call_count = {"value": 0}

        def fake_flash_forward(**kwargs):
            call_count["value"] += 1
            q = kwargs["q"]
            lse = torch.zeros(
                q.size(1), q.size(0), device=q.device, dtype=torch.float32
            )
            return q + 0.25, lse, None, None

        original_resolve = moba_module._resolve_flash_varlen_ops
        original_can_use = moba_module.MoBAAttentionImpl._can_use_flash_moba
        original_ops = moba_module._FLASH_MOBA_OPS
        try:
            moba_module._FLASH_MOBA_OPS = None
            moba_module._resolve_flash_varlen_ops = lambda: (
                fake_flash_forward,
                object(),
            )
            moba_module.MoBAAttentionImpl._can_use_flash_moba = (
                lambda self, q, k, v, attn_mask, key_padding_mask, is_causal, need_weights: (
                    True
                )
            )

            moba = MultiAttention(
                d_model=32,
                n_heads=4,
                dropout=0.0,
                attention_type="moba",
                use_mla=False,
                chunk_size=4,
                moba_topk=2,
            ).eval()
            x = torch.randn(2, 12, 32)
            with torch.no_grad():
                out, _, _ = moba(x, x, x, is_causal=True)

            self.assertEqual(tuple(out.shape), (2, 12, 32))
            self.assertGreaterEqual(call_count["value"], 4)
        finally:
            moba_module._resolve_flash_varlen_ops = original_resolve
            moba_module.MoBAAttentionImpl._can_use_flash_moba = original_can_use
            moba_module._FLASH_MOBA_OPS = original_ops


if __name__ == "__main__":
    unittest.main()
