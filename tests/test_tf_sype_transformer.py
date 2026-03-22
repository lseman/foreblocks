import unittest

import torch

from foreblocks.tf.attention.multi_att import MultiAttention
from foreblocks.tf.embeddings import LearnablePositionalEncoding, PositionalEncoding
from foreblocks.tf.transformer import TransformerEncoder


class TestSyPETransformer(unittest.TestCase):
    def test_sype_chunked_transform_matches_full_sequence_with_state(self):
        torch.manual_seed(0)
        attn = MultiAttention(
            d_model=8,
            n_heads=2,
            dropout=0.0,
            attention_type="sype",
            use_mla=False,
            use_paged_cache=False,
        )

        B, H, T, Dh = 2, 2, 6, 4
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        query = torch.randn(B, T, 8)
        key = query.clone()

        q_full, k_full = attn._sype_position_transform(
            q, k, {"query": query, "key": key}
        )

        state = {}
        q_parts = []
        k_parts = []
        for start in range(0, T, 2):
            stop = min(T, start + 2)
            q_chunk, k_chunk = attn._sype_position_transform(
                q[:, :, start:stop, :],
                k[:, :, start:stop, :],
                {
                    "query": query[:, start:stop, :],
                    "key": key[:, start:stop, :],
                    "layer_state": state,
                },
            )
            q_parts.append(q_chunk)
            k_parts.append(k_chunk)

        q_chunked = torch.cat(q_parts, dim=2)
        k_chunked = torch.cat(k_parts, dim=2)

        self.assertIn("sype_tau", state)
        self.assertEqual(tuple(state["sype_tau"].shape), (B,))
        self.assertTrue(torch.allclose(q_full, q_chunked, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(k_full, k_chunked, atol=1e-6, rtol=1e-5))

    def test_transformer_encoder_uses_learnable_positions_for_sype(self):
        model = TransformerEncoder(
            input_size=3,
            d_model=16,
            nhead=4,
            num_layers=1,
            dropout=0.0,
            att_type="sype",
            patch_encoder=False,
        )
        self.assertIsInstance(model.pos_encoder, LearnablePositionalEncoding)

    def test_transformer_encoder_keeps_sinusoidal_positions_for_standard(self):
        model = TransformerEncoder(
            input_size=3,
            d_model=16,
            nhead=4,
            num_layers=1,
            dropout=0.0,
            att_type="standard",
            patch_encoder=False,
        )
        self.assertIsInstance(model.pos_encoder, PositionalEncoding)


if __name__ == "__main__":
    unittest.main()
