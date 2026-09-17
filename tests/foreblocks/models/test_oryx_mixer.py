import unittest

import torch

from foreblocks.models.popular.oryx import OryxMixerBlock, OryxTransformer


class TestOryxMixer(unittest.TestCase):
    def test_oryx_block_forward_attention_and_gdn(self):
        torch.manual_seed(0)
        block = OryxMixerBlock(
            d_model=16,
            n_heads=4,
            dropout=0.0,
            attention_type="standard",
            linear_mode="gdn",
            use_short_conv=True,
            conv_kernel=3,
            gate=True,
            norm_type="rms",
        )

        x = torch.randn(2, 10, 16)
        out_attn, state_attn = block(x, mode="attention")
        out_gdn, state_gdn = block(x, mode="gdn")

        self.assertEqual(out_attn.shape, (2, 10, 16))
        self.assertEqual(out_gdn.shape, (2, 10, 16))
        self.assertIs(block.attn.k_proj, block.gdn.k_proj)
        self.assertIs(block.attn.v_proj, block.gdn.v_proj)
        self.assertNotEqual(
            out_attn.detach().cpu().numpy().tolist(),
            out_gdn.detach().cpu().numpy().tolist(),
        )
        self.assertTrue(state_attn is None or isinstance(state_attn, dict))
        self.assertTrue(state_gdn is None or isinstance(state_gdn, dict))

    def test_oryx_transformer_stack(self):
        torch.manual_seed(0)
        model = OryxTransformer(
            num_layers=2,
            d_model=16,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
            attention_type="standard",
            linear_mode="gdn",
            use_short_conv=False,
            gate=True,
            norm_type="rms",
        )

        x = torch.randn(2, 12, 16)
        out, states = model(x, mode="attention")

        self.assertEqual(out.shape, (2, 12, 16))
        self.assertEqual(len(states), 2)
        self.assertTrue(
            all(state is None or isinstance(state, dict) for state in states)
        )


if __name__ == "__main__":
    unittest.main()
