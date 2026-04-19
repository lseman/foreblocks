import unittest

import torch

from foreblocks.tf.skip.gateskip import ResidualGate, gateskip_apply


class TestResidualGate(unittest.TestCase):
    def test_token_conditioned_gate_outputs_scalar_tokens(self):
        torch.manual_seed(0)
        gate = ResidualGate(d_model=8, gate_dim=1)
        x = torch.randn(2, 4, 8)
        o = torch.randn(2, 4, 8)

        out, g, gbar = gate(x, o)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(g.shape, (2, 4, 1))
        self.assertEqual(gbar.shape, (2, 4))
        self.assertTrue(torch.all(g >= 0))
        self.assertTrue(torch.all(g <= 1))

    def test_gateskip_apply_masks_inactive_tokens(self):
        torch.manual_seed(0)
        gate = ResidualGate(d_model=8, gate_dim=1)
        x = torch.randn(1, 3, 8)
        o = torch.randn(1, 3, 8)
        active_mask = torch.tensor([[True, False, True]])

        out, skip_mask = gateskip_apply(
            enabled=True,
            h_prev=x,
            o=o,
            gate=gate,
            budget=1.0,
            aux_l2_terms=[],
            lambda_s=0.1,
            active_mask=active_mask,
        )

        self.assertEqual(skip_mask.shape, (1, 3))
        self.assertTrue(skip_mask[0, 1].item())
        self.assertTrue(torch.allclose(out[:, 1, :], x[:, 1, :]))


if __name__ == "__main__":
    unittest.main()
