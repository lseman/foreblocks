import unittest

import torch

from foreblocks.modules.skip.gateskip import (
    ResidualGate,
    gateskip_apply,
    GateStats,
)
from foreblocks.modules.skip.mod import (
    MoDRouter,
    mod_topk_mask,
    mod_routed_indices,
    mod_router_aux_loss,
    MoDBudgetScheduler,
)
from foreblocks.modules.skip.gateskip import (
    BudgetScheduler,
    _exact_topk_keep_mask,
    _compute_gate_stats,
)
from foreblocks.modules.attention.utils.residuals import (
    AttentionResidual,
    BlockAttentionResidual,
    normalize_attention_residual_mode,
)


class TestResidualGate(unittest.TestCase):
    def test_token_conditioned_gate_outputs_scalar_tokens(self):
        """ResidualGate produces scalar gates [B,T,1] and importance scores [B,T]."""
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
        # Gate bias defaults to 5.0: σ(5) ≈ 0.993
        self.assertAlmostEqual(gate.fc.bias[0].item(), 5.0)
        self.assertAlmostEqual(torch.sigmoid(gate.fc.bias[0]).item(), 0.993, places=2)

    def test_gateskip_apply_masks_inactive_tokens(self):
        """gateskip_apply returns 3-tuple and masks inactive tokens via skip_mask."""
        torch.manual_seed(0)
        gate = ResidualGate(d_model=8, gate_dim=1)
        x = torch.randn(1, 3, 8)
        o = torch.randn(1, 3, 8)
        active_mask = torch.tensor([[True, False, True]])

        out, skip_mask, stats = gateskip_apply(
            enabled=True,
            h_prev=x,
            o=o,
            gate=gate,
            budget=1.0,
            aux_terms=[],
            lambda_s=0.1,
            active_mask=active_mask,
        )

        self.assertEqual(skip_mask.shape, (1, 3))
        self.assertTrue(skip_mask[0, 1].item())  # inactive token is skipped
        self.assertTrue(torch.allclose(out[:, 1, :], x[:, 1, :]))  # residual copy
        self.assertIsInstance(stats, GateStats)

    def test_gateskip_disabled(self):
        """Disabled gateskip returns plain residual with no skip mask."""
        torch.manual_seed(0)
        gate = ResidualGate(d_model=8, gate_dim=1)
        x = torch.randn(1, 3, 8)
        o = torch.randn(1, 3, 8)

        out, skip_mask, stats = gateskip_apply(
            enabled=False,
            h_prev=x,
            o=o,
            gate=gate,
            budget=None,
            aux_terms=[],
            lambda_s=0.0,
        )

        self.assertTrue(torch.allclose(out, x + o))
        self.assertIsNone(skip_mask)

    def test_gateskip_with_aux_terms(self):
        """Auxiliary terms are appended to aux_terms list."""
        torch.manual_seed(0)
        gate = ResidualGate(d_model=8, gate_dim=1)
        x = torch.randn(1, 3, 8)
        o = torch.randn(1, 3, 8)
        aux = []

        gateskip_apply(
            enabled=True,
            h_prev=x,
            o=o,
            gate=gate,
            budget=0.5,
            aux_terms=aux,
            lambda_s=0.1,
            lambda_budget=0.05,
            lambda_smooth=0.02,
        )

        # Should have 3 terms: sparsity + budget + smoothness
        self.assertEqual(len(aux), 3)

    def test_gateskip_per_feature_gating(self):
        """ResidualGate with gate_dim=d_model produces per-feature gates."""
        torch.manual_seed(0)
        gate = ResidualGate(d_model=8, gate_dim=8)
        x = torch.randn(2, 4, 8)
        o = torch.randn(2, 4, 8)

        out, g, gbar = gate(x, o)

        self.assertEqual(g.shape, (2, 4, 8))
        self.assertEqual(gbar.shape, (2, 4))  # mean over features

    def test_gateskip_vector_gate_default(self):
        """ResidualGate defaults to vector gate (gate_dim=d_model)."""
        torch.manual_seed(0)
        gate = ResidualGate(d_model=8)
        self.assertEqual(gate.gate_dim, 8)
        x = torch.randn(2, 4, 8)
        o = torch.randn(2, 4, 8)
        out, g, gbar = gate(x, o)
        self.assertEqual(g.shape, (2, 4, 8))

    def test_gateskip_init_defaults(self):
        """ResidualGate init: bias=5.0, std=0.01, single linear gate."""
        gate = ResidualGate(d_model=8, gate_dim=1)
        self.assertAlmostEqual(gate.fc.bias[0].item(), 5.0)
        # Should have exactly one Linear, no fc1/fc2 or norm
        self.assertFalse(hasattr(gate, 'fc1'))
        self.assertFalse(hasattr(gate, 'fc2'))
        self.assertFalse(hasattr(gate, 'act'))
        self.assertFalse(hasattr(gate, 'norm'))
        gate2 = ResidualGate(d_model=8, gate_dim=1, init_bias=2.0)
        self.assertAlmostEqual(gate2.fc.bias[0].item(), 2.0)


class TestBudgetScheduler(unittest.TestCase):
    def test_budget_decreases_linearly(self):
        scheduler = BudgetScheduler(b_start=1.0, b_end=0.5, total_steps=100)
        b1 = scheduler.get_budget(current_step=0)
        b50 = scheduler.get_budget(current_step=50)
        b100 = scheduler.get_budget(current_step=100)
        self.assertGreater(b1, b50)
        self.assertGreater(b50, b100)
        self.assertAlmostEqual(b100, 0.5)

    def test_budget_no_total_steps(self):
        scheduler = BudgetScheduler(b_start=1.0, b_end=0.5)
        # Without total_steps, should return b_end
        self.assertAlmostEqual(scheduler.get_budget(), 0.5)


class TestMoDRouter(unittest.TestCase):
    def test_router_token_mode(self):
        router = MoDRouter(d_model=16, mode="token")
        x = torch.randn(2, 8, 16)
        scores = router(x)
        self.assertEqual(scores.shape, (2, 8, 1))

    def test_router_seq_mode(self):
        router = MoDRouter(d_model=16, mode="seq")
        x = torch.randn(2, 8, 16)
        scores = router(x)
        self.assertEqual(scores.shape, (2, 1, 1))

    def test_router_two_layer_head(self):
        router = MoDRouter(d_model=16, mode="token", hidden=32)
        x = torch.randn(2, 8, 16)
        scores = router(x)
        self.assertEqual(scores.shape, (2, 8, 1))


class TestModTopK(unittest.TestCase):
    def test_mod_topk_mask_selects_exact_count(self):
        scores = torch.randn(2, 10, 1)
        keep_rate = 0.5
        mask = mod_topk_mask(scores, keep_rate=keep_rate)
        self.assertEqual(mask.shape, (2, 10))
        # Should keep exactly ceil(0.5 * 10) = 5 per row
        self.assertEqual(mask.sum().item(), 10)

    def test_mod_topk_mask_edges(self):
        scores = torch.randn(1, 5, 1)
        self.assertTrue((mod_topk_mask(scores, 0.0) == False).all())
        self.assertTrue((mod_topk_mask(scores, 1.0) == True).all())

    def test_mod_topk_mask_with_active_mask(self):
        scores = torch.randn(1, 5, 1)
        active = torch.tensor([[True, True, False, False, True]])
        mask = mod_topk_mask(scores, keep_rate=1.0, active_mask=active)
        # Only active positions should be True (mask is subset of active)
        self.assertTrue((mask | ~active).all())  # every True in mask is also in active
        # Inactive positions should never be True
        self.assertTrue((mask[0, 2:4].logical_not()).all())  # positions 2,3 are inactive


class TestModRoutedIndices(unittest.TestCase):
    def test_mod_routed_indices_basic(self):
        keep = torch.tensor([[True, False, True, True, False]])
        indices, slot_mask = mod_routed_indices(keep)
        self.assertEqual(indices.shape, (1, 3))
        self.assertEqual(slot_mask.sum().item(), 3)

    def test_mod_routed_indices_with_capacity(self):
        keep = torch.tensor([[True, False, True, False, False]])
        indices, slot_mask = mod_routed_indices(keep, capacity=5)
        self.assertEqual(indices.shape, (1, 5))


class TestModAuxLoss(unittest.TestCase):
    def test_mod_router_aux_loss_scalar(self):
        scores = torch.randn(2, 4, 1)
        keep = torch.zeros(2, 4, dtype=torch.bool)
        keep[:, :2] = True
        aux = mod_router_aux_loss(scores, keep)
        self.assertEqual(aux.dim(), 0)  # scalar


class TestMoDBudgetScheduler(unittest.TestCase):
    def test_flat_profile(self):
        sched = MoDBudgetScheduler(num_layers=4, start_keep=1.0, end_keep=0.7,
                                   total_steps=10, layer_profile="flat")
        rates = [sched.get_keep_rate(i) for i in range(4)]
        self.assertTrue(all(r == rates[0] for r in rates))

    def test_deeper_more_profile(self):
        sched = MoDBudgetScheduler(num_layers=4, start_keep=1.0, end_keep=0.7,
                                   total_steps=0, layer_profile="deeper_more")
        rates = [sched.get_keep_rate(i) for i in range(4)]
        self.assertTrue(rates[-1] >= rates[0])

    def test_deeper_less_profile(self):
        sched = MoDBudgetScheduler(num_layers=4, start_keep=1.0, end_keep=0.7,
                                   total_steps=0, layer_profile="deeper_less")
        rates = [sched.get_keep_rate(i) for i in range(4)]
        self.assertTrue(rates[0] >= rates[-1])

    def test_step_advances(self):
        sched = MoDBudgetScheduler(num_layers=4, start_keep=1.0, end_keep=0.5,
                                   total_steps=10, layer_profile="flat")
        r0 = sched.get_keep_rate(0)
        sched.step()
        r1 = sched.get_keep_rate(0)
        self.assertLess(r1, r0)  # budget decreases from start_keep to end_keep


class TestExactTopKKeepMask(unittest.TestCase):
    def test_exact_topk_selects_correct_tokens(self):
        gbar = torch.tensor([[0.1, 0.9, 0.5, 0.2, 0.8]])
        mask = _exact_topk_keep_mask(gbar, keep=0.4)
        # Top 40% of 5 = 2 tokens: indices 1 and 4 (scores 0.9, 0.8)
        self.assertEqual(mask.sum().item(), 2)

    def test_exact_topk_with_active_mask(self):
        gbar = torch.tensor([[0.1, 0.9, 0.0, 0.5, 0.8]])
        active = torch.tensor([[True, True, False, True, True]])
        mask = _exact_topk_keep_mask(gbar, keep=0.5, active_mask=active)
        # 4 active tokens, top 50% = 2 tokens among active
        self.assertEqual(mask.sum().item(), 2)


class TestComputeGateStats(unittest.TestCase):
    def test_gate_stats_shapes(self):
        gbar = torch.randn(2, 4)
        keep = torch.zeros(2, 4, dtype=torch.bool)
        stats = _compute_gate_stats(gbar, keep_mask=keep, budget=0.5)
        self.assertIsInstance(stats, GateStats)
        self.assertEqual(stats.gate_mean.dim(), 0)
        self.assertEqual(stats.token_keep_ratio.dim(), 0)

    def test_budget_error(self):
        gbar = torch.ones(1, 4)
        keep = torch.ones(1, 4, dtype=torch.bool)
        stats = _compute_gate_stats(gbar, keep_mask=keep, budget=1.0)
        self.assertAlmostEqual(stats.budget_error.item(), 0.0)


class TestAttentionResidual(unittest.TestCase):
    def test_attention_residual_output_shape(self):
        ar = AttentionResidual(dim=32)
        history = [torch.randn(2, 8, 32) for _ in range(4)]
        out = ar(history)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_attention_residual_single_layer(self):
        ar = AttentionResidual(dim=16)
        history = [torch.randn(1, 4, 16)]
        out = ar(history)
        self.assertEqual(out.shape, (1, 4, 16))

    def test_attention_residual_gradient(self):
        ar = AttentionResidual(dim=16)
        history = [torch.randn(1, 4, 16, requires_grad=True) for _ in range(3)]
        out = ar(history)
        out.sum().backward()
        self.assertIsNotNone(ar.query.grad)


class TestBlockAttentionResidual(unittest.TestCase):
    def test_block_attnres_output_shape(self):
        bar = BlockAttentionResidual(dim=32)
        blocks = [torch.randn(2, 8, 32) for _ in range(3)]
        out = bar(blocks)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_block_attnres_with_partial(self):
        bar = BlockAttentionResidual(dim=16)
        blocks = [torch.randn(1, 4, 16) for _ in range(2)]
        partial = torch.randn(1, 4, 16)
        out = bar(blocks, partial=partial)
        self.assertEqual(out.shape, (1, 4, 16))

    def test_block_attnres_requires_values(self):
        bar = BlockAttentionResidual(dim=16)
        with self.assertRaises(ValueError):
            bar([])


class TestNormalizeMode(unittest.TestCase):
    def test_normalize_modes(self):
        self.assertEqual(normalize_attention_residual_mode("full"), "full")
        self.assertEqual(normalize_attention_residual_mode("block"), "block")

    def test_normalize_invalid(self):
        with self.assertRaises(ValueError):
            normalize_attention_residual_mode("unknown")


if __name__ == "__main__":
    unittest.main()
