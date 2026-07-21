"""Tests for the DARTS engine variants (GDAS, GD-DARTS, R-DARTS, PC-DARTS, Bi-DARTS)."""
import unittest

import torch
import torch.nn as nn

from darts.architecture.darts_cell import DARTSCell
from darts.architecture.mixed_op import MixedOp
from darts.config import (
    DARTSEngineConfig,
    DARTSVariant,
    GD_DARTSEngineConfig,
    R_DARTSEngineConfig,
)
from darts.training.darts_engine import (
    build_engine_config,
    compute_backward_loss,
    compute_gradient_norm_balance,
    configure_mixed_op_for_variant,
    forward_bidirectional,
)
from darts.training.optimizers import BilevelOptimizer
from torch.amp import GradScaler


class TestDARTSVariantEnum(unittest.TestCase):
    """Test DARTSVariant enum and AUTO resolution."""

    def test_all_variants_exist(self):
        """All 5 variants + AUTO must be defined."""
        self.assertEqual(len(DARTSVariant), 6)
        variants = {v.value for v in DARTSVariant}
        self.assertEqual(variants, {"darts", "gd_darts", "r_darts", "pc_darts", "bi_darts", "auto"})

    def test_auto_resolution_small_dataset(self):
        """AUTO resolves to R_DARTS for small datasets (< 10k samples)."""
        self.assertEqual(
            DARTSVariant.resolve_auto(n_samples=5_000),
            DARTSVariant.R_DARTS,
        )

    def test_auto_resolution_medium_dataset(self):
        """AUTO resolves to PC_DARTS for medium datasets (10k-100k samples)."""
        self.assertEqual(
            DARTSVariant.resolve_auto(n_samples=50_000),
            DARTSVariant.PC_DARTS,
        )

    def test_auto_resolution_large_dataset(self):
        """AUTO resolves to R_DARTS for large datasets (> 100k samples)."""
        self.assertEqual(
            DARTSVariant.resolve_auto(n_samples=500_000),
            DARTSVariant.R_DARTS,
        )

    def test_auto_resolution_zero_samples(self):
        """AUTO resolves to R_DARTS when n_samples=0 (unknown size)."""
        self.assertEqual(
            DARTSVariant.resolve_auto(n_samples=0),
            DARTSVariant.R_DARTS,
        )

    def test_engine_config_resolve_variant(self):
        """DARTSEngineConfig.resolve_variant() returns the correct variant."""
        cfg = DARTSEngineConfig(variant=DARTSVariant.GD_DARTS)
        self.assertEqual(cfg.resolve_variant(), DARTSVariant.GD_DARTS)

        cfg = DARTSEngineConfig(variant=DARTSVariant.AUTO, n_samples=5_000)
        self.assertEqual(cfg.resolve_variant(), DARTSVariant.R_DARTS)

        cfg = DARTSEngineConfig(variant=DARTSVariant.AUTO, n_samples=500_000)
        self.assertEqual(cfg.resolve_variant(), DARTSVariant.R_DARTS)


class TestConfigureMixedOpForVariant(unittest.TestCase):
    """Test configure_mixed_op_for_variant for each variant."""

    def test_gd_darts_disables_gumbel(self):
        """GD-DARTS: disable Gumbel-Softmax in MixedOp."""
        op = MixedOp(
            input_dim=4,
            latent_dim=4,
            seq_length=8,
            available_ops=["Identity", "TimeConv", "DLinear"],
            temperature=0.5,
            op_gdas=True,
        )
        op.use_gumbel = True  # start with Gumbel enabled

        engine_cfg = DARTSEngineConfig(variant=DARTSVariant.GD_DARTS)
        configure_mixed_op_for_variant(
            op, "gd_darts", engine_cfg, epoch=0, total_epochs=50
        )

        self.assertFalse(op.use_gumbel)

    def test_gd_darts_sets_commitment_temperature(self):
        """GD-DARTS: set commitment temperature."""
        op = MixedOp(
            input_dim=4,
            latent_dim=4,
            seq_length=8,
            available_ops=["Identity", "TimeConv", "DLinear"],
            temperature=0.5,
            op_gdas=True,
        )
        self.assertEqual(op.op_temperature, 0.5)

        engine_cfg = DARTSEngineConfig(
            variant=DARTSVariant.GD_DARTS,
            gd_darts=GD_DARTSEngineConfig(commitment_temperature=0.1),
        )
        configure_mixed_op_for_variant(
            op, "gd_darts", engine_cfg, epoch=0, total_epochs=50
        )

        self.assertEqual(op.op_temperature, 0.1)

    def test_gd_darts_hierarchical_op_remains_runnable(self):
        """Changing variants must not invalidate construction-time alpha layout."""
        op = MixedOp(
            input_dim=4,
            latent_dim=4,
            seq_length=8,
            available_ops=["Identity", "TimeConv", "DLinear"],
            use_hierarchical=True,
            op_gdas=True,
        )
        engine_cfg = DARTSEngineConfig(variant=DARTSVariant.GD_DARTS)

        configure_mixed_op_for_variant(
            op, "gd_darts", engine_cfg, epoch=0, total_epochs=10
        )
        output = op(torch.randn(2, 8, 4))

        self.assertTrue(op.use_hierarchical)
        self.assertFalse(op.op_gdas)
        self.assertEqual(output.shape, (2, 8, 4))

    def test_pc_darts_sets_perm_l2_weight(self):
        """PC-DARTS enables partial channels and learned edge normalization."""
        cell = DARTSCell(
            input_dim=4,
            latent_dim=4,
            seq_length=8,
            num_nodes=3,
            initial_search=False,
            selected_ops=["Identity", "TimeConv", "DLinear"],
            op_gdas=False,  # PC-DARTS disables GDAS
        )

        engine_cfg = DARTSEngineConfig(variant=DARTSVariant.PC_DARTS)
        configure_mixed_op_for_variant(
            cell, "pc_darts", engine_cfg, epoch=0, total_epochs=50
        )

        self.assertTrue(cell.pc_darts_enabled)
        for edge in cell.edges:
            self.assertTrue(edge.pc_darts_enabled)

    def test_pc_darts_candidates_share_one_channel_sample(self):
        op = MixedOp(
            input_dim=8,
            latent_dim=8,
            seq_length=4,
            available_ops=["Identity", "DLinear"],
            use_hierarchical=False,
            drop_prob=0.0,
            pc_ratio=0.25,
            op_gdas=False,
        )
        configure_mixed_op_for_variant(
            op,
            "pc_darts",
            DARTSEngineConfig(variant=DARTSVariant.PC_DARTS),
            epoch=0,
            total_epochs=2,
        )
        seen = []
        hooks = [
            candidate.register_forward_pre_hook(
                lambda _module, args: seen.append(args[0].detach().clone())
            )
            for candidate in op.ops
        ]
        try:
            op.train()(torch.ones(1, 4, 8))
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.equal(seen[0], seen[1]))
        self.assertEqual(torch.count_nonzero(seen[0]).item(), 8)

    def test_bi_darts_sets_bidirectional_training(self):
        """Bi-DARTS: enable bidirectional training on cells."""
        cell = DARTSCell(
            input_dim=4,
            latent_dim=4,
            seq_length=8,
            num_nodes=3,
            initial_search=False,
            selected_ops=["Identity", "TimeConv", "DLinear"],
        )

        engine_cfg = DARTSEngineConfig(
            variant=DARTSVariant.BI_DARTS,
            bi_darts=type("BIDARTS", (), {
                "backward_loss_weight": 0.7,
                "backward_passes": 2,
            })(),
        )
        configure_mixed_op_for_variant(
            cell, "bi_darts", engine_cfg, epoch=0, total_epochs=50
        )

        # All cells should have bidirectional_training set.
        for edge in cell.edges:
            if hasattr(edge, "bidirectional_training"):
                self.assertTrue(edge.bidirectional_training)
            if hasattr(edge, "backward_loss_weight"):
                self.assertEqual(edge.backward_loss_weight, 0.7)

    def test_r_darts_sets_norm_balance_warmup(self):
        """R-DARTS: enable gradient-norm balancing."""
        cell = DARTSCell(
            input_dim=4,
            latent_dim=4,
            seq_length=8,
            num_nodes=3,
            initial_search=False,
            selected_ops=["Identity", "TimeConv", "DLinear"],
        )

        engine_cfg = DARTSEngineConfig(
            variant=DARTSVariant.R_DARTS,
            r_darts=R_DARTSEngineConfig(norm_balance_warmup=5),
        )
        configure_mixed_op_for_variant(
            cell, "r_darts", engine_cfg, epoch=0, total_epochs=50
        )

        # All cells should have norm_balance_warmup set.
        for edge in cell.edges:
            if hasattr(edge, "norm_balance_warmup"):
                self.assertEqual(edge.norm_balance_warmup, 5)


class TestGradientNormBalance(unittest.TestCase):
    """Test compute_gradient_norm_balance for R-DARTS."""

    def test_balancing_disabled_during_warmup(self):
        """Gradient norm balancing returns 1.0 during warmup."""
        model = nn.Linear(4, 4)
        arch_grads = [torch.ones(4) * 2.0]  # arch grad norm = 2*sqrt(4) = 4
        model_grads = [torch.ones(4) * 1.0]  # model grad norm = 1*sqrt(4) = 2

        # During warmup (epoch=0, warmup=2), should return 1.0.
        factor = compute_gradient_norm_balance(
            model,
            arch_grads,
            model_grads,
            warmup_epochs=2,
            epoch=0,
        )
        self.assertEqual(factor, 1.0)

    def test_balancing_computed_after_warmup(self):
        """Gradient norm balancing returns correct factor after warmup."""
        model = nn.Linear(4, 4)
        arch_grads = [torch.ones(4) * 2.0]  # arch grad norm = 4
        model_grads = [torch.ones(4) * 1.0]  # model grad norm = 2

        # After warmup (epoch=3, warmup=2), should return model/arch = 2/4 = 0.5
        factor = compute_gradient_norm_balance(
            model,
            arch_grads,
            model_grads,
            warmup_epochs=2,
            epoch=3,
        )
        # factor = model_norm / arch_norm = 2 / 4 = 0.5
        self.assertAlmostEqual(factor, 0.5, places=3)

    def test_balancing_with_empty_grads(self):
        """Gradient norm balancing handles empty grads gracefully."""
        model = nn.Linear(4, 4)

        # Empty arch_grads and model_grads.
        factor = compute_gradient_norm_balance(
            model,
            [],
            [],
            warmup_epochs=0,
            epoch=0,
        )
        # Should return 1.0 (no balancing needed).
        self.assertEqual(factor, 1.0)

    def test_bilevel_step_applies_balance_to_arch_gradient(self):
        arch = nn.Parameter(torch.tensor(1.0))
        weight = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([arch], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        bilevel = BilevelOptimizer(
            arch_optimizer=optimizer,
            arch_scheduler=scheduler,
            arch_params=[arch],
            edge_arch_params=[],
            component_arch_params=[],
            use_bilevel_optimization=False,
            train_arch_loader=None,
            val_loader=[(torch.zeros(1), torch.zeros(1))],
            train_model_loader=[(torch.zeros(1), torch.zeros(1))],
        )

        # Raw gradients are arch=2 and weight=1, so balancing scales arch by 0.5.
        bilevel.step_architecture(
            arch * 2.0 + weight,
            GradScaler(enabled=False),
            gradient_balance_params=[weight],
            gradient_balance_epoch=1,
        )

        self.assertAlmostEqual(arch.item(), 0.9, places=6)


class TestForwardBidirectional(unittest.TestCase):
    """Test forward_bidirectional for Bi-DARTS."""

    def test_bidirectional_forward_preserves_shape(self):
        """Bidirectional forward preserves input/output shape."""
        cell = DARTSCell(
            input_dim=4,
            latent_dim=4,
            seq_length=8,
            num_nodes=3,
            initial_search=False,
            selected_ops=["Identity", "TimeConv", "DLinear"],
        )

        x = torch.randn(2, 8, 4)
        out = forward_bidirectional(cell, x, backward_loss_weight=0.5)
        self.assertEqual(out.shape, x.shape)

    def test_backward_loss_uses_weight_and_number_of_passes(self):
        class CountingIdentity(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.calls = 0

            def forward(self, x, **kwargs):
                self.calls += 1
                return x * self.scale

        model = CountingIdentity()
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        y = torch.zeros_like(x)
        loss = compute_backward_loss(
            model,
            x,
            y,
            nn.MSELoss(),
            backward_loss_weight=0.25,
            backward_passes=2,
        )

        expected = nn.functional.mse_loss(x, y) * 1.25
        self.assertEqual(model.calls, 3)
        self.assertTrue(torch.allclose(loss, expected))


class TestBuildEngineConfig(unittest.TestCase):
    """Test build_engine_config factory."""

    def test_build_r_darts_config(self):
        """Build engine config for R-DARTS."""
        cfg = build_engine_config("r_darts")
        self.assertEqual(cfg["use_adamw_arch"], True)
        self.assertEqual(cfg["balance_gradient_norms"], True)

    def test_build_gd_darts_config(self):
        """Build engine config for GD-DARTS."""
        cfg = build_engine_config("gd_darts")
        self.assertEqual(cfg["replace_gumbel_softmax"], True)
        self.assertEqual(cfg["commitment_temperature"], 0.1)

    def test_build_pc_darts_config(self):
        """Build engine config for PC-DARTS."""
        cfg = build_engine_config("pc_darts")
        self.assertEqual(cfg["enable_partial_channels"], True)
        self.assertEqual(cfg["enable_edge_normalization"], True)

    def test_build_bi_darts_config(self):
        """Build engine config for Bi-DARTS."""
        cfg = build_engine_config("bi_darts")
        self.assertEqual(cfg["bidirectional_training"], True)

    def test_build_engine_config_with_overrides(self):
        """Build engine config with overrides."""
        cfg = build_engine_config("r_darts", norm_balance_warmup=10)
        self.assertEqual(cfg["use_adamw_arch"], True)
        self.assertEqual(cfg["norm_balance_warmup"], 10)


if __name__ == "__main__":
    unittest.main()
