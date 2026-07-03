"""Tests for LLRD + WarmupCosineLR scheduler + per-layer dropout schedule."""

import pytest
import torch
import torch.nn as nn

from foreblocks.config import TrainingConfig
from foreblocks.core.training.llrd import (
    WarmupCosineLR,
    get_llrd_param_groups,
)
from foreblocks.models.transformer.tf_base import TransformerEncoder, TransformerDecoder
from foreblocks.modules.skip.mod import LayerDropoutSchedule


class TestLayerDropoutSchedule:
    """Test per-layer dropout schedule."""

    def test_flat_profile(self):
        """Test flat profile (no depth scaling)."""
        schedule = LayerDropoutSchedule(
            num_layers=6,
            base_dropout=0.1,
            max_dropout=0.3,
            profile="flat",
        )
        for i in range(6):
            assert schedule.get_dropout(i) == 0.1

    def test_deeper_more_profile(self):
        """Test deeper_more profile."""
        schedule = LayerDropoutSchedule(
            num_layers=6,
            base_dropout=0.05,
            max_dropout=0.25,
            profile="deeper_more",
        )
        # Layer 0 (shallow) should have lower dropout
        d0 = schedule.get_dropout(0)
        # Layer 5 (deep) should have higher dropout
        d5 = schedule.get_dropout(5)
        assert d0 < d5
        assert d0 >= 0.05
        assert d5 <= 0.25

    def test_deeper_less_profile(self):
        """Test deeper_less profile."""
        schedule = LayerDropoutSchedule(
            num_layers=6,
            base_dropout=0.25,
            max_dropout=0.05,
            profile="deeper_less",
        )
        # Layer 0 should have higher dropout
        d0 = schedule.get_dropout(0)
        # Layer 5 should have lower dropout
        d5 = schedule.get_dropout(5)
        assert d0 > d5
        assert d0 <= 0.25
        assert d5 >= 0.05

    def test_none_max_dropout(self):
        """Test None max_dropout (returns base_dropout)."""
        schedule = LayerDropoutSchedule(
            num_layers=6,
            base_dropout=0.1,
            max_dropout=None,
            profile="deeper_more",
        )
        for i in range(6):
            assert schedule.get_dropout(i) == 0.1


class TestGetLLRDParamGroups:
    """Test LLRD param group builder."""

    def test_simple_model(self):
        """Test LLRD on a simple transformer."""
        model = TransformerEncoder(
            input_size=1,
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
        )
        param_groups = get_llrd_param_groups(
            model,
            base_lr=1e-3,
            weight_decay=0.01,
            decay=0.9,
        )
        assert len(param_groups) > 0
        # Check that param groups have decreasing LR for deeper layers
        layer_groups = [g for g in param_groups if "layer" in g.get("group_name", "")]
        if layer_groups:
            # Sort by group name to ensure layer order
            layer_groups_sorted = sorted(layer_groups, key=lambda g: int(g["group_name"].split("_")[1]))
            lrs = [g["lr"] for g in layer_groups_sorted]
            # Deeper layers should have lower LR (decay is applied)
            if len(lrs) > 1:
                assert lrs[0] > lrs[-1]

    def test_param_deduplication(self):
        """Test that no params are dropped or duplicated."""
        model = TransformerEncoder(
            input_size=1,
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
        )
        param_groups = get_llrd_param_groups(
            model,
            base_lr=1e-3,
            weight_decay=0.01,
        )
        total_params = sum(len(g["params"]) for g in param_groups)
        model_params = len(list(model.parameters()))
        assert total_params == model_params

    def test_no_decay_params(self):
        """Test that bias and norm params get weight_decay=0."""
        model = TransformerEncoder(
            input_size=1,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
        )
        param_groups = get_llrd_param_groups(
            model,
            base_lr=1e-3,
            weight_decay=0.01,
        )
        # Check for groups with weight_decay=0 (should include bias, norm)
        no_decay_groups = [g for g in param_groups if g["weight_decay"] == 0]
        assert len(no_decay_groups) > 0


class TestWarmupCosineLR:
    """Test WarmupCosineLR scheduler."""

    def test_warmup_phase(self):
        """Test linear warmup phase."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.01,
        )
        # Check LR ramps from 0 to base during warmup
        lrs = []
        for _ in range(150):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        # LR should increase monotonically during warmup (0-100 steps)
        assert all(lrs[i] <= lrs[i + 1] for i in range(99))
        # LR should start near 0
        assert lrs[0] < lrs[50]
        assert lrs[50] < lrs[99]

    def test_cosine_phase(self):
        """Test cosine annealing phase."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.01,
        )
        # Step through warmup
        for _ in range(100):
            scheduler.step()
        base_lr = optimizer.param_groups[0]["lr"]
        # Step through cosine annealing (should decrease)
        lrs = []
        for _ in range(900):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        # LR should decrease monotonically in cosine phase
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))
        # Final LR should be near min_lr_ratio * base_lr
        min_lr = 0.01 * 1e-3
        assert lrs[-1] >= min_lr * 0.95

    def test_state_dict(self):
        """Test save/load state dict."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.01,
        )
        # Step a few times
        for _ in range(50):
            scheduler.step()
        state = scheduler.state_dict()
        assert state["_step"] == 50
        # Load into new scheduler
        scheduler2 = WarmupCosineLR(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.01,
        )
        scheduler2.load_state_dict(state)
        assert scheduler2._step == 50


class TestTransformerWithLayerDropoutSchedule:
    """Test transformer with layer dropout schedule."""

    def test_encoder_with_dropout_schedule(self):
        """Test encoder instantiation with dropout schedule."""
        schedule = LayerDropoutSchedule(
            num_layers=4,
            base_dropout=0.05,
            max_dropout=0.2,
            profile="deeper_more",
        )
        encoder = TransformerEncoder(
            input_size=1,
            d_model=64,
            nhead=4,
            num_layers=4,
            dim_feedforward=256,
            layer_dropout_schedule=schedule,
        )
        # Check that encoder was created
        assert encoder is not None
        assert encoder.num_layers == 4
        assert encoder.layer_dropout_schedule == schedule

    def test_decoder_with_dropout_schedule(self):
        """Test decoder instantiation with dropout schedule."""
        schedule = LayerDropoutSchedule(
            num_layers=4,
            base_dropout=0.05,
            max_dropout=0.2,
            profile="deeper_more",
        )
        decoder = TransformerDecoder(
            input_size=1,
            output_size=1,
            d_model=64,
            nhead=4,
            num_layers=4,
            dim_feedforward=256,
            layer_dropout_schedule=schedule,
        )
        assert decoder is not None
        assert decoder.num_layers == 4
        assert decoder.layer_dropout_schedule == schedule


class TestTrainingConfigExtensions:
    """Test new TrainingConfig fields."""

    def test_llrd_fields(self):
        """Test LLRD fields in TrainingConfig."""
        config = TrainingConfig(
            use_llrd=True,
            llrd_decay=0.95,
            learning_rate=1e-3,
            weight_decay=0.01,
        )
        assert config.use_llrd is True
        assert config.llrd_decay == 0.95
        assert config.learning_rate == 1e-3

    def test_warmup_fields(self):
        """Test warmup fields in TrainingConfig."""
        config = TrainingConfig(
            scheduler_type="warmup_cosine",
            warmup_steps=100,
            warmup_ratio=0.1,
            steps_per_epoch=1000,
        )
        assert config.scheduler_type == "warmup_cosine"
        assert config.warmup_steps == 100
        assert config.warmup_ratio == 0.1
        assert config.steps_per_epoch == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
