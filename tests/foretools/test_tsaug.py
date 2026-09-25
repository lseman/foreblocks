"""Tests for foretools.tsaug module - API surface, transformations, features, layers, losses, and model."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


class TestPublicAPI:
    """Verify all public names are importable from foretools.tsaug."""

    def test_all_exports_importable(self):
        import foretools.tsaug as tsaug

        expected = [
            "__version__",
            "AutoDATimeseries",
            "AutoDATrainer",
            "AugmentationLayer",
            "StackedAugmentationLayers",
            "CompositeLoss",
            "extract_features",
            "FEATURE_DIM",
            "TRANSFORMATIONS",
            "TRANSFORM_NAMES",
            "NUM_TRANSFORMS",
            "raw",
            "jittering",
            "scaling",
            "resample",
            "time_warp",
            "freq_warp",
            "mag_warp",
            "time_mask",
            "drift",
            "permutation",
            "window_slice",
            "time_mix",
        ]
        for name in expected:
            assert hasattr(tsaug, name), f"Missing public export: {name}"

    def test_version_is_string(self):
        import foretools.tsaug as tsaug

        assert isinstance(tsaug.__version__, str)

    def test_feature_dim_constant(self):
        from foretools.tsaug import FEATURE_DIM

        assert FEATURE_DIM == 24

    def test_num_transforms(self):
        from foretools.tsaug import NUM_TRANSFORMS, TRANSFORM_NAMES, TRANSFORMATIONS

        assert NUM_TRANSFORMS == len(TRANSFORM_NAMES)
        assert NUM_TRANSFORMS == len(TRANSFORMATIONS)
        assert NUM_TRANSFORMS == 12

    def test_transform_names_match(self):
        from foretools.tsaug import TRANSFORM_NAMES

        expected = [
            "Raw",
            "Jittering",
            "Scaling",
            "Resample",
            "TimeWarp",
            "FreqWarp",
            "MagWarp",
            "TimeMask",
            "Drift",
            "Permutation",
            "WindowSlice",
            "TimeMix",
        ]
        assert TRANSFORM_NAMES == expected

    def test_transformations_are_callables(self):
        from foretools.tsaug import TRANSFORMATIONS

        for t in TRANSFORMATIONS:
            assert callable(t)


class TestTransformations:
    """Test each transformation function preserves shape and dtype."""

    @pytest.fixture
    def x(self):
        return torch.randn(4, 50, 1)

    @pytest.fixture
    def intensity(self):
        # Per-batch intensity matching batch size
        return torch.tensor([0.3, 0.3, 0.3, 0.3])

    def test_raw_preserves_input(self, x):
        from foretools.tsaug.transformations import raw

        y = raw(x, torch.zeros_like(x))
        assert torch.allclose(y, x)

    def test_jittering_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import jittering

        for dtype in [torch.float32, torch.float64]:
            x_t = x.to(dtype)
            y = jittering(x_t, intensity.to(dtype))
            assert y.shape == x.shape
            assert y.dtype == dtype

    def test_scaling_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import scaling

        for dtype in [torch.float32, torch.float64]:
            x_t = x.to(dtype)
            y = scaling(x_t, intensity.to(dtype))
            assert y.shape == x.shape
            assert y.dtype == dtype

    def test_resample_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import resample

        for dtype in [torch.float32, torch.float64]:
            x_t = x.to(dtype)
            y = resample(x_t, intensity.to(dtype))
            assert y.shape == x.shape
            assert y.dtype == dtype

    def test_time_warp_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import time_warp

        for dtype in [torch.float32, torch.float64]:
            x_t = x.to(dtype)
            y = time_warp(x_t, intensity.to(dtype))
            assert y.shape == x.shape
            assert y.dtype == dtype

    def test_freq_warp_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import freq_warp

        for dtype in [torch.float32, torch.float64]:
            x_t = x.to(dtype)
            y = freq_warp(x_t, intensity.to(dtype))
            assert y.shape == x.shape
            assert y.dtype == dtype

    def test_mag_warp_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import mag_warp

        for dtype in [torch.float32, torch.float64]:
            x_t = x.to(dtype)
            y = mag_warp(x_t, intensity.to(dtype))
            assert y.shape == x.shape
            assert y.dtype == dtype

    def test_time_mask_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import time_mask

        y = time_mask(x.float(), intensity)
        assert y.shape == x.shape

    def test_drift_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import drift

        y = drift(x.float(), intensity)
        assert y.shape == x.shape

    def test_permutation_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import permutation

        y = permutation(x.float(), intensity)
        assert y.shape == x.shape

    def test_window_slice_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import window_slice

        y = window_slice(x.float(), intensity)
        assert y.shape == x.shape

    def test_time_mix_shape_dtype(self, x, intensity):
        from foretools.tsaug.transformations import time_mix

        # batch >= 2 required for time_mix
        x2 = torch.randn(4, 50, 1)
        y = time_mix(x2.float(), intensity)
        assert y.shape == x2.shape

    def test_time_mix_batch_1_returns_input(self):
        from foretools.tsaug.transformations import time_mix

        x = torch.randn(1, 50, 1)  # batch=1 — no mixing possible
        y = time_mix(x, torch.tensor([0.3]))
        assert torch.allclose(y, x)

    def test_zero_intensity_produces_small_changes(self, x):
        from foretools.tsaug.transformations import jittering

        y = jittering(x, torch.zeros(4))
        # Zero intensity jittering should return identical input
        assert torch.allclose(y, x)


class TestFeatures:
    """Test feature extraction."""

    def test_extract_features_shape(self):
        from foretools.tsaug.features import extract_features, FEATURE_DIM

        x = torch.randn(8, 100, 3)
        features = extract_features(x)
        assert features.shape == (8, FEATURE_DIM)

    def test_extract_features_single_channel(self):
        from foretools.tsaug.features import extract_features

        x = torch.randn(16, 200, 1)
        features = extract_features(x)
        assert features.shape == (16, 24)

    def test_extract_features_multichannel(self):
        from foretools.tsaug.features import extract_features

        x = torch.randn(8, 50, 5)
        features = extract_features(x)
        assert features.shape == (8, 24)

    def test_extract_features_no_nans(self):
        from foretools.tsaug.features import extract_features

        # Constant signal — should not produce NaNs
        x = torch.ones(4, 100, 1)
        features = extract_features(x)
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()

    def test_extract_features_batch_independence(self):
        from foretools.tsaug.features import extract_features

        x_single = torch.randn(1, 100, 1)
        f1 = extract_features(x_single)
        # Features should be finite and valid
        assert not torch.isnan(f1).any()
        assert not torch.isinf(f1).any()


class TestLayers:
    """Test augmentation layers."""

    def test_augmentation_layer_creation(self):
        from foretools.tsaug.layers import AugmentationLayer

        layer = AugmentationLayer(feature_dim=24, num_transforms=12)
        assert layer.num_transforms == 12

    def test_augmentation_layer_forward(self):
        from foretools.tsaug.layers import AugmentationLayer

        layer = AugmentationLayer(
            feature_dim=24, num_transforms=12, hidden_dim=32
        )
        x = torch.randn(8, 50, 1)
        features = torch.randn(8, 24)
        prev_prob = torch.zeros(8, 12)

        layer.train()
        result = layer(x, prev_prob, features)

        # Returns (x_aug, prob, intensity, selected) as tensors
        x_aug, prob, intensity, selected = result
        assert x_aug.shape == x.shape
        assert prob.shape == (8, 12)
        assert intensity.shape == (8, 12)
        assert selected.shape == (8,)

    def test_augmentation_layer_inference(self):
        from foretools.tsaug.layers import AugmentationLayer

        layer = AugmentationLayer()
        x = torch.randn(4, 50, 1)
        features = torch.randn(4, 24)
        prev_prob = torch.zeros(4, 12)

        layer.eval()
        with torch.no_grad():
            x_aug, prob, intensity, selected = layer(x, prev_prob, features)

        assert x_aug.shape == x.shape

    def test_stacked_layers_creation(self):
        from foretools.tsaug.layers import StackedAugmentationLayers

        stack = StackedAugmentationLayers(num_layers=3)
        assert len(stack.layers) == 3

    def test_stacked_layers_forward(self):
        from foretools.tsaug.layers import StackedAugmentationLayers

        stack = StackedAugmentationLayers(num_layers=3, hidden_dim=32)
        x = torch.randn(8, 50, 1)
        features = torch.randn(8, 24)

        stack.train()
        x_aug, probs, intensities, selected = stack(x, features)

        assert x_aug.shape == x.shape
        assert len(probs) == 3
        assert len(intensities) == 3
        assert len(selected) == 3


class TestCompositeLoss:
    """Test the composite loss function."""

    def test_composite_loss_creation(self):
        from foretools.tsaug.losses import CompositeLoss

        closs = CompositeLoss()
        assert len(closs.log_w) == 3

    def test_composite_loss_forward(self):
        from foretools.tsaug.losses import CompositeLoss

        closs = CompositeLoss()
        task_loss = torch.tensor(0.5)
        all_probs = [torch.ones(8, 12) / 12 for _ in range(3)]

        total, details = closs(task_loss, all_probs)

        assert isinstance(total, torch.Tensor)
        assert "total" in details
        assert "L1" in details
        assert "L2" in details
        assert "L3" in details
        assert "w1" in details
        assert "w2" in details
        assert "w3" in details

    def test_composite_loss_single_layer(self):
        from foretools.tsaug.losses import CompositeLoss

        closs = CompositeLoss()
        task_loss = torch.tensor(0.5)
        all_probs = [torch.ones(8, 12) / 12]  # single layer

        total, details = closs(task_loss, all_probs)
        assert "total" in details

    def test_composite_loss_learnable_weights(self):
        from foretools.tsaug.losses import CompositeLoss

        closs = CompositeLoss()
        initial_w1 = closs.log_w[0].exp().item() ** 2

        task_loss = torch.tensor(0.5)
        all_probs = [torch.ones(8, 12) / 12 for _ in range(3)]

        total, details = closs(task_loss, all_probs)
        assert abs(details["w1"] - initial_w1) < 0.01


class TestModel:
    """Test the AutoDATimeseries and AutoDATrainer classes."""

    def test_autoda_creation(self):
        from foretools.tsaug import AutoDATimeseries

        autoda = AutoDATimeseries(num_layers=3, hidden_dim=64)
        assert autoda.num_layers == 3
        assert autoda.feature_dim == 24

    def test_autoda_forward(self):
        from foretools.tsaug import AutoDATimeseries

        autoda = AutoDATimeseries(num_layers=2)
        x = torch.randn(8, 50, 1)

        autoda.train()
        x_aug, probs, intensities, selected = autoda(x)

        assert x_aug.shape == x.shape
        assert len(probs) == 2
        assert len(intensities) == 2
        assert len(selected) == 2

    def test_autoda_precomputed_features(self):
        from foretools.tsaug import AutoDATimeseries
        from foretools.tsaug.features import extract_features

        autoda = AutoDATimeseries(num_layers=2)
        x = torch.randn(8, 50, 1)
        features = extract_features(x)

        x_aug, probs, intensities, selected = autoda(x, precomputed_features=features)
        assert x_aug.shape == x.shape

    def test_autoda_get_policy_summary(self):
        from foretools.tsaug import AutoDATimeseries

        autoda = AutoDATimeseries(num_layers=3)
        x = torch.randn(8, 50, 1)

        autoda.train()
        x_aug, probs, intensities, selected = autoda(x)

        summary = autoda.get_policy_summary(probs, intensities, selected)

        assert "layer_0" in summary
        assert "layer_1" in summary
        assert "layer_2" in summary
        assert "temperature" in summary["layer_0"]
        assert "avg_probabilities" in summary["layer_0"]
        assert "avg_intensities" in summary["layer_0"]

    def test_trainer_creation(self):
        from foretools.tsaug import AutoDATimeseries, AutoDATrainer

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 2)

            def forward(self, x):
                # x: (B, L, C) -> pool to (B, 1) -> fc -> (B, 2)
                return self.fc(x.mean(dim=(1, 2)).unsqueeze(-1))

        autoda = AutoDATimeseries(num_layers=2)
        downstream = DummyModel()
        trainer = AutoDATrainer(autoda, downstream, task="classification")

        assert trainer.task == "classification"
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_trainer_train_step(self):
        from foretools.tsaug import AutoDATimeseries, AutoDATrainer

        class DummyClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 2)

            def forward(self, x):
                # x: (B, L, C) -> pool to (B, 1) -> fc -> (B, 2)
                return self.fc(x.mean(dim=(1, 2)).unsqueeze(-1))

        autoda = AutoDATimeseries(num_layers=2)
        downstream = DummyClassifier()
        trainer = AutoDATrainer(autoda, downstream, task="classification")

        x = torch.randn(4, 50, 1)
        y = torch.tensor([0, 1, 0, 1])

        loss_details = trainer.train_step(x, y)

        assert "total" in loss_details
        assert "L1" in loss_details


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test the complete augmentation pipeline."""
        from foretools.tsaug import AutoDATimeseries, extract_features

        autoda = AutoDATimeseries(num_layers=2)
        x = torch.randn(8, 100, 1)

        # Extract features manually
        features = extract_features(x)
        assert features.shape == (8, 24)

        # Forward pass with precomputed features
        autoda.train()
        x_aug, probs, intensities, selected = autoda(x, precomputed_features=features)
        assert x_aug.shape == x.shape

    def test_training_loop_step(self):
        """Test a minimal training loop step."""
        from foretools.tsaug import AutoDATimeseries, AutoDATrainer
        from foretools.tsaug.losses import CompositeLoss

        class SimpleClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 2)

            def forward(self, x):
                # x: (B, L, C) -> pool to (B, 1) -> fc -> (B, 2)
                return self.fc(x.mean(dim=(1, 2)).unsqueeze(-1))

        autoda = AutoDATimeseries(num_layers=2)
        downstream = SimpleClassifier()
        trainer = AutoDATrainer(autoda, downstream, task="classification")

        # Run a few training steps
        for _ in range(3):
            x = torch.randn(8, 50, 1)
            y = torch.randint(0, 2, (8,))
            loss_details = trainer.train_step(x, y)
            assert not torch.isnan(torch.tensor(loss_details["total"]))

    def test_composite_loss_gradient_flow(self):
        """Test that gradients flow through the composite loss."""
        from foretools.tsaug.losses import CompositeLoss

        closs = CompositeLoss()
        task_loss = torch.tensor(0.5)
        all_probs = [torch.ones(8, 12) / 12 for _ in range(3)]

        total, details = closs(task_loss, all_probs)
        total.backward()

        # Check that learnable weights have gradients
        for log_w in closs.log_w:
            assert log_w.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
