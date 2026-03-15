"""
AutoDA-Timeseries: Automated Data Augmentation for Time Series.

Main framework module that ties together:
  - Time series feature extraction
  - Stacked augmentation layers (augmented data generator A_theta)
  - Composite loss with learnable weights
  - Joint end-to-end training with downstream models

Usage:
    autoda = AutoDATimeseries(num_layers=3)
    trainer = AutoDATrainer(autoda, downstream_model, task='forecasting')
    trainer.fit(train_loader, val_loader, epochs=50)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List

from .features import extract_features, FEATURE_DIM
from .layers import StackedAugmentationLayers
from .losses import CompositeLoss
from .transformations import NUM_TRANSFORMS, TRANSFORM_NAMES


class AutoDATimeseries(nn.Module):
    """AutoDA-Timeseries framework: feature-aware augmented data generator.

    Implements the complete augmentation pipeline from Figure 2:
      Raw Time Series -> Feature Extractor -> Adaptive Policy Generator
      -> Stacked Augmentation Layers -> Augmented Time Series

    Args:
        num_layers: K, number of stacked augmentation layers (default: 3).
        num_transforms: Number of available transformations (default: 7).
        feature_dim: Dimension of extracted feature vector (default: 24).
        hidden_dim: Hidden dimension for policy MLPs.
        init_temperature: Initial Gumbel-Softmax temperature.
        raw_bias: Probability of raw transform selection per layer (p_rb).
    """

    def __init__(
        self,
        num_layers: int = 3,
        num_transforms: int = NUM_TRANSFORMS,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        init_temperature: float = 1.0,
        raw_bias: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_transforms = num_transforms
        self.feature_dim = feature_dim

        # Feature projection MLP (optional, to align feature dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Stacked augmentation layers (Section 3.4)
        self.aug_layers = StackedAugmentationLayers(
            num_layers=num_layers,
            feature_dim=feature_dim,
            num_transforms=num_transforms,
            hidden_dim=hidden_dim,
            init_temperature=init_temperature,
            raw_bias=raw_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        precomputed_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Generate augmented time series with adaptive policy.

        Args:
            x: (B, L, C) raw time series.
            precomputed_features: Optional (B, feature_dim) pre-extracted features.
                If None, features are computed on-the-fly.

        Returns:
            x_aug: (B, L, C) augmented time series.
            all_probs: List of K probability vectors.
            all_intensities: List of K intensity vectors.
            all_selected: List of K selected transform indices.
        """
        # Feature extraction (Section 3.3)
        if precomputed_features is not None:
            features = precomputed_features
        else:
            features = extract_features(x)  # (B, 24)

        # Project features
        features = self.feature_proj(features)  # (B, feature_dim)

        # Apply stacked augmentation layers
        x_aug, all_probs, all_intensities, all_selected = self.aug_layers(
            x, features
        )

        return x_aug, all_probs, all_intensities, all_selected

    def get_policy_summary(
        self,
        all_probs: List[torch.Tensor],
        all_intensities: List[torch.Tensor],
        all_selected: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """Get a human-readable summary of the augmentation policy.

        Args:
            all_probs: List of K probability vectors.
            all_intensities: List of K intensity vectors.
            all_selected: List of K selected indices.

        Returns:
            Dictionary with policy summary per layer.
        """
        summary = {}
        for k in range(self.num_layers):
            avg_prob = all_probs[k].mean(dim=0).detach().cpu().numpy()
            avg_intensity = all_intensities[k].mean(dim=0).detach().cpu().numpy()
            temp = self.aug_layers.layers[k].temperature.item()

            layer_info = {
                "temperature": temp,
                "avg_probabilities": {
                    name: float(p) for name, p in zip(TRANSFORM_NAMES, avg_prob)
                },
                "avg_intensities": {
                    name: float(t) for name, t in zip(TRANSFORM_NAMES, avg_intensity)
                },
            }
            summary[f"layer_{k}"] = layer_info

        return summary


class AutoDATrainer:
    """End-to-end trainer for AutoDA-Timeseries with downstream model.

    Jointly optimizes the augmentation framework parameters theta and
    downstream model parameters theta_M (Section 3.1, Eqs. 2-3).

    Args:
        autoda: AutoDATimeseries instance (augmented data generator).
        downstream_model: Any nn.Module for the downstream task.
        task: Task type ('classification', 'forecasting', 'regression', 'anomaly').
        task_loss_fn: Loss function for the downstream task.
            Default: CrossEntropyLoss for classification, MSELoss otherwise.
        lr: Learning rate.
        aug_lr: Learning rate for augmentation parameters (if different).
        weight_decay: Weight decay for optimizer.
        device: Device to train on.
    """

    def __init__(
        self,
        autoda: AutoDATimeseries,
        downstream_model: nn.Module,
        task: str = "forecasting",
        task_loss_fn: Optional[nn.Module] = None,
        lr: float = 1e-3,
        aug_lr: Optional[float] = None,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.autoda = autoda.to(device)
        self.downstream_model = downstream_model.to(device)
        self.task = task
        self.device = device

        # Task loss
        if task_loss_fn is not None:
            self.task_loss_fn = task_loss_fn
        elif task == "classification":
            self.task_loss_fn = nn.CrossEntropyLoss()
        else:
            self.task_loss_fn = nn.MSELoss()

        # Composite loss (Section 3.5.2)
        self.composite_loss = CompositeLoss().to(device)

        # Optimizer with separate param groups
        aug_lr = aug_lr or lr
        self.optimizer = torch.optim.Adam([
            {"params": self.downstream_model.parameters(), "lr": lr, "weight_decay": weight_decay},
            {"params": self.autoda.parameters(), "lr": aug_lr},
            {"params": self.composite_loss.parameters(), "lr": aug_lr},
        ])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        precomputed_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            x: (B, L, C) input time series.
            y: Target labels/values.
            precomputed_features: Optional pre-extracted features.

        Returns:
            Dictionary with loss values.
        """
        self.autoda.train()
        self.downstream_model.train()

        x = x.to(self.device)
        y = y.to(self.device)
        if precomputed_features is not None:
            precomputed_features = precomputed_features.to(self.device)

        # Generate augmented data
        x_aug, all_probs, all_intensities, all_selected = self.autoda(
            x, precomputed_features
        )

        # Forward through downstream model
        output = self.downstream_model(x_aug)

        # Compute task loss
        task_loss = self.task_loss_fn(output, y)

        # Compute composite loss
        total_loss, loss_details = self.composite_loss(task_loss, all_probs)

        # Backprop and update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.autoda.parameters()) +
            list(self.downstream_model.parameters()) +
            list(self.composite_loss.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return loss_details

    @torch.no_grad()
    def eval_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluation step on original (non-augmented) data.

        At test time, only the downstream model is used (Section 3.1, Eq. 3).
        """
        self.downstream_model.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        output = self.downstream_model(x)
        loss = self.task_loss_fn(output, y)

        result = {"eval_loss": loss.item()}

        if self.task == "classification":
            preds = output.argmax(dim=-1)
            acc = (preds == y).float().mean().item()
            result["accuracy"] = acc

        return result

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 50,
        log_interval: int = 10,
        precompute_features: bool = True,
    ) -> Dict[str, list]:
        """Full training loop.

        Args:
            train_loader: DataLoader yielding (x, y) batches.
            val_loader: Optional validation DataLoader.
            epochs: Number of training epochs.
            log_interval: Print progress every N epochs.
            precompute_features: Whether to precompute and cache features.

        Returns:
            Dictionary with training history.
        """
        history = {"train_loss": [], "val_loss": []}
        if self.task == "classification":
            history["val_accuracy"] = []

        # Optionally precompute features for training data
        feature_cache = {}
        if precompute_features:
            print("Precomputing time series features...")
            self.autoda.eval()
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                features = extract_features(x)
                feature_cache[batch_idx] = features.cpu()
            self.autoda.train()
            print(f"  Cached features for {len(feature_cache)} batches.")

        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for batch_idx, (x, y) in enumerate(train_loader):
                feats = feature_cache.get(batch_idx)
                loss_details = self.train_step(x, y, feats)
                epoch_losses.append(loss_details["total"])

            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            history["train_loss"].append(avg_train_loss)

            # Validation
            if val_loader is not None:
                val_results = self._evaluate(val_loader)
                history["val_loss"].append(val_results["eval_loss"])
                if "accuracy" in val_results:
                    history["val_accuracy"].append(val_results["accuracy"])

            # Learning rate scheduling
            self.scheduler.step()

            # Logging
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}"
                if val_loader is not None:
                    msg += f" | Val Loss: {val_results['eval_loss']:.4f}"
                    if "accuracy" in val_results:
                        msg += f" | Val Acc: {val_results['accuracy']:.4f}"

                # Policy summary
                policy = self.autoda.get_policy_summary(
                    *self._get_sample_policy(train_loader)
                )
                temps = [policy[f"layer_{k}"]["temperature"] for k in range(self.autoda.num_layers)]
                msg += f" | Temps: {[f'{t:.3f}' for t in temps]}"
                print(msg)

        return history

    def _evaluate(self, loader) -> Dict[str, float]:
        """Evaluate on a DataLoader."""
        all_results = []
        for x, y in loader:
            result = self.eval_step(x, y)
            all_results.append(result)

        avg = {}
        for key in all_results[0]:
            avg[key] = sum(r[key] for r in all_results) / len(all_results)
        return avg

    def _get_sample_policy(self, loader):
        """Get a sample policy for logging."""
        self.autoda.eval()
        x, _ = next(iter(loader))
        x = x.to(self.device)
        with torch.no_grad():
            _, probs, intensities, selected = self.autoda(x)
        self.autoda.train()
        return probs, intensities, selected
