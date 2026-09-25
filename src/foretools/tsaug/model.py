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

from typing import Any

import torch
import torch.nn as nn

from .features import FEATURE_DIM, extract_features
from .layers import StackedAugmentationLayers
from .losses import CompositeLoss
from .transformations import NUM_TRANSFORMS, TRANSFORM_NAMES


class AutoDATimeseries(nn.Module):
    """AutoDA-Timeseries framework: feature-aware augmented data generator.

    Implements the complete augmentation pipeline:

        Raw Time Series → Feature Extractor → Adaptive Policy Generator
            → Stacked Augmentation Layers → Augmented Time Series

    The framework learns to adaptively select which transformations to apply,
    at what intensity, and in what sequence, conditioned on global time series
    features.

    Architecture
    ------------
    1. Feature extraction: 24 descriptive statistics (see :mod:`features`)
    2. Feature projection: MLP to align feature dimension
    3. Stacked augmentation layers: K layers of adaptive policy generation
       (see :class:`~foretools.tsaug.layers.StackedAugmentationLayers`)

    Parameters
    ----------
    num_layers : int
        K, number of stacked augmentation layers. Default 3.
    num_transforms : int
        Number of available transformations in T. Default is NUM_TRANSFORMS (12).
    feature_dim : int
        Dimension of extracted feature vector F_i. Default is FEATURE_DIM (24).
    hidden_dim : int
        Hidden dimension for policy MLPs in each layer and feature projection.
            Default 64.
    init_temperature : float
        Initial Gumbel-Softmax temperature τ for all layers. Default 1.0.
    raw_bias : float
        Per-layer probability of selecting Raw transform during training.
            Default 0.1 (10% chance of no augmentation per layer).

    Attributes
    ----------
    feature_proj : nn.Sequential
        MLP projecting features from ``feature_dim → hidden_dim → feature_dim``.
    aug_layers : StackedAugmentationLayers
        The K stacked augmentation layers.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug import AutoDATimeseries
    >>> autoda = AutoDATimeseries(num_layers=3, hidden_dim=64)
    >>> x = torch.randn(8, 100, 1)  # (batch, length, channels)
    >>> x_aug, probs, intensities, selected = autoda(x)
    >>> print(f"Augmented shape: {x_aug.shape}")  # doctest: +SKIP
    Augmented shape: torch.Size([8, 100, 1])
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
        precomputed_features: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        """Generate augmented time series with adaptive policy.

        Parameters
        ----------
        x : torch.Tensor
            Raw time series tensor of shape (batch_size, length, channels).
        precomputed_features : torch.Tensor | None
            Optional pre-extracted features of shape (batch_size, feature_dim).
                If None, features are computed on-the-fly via :func:`extract_features`.

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
            ``(x_aug, all_probs, all_intensities, all_selected)`` where:

            - ``x_aug``: Augmented time series of shape (batch_size, length, channels)
            - ``all_probs``: List of K tensors, each (batch_size, num_transforms)
            - ``all_intensities``: List of K tensors, each (batch_size, num_transforms)
            - ``all_selected``: List of K tensors, each (batch_size,) transform indices

        Examples
        --------
        >>> import torch
        >>> from foretools.tsaug import AutoDATimeseries
        >>> autoda = AutoDATimeseries(num_layers=3)
        >>> x = torch.randn(4, 100, 1)
        >>> # Forward pass (training mode: Gumbel-Softmax sampling)
        >>> x_aug, probs, intensities, selected = autoda(x)
        >>> print(f"Augmented shape: {x_aug.shape}")  # doctest: +SKIP
        Augmented shape: torch.Size([4, 100, 1])
        """
        # Feature extraction (Section 3.3)
        if precomputed_features is not None:
            features = precomputed_features
        else:
            features = extract_features(x)  # (B, 24)

        # Project features
        features = self.feature_proj(features)  # (B, feature_dim)

        # Apply stacked augmentation layers
        x_aug, all_probs, all_intensities, all_selected = self.aug_layers(x, features)

        return x_aug, all_probs, all_intensities, all_selected

    def get_policy_summary(
        self,
        all_probs: list[torch.Tensor],
        all_intensities: list[torch.Tensor],
        all_selected: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Get a human-readable summary of the augmentation policy.

        Computes average probabilities and intensities per layer and transform,
        along with the current Gumbel-Softmax temperature.

        Parameters
        ----------
        all_probs : list[torch.Tensor]
            List of K probability tensors, each (batch_size, num_transforms).
        all_intensities : list[torch.Tensor]
            List of K intensity tensors, each (batch_size, num_transforms).
        all_selected : list[torch.Tensor]
            List of K selected index tensors, each (batch_size,).

        Returns
        -------
        dict[str, Any]
            Policy summary dictionary with keys ``"layer_0"``, ``"layer_1"``, ...:

            .. code-block:: python

                {
                    "layer_0": {
                        "temperature": 0.85,
                        "avg_probabilities": {"Raw": 0.12, "Jittering": 0.23, ...},
                        "avg_intensities": {"Raw": 0.0, "Jittering": 0.45, ...},
                    },
                    ...
                }

        Examples
        --------
        >>> import torch
        >>> from foretools.tsaug import AutoDATimeseries
        >>> autoda = AutoDATimeseries(num_layers=3)
        >>> x = torch.randn(8, 100, 1)
        >>> x_aug, probs, intensities, selected = autoda(x)
        >>> summary = autoda.get_policy_summary(probs, intensities, selected)
        >>> print(f"Layer 0 temperature: {summary['layer_0']['temperature']:.3f}")  # doctest: +SKIP
        Layer 0 temperature: 0.850
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

    Jointly optimizes the augmentation framework parameters θ and
    downstream model parameters θ_M (Section 3.1, Eqs. 2-3).

    Training Procedure
    ------------------
    The trainer uses a single Adam optimizer with three parameter groups:

    1. Downstream model parameters — learning rate ``lr``
    2. Augmentation framework parameters — learning rate ``aug_lr``
    3. Composite loss weights — learning rate ``aug_lr``

    A cosine annealing scheduler is used with T_max=100.

    Loss Function
    -------------
    The composite loss combines three terms (Section 3.5.2):

        L_composite = w_1^2 * L_task + w_2^2 * L_intra_diversity + w_3^2 * L_inter_diversity

    where the weights w_z are learnable and optimized jointly.

    Parameters
    ----------
    autoda : AutoDATimeseries
        Augmented data generator instance.
    downstream_model : torch.nn.Module
        Any PyTorch module for the downstream task (e.g., classifier, regressor).
    task : str
        Task type determining loss function. One of ``'classification'``,
            ``'forecasting'``, ``'regression'``, or ``'anomaly'``.
            Default 'forecasting'.
    task_loss_fn : torch.nn.Module | None
        Custom loss function for the downstream task. If None, uses
            CrossEntropyLoss for classification and MSELoss otherwise.
    lr : float
        Learning rate for downstream model parameters. Default 1e-3.
    aug_lr : float | None
        Learning rate for augmentation parameters. Defaults to ``lr`` if None.
    weight_decay : float
        Weight decay (L2 regularization) for downstream model optimizer. Default 1e-4.
    device : str
        PyTorch device string ('cpu', 'cuda', etc.). Default 'cpu'.

    Attributes
    ----------
    autoda : AutoDATimeseries
        The augmentation framework (moved to ``device``).
    downstream_model : torch.nn.Module
        The downstream task model (moved to ``device``).
    task_loss_fn : torch.nn.Module
        Loss function for the downstream task.
    composite_loss : CompositeLoss
        Learnable loss balancing module.
    optimizer : torch.optim.Adam
        Optimizer with three parameter groups.
    scheduler : torch.optim.lr_scheduler.CosineAnnealingLR
        Cosine annealing learning rate scheduler.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug import AutoDATimeseries, AutoDATrainer
    >>> class SimpleClassifier(torch.nn.Module):
    ...     def __init__(self, input_dim: int = 10):
    ...         super().__init__()
    ...         self.fc = torch.nn.Linear(input_dim, 2)
    ...     def forward(self, x):
    ...         return self.fc(x.mean(dim=1))
    >>> autoda = AutoDATimeseries(num_layers=3)
    >>> downstream = SimpleClassifier()
    >>> trainer = AutoDATrainer(autoda, downstream, task="classification")
    """

    def __init__(
        self,
        autoda: AutoDATimeseries,
        downstream_model: nn.Module,
        task: str = "forecasting",
        task_loss_fn: nn.Module | None = None,
        lr: float = 1e-3,
        aug_lr: float | None = None,
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
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.downstream_model.parameters(),
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
                {"params": self.autoda.parameters(), "lr": aug_lr},
                {"params": self.composite_loss.parameters(), "lr": aug_lr},
            ]
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        precomputed_features: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Execute a single training step.

        Generates augmented data via the AutoDA framework, computes task loss
        through the downstream model, applies composite loss weighting, and
        performs a gradient update with gradient clipping.

        Parameters
        ----------
        x : torch.Tensor
            Input time series tensor of shape (batch_size, length, channels).
        y : torch.Tensor
            Target labels or values (shape depends on task).
        precomputed_features : torch.Tensor | None
            Optional pre-extracted features of shape (batch_size, feature_dim).
                If None, features are computed on-the-fly.

        Returns
        -------
        dict[str, float]
            Loss dictionary with keys ``'total'``, ``'L1'`` (task loss),
                ``'w1'``, ``'w2'``, ``'w3'``, ``'intra_entropy'``, and ``'inter_kl'``.

        Examples
        --------
        >>> import torch
        >>> from foretools.tsaug import AutoDATimeseries, AutoDATrainer
        >>> autoda = AutoDATimeseries(num_layers=2)
        >>> downstream = torch.nn.Linear(10, 2)
        >>> trainer = AutoDATrainer(autoda, downstream, task="classification")
        >>> x = torch.randn(4, 50, 1)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> loss_details = trainer.train_step(x, y)  # doctest: +SKIP
        >>> print(f"Total loss: {loss_details['total']:.4f}")  # doctest: +SKIP
        Total loss: 0.6931
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
            list(self.autoda.parameters())
            + list(self.downstream_model.parameters())
            + list(self.composite_loss.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return loss_details

    @torch.no_grad()
    def eval_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        """Evaluation step on original (non-augmented) data.

        At test time, only the downstream model is used without augmentation
        (Section 3.1, Eq. 3). For classification tasks, accuracy is also computed.

        Parameters
        ----------
        x : torch.Tensor
            Input time series tensor of shape (batch_size, length, channels).
        y : torch.Tensor
            Target labels or values.

        Returns
        -------
        dict[str, float]
            Evaluation results with ``'eval_loss'`` and optionally ``'accuracy'``
                for classification tasks.

        Examples
        --------
        >>> import torch
        >>> from foretools.tsaug import AutoDATimeseries, AutoDATrainer
        >>> autoda = AutoDATimeseries(num_layers=2)
        >>> downstream = torch.nn.Linear(10, 2)
        >>> trainer = AutoDATrainer(autoda, downstream, task="classification")
        >>> x = torch.randn(4, 50, 1)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> results = trainer.eval_step(x, y)  # doctest: +SKIP
        >>> print(f"Loss: {results['eval_loss']:.4f}")  # doctest: +SKIP
        Loss: 0.6931
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
    ) -> dict[str, list]:
        """Execute the full training loop.

        Trains both the augmentation framework and downstream model jointly
        over multiple epochs. Optionally precomputes features for efficiency.
        Prints progress at regular intervals including temperature values
        for each layer.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Training DataLoader yielding (x, y) batches.
        val_loader : torch.utils.data.DataLoader | None
            Optional validation DataLoader. If provided, validation loss is
                tracked and printed each epoch.
        epochs : int
            Number of training epochs. Default 50.
        log_interval : int
            Print progress every N epochs. Default 10.
        precompute_features : bool
            Whether to precompute and cache features before training for
                efficiency. Default True (recommended).

        Returns
        -------
        dict[str, list]
            Training history dictionary with keys:

            - ``'train_loss'``: List of average training loss per epoch
            - ``'val_loss'``: List of validation loss per epoch (if val_loader provided)
            - ``'val_accuracy'``: List of validation accuracy per epoch (classification only)

        Examples
        --------
        >>> import torch
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> from foretools.tsaug import AutoDATimeseries, AutoDATrainer
        >>> autoda = AutoDATimeseries(num_layers=2)
        >>> downstream = torch.nn.Linear(10, 2)
        >>> trainer = AutoDATrainer(autoda, downstream, task="classification")
        >>> x_train = torch.randn(32, 50, 1)
        >>> y_train = torch.randint(0, 2, (32,))
        >>> train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8)
        >>> history = trainer.fit(train_loader, epochs=5)  # doctest: +SKIP
        >>> print(f"Final train loss: {history['train_loss'][-1]:.4f}")  # doctest: +SKIP
        Final train loss: 0.3456
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
                msg = f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}"
                if val_loader is not None:
                    msg += f" | Val Loss: {val_results['eval_loss']:.4f}"
                    if "accuracy" in val_results:
                        msg += f" | Val Acc: {val_results['accuracy']:.4f}"

                # Policy summary
                policy = self.autoda.get_policy_summary(
                    *self._get_sample_policy(train_loader)
                )
                temps = [
                    policy[f"layer_{k}"]["temperature"]
                    for k in range(self.autoda.num_layers)
                ]
                msg += f" | Temps: {[f'{t:.3f}' for t in temps]}"
                print(msg)

        return history

    def _evaluate(self, loader) -> dict[str, float]:
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
