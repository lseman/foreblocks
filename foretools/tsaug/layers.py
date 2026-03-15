"""
Stacked augmentation layers for AutoDA-Timeseries.

Each augmentation layer A^(k) receives:
  (i)   input time series from the previous layer
  (ii)  previous probability vector p^(k-1)
  (iii) global feature vector F_i

and generates probability p^(k)_{i,j} and intensity t^(k)_{i,j} via MLPs,
then samples a transformation using Gumbel-Softmax (Section 3.4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformations import TRANSFORMATIONS, NUM_TRANSFORMS


class AugmentationLayer(nn.Module):
    """Single augmentation layer A^(k)_{theta_k}.

    Generates augmentation policy (probability + intensity) conditioned on
    time series features and previous layer's probability, then applies
    a differentiably-selected transformation.

    Args:
        feature_dim: Dimension of the time series feature vector.
        num_transforms: Number of available transformations.
        hidden_dim: Hidden dimension for policy MLPs.
        init_temperature: Initial Gumbel-Softmax temperature.
    """

    def __init__(
        self,
        feature_dim: int = 24,
        num_transforms: int = NUM_TRANSFORMS,
        hidden_dim: int = 64,
        init_temperature: float = 1.0,
    ):
        super().__init__()
        self.num_transforms = num_transforms

        input_dim = feature_dim + num_transforms  # [p^(k-1), F_i]

        # MLP for probability generation: f^(k)_p (Eq. 4)
        self.prob_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_transforms),
        )

        # MLP for intensity generation: f^(k)_t (Eq. 5)
        self.intensity_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_transforms),
            nn.Softplus(),  # Ensure t >= 0
        )

        # Learnable temperature for Gumbel-Softmax (Section 3.5.1)
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(min=0.01, max=10.0)

    def forward(
        self,
        x: torch.Tensor,
        prev_prob: torch.Tensor,
        features: torch.Tensor,
    ):
        """Forward pass through augmentation layer.

        Args:
            x: (B, L, C) input time series.
            prev_prob: (B, num_transforms) probability vector from previous layer.
            features: (B, feature_dim) global time series feature vector.

        Returns:
            x_aug: (B, L, C) augmented time series.
            prob: (B, num_transforms) augmentation probability vector.
            intensity: (B, num_transforms) augmentation intensity vector.
            selected_idx: (B,) index of selected transformation.
        """
        B = x.size(0)

        # Concatenate previous probability and features
        mlp_input = torch.cat([prev_prob, features], dim=1)  # (B, n + feature_dim)

        # Generate probability logits and intensity (Eqs. 4-5)
        prob_logits = self.prob_mlp(mlp_input)  # (B, n)
        intensity = self.intensity_mlp(mlp_input)  # (B, n)

        # Probability via softmax
        prob = F.softmax(prob_logits, dim=-1)  # (B, n)

        # Gumbel-Softmax sampling for differentiable selection (Eq. 6)
        if self.training:
            selection = F.gumbel_softmax(
                prob_logits, tau=self.temperature, hard=True, dim=-1
            )  # (B, n) one-hot (hard) with soft gradients
        else:
            # At inference, select the most probable transformation
            idx = prob_logits.argmax(dim=-1)
            selection = F.one_hot(idx, self.num_transforms).float()

        selected_idx = selection.argmax(dim=-1)  # (B,)

        # Get intensity for the selected transform
        selected_intensity = (intensity * selection).sum(dim=-1)  # (B,)

        # Apply the selected transformation (Eq. 7)
        # For differentiability, apply all transforms and mix via soft selection
        if self.training:
            # Straight-through: use hard selection in forward, soft gradients in backward
            x_aug = self._apply_mixed(x, selection, intensity)
        else:
            x_aug = self._apply_selected(x, selected_idx, selected_intensity)

        return x_aug, prob, intensity, selected_idx

    def _apply_mixed(
        self,
        x: torch.Tensor,
        selection: torch.Tensor,
        intensity: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transformations with Gumbel-Softmax mixing for training.

        Since selection is hard one-hot with straight-through gradients,
        only one transform is actually applied per sample.
        """
        B, L, C = x.shape
        results = []

        for j, transform_fn in enumerate(TRANSFORMATIONS):
            t_j = intensity[:, j]  # (B,)
            x_j = transform_fn(x, t_j)  # (B, L, C)
            results.append(x_j)

        # Stack: (n, B, L, C) -> select via one-hot
        results = torch.stack(results, dim=0)  # (n, B, L, C)
        # selection: (B, n) -> (n, B, 1, 1)
        weights = selection.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
        x_aug = (results * weights).sum(dim=0)  # (B, L, C)
        return x_aug

    def _apply_selected(
        self,
        x: torch.Tensor,
        selected_idx: torch.Tensor,
        selected_intensity: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the selected transformation at inference."""
        B, L, C = x.shape
        x_aug = torch.zeros_like(x)

        for j, transform_fn in enumerate(TRANSFORMATIONS):
            mask = (selected_idx == j)
            if mask.any():
                x_j = transform_fn(x[mask], selected_intensity[mask])
                x_aug[mask] = x_j

        return x_aug


class StackedAugmentationLayers(nn.Module):
    """Stack of K augmentation layers forming the augmented data generator A_theta.

    A_theta = A^(1) o A^(2) o ... o A^(K)

    Args:
        num_layers: K, number of stacked augmentation layers.
        feature_dim: Dimension of the time series feature vector.
        num_transforms: Number of available transformations.
        hidden_dim: Hidden dimension for policy MLPs.
        init_temperature: Initial Gumbel-Softmax temperature.
        raw_bias: Probability of selecting Raw transform (Section 3.5.3).
    """

    def __init__(
        self,
        num_layers: int = 3,
        feature_dim: int = 24,
        num_transforms: int = NUM_TRANSFORMS,
        hidden_dim: int = 64,
        init_temperature: float = 1.0,
        raw_bias: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_transforms = num_transforms
        self.raw_bias = raw_bias

        self.layers = nn.ModuleList([
            AugmentationLayer(
                feature_dim=feature_dim,
                num_transforms=num_transforms,
                hidden_dim=hidden_dim,
                init_temperature=init_temperature,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, features: torch.Tensor):
        """Forward pass through all K augmentation layers.

        Args:
            x: (B, L, C) raw time series.
            features: (B, feature_dim) global feature vector.

        Returns:
            x_aug: (B, L, C) augmented time series after K layers.
            all_probs: list of (B, n) probability vectors per layer.
            all_intensities: list of (B, n) intensity vectors per layer.
            all_selected: list of (B,) selected transform indices per layer.
        """
        B = x.size(0)
        prev_prob = torch.zeros(B, self.num_transforms, device=x.device)

        all_probs = []
        all_intensities = []
        all_selected = []

        x_current = x
        for k, layer in enumerate(self.layers):
            # Raw transform bias (Section 3.5.3)
            if self.training and self.raw_bias > 0:
                use_raw = torch.rand(B, device=x.device) < self.raw_bias
                # For samples that get raw bias, skip augmentation
                x_before = x_current.clone()

            x_aug, prob, intensity, selected_idx = layer(
                x_current, prev_prob, features
            )

            # Apply raw transform bias
            if self.training and self.raw_bias > 0:
                x_aug = torch.where(
                    use_raw.view(-1, 1, 1).expand_as(x_aug),
                    x_before,
                    x_aug,
                )

            all_probs.append(prob)
            all_intensities.append(intensity)
            all_selected.append(selected_idx)

            prev_prob = prob.detach()  # Detach to prevent gradient through prob chain
            x_current = x_aug

        return x_current, all_probs, all_intensities, all_selected
