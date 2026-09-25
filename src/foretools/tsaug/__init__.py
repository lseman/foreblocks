"""AutoDA-Timeseries: Automated Data Augmentation for Time Series.

A general-purpose automated data augmentation framework that incorporates
time series features into augmentation policy design and adaptively optimizes
both augmentation probability and intensity in a single-stage, end-to-end manner.

Reference
---------
"AutoDA-Timeseries: Automated Data Augmentation for Time Series"
Under review at ICLR 2026.

Architecture Overview
---------------------
The framework consists of three main components:

1. **Feature Extraction** — Extracts 24 descriptive statistics from each time series,
   forming a feature vector F_i that captures autocorrelation, distribution, and
   higher-order properties (inspired by Catch22 and tsfresh).

2. **Stacked Augmentation Layers** — K layers of adaptive augmentation where each
   layer generates probability p^(k)_{i,j} and intensity t^(k)_{i,j} via MLPs
   conditioned on time series features and previous layer's probability, then
   samples a transformation using Gumbel-Softmax.

3. **Composite Loss** — Combines task loss with diversity regularization:
   - L1: Task-specific loss (MSE, CE, etc.)
   - L2: Intra-layer diversity (Shannon entropy of probabilities)
   - L3: Inter-layer diversity (KL divergence between layers)

Quick Start
-----------
>>> import torch
>>> from foretools.tsaug import AutoDATimeseries, AutoDATrainer

>>> # Create the augmentation framework
>>> autoda = AutoDATimeseries(num_layers=3, hidden_dim=64)

>>> # Wrap with a downstream model (example: simple linear classifier)
>>> class SimpleClassifier(torch.nn.Module):
...     def __init__(self, input_dim: int = 10):
...         super().__init__()
...         self.fc = torch.nn.Linear(input_dim, 2)
...     def forward(self, x):
...         return self.fc(x.mean(dim=1))

>>> downstream = SimpleClassifier()
>>> trainer = AutoDATrainer(autoda, downstream, task="classification")

>>> # Generate augmented data (training mode uses Gumbel-Softmax sampling)
>>> x = torch.randn(32, 100, 1)  # (batch, length, channels)
>>> x_aug, probs, intensities, selected = autoda(x)
>>> print(f"Augmented shape: {x_aug.shape}")  # doctest: +SKIP
Augmented shape: torch.Size([32, 100, 1])

Transformations (T)
-------------------
The framework supports 12 transformations applied in sequence across K layers:

| Index | Name         | Description                                    |
|-------|--------------|------------------------------------------------|
| T1    | Raw          | Identity — returns input unchanged             |
| T2    | Jittering    | Add Gaussian noise scaled by intensity         |
| T3    | Scaling      | Multiply by random scaling factor              |
| T4    | Resample     | Interpolate to shorter/longer length           |
| T5    | TimeWarp     | Warp time axis using smooth random curve       |
| T6    | FreqWarp     | Perturb phase in Fourier domain                |
| T7    | MagWarp      | Multiply by smooth random curve along time     |
| T8    | TimeMask     | Mask contiguous window, fill with local mean   |
| T9    | Drift        | Add smooth low-frequency trend                 |
| T10   | Permutation  | Randomly permute temporal segments             |
| T11   | WindowSlice  | Crop a window and resize back to original      |
| T12   | TimeMix      | Mix segments between paired samples            |

Modules
-------
See the individual module docstrings for details:
- :mod:`foretools.tsaug.transformations` — All augmentation transformation functions
- :mod:`foretools.tsaug.features` — Feature extraction (24 statistics)
- :mod:`foretools.tsaug.layers` — AugmentationLayer and StackedAugmentationLayers
- :mod:`foretools.tsaug.losses` — CompositeLoss with learnable weights
- :mod:`foretools.tsaug.model` — AutoDATimeseries and AutoDATrainer

"""

from __future__ import annotations

__version__ = "0.1.0"

# Core framework classes
from .model import AutoDATimeseries, AutoDATrainer

# Augmentation layers
from .layers import AugmentationLayer, StackedAugmentationLayers

# Loss function
from .losses import CompositeLoss

# Feature extraction
from .features import FEATURE_DIM, extract_features

# Transformation functions and registry
from .transformations import (
    NUM_TRANSFORMS,
    TRANSFORM_NAMES,
    TRANSFORMATIONS,
    drift,
    freq_warp,
    jittering,
    mag_warp,
    permutation,
    raw,
    resample,
    scaling,
    time_mask,
    time_mix,
    time_warp,
    window_slice,
)

__all__ = [
    # Version
    "__version__",
    # Core framework
    "AutoDATimeseries",
    "AutoDATrainer",
    # Layers
    "AugmentationLayer",
    "StackedAugmentationLayers",
    # Loss
    "CompositeLoss",
    # Features
    "extract_features",
    "FEATURE_DIM",
    # Transformations registry
    "TRANSFORMATIONS",
    "TRANSFORM_NAMES",
    "NUM_TRANSFORMS",
    # Individual transformation functions
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
