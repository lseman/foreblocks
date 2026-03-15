# AutoDA-Timeseries (tsaug)

`foretools.tsaug` is an automated data augmentation framework for time series. It implements the AutoDA-Timeseries method, which jointly learns *which* augmentations to apply and *how strongly* to apply them, in a single end-to-end training pass.

!!! note "Research module"
    This module implements a method currently under review. The API may change between minor versions.

## Overview

Traditional augmentation pipelines pick a fixed policy manually. AutoDA-Timeseries extracts statistical features from each batch and uses them to predict per-sample augmentation probabilities and intensities, so the augmentation policy adapts to the data distribution automatically.

**Transformations available:**

| Name | Effect |
|---|---|
| `raw` | Identity — no change |
| `jittering` | Additive Gaussian noise |
| `scaling` | Random multiplicative scale |
| `resample` | Interpolate to random length and back |
| `time_warp` | Nonlinear time-axis distortion |
| `freq_warp` | Frequency-domain amplitude perturbation |
| `mag_warp` | Smooth random magnitude envelope |
| `time_mask` | Zero-out a random contiguous window |
| `drift` | Add a low-frequency drift trend |

## Quick start

```python
import torch
from foretools.tsaug import AutoDATimeseries, AutoDATrainer

# x: (batch, length, channels) — your training batch
x = torch.randn(32, 96, 7)

# Build model: wraps your backbone with augmentation
model = AutoDATimeseries(
    backbone=your_forecasting_model,
    feature_dim=32,      # internal feature size for augmentation policy
    num_layers=2,        # depth of the policy network
)

# Standard training step
trainer = AutoDATrainer(model, lr=1e-3)
loss = trainer.step(x, y_target)
```

## `AugmentationLayer`

A single differentiable augmentation step. Takes a batch and returns an augmented batch, where augmentation probabilities and intensities come from learned weights conditioned on input features.

```python
from foretools.tsaug import AugmentationLayer

layer = AugmentationLayer(feature_dim=32)

x_aug = layer(x)  # (batch, length, channels)
```

## `StackedAugmentationLayers`

Chains multiple `AugmentationLayer` instances sequentially. Each layer applies an independent augmentation decision.

```python
from foretools.tsaug import StackedAugmentationLayers

aug = StackedAugmentationLayers(num_layers=3, feature_dim=32)
x_aug = aug(x)
```

## Feature extraction

The policy network is conditioned on a fixed-size feature vector extracted from the input:

```python
from foretools.tsaug import extract_features, FEATURE_DIM

feats = extract_features(x)   # (batch, FEATURE_DIM)
print(FEATURE_DIM)             # number of features extracted
```

Features include statistical moments, autocorrelation, spectral energy, and trend strength — computed per channel and aggregated across the batch dimension.

## `CompositeLoss`

Combines the task loss with an augmentation regularisation term that encourages diverse augmentation usage:

```python
from foretools.tsaug import CompositeLoss

criterion = CompositeLoss(task_loss=nn.MSELoss(), diversity_weight=0.01)
loss = criterion(y_pred, y_true, aug_probs)
```

## Applying transformations directly

Each transformation function is available standalone:

```python
from foretools.tsaug import jittering, time_mask

x_noisy = jittering(x, intensity=0.05)
x_masked = time_mask(x, intensity=0.1)
```

The `intensity` argument can be a scalar or a `(batch,)` tensor for per-sample control.
