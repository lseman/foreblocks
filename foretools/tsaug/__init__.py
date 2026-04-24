"""
AutoDA-Timeseries: Automated Data Augmentation for Time Series.

A general-purpose automated data augmentation framework that incorporates
time series features into augmentation policy design and adaptively optimizes
both augmentation probability and intensity in a single-stage, end-to-end manner.

Reference:
    "AutoDA-Timeseries: Automated Data Augmentation for Time Series"
    Under review at ICLR 2026.
"""

from .features import FEATURE_DIM, extract_features
from .layers import AugmentationLayer, StackedAugmentationLayers
from .losses import CompositeLoss
from .model import AutoDATimeseries, AutoDATrainer
from .transformations import (
    NUM_TRANSFORMS,
    TRANSFORM_NAMES,
    TRANSFORMATIONS,
    drift,
    freq_warp,
    jittering,
    mag_warp,
    raw,
    resample,
    scaling,
    time_mask,
    time_warp,
)


__version__ = "0.1.0"
__all__ = [
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
]
