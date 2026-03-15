"""
AutoDA-Timeseries: Automated Data Augmentation for Time Series.

A general-purpose automated data augmentation framework that incorporates
time series features into augmentation policy design and adaptively optimizes
both augmentation probability and intensity in a single-stage, end-to-end manner.

Reference:
    "AutoDA-Timeseries: Automated Data Augmentation for Time Series"
    Under review at ICLR 2026.
"""

from .model import AutoDATimeseries, AutoDATrainer
from .layers import AugmentationLayer, StackedAugmentationLayers
from .losses import CompositeLoss
from .features import extract_features, FEATURE_DIM
from .transformations import (
    TRANSFORMATIONS,
    TRANSFORM_NAMES,
    NUM_TRANSFORMS,
    raw,
    jittering,
    scaling,
    resample,
    time_warp,
    freq_warp,
    mag_warp,
    time_mask,
    drift,
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
