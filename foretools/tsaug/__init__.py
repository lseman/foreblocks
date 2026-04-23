"""
AutoDA-Timeseries: Automated Data Augmentation for Time Series.

A general-purpose automated data augmentation framework that incorporates
time series features into augmentation policy design and adaptively optimizes
both augmentation probability and intensity in a single-stage, end-to-end manner.

Reference:
    "AutoDA-Timeseries: Automated Data Augmentation for Time Series"
    Under review at ICLR 2026.
"""

from .features import FEATURE_DIM
from .features import extract_features
from .layers import AugmentationLayer
from .layers import StackedAugmentationLayers
from .losses import CompositeLoss
from .model import AutoDATimeseries
from .model import AutoDATrainer
from .transformations import NUM_TRANSFORMS
from .transformations import TRANSFORM_NAMES
from .transformations import TRANSFORMATIONS
from .transformations import drift
from .transformations import freq_warp
from .transformations import jittering
from .transformations import mag_warp
from .transformations import raw
from .transformations import resample
from .transformations import scaling
from .transformations import time_mask
from .transformations import time_warp


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
