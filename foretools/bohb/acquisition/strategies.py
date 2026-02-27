"""Backward-compatible re-exports for legacy imports."""

from .base import AcquisitionStrategy
from .factory import build_acquisition_strategy
from .log_ratio import LogRatioAcquisition

__all__ = [
    "AcquisitionStrategy",
    "LogRatioAcquisition",
    "build_acquisition_strategy",
]
