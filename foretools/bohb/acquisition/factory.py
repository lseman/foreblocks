from __future__ import annotations

from .base import AcquisitionStrategy
from .log_ratio import LogRatioAcquisition


def build_acquisition_strategy(*args, **kwargs) -> AcquisitionStrategy:
    return LogRatioAcquisition(*args, **kwargs)

