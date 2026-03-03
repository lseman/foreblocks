"""
Backward-compatibility shim for NAS metrics.

All classes (Config, Result, MetricsComputer, ZeroCostNAS …) have moved to
search/nas_metrics.py. Existing imports continue to work unchanged.
"""

from .search.nas_metrics import *  # noqa: F401, F403
