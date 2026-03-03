"""
Backward-compatibility shim for DARTS plot / analysis utilities.

StreamlinedDARTSAnalyzer has moved to evaluation/analyzer.py.
Existing imports from this location continue to work unchanged.
"""

from .evaluation.analyzer import *  # noqa: F401, F403
