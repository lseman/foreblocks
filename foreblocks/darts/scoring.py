"""
Backward-compatibility shim for scoring utilities.

Scoring functions have moved to search/scoring.py, their natural home
alongside the other search modules that consume them.
"""

from .search.scoring import normalize_metric_value
from .search.scoring import score_from_metrics


__all__ = ["normalize_metric_value", "score_from_metrics"]
