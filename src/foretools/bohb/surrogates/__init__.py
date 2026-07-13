"""Surrogate models layer (GP, KDE, ensemble)."""

from .base import Surrogate
from .gp import GPSurrogate, GPEnsemble, ExpectedImprovement, UpperConfidenceBound, vectorized_ei_score

__all__ = [
    "Surrogate",
    "GPSurrogate",
    "GPEnsemble",
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "vectorized_ei_score",
]
