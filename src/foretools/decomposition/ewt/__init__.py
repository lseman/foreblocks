"""Empirical Wavelet Transform (EWT) decomposition."""

from .ewt_core import EWT1D, EWT_Boundaries_Detect, EWT_Meyer_FilterBank

__all__ = [
    "EWT1D",
    "EWT_Boundaries_Detect",
    "EWT_Meyer_FilterBank",
]
