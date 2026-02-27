"""Backward-compatible re-exports for legacy imports."""

from .base import GammaStrategy
from .default import DefaultGammaStrategy
from .factory import build_gamma_strategy

__all__ = ["GammaStrategy", "DefaultGammaStrategy", "build_gamma_strategy"]
