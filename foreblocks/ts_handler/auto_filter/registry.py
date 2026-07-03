"""foreblocks.ts_handler.auto_filter.registry.

This module implements the registry pieces for its package.
It belongs to the automatic signal filtering and denoising pipelines area of Foreblocks.
It exposes functions such as register_filter.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


_FILTER_REGISTRY: dict[str, Callable[[pd.Series], pd.Series]] = {}
_SLOW_FILTERS: set[str] = set()


def register_filter(name: str, *, slow: bool = False):
    """Decorator that adds a filter function to the auto-selection registry."""

    def decorator(fn: Callable[[pd.Series], pd.Series]):
        _FILTER_REGISTRY[name] = fn
        if slow:
            _SLOW_FILTERS.add(name)
        return fn

    return decorator
