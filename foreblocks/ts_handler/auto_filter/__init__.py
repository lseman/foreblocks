"""foreblocks.ts_handler.auto_filter.

Automatic time-series denoising with filter selection.

Provides a curated set of signal processing filters with automatic selection
based on signal characteristics. Supports wavelet, Kalman, lowess, Savitzky-
Golay, and deep learning-based denoising.

Core API:
- AutoFilter: main auto-selection denoising interface
- register_filter: register custom filters in the auto-selection registry

"""

from __future__ import annotations

import sys as _sys
import types as _types

from foreblocks.ts_handler.auto_filter import core as _core

__all__ = _core.__all__


class _AutoFilterPackage(_types.ModuleType):
    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if not name.startswith("__"):
            setattr(_core, name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        if hasattr(_core, name):
            delattr(_core, name)


def __getattr__(name: str):
    return getattr(_core, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_core)))


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

_sys.modules[__name__].__class__ = _AutoFilterPackage
