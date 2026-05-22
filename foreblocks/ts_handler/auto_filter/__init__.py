from __future__ import annotations

import sys as _sys
import types as _types

from . import core as _core

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
