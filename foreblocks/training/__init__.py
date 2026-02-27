from typing import TYPE_CHECKING

__all__ = ["Trainer"]

if TYPE_CHECKING:
    from .trainer import Trainer


def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer as _Trainer

        return _Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
