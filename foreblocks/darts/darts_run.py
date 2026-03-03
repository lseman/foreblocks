"""
Backward-compatibility shim.

The monolithic DARTSTrainer has been refactored into focused sub-modules.
Import from this location (or from foreblocks.darts) continues to work.

New canonical import:
    from foreblocks.darts.trainer import DARTSTrainer
"""

from .trainer import DARTSTrainer

__all__ = ["DARTSTrainer"]
