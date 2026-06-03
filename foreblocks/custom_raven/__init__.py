from __future__ import annotations

import sys
from pathlib import Path


def _fla_checkout_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "third-party" / "flash-linear-attention"
    if path.exists():
        return path
    return repo_root / "flash-linear-attention"


_fla_path = _fla_checkout_path()
if not (_fla_path / "fla").is_dir():
    raise ModuleNotFoundError(
        "flash-linear-attention checkout not found. Run "
        "`git submodule update --init --recursive` from the repository root."
    )

_fla_path_str = str(_fla_path)
if _fla_path_str not in sys.path:
    sys.path.insert(0, _fla_path_str)

from fla.layers.raven import Raven

__all__ = ["Raven"]
