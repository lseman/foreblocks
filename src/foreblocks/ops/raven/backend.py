"""foreblocks.ops.raven.backend.

Import-path management and lazy loading for the flash-linear-attention (FLA) package.

Provides helpers to locate the FLA checkout, temporarily prepend it to
sys.path, and lazily import FLA submodules — only when actually needed.
Raises a clear error if FLA is neither installed nor checked out. Use
when building integration layers that optionally depend on the upstream FLA library.

Core API:
- fla_path: resolve the FLA repository root path
- has_fla_checkout: check whether the FLA checkout directory exists
- fla_import_path: context manager for temporarily adding FLA to sys.path
- import_fla_module: lazily import an FLA submodule (installs/checks out if needed)

"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from importlib import import_module
from pathlib import Path
from types import ModuleType


def fla_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    submodule = repo_root / "third_party" / "flash-linear-attention"
    if submodule.exists():
        return submodule
    return repo_root / "flash-linear-attention"


def has_fla_checkout() -> bool:
    return (fla_path() / "fla").is_dir()


@contextmanager
def fla_import_path() -> Iterator[None]:
    path = str(fla_path())
    inserted = False
    if path not in sys.path:
        sys.path.insert(0, path)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


@cache
def import_fla_module(module_name: str) -> ModuleType:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
            raise
    if not has_fla_checkout():
        raise ModuleNotFoundError(
            "flash-linear-attention is not installed and checkout not found. "
            "Install the `fla` extra or run "
            "`git submodule update --init --recursive` from the repository root."
        )
    with fla_import_path():
        return import_module(module_name)


def get_fla_attr(module_name: str, attr_name: str):
    return getattr(import_fla_module(module_name), attr_name)


__all__ = [
    "fla_import_path",
    "fla_path",
    "get_fla_attr",
    "has_fla_checkout",
    "import_fla_module",
]
