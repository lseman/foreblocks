from __future__ import annotations

import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Iterator


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


def import_fla_module(module_name: str) -> ModuleType:
    if not has_fla_checkout():
        raise ModuleNotFoundError(
            "flash-linear-attention checkout not found. Run "
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
