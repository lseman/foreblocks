"""foreblocks.ops.attention.fla_backend.

Lazy adapter for the optional flash-linear-attention (FLA) submodule.

Manages the FLA git submodule path, provides context managers for temporary
sys.path injection, and caches upstream module imports. Exposes typed helpers
for delta rule, gated delta rule, GDN-2, linear attention, and GLA kernels.
Use when your code needs to conditionally use FLA kernels without making them
hard dependencies.

Core API:
- import_fla_module: import and cache an upstream FLA module
- fla_import_path: context manager to temporarily add FLA to sys.path
- is_fla_available: check FLA module availability
- fla_path / has_fla_checkout: FLA checkout detection

"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from types import ModuleType

_SUBMODULE_REL = Path("third_party") / "flash-linear-attention"
_FALLBACK_REL = Path("flash-linear-attention")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def fla_path() -> Path:
    root = _repo_root()
    path = root / _SUBMODULE_REL
    if path.exists():
        return path
    return root / _FALLBACK_REL


def has_fla_checkout() -> bool:
    path = fla_path()
    return (path / "fla").is_dir()


@contextmanager
def fla_import_path() -> Iterator[Path]:
    path = fla_path()
    path_str = str(path)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield path
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


@cache
def import_fla_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
            raise
    if not has_fla_checkout():
        raise ModuleNotFoundError(
            f"flash-linear-attention is not installed and checkout not found at {fla_path()}"
        )
    with fla_import_path():
        return importlib.import_module(module_name)


def get_fla_attr(module_name: str, attr_name: str):
    return getattr(import_fla_module(module_name), attr_name)


def is_fla_available(module_name: str = "fla.ops.linear_attn") -> bool:
    try:
        import_fla_module(module_name)
    except Exception:
        return False
    return True


def fla_chunk_delta_rule():
    return get_fla_attr("fla.ops.delta_rule", "chunk_delta_rule")


def fla_fused_chunk_delta_rule():
    return get_fla_attr("fla.ops.delta_rule", "fused_chunk_delta_rule")


def fla_fused_recurrent_delta_rule():
    return get_fla_attr("fla.ops.delta_rule", "fused_recurrent_delta_rule")


def fla_chunk_gla():
    return get_fla_attr("fla.ops.gla", "chunk_gla")


def fla_fused_recurrent_gla():
    return get_fla_attr("fla.ops.gla", "fused_recurrent_gla")


def fla_chunk_linear_attn():
    return get_fla_attr("fla.ops.linear_attn", "chunk_linear_attn")


def fla_fused_recurrent_linear_attn():
    return get_fla_attr("fla.ops.linear_attn", "fused_recurrent_linear_attn")


def fla_chunk_gated_delta_rule():
    return get_fla_attr("fla.ops.gated_delta_rule", "chunk_gated_delta_rule")


def fla_fused_recurrent_gated_delta_rule():
    return get_fla_attr("fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule")


def fla_chunk_kda():
    return get_fla_attr("fla.ops.kda", "chunk_kda")


def fla_fused_recurrent_kda():
    return get_fla_attr("fla.ops.kda", "fused_recurrent_kda")


def fla_chunk_gdn2():
    return get_fla_attr("fla.ops.gdn2", "chunk_gdn2")


def fla_fused_recurrent_gdn2():
    return get_fla_attr("fla.ops.gdn2", "fused_recurrent_gdn2")


def fla_rms_norm_gated():
    return get_fla_attr("fla.modules.layernorm_gated", "RMSNormGated")


__all__ = [
    "fla_chunk_delta_rule",
    "fla_chunk_gated_delta_rule",
    "fla_chunk_gla",
    "fla_chunk_gdn2",
    "fla_chunk_kda",
    "fla_chunk_linear_attn",
    "fla_fused_recurrent_gdn2",
    "fla_fused_chunk_delta_rule",
    "fla_fused_recurrent_gated_delta_rule",
    "fla_fused_recurrent_delta_rule",
    "fla_fused_recurrent_gla",
    "fla_fused_recurrent_kda",
    "fla_fused_recurrent_linear_attn",
    "fla_import_path",
    "fla_path",
    "fla_rms_norm_gated",
    "get_fla_attr",
    "has_fla_checkout",
    "import_fla_module",
    "is_fla_available",
]
