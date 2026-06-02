from __future__ import annotations

import importlib
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load

EXTENSION_NAME = "custom_mamba_selective_scan"
PACKAGE_ROOT = Path(__file__).resolve().parent
CSRC_DIR = PACKAGE_ROOT / "csrc"
DEFAULT_BUILD_DIR = PACKAGE_ROOT / ".build"
FORCE_ENV_VAR = "CUSTOM_MAMBA_FORCE_CUDA_VERSION"
DEFAULT_ARCH_LIST_ENV_VAR = "CUSTOM_MAMBA_CUDA_ARCH_LIST"
DEFAULT_ARCH_LIST = "8.0;8.6;8.9;9.0"
STALE_LOCK_SECONDS_ENV_VAR = "CUSTOM_MAMBA_STALE_LOCK_SECONDS"
DEFAULT_STALE_LOCK_SECONDS = 60 * 60

_EXT = None


def extension_available() -> bool:
    return torch.cuda.is_available()


def get_default_build_dir() -> Path:
    raw = os.environ.get("CUSTOM_MAMBA_BUILD_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_BUILD_DIR / _build_cache_tag()


def _sanitize_tag_part(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _build_cache_tag() -> str:
    parts = [
        f"py{sys.version_info.major}{sys.version_info.minor}",
        f"torch{_sanitize_tag_part(torch.__version__)}",
        f"cuda{_sanitize_tag_part(torch.version.cuda or 'none')}",
    ]
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        parts.append(f"sm{major}{minor}")
    return "-".join(parts)


def _current_device_arch_list() -> str | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def _should_force_cuda_version() -> bool:
    return os.environ.get(FORCE_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _ensure_cuda_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    arch_list = os.environ.get(DEFAULT_ARCH_LIST_ENV_VAR, "").strip()
    if arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
        return
    current_arch = _current_device_arch_list()
    if current_arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = current_arch
        return
    if not torch.cuda.is_available():
        os.environ["TORCH_CUDA_ARCH_LIST"] = DEFAULT_ARCH_LIST


def _stale_lock_seconds() -> int:
    raw = os.environ.get(STALE_LOCK_SECONDS_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_STALE_LOCK_SECONDS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_STALE_LOCK_SECONDS


def _clear_stale_build_lock(build_dir: Path) -> None:
    lock_path = build_dir / "lock"
    if not lock_path.exists():
        return
    stale_after = _stale_lock_seconds()
    if stale_after == 0:
        return
    try:
        age_seconds = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return
    if age_seconds >= stale_after:
        lock_path.unlink(missing_ok=True)


def load_selective_scan_extension(
    *,
    verbose: bool = False,
    force: bool = False,
    build_directory: str | os.PathLike[str] | None = None,
):
    global _EXT
    if _EXT is not None and not force:
        return _EXT

    if not force:
        try:
            _EXT = importlib.import_module(EXTENSION_NAME)
            return _EXT
        except ImportError:
            pass

    if _should_force_cuda_version():
        cpp_extension._check_cuda_version = lambda *args, **kwargs: None
    _ensure_cuda_arch_list()

    build_dir = Path(build_directory) if build_directory else get_default_build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_build_lock(build_dir)

    _EXT = load(
        name=EXTENSION_NAME,
        sources=[
            str(CSRC_DIR / "selective_scan_bindings.cpp"),
            str(CSRC_DIR / "selective_scan.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        with_cuda=True,
        verbose=verbose,
        build_directory=str(build_dir),
    )
    return _EXT


def precompile_selective_scan_extension(
    *,
    verbose: bool = True,
    force: bool = False,
    build_directory: str | os.PathLike[str] | None = None,
):
    return load_selective_scan_extension(
        verbose=verbose,
        force=force,
        build_directory=build_directory,
    )
