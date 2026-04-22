from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load

EXTENSION_NAME = "hybrid_mamba_selective_scan"
PACKAGE_ROOT = Path(__file__).resolve().parent
CSRC_DIR = PACKAGE_ROOT / "csrc"
DEFAULT_BUILD_DIR = PACKAGE_ROOT / ".build"
FORCE_ENV_VAR = "HYBRID_MAMBA_FORCE_CUDA_VERSION"
DEFAULT_ARCH_LIST_ENV_VAR = "HYBRID_MAMBA_CUDA_ARCH_LIST"
DEFAULT_ARCH_LIST = "8.0;8.6;8.9;9.0"

_EXT = None


def extension_available() -> bool:
    return torch.cuda.is_available()


def get_default_build_dir() -> Path:
    raw = os.environ.get("HYBRID_MAMBA_BUILD_DIR")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_BUILD_DIR


def _should_force_cuda_version() -> bool:
    return os.environ.get(FORCE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _ensure_cuda_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    arch_list = os.environ.get(DEFAULT_ARCH_LIST_ENV_VAR, "").strip()
    if arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
        return
    if not torch.cuda.is_available():
        os.environ["TORCH_CUDA_ARCH_LIST"] = DEFAULT_ARCH_LIST


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
