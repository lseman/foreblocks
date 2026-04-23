from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.utils.cpp_extension as cpp_extension
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
EXTENSION_NAME = "hybrid_mamba_selective_scan"
FORCE_ENV_VAR = "HYBRID_MAMBA_FORCE_CUDA_VERSION"
DEFAULT_ARCH_LIST_ENV_VAR = "HYBRID_MAMBA_CUDA_ARCH_LIST"
DEFAULT_ARCH_LIST = "8.0;8.6;8.9;9.0"


def _get_version() -> str:
    return os.environ.get("HYBRID_MAMBA_VERSION", "0.1.0")


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
    if not torch.cuda.is_available():
        os.environ["TORCH_CUDA_ARCH_LIST"] = DEFAULT_ARCH_LIST


if _should_force_cuda_version():
    cpp_extension._check_cuda_version = lambda *args, **kwargs: None

_ensure_cuda_arch_list()


def _make_extensions():
    return [
        CUDAExtension(
            name=EXTENSION_NAME,
            sources=[
                str(CSRC / "selective_scan_bindings.cpp"),
                str(CSRC / "selective_scan.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ]


setup(
    name="hybrid-mamba",
    version=_get_version(),
    description="Hybrid Triton + CUDA Mamba starter components",
    packages=["hybrid_mamba", "hybrid_mamba.ops"],
    package_dir={
        "hybrid_mamba": ".",
        "hybrid_mamba.ops": "ops",
    },
    include_package_data=True,
    ext_modules=_make_extensions(),
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=[
        "torch",
    ],
    extras_require={
        "triton": ["triton"],
        "fast": ["triton"],
    },
)
