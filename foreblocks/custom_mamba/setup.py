from __future__ import annotations

import os

from setuptools import setup


def _get_version() -> str:
    return os.environ.get("CUSTOM_MAMBA_VERSION", "0.1.0")


setup(
    name="custom-mamba",
    version=_get_version(),
    description="Mamba2-style PyTorch and Triton sequence-mixing blocks",
    packages=["custom_mamba", "custom_mamba.blocks", "custom_mamba.ops"],
    package_dir={
        "custom_mamba": ".",
        "custom_mamba.blocks": "blocks",
        "custom_mamba.ops": "ops",
    },
    include_package_data=True,
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
