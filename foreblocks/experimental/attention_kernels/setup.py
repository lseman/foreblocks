"""foreblocks.experimental.attention_kernels.setup.

Build and setup helpers for the local extension package.
It belongs to the experimental attention kernel implementations and benchmarks area of Foreblocks.

"""

from setuptools import setup

setup(
    name="custom_att",
    version="0.1.0",
    packages=["custom_att", "custom_att.src"],
    package_dir={"custom_att": ".", "custom_att.src": "src"},
    python_requires=">=3.10",
    install_requires=["torch", "triton"],
)
