from setuptools import setup

setup(
    name="custom_raven",
    version="0.1.0",
    packages=["custom_raven"],
    package_dir={"custom_raven": "."},
    python_requires=">=3.10",
    install_requires=["torch", "triton", "einops", "transformers>=4.45.0"],
)
