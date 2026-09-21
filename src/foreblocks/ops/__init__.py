"""Tensor operations, reference algorithms, and execution dispatch.

Hardware implementations live in foreblocks.kernels; optional external
backends live in foreblocks.integrations. Import the needed operation family
explicitly (attention or mamba) to avoid loading unrelated backends.
"""
