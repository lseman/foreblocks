"""Accelerated kernels and their autograd wrappers, grouped by operation family.

No backend is imported at package load time. Individual implementations may
provide a PyTorch fallback or require an optional GPU runtime.
"""
