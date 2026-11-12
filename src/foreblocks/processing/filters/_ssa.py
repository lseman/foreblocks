"""Shared NumPy reconstruction for the SSA filter adapters.

Adapters own window selection, missing-data handling, and output containers.
This module only embeds a one-dimensional signal, truncates its SVD, and
averages the reconstructed trajectory matrix along anti-diagonals.
"""

from __future__ import annotations

import numpy as np


def _ssa_reconstruct(x: np.ndarray, window: int, n_components: int) -> np.ndarray:
    """Reconstruct a finite 1-D signal with a caller-selected window and rank.

    Callers supply a window between 1 and len(x). Nonpositive ranks produce
    zeros; ranks above the matrix rank retain all singular components.
    SVD failures propagate so each adapter can apply its own failure policy.
    """
    length = len(x)
    columns = length - window + 1
    trajectory = np.column_stack([x[j : j + window] for j in range(columns)])
    u, singular_values, vh = np.linalg.svd(trajectory, full_matrices=False)

    reconstructed = np.zeros_like(trajectory)
    for j in range(min(n_components, len(singular_values))):
        reconstructed += singular_values[j] * np.outer(u[:, j], vh[j, :])

    result = np.zeros(length)
    counts = np.zeros(length)
    for column in range(columns):
        for row in range(window):
            result[row + column] += reconstructed[row, column]
            counts[row + column] += 1
    return result / np.maximum(counts, 1)
