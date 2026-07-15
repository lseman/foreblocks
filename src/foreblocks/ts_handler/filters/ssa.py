"""foreblocks.ts_handler.filters.ssa.

Singular Spectrum Analysis (SSA) filter for time-series denoising.

"""

from __future__ import annotations

import numpy as np

from foreblocks.ts_handler.filters.utils import _as_2d, _nan_interp_1d, _odd_at_least


def ssa_filter(
    data: np.ndarray,
    window_length: int | None = None,
    n_components: int = 2,
    *,
    fill_nans_for_filter: bool = True,
) -> np.ndarray:
    """
    SOTA Singular Spectrum Analysis (SSA) filter.
    Decomposes the time series into a trajectory matrix, performs SVD,
    and reconstructs the signal using only the top `n_components`
    (assumed to be trend + dominant oscillations, discarding noise).
    """
    x = _as_2d(data)
    T, F = x.shape
    out = np.full_like(x, np.nan)

    # Heuristic for embedding window length if not provided
    if window_length is None:
        window_length = min(T // 2, 50)

    L = int(max(2, window_length))
    K = T - L + 1

    if K < 1:
        # Time series too short for the window
        return x.copy()

    for i in range(F):
        col = x[:, i]
        mask = ~np.isnan(col)

        if mask.sum() < L + 2:
            out[mask, i] = col[mask]
            continue

        col_work = col.copy()
        if fill_nans_for_filter and np.isnan(col_work).any():
            col_work = _nan_interp_1d(col_work)

        # 1. Embedding
        X = np.column_stack([col_work[j : j + L] for j in range(K)])

        # 2. SVD
        try:
            U, s, Vh = np.linalg.svd(X, full_matrices=False)

            # 3. Grouping (select top components)
            d = min(n_components, len(s))
            X_elem = np.zeros_like(X)
            for j in range(d):
                X_elem += s[j] * np.outer(U[:, j], Vh[j, :])

            # 4. Diagonal Averaging (Reconstruction)
            rec = np.zeros(T)
            counts = np.zeros(T)

            for c in range(K):
                for r in range(L):
                    rec[r + c] += X_elem[r, c]
                    counts[r + c] += 1

            rec = rec / np.maximum(counts, 1)

            out[mask, i] = rec[mask]
        except Exception:
            out[mask, i] = col[mask]

    return out
