"""foreblocks.ts_handler.auto_filter.filters.ssa.

SSA — Singular Spectrum Analysis (pure NumPy).

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foreblocks.ts_handler.auto_filter.filters.utils import _as_series
from foreblocks.ts_handler.auto_filter.registry import register_filter


def _ssa_reconstruct(x: np.ndarray, window: int, n_components: int) -> np.ndarray:
    N = len(x)
    L = window
    K = N - L + 1

    # Trajectory (Hankel) matrix  (L × K)
    X_emb = np.stack([x[i : i + L] for i in range(K)], axis=1)

    U, s, Vt = np.linalg.svd(X_emb, full_matrices=False)

    n = min(n_components, len(s))
    X_rec = sum(s[i] * np.outer(U[:, i], Vt[i]) for i in range(n))

    # Anti-diagonal averaging (Hankelization → 1-D)
    result = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            result[i + j] += X_rec[i, j]
            counts[i + j] += 1
    return result / np.maximum(counts, 1)


@register_filter("SSA")
def ssa_filter(
    ts: pd.Series, window: int | None = None, n_components: int = 3
) -> pd.Series:
    x = ts.values.astype(float)
    N = len(x)
    if N < 16:
        return ts.copy()
    if window is None:
        window = int(np.clip(N // 4, 8, 200))
    window = int(np.clip(window, 4, N // 2))
    n_components = max(1, int(n_components))
    reconstructed = _ssa_reconstruct(x, window=window, n_components=n_components)
    return _as_series(reconstructed, ts.index, name="ssa")
