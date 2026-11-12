"""foreblocks.processing.auto_filter.filters.ssa.

SSA — Singular Spectrum Analysis (pure NumPy).

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foreblocks.processing.auto_filter.filters.utils import _as_series
from foreblocks.processing.auto_filter.registry import register_filter
from foreblocks.processing.filters._ssa import _ssa_reconstruct


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
