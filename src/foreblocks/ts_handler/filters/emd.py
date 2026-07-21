"""foreblocks.ts_handler.filters.emd.

Empirical Mode Decomposition (EMD) filter for time-series denoising.

"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from foreblocks.ts_handler.filters.utils import _as_2d

# Optional import
try:
    from PyEMD import EMD  # type: ignore
except Exception:
    EMD = None


def emd_filter(
    data: np.ndarray,
    keep_ratio: float = 0.5,
    *,
    n_jobs: int = 1,
    backend: str = "loky",
    min_imfs: int = 1,
    min_points: int = 32,
) -> np.ndarray:
    if EMD is None:
        raise ImportError("PyEMD not installed")

    x = _as_2d(data)
    T, F = x.shape
    out = x.copy()

    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))

    def _emd_one(j: int) -> tuple[int, np.ndarray]:
        col = x[:, j]
        if np.isnan(col).any() or col.size < min_points:
            return j, col

        try:
            imfs = EMD().emd(col)
            if imfs is None or len(imfs) == 0:
                return j, col

            k = int(np.floor(len(imfs) * keep_ratio))
            k = int(np.clip(k, min_imfs, len(imfs)))
            rec = np.sum(imfs[:k], axis=0)
            return j, rec
        except Exception:
            return j, col

    if n_jobs == 1:
        cols = [_emd_one(j) for j in range(F)]
    else:
        cols = Parallel(n_jobs=n_jobs, backend=backend, prefer="processes")(
            delayed(_emd_one)(j) for j in range(F)
        )

    cols.sort(key=lambda t: t[0])
    for j, colf in cols:
        out[:, j] = colf

    return out
