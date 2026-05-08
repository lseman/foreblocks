from __future__ import annotations

import numpy as np


def _energy(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
