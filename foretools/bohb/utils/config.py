from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _reflect_into_bounds(x: float, lo: float, hi: float) -> float:
    """
    Reflect x into [lo, hi]. This reduces boundary bias compared to raw clamp.
    """
    if lo >= hi:
        return float(lo)

    width = hi - lo
    # Map into [0, 2*width) then reflect
    y = (x - lo) % (2.0 * width)
    if y > width:
        y = 2.0 * width - y
    return float(lo + y)


def _robust_scale_1d(values: np.ndarray) -> float:
    """
    Robust scale estimate: min(std, IQR/1.349). Falls back safely.
    """
    v = np.asarray(values, dtype=float)
    if v.size <= 1:
        return 1.0
    std = float(np.std(v))
    q25, q75 = np.percentile(v, [25, 75])
    iqr = float(q75 - q25)
    robust = iqr / 1.349 if iqr > 0 else std
    s = min(std, robust) if (std > 0 and robust > 0) else max(std, robust, 1e-12)
    return float(max(s, 1e-12))


def _canonical_config_key(config: Dict[str, Any], float_round: int = 12) -> str:
    """
    Stable serialization for config hashing. Floats are rounded to reduce tiny drift.
    """
    items: Dict[str, Any] = {}
    for k in sorted(config.keys()):
        v = config[k]
        if isinstance(v, float):
            v = round(v, float_round)
        items[k] = v
    return json.dumps(items, sort_keys=True, separators=(",", ":"))
