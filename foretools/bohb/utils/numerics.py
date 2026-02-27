from __future__ import annotations

import math
from typing import Optional

import numpy as np


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-12 else default


def safe_log(x: float, floor: float = -30.0) -> float:
    if x <= 0:
        return float(floor)
    return math.log(max(x, 1e-300))


def safe_normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    if s > 1e-12:
        return w / s
    if w.size == 0:
        return w
    return np.full_like(w, 1.0 / float(len(w)))


def make_positive_definite(cov: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return cov
    try:
        eigs = np.linalg.eigvals(cov)
        minv = float(np.min(eigs))
    except Exception:
        return cov
    if minv < min_eig:
        cov = cov + np.eye(cov.shape[0]) * (min_eig - minv + 1e-10)
    return cov


def inv_yeojohnson(x: float | np.ndarray, lmbda: float) -> float | np.ndarray:
    """
    Inverse Yeo-Johnson transform. Supports scalars and numpy arrays.
    """
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    
    # Case 1: x >= 0
    mask_pos = x >= 0
    if np.any(mask_pos):
        x_pos = x[mask_pos]
        if abs(lmbda) < 1e-9:
             out[mask_pos] = np.exp(x_pos) - 1.0
        else:
             val = x_pos * lmbda + 1.0
             # Avoid invalid pow
             val = np.maximum(0.0, val)
             out[mask_pos] = np.power(val, 1.0 / lmbda) - 1.0
             
    # Case 2: x < 0
    mask_neg = ~mask_pos
    if np.any(mask_neg):
        x_neg = x[mask_neg]
        if abs(lmbda - 2.0) < 1e-9:
             out[mask_neg] = 1.0 - np.exp(-x_neg)
        else:
             val = 1.0 - x_neg * (2.0 - lmbda)
             val = np.maximum(0.0, val)
             out[mask_neg] = 1.0 - np.power(val, 1.0 / (2.0 - lmbda))
             
    if out.ndim == 0:
        return float(out)
    return out


def yeojohnson_forward(x: float | np.ndarray, lmbda: float) -> float | np.ndarray:
    """
    Forward Yeo-Johnson transform. Supports scalars and numpy arrays.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    
    # Case 1: x >= 0
    mask_pos = x >= 0
    if np.any(mask_pos):
        x_pos = x[mask_pos]
        if abs(lmbda) < 1e-9:
            out[mask_pos] = np.log1p(x_pos)
        else:
            out[mask_pos] = (np.power(x_pos + 1.0, lmbda) - 1.0) / lmbda
            
    # Case 2: x < 0
    mask_neg = ~mask_pos
    if np.any(mask_neg):
        x_neg = x[mask_neg]
        if abs(lmbda - 2.0) < 1e-9:
            out[mask_neg] = -np.log1p(-x_neg)
        else:
            out[mask_neg] = -((np.power(-x_neg + 1.0, 2.0 - lmbda) - 1.0) / (2.0 - lmbda))
            
    if out.ndim == 0:
        return float(out)
    return out


def yeojohnson_log_jacobian(x: float | np.ndarray, lmbda: float) -> float | np.ndarray:
    """
    Log of the Jacobian of the Yeo-Johnson transform |dy/dx|.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    
    mask_pos = x >= 0
    if np.any(mask_pos):
        out[mask_pos] = (lmbda - 1.0) * np.log1p(x[mask_pos])
        
    mask_neg = ~mask_pos
    if np.any(mask_neg):
        out[mask_neg] = (1.0 - lmbda) * np.log1p(-x[mask_neg])

    if out.ndim == 0:
        return float(out)
    return out
