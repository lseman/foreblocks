# foreblocks/fusions.py
# -*- coding: utf-8 -*-
"""
Lightweight fused ops:
- Dropout → Residual Add
- Dropout → Residual Add → Norm
- Dropout → GateSkip(residual, update) → Norm

Designed to be:
  - Autocast-safe
  - Compatible with NormWrapper (uses `.norm(x)`) and plain norms (callable)
  - Plug-and-play with your ResidualGate + gateskip_apply
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "fused_dropout_add",
    "fused_dropout_add_norm",
    "fused_dropout_gateskip_norm",
    "get_dropout_p",
]

# Optional GateSkip import
try:
    from .gateskip import ResidualGate, gateskip_apply  # type: ignore
    _HAS_GATESKIP = True
except Exception:
    ResidualGate = None  # type: ignore[assignment]
    _HAS_GATESKIP = False

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_dropout_p(layer: Optional[nn.Module]) -> float:
    """
    Safely extract p from a Dropout-like module or return 0.0 for Identity/None.
    """
    if layer is None:
        return 0.0
    # nn.Dropout has attribute .p; Identity doesn't.
    p = getattr(layer, "p", None)
    try:
        return float(p) if p is not None else 0.0
    except Exception:
        return 0.0


def _apply_norm(norm_layer: Optional[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """
    Supports either:
      - NormWrapper (has .norm(x))
      - Plain nn.Module like LayerNorm/RMSNorm (callable)
      - None (no-op)
    """
    if norm_layer is None:
        return x
    norm_fn = getattr(norm_layer, "norm", None)
    if callable(norm_fn):
        return norm_fn(x)  # NormWrapper path
    return norm_layer(x)   # Plain norm module


# ---------------------------------------------------------------------------
# Fused primitives
# ---------------------------------------------------------------------------

@torch.jit.script_if_tracing
def fused_dropout_add(
    residual: torch.Tensor,
    update: torch.Tensor,
    p: float,
    training: bool,
) -> torch.Tensor:
    """
    Fuse: Dropout(update) + Residual add

    Args:
        residual: [B, T, D] (or compatible shape)
        update:   same shape as residual
        p:        dropout probability
        training: apply dropout only if True

    Returns:
        out = residual + dropout(update)
    """
    if p > 0.0 and training:
        update = F.dropout(update, p=p, training=True)
    return residual + update


def fused_dropout_add_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    norm_layer: Optional[nn.Module],
    p: float,
    training: bool,
) -> torch.Tensor:
    """
    Fuse: Dropout(update) → Residual add → Norm  (post-norm style)

    Use this in branches where you currently do:
        o = dropout(o); src = src + o; src = norm(src)

    Works with NormWrapper or plain norm modules.
    """
    out = fused_dropout_add(residual, update, p=p, training=training)
    out = _apply_norm(norm_layer, out)
    return out


def fused_dropout_gateskip_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: Optional[nn.Module],         # expected: ResidualGate
    use_gateskip: bool,
    gate_budget: Optional[float],
    aux_l2_terms: Optional[List[torch.Tensor]],
    gate_lambda: float,
    norm_layer: Optional[nn.Module],
    p: float,
    training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Fuse: Dropout(update) → GateSkip(residual, update) → Norm

    Returns:
        out: fused output tensor
        skip_mask: optional boolean mask from gateskip_apply (for KV cache copy)

    Notes:
        - If use_gateskip=False, this reduces to: residual + dropout(update) → Norm
        - If GateSkip is requested but unavailable, raises a RuntimeError.
    """
    if p > 0.0 and training:
        update = F.dropout(update, p=p, training=True)

    if use_gateskip:
        if not _HAS_GATESKIP or gate is None:
            raise RuntimeError("GateSkip requested but ResidualGate/gateskip_apply not available.")
        aux = aux_l2_terms if aux_l2_terms is not None else []
        out, skip_mask = gateskip_apply(
            use_gateskip, residual, update, gate, gate_budget, aux, gate_lambda
        )
    else:
        out = residual + update
        skip_mask = None

    out = _apply_norm(norm_layer, out)
    return out, skip_mask
