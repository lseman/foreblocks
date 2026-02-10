# foreblocks/fusions.py
# -*- coding: utf-8 -*-
"""
Lightweight fused ops:
- Dropout → Residual Add
- Dropout → Residual Add → Norm
- Dropout → GateSkip(residual, update) → Norm

Goals:
  - Autocast-safe
  - Compatible with NormWrapper (uses `.norm(x)`) and plain norms (callable)
  - Plug-and-play with ResidualGate + gateskip_apply
  - Keep the hot path small and JIT-friendly where it matters
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "get_dropout_p",
    "fused_dropout_add",
    "fused_dropout_add_norm",
    "fused_dropout_gateskip_norm",
    # Convenience, module-based wrappers:
    "fused_dropout_add_from_layer",
    "fused_dropout_add_norm_from_layers",
    "fused_dropout_gateskip_norm_from_layers",
]

# Optional GateSkip import
try:
    from .gateskip import ResidualGate, gateskip_apply  # type: ignore

    _HAS_GATESKIP = True
except Exception:  # pragma: no cover
    ResidualGate = None  # type: ignore[assignment]
    _HAS_GATESKIP = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_dropout_p(layer: Optional[nn.Module]) -> float:
    """
    Safely extract p from a Dropout-like module or return 0.0 for Identity/None.

    This is robust to passing nn.Identity, None, or any module without `.p`.
    """
    if layer is None:
        return 0.0
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
        # NormWrapper path
        return norm_fn(x)
    # Plain norm module
    return norm_layer(x)


# ---------------------------------------------------------------------------
# Fused primitives (tensor + scalar p)
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
        residual: [..., D]
        update:   same shape as residual
        p:        dropout probability
        training: apply dropout only if True

    Returns:
        out = residual + dropout(update)
    """
    # Fast-path for p == 0.0 or eval mode avoids F.dropout call.
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

    Use this where you currently do:
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
        out: fused output tensor (same shape as residual)
        skip_mask: optional boolean mask from gateskip_apply (for KV cache copy)

    Notes:
        - If use_gateskip=False, this reduces to:
              out = residual + dropout(update); out = norm(out)
        - If GateSkip is requested but unavailable, raises a RuntimeError.
    """
    if p > 0.0 and training:
        update = F.dropout(update, p=p, training=True)

    if use_gateskip:
        if not _HAS_GATESKIP or gate is None:
            raise RuntimeError(
                "GateSkip requested but ResidualGate/gateskip_apply not available."
            )
        aux = aux_l2_terms if aux_l2_terms is not None else []
        out, skip_mask = gateskip_apply(
            use_gateskip, residual, update, gate, gate_budget, aux, gate_lambda
        )
    else:
        out = residual + update
        skip_mask = None

    out = _apply_norm(norm_layer, out)
    return out, skip_mask


# ---------------------------------------------------------------------------
# Convenience wrappers (module-based)
# ---------------------------------------------------------------------------

def fused_dropout_add_from_layer(
    residual: torch.Tensor,
    update: torch.Tensor,
    dropout: Optional[nn.Module],
    training: bool,
) -> torch.Tensor:
    """
    Convenience wrapper:

        fused_dropout_add(residual, update, p=get_dropout_p(dropout), training=training)
    """
    p = get_dropout_p(dropout)
    return fused_dropout_add(residual, update, p=p, training=training)


def fused_dropout_add_norm_from_layers(
    residual: torch.Tensor,
    update: torch.Tensor,
    dropout: Optional[nn.Module],
    norm_layer: Optional[nn.Module],
    training: bool,
) -> torch.Tensor:
    """
    Convenience wrapper:

        o = dropout(update)
        residual = residual + o
        residual = norm(residual)

    but in one call, using the given dropout/norm modules directly.
    """
    p = get_dropout_p(dropout)
    return fused_dropout_add_norm(
        residual=residual,
        update=update,
        norm_layer=norm_layer,
        p=p,
        training=training,
    )


def fused_dropout_gateskip_norm_from_layers(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: Optional[nn.Module],
    use_gateskip: bool,
    gate_budget: Optional[float],
    aux_l2_terms: Optional[List[torch.Tensor]],
    gate_lambda: float,
    dropout: Optional[nn.Module],
    norm_layer: Optional[nn.Module],
    training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convenience wrapper for the full GateSkip branch:

        - Reads `p` from a Dropout-like module
        - Applies dropout → GateSkip (if enabled) → Norm
    """
    p = get_dropout_p(dropout)
    return fused_dropout_gateskip_norm(
        residual=residual,
        update=update,
        gate=gate,
        use_gateskip=use_gateskip,
        gate_budget=gate_budget,
        aux_l2_terms=aux_l2_terms,
        gate_lambda=gate_lambda,
        norm_layer=norm_layer,
        p=p,
        training=training,
    )
