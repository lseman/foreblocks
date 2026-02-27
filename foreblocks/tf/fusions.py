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

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    TRITON_AVAILABLE = False

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
    from .skip.gateskip import ResidualGate, gateskip_apply  # type: ignore

    _HAS_GATESKIP = True
except Exception:  # pragma: no cover
    ResidualGate = None  # type: ignore[assignment]
    _HAS_GATESKIP = False

# Optional Triton norm backend import
try:
    from .norms.layer_norm import AdaptiveLayerNorm, FastLayerNorm  # type: ignore
    from .norms.rms_norm import AdaptiveRMSNorm, RMSNorm  # type: ignore
    from .norms.triton_backend import (
        TRITON_AVAILABLE as _NORM_TRITON_AVAILABLE,
    )
    from .norms.triton_backend import (  # type: ignore
        LayerNormTritonFunction,
        RMSNormTritonFunction,
        triton_fused_rmsnorm_scale_bias,
        triton_scale_bias,
    )
    from .norms.triton_backend import (
        _should_use_triton as _norm_should_use_triton,
    )

    _HAS_TRITON_NORM_BACKEND = True
except Exception:  # pragma: no cover
    FastLayerNorm = None  # type: ignore[assignment]
    AdaptiveLayerNorm = None  # type: ignore[assignment]
    RMSNorm = None  # type: ignore[assignment]
    AdaptiveRMSNorm = None  # type: ignore[assignment]
    LayerNormTritonFunction = None  # type: ignore[assignment]
    RMSNormTritonFunction = None  # type: ignore[assignment]
    _NORM_TRITON_AVAILABLE = False
    _HAS_TRITON_NORM_BACKEND = False


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


def _resolve_norm_module(norm_layer: Optional[nn.Module]) -> Optional[nn.Module]:
    if norm_layer is None:
        return None
    norm_fn = getattr(norm_layer, "norm", None)
    if callable(norm_fn) and isinstance(norm_fn, nn.Module):
        return norm_fn
    return norm_layer


def _apply_norm_triton_if_possible(
    norm_layer: Optional[nn.Module], x: torch.Tensor
) -> torch.Tensor:
    """
    Dedicated Triton norm fast path for post-norm branches.
    Falls back to module forward when unsupported.
    """
    mod = _resolve_norm_module(norm_layer)
    if mod is None:
        return x
    if not _HAS_TRITON_NORM_BACKEND:
        return _apply_norm(norm_layer, x)
    if not _NORM_TRITON_AVAILABLE:
        return _apply_norm(norm_layer, x)
    if not x.is_cuda:
        return _apply_norm(norm_layer, x)

    # FastLayerNorm: uses custom autograd Triton function.
    if FastLayerNorm is not None and isinstance(mod, FastLayerNorm):
        if (
            getattr(mod, "elementwise_affine", False)
            and getattr(mod, "weight", None) is not None
            and getattr(mod, "bias", None) is not None
            and _norm_should_use_triton(x, min_numel=1024)
        ):
            return LayerNormTritonFunction.apply(x, mod.weight, mod.bias, mod.eps)
        return mod(x)

    # RMSNorm: uses custom autograd Triton function.
    if RMSNorm is not None and isinstance(mod, RMSNorm):
        if (
            getattr(mod, "elementwise_affine", False)
            and getattr(mod, "weight", None) is not None
            and _norm_should_use_triton(x, min_numel=1024)
        ):
            return RMSNormTritonFunction.apply(x, mod.weight, mod.eps)
        return mod(x)

    # AdaptiveLayerNorm: Triton LN + Triton scale/bias (inference/no-grad only).
    if AdaptiveLayerNorm is not None and isinstance(mod, AdaptiveLayerNorm):
        if not torch.is_grad_enabled() and _norm_should_use_triton(x, min_numel=2048):
            base = mod.norm
            y = LayerNormTritonFunction.apply(x, base.weight, base.bias, base.eps)
            y = triton_scale_bias(y, mod.alpha, mod.beta)
            return mod.dropout(y)
        return mod(x)

    # AdaptiveRMSNorm: Triton fused path only in non-global + inference/no-grad mode.
    if AdaptiveRMSNorm is not None and isinstance(mod, AdaptiveRMSNorm):
        if (
            (not getattr(mod, "global_rms", False))
            and not torch.is_grad_enabled()
            and _norm_should_use_triton(x, min_numel=2048)
        ):
            y = triton_fused_rmsnorm_scale_bias(
                x=x,
                rms_weight=mod.weight,
                alpha=mod.alpha,
                beta=mod.beta,
                eps=mod.eps,
            )
            return mod.dropout(y)
        return mod(x)

    # Unknown norm type -> existing behavior.
    return _apply_norm(norm_layer, x)


def _can_use_triton_add(
    residual: torch.Tensor,
    update: torch.Tensor,
    *,
    p: float,
    training: bool,
) -> bool:
    # Keep training/dropout path on PyTorch to preserve RNG semantics.
    if (p > 0.0) and training:
        return False
    if not TRITON_AVAILABLE:
        return False
    if (not residual.is_cuda) or (not update.is_cuda):
        return False
    if residual.shape != update.shape:
        return False
    if residual.dtype != update.dtype:
        return False
    if residual.numel() == 0:
        return False
    return True


if TRITON_AVAILABLE:

    @triton.jit
    def _add_kernel(
        residual_ptr,
        update_ptr,
        out_ptr,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        a = tl.load(residual_ptr + offs, mask=mask, other=0.0)
        b = tl.load(update_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, a + b, mask=mask)


def _triton_add(residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(residual)
    r = residual.contiguous().view(-1)
    u = update.contiguous().view(-1)
    o = out.view(-1)
    n = o.numel()
    block = 1024
    grid = (triton.cdiv(n, block),)
    _add_kernel[grid](r, u, o, n, BLOCK=block)
    return out


# ---------------------------------------------------------------------------
# Fused primitives (tensor + scalar p)
# ---------------------------------------------------------------------------


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
    if _can_use_triton_add(residual, update, p=p, training=training):
        return _triton_add(residual, update)
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
    out = _apply_norm_triton_if_possible(norm_layer, out)
    return out


def fused_dropout_gateskip_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: Optional[nn.Module],  # expected: ResidualGate
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
        if _can_use_triton_add(residual, update, p=p, training=training):
            out = _triton_add(residual, update)
        else:
            out = residual + update
        skip_mask = None

    out = _apply_norm_triton_if_possible(norm_layer, out)
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
