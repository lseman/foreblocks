# foreblocks/fusions.py
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
    from .skip.gateskip import ResidualGate  # type: ignore
    from .skip.gateskip import gateskip_apply

    _HAS_GATESKIP = True
except Exception:  # pragma: no cover
    ResidualGate = None  # type: ignore[assignment]
    _HAS_GATESKIP = False

# Optional Triton norm backend import
try:
    from .norms.layer_norm import AdaptiveLayerNorm  # type: ignore
    from .norms.layer_norm import FastLayerNorm
    from .norms.rms_norm import AdaptiveRMSNorm  # type: ignore
    from .norms.rms_norm import RMSNorm
    from .norms.triton_backend import (
        TRITON_AVAILABLE as _NORM_TRITON_AVAILABLE,  # type: ignore
    )
    from .norms.triton_backend import FusedAddRMSNormFunction
    from .norms.triton_backend import LayerNormTritonFunction
    from .norms.triton_backend import RMSNormTritonFunction
    from .norms.triton_backend import _should_use_triton as _norm_should_use_triton
    from .norms.triton_backend import triton_fused_rmsnorm_scale_bias
    from .norms.triton_backend import triton_scale_bias

    _HAS_TRITON_NORM_BACKEND = True
except Exception:  # pragma: no cover
    FastLayerNorm = None  # type: ignore[assignment]
    AdaptiveLayerNorm = None  # type: ignore[assignment]
    RMSNorm = None  # type: ignore[assignment]
    AdaptiveRMSNorm = None  # type: ignore[assignment]
    LayerNormTritonFunction = None  # type: ignore[assignment]
    RMSNormTritonFunction = None  # type: ignore[assignment]
    FusedAddRMSNormFunction = None  # type: ignore[assignment]
    _NORM_TRITON_AVAILABLE = False
    _HAS_TRITON_NORM_BACKEND = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_dropout_p(layer: nn.Module | None) -> float:
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


def _apply_norm(norm_layer: nn.Module | None, x: torch.Tensor) -> torch.Tensor:
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


def _resolve_norm_module(norm_layer: nn.Module | None) -> nn.Module | None:
    if norm_layer is None:
        return None
    norm_fn = getattr(norm_layer, "norm", None)
    if callable(norm_fn) and isinstance(norm_fn, nn.Module):
        return norm_fn
    return norm_layer


def _apply_norm_triton_if_possible(
    norm_layer: nn.Module | None, x: torch.Tensor
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


def _can_fused_add_rmsnorm(
    residual: torch.Tensor,
    update: torch.Tensor,
    norm_layer: nn.Module | None,
) -> bool:
    if not (_HAS_TRITON_NORM_BACKEND and _NORM_TRITON_AVAILABLE):
        return False
    if not (residual.is_cuda and update.is_cuda):
        return False
    if residual.shape != update.shape or residual.dtype != update.dtype:
        return False
    if residual.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if torch.jit.is_scripting():
        return False
    if residual.shape[-1] > 2048:
        return False
    mod = _resolve_norm_module(norm_layer)
    if mod is None:
        return False
    if RMSNorm is None or not isinstance(mod, RMSNorm):
        return False
    if not getattr(mod, "elementwise_affine", False):
        return False
    if getattr(mod, "weight", None) is None:
        return False
    return True


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
    norm_layer: nn.Module | None,
    p: float,
    training: bool,
) -> torch.Tensor:
    """
    Fuse: Dropout(update) → Residual add → Norm  (post-norm style)

    Use this where you currently do:
        o = dropout(o); src = src + o; src = norm(src)

    Works with NormWrapper or plain norm modules.
    Fast path: single-kernel fused add+RMSNorm when p==0, CUDA, D<=2048.
    """
    if (p == 0.0 or not training) and _can_fused_add_rmsnorm(residual, update, norm_layer):
        mod = _resolve_norm_module(norm_layer)
        return FusedAddRMSNormFunction.apply(residual, update, mod.weight, mod.eps)

    out = fused_dropout_add(residual, update, p=p, training=training)
    out = _apply_norm_triton_if_possible(norm_layer, out)
    return out


def fused_dropout_gateskip_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: nn.Module | None,  # expected: ResidualGate
    use_gateskip: bool,
    gate_budget: float | None,
    aux_l2_terms: list[torch.Tensor] | None,
    gate_lambda: float,
    norm_layer: nn.Module | None,
    p: float,
    training: bool,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
            use_gateskip,
            residual,
            update,
            gate,
            gate_budget,
            aux,
            gate_lambda,
            active_mask=active_mask,
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
    dropout: nn.Module | None,
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
    dropout: nn.Module | None,
    norm_layer: nn.Module | None,
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
    gate: nn.Module | None,
    use_gateskip: bool,
    gate_budget: float | None,
    aux_l2_terms: list[torch.Tensor] | None,
    gate_lambda: float,
    dropout: nn.Module | None,
    norm_layer: nn.Module | None,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
