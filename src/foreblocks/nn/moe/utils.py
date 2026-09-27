"""foreblocks.nn.moe.utils.

MoE utilities: TorchScript-compiled routing, compile wrappers, and dtype helpers.

Provides optimized top-K routing functions (TorchScript and eager fallbacks),
safe torch.compile wrapping, bf16 autocast detection, and grouped kernel capability
detection. Use for low-level MoE infrastructure in performance-critical paths.

Core API:
- maybe_compile: safe torch.compile wrapper with graceful fallback
- optimized_topk_routing: TorchScript-compiled top-K routing
- eager_topk_routing: non-scripted fallback routing
- autocast_bf16_enabled: bf16 autocast availability check
- supports_grouped_prepacked: grouped kernel prepacked-args detection

"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

try:
    from foreblocks.kernels.activations.swiglu import (
        grouped_mlp_swiglu,  # type: ignore
    )
except Exception:
    grouped_mlp_swiglu = None  # type: ignore


def maybe_compile(
    mod: nn.Module, enabled: bool = True, dynamic: bool = True
) -> nn.Module:
    if not enabled:
        return mod
    try:
        return torch.compile(mod, dynamic=dynamic)
    except Exception:
        return mod


def autocast_bf16_enabled(device_type: str) -> bool:
    if device_type != "cuda":
        return False
    return torch.cuda.is_available()


def supports_grouped_prepacked() -> bool:
    if grouped_mlp_swiglu is None:
        return False
    try:
        sig = inspect.signature(grouped_mlp_swiglu)
        return ("B12_cat_prepacked" in sig.parameters) and (
            "B3_cat_prepacked" in sig.parameters
        )
    except Exception:
        return False


def eager_topk_routing(
    logits: torch.Tensor, k: int, selection_logits: torch.Tensor | None = None
):
    scores = logits if selection_logits is None else selection_logits
    top_i = torch.topk(scores, k, dim=-1, sorted=False).indices
    if k == 1:
        # Keep the selected probability's task gradient (Switch-style gating).
        top_p = torch.softmax(logits.float(), dim=-1).gather(-1, top_i)
    else:
        top_p = torch.softmax(logits.float().gather(-1, top_i), dim=-1)
    return top_p.to(logits.dtype), top_i


optimized_topk_routing = torch.compile(eager_topk_routing)


def should_fallback_router_topk(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "nvrtc" in msg or "libnvrtc-builtins" in msg or "torchscript interpreter" in msg
    )


__all__ = [
    "autocast_bf16_enabled",
    "eager_topk_routing",
    "maybe_compile",
    "optimized_topk_routing",
    "should_fallback_router_topk",
    "supports_grouped_prepacked",
]
