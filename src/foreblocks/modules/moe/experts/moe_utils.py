"""foreblocks.modules.moe.experts.moe_utils.

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
    from foreblocks.modules.moe.kernels.kernels import (
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


@torch.jit.script
def optimized_topk_routing(logits: torch.Tensor, k: int):
    if k == 1:
        top_v, top_i = torch.max(logits, dim=-1, keepdim=True)
        top_p = torch.ones_like(top_v)
        return top_p, top_i
    top_v, top_i = torch.topk(logits, k, dim=-1, sorted=False)
    # Match production MoE routers (for example Mixtral): normalize routing
    # probabilities in fp32 even when the expert block runs under autocast.
    route_dtype = top_v.dtype
    top_v = top_v.float()
    m = top_v.max(dim=-1, keepdim=True).values
    expv = torch.exp(top_v - m)
    top_p = expv / (expv.sum(dim=-1, keepdim=True) + 1e-12)
    return top_p.to(route_dtype), top_i


def eager_topk_routing(logits: torch.Tensor, k: int):
    if k == 1:
        top_v, top_i = torch.max(logits, dim=-1, keepdim=True)
        top_p = torch.ones_like(top_v)
        return top_p, top_i
    top_v, top_i = torch.topk(logits, k, dim=-1, sorted=False)
    route_dtype = top_v.dtype
    top_v = top_v.float()
    m = top_v.max(dim=-1, keepdim=True).values
    expv = torch.exp(top_v - m)
    top_p = expv / (expv.sum(dim=-1, keepdim=True) + 1e-12)
    return top_p.to(route_dtype), top_i


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
