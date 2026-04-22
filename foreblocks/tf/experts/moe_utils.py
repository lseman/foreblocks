from __future__ import annotations

import inspect

import torch
import torch.nn as nn

try:
    from ..compute.kernels import grouped_mlp_swiglu  # type: ignore
except Exception:
    grouped_mlp_swiglu = None  # type: ignore


def maybe_compile(
    mod: nn.Module, enabled: bool = True, dynamic: bool = True
) -> nn.Module:
    """
    Wrap with torch.compile if available and enabled, else return the module.
    Kept tiny/defensive to avoid surprising failures.
    """
    if not enabled:
        return mod
    try:
        return torch.compile(mod, dynamic=dynamic)
    except Exception:
        return mod


def autocast_bf16_enabled(device_type: str) -> bool:
    """
    Decide whether to enable bf16 autocast for the given device_type.
    For now: only CUDA; CPU autocast in bf16 is still not universally stable.
    """
    if device_type != "cuda":
        return False
    return torch.cuda.is_available()


def supports_grouped_prepacked() -> bool:
    """
    Detect once whether grouped_mlp_swiglu supports prepacked expert args.
    Avoids exception-driven probing on every forward.
    """
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
    """
    Returns (top_p, top_i) for K experts per token.
    - Take topk on raw logits (no full softmax over E).
    - Normalize only within the chosen K.

    logits: [T, E]
    top_p:  [T, K]  (probabilities within top-k)
    top_i:  [T, K]  (expert indices)
    """
    if k == 1:
        top_v, top_i = torch.max(logits, dim=-1, keepdim=True)
        top_p = torch.ones_like(top_v)
        return top_p, top_i
    top_v, top_i = torch.topk(logits, k, dim=-1, sorted=False)
    m = top_v.max(dim=-1, keepdim=True).values
    expv = torch.exp(top_v - m)
    top_p = expv / (expv.sum(dim=-1, keepdim=True) + 1e-12)
    return top_p, top_i


def eager_topk_routing(logits: torch.Tensor, k: int):
    """
    Non-scripted fallback for environments where the TorchScript/fused routing
    path fails at runtime (for example missing NVRTC builtins on CUDA).
    """
    if k == 1:
        top_v, top_i = torch.max(logits, dim=-1, keepdim=True)
        top_p = torch.ones_like(top_v)
        return top_p, top_i
    top_v, top_i = torch.topk(logits, k, dim=-1, sorted=False)
    m = top_v.max(dim=-1, keepdim=True).values
    expv = torch.exp(top_v - m)
    top_p = expv / (expv.sum(dim=-1, keepdim=True) + 1e-12)
    return top_p, top_i


def should_fallback_router_topk(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "nvrtc" in msg
        or "libnvrtc-builtins" in msg
        or "torchscript interpreter" in msg
    )


__all__ = [
    "autocast_bf16_enabled",
    "eager_topk_routing",
    "maybe_compile",
    "optimized_topk_routing",
    "should_fallback_router_topk",
    "supports_grouped_prepacked",
]
