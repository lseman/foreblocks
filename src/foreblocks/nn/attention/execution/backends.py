"""Registry and adapters for concrete attention execution backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F

from foreblocks.nn.attention.preparation.masking import build_attention_mask


@dataclass(frozen=True)
class AttentionBackendSpec:
    name: str
    runner: Callable | None = None
    mask_builder: Callable | None = None
    supports_gqa: bool = True
    supports_attention_weights: bool = True
    supports_compile: bool = True
    supports_boolean_mask: bool = True
    devices: tuple[str, ...] = ("cpu", "cuda", "mps")
    available: bool = True
    unavailable_reason: str | None = None

    def supports(self, feature: str) -> bool:
        field = f"supports_{feature}"
        if not hasattr(self, field):
            raise KeyError(f"unknown attention backend capability {feature!r}")
        return bool(getattr(self, field))


class AttentionBackendRegistry:
    def __init__(self):
        self._specs: dict[str, AttentionBackendSpec] = {}

    def register(
        self,
        name: str,
        runner: Callable | None = None,
        mask_builder: Callable | None = None,
        *,
        replace: bool = False,
        **capabilities,
    ) -> None:
        key = str(name).lower()
        if key in self._specs and not replace:
            raise ValueError(f"attention backend {key!r} is already registered")
        self._specs[key] = AttentionBackendSpec(
            key, runner, mask_builder, **capabilities
        )

    def get(self, name: str) -> AttentionBackendSpec:
        key = str(name).lower()
        if key not in self._specs:
            raise KeyError(f"unknown attention backend {key!r}")
        return self._specs[key]

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def validate(
        self,
        name: str,
        *,
        device_type: str | None = None,
        require_attention_weights: bool = False,
        require_gqa: bool = False,
        compiling: bool = False,
    ) -> AttentionBackendSpec:
        spec = self.get(name)
        if not spec.available:
            raise RuntimeError(
                f"attention backend {spec.name!r} is unavailable: "
                f"{spec.unavailable_reason}"
            )
        if spec.mask_builder is None:
            raise RuntimeError(
                f"attention backend {spec.name!r} has no registered mask builder"
            )
        if device_type is not None and device_type not in spec.devices:
            raise RuntimeError(
                f"attention backend {spec.name!r} does not support {device_type}"
            )
        if require_attention_weights and not spec.supports_attention_weights:
            raise RuntimeError(
                f"attention backend {spec.name!r} does not return weights"
            )
        if require_gqa and not spec.supports_gqa:
            raise RuntimeError(f"attention backend {spec.name!r} does not support GQA")
        if compiling and not spec.supports_compile:
            raise RuntimeError(
                f"attention backend {spec.name!r} is not compile-compatible"
            )
        return spec


ATTENTION_BACKENDS = AttentionBackendRegistry()
ATTENTION_BACKENDS.register("auto", mask_builder=build_attention_mask)
ATTENTION_BACKENDS.register("eager", mask_builder=build_attention_mask)
ATTENTION_BACKENDS.register(
    "sdpa", mask_builder=build_attention_mask, supports_attention_weights=False
)


def register_attention_backend(
    name: str,
    runner: Callable,
    mask_builder: Callable | None = None,
    *,
    replace: bool = False,
    **capabilities,
) -> None:
    ATTENTION_BACKENDS.register(
        name, runner, mask_builder, replace=replace, **capabilities
    )


def _flash_attention_runner(
    q, k, v, *, attention_mask=None, dropout_p=0.0, scale=None, causal=False, **_
):
    if attention_mask is not None:
        additive = torch.full_like(
            attention_mask, float("-inf"), dtype=q.dtype
        ).masked_fill_(~attention_mask, 0.0)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive,
            dropout_p=dropout_p,
            scale=scale,
            enable_gqa=q.size(1) != k.size(1),
        )
    from flash_attn import flash_attn_func

    return flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal,
    ).transpose(1, 2)


@cache
def _compiled_flex_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention)


def _flex_attention_runner(
    q, k, v, *, attention_mask=None, scale=None, causal=False, dropout_p=0.0, **_
):
    # FlexAttention has no attention-dropout argument. Preserve training
    # semantics through SDPA rather than silently discarding dropout. Triton's
    # FlexAttention lowering also requires head dimensions of at least 16.
    if dropout_p or min(q.size(-1), v.size(-1)) < 16:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=~attention_mask if attention_mask is not None else None,
            dropout_p=dropout_p,
            is_causal=causal and attention_mask is None,
            scale=scale,
            enable_gqa=q.size(1) != k.size(1),
        )
    from torch.nn.attention.flex_attention import create_block_mask

    block_mask = None
    if attention_mask is not None:

        def mask_mod(b, h, q_idx, kv_idx):
            return ~attention_mask[b, h, q_idx, kv_idx]

        block_mask = create_block_mask(
            mask_mod, q.size(0), q.size(1), q.size(2), k.size(2), device=q.device
        )
    elif causal:

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask, None, None, q.size(2), k.size(2), device=q.device
        )

    return _compiled_flex_attention()(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=q.size(1) != k.size(1),
    )


try:
    from flash_attn import flash_attn_func as _flash_available  # noqa: F401
except (ImportError, AttributeError):
    ATTENTION_BACKENDS.register(
        "flash_attention_2",
        _flash_attention_runner,
        build_attention_mask,
        supports_attention_weights=False,
        devices=("cuda",),
        available=False,
        unavailable_reason="flash-attn with flash_attn_func is not installed",
    )
else:
    ATTENTION_BACKENDS.register(
        "flash_attention_2",
        _flash_attention_runner,
        build_attention_mask,
        supports_attention_weights=False,
        devices=("cuda",),
    )

try:
    from torch.nn.attention.flex_attention import (  # noqa: F401
        flex_attention as _flex_available,
    )
except ImportError:
    ATTENTION_BACKENDS.register(
        "flex_attention",
        _flex_attention_runner,
        build_attention_mask,
        supports_attention_weights=False,
        devices=("cuda",),
        available=False,
        unavailable_reason="torch FlexAttention is unavailable",
    )
else:
    ATTENTION_BACKENDS.register(
        "flex_attention",
        _flex_attention_runner,
        build_attention_mask,
        supports_attention_weights=False,
        devices=("cuda",),
    )


__all__ = [
    "ATTENTION_BACKENDS",
    "AttentionBackendRegistry",
    "AttentionBackendSpec",
    "register_attention_backend",
]


def _get_available_backends() -> dict[str, bool]:
    backends = {"flash": False, "xformers": False, "sdp": False, "softpick": False}

    # FlashAttention
    try:
        from flash_attn import flash_attn_func  # noqa: F401

        backends["flash"] = True
    except ImportError:
        pass

    # xFormers
    try:
        import xformers.ops  # noqa: F401

        backends["xformers"] = True
    except ImportError:
        pass

    # PyTorch scaled_dot_product_attention
    backends["sdp"] = hasattr(F, "scaled_dot_product_attention")

    # Optional third_party SoftPick
    try:
        from foreblocks.third_party.flash_softpick_attn import (  # noqa: F401
            parallel_softpick_attn,
        )

        backends["softpick"] = True
    except ImportError:
        pass

    return backends
