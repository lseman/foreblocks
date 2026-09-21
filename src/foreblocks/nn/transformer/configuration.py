"""Resolve constructor arguments before constructing transformer modules.

The public constructors accept a config or legacy flat arguments. This module
owns precedence, legacy defaults, and model presets; module constructors only
consume the resulting config. No caller-owned config or mapping is mutated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from foreblocks.nn.transformer.config import TransformerConfig


def _resolve_config(
    primary: int | TransformerConfig | None,
    config: TransformerConfig | None,
    *,
    primary_name: str,
    overrides: Mapping[str, Any],
    legacy_defaults: Mapping[str, Any] | None = None,
) -> TransformerConfig:
    values = dict(overrides)
    if isinstance(primary, TransformerConfig):
        if config is not None:
            raise ValueError("pass TransformerConfig once, positionally or as config")
        config = primary
    elif primary is not None:
        values[primary_name] = primary

    if config is not None:
        return config.with_overrides(**values) if values else config
    return TransformerConfig.from_legacy_dict({**(legacy_defaults or {}), **values})


def resolve_model_config(
    input_size: int | TransformerConfig | None,
    config: TransformerConfig | None,
    *,
    role: Literal["encoder", "decoder"],
    overrides: Mapping[str, Any],
) -> TransformerConfig:
    """Apply explicit overrides, model presets, then role compatibility checks."""
    config = _resolve_config(
        input_size, config, primary_name="input_size", overrides=overrides
    )
    preset = {}
    if config.model_type == "informer-like":
        preset["use_time_encoding"] = True
        if role == "decoder":
            preset["informer_like"] = True
    if role == "encoder" and config.ct_patchtst:
        preset["patch_encoder"] = False
    if preset:
        config = config.with_overrides(**preset)
    config.validate_compatibility(role=role)
    return config


def resolve_layer_config(
    d_model: int | TransformerConfig | None,
    nhead: int | None,
    config: TransformerConfig | None,
    *,
    role: Literal["encoder", "decoder"],
    overrides: Mapping[str, Any],
) -> TransformerConfig:
    """Retain standalone layer defaults without leaking them into stack configs."""
    if config is None and not isinstance(d_model, TransformerConfig):
        if d_model is None or nhead is None:
            raise TypeError("d_model and nhead are required without config")
    values = dict(overrides)
    if nhead is not None:
        values["nhead"] = nhead
    defaults: dict[str, Any] = {"dim_feedforward": 2048}
    if role == "encoder":
        if values.get("attention") is None:
            defaults["freq_modes"] = 16
    else:
        defaults["informer_like"] = False
    return _resolve_config(
        d_model,
        config,
        primary_name="d_model",
        overrides=values,
        legacy_defaults=defaults,
    )


__all__ = ["resolve_layer_config", "resolve_model_config"]
