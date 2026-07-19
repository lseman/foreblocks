"""Typed, backward-compatible runtime state for incremental decoding."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch

from foreblocks.modules.attention.cache.base import KVCacheProtocol


class AttentionCacheState(dict[str, Any]):
    """State owned by one attention sublayer.

    It remains a ``dict`` so existing attention backends can migrate one at a
    time, while callers gain canonical accessors for cache and position state.
    """

    @classmethod
    def from_legacy(cls, value: Mapping[str, Any] | None) -> "AttentionCacheState":
        return value if isinstance(value, cls) else cls(value or {})

    @property
    def cache(self) -> KVCacheProtocol | None:
        for key in ("static_cache", "paged_cache"):
            value = self.get(key)
            if isinstance(value, KVCacheProtocol):
                return value
        return None

    @property
    def cache_position(self) -> torch.Tensor | None:
        value = self.get("cache_position")
        return value if isinstance(value, torch.Tensor) else None

    @cache_position.setter
    def cache_position(self, value: torch.Tensor | None) -> None:
        if value is None:
            self.pop("cache_position", None)
        else:
            self["cache_position"] = value

    @property
    def cache_update_mask(self) -> torch.Tensor | None:
        value = self.get("cache_update_mask")
        return value if isinstance(value, torch.Tensor) else None

    @cache_update_mask.setter
    def cache_update_mask(self, value: torch.Tensor | None) -> None:
        if value is None:
            self.pop("cache_update_mask", None)
        else:
            self["cache_update_mask"] = value


class DecoderLayerState(dict[str, AttentionCacheState]):
    @classmethod
    def from_legacy(cls, value: Mapping[str, Any] | None) -> "DecoderLayerState":
        value = value or {}
        return cls(
            self_attn=AttentionCacheState.from_legacy(value.get("self_attn")),
            cross_attn=AttentionCacheState.from_legacy(value.get("cross_attn")),
        )

    @property
    def self_attention(self) -> AttentionCacheState:
        return self["self_attn"]

    @property
    def cross_attention(self) -> AttentionCacheState:
        return self["cross_attn"]


class DecoderState(dict[str, Any]):
    @classmethod
    def from_legacy(
        cls,
        value: Mapping[str, Any] | None,
        *,
        num_layers: int,
    ) -> "DecoderState":
        raw = dict(value or {})
        raw_layers: Iterable[Mapping[str, Any] | None] = raw.get("layers") or ()
        layers = [DecoderLayerState.from_legacy(item) for item in raw_layers]
        if layers and len(layers) != num_layers:
            raise ValueError(
                f"decoder state has {len(layers)} layers; expected {num_layers}"
            )
        if not layers:
            layers = [DecoderLayerState.from_legacy(None) for _ in range(num_layers)]
        raw["layers"] = layers
        return cls(raw)

    @property
    def layers(self) -> list[DecoderLayerState]:
        return self["layers"]

    @property
    def decoded_length(self) -> int:
        return int(self.get("_decoded_len", 0))

    @decoded_length.setter
    def decoded_length(self, value: int) -> None:
        self["_decoded_len"] = int(value)


__all__ = ["AttentionCacheState", "DecoderLayerState", "DecoderState"]
