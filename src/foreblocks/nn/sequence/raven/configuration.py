"""Local configuration for Raven sequence mixers and hybrid blocks.

Uses the same dataclass/dictionary conventions as TransformerConfig and does
not require Hugging Face model or configuration infrastructure.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any, ClassVar


@dataclass
class RavenConfig:
    model_type: ClassVar[str] = "raven"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    hidden_size: int = 2048
    gate_logit_normalizer: int | None = 8
    hidden_ratio: int | None = 4
    intermediate_size: int | None = None
    num_hidden_layers: int = 24
    num_heads: int = 4
    num_kv_heads: int | None = None
    num_slots: int | None = 64
    expand_k: float = 1
    expand_v: float = 1
    feature_map: str = "swish"
    use_output_gate: bool = False
    max_position_embeddings: int = 2048
    hidden_act: str = "swish"
    decay_type: str = "Mamba2"
    topk: int = 32
    bias_rmm: bool = False
    add_gumbel_noise: bool = True
    router_score: str = "sigmoid"
    router_type: str = "lin"
    use_rope: bool = False
    rope_theta: float = 10000.0
    elementwise_affine: bool | None = True
    norm_eps: float = 1e-6
    attn: dict | None = None
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    fuse_norm: bool = True
    fuse_swiglu: bool = True
    fuse_cross_entropy: bool = True
    fuse_linear_cross_entropy: bool = False
    use_l2warp: bool = False
    vocab_size: int = 32000
    attnres_block_size: int | None = None

    def __post_init__(self) -> None:
        if self.fuse_cross_entropy and self.fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot both be True."
            )
        if self.fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` can improve memory efficiency at the potential "
                "cost of reduced precision.",
                stacklevel=2,
            )

        if self.attn is not None:
            if not isinstance(self.attn, dict):
                raise ValueError("attn must be a dictionary")
            self.attn = deepcopy(self.attn)
            if "layers" not in self.attn:
                raise ValueError(
                    "Layer indices are required for hybrid attention layers"
                )
            if "num_heads" not in self.attn:
                raise ValueError("num_heads is required for hybrid attention layers")
            self.attn["num_kv_heads"] = self.attn.get(
                "num_kv_heads", self.attn["num_heads"]
            )
            self.attn["qkv_bias"] = self.attn.get("qkv_bias", False)
            self.attn["window_size"] = self.attn.get("window_size", None)
            self.attn["rope_theta"] = self.attn.get("rope_theta", 10000.0)

        if self.attnres_block_size is not None and self.attnres_block_size != 1:
            if self.attnres_block_size < 2 or self.attnres_block_size % 2 != 0:
                raise ValueError(
                    "`attnres_block_size` must be None, 1, or an even integer."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return an independent dictionary of Raven construction settings."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RavenConfig:
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> RavenConfig:
        return replace(self, **overrides)


__all__ = ["RavenConfig"]
