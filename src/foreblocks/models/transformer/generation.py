"""Generation policy independent from decoder model construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 1
    return_dict: bool = True
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")


__all__ = ["GenerationConfig"]
