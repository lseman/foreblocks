"""Layer-wise learning-rate decay and warmup scheduling.

Layer-wise learning rate decay and warmup-cosine scheduling.

Implements LLRD for fine-tuning pre-trained sequence models, where early
layers receive lower learning rates and task-facing layers receive the base
LR. Provides a WarmupCosineLR schedule compatible with multi-param-group
optimizers. Designed for transformer and Mamba-based models.

Core API:
- WarmupCosineLR: linear warmup with cosine annealing to min LR
- get_llrd_param_groups: create layer-wise LR-decayed param groups

"""

import math
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class WarmupCosineLR:
    optimizer: torch.optim.Optimizer
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.01
    _step: int = field(default=0, init=False)
    _base_lrs: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps cannot exceed total_steps")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self._base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        if self.warmup_steps > 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

    def step(self) -> None:
        self._step += 1
        s = self._step

        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self._base_lrs[i]
            min_lr = self.min_lr_ratio * base_lr

            if s <= self.warmup_steps and self.warmup_steps > 0:
                # Linear warmup: [0, base_lr]
                lr = (s / max(1, self.warmup_steps)) * base_lr
            else:
                # Cosine annealing from base_lr to min_lr
                cosine_steps = max(1, self.total_steps - self.warmup_steps)
                progress = (s - self.warmup_steps) / cosine_steps
                progress = max(0.0, min(progress, 1.0))
                lr = min_lr + 0.5 * (base_lr - min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )

            param_group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "_step": self._step,
            "_base_lrs": self._base_lrs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.total_steps = state_dict.get("total_steps", self.total_steps)
        self.min_lr_ratio = state_dict.get("min_lr_ratio", self.min_lr_ratio)
        self._step = state_dict.get("_step", 0)
        self._base_lrs = state_dict.get("_base_lrs", self._base_lrs)


def get_llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    decay: float = 0.9,
    num_layers: int | None = None,
) -> list[dict[str, Any]]:
    param_groups: list[dict[str, Any]] = []
    pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    max_layer_idx = -1

    # Scan all parameters to classify them
    for name, param in model.named_parameters():
        match = pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            max_layer_idx = max(max_layer_idx, layer_idx)
        # We'll compute LR on the fly during final loop

    # Infer num_layers if not provided
    if num_layers is None and max_layer_idx >= 0:
        num_layers = max_layer_idx + 1

    # Build param groups: one group per param, with appropriate LR
    for name, param in model.named_parameters():
        match = pattern.search(name)
        is_no_decay = _is_no_decay_param(name)
        wd = 0.0 if is_no_decay else weight_decay

        if match and max_layer_idx >= 0 and num_layers is not None:
            # Layer param: use depth-scaled LR
            layer_idx = int(match.group(1))
            depth_offset = num_layers - 1 - layer_idx
            lr = base_lr * (decay**depth_offset)
            group_name = f"layer_{layer_idx}_{name.split('.')[-1][:20]}"
        else:
            # Non-layer param or no numbered layers found: use base LR
            lr = base_lr
            layer_idx = None
            group_name = f"other_{name[:30]}"

        param_groups.append(
            {
                "params": [param],
                "lr": lr,
                "weight_decay": wd,
                "group_name": group_name,
                "layer_idx": layer_idx,
            }
        )

    return param_groups


def _is_no_decay_param(name: str) -> bool:
    return any(x in name for x in ["bias", "norm", "embedding"])
