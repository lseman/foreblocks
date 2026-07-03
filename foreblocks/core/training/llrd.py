"""foreblocks.core.training.llrd.

Layer-wise learning rate decay and warmup-cosine scheduling.

Implements LLRD for fine-tuning pre-trained sequence models, where deeper
layers receive lower learning rates. Provides WarmupCosineLR and
ExponentialLR schedules compatible with multi-param-group optimizers.
Designed for transformer and Mamba-based models.

Core API:
- WarmupCosineLR: linear warmup with cosine annealing to min LR
- ExponentialLR: exponential decay scheduling
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
    """
    Linear warmup followed by cosine annealing to min_lr.

    Works with multi-param-group optimizers (e.g., LLRD-grouped optimizers).
    Step-level scheduler (call .step() after each optimizer.step()).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    warmup_steps : int
        Number of steps during which LR ramps linearly from ~0 to base LR.
    total_steps : int
        Total training steps over which cosine annealing happens
        (measured from the END of warmup).
    min_lr_ratio : float, optional
        Floor LR as a fraction of each param group's base LR (default 0.01).
    """

    optimizer: torch.optim.Optimizer
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.01
    _step: int = field(default=0, init=False)
    _base_lrs: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Capture base LR from each param group."""
        self._base_lrs = [g["lr"] for g in self.optimizer.param_groups]

    def step(self) -> None:
        """Advance the schedule and update optimizer param groups."""
        s = self._step
        self._step += 1

        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self._base_lrs[i]
            min_lr = self.min_lr_ratio * base_lr

            if s < self.warmup_steps:
                # Linear warmup: [0, base_lr]
                lr = (s / max(1, self.warmup_steps)) * base_lr
            else:
                # Cosine annealing from base_lr to min_lr
                progress = (s - self.warmup_steps) / max(1, self.total_steps)
                progress = min(progress, 1.0)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )

            param_group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        """Return the last set of LRs (for compatibility with torch API)."""
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "_step": self._step,
            "_base_lrs": self._base_lrs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
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
    """
    Build parameter groups with layer-wise LR decay (LLRD).

    Detects transformer layer depth via the `.layers.<idx>.` naming pattern
    in parameter names. Deeper layers get progressively lower LR.

    Parameters
    ----------
    model : nn.Module
        The model to extract parameters from.
    base_lr : float
        Base learning rate (for the last layer or non-layer params).
    weight_decay : float
        Weight decay; set to 0 for bias and norm weights per standard practice.
    decay : float, optional
        Multiplicative decay factor per layer (default 0.9). Deeper layers get
        ``base_lr * decay ** depth_offset``.
    num_layers : int, optional
        Expected number of layers (for validation). If None, inferred from
        the deepest layer index found.

    Returns
    -------
    list[dict]
        List of param groups suitable for ``torch.optim.Optimizer(..., param_groups)``.

    Notes
    -----
    - Non-layer params (embedding, input_adapter, final_norm, output head) get
      ``base_lr`` unscaled (treated as depth 0).
    - Bias and LayerNorm/BatchNorm weights get ``weight_decay=0``.
    - If ``share_layers=True`` (single ``shared_layer``, not numbered ``.layers.<i>.``),
      the function falls back to flat LR for that layer.
    """
    param_groups: list[dict[str, Any]] = []
    pattern = re.compile(r"\.layers\.(\d+)\.")
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
            group_name = f"non_layer_{name[:30]}"

        param_groups.append(
            {
                "params": [param],
                "lr": lr,
                "weight_decay": wd,
                "group_name": group_name,
            }
        )

    return param_groups


def _is_no_decay_param(name: str) -> bool:
    """Heuristic: no weight decay for bias and norm parameters."""
    return any(x in name for x in ["bias", "norm", "embedding"])
