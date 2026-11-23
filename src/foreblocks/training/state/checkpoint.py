"""Trainer checkpoint serialization and restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch


class CheckpointableTrainer(Protocol):
    model: Any
    optimizer: Any
    config: Any
    history: Any
    alpha_optimizer: Any
    conformal_engine: Any
    device: torch.device


def save_trainer_checkpoint(trainer: CheckpointableTrainer, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "config": (
            trainer.config.__dict__
            if hasattr(trainer.config, "__dict__")
            else dict(trainer.config)
        ),
        "history": {
            "train_losses": trainer.history.train_losses,
            "val_losses": trainer.history.val_losses,
            "learning_rates": trainer.history.learning_rates,
            "alpha_values": trainer.history.alpha_values,
        },
    }
    if trainer.alpha_optimizer is not None:
        checkpoint["alpha_optimizer_state_dict"] = (
            trainer.alpha_optimizer.state_dict()
        )
    engine = trainer.conformal_engine
    if engine is not None and getattr(engine, "radii", None) is not None:
        checkpoint["conformal_radii"] = engine.radii
        checkpoint["conformal_method"] = engine.method
    torch.save(checkpoint, target)


def _restore_optimizer(trainer: CheckpointableTrainer, saved: dict[str, Any]) -> None:
    """Restore older optimizers that were built before forecast weights existed."""
    from foreblocks.models.baselines.dlinear import DLinear
    from foreblocks.models.forecasting import ForecastingModel

    groups = trainer.optimizer.param_groups
    saved_groups = saved["param_groups"]
    if len(groups) == len(saved_groups) and any(
        len(current["params"]) != len(old["params"]) for current, old in zip(groups, saved_groups)
    ):
        formerly_lazy = set()
        for module in trainer.model.modules():
            if isinstance(module, DLinear):
                formerly_lazy.update(module.parameters(recurse=False))
            elif isinstance(module, ForecastingModel) and module._direct_backbone_projection is not None:
                formerly_lazy.update(module._direct_backbone_projection.parameters())

        # Optimizer checkpoints identify parameters by position within a group.
        # Preserve every old position and give newly tracked weights empty state.
        next_id = 1 + max((i for g in saved_groups for i in g["params"]), default=-1)
        restored_groups = []
        for current, old in zip(groups, saved_groups):
            params = current["params"]
            old_ids = old["params"]
            if len(params) != len(old_ids) and sum(p not in formerly_lazy for p in params) == len(old_ids):
                existing = iter(old_ids)
                ids = []
                for param in params:
                    if param in formerly_lazy:
                        ids.append(next_id)
                        next_id += 1
                    else:
                        ids.append(next(existing))
                old = {**old, "params": ids}
            restored_groups.append(old)
        saved = {**saved, "param_groups": restored_groups}
    trainer.optimizer.load_state_dict(saved)


def load_trainer_checkpoint(trainer: CheckpointableTrainer, path: str | Path) -> None:
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.to(trainer.device)
    _restore_optimizer(trainer, checkpoint["optimizer_state_dict"])
    if "alpha_optimizer_state_dict" in checkpoint and trainer.alpha_optimizer is not None:
        trainer.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
    if "config" in checkpoint:
        if hasattr(trainer.config, "update"):
            trainer.config.update(**checkpoint["config"])
        elif hasattr(trainer.config, "__dict__"):
            trainer.config.__dict__.update(checkpoint["config"])
    if "conformal_radii" in checkpoint and trainer.conformal_engine is not None:
        trainer.conformal_engine.radii = checkpoint["conformal_radii"]


__all__ = ["load_trainer_checkpoint", "save_trainer_checkpoint"]
