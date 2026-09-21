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


def load_trainer_checkpoint(trainer: CheckpointableTrainer, path: str | Path) -> None:
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.to(trainer.device)
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
