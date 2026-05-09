from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from foreblocks.config import TrainingConfig
from foreblocks.training.trainer import Trainer


class TinyForecast(nn.Module):
    def __init__(self, horizon: int = 2, output_dim: int = 1) -> None:
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.proj = nn.Linear(3, horizon * output_dim)

    def forward(self, x: torch.Tensor, *_, **__) -> torch.Tensor:
        x_last = x[:, -1, :]
        out = self.proj(x_last)
        return out.reshape(x.size(0), self.horizon, self.output_dim)


def _loader() -> DataLoader:
    x = torch.randn(4, 5, 3)
    y = torch.randn(4, 2, 1)
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def test_trainer_resolves_device_to_torch_device_and_cpu_safe_amp() -> None:
    trainer = Trainer(
        TinyForecast(),
        config=TrainingConfig(num_epochs=1, use_amp=True),
        auto_track=False,
        device=torch.device("cpu"),
    )

    assert trainer.device == torch.device("cpu")
    assert not trainer._amp_enabled

    loss, components = trainer.train_epoch(_loader())

    assert loss > 0
    assert "task_loss" in components


def test_step_scheduler_does_not_treat_loss_as_epoch() -> None:
    trainer = Trainer(
        TinyForecast(),
        config=TrainingConfig(
            num_epochs=2,
            use_amp=False,
            learning_rate=0.1,
            scheduler_type="step",
            lr_step_size=1,
            lr_gamma=0.5,
        ),
        auto_track=False,
        device="cpu",
    )

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.optimizer.step()
    trainer._step_scheduler(train_loss=123.0, val_loss=456.0)

    assert trainer.optimizer.param_groups[0]["lr"] == 0.05


def test_train_zero_epochs_does_not_reference_missing_epoch() -> None:
    trainer = Trainer(
        TinyForecast(),
        config=TrainingConfig(num_epochs=0, use_amp=False),
        auto_track=False,
        device="cpu",
    )

    history = trainer.train(_loader())

    assert history.train_losses == []


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    trainer = Trainer(
        TinyForecast(),
        config=TrainingConfig(num_epochs=1, use_amp=False),
        auto_track=False,
        device="cpu",
    )
    path = tmp_path / "nested" / "trainer.pt"

    trainer.save(path)

    assert path.exists()
