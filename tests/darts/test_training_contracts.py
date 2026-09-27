"""Training modules share batching semantics and explicit run configuration."""

from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset, default_collate

from darts.config import DARTSTrainConfig
from darts.trainer import DARTSTrainer
from darts.training.data import create_bilevel_loaders
from darts.training.training_loop import (
    _run_model_training_epoch,
    _run_validation_epoch,
)


class ForecastProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.alpha = nn.Parameter(torch.zeros(2))
        self.calls = []

    def forward(
        self, x, x_future=None, decoder_targets=None, teacher_forcing_ratio=None
    ):
        self.calls.append((x_future, decoder_targets, teacher_forcing_ratio))
        return x_future * self.weight * self.alpha.softmax(0)[0]


def forecasting_loader():
    return DataLoader(
        TensorDataset(
            torch.randn(10, 4, 2), torch.randn(10, 2, 2), torch.randn(10, 2, 2)
        ),
        batch_size=2,
    )


@pytest.mark.parametrize("metadata_dict", [False, True])
def test_training_and_validation_preserve_covariates_without_label_leakage(
    metadata_dict,
):
    model = ForecastProbe()
    loader = forecasting_loader()
    if metadata_dict:
        loader = DataLoader(loader.dataset, batch_size=2, collate_fn=custom_collate)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    _run_model_training_epoch(
        model=model,
        train_model_loader=loader,
        model_params=list(model.parameters()),
        model_optimizer=optimizer,
        model_scheduler=scheduler,
        scaler=GradScaler(enabled=False),
        loss_fn=nn.MSELoss(),
        gradient_accumulation_steps=2,
        device="cpu",
        use_amp=False,
        verbose=False,
        epoch=0,
    )
    assert len(model.calls) == len(loader)
    assert all(
        future is not None and target is not None for future, target, _ in model.calls
    )
    model.calls.clear()
    _run_validation_epoch(
        model=model,
        val_loader=loader,
        loss_fn=nn.MSELoss(),
        device="cpu",
        use_amp=False,
        verbose=False,
    )
    assert all(
        future is not None and target is None and ratio == 0.0
        for future, target, ratio in model.calls
    )


def custom_collate(batch):
    x, y, future = default_collate(batch)
    return x, y, {"x_future": future}


def test_bilevel_split_preserves_collation_and_seeded_order():
    data = TensorDataset(torch.arange(30), torch.arange(30), torch.arange(30))
    loader = DataLoader(data, batch_size=3, collate_fn=custom_collate)
    first = create_bilevel_loaders(loader, seed=12)
    second = create_bilevel_loaders(loader, seed=12)
    different = create_bilevel_loaders(loader, seed=13)
    for a, b, c in zip(first, second, different):
        assert a.collate_fn is custom_collate
        assert a.persistent_workers == loader.persistent_workers
        assert list(a.sampler) == list(b.sampler)
        # Compare fresh generators because iteration advances generator state.
        assert a.generator.initial_seed() != c.generator.initial_seed()
        assert isinstance(next(iter(a))[2], dict)
    assert list(first[1].dataset.indices) == list(range(21))
    assert list(first[0].dataset.indices) == list(range(21, 30))


def test_bilevel_split_rejects_empty_drop_last_partition():
    loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=3, drop_last=True)
    with pytest.raises(ValueError, match="no batches"):
        create_bilevel_loaders(loader)


def test_complete_search_owns_history_and_does_not_mutate_config():
    model = ForecastProbe()
    trainer = DARTSTrainer(
        input_dim=2, hidden_dims=[4], forecast_horizon=2, seq_length=4, device="cpu"
    )
    config = DARTSTrainConfig(
        epochs=2,
        warmup_epochs=5,
        use_amp=False,
        verbose=False,
        progressive_shrinking=False,
        max_train_batches=1,
        max_val_batches=1,
    )
    before = asdict(config)
    result = trainer.train_darts_model(
        model,
        forecasting_loader(),
        forecasting_loader(),
        train_config=config,
        compute_metrics=False,
    )
    assert asdict(config) == before
    assert trainer.training_history == [result]
    assert len(result["train_losses"]) == 2
    assert torch.isfinite(torch.tensor(result["best_val_loss"]))
    assert all(future is not None for future, _, _ in model.calls)
