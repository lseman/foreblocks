import pytest
import torch
from torch import nn

from foreblocks import ForecastingModel, Trainer, TrainingConfig
from foreblocks.models.baselines.dlinear import DLinear
from foreblocks.training.execution.epochs import backward_step


@pytest.mark.parametrize("backbone", [False, True])
def test_forecast_weights_train_and_restore_before_first_forward(tmp_path, backbone):
    def build():
        if backbone:
            return ForecastingModel(
                forecasting_strategy="direct",
                model_type="mamba",
                encoder=nn.Identity(),
                target_len=2,
                output_size=3,
            )
        return DLinear(2, ma_kernel=3)

    def trainer():
        return Trainer(
            build(),
            device="cpu",
            auto_track=False,
            config=TrainingConfig(use_amp=False),
        )

    first = trainer()
    x = torch.randn(4, 8, 3)
    output = first.model(x)
    weights = (
        (first.model._direct_backbone_projection.weight,)
        if backbone
        else (
            first.model.W_season,
            first.model.W_trend,
        )
    )
    before = [w.detach().clone() for w in weights]
    output.square().mean().backward()
    first.optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, weights))
    path = tmp_path / "forecast.pt"
    first.save(path)
    fresh = trainer()
    fresh.load(path)
    torch.testing.assert_close(fresh.model(x), first.model(x))
    assert all(p in fresh.optimizer.state for p in fresh.model.parameters())


@pytest.mark.parametrize(("batches", "updates"), [(2, 1), (4, 1), (5, 2), (7, 2)])
def test_accumulation_averages_each_group(batches, updates):
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    config = TrainingConfig(gradient_accumulation_steps=4, gradient_clip_val=None)
    for i in range(batches):
        backward_step(
            model.weight.sum(), model, config, optimizer, i, batches, None, False
        )
    assert model.weight.item() == -updates


def test_streaming_loader_flushes_partial_accumulation_group():
    class Samples(torch.utils.data.IterableDataset):
        def __iter__(self):
            for _ in range(5):
                yield torch.ones(1), torch.ones(1)

    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model,
        device="cpu",
        auto_track=False,
        optimizer=optimizer,
        config=TrainingConfig(
            use_amp=False, gradient_accumulation_steps=4, gradient_clip_val=None
        ),
    )
    trainer.train_epoch(torch.utils.data.DataLoader(Samples(), batch_size=1))
    # Two mean-squared-error updates, first 0 -> 0.2 and then 0.2 -> 0.36.
    torch.testing.assert_close(model.weight, torch.tensor([[0.36]]))


def test_legacy_checkpoint_restores_optimizer_moments_to_channel_mixer(tmp_path):
    model = DLinear(2, in_channels=3, out_channels=1, ma_kernel=3)
    legacy_optimizer = torch.optim.AdamW(model.channel_mixer.parameters())
    trainer = Trainer(model, optimizer=legacy_optimizer, auto_track=False, device="cpu")
    x = torch.randn(4, 8, 3)
    model(x).square().mean().backward()
    legacy_optimizer.step()
    path = tmp_path / "legacy.pt"
    trainer.save(path)
    fresh = Trainer(
        DLinear(2, in_channels=3, out_channels=1, ma_kernel=3),
        auto_track=False,
        device="cpu",
    )
    fresh.load(path)
    torch.testing.assert_close(fresh.model(x), model(x))
    torch.testing.assert_close(
        fresh.optimizer.state[fresh.model.channel_mixer.weight]["exp_avg"],
        legacy_optimizer.state[model.channel_mixer.weight]["exp_avg"],
    )
    assert fresh.model.W_season not in fresh.optimizer.state
