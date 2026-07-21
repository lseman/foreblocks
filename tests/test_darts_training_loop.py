import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from darts.training.training_loop import (
    _optimizer_steps_per_epoch,
    _run_model_training_epoch,
)
from darts.training.utils import snapshot_state_dict
from darts.architecture.core_blocks import DARTSCell


class _CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_snapshot_state_dict_is_cpu_non_aliasing_and_dtype_preserving() -> None:
    model = nn.Module()
    model.weight = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    model.register_buffer("steps", torch.tensor([3], dtype=torch.int64))

    snapshot = snapshot_state_dict(model)
    with torch.no_grad():
        model.weight.fill_(9.0)
        model.steps.fill_(7)

    assert snapshot["weight"].device.type == "cpu"
    assert snapshot["weight"].dtype == torch.float64
    assert snapshot["steps"].dtype == torch.int64
    assert snapshot["weight"].item() == 1.0
    assert snapshot["steps"].item() == 3


def test_optimizer_step_count_uses_ceiling_and_batch_cap() -> None:
    assert _optimizer_steps_per_epoch(5, None, 2) == 3
    assert _optimizer_steps_per_epoch(5, 3, 2) == 2
    assert _optimizer_steps_per_epoch(3, 99, 2) == 2


def test_training_epoch_steps_partial_accumulation_window() -> None:
    model = nn.Linear(2, 2)
    loader = DataLoader(
        TensorDataset(torch.randn(5, 2), torch.randn(5, 2)),
        batch_size=1,
        shuffle=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = _CountingScheduler()

    loss = _run_model_training_epoch(
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

    assert scheduler.steps == 3
    assert torch.isfinite(torch.tensor(loss))


def test_cell_caches_topology_and_residual_search_receives_gradients() -> None:
    torch.manual_seed(7)
    cell = DARTSCell(
        input_dim=4,
        latent_dim=4,
        seq_length=6,
        num_nodes=3,
        initial_search=True,
        op_gdas=False,
    )
    cell.train()
    x = torch.randn(2, 6, 4)

    first = cell(x)
    cached_routing = cell._edge_routing
    second = cell(x)
    second.square().mean().backward()

    assert cached_routing is not None
    assert cell._edge_routing is cached_routing
    assert first.shape == second.shape == (2, 6, 4)
    assert cell.residual_pattern_alphas.grad is not None
    assert torch.isfinite(cell.residual_pattern_alphas.grad).all()
