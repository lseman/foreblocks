import torch

from foreblocks.tf.skip.gateskip import BudgetScheduler
from foreblocks.tf.skip.mod import MoDBudgetScheduler
from foreblocks.tf.transformer import TransformerEncoder


def test_transformer_schedulers_do_not_step_during_eval():
    gate_scheduler = BudgetScheduler(b_start=1.0, b_end=0.5, total_steps=10)

    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        use_gateskip=True,
        gate_budget=0.8,
        use_mod=False,
    )
    model.set_budget_scheduler(gate_scheduler)

    x = torch.randn(2, 6, 2)

    model.eval()
    with torch.no_grad():
        _ = model(x)

    assert gate_scheduler._step == 0


def test_transformer_gate_scheduler_steps_during_training():
    gate_scheduler = BudgetScheduler(b_start=1.0, b_end=0.5, total_steps=10)

    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        use_gateskip=False,
        use_mod=False,
    )
    model.set_budget_scheduler(gate_scheduler)

    x = torch.randn(2, 6, 2)

    model.train()
    _ = model(x)

    assert gate_scheduler._step == 1


def test_transformer_mod_scheduler_steps_only_during_training():
    mod_scheduler = MoDBudgetScheduler(num_layers=1, total_steps=10)

    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        use_gateskip=False,
        use_mod=True,
        mod_budget_scheduler=mod_scheduler,
    )

    x = torch.randn(2, 6, 2)

    model.eval()
    with torch.no_grad():
        _ = model(x)
    assert mod_scheduler._step == 0

    model.train()
    _ = model(x)
    assert mod_scheduler._step == 1
