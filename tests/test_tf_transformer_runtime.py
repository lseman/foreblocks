import torch

from foreblocks.models.transformer.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from foreblocks.modules.skip.gateskip import BudgetScheduler
from foreblocks.modules.skip.mod import MoDBudgetScheduler


def _optimizer_param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }


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


def test_transformer_attention_params_are_optimizer_visible_before_forward():
    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        patch_encoder=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before_param_count = len(list(model.parameters()))

    _ = model(torch.randn(2, 6, 2))

    assert len(list(model.parameters())) == before_param_count
    assert {id(p) for p in model.parameters()} <= _optimizer_param_ids(optimizer)


def test_shared_hybrid_attention_materializes_configured_backends():
    model = TransformerEncoder(
        input_size=2,
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        attention_mode="hybrid",
        share_layers=True,
        patch_encoder=False,
    )
    layer = model.shared_layer

    assert layer is not None
    assert layer.self_attn_lin is not None
    assert layer.self_attn_std is not None

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert {id(p) for p in model.parameters()} <= _optimizer_param_ids(optimizer)


def test_standalone_encoder_layer_materializes_selected_attention():
    layer = TransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        layer_attention_type="linear",
    )

    assert layer.self_attn_lin is not None
    assert layer.self_attn_std is None
