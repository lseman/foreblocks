from types import SimpleNamespace

import torch
import torch.nn as nn

from darts.training.architecture_step import compose_architecture_loss


class _SearchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([0.2, -0.1]))
        self.cells = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.softmax(self.alpha, dim=0)[0]

    def get_orthogonal_regularization(self) -> torch.Tensor:
        return self.alpha.square().mean()

    def get_moe_balance_loss(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha).mean()


class _Regularizer:
    def compute_regularization(self, model, arch_params, epoch, epochs):
        return {"total": model.alpha.sum() * 0.0}


class _AlphaTracker:
    def component_alpha_sources(self, model):
        return [{"name": "encoder_block", "alpha": model.alpha}]


def test_composed_architecture_loss_preserves_architecture_gradients() -> None:
    model = _SearchModel()
    result = compose_architecture_loss(
        model=model,
        x=torch.ones(2, 3),
        y=torch.zeros(2, 3),
        model_kwargs={},
        loss_fn=nn.MSELoss(),
        regularizer=_Regularizer(),
        alpha_tracker=_AlphaTracker(),
        arch_params=[model.alpha],
        epoch=2,
        epochs=10,
        warmup_epochs=1,
        device="cpu",
        engine_variant=None,
        engine_cfg=SimpleNamespace(),
        state_mix_ortho_reg_weight=0.1,
        beta_darts_weight=0.1,
        edge_diversity_weight=0.0,
        edge_usage_balance_weight=0.0,
        edge_identity_cap=1.0,
        edge_identity_cap_weight=0.0,
        moe_balance_weight=0.1,
        transformer_exploration_weight=0.1,
        edge_sharpening_max_weight=0.0,
        edge_sharpening_start_frac=0.5,
    )

    result.loss.backward()

    assert result.loss.ndim == 0
    assert result.edge_diversity_pairs == 0
    assert model.alpha.grad is not None
    assert torch.isfinite(model.alpha.grad).all()
