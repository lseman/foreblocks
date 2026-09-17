import torch

from foreblocks.models.kan import KANModel, Model


def test_kan_model_returns_time_major_output_without_revin() -> None:
    model = KANModel(
        c_in=4,
        context_window=16,
        target_window=6,
        patch_len=4,
        stride=2,
        revin=False,
    )

    out = model(torch.randn(2, 16, 4))

    assert out.shape == (2, 6, 4)


def test_kan_model_supports_configurable_head_and_baseline_block_options() -> None:
    model = KANModel(
        c_in=3,
        context_window=20,
        target_window=5,
        patch_len=5,
        stride=5,
        revin=False,
        head_hidden_dims=(32, 16),
        head_dropout=0.0,
        block_dropout=0.0,
        block_use_norm=False,
        final_norm=False,
    )

    out = model(torch.randn(2, 20, 3))

    assert out.shape == (2, 5, 3)


def test_model_alias_points_to_kan_model() -> None:
    assert Model is KANModel
