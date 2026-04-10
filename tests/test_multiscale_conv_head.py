import torch

from foreblocks.core.heads.multiscale_conv_head import MultiScaleConvHead


def test_multiscale_conv_head_preserves_shape_in_legacy_mode():
    head = MultiScaleConvHead(
        feature_dim=3,
        num_scales=4,
        pool_factor=2,
        use_nhits_style_refinement=False,
    )
    x = torch.randn(2, 64, 3)

    y = head(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_multiscale_conv_head_supports_nhits_like_refinement():
    head = MultiScaleConvHead(
        feature_dim=2,
        num_scales=4,
        pool_factor=2,
        use_fft_filter=True,
        use_nhits_style_refinement=True,
        interpolation_mode="linear",
        refinement_hidden_mult=2.0,
        refinement_dropout=0.1,
    )
    x = torch.randn(4, 96, 2)

    y = head(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_multiscale_conv_head_supports_nearest_interpolation_refinement():
    head = MultiScaleConvHead(
        feature_dim=1,
        num_scales=3,
        pool_factor=2,
        use_nhits_style_refinement=True,
        interpolation_mode="nearest",
    )
    x = torch.randn(3, 48, 1)

    y = head(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
