import torch

from foreblocks.models.kan import (
    DEFAULT_POLY_FAMILIES,
    POLY_FAMILIES,
    PolyLayerConfig,
    build_poly_layer,
)


def test_build_poly_layer_supports_gegenbauer_and_laguerre() -> None:
    x = torch.randn(2, 5, 3)

    gegen = build_poly_layer(
        "gegenbauer",
        input_dim=3,
        output_dim=4,
        config=PolyLayerConfig(degree=4, gegen_alpha=1.25),
    )
    laguerre = build_poly_layer(
        "laguerre",
        input_dim=3,
        output_dim=4,
        config=PolyLayerConfig(degree=4, laguerre_alpha=0.5),
    )

    y_gegen = gegen(x)
    y_laguerre = laguerre(x)

    assert y_gegen.shape == (2, 5, 4)
    assert y_laguerre.shape == (2, 5, 4)


def test_poly_family_registry_exposes_new_families() -> None:
    assert "gegenbauer" in POLY_FAMILIES
    assert "laguerre" in POLY_FAMILIES
    assert "gegenbauer" in DEFAULT_POLY_FAMILIES
