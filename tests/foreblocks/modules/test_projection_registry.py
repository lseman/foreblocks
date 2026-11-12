"""Projection registration must support float settings and dotted names."""

import torch
from torch import nn

from foreblocks.nn.heads.core.projection import ProjectionRegistry


def test_float_options_are_cached_separately_and_roundtrip_checkpoint():
    registry = ProjectionRegistry()
    options = dict(
        category="fusion",
        name="example",
        in_dim=4,
        out_dim=2,
        proj_type="hyper_mlp",
        hidden_dim=8,
    )
    first = registry.get_or_create(**options, dropout=0.1)
    second = registry.get_or_create(**options, dropout=0.2)
    assert registry.get_or_create(**options, dropout=0.1) is first
    assert first is not second
    assert all("." not in key for key in registry.registry)

    restored = ProjectionRegistry()
    for dropout in [0.1, 0.2]:
        restored.get_or_create(**options, dropout=dropout)
    restored.load_state_dict(registry.state_dict(), strict=True)
    value = torch.randn(2, 4)
    first.eval()
    actual = restored.get_or_create(**options, dropout=0.1).eval()(value)
    torch.testing.assert_close(actual, first(value))


def test_dotted_and_literal_escape_names_do_not_collide():
    registry = ProjectionRegistry()
    names = ["head.v1", "head%2Ev1", "head_v1"]
    modules = [nn.Linear(3, 3) for _ in names]
    for name, module in zip(names, modules):
        registry.register_custom(category="carry", name=name, module=module)
    for name, module in zip(names, modules):
        assert registry.get_registered(category="carry", name=name) is module
    assert len(registry.registry) == len(names)


def test_existing_simple_projection_keys_are_unchanged():
    registry = ProjectionRegistry()
    registry.get_or_create(
        category="stage_proj",
        name="parallel_out",
        in_dim=3,
        out_dim=4,
        allow_identity=True,
    )
    assert list(registry.registry) == [
        "stage_proj::parallel_out::3::4::linear::allow_identity=True"
    ]


def test_lazy_projection_inherits_and_tracks_training_mode():
    registry = ProjectionRegistry().eval()
    module = registry.get_or_create(
        category="fusion",
        name="lazy",
        in_dim=4,
        out_dim=2,
        proj_type="hyper_mlp",
        dropout=0.5,
    )
    assert all(not child.training for child in module.modules())
    value = torch.randn(3, 4)
    torch.testing.assert_close(module(value), module(value))
    registry.train()
    assert all(child.training for child in module.modules())
    registry.eval()
    assert all(not child.training for child in module.modules())
