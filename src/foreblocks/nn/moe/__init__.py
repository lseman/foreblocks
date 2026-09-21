"""Mixture-of-experts layers, feed-forward construction, routing, and dispatch."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.nn.moe.feedforward import FeedForwardBlock as FeedForwardBlock
    from foreblocks.nn.moe.layer import MoEFeedForwardDMoE as MoEFeedForwardDMoE
    from foreblocks.nn.moe.layer import MoERoutingState as MoERoutingState
    from foreblocks.nn.moe.experts import MTPHead as MTPHead

_EXPORTS = {
    'FeedForwardBlock': 'foreblocks.nn.moe.feedforward',
    'MoEFeedForwardDMoE': 'foreblocks.nn.moe.layer',
    'MoERoutingState': 'foreblocks.nn.moe.layer',
    'MTPHead': 'foreblocks.nn.moe.experts',
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
