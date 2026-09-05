"""Composable model modules (attention, MoE, blocks, heads, skip).

For attention, import from ``foreblocks.attention`` directly.
This module is kept for backward compat: ``foreblocks.modules.attention``
still works by re-exporting from the top-level package.
"""

import foreblocks.attention as attention  # noqa: F401
from foreblocks.modules import blocks, heads, moe, skip  # noqa: F401

__all__ = ["attention", "blocks", "heads", "moe", "skip"]
