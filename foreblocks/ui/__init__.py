"""foreblocks.ui.

Package initializer that exposes the public symbols for this namespace.
It belongs to the node metadata, discovery, and UI-facing schema helpers area of Foreblocks.
"""

from foreblocks.ui.auto_spec import build_node_spec
from foreblocks.ui.discovery import discover_nodes, discover_nodes_payload
from foreblocks.ui.node_spec import node


__all__ = [
    "node",
    "build_node_spec",
    "discover_nodes",
    "discover_nodes_payload",
]
