"""foreblocks.ui.

Node metadata, discovery, and UI-facing schema helpers.

Provides the node decorator, auto-spec inference, and discovery utilities
used by the foreblocks Studio for visual workflow construction and node
palette population.

Core API:
- node: decorator for registering discoverable node classes
- build_node_spec: infer node spec from class annotations
- discover_nodes: find all registered foreblocks nodes
- discover_nodes_payload: JSON-serializable discovery data

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
