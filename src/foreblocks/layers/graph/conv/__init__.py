"""foreblocks.layers.graph.conv.

Standard graph convolution layer implementations.

Provides GCN, GAT/GATv2, GIN/GINE, SAGE, and EdgeCond graph convolutions with proper
message passing infrastructure. Includes CachedNorm for batch normalization
caching and StochasticDepth for layer dropout. Supports batched and
unbatched graph inputs.

Core API:
- MessagePassing: base message passing layer
- GCNConv: graph convolutional layer
- GATConv: graph attention layer
- GATv2Conv: dynamic graph attention layer
- GINConv, GINEConv: expressive sum-aggregation layers
- SAGEConv: simplifying graph convolution
- EdgeCondGCN: edge-conditioned GCN
- StochasticDepth: stochastic depth regularization

"""

from foreblocks.layers.graph.conv.edge_cond import EdgeCondGCN
from foreblocks.layers.graph.conv.gat import GATConv, GATv2Conv
from foreblocks.layers.graph.conv.gcn import GCNConv
from foreblocks.layers.graph.conv.gin import GINConv, GINEConv
from foreblocks.layers.graph.conv.jump_knowledge import JumpKnowledge
from foreblocks.layers.graph.conv.message_passing import (
    CachedNorm,
    GraphConvBase,
    MessagePassing,
)
from foreblocks.layers.graph.conv.sage import SAGEConv
from foreblocks.layers.graph.conv.stochastic_depth import StochasticDepth

__all__ = [
    "CachedNorm",
    "EdgeCondGCN",
    "GATConv",
    "GATv2Conv",
    "GCNConv",
    "GINConv",
    "GINEConv",
    "GraphConvBase",
    "JumpKnowledge",
    "MessagePassing",
    "SAGEConv",
    "StochasticDepth",
]
