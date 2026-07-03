"""foreblocks.layers.graph.

Graph neural network layers and spatio-temporal architectures.

Provides GCN, GAT, SAGE, EdgeCondGCN, MTGNN, TGGC, and GraphWaveNet
graph convolution layers with spatio-temporal modeling. Includes latent
graph correlation learners and adaptive edge sparsification.

Core API:
- GCNConv, GATConv, SAGEConv: standard GNN convolutions
- MTGNNBlock, MTGNNGraphConstructor: MTGNN spatio-temporal blocks
- TGGCBlock, TGGCModern: temporal spectral graph convolutions
- LatentCorrelationLearner: learnable latent graph structure

"""

from foreblocks.layers.graph.latent import (
    AdaptiveEdgeSparsifier,
    CorrelationConfig,
    LatentCorrelationLearner,
)
from foreblocks.layers.graph.layers import (
    EdgeCondGCN,
    GATConv,
    GCNConv,
    JumpKnowledge,
    MessagePassing,
    SAGEConv,
    StochasticDepth,
)
from foreblocks.layers.graph.mtgnn import (
    MTGNNBlock,
    MTGNNDilatedInception,
    MTGNNGraphConstructor,
    MTGNNMixProp,
    MTGNNProp,
    MTGNNTemporalGatedUnit,
)
from foreblocks.layers.graph.norms import GraphNorm
from foreblocks.layers.graph.spatio_temporal import GraphWaveNetBlock
from foreblocks.layers.graph.tggc import (
    ComplexLinearModes,
    FineTemporalSpectralFilter,
    GraphGegenbauerConv,
    LatentCorrelationLayer,
    MovingAverage,
    SeriesDecomposition,
    TemporalSpectralFilter,
    TGGCBlock,
    TGGCModern,
    TGGCModernConfig,
    build_frequency_indices,
    normalized_laplacian_from_adjacency,
    symmetric_normalize_adjacency,
)

__all__ = [
    "AdaptiveEdgeSparsifier",
    "ComplexLinearModes",
    "CorrelationConfig",
    "EdgeCondGCN",
    "FineTemporalSpectralFilter",
    "GATConv",
    "GCNConv",
    "GraphWaveNetBlock",
    "GraphGegenbauerConv",
    "GraphNorm",
    "JumpKnowledge",
    "LatentCorrelationLayer",
    "LatentCorrelationLearner",
    "MessagePassing",
    "MTGNNBlock",
    "MTGNNDilatedInception",
    "MTGNNGraphConstructor",
    "MTGNNMixProp",
    "MTGNNProp",
    "MTGNNTemporalGatedUnit",
    "MovingAverage",
    "SeriesDecomposition",
    "SAGEConv",
    "StochasticDepth",
    "TGGCBlock",
    "TGGCModern",
    "TGGCModernConfig",
    "TemporalSpectralFilter",
    "build_frequency_indices",
    "normalized_laplacian_from_adjacency",
    "symmetric_normalize_adjacency",
]
