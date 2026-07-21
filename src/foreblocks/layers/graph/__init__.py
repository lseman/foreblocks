"""foreblocks.layers.graph.

Graph neural network layers and spatio-temporal architectures.

Provides GCN, GAT/GATv2, GIN/GINE, SAGE, EdgeCondGCN, MTGNN, TGGC, and GraphWaveNet
graph convolution layers with spatio-temporal modeling. Includes latent
graph correlation learners and adaptive edge sparsification.

Core API:
- GCNConv, GATConv, GATv2Conv, GINConv, GINEConv, SAGEConv: GNN convolutions
- MTGNNBlock, MTGNNGraphConstructor: MTGNN spatio-temporal blocks
- TGGCBlock, TGGCModern: temporal spectral graph convolutions
- LatentCorrelationLearner: learnable latent graph structure

"""

from foreblocks.layers.graph.latent import (
    AdaptiveEdgeSparsifier,
    CorrelationConfig,
    LatentCorrelationLearner,
)
from foreblocks.layers.graph.conv import (
    EdgeCondGCN,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
    JumpKnowledge,
    MessagePassing,
    SAGEConv,
    StochasticDepth,
)
from foreblocks.layers.graph.norms import GraphNorm
from foreblocks.layers.graph.spatiotemporal import (
    BackwardDiffusionConv,
    ComplexLinearModes,
    DiffusionConv,
    FineTemporalSpectralFilter,
    GraphGegenbauerConv,
    GraphWaveNetBlock,
    GraphWaveNetTemporalConv,
    LatentCorrelationLayer,
    MTGNNBlock,
    MTGNNDilatedInception,
    MTGNNGraphConstructor,
    MTGNNMixProp,
    MTGNNProp,
    MTGNNTemporalGatedUnit,
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
    "BackwardDiffusionConv",
    "ComplexLinearModes",
    "CorrelationConfig",
    "DiffusionConv",
    "EdgeCondGCN",
    "FineTemporalSpectralFilter",
    "GATConv",
    "GATv2Conv",
    "GCNConv",
    "GINConv",
    "GINEConv",
    "GraphWaveNetBlock",
    "GraphWaveNetTemporalConv",
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
