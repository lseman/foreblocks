"""foreblocks.layers.graph.spatiotemporal.

Spatio-temporal and temporal spectral graph convolution architectures.

Provides MTGNN, TGGC, and GraphWaveNet spatio-temporal modeling blocks.

Core API:
- MTGNNBlock, MTGNNGraphConstructor: MTGNN spatio-temporal blocks
- TGGCBlock, TGGCModern: temporal spectral graph convolutions
- GraphWaveNetBlock, DiffusionConv: real GraphWaveNet with diffusion propagation

"""

from foreblocks.layers.graph.spatiotemporal.graph_wavenet import (
    BackwardDiffusionConv,
    DiffusionConv,
    GraphWaveNetBlock,
    GraphWaveNetTemporalConv,
)
from foreblocks.layers.graph.spatiotemporal.mtgnn import (
    MTGNNBlock,
    MTGNNDilatedInception,
    MTGNNGraphConstructor,
    MTGNNMixProp,
    MTGNNProp,
    MTGNNTemporalGatedUnit,
)
from foreblocks.layers.graph.spatiotemporal.tggc import (
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
    "BackwardDiffusionConv",
    "ComplexLinearModes",
    "DiffusionConv",
    "FineTemporalSpectralFilter",
    "GraphGegenbauerConv",
    "GraphWaveNetBlock",
    "GraphWaveNetTemporalConv",
    "LatentCorrelationLayer",
    "MTGNNBlock",
    "MTGNNDilatedInception",
    "MTGNNGraphConstructor",
    "MTGNNMixProp",
    "MTGNNProp",
    "MTGNNTemporalGatedUnit",
    "MovingAverage",
    "SeriesDecomposition",
    "TGGCBlock",
    "TGGCModern",
    "TGGCModernConfig",
    "TemporalSpectralFilter",
    "build_frequency_indices",
    "normalized_laplacian_from_adjacency",
    "symmetric_normalize_adjacency",
]
