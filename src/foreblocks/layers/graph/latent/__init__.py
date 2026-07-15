"""foreblocks.layers.graph.latent.

Latent graph structure learning and adaptive edge sparsification.

Provides learnable latent graph correlation learners and adaptive edge
sparsification for dynamic graph structure learning.

Core API:
- LatentCorrelationLearner: learnable latent graph structure
- AdaptiveEdgeSparsifier: adaptive edge sparsification
- CorrelationConfig: correlation configuration

"""

from foreblocks.layers.graph.latent.latent import (
    AdaptiveEdgeSparsifier,
    CorrelationConfig,
    LatentCorrelationLearner,
)

__all__ = [
    "AdaptiveEdgeSparsifier",
    "CorrelationConfig",
    "LatentCorrelationLearner",
]
