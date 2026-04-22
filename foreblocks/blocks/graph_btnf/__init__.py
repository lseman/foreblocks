from .latent import AdaptiveEdgeSparsifier, CorrelationConfig, LatentCorrelationLearner
from .layers import (
    EdgeCondGCN,
    GATConv,
    GCNConv,
    JumpKnowledge,
    MessagePassing,
    SAGEConv,
    StochasticDepth,
)
from .network import GraphPreprocessor, LatentGraphNetwork
from .norms import GraphNorm

# Backward-compatible aliases.
CorrelationConfigNTF = CorrelationConfig
LatentCorrelationLearnerNTF = LatentCorrelationLearner
MessagePassingNTF = MessagePassing
GCNConvNTF = GCNConv
SAGEConvNTF = SAGEConv
GATConvNTF = GATConv
EdgeCondGCNNTF = EdgeCondGCN
JumpKnowledgeNTF = JumpKnowledge
LatentGraphNetworkNTF = LatentGraphNetwork
GraphPreprocessorNTF = GraphPreprocessor

__all__ = [
    "AdaptiveEdgeSparsifier",
    "CorrelationConfig",
    "CorrelationConfigNTF",
    "EdgeCondGCN",
    "EdgeCondGCNNTF",
    "GATConv",
    "GATConvNTF",
    "GCNConv",
    "GCNConvNTF",
    "GraphNorm",
    "GraphPreprocessor",
    "GraphPreprocessorNTF",
    "JumpKnowledge",
    "JumpKnowledgeNTF",
    "LatentCorrelationLearner",
    "LatentCorrelationLearnerNTF",
    "LatentGraphNetwork",
    "LatentGraphNetworkNTF",
    "MessagePassing",
    "MessagePassingNTF",
    "SAGEConv",
    "SAGEConvNTF",
    "StochasticDepth",
]
