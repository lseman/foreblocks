"""
ForeBlocks DARTS: Neural Architecture Search for Time Series Forecasting
"""

# Neural Building Blocks
# Core DARTS Components
from .darts import (  # Temporal Operations; Attention Mechanisms; Spectral Analysis; Advanced Neural Blocks
    AttentionOp,
    ConvMixerOp,
    DARTSCell,
    FixedOp,
    FourierOp,
    GRNOp,
    IdentityOp,
    MixedOp,
    ResidualMLPOp,
    TCNOp,
    TimeConvOp,
    TimeSeriesDARTS,
    TransformerOp,
    WaveletOp,
)

# Zero-Cost Metrics and Search Functions
from .darts_run import (  # Zero-Cost Metrics; Training and Search Functions; Utility Functions
    adapt_metric_weights,
    aggregate_metrics,
    compute_fisher,
    compute_grasp,
    compute_jacob_cov,
    compute_naswot,
    compute_sensitivity,
    compute_snip,
    compute_synflow,
    compute_weight_conditioning,
    derive_final_architecture,
    detect_correlations,
    estimate_flops,
    evaluate_zero_cost_metrics,
    expand_operations,
    gumbel_softmax,
    multi_fidelity_darts_search,
    track_nas_performance,
    train_darts_model,
    train_final_model,
)

__version__ = "1.0.0"
__author__ = "ForeBlocks Team"

__all__ = [
    # Core Components
    "TimeSeriesDARTS",
    "DARTSCell",
    "MixedOp",
    "FixedOp",
    # Neural Operations
    "TimeConvOp",
    "TCNOp",
    "ConvMixerOp",
    "AttentionOp",
    "TransformerOp",
    "WaveletOp",
    "FourierOp",
    "GRNOp",
    "ResidualMLPOp",
    "IdentityOp",
    # Zero-Cost Metrics
    "evaluate_zero_cost_metrics",
    "compute_synflow",
    "compute_naswot",
    "compute_grasp",
    "compute_fisher",
    "compute_snip",
    "compute_jacob_cov",
    "compute_sensitivity",
    "compute_weight_conditioning",
    "estimate_flops",
    "aggregate_metrics",
    # Search Functions
    "multi_fidelity_darts_search",
    "train_darts_model",
    "train_final_model",
    "derive_final_architecture",
    "expand_operations",
    # Utilities
    "track_nas_performance",
    "adapt_metric_weights",
    "detect_correlations",
    "gumbel_softmax",
]
