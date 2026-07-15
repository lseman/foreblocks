# Darts - Neural Architecture Search for Time Series Forecasting

Darts is a comprehensive Neural Architecture Search (NAS) and time series forecasting framework. It contains advanced search algorithms, zero-cost proxies, multi-fidelity optimization, and a rich collection of neural network architecture blocks specifically designed for time series and sequence modeling.

## Overview

Darts provides:

- **Neural Architecture Search (NAS)**: Advanced search algorithms with zero-cost proxies, gradient-based metrics, and multi-fidelity optimization
- **Architecture Blocks**: Comprehensive collection of neural network building blocks including transformers, convolutions, MLPs, spectral operations, and decomposition operations
- **Search Metrics**: Rich set of architecture evaluation metrics including FLOPs, parameter count, Jacobian conditioning, Fisher information, GRASP, SNIP, SynFlow, and activation diversity
- **Training & Evaluation**: Complete training loops, evaluation metrics, and backtesting utilities

## Directory Structure

```
darts/
├── architecture/                 # Neural network architecture implementations
│   ├── base_blocks.py           # Base architecture blocks and primitives
│   ├── operation_blocks.py      # Operation-level building blocks
│   ├── core_blocks.py           # Core architectural components
│   ├── bb_primitives.py         # Basic search space primitives
│   ├── bb_positional.py         # Positional encoding blocks
│   ├── bb_sequence.py           # Sequence modeling blocks
│   ├── bb_attention.py          # Attention mechanism blocks
│   ├── bb_mixed.py              # Mixed operation blocks
│   ├── bb_moe.py                # Mixture of Experts blocks
│   ├── bb_transformers.py       # Transformer architecture blocks
│   ├── norms.py                 # Normalization layers (LayerNorm, RMSNorm, etc.)
│   ├── conv_ops.py              # Convolutional operations
│   ├── mlp_ops.py               # MLP operations
│   ├── spectral_ops.py          # Spectral/frequency operations
│   ├── decomposition_ops.py     # Signal decomposition operations
│   ├── fixed_encoder_decoder.py # Fixed encoder-decoder architectures
│   ├── inspector.py             # Architecture inspection utilities
│   ├── converter.py             # Architecture conversion utilities
│   ├── bridges.py               # Bridge components between blocks
│   ├── helpers.py               # Helper functions
│   └── utils.py                 # General utilities
│
├── search/                       # Neural Architecture Search framework
│   ├── search.py                # Main search orchestrator
│   ├── orchestrator.py          # Search orchestration and workflow
│   ├── multi_fidelity.py        # Multi-fidelity optimization strategies
│   ├── zero_cost.py             # Zero-cost proxy metrics
│   ├── lr_sensitivity.py        # Learning rate sensitivity analysis
│   ├── phase_utils.py           # Search phase utilities
│   ├── stats.py                 # Search statistics tracking
│   ├── stats_reporting.py       # Statistics reporting utilities
│   ├── candidate_config.py      # Candidate configuration management
│   ├── candidate_scoring.py     # Candidate scoring mechanisms
│   ├── weight_schemes.py        # Weight scheme implementations
│   ├── ablation.py              # Ablation study utilities
│   ├── robust_pool.py           # Robust candidate pooling
│   ├── scoring.py               # General scoring utilities
│   └── metrics/                 # Architecture evaluation metrics
│       ├── __init__.py
│       ├── flops.py             # FLOPs estimation
│       ├── params.py            # Parameter counting
│       ├── conditioning.py      # Jacobian conditioning metrics
│       ├── jacobian.py          # Jacobian-based metrics
│       ├── fisher.py            # Fisher information metrics
│       ├── grasp.py             # GRASP (Gradient-based Proxy) metrics
│       ├── naswot.py            # NASWOT (Neural Architecture Search without Training) metrics
│       ├── snip.py              # SNIP (Synaptic Intelligence) metrics
│       ├── synflow.py           # SynFlow metrics
│       ├── sensitivity.py       # Sensitivity analysis metrics
│       └── activation_diversity.py # Activation diversity metrics
│
├── training/                     # Training utilities and loops
│   (Shared with foreblocks/core/training/)
│
├── evaluation/                   # Model evaluation and benchmarking
│   (Shared with foreblocks/core/evaluation/)
│
├── search/                       # Search and optimization utilities
├── trainer.py                    # Main trainer implementation
├── config.py                     # Configuration management
├── scoring.py                    # Scoring utilities
├── __init__.py                   # Package initialization
├── methodology_illustrative.svg  # Methodology illustration diagram
├── methodology.tex               # Methodology LaTeX documentation
├── transformer_block_illustrative_v2.svg # Transformer block diagram
└── transformer_diagram.py        # Transformer diagram generation script

## Core API

### Architecture Search

```python
from darts.search import (
    NASOrchestrator,
    ZeroCostProxy,
    MultiFidelityOptimizer,
    # Search metrics
    FLOPsEstimator,
    ParameterCounter,
    JacobianConditioning,
    FisherInformation,
    GRASPProxy,
    NASWOTProxy,
    SNIPProxy,
    SynFlowProxy,
)
```

### Architecture Blocks

```python
from darts.architecture import (
    # Basic blocks
    BaseBlock,
    OperationBlock,
    CoreBlock,
    # Search space primitives
    SearchSpacePrimitive,
    PositionalEncodingBlock,
    SequenceBlock,
    AttentionBlock,
    MixedOperationBlock,
    MoEBlock,
    TransformerBlock,
    # Operations
    ConvOperation,
    MLPOperation,
    SpectralOperation,
    DecompositionOperation,
    # Norms
    LayerNorm,
    RMSNorm,
)
```

### Search Metrics

- **FLOPs**: Floating point operations estimation
- **Parameters**: Model parameter counting
- **Jacobian Conditioning**: Condition number of the Jacobian matrix
- **Fisher Information**: Fisher information matrix-based metrics
- **GRASP**: Gradient-based architecture search proxy
- **NASWOT**: Neural Architecture Search without Training
- **SNIP**: Synaptic Intelligence for architecture search
- **SynFlow**: SynFlow-based architecture evaluation
- **Activation Diversity**: Measure of activation pattern diversity
- **Sensitivity**: Learning rate sensitivity analysis

## Key Features

1. **Zero-Cost Proxies**: Fast architecture evaluation without full training
2. **Multi-Fidelity Optimization**: Progressive search from low to high fidelity
3. **Rich Search Space**: Comprehensive collection of architecture blocks and operations
4. **Gradient-Based Metrics**: Jacobian, Fisher information, GRASP for architecture evaluation
5. **Training-Free Evaluation**: NASWOT, SNIP, SynFlow for rapid architecture scoring
6. **Transformer Architectures**: Specialized transformer blocks for time series

## Dependencies

- PyTorch
- NumPy
- SciPy
- Additional search and optimization libraries
