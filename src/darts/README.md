# Darts - Neural Architecture Search for Time Series Forecasting

Darts is a differentiable Neural Architecture Search (DARTS-family NAS) and time
series forecasting framework. It contains gradient-based and zero-cost search
algorithms, multi-fidelity/ablation search phases, and a rich collection of
neural network architecture blocks for time series and sequence modeling.

## Overview

Darts provides:

- **Neural Architecture Search (NAS)**: bilevel DARTS training, zero-cost proxies,
  candidate scoring/pooling, and multi-fidelity/ablation search phases
- **Architecture Blocks**: a search space of building blocks (attention, MoE,
  positional encodings, sequence/transformer blocks, conv/MLP/spectral/decomposition
  operations) plus the DARTS-specific mixed-op cell and finalization machinery
- **Search Metrics**: FLOPs, parameter count, Jacobian conditioning, Fisher
  information, GRASP, NASWOT, SNIP, SynFlow, and activation diversity
- **Training & Evaluation**: bilevel/final training loops, evaluation metrics,
  backtesting, and architecture diagram visualization

## Methodology notes

The bilevel search splits the supplied training dataset in chronological order:
the first 70% trains model weights and the last 30% updates architecture
parameters. Supply a separate, later validation set for model selection. If
dataset items are overlapping windows, construct the dataset with a suitable
time gap at the boundary; splitting window indices alone does not remove
overlap. Architecture updates use a first-order validation gradient by default;
the optional Hessian correction is an approximation to unrolled DARTS.

Zero-cost scores are ranking proxies, not calibrated estimates of forecast
accuracy. FLOPs count multiply and add operations in Linear and Conv layers per
example; attention, normalization, and other operations are excluded. The
default `snip_mode="current"` evaluates current weights rather than SNIP's
initialization-time saliency. The default `fisher_per_sample=False` uses a
squared batch gradient rather than per-sample empirical Fisher. Use
`snip_mode="init"` and `fisher_per_sample=True` when those definitions matter.

Attention search uses five self-attention kernels in both encoders and causal
decoders (`sdp`, `linear`, `probsparse`, `cosine`, `local`). Causal `linear`
uses prefix feature statistics; causal `probsparse` samples only prefix keys,
selects queries online, and uses prefix means for unselected queries. Linear
attention applies ALiBi, seasonal, and learned relative biases as lag weights
using FFT convolution, without constructing a time-by-time matrix. A soft
attention supernet still evaluates its other, quadratic kernels; the fast
linear path applies to fixed or hard single-path selection.
Search-to-fixed tests cover each available attention mode, position mode,
encoder patch mode, and decoder cross-attention mode.
Three-logit causal-attention checkpoints from the restricted search space load
with `linear` and `probsparse` initialized to low logits. Recreate architecture
optimizer state when expanding those logits.

## Directory Structure

```
darts/
├── architecture/                  # Neural network architecture implementations
│   ├── blocks/                    # Search-space building blocks
│   │   ├── attention.py           # Attention mechanisms
│   │   ├── bridges.py             # Bridge components between blocks
│   │   ├── moe.py                 # Mixture of Experts blocks
│   │   ├── positional.py          # Positional encoding blocks
│   │   ├── primitives.py          # Basic search-space primitives
│   │   ├── sequence.py            # Sequence modeling blocks
│   │   └── transformers.py        # Transformer architecture blocks
│   ├── common/                    # Shared architecture utilities
│   │   ├── attention_math.py      # ALiBi slopes, causal masks, sinusoidal features
│   │   ├── block_wrappers.py      # Searchable mixed-block / fixed-deployment wrappers
│   │   ├── freeze.py              # Parameter freezing helpers
│   │   ├── inspector.py           # Architecture inspection utilities
│   │   └── norms.py               # Normalization layers (LayerNorm, RMSNorm, ...)
│   ├── darts/                     # DARTS differentiable search-space mechanics
│   │   ├── converter.py           # Architecture conversion utilities
│   │   ├── darts_cell.py          # DARTSCell: mixed-op cell w/ progressive search
│   │   ├── finalization.py        # Derive a fixed architecture from learned alphas
│   │   ├── fixed_encoder_decoder.py # Fixed (post-search) encoder/decoder
│   │   ├── genotype.py            # Genotype record + rebuild-from-genotype
│   │   ├── mixed_encoder_decoder.py # Differentiable (search-time) encoder/decoder
│   │   ├── mixed_op.py            # MixedOp: per-edge differentiable operation mix
│   │   └── time_series_darts.py   # TimeSeriesDARTS: top-level searchable model
│   └── ops/                       # Concrete candidate operations
│       ├── conv.py                # Convolutional operations
│       ├── decomposition.py       # Signal decomposition operations
│       ├── fixed.py               # Fixed (non-searched) operations
│       ├── misc.py                # ConvMixer/GRN/PatchEmbed/gated-FFN ops
│       ├── mlp.py                 # MLP operations
│       ├── registry.py            # Operation family/name registry
│       ├── spectral.py            # Spectral/frequency operations
│       └── ssm.py                 # State-space model operations
│
├── search/                        # NAS search orchestration
│   ├── candidates/                # Candidate config, scoring, pooling
│   │   ├── candidate_config.py    # Random search-space sampling / op-family selection
│   │   ├── pool_scoring.py        # Pool-wide diversity, dedup, poolwise rescoring
│   │   ├── scoring.py             # Single-candidate score from weighted metrics
│   │   └── weight_schemes.py
│   ├── metrics/                   # Zero-cost / gradient-based architecture metrics
│   │   ├── activation_diversity.py
│   │   ├── compatibility.py
│   │   ├── computer.py            # Metric computation orchestration
│   │   ├── conditioning.py        # Jacobian conditioning
│   │   ├── config.py
│   │   ├── fisher.py              # Fisher information
│   │   ├── flops.py               # FLOPs estimation
│   │   ├── grasp.py               # GRASP
│   │   ├── jacobian.py
│   │   ├── naswot.py              # NASWOT
│   │   ├── params.py              # Parameter counting
│   │   ├── sensitivity.py         # Learning-rate sensitivity
│   │   ├── snip.py                # SNIP
│   │   ├── synflow.py             # SynFlow
│   │   └── zero_cost_nas.py
│   ├── phases/                    # Multi-stage search pipeline
│   │   ├── ablation.py            # Weight-scheme ablation
│   │   ├── lr_sensitivity.py      # LR sensitivity phase
│   │   ├── multi_fidelity.py      # Multi-fidelity search phase (run_multi_fidelity_search)
│   │   ├── phase_stats.py         # Phase-3 stats payload construction/persistence
│   │   └── phase_utils.py
│   ├── reporting/                 # Reusable search statistics reporting
│   │   └── stats_reporting.py     # save_json/save_csv/mean_std/lpt_estimate/...
│   ├── orchestrator.py            # Candidate evaluation/selection orchestration
│   ├── robust_pool.py             # Op-pool robustness evaluation
│   └── zero_cost.py               # Zero-cost proxy entry points
│
├── training/                      # Training loops and schedules
│   ├── architecture_step.py       # Architecture (alpha) optimization step
│   ├── darts_engine.py            # DARTS engine variants (DARTS/GDAS/DrNAS/PC-DARTS)
│   ├── dynamic_scheduling.py      # Dynamic search-stage scheduling
│   ├── edge_regularization.py     # Edge/entropy regularization
│   ├── final_trainer.py           # Final (post-search) model training
│   ├── optimizers.py              # Optimizer construction
│   ├── perturbation_hessian.py    # Perturbation-based Hessian estimates
│   ├── regularization.py
│   ├── schedulers.py               # LR/temperature schedulers
│   ├── training_loop.py           # Bilevel DARTS training loop
│   └── utils.py
│
├── evaluation/                    # Model evaluation and benchmarking
│   ├── analyzer.py                # StreamlinedDARTSAnalyzer
│   ├── metrics.py                 # compute_metrics / evaluate_on_loader
│   └── plotting.py                # Training-curve / prediction plotting
│
├── visualization/                 # Architecture diagram rendering
│   └── transformer_diagram.py     # Matplotlib transformer/DARTS-cell diagrams
│
├── utils/                         # Shared low-level utilities
│   ├── io.py                      # Checkpoint / genotype (de)serialization
│   ├── tensors.py
│   └── training_helpers.py        # Loss registry, param grouping, progress bars
│
├── trainer.py                     # DARTSTrainer: thin orchestrator over the above
├── config.py                      # Configuration dataclasses (DARTSConfig, ...)
└── __init__.py                    # Package initialization (lazy public API)
```

Every package directory (`architecture`, `search`, `training`, `evaluation`,
`utils`, and the package root itself) exposes its public API through a lazy
`__getattr__` in `__init__.py`, so importing one symbol doesn't pull in every
submodule's dependencies.

## Core API

### Top-level

```python
from darts import (
    DARTSTrainer,
    DARTSConfig,
    TimeSeriesDARTS,
    DARTSCell,
    StreamlinedDARTSAnalyzer,
    ArchitectureInspector,
)
```

### Architecture Search

```python
from darts.search import (
    evaluate_search_candidate,
    select_top_candidates,
    run_parallel_candidate_collection,
    build_weight_schemes,
)
from darts.search import metrics, zero_cost, robust_pool, ablation
```

### Architecture Components

```python
from darts.architecture import (
    TimeSeriesDARTS,
    DARTSCell,
    MixedOp,
    MixedEncoder,
    MixedDecoder,
    FixedEncoder,
    FixedDecoder,
    ArchitectureConverter,
    derive_final_architecture,
)
```

### Visualization

```python
from darts.visualization import (
    make_encoder_layers,
    make_decoder_layers,
    draw_single_block,
    draw_encoder_decoder,
    draw_selected_transformer_architecture,
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
- matplotlib (for `darts.visualization`)
