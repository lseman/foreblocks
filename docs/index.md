# ForeBlocks Docs

<div class="hero-panel">
  <span class="eyebrow">ForeBlocks v{{ $frontmatter.foreBlocksVersion }} | Docs Home</span>
  <h1>Forecasting, preprocessing, search, and tooling in one documentation hub</h1>
  <p class="hero-lead">
    <code>foreblocks</code> is the forecasting library. <code>foretools</code> is the
    companion toolbox. The safest way in is still the small public API path, but the docs
    now make it easier to branch into preprocessing, architecture search, uncertainty,
    or dashboard tooling once the baseline loop is already working.
  </p>
  <div class="hero-actions">
    <a class="md-button md-button--primary" href="getting-started/">Run your first model</a>
    <a class="md-button" href="overview/">Read the overview</a>
    <a class="md-button" href="reference/public-api/">Browse the public API</a>
  </div>
</div>

<div class="metric-strip">
  <div class="metric-chip"><strong>Stable first step</strong><span><code>ForecastingModel</code> + <code>Trainer</code> + <code>ModelEvaluator</code> remain the best first run.</span></div>
  <div class="metric-chip"><strong>Raw-series bridge</strong><span><code>TimeSeriesHandler</code> helps when your input is a raw <code>[T, D]</code> array instead of ready-made windows.</span></div>
  <div class="metric-chip"><strong>Search stack</strong><span>DARTS guides cover screening, bilevel search, retraining, and result analysis.</span></div>
  <div class="metric-chip"><strong>Companion tooling</strong><span><code>foretools</code> adds BOHB, synthetic series generation, VMD, and feature engineering.</span></div>
</div>

## Choose your route

<div class="path-grid">
  <div class="path-card">
    <p class="route-kicker">Path 01</p>
    <h3><a href="getting-started/">Train a baseline first</a></h3>
    <p>Use the smallest reliable path through the public API before adding preprocessing, search, or dashboard features.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Path 02</p>
    <h3><a href="preprocessor/">Start from raw series</a></h3>
    <p>Scale, filter, impute, and slice a raw multivariate array into training windows with the preprocessing stack.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Path 03</p>
    <h3><a href="transformer/">Customize architectures</a></h3>
    <p>Transformer, MoE, custom blocks, and Hybrid Mamba guides explain the more configurable internal subsystems.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Path 04</p>
    <h3><a href="darts/">Search architectures</a></h3>
    <p>Move into DARTS once the baseline path is healthy and you want automated structure search instead of hand-picking modules.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Path 05</p>
    <h3><a href="uncertainty/">Add uncertainty intervals</a></h3>
    <p>Use conformal workflows when point predictions are not enough for reporting, coverage analysis, or online adaptation.</p>
  </div>
  <div class="path-card">
    <p class="route-kicker">Path 06</p>
    <h3><a href="foretools/">Open the companion toolbox</a></h3>
    <p>Explore synthetic data, BOHB, VMD decomposition, and feature-engineering utilities documented in the same site.</p>
  </div>
</div>

## Recommended reading order

<div class="step-grid">
  <div class="step-card">
    <strong>1. Confirm the baseline path</strong>
    <span>Start with <a href="getting-started/">Getting Started</a> and make sure one small training loop runs end to end.</span>
  </div>
  <div class="step-card">
    <strong>2. Learn the mental model</strong>
    <span>Read <a href="overview/">Overview</a> to understand how <code>foreblocks</code>, extras, and <code>foretools</code> fit together.</span>
  </div>
  <div class="step-card">
    <strong>3. Stay on the public surface first</strong>
    <span>Use <a href="reference/public-api/">Public API</a> before reaching for subsystem-level imports.</span>
  </div>
  <div class="step-card">
    <strong>4. Branch by workflow</strong>
    <span>Open the guide that matches your next step: preprocessing, transformers, DARTS, uncertainty, or tooling.</span>
  </div>
</div>

## Install by intent

| Need | Suggested install | Best next page |
| --- | --- | --- |
| Core forecasting only | `pip install foreblocks` | [Getting Started](getting-started.md) |
| Raw-series preprocessing | `pip install "foreblocks[preprocessing]"` | [Preprocessor Guide](preprocessor.md) |
| DARTS training, search, and analysis | `pip install "foreblocks[darts]"` | [DARTS Guide](darts.md) |
| Tracking UI and API clients | `pip install "foreblocks[mltracker]"` | [Web UI](webui.md) |
| Everything | `pip install "foreblocks[all]"` | [Overview](overview.md) |

## Documentation map

### Tutorials

Step-by-step runnable paths:

- [Getting Started](getting-started.md)
- [Train A Direct Model](tutorials/train-direct-model.md)
- [Run A DARTS Search](tutorials/darts-multifidelity-search.md)
- [Generate Synthetic Series](tutorials/generate-synthetic-series.md)
- [Optimize With BOHB](tutorials/optimize-with-bohb.md)

### Guides

Topic-based deep dives:

- [Preprocessor](preprocessor.md)
- [Custom Blocks](custom_blocks.md)
- [Transformer](transformer.md)
- [Mixture of Experts](moe.md)
- [Hybrid Mamba](hybrid-mamba.md)
- [DARTS](darts.md)
- [Evaluation & Metrics](evaluation.md)
- [Uncertainty Quantification](uncertainty.md)
- [Web UI](webui.md)
- [Troubleshooting](troubleshooting.md)

### Foretools

Companion utilities and helpers:

- [Foretools Overview](foretools/index.md)
- [Time Series Generator](foretools/tsgen.md)
- [BOHB Search](foretools/bohb.md)
- [VMD Decomposition](foretools/vmd.md)
- [AutoDA Augmentation](foretools/tsaug.md)
- [Feature Engineering](foretools/feature-engineering.md)

### Architecture

Internals and system notes:

- [System Overview](architecture/system-overview.md)
- [Forecasting Pipeline](architecture/forecasting-pipeline.md)
- [DARTS Search Pipeline](architecture/darts-pipeline.md)

## Stable public entry points

Start here before using deep imports:

### Core

```python
from foreblocks import (
    ForecastingModel,
    Trainer,
    ModelEvaluator,
    TimeSeriesHandler,
    TimeSeriesDataset,
    create_dataloaders,
    ModelConfig,
    TrainingConfig,
)
```

### Architecture search

```python
from foreblocks.darts import (
    DARTSTrainer,
    DARTSConfig,
    DARTSTrainConfig,
    FinalTrainConfig,
    MultiFildelitySearchConfig,
)
```

### Uncertainty

```python
from foreblocks.core import ConformalPredictionEngine
```

### Hybrid Mamba

```python
from foreblocks.hybrid_mamba import (
    HybridMambaBlock,
    HybridMamba2Block,
    TinyHybridMamba2LM,
)
```

<div class="docs-callout">
  The docs now mirror the package split more intentionally: stable forecasting first,
  optional extras second, and specialist subsystems framed as branches instead of default prerequisites.
</div>

## Notes

- The canonical docs source lives in `docs/`.
- The published versioned docs live under `/docs/`.
- The site root landing page lives in `web/`.
- `foretools` is documented here too, even when utilities are not re-exported from top-level `foreblocks`.
