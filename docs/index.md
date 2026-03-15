# foreBlocks Docs

<div class="hero-copy">
  <span class="eyebrow">Versioned Docs</span>
  <h1>Forecasting, preprocessing, search, and tooling in one repo</h1>
  <p class="hero-lead">
    <code>foreblocks</code> is the forecasting library. <code>foretools</code> is the
    companion toolbox. The docs site is organized so you can start from the stable
    training path, branch into preprocessing or transformers, and only then move into
    heavier workflows like DARTS search, VMD, or experiment tracking.
  </p>
</div>

<div class="metric-strip">
  <div class="metric-chip"><strong>Core install</strong><span>Minimal modeling stack with optional extras layered on top.</span></div>
  <div class="metric-chip"><strong>Stable path</strong><span><code>ForecastingModel</code> + <code>Trainer</code> + <code>ModelEvaluator</code>.</span></div>
  <div class="metric-chip"><strong>Search stack</strong><span>DARTS supports zero-cost screening, bilevel search, and final retraining.</span></div>
  <div class="metric-chip"><strong>Docs layout</strong><span>Tutorials for runnable paths, guides for subsystems, architecture notes for internals.</span></div>
</div>

## Start here

If you are new to the project, this is the safest reading order:

1. [Overview](overview.md)
2. [Getting Started](getting-started.md)
3. [Public API](reference/public-api.md)
4. The subsystem guide that matches your workflow

<div class="doc-grid">
  <div class="doc-card">
    <h3><a href="getting-started.md">Train a baseline first</a></h3>
    <p>Use the smallest reliable path through <code>ForecastingModel</code>, <code>Trainer</code>, and NumPy-backed dataloaders.</p>
  </div>
  <div class="doc-card">
    <h3><a href="preprocessor.md">Start from raw series</a></h3>
    <p>Use <code>TimeSeriesHandler</code> when you need scaling, filtering, imputation, and window generation from a <code>[T, D]</code> array.</p>
  </div>
  <div class="doc-card">
    <h3><a href="transformer.md">Go deeper on model blocks</a></h3>
    <p>Transformer, attention, patching, and MoE guides cover the more configurable internals.</p>
  </div>
  <div class="doc-card">
    <h3><a href="darts.md">Search architectures</a></h3>
    <p>The DARTS docs explain the staged NAS pipeline, zero-cost ranking, and the search-result analysis workflow.</p>
  </div>
</div>

## Documentation map

### Tutorials

- [Getting Started](getting-started.md)
- [Train A Direct Model](tutorials/train-direct-model.md)
- [Run A DARTS Search](tutorials/darts-multifidelity-search.md)
- [Generate Synthetic Series](tutorials/generate-synthetic-series.md)
- [Optimize With BOHB](tutorials/optimize-with-bohb.md)

### Guides

- [Preprocessor Guide](preprocessor.md)
- [Custom Blocks Guide](custom_blocks.md)
- [Transformer Guide](transformer.md)
- [MoE Guide](moe.md)
- [DARTS Guide](darts.md)
- [Troubleshooting](troubleshooting.md)

### Architecture notes

- [System Overview](architecture/system-overview.md)
- [Forecasting Pipeline](architecture/forecasting-pipeline.md)
- [DARTS Search Pipeline](architecture/darts-pipeline.md)

### Companion tooling

- [Foretools Overview](foretools/index.md)
- [Time Series Generator](foretools/tsgen.md)
- [BOHB Search](foretools/bohb.md)
- [VMD Decomposition](foretools/vmd.md)

## Stable public entry points

Start from the top-level exports before you reach for deep imports:

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

For architecture search, the main public surface is:

```python
from foreblocks.darts import (
    DARTSTrainer,
    DARTSConfig,
    DARTSTrainConfig,
    FinalTrainConfig,
    MultiFildelitySearchConfig,
)
```

<div class="docs-callout">
  The docs now mirror the packaging split: the default install stays lean, while preprocessing,
  DARTS analysis, MLTracker, VMD, benchmarking, and other heavier workflows are documented with
  the exact extras that enable them.
</div>

## Notes

- The canonical docs source lives in `docs/`.
- The published versioned docs live under `/docs/`.
- The static landing page at site root still lives in `web/`.
- `foretools` is documented here too, even when those utilities are not re-exported from top-level `foreblocks`.
