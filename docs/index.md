# foreBlocks Docs

<div class="hero-copy">
  <span class="eyebrow">v0.1.15 &mdash; Modular Forecasting</span>
  <h1>Forecasting, preprocessing, search, and tooling in one repo</h1>
  <p class="hero-lead">
    <code>foreblocks</code> is the forecasting library. <code>foretools</code> is the
    companion toolbox. Start from the stable training path, branch into preprocessing
    or transformers, and only then move into heavier workflows like DARTS search,
    conformal intervals, or the visual Web UI.
  </p>
</div>

<div class="metric-strip">
  <div class="metric-chip"><strong>Minimal install</strong><span>Core stack with optional extras layered on top.</span></div>
  <div class="metric-chip"><strong>Stable path</strong><span><code>ForecastingModel</code> + <code>Trainer</code> + <code>ModelEvaluator</code>.</span></div>
  <div class="metric-chip"><strong>NAS</strong><span>DARTS with zero-cost screening, bilevel search, and retraining.</span></div>
  <div class="metric-chip"><strong>Uncertainty</strong><span>10 conformal methods — static, adaptive, and attention-based.</span></div>
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
    <h3><a href="transformer.md">Transformer and MoE</a></h3>
    <p>Transformer, attention, patching, and MoE guides cover the more configurable internals.</p>
  </div>
  <div class="doc-card">
    <h3><a href="hybrid-mamba.md">Hybrid Mamba (SSM)</a></h3>
    <p>Pure SSM and hybrid SSM+attention blocks with Triton/CUDA kernel backends.</p>
  </div>
  <div class="doc-card">
    <h3><a href="darts.md">Search architectures</a></h3>
    <p>DARTS staged NAS pipeline: zero-cost ranking, bilevel search, and result analysis.</p>
  </div>
  <div class="doc-card">
    <h3><a href="uncertainty.md">Prediction intervals</a></h3>
    <p>Post-hoc conformal prediction — 10 methods from split conformal to online ACI variants.</p>
  </div>
</div>

## Documentation map

=== "Tutorials"

    Step-by-step runnable examples:

    - [Getting Started](getting-started.md)
    - [Train A Direct Model](tutorials/train-direct-model.md)
    - [Run A DARTS Search](tutorials/darts-multifidelity-search.md)
    - [Generate Synthetic Series](tutorials/generate-synthetic-series.md)
    - [Optimize With BOHB](tutorials/optimize-with-bohb.md)

=== "Guides"

    Subsystem deep dives:

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

=== "Foretools"

    Companion utilities:

    - [Foretools Overview](foretools/index.md)
    - [Time Series Generator](foretools/tsgen.md)
    - [BOHB Search](foretools/bohb.md)
    - [VMD Decomposition](foretools/vmd.md)
    - [AutoDA Augmentation](foretools/tsaug.md)

=== "Architecture"

    Internals and design notes:

    - [System Overview](architecture/system-overview.md)
    - [Forecasting Pipeline](architecture/forecasting-pipeline.md)
    - [DARTS Search Pipeline](architecture/darts-pipeline.md)

## Stable public entry points

Start from the top-level exports before reaching for deep imports:

=== "Core"

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

=== "Architecture search"

    ```python
    from foreblocks.darts import (
        DARTSTrainer,
        DARTSConfig,
        DARTSTrainConfig,
        FinalTrainConfig,
        MultiFildelitySearchConfig,
    )
    ```

=== "Uncertainty"

    ```python
    from foreblocks.core import ConformalPredictionEngine
    ```

=== "Hybrid Mamba"

    ```python
    from foreblocks.hybrid_mamba import (
        HybridMambaBlock,
        HybridMamba2Block,
        TinyHybridMamba2LM,
    )
    ```

<div class="docs-callout">
  The docs mirror the packaging split: the default install stays lean, while preprocessing,
  DARTS, MLTracker, VMD, and other heavier workflows are documented with the exact extras that enable them.
</div>

## Notes

- The canonical docs source lives in `docs/`.
- The published versioned docs live under `/docs/`.
- The static landing page at site root lives in `web/`.
- `foretools` is documented here too, even when utilities are not re-exported from top-level `foreblocks`.
