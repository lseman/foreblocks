# Architecture

This document maps the repository as it actually exists on disk. It exists because
`README.md`'s "Repository map" table had drifted from the real layout (paths like
`foreblocks/transformer`, `foreblocks/kan`, `foreblocks/mamba`, `foreblocks/blocks`
don't exist — see "Corrections to README.md" at the bottom). If you move a package,
update this file in the same change.

## Packages under `src/`

`src/` holds more than the `foreblocks` package. Four things are real, importable
Python packages; two are unrelated side-projects that happen to live at the repo
root, not under `src/`.

| Package | Path | Packaged in `pyproject.toml`? | What it is |
| --- | --- | --- | --- |
| `foreblocks` | `src/foreblocks/` | Yes | The main library — forecasting models, training, preprocessing, attention/sequence backbones, kernels. Everything below this line describes its internals. |
| `darts` | `src/darts/` | Yes | Standalone DARTS-style differentiable NAS for time-series forecasting. Imported as `import darts`, not `foreblocks.darts`. |
| `foretools` | `src/foretools/` | Yes | Companion utilities: synthetic data generation, feature engineering, decomposition, BOHB hyperparameter search. Imported as `import foretools`. |
| `mltracker` | `src/mltracker/` | **No** | Experiment tracking (local SQLite + optional remote API sync). Has a real `__init__.py` but is not in `pyproject.toml`'s `packages.find.include` — not currently distributed in the wheel despite living under `src/`. |
| `tree` (ForeTree) | `src/tree/` | N/A | C++23 tree-model library (histogram splitting, CUDA) with planned nanobind Python bindings. No `__init__.py` — not a Python package today. |

Not under `src/`, not part of the `foreblocks` distribution, and not referenced by
the root README: `scheduling/` (independent side-project, has its own `.venv` and
`README.md`) and `cubinho/` (image/dataset assets plus a vendored third-party
`sd-scripts` checkout) at the repo root. Don't treat either as part of this
library's architecture.

## `src/foreblocks/` top-level layout

```
foreblocks/
├── core/               ForecastingModel base class, attention-layer abstraction, training/eval/quantization
│   ├── training/          Trainer — the main training-loop entry point (NAS hooks, conformal, MLTracker, checkpointing)
│   ├── evaluation/         ModelEvaluator — metrics, cross-validation, prediction helpers
│   └── quantization/        Model quantization configs/modules
├── models/              High-level model assembly
│   ├── transformer/        Modular transformer encoder/decoder stack — see models/transformer/README.md
│   ├── kan/                 Kolmogorov-Arnold Network forecasting models (patch-based, multiple polynomial bases)
│   └── popular/              Published architectures: DLinear, Informer, Autoformer, PatchTST, N-BEATS/N-HiTS, FEDformer, ETSformer, TimesNet, CrossFormer, TimeMixer, TimeXer, TFT, Oryx, ...
├── modules/             Composable building blocks shared across models
│   ├── attention/          Multi-backend attention (dense/sparse/linear/spectral) — see modules/attention/README.md
│   ├── moe/                 Mixture-of-Experts feed-forward layers
│   ├── skip/                 GateSkip / Mixture-of-Depths routing, layer dropout
│   ├── heads/                  Forecasting head composition/projection
│   └── blocks/                  Lower-level reusable blocks: recurrent, spectral, graph, ODE, normalization
├── sequence/            Stateful sequence-model backbones
│   ├── mamba/              Mamba2/Mamba3 SSM blocks (Triton-accelerated)
│   └── raven/                FLA-backed gated sparse attention (GSA) sequence blocks
├── ops/                 Triton/CUDA kernels — the performance layer everything above calls into
│   ├── attention/          FLA integration, fused RoPE, fused RMSNorm-gate, paged decode
│   ├── mamba/               Causal conv1d, fused Δt, chunked SSD scan
│   ├── kernels/              Generic kernels: layer norm, RMS norm, grouped GEMM, SwiGLU, softmax, GELU
│   ├── graph/                 Graph message-passing kernels
│   └── raven/                  Lazy proxy over upstream FLA symbols
├── layers/              Lower-level nn.Module primitives: embeddings (RoPE/ALiBi/time), norms, graph convolutions
├── anomaly/             Anomaly detection: reconstruction/forecasting/representation-based detectors (TranAD, OmniAnomaly, DAGMM, AnomalyTransformer, PatchTST-based, diffusion)
├── ts_handler/           TimeSeriesHandler — preprocessing, filtering, imputation, outlier handling
│   ├── auto_filter/         Optuna-based automatic filter selection/tuning
│   └── filters/               Filter implementations: Savitzky-Golay, Kalman, LOWESS, Wiener, EMD, SSA, STL
├── data/                Dataset/DataLoader construction, windowing, normalization, splitting
├── ui/                  Studio UI support: node metadata, auto-spec inference, model discovery
├── experimental/         Not-yet-stable code
│   └── attention_kernels/   Vendored "custom_att" Triton exact-online-softmax attention scaffold (own egg-info/src, not a normal subpackage)
├── third_party/          Vendored fallback implementations (flash-softpick-attn, variational SGD)
├── config.py             Library-wide configuration
└── studio_server.py       Studio web server entry point
```

### Dependency direction

`ops/` is the lowest layer — raw Triton/CUDA kernels with no knowledge of models.
`layers/` and `modules/` build `nn.Module`s on top of `ops/`. `sequence/` and
`models/` assemble those into full backbones. `core/` (training/evaluation) and
`ts_handler/` operate on top of whatever model you built. `anomaly/` and
`models/popular/` are consumers of `modules/` and `models/transformer/`, not
dependencies of them.

Two subpackages have their own `README.md` with dependency-direction and
naming-convention detail that this file doesn't duplicate:
- [`models/transformer/README.md`](src/foreblocks/models/transformer/README.md)
- [`modules/attention/README.md`](src/foreblocks/modules/attention/README.md)

If you write a README for another subpackage, link it here rather than inlining
its structure into this file — this file should stay a map, not the territory.

## Public API surface

`src/foreblocks/__init__.py` lazy-loads (`__getattr__` + `TYPE_CHECKING`) a small,
deliberately stable `__all__`:

```python
__all__ = [
    "ForecastingModel", "GraphForecastingModel",
    "Trainer", "ModelEvaluator",
    "TimeSeriesHandler", "TimeSeriesDataset", "create_dataloaders",
    "ModelConfig", "TrainingConfig",
    "LSTMEncoder", "LSTMDecoder", "GRUEncoder", "GRUDecoder",
    "TransformerEncoder", "TransformerDecoder", "ModernTransformerTuner",
    "AttentionLayer",
]
```

Resolution targets (where each name actually lives — useful when the name alone
doesn't tell you the subpackage):

| Export | Resolves to |
| --- | --- |
| `AttentionLayer` | `foreblocks.core.att` |
| `ForecastingModel`, `GraphForecastingModel` | `foreblocks.models` |
| `Trainer` | `foreblocks.core.training` |
| `ModelEvaluator` | `foreblocks.core.evaluation` |
| `TimeSeriesHandler` | `foreblocks.ts_handler` |
| `TimeSeriesDataset`, `create_dataloaders` | `foreblocks.data` |
| `ModelConfig`, `TrainingConfig` | `foreblocks.config` |
| `LSTMEncoder`, `LSTMDecoder`, `GRUEncoder`, `GRUDecoder` | `foreblocks.modules.blocks.enc_dec` |
| `TransformerEncoder` | `foreblocks.models.transformer.core.encoder` |
| `TransformerDecoder` | `foreblocks.models.transformer.core.decoder` |
| `ModernTransformerTuner` | `foreblocks.models.transformer.tuner` |

Anything not in this list is an internal import path and can move without a
deprecation cycle. Anything in this list moving is a breaking change.

## Corrections to `README.md`

The root README's "Repository map" table describes a layout that doesn't match
the filesystem. Until that table is rewritten, treat this file as authoritative
for structure and the README as authoritative for install/quickstart. Specific
drift found:

| README claims | Reality |
| --- | --- |
| `foreblocks/training` | `foreblocks/core/training/` |
| `foreblocks/evaluation` | `foreblocks/core/evaluation/` |
| `foreblocks/transformer` | `foreblocks/models/transformer/` |
| `foreblocks/kan` | `foreblocks/models/kan/` |
| `foreblocks/mamba` | `foreblocks/sequence/mamba/` (blocks) + `foreblocks/ops/mamba/` (kernels) — no single `mamba` dir |
| `foreblocks/custom_mamba` | Does not exist under this name; same split as above |
| `foreblocks/custom_raven` | `foreblocks/sequence/raven/` + `foreblocks/ops/raven/` |
| `foreblocks/custom_att` | `foreblocks/experimental/attention_kernels/` (vendored, importable as `custom_att`) |
| `foreblocks/blocks` | `foreblocks/modules/blocks/` |
| `foreblocks/layers` | Exists, but is broader than "graph-focused primitives" — also holds embeddings and norms |
