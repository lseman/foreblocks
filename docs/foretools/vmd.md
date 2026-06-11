---
title: VMD Decomposition
description: Variational Mode Decomposition, EMD variants, hierarchical decomposition, and parameter search.
editLink: true
---


[[toc]]
# VMD Decomposition

`foretools/emd_like` provides a decomposition toolkit covering Variational Mode Decomposition (VMD), empirical mode decomposition variants, hierarchical decomposition, multivariate support, and Optuna-based parameter search.

Use it when you want to split a signal into interpretable oscillatory modes before forecasting, diagnostics, denoising, or downstream feature extraction.

## Installation

The VMD module requires the `vmd` extra:

```bash
pip install "foreblocks[vmd]"
```python

Other useful exports:

- `VMDOptimizer` for the lower-level optimization pipeline
- `SignalAnalyzer` for auto-parameter heuristics
- `ModeProcessor` for post-processing and mode ordering
- `VMDCore` for the low-level solver

## Fast path

`FastVMD` is the main entry point.

```python
import numpy as np
from foretools.emd_like import FastVMD

fs = 100.0
t = np.arange(0, 10, 1 / fs)
signal = (
    np.sin(2 * np.pi * 3.0 * t)
    + 0.6 * np.sin(2 * np.pi * 11.0 * t)
    + 0.15 * np.random.default_rng(0).normal(size=t.shape[0])
)

vmd = FastVMD()
raw_modes, raw_freqs, optinfo = vmd.decompose(
    signal,
    fs=fs,
    method="standard",
    auto_params=True,
    refine_modes=False,
    return_raw_modes=True,
)

post_modes = optinfo["post_modes"]
post_freqs = optinfo["post_freqs"]
best_K, best_alpha, best_cost = optinfo["best"]
```toml

## Common controls

`FastVMD.decompose(...)` forwards most tuning arguments into `VMDParameters`.

Common overrides:

| Argument | Why you would change it |
| --- | --- |
| `auto_params` | disable heuristics and use explicit parameter overrides |
| `n_trials` | control the Optuna search budget |
| `max_K` | cap the number of modes considered |
| `alpha_min`, `alpha_max` | narrow or widen the bandwidth-penalty search range |
| `k_selection` | choose `penalized`, `fbd`, or `entropy` preselection logic |
| `search_method` | choose `optuna` or `entropy` for the search path |
| `boundary_method` | control edge extension such as `mirror`, `reflect`, `linear`, `constant`, or `none` |
| `apply_tapering` | taper mode boundaries before returning them |
| `refine_modes` | enable neural post-refinement |
| `refine_method` | switch between `informer` and `cross_mode` refinement |
| `fft_backend`, `fft_device` | choose FFT backend and device routing |

Example with explicit overrides:

```python
modes, freqs, best = vmd.decompose(
    signal,
    fs=fs,
    return_raw_modes=False,
    auto_params=False,
    n_trials=20,
    max_K=5,
    alpha_min=800,
    alpha_max=4000,
    k_selection="penalized",
    boundary_method="mirror",
    apply_tapering=True,
)
```python

Important parameter groups:

- Search: `n_trials`, `max_K`, `search_method`, `k_selection`
- Solver: `tol`, `max_iter`, `tau`, `DC`, `init`
- Boundary handling: `boundary_method`, `use_soft_junction`, `window_alpha`
- Stabilization: `admm_over_relax`, `omega_momentum`, `omega_shrinkage`, `omega_max_step`
- Advanced bandwidth handling: `adaptive_alpha`, related `adaptive_alpha_*` fields
- Backend selection: `fft_backend`, `fft_device`
- Special modes: `if_tracking`, `use_mvmd`

## Automatic parameter selection

With `auto_params=True`, the pipeline uses `SignalAnalyzer.assess_complexity(...)` to pick a search budget and alpha range from spectral entropy, frequency spread, variability, kurtosis, and an SNR estimate.

This is a good first step when:

- you do not know a reasonable `K` range yet
- signals vary a lot across datasets
- you want a usable baseline before manual tuning

Prefer `auto_params=False` when you are doing ablations or want strict reproducibility of the search space.

## Hierarchical decomposition

Use `method="hierarchical"` when you want multi-scale decomposition across progressively downsampled residuals.

```python
vmd = FastVMD()

modes, freqs, level_info = vmd.decompose(
    signal,
    fs=fs,
    method="hierarchical",
    max_levels=3,
    energy_threshold=0.01,
    min_samples_per_level=100,
    use_emd_hybrid=False,
    refine_modes=False,
)
```text

Notes:

- MVMD expects a 2D array shaped `[channels, samples]`
- the raw return is flattened for convenience, while `optinfo["mv_shape"]` preserves the original multivariate mode layout
- entropy-based search currently falls back to Optuna for MVMD in this pipeline

## Low-level optimizer

If you want more control than `FastVMD`, use `VMDOptimizer` directly.

```python
from foretools.emd_like import FFTWManager, VMDOptimizer

fftw = FFTWManager()
optimizer = VMDOptimizer(fftw)

raw_modes, raw_freqs, optinfo = optimizer.optimize(
    signal,
    fs=fs,
    auto_params=True,
    refine_modes=False,
    return_raw_modes=True,
)
