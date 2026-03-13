# VMD Decomposition

`foretools/vmd` provides a decomposition toolkit built around Variational Mode Decomposition (VMD), plus hierarchical decomposition, multivariate support, and Optuna-based parameter search.

Use it when you want to split a signal into interpretable oscillatory modes before forecasting, diagnostics, denoising, or downstream feature extraction.

## Installation

The VMD module requires the `vmd` extra:

```bash
pip install "foreblocks[vmd]"
```

This currently pulls in the FFTW and Optuna dependencies that `foretools.vmd` imports eagerly.

## Import surface

For most users, start here:

```python
from foretools.vmd import FastVMD, HierarchicalParameters, VMDParameters
```

Other useful exports:

- `VMDOptimizer` for the lower-level optimization pipeline
- `SignalAnalyzer` for auto-parameter heuristics
- `ModeProcessor` for post-processing and mode ordering
- `VMDCore` for the low-level solver

## Fast path

`FastVMD` is the main entry point.

```python
import numpy as np
from foretools.vmd import FastVMD

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
```

### What comes back

When `return_raw_modes=True`, `FastVMD.decompose(...)` returns:

- `raw_modes`: NumPy array shaped `[K_raw, N]`
- `raw_freqs`: dominant frequency estimate for each raw mode
- `optinfo`: metadata dictionary with:
  - `best`: `(K, alpha, cost)`
  - `raw_modes`, `raw_freqs`
  - `post_modes`, `post_freqs`

The post-processed modes are usually the ones you want to keep for analysis, because the pipeline may drop low-energy modes, merge nearby frequencies, and sort modes from low to high frequency.

If `return_raw_modes=False`, the return shape is simpler:

```python
modes, freqs, best = vmd.decompose(signal, fs=fs, return_raw_modes=False)
```

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
```

## Parameter objects

Use `VMDParameters` when you want a typed view of the available controls.

```python
from foretools.vmd import VMDParameters

params = VMDParameters(
    n_trials=24,
    max_K=6,
    alpha_min=600,
    alpha_max=5000,
    boundary_method="mirror",
    k_selection="penalized",
    search_method="optuna",
    apply_tapering=True,
)
```

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
```

`level_info` contains per-level metadata such as:

- level number
- downsampling factor
- sampling rate used at that level
- number of modes found
- dominant frequencies
- energy ratio
- computation time
- selected optimization parameters

This path is useful when a single flat decomposition struggles to separate very low-frequency structure from faster local oscillations.

## Multivariate decomposition

The optimizer also supports MVMD-style workflows for signals shaped `[channels, samples]`.

```python
signals = np.stack(
    [
        signal,
        0.8 * signal + 0.1 * np.random.default_rng(1).normal(size=signal.shape[0]),
    ],
    axis=0,
)

raw_modes, raw_freqs, optinfo = vmd.decompose(
    signals,
    fs=fs,
    use_mvmd=True,
    return_raw_modes=True,
    refine_modes=False,
)
```

Notes:

- MVMD expects a 2D array shaped `[channels, samples]`
- the raw return is flattened for convenience, while `optinfo["mv_shape"]` preserves the original multivariate mode layout
- entropy-based search currently falls back to Optuna for MVMD in this pipeline

## Low-level optimizer

If you want more control than `FastVMD`, use `VMDOptimizer` directly.

```python
from foretools.vmd import FFTWManager, VMDOptimizer

fftw = FFTWManager()
optimizer = VMDOptimizer(fftw)

raw_modes, raw_freqs, optinfo = optimizer.optimize(
    signal,
    fs=fs,
    auto_params=True,
    refine_modes=False,
    return_raw_modes=True,
)
```

This is the same core path used by `FastVMD`, just without the convenience wrapper.

## Practical behavior

- The pipeline caches candidate evaluations internally during search, so repeated `(K, alpha)` evaluations are cheaper within one optimizer instance.
- FFTW wisdom is loaded on startup and saved again after decomposition, which can speed up repeated runs.
- Returned post-processed modes may be fewer than the raw `K` because low-energy modes can be removed and nearby modes can be merged.
- `boundary_method` and `window_alpha` interact: the low-level solver treats boundary extension and Tukey windowing as mutually exclusive paths.
- `return_raw_modes=True` is useful for debugging search behavior; `return_raw_modes=False` is cleaner for downstream use.

## Suggested workflow

1. Install the extra with `pip install "foreblocks[vmd]"`.
2. Start with `FastVMD(...).decompose(..., auto_params=True, refine_modes=False)`.
3. Inspect `optinfo["best"]`, `optinfo["post_freqs"]`, and the reconstructed sum of `post_modes`.
4. Only then start tightening `max_K`, `alpha_min`, `alpha_max`, or enabling hierarchical decomposition.

## Related pages

- [Foretools Overview](index.md)
- [Repository Map](../reference/repository-map.md)
- [Troubleshooting](../troubleshooting.md)
