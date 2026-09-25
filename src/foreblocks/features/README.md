# Features

Reusable feature transforms for time-series classification, anomaly detection
and regression. Every transform follows the scikit-learn transformer API
(`fit`, `transform`, `fit_transform`, `get_params`), so it composes with
`sklearn.pipeline.make_pipeline` and `sklearn.base.clone`.

Requires the `scientific` extra (numba, scipy, scikit-learn); `SignalFeatures`
also needs PyWavelets (`wavelets` extra).

## Package layout

```text
features/
  _validation.py        # [N, L] / [N, C, L] input contract, normalize_series
  signal.py             # SignalFeatures: engineered amplitude/spectral/wavelet/pulse features
  rocket/
    rocket.py           # Rocket
    minirocket.py       # MiniRocket (+ dilation and bias fitting shared with MultiRocket)
    multirocket.py      # MultiRocket
    fused.py            # FusedRocket (Numba/Triton fused conv + pooling)
    hydra.py            # Hydra, SparseScaler
    selfrocket.py       # SelfRocket (learned representation + pooling operator)
    pruning.py          # pocket_select, srocket_select (kernel pruning)
    classifier.py       # RocketClassifier, make_rocket_transform
    _kernels.py         # Numba kernels for Rocket/MiniRocket/MultiRocket
    _fused_kernels.py   # Numba/Triton kernels for FusedRocket
```

## Rocket family

| Transform | Kernels | Pooling | Output size | Input |
|---|---|---|---|---|
| `Rocket` | random length {7, 9, 11}, Normal weights, random dilation/padding | PPV, max | `2 * num_kernels` | `[N, L]`, `[N, C, L]` |
| `MiniRocket` | 84 fixed kernels, biases fitted from training data | PPV | `84 * (num_features // 84)` | `[N, L]`, `[N, C, L]`; `L >= 9` |
| `MultiRocket` | MiniRocket kernels on the series and its first difference | PPV, MPV, MIPV, LSPV | `8 * 84 * (num_features // 8 // 84)` | `[N, L]`, `[N, C, L]`; `L >= 10` |
| `FusedRocket` | length-9 ROCKET kernels, fused conv + pooling on CPU (Numba) or CUDA (Triton) | PPV, MPV on series and difference | `4 * num_kernels` | `[N, L]` |
| `Hydra` | `g` groups of `k` competing length-9 kernels on the series and difference, power-of-two dilations | per group and time step: arg-max kernel gains its response, arg-min kernel gains 1 | `num_dilations * 2 * g * k` | `[N, L]`, `[N, C, L]` |
| `SelfRocket` | MiniRocket kernels on the series and difference; supervised `fit(X, y)` picks one representation and operator | one of PPV, GMP, MPV, MIPV, LSPV | `84 * (num_features // 84)`, doubled for MIX | `[N, L]`, `[N, C, L]`; `L >= 10` |

Multivariate kernels sum over a random channel subset per kernel (per group for
`Hydra`), as in the reference implementations. `Rocket`, `FusedRocket` and
`Hydra` are data-independent; `MiniRocket` and `MultiRocket` fit biases from
training examples; `SelfRocket` also uses the labels.

- **Hydra** (Dempster et al., 2023) counts which kernel of each group wins, a
  dictionary-style method on top of random convolutions. Its sparse count
  features are scaled with `SparseScaler` (sqrt, then masked standardization),
  which `RocketClassifier` picks automatically.
- **SelfRocket** (Lo et al., 2024, "SelF-Rocket") scores all 15 combinations of
  representation (BASE, DIFF, MIX) and pooling operator (PPV, GMP, MPV, MIPV,
  LSPV) with ridge voters on repeated stratified splits of the training set.
  The best median accuracy wins if at least `vote_threshold` of the voters
  rank it in their top `top`; otherwise `default` (`("mix", "ppv")`, our
  choice) is used. `selection_` and `scores_` expose the result.

```python
from foreblocks.features import MiniRocket, RocketClassifier

model = RocketClassifier(transformer="multirocket", transformer_params={"seed": 0})
model.fit(x_train, y_train)          # x: [N, L] or [N, C, L]
model.predict(x_test)

features = MiniRocket(seed=0).fit(x_train).transform(x_test)
```

`RocketClassifier` z-scores each series (`normalize=True`), applies the
transform(s), scales each branch, and fits `RidgeClassifierCV`. `transformer`
may be a list, or `"multirocket-hydra"` for the MultiRocket + Hydra
concatenation. `transformer_params` is either one dict for every named
branch, or a dict per name:

```python
RocketClassifier(
    "multirocket-hydra",
    transformer_params={"multirocket": {"seed": 0}, "hydra": {"seed": 0}},
)
```
`predict_proba` is a softmax over ridge scores, not a calibrated probability.

## Kernel pruning

`RocketClassifier(pruning="pocket" | "s-rocket", prune_keep=0.4)` keeps a
fraction (or count) of the kernels of a single `Rocket`, `MiniRocket` or
`MultiRocket` transform. It then refits the scaler and ridge on the surviving
features. Pruned kernels are not computed at predict time: each transform
exposes `kernel_groups_` (the kernel id of every feature column) and
`select_kernels(ids)`, which returns a smaller fitted copy.

- `pocket_select` — POCKET (Chen et al., 2024), stage 1: group-lasso
  least-squares on ±1 class indicators (FISTA). The soft threshold is re-chosen
  every iteration so exactly `keep` kernel groups stay non-zero. Stage 2 (ridge
  refit) is the classifier fit.
- `srocket_select` — S-ROCKET (Salehinejad et al., 2022): a genetic search over
  binary kernel masks. Each mask is scored by the held-out accuracy of ridge
  heads trained once on all kernels, with pruned kernels' score contributions
  removed. This implementation differs from the paper in two ways:
  - The active-kernel count is fixed rather than traded off in the fitness.
  - Fitness is averaged over `n_folds` folds, because a single small
    validation split was easy to overfit.

Deviation from the reference MultiRocket code: MPV is the mean of positive
`conv - bias` values, as the paper defines it; the reference code accumulates
`conv + bias`.

## Engineered features

`SignalFeatures(fs, bands=None, ...)` computes, per series (and per channel for
`[N, C, L]` input): log RMS, log peak, crest factor, kurtosis, skewness;
Welch spectral centroid, bandwidth, entropy, flatness, roll-off and dominant
frequency; band-power fractions; wavelet entropy and per-level energy; and pulse
rate, occupancy and inter-pulse gap CV. `bands` defaults to four equal-width
bands spanning 0 to Nyquist. `get_feature_names_out()` names every column.

The discharge baselines (`foreblocks.models.discharge`) are thin wrappers over
these modules.
