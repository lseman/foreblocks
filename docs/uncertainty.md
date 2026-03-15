# Uncertainty Quantification

`foreblocks.core.ConformalPredictionEngine` provides post-hoc prediction intervals for any trained `nn.Module`. It requires no model retraining — calibration uses a held-out set after training is complete.

## Concepts

Conformal prediction is a distribution-free framework that produces intervals with a **guaranteed marginal coverage** at a target confidence level. Given a calibration split of size `N`, the interval at level `1 − α` contains the true value with probability at least `1 − α`, regardless of the true data distribution.

The engine supports ten methods ranging from basic split conformal to online adaptive variants:

| Method | Type | Use case |
|---|---|---|
| `split` | Static, global | Baseline; fast, assumes stationarity |
| `local` | Static, adaptive | Non-stationary input distribution; uses KNN weighting |
| `jackknife` | Static, CV+ | Tighter intervals via leave-one-out; requires CV models |
| `quantile` | Static, CQR | Model outputs explicit quantiles; very efficient |
| `tsp` | Static, temporal | Temporal partition with exponential recency weighting |
| `rolling` | Online, ACI | Non-stationary targets; adapts α per step |
| `agaci` | Online, ACI | Aggregated experts over multiple learning rates |
| `enbpi` | Online, ensemble | Ensemble batch prediction intervals |
| `cptc` | Online, covariate-shift | State-conditioned conformal for covariate drift |
| `afocp` | Online, attention | Attention-weighted feature-based online conformal |

## Quick start

```python
import numpy as np
from foreblocks.core import ConformalPredictionEngine

# 1. Split your data: train / calibration / test
# X_cal, y_cal: held-out calibration split (never seen during training)
# X_test, y_test: evaluation split

engine = ConformalPredictionEngine(
    method="split",
    quantile=0.9,   # 90% coverage target
)

# 2. Calibrate: computes conformal scores on calibration residuals
engine.calibrate(model, X_cal, y_cal, device="cuda")

# 3. Predict: returns point predictions + lower/upper bounds
preds, lower, upper = engine.predict(model, X_test, device="cuda")
# preds, lower, upper: numpy arrays, shape [N, H, D]
```

## Online update

Adaptive methods (`rolling`, `agaci`, `enbpi`, `cptc`, `afocp`) can be updated incrementally as new labelled points arrive, without recalibrating from scratch:

```python
engine = ConformalPredictionEngine(method="rolling", quantile=0.9, aci_gamma=0.01)
engine.calibrate(model, X_cal, y_cal, device="cuda")

# During deployment, after each batch of new observations:
engine.update(model, X_new, y_new, device="cuda")
preds, lower, upper = engine.predict(model, X_next, device="cuda")
```

`update()` for `rolling` and `agaci` defaults to sequential (point-by-point) update to maintain exact ACI guarantees. Pass `sequential=False` for a faster batch approximation.

## Method reference

### `split`

Global radius: the `q`-quantile of absolute calibration residuals. Cheapest method; assumes exchangeability.

```python
engine = ConformalPredictionEngine(method="split", quantile=0.9)
```

### `local`

Per-sample radius estimated via KNN in feature space. More expressive than split conformal when the input distribution is non-stationary.

```python
engine = ConformalPredictionEngine(
    method="local",
    quantile=0.9,
    knn_k=50,
    local_window=5000,   # max calibration samples to keep
)
```

### `jackknife` (CV+)

Proper Jackknife+ / CV+ intervals using leave-one-out residuals. Provides tighter valid intervals than split conformal. Requires CV models at both calibration and predict time.

```python
from sklearn.model_selection import KFold
# Train cv_models: list of models, each trained on a different fold
engine = ConformalPredictionEngine(method="jackknife", quantile=0.9)
engine.calibrate(
    model, X_cal, y_cal,
    jackknife_cv_models=cv_models,
    jackknife_cv_indices=cv_indices,
)
preds, lower, upper = engine.predict(model, X_test)
```

### `quantile` (CQR)

Conformalized Quantile Regression. The model must output two quantile predictions per step: `[lower_q, upper_q]` as the last dimension.

```python
engine = ConformalPredictionEngine(method="quantile", quantile=0.9)
# model output shape must be [N, H, 2]
engine.calibrate(quantile_model, X_cal, y_cal)
```

### `rolling` (ACI)

Adaptive Conformal Inference with a single learning rate. The coverage level α is updated step-by-step to track temporal drift.

```python
engine = ConformalPredictionEngine(
    method="rolling",
    quantile=0.9,
    rolling_alpha=0.05,
    aci_gamma=0.01,     # learning rate for α updates
)
```

### `agaci`

Aggregated ACI: maintains an expert pool with multiple γ values and aggregates their intervals. More robust than a single learning rate.

```python
engine = ConformalPredictionEngine(
    method="agaci",
    quantile=0.9,
    agaci_gammas=[0.001, 0.005, 0.01, 0.05, 0.1],
)
```

### `tsp`

Temporal partition with exponential recency weighting. Older calibration residuals are discounted.

```python
engine = ConformalPredictionEngine(
    method="tsp",
    quantile=0.9,
    tsp_lambda=0.01,    # exponential decay rate
    tsp_window=5000,
)
```

### `enbpi`

Ensemble Batch Prediction Intervals. Calibrate with out-of-bag residuals from an ensemble; online update uses the same ensemble.

```python
engine = ConformalPredictionEngine(
    method="enbpi",
    quantile=0.9,
    enbpi_B=20,
    enbpi_window=500,
)
engine.calibrate(
    model, X_cal, y_cal,
    enbpi_member_models=ensemble_members,
    enbpi_boot_indices=boot_indices,   # [B, N] bootstrap index array
)
```

### `cptc`

Conformal Prediction under Temporal Covariate shift. Weights calibration residuals by how closely the current regime matches past regimes, via a user-supplied `state_model`.

```python
engine = ConformalPredictionEngine(
    method="cptc",
    quantile=0.9,
    cptc_window=500,
    cptc_tau=1.0,
)
engine.calibrate(model, X_cal, y_cal, state_model=my_state_fn)
```

`state_model` is any callable that maps `X → state_features` (array/tensor).

### `afocp`

Attention-based Feature-weighted Online Conformal Prediction. Learns a weighting over calibration residuals using pairwise attention on input features. Optionally updates the attention network online.

```python
engine = ConformalPredictionEngine(
    method="afocp",
    quantile=0.9,
    afocp_feature_dim=128,
    afocp_attn_hidden=64,
    afocp_window=500,
    afocp_online_lr=1e-4,    # 0.0 disables online learning
    afocp_online_steps=1,
)
engine.calibrate(model, X_cal, y_cal, feature_extractor=feat_fn)
```

## Persistence

```python
engine.save("engine.pkl")
engine2 = ConformalPredictionEngine(method="split")
engine2.load("engine.pkl")
```

## Checking coverage

After collecting predictions on a labelled test set:

```python
in_interval = (y_test >= lower) & (y_test <= upper)
empirical_coverage = in_interval.float().mean().item()
print(f"Empirical coverage: {empirical_coverage:.3f}")
```

For a well-calibrated engine on exchangeable data, this should be ≥ `quantile`.
