# Anomaly detection

## Package layout

```text
anomaly/
  config.py           # AnomalyDetectorConfig
  detector.py         # Unified fit/predict orchestration
  blocks.py           # Block protocol, registry, composition, and decisions
  modes.py            # Strategies connecting backbones/scorers to the detector
  backbones/          # Neural architectures and their output types
  scorers/            # Classical, statistical, and empirical NumPy scorers
  tranad_detector.py  # Dedicated TranAD training API
  windows.py          # Anomaly preprocessing, score alignment, thresholds
  calibration.py      # Confidence and score calibration
  online.py           # Streaming and adaptation
```

Import public APIs from `foreblocks.models.anomaly` as before. Internal imports
now use `anomaly.backbones.*` for neural architectures, `anomaly.scorers.*` for
numerical scorers, and `anomaly.blocks` for composition. The old nested
`anomaly.models` package has been removed. The dedicated TranAD wrapper moved
from `anomaly.tranad` to `anomaly.tranad_detector`; `TranAD` itself lives in
`anomaly.backbones.tranad`.

Shared components have one owner:

- `foreblocks.data.windowing` owns NumPy windows, tensor window views, and
  `SlidingWindowDataset`. Existing anomaly exports, including `TranADDataset`,
  re-export these implementations.
- `foreblocks.nn.embeddings.positional_encoding` supplies TranAD's positional
  encoding. A small adapter preserves its odd-width sine convention and saved
  `pe` buffer, so existing model state dictionaries still load strictly.
- `BaseAnomalyBlock` supplies common decision and block-name behavior. Built-in
  modes inherit it instead of being patched at import time.
- `config.py` owns the unified configuration; importing it from `detector.py`
  remains supported.

Backbones and scorers do not depend on the orchestration modules. Add a neural
architecture to `backbones`, a numerical method to `scorers`, and integrate it
through a mode. Shared data helpers must not import the anomaly package.

## Empirical anomaly detectors

ECOD, COPOD, and HBOS are available through the existing time-series pipeline:

```python
from foreblocks.models.anomaly import ForeblocksAnomalyDetector

model = ForeblocksAnomalyDetector(
    model_type="ecod",  # also "copod" or "hbos"
    window_size=32,
    contamination=0.01,
    batch_size=128,
)
model.fit(train_series)  # [time, channels], or a univariate [time] array
result = model.predict(test_series)
# result.scores, result.labels, result.window_scores, result.threshold
```

`detection_mode="auto"` selects classical detection for these models;
`detection_mode="classical"` also works. They fit without gradient training.
The existing scaler, score alignment, and training-derived robust threshold
remain in use. With end alignment, the first `window_size - 1` scores are NaN.
`contamination` controls the threshold quantile; the pipeline also applies its
MAD-based lower bound, so it does not guarantee that fraction of positive labels.

| Model | Scoring method | Parameters |
| --- | --- | --- |
| ECOD | Empirical left/right tail probabilities with skew adjustment | None |
| COPOD | Empirical copula tails, combining average and skew-selected surprisal | None |
| HBOS | Independent equal-width marginal histograms | `hbos_bins=10`, `hbos_alpha=0.1` |

Each window is flattened into lag/channel features. These methods model marginal
rarity, rather than cross-channel dependencies. Use a representative training
period: the reference distribution stays fixed until the next `fit` call.
No PyOD dependency is required by these implementations.

### Direct scoring and explanations

```python
from foreblocks.models.anomaly import ECOD, COPOD, HBOS, ecod_score

scorer = ECOD().fit(train_windows)  # [samples, window_length, channels]
scores = scorer.decision_function(test_windows)  # [samples]
contributions = scorer.feature_scores(test_windows)  # [samples, flattened_features]
# contributions.sum(axis=1) == scores

# Feature matrices [samples, features] are also supported.
scores = ecod_score(test_windows, reference=train_windows)
```

`copod_score` and `hbos_score` provide the same reference argument. Omitting
`reference` fits and scores on the supplied input; for held-out prediction,
provide a training reference or reuse a fitted scorer. Direct scorers reject
nonfinite data and incompatible shapes; the time-series pipeline retains its
existing forward-fill preprocessing.

Training scores are stored in `decision_scores_`. Higher scores indicate greater
anomaly evidence, not calibrated probabilities. Prediction never recomputes
ranks, skew, or histograms from the query batch. Thus scoring a sample alone or
alongside other samples gives the same result.

The empirical scorers use inclusive ranks for ties and floor unseen tail
probabilities at `1 / (n_train + 1)`. ECOD and COPOD implement fixed-reference
prediction, which deliberately differs from transductive implementations that
combine training and query samples before computing ranks. HBOS normalizes bin
densities by their marginal maximum, regularizes with `alpha`, and treats values
outside the training range as zero-density; it does not use boundary tolerance
bands. A constant marginal contributes zero at its training value and positive
anomaly evidence for a changed value, including with one training sample.

References: [ECOD](https://arxiv.org/abs/2201.00382),
[COPOD](https://arxiv.org/abs/2009.09463), and Goldstein & Dengel,
*Histogram-based Outlier Score (HBOS): A Fast Unsupervised Anomaly Detection
Algorithm*, KI 2012.

## Native fitted methods

These implementations live in Foreblocks and use NumPy and PyTorch directly.
They do not wrap external anomaly estimators and require no additional dependency.

| `model_type` | Public scorer | Algorithm / main options |
| --- | --- | --- |
| `inne` | `INNE` | Nearest-neighbor hypersphere ensembles; `n_estimators`, `max_samples` |
| `loda` | `LODA` | Sparse projected histograms; `n_projections`, `n_bins`, `alpha` |
| `knn` | `KNNScorer` | Chunked training-neighbor distances; `n_neighbors`, `method`, `chunk_size` |
| `gmm` | `GaussianMixtureScorer` | Diagonal Gaussian mixture EM; `n_components`, `max_iter`, `reg_covar` |
| `dif` | `DeepIsolationForest` | Random MLP representations and native isolation trees; `n_ensemble`, `n_estimators`, `hidden_sizes`, `representation_dim` |
| `autoencoder` | `AutoEncoderScorer` | Denoising bottleneck MLP; `hidden_sizes`, `latent_dim`, `noise_std` |
| `vae` | `VAEScorer` | Gaussian latent VAE; `hidden_sizes`, `latent_dim`, `beta` |
| `deep_svdd` | `DeepSVDD` | Bias-free deep hypersphere; `objective`, `nu`, `warmup_epochs` |

```python
from foreblocks.models.anomaly import ForeblocksAnomalyDetector, DeepSVDD

model = ForeblocksAnomalyDetector(
    model_type="dif",
    window_size=32,
    scorer_kwargs={"n_ensemble": 10, "n_estimators": 6},
).fit(train_series)
result = model.predict(test_series)

# Neural methods use the same time-series pipeline.
model = ForeblocksAnomalyDetector(
    model_type="deep_svdd",
    window_size=32,
    epochs=30,
    device="cpu",
    scorer_kwargs={"hidden_sizes": (64, 32), "latent_dim": 8},
).fit(train_series)

# Direct APIs accept [samples, features] or [samples, time, channels].
scorer = DeepSVDD(epochs=30, seed=42, device="cpu").fit(train_windows)
scores = scorer.decision_function(test_windows)
training_scores = scorer.decision_scores_
```

`auto` selects `NativeMode`; `detection_mode="native"` also works. Each native
scorer owns its fit lifecycle, and Foreblocks handles series scaling, window
construction, alignment, and its existing training-derived threshold. The
pipeline forwards `seed` to all methods, `batch_size` to DIF and neural methods,
and `epochs`, `learning_rate`, `weight_decay`, and `device` to neural methods.
`scorer_kwargs` overrides these settings and supplies method-specific options.
Invalid constructor arguments fail explicitly. DIF runs on the CPU in NumPy;
neural methods support PyTorch devices.

Direct scorers standardize using training means and scales by default; pass
`standardize=False` to disable this. The unified pipeline disables the extra
standardization because its selected series scaler already preprocesses data.
The native fit uses all supplied training windows. `validation_split`, `patience`,
and mixed precision from the unified trainer do not apply to these self-contained
training loops. Neural optimization uses float32, gradient clipping, and seeded
initialization. Direct native fitting preserves the caller's random generator
state. Scorers can be refitted; all learned state is replaced. Persist the whole
fitted scorer (including scaling), not just the neural weights.

### Algorithm choices and limits

- INNE uses the smallest enclosing open hypersphere and its local radius ratio.
  Duplicate sampled centers are collapsed; a constant reference is a zero-radius
  sphere that accepts only an exact match.
- LODA uses normalized sparse Gaussian projections with roughly sqrt(features)
  nonzero entries, fixed equal-width bins, and pseudocount smoothing. This omits
  the original adaptive projection/bin-count selection. Outside the training
  support, it uses the empty-bin density. Constant projections use exact matching.
- KNN bounds pairwise-distance memory by `chunk_size` in both query and reference
  dimensions. Its training scores include self-neighbors, matching query semantics;
  these are not leave-one-out estimates. Use at least two neighbors when calibrating
  training scores. `method` can be `largest`, `mean`, or `median`.
- GMM fits diagonal covariances with a variance floor using a custom EM loop.
  It exposes `lower_bounds_`, `converged_`, and `n_iter_`; inspect convergence for
  difficult fits. Scores are negative log densities and can be negative.
- DIF samples independent random MLPs instead of implementing CERE. Custom trees
  combine normalized path length with mean split deviation (DEAS). Representation
  scaling is fitted on training data and frozen at inference. Constant training
  representations produce no splits and therefore zero DEAS evidence.
- AutoEncoder trains on noisy inputs with clean reconstruction targets. Set
  `noise_std=0` for a plain autoencoder.
- VAE samples latents during training but uses their posterior means for
  deterministic reconstruction scores at inference. The training objective sums
  reconstruction error over features and adds `beta` times KL divergence;
  reported scores are reconstruction MSE, not a likelihood estimate.
- Deep SVDD uses a fixed, nonzero training-initialized center and a bias-free
  encoder. `objective="one_class"` minimizes distance to that center;
  `objective="soft_boundary"` uses `nu` and updates a training-distance quantile
  radius after `warmup_epochs`. Scores are squared distances (minus radius squared
  for soft-boundary mode). No autoencoder pretraining is performed.

All query scores use frozen training state and are independent of query batch
composition, within floating-point tolerance. Windows are flattened: these are
lag-feature detectors, while the existing temporal backbones model sequence
structure directly. Scores are anomaly evidence, not calibrated probabilities.
Validate accuracy on chronological holdouts with windows kept within each split;
method availability alone does not establish state-of-the-art performance.

References: [Deep Isolation Forest](https://arxiv.org/abs/2206.06602),
[Deep SVDD](https://proceedings.mlr.press/v80/ruff18a.html),
[iNNE paper](https://www.researchgate.net/publication/322359651_Isolation-based_anomaly_detection_using_nearest-neighbor_ensembles_iNNE),
and Pevny, *Loda: Lightweight on-line detector of anomalies*, Machine Learning,
2016. These are independent implementations with the choices stated above.
