# Acoustic partial-discharge classification

`DischargeClassifier` learns known discharge classes from `[contexts, samples]`
waveforms. A shared multiscale waveform encoder processes each short frame;
a dilated envelope encoder processes the entire context. The fused embedding
supports both classification and a separately fitted novelty reference.

```python
from foreblocks.models.discharge import DischargeClassifier, DischargeClassifierConfig

config = DischargeClassifierConfig(
    fs=256_000,
    context_ms=100,
    frame_ms=10,
    frame_pooling="attention",
    novelty_method="mahalanobis",
    label_smoothing=0.05,
    epochs=30,
    device="cpu",
)
model = DischargeClassifier(config)
model.fit(x_train, y_train, groups=recording_ids, validation_split=0.2)

# Reserve calibration contexts separately from training and model selection.
model.calibrate_probabilities(x_probability_calibration, y_probability_calibration)
model.calibrate_novelty(x_novelty_calibration, quantile=0.95)
result = model.predict(x_test)

# result.predicted_class: known-class labels, even when unfamiliar
# result.class_probabilities: softmax probabilities after temperature scaling
# result.embedding: fused features
# result.novelty_score: minimum distance/evidence across known classes
# result.is_unfamiliar: threshold flag, or None before novelty calibration
# result.novelty_pvalue: conservative known-reference tail rank, or None
```

Inputs must be finite two-dimensional matrices of the configured context length.
Labels are a matching vector with at least two classes. Empty prediction batches
are supported. Probability columns follow `model.classes_`.

## Representation and training

`frame_pooling="mean"` preserves the original frame aggregation architecture.
`"attention"` uses a native depthwise temporal residual convolution followed by
attention-weighted mean and standard deviation. This gives the waveform branch
access to frame order and variation instead of only the frame average. It is
inspired by [attentive statistics pooling](https://www.isca-archive.org/interspeech_2018/okabe18_interspeech.html),
with a local temporal mixer specific to this implementation.

Three explicit log-amplitude/shape features (RMS, peak, crest factor) remain
available through `use_context_features=True`. Their scaling is fitted on the
training subset only. Set this option to `False` for an amplitude-cue ablation;
the waveform and envelope inputs are per-context z-normalized. Normalization
uses float64 intermediate moments to avoid float32 variance overflow.

Training uses class-balanced sampling, AdamW, optional `label_smoothing`, and
`gradient_clip`. Validation uses unsmoothed cross-entropy. The best checkpoint is
restored after early stopping. `history_` contains sample-weighted training loss
and validation loss; `best_epoch_` identifies the selected epoch (zero-based).
Seeded fitting restores the caller's PyTorch random state.

## Recording separation

Split raw recordings **before window construction**, and never let a context
cross a recording or split boundary. Pass one recording/session ID per context
through `groups`. The native split helper holds out whole groups and requires
every known class to remain in training. It searches a bounded set of candidate
splits; the requested fraction and validation class balance are approximate for
scarce or unequal-size groups. An infeasible split raises an error instead of
silently falling back to a row split. Inspect `train_indices_` and
`validation_indices_` to audit the resulting split.

Without groups, splitting is stratified by class at the context level; this
cannot protect against dependence between overlapping contexts. Use
`validation_split=0` to train on all supplied contexts without early stopping.
Novelty references and amplitude-feature scaling use **only training indices**,
even when validation data was provided for checkpoint selection.

## Novelty and probability calibration

Novelty is implemented locally, with no external anomaly estimator:

| Method | Reference / score |
| --- | --- |
| `mahalanobis` | Class means, pooled residual covariance, native OAS shrinkage plus a positive ridge; squared Mahalanobis distance |
| `ecod` | Existing native ECOD fitted separately per class |
| `copod` | Existing native COPOD fitted separately per class |
| `cosine` | Angular distance to each class prototype, ignoring embedding magnitude |

`ClassConditionalNovelty.score_per_class()` returns `[samples, classes]` scores
in `classes_` order. `score()` takes their minimum: an unfamiliar context should
resemble none of the known classes. The default covariance estimator now uses
OAS rather than the prior Ledoit-Wolf dependency; the ridge ensures zero-variance
training embeddings still produce meaningful distances. Direct novelty objects
accept an explicit `covariance_shrinkage` in `[0, 1]` or estimate it automatically.
Cosine distance is an ablation and may miss radial departures from a prototype.

For `n` reserved known-class calibration scores and quantile `q`, rejection uses
the `ceil((n + 1) * q)`-th ordered score, or infinity when that rank exceeds `n`.
Small calibration sets can therefore produce no hard rejections at high quantiles;
this is deliberate finite-sample behavior. Calibration rejects empty/nonfinite
scores and invalid quantiles. Tail p-values are
`(1 + count(calibration_score >= query_score)) / (n + 1)`, including ties
conservatively. Smaller values indicate less resemblance to the reference.

These are pooled known-class tail ranks, not posterior probabilities of an
unknown defect or physical failure. Their usual finite-sample interpretation
requires a fixed scoring model and exchangeable calibration/query samples;
correlated windows and new recording domains need separate evaluation. See
[split conformal prediction under non-exchangeability](https://www.jmlr.org/papers/v25/23-1553.html).

`calibrate_probabilities` fits one positive temperature to reserved labeled
logits by minimizing negative log likelihood. It preserves the predicted known
class and does not change embeddings or novelty scores. It is separate from
novelty calibration; neither is performed automatically on training data.
Refitting resets both the temperature and novelty calibration.

## Baselines and preprocessing

`RocketDischargeClassifier` and `FeatureDischargeClassifier` retain their existing
APIs and now expose `novelty_pvalue` after novelty calibration. Their transforms
live in the reusable `foreblocks.features` package: `RocketFeatures` is an alias
of `foreblocks.features.FusedRocket`, and `extract_features` is
`foreblocks.features.SignalFeatures` configured with the acoustic bands.
`RocketDischargeClassifier(transformer="minirocket" | "multirocket" | "rocket")`
swaps in the published Rocket-family transforms. Ridge softmax
outputs are uncalibrated scores; temperature calibration above applies to the
neural classifier.

The engineered pulse features use a signal-relative numerical floor instead of
one raw amplitude unit, making low-amplitude pulses measurable. Hilbert envelope
resampling clips negative FIR ringing to zero. Direct `compute_envelope` supports
rational sampling-rate ratios; the classifier does not require integer envelope
decimation.

All changes are covered by synthetic and numerical regression tests. No improved
accuracy on real discharge recordings is claimed without recording-held-out and
external-domain benchmarks. Compare attention against mean pooling and ablate
amplitude features on the same untouched test recordings.
