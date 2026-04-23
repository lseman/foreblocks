# Configuration

This page summarizes the main configuration objects exposed through the current API.

The canonical configuration definitions live in:

- `foreblocks/config.py` for `ModelConfig` and `TrainingConfig`

## `ModelConfig`

`ModelConfig` is a lightweight dataclass for architecture-level settings.

Core fields:

| Field | Purpose |
| --- | --- |
| `model_type` | selects the backbone family such as LSTM-style vs transformer-style workflows |
| `input_size` | feature dimension presented to the model |
| `output_size` | output feature dimension |
| `hidden_size` | latent width for recurrent or sequence blocks |
| `seq_len` | source window length |
| `target_len` | forecast horizon |
| `strategy` | forecasting strategy such as `seq2seq` |
| `teacher_forcing_ratio` | decoder teacher forcing during training |

Extended architecture fields include `dim_feedforward`, `dropout`, `num_encoder_layers`, `num_decoder_layers`, `multi_encoder_decoder`, `latent_size`, and `nheads`.

It is useful when you want a typed container for model-level settings, even though many examples instantiate modules directly.

## `TrainingConfig`

`TrainingConfig` controls the training loop.

### Core training

| Field | Purpose |
| --- | --- |
| `num_epochs` | maximum epoch count |
| `learning_rate` | optimizer learning rate |
| `weight_decay` | optimizer weight decay |
| `batch_size` | default batch size used in examples and helpers |
| `patience` | early-stopping patience |
| `min_delta` | minimum improvement threshold for early stopping |
| `use_amp` | enables automatic mixed precision |
| `gradient_clip_val` | gradient clipping threshold |
| `gradient_accumulation_steps` | number of mini-batches to accumulate before stepping |
| `l1_regularization` | optional L1 penalty |
| `kl_weight` | optional KL weight for compatible models |

### Scheduler and logging

| Field | Purpose |
| --- | --- |
| `scheduler_type` | scheduler selector |
| `lr_step_size` | step scheduler interval |
| `lr_gamma` | step scheduler decay |
| `min_lr` | lower learning-rate bound |
| `verbose` | trainer logging verbosity |
| `log_interval` | batch logging cadence |
| `save_best_model` | whether to retain best checkpoint state |
| `save_model_path` | optional filesystem path for saved weights |
| `experiment_name` | MLTracker experiment name when tracking is enabled |

### Mixture-of-experts logging

| Field | Purpose |
| --- | --- |
| `moe_logging` | enables MoE report collection |
| `moe_log_latency` | records latency metrics in MoE logs |
| `moe_condition_name` | condition label for segmented MoE analysis |
| `moe_condition_cardinality` | category count for condition-aware reports |

### NAS settings

| Field | Purpose |
| --- | --- |
| `train_nas` | enables alternating optimization for architecture parameters |
| `nas_alpha_lr` | learning rate for architecture parameters |
| `nas_alpha_weight_decay` | weight decay for architecture parameters |
| `nas_warmup_epochs` | epochs before alpha updates begin |
| `nas_alternate_steps` | alpha-step frequency |
| `nas_use_val_for_alpha` | uses validation loss for alpha updates |
| `nas_discretize_at_end` | discretizes architecture choices after training |
| `nas_discretize_threshold` | threshold used during discretization |
| `nas_log_alphas` | logs alpha values during training |

### Conformal prediction settings

Shared fields:

| Field | Purpose |
| --- | --- |
| `conformal_enabled` | master switch |
| `conformal_method` | method name such as `split`, `rolling`, `agaci`, `enbpi`, `cptc`, or `afocp` |
| `conformal_quantile` | target coverage level |
| `conformal_knn_k` | neighborhood size for local conformal methods |
| `conformal_local_window` | calibration window for local methods |
| `conformal_aci_gamma` | adaptation rate for ACI-style methods |
| `conformal_rolling_alpha` | rolling update rate |
| `conformal_agaci_gammas` | gamma grid for AgACI |
| `conformal_enbpi_B` | bootstrap count for EnbPI |
| `conformal_enbpi_window` | rolling window for EnbPI |
| `conformal_tsp_lambda` | regularization parameter for TSP |
| `conformal_tsp_window` | calibration window for TSP |
| `conformal_cptc_window` | state-aware rolling window |
| `conformal_cptc_tau` | state filter sharpness |
| `conformal_cptc_hard_state_filter` | switches CPTC filtering mode |
| `conformal_afocp_feature_dim` | feature dimension for AFOCP |
| `conformal_afocp_attn_hidden` | attention hidden size for AFOCP |
| `conformal_afocp_window` | context window for AFOCP |
| `conformal_afocp_tau` | temperature-like scaling for AFOCP |
| `conformal_afocp_internal_feat_hidden` | hidden size of the internal feature network |
| `conformal_afocp_internal_feat_depth` | depth of the internal feature network |
| `conformal_afocp_internal_feat_dropout` | dropout inside the internal feature network |
| `conformal_afocp_online_lr` | online update learning rate |
| `conformal_afocp_online_steps` | number of online update steps |

### Helper methods

`TrainingConfig` also provides:

- `update(**kwargs)` for safe field updates
- `get_conformal_params()` to build the trainer-facing conformal configuration payload

### Workflow recipes

**Smoke test — no optional features:**

```python
config = TrainingConfig(
    num_epochs=5,
    batch_size=16,
    patience=3,
    use_amp=False,
)
trainer = Trainer(model, config=config, auto_track=False)
```

**Preprocessing-heavy run — longer training, AMP, gradient clipping:**

```python
config = TrainingConfig(
    num_epochs=50,
    learning_rate=3e-4,
    batch_size=32,
    patience=10,
    use_amp=True,
    gradient_clip_val=1.0,
    scheduler_type="cosine",
    min_lr=1e-6,
    save_best_model=True,
    experiment_name="revin_decomp_baseline",
)
```

**DARTS / NAS run — architecture parameter alternation:**

```python
config = TrainingConfig(
    num_epochs=30,
    learning_rate=1e-3,
    batch_size=32,
    train_nas=True,
    nas_alpha_lr=3e-4,
    nas_warmup_epochs=5,
    nas_alternate_steps=1,
    nas_use_val_for_alpha=True,
    nas_discretize_at_end=True,
    nas_log_alphas=True,
)
```

**MoE monitoring run — enable router diagnostics:**

```python
config = TrainingConfig(
    num_epochs=40,
    learning_rate=3e-4,
    use_amp=True,
    moe_logging=True,
    moe_log_latency=True,
    moe_condition_name="season",
    moe_condition_cardinality=4,
    experiment_name="moe_router_audit",
    save_best_model=True,
)
```

**Conformal prediction run — split conformal with 90 % coverage:**

```python
config = TrainingConfig(
    num_epochs=30,
    batch_size=32,
    conformal_enabled=True,
    conformal_method="split",
    conformal_quantile=0.90,
)
```

**Tracking-oriented production run:**

```python
config = TrainingConfig(
    experiment_name="baseline_direct",
    save_best_model=True,
    save_model_path="checkpoints/best.pt",
    use_amp=True,
    gradient_clip_val=1.0,
    scheduler_type="cosine",
)
```

## `TimeSeriesHandler` settings

Important preprocessor controls include:

- `window_size`
- `horizon`
- `normalize`
- `differencing`
- `detrend`
- `apply_filter`
- `apply_ewt`
- `remove_outliers`
- `apply_imputation`
- `generate_time_features`
- `self_tune`

Use the dedicated guide for more detail:

- [Preprocessor Guide](../preprocessor.md)

## Practical guidance

- For first runs, keep configuration small and explicit.
- Disable optional features until the core training loop is working.
- Enable conformal and NAS only after validating your baseline training path.
- Use `auto_track=False` on `Trainer` when you do not want MLTracker involved in local smoke tests.

## Related pages

- [Public API](public-api.md)
- [Getting Started](../getting-started.md)
- [Troubleshooting](../troubleshooting.md)
