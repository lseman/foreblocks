# Configuration

This page summarizes the most important configuration objects and settings exposed through the current API.

## `ModelConfig`

`ModelConfig` is a lightweight dataclass for architecture-level settings such as:

- `model_type`
- `input_size`
- `output_size`
- `hidden_size`
- `seq_len`
- `target_len`
- `strategy`
- `teacher_forcing_ratio`

It is useful when you want a typed container for model-level settings, even though many examples instantiate modules directly.

## `TrainingConfig`

`TrainingConfig` controls the training loop.

Core fields:

- `num_epochs`
- `learning_rate`
- `weight_decay`
- `batch_size`
- `patience`
- `min_delta`
- `use_amp`
- `gradient_clip_val`
- `scheduler_type`

NAS-related fields:

- `train_nas`
- `nas_alpha_lr`
- `nas_alpha_weight_decay`
- `nas_warmup_epochs`
- `nas_alternate_steps`
- `nas_use_val_for_alpha`
- `nas_discretize_at_end`

Conformal-related fields:

- `conformal_enabled`
- `conformal_method`
- `conformal_quantile`
- `conformal_knn_k`
- `conformal_rolling_alpha`
- `conformal_aci_gamma`
- `conformal_enbpi_B`
- `conformal_enbpi_window`
- `conformal_cptc_window`
- `conformal_afocp_feature_dim`

## `TimeSeriesPreprocessor` settings

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

## Related pages

- [Public API](public-api.md)
- [Getting Started](../getting-started.md)
