# Changelog

Release history for `foreblocks` and `foretools`.

## v0.1.15 (current)

### foreblocks

- Added `mHC` (multi-Hyper-Connections) residual stream mixing to `TransformerEncoder` and `TransformerDecoder`
- Added paper-style Attention Residuals (`use_attention_residual`, `attn_residual_type`, `attention_residual_block_size`)
- Added Mixture-of-Depths token routing (`use_mod`, `mod_mode`, `mod_budget_scheduler`)
- Added latent-MoE path (`moe_use_latent`, `moe_latent_dim`, `moe_latent_d_ff`) for higher expert count at lower cost
- Added `adaptive_noisy_topk`, `hash_topk`, and `multi_hash_topk` router families
- Added `afocp` conformal method with attention-based feature network
- Added `cptc` (state-aware rolling conformal) method
- Removed classic dense load-balancing auxiliary loss; expert utilization now handled via router expert-bias adaptation
- `label_len <= 0` in Informer-like mode no longer generates a full-decoder Informer padding mask
- `forward_one_step(...)` is now incompatible with `use_mod=True` and `use_mhc=True` (explicitly guarded)

### foretools

- Added `tsgen` synthetic series generator with AR, seasonal, trend, and noise components
- Added `tsaug` AutoDA augmentation search
- Added `foreminer` changepoint detection, cluster analysis, and stationarity diagnostics
- `bohb` plotter and observation store improvements

### Breaking changes

- Code importing `TimeSeriesSeq2Seq` directly should switch to `ForecastingModel`
- `load_balance_weight` no longer drives the primary balancing mechanism in MoE; tune `z_loss_weight` and router type instead

---

## v0.1.14

- Initial public release of `foretools/bohb` Bayesian hyperparameter optimization
- Added `ConformalPredictionEngine` with `split`, `rolling`, `agaci`, and `enbpi` methods
- Added `GateSkip` sublayer-level residual gating
- Added CT-PatchTST encoder tokenization path (`ct_patchtst=True`)
- Added `foretools/fengineer` feature engineering pipeline with RFECV and mutual-information selection
- Added `foretools/vmd` decomposition toolkit (VMD, EMD-family, hierarchical VMD)
- Stabilized `MLTracker` local FastAPI server and `mltracker-tui` TUI

---

## v0.1.x and earlier

See [git log](https://github.com/lseman/foreblocks/commits/main) for commit-level history.
