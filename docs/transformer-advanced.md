---
title: Advanced Transformer Features
description: Comprehensive guide to SOTA transformer features including LLRD, per-layer dropout, GateSkip, MoD, mHC, and attention variants.
editLink: true
---

[[toc]]

# Advanced Transformer Features

This guide covers production-grade transformer optimizations and architectural features in ForeBlocks, including recent SOTA additions like layer-wise LR decay (LLRD) and per-layer dropout scheduling.

## Training Optimizations (LLRD + Warmup-Cosine Scheduler)

### Layer-wise Learning Rate Decay (LLRD)

Deep transformers benefit from layer-dependent learning rates: early layers (feature extraction) learn at a coarser rate, later layers (fine-tuning) at a finer rate. ForeBlocks implements LLRD automatically via parameter grouping.

#### Enable LLRD

```python
from foreblocks.training.config import TrainingConfig
from foreblocks.training.trainer import Trainer

config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=0.01,
    # ── LLRD settings ──
    use_llrd=True,
    llrd_decay=0.9,  # Decay factor per layer depth (0-1)
)

trainer = Trainer(model=your_model, config=config)
```

**How it works:**
- Layer 0 (first/shallowest) gets `base_lr`
- Layer i gets `base_lr * decay^i`
- Non-layer params (embeddings, input_adapter, final_norm) get `base_lr` unscaled
- Bias and norm weights automatically get `weight_decay=0` (standard practice)

**Effect:** Early layers train slower, later layers faster. Reduces catastrophic forgetting in fine-tuning.

#### LLRD Decay Factor Guidelines

| Use Case | llrd_decay | Rationale |
|----------|-----------|-----------|
| Fine-tuning (pretrained) | 0.85–0.95 | Conservative, early-layer decay = 30–50% |
| Training from scratch | 0.9–1.0 | Larger decay or flat (depends on init) |
| Large model (12+ layers) | 0.85 | Steeper decay for deeper networks |
| Small model (2–4 layers) | 0.95 | Milder decay |

### Warmup-Cosine Scheduler

Linear warmup + cosine annealing to `min_lr`. Measured in **optimizer steps**, not epochs (crucial for gradient accumulation).

#### Enable Warmup-Cosine

```python
config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    scheduler_type="warmup_cosine",
    # ── Warmup options (choose one) ──
    warmup_steps=1000,  # Explicit warmup steps
    # OR
    warmup_ratio=0.1,  # 10% of total training steps
    steps_per_epoch=2500,  # Required if using warmup_ratio
    # ── Optional ──
    min_lr=1e-6,  # Cosine floor (default: 0.01 * learning_rate)
)

trainer = Trainer(model=your_model, config=config)
# Scheduler is stepped automatically after each optimizer.step()
```

**Warmup phase (0 to warmup_steps):**
```
LR = (step / warmup_steps) * base_lr
```

**Cosine phase (warmup_steps to total_steps):**
```
progress = (step - warmup_steps) / (total_steps - warmup_steps)
LR = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
```

#### Warmup Ratio vs Steps

Use `warmup_ratio` for cleaner experiment reproduction:
```python
config = TrainingConfig(
    num_epochs=100,
    warmup_ratio=0.1,  # 10% of training
    steps_per_epoch=2500,
    # Computed: warmup_steps = 0.1 * 100 * 2500 = 25,000
)
```

Use `warmup_steps` when you have a fixed schedule (e.g., 5000 steps always):
```python
config = TrainingConfig(
    warmup_steps=5000,
)
```

### Combining LLRD + Warmup-Cosine (Recommended)

```python
config = TrainingConfig(
    num_epochs=50,
    learning_rate=1e-3,
    weight_decay=0.01,
    batch_size=32,
    steps_per_epoch=None,  # Will be set by trainer or computed on first epoch
    # ── LLRD ──
    use_llrd=True,
    llrd_decay=0.9,
    # ── Warmup-Cosine ──
    scheduler_type="warmup_cosine",
    warmup_ratio=0.1,
    # ── Checkpointing ──
    use_gradient_checkpointing=True,
)

trainer = Trainer(model=model, config=config)
history = trainer.train(train_loader=train_dl, val_loader=val_dl)
```

**Result:** Each parameter group gets its own warmup + cosine schedule scaled to its base LR.

---

## Per-Layer Dropout Schedule

### Motivation

Deeper transformer layers often benefit from higher dropout (stochastic-depth style): early layers preserve features broadly, later layers refine with more noise to prevent overfitting.

### Enable Per-Layer Dropout

```python
from foreblocks.nn.routing.mod import LayerDropoutSchedule
from foreblocks.nn.transformer.encoder import TransformerEncoder

# Create the schedule
dropout_schedule = LayerDropoutSchedule(
    num_layers=6,
    base_dropout=0.05,     # Shallow-layer dropout
    max_dropout=0.2,       # Deep-layer dropout
    profile="deeper_more",  # or "deeper_less", "flat"
)

# Pass to encoder/decoder
encoder = TransformerEncoder(
    input_size=1,
    d_model=256,
    n_heads=8,
    num_layers=6,
    ff_dim=1024,
    dropout_schedule=dropout_schedule,
)
```

### Profiles

| Profile | Behavior | Use Case |
|---------|----------|----------|
| `"flat"` | Identical dropout all layers | Baseline, comparison |
| `"deeper_more"` | Dropout increases with depth | Standard (early preserve, late refine) |
| `"deeper_less"` | Dropout decreases with depth | Rare (opposite intuition) |

Example: `deeper_more` with 6 layers, base=0.05, max=0.2:
```
Layer 0: 0.05
Layer 1: 0.083
Layer 2: 0.117
Layer 3: 0.150
Layer 4: 0.183
Layer 5: 0.200
```

### Integration with Trainer

```python
from foreblocks.nn.routing.mod import LayerDropoutSchedule
from foreblocks.nn.transformer.decoder import TransformerDecoder

dropout_schedule = LayerDropoutSchedule(
    num_layers=4,
    base_dropout=0.03,
    max_dropout=0.15,
    profile="deeper_more",
)

decoder = TransformerDecoder(
    input_size=1,
    output_size=1,
    d_model=256,
    n_heads=8,
    num_layers=4,
    dropout_schedule=dropout_schedule,
)

trainer = Trainer(model=decoder, config=config)
```

No config changes needed — dropout schedule is a model-level choice, not a training-loop choice.

---

## Advanced Attention Mechanisms

### Attention settings

Attention is configured with flat keywords on the transformer:

```python
from foreblocks import TransformerEncoder

encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    n_heads=8,
    n_kv_heads=2,              # grouped-query attention; None = one per head
    max_seq_len=4096,
    attention="standard",      # backend used by every layer
    attention_kernel="auto",   # "auto", "sdpa", "eager", ...
    position="rope",           # "rope", "alibi", "sinusoidal", "learnable", "none"
    rope_base=10000.0,
    rope_scaling="none",       # "none", "yarn", "ntk", "linear"
    kv_cache="auto",           # "auto"/"paged" use a paged KV cache
    attention_options={        # any other foreblocks.nn.attention.config field
        "qk_norm": True,
        "logit_softcap": 30.0,
        "gated_attention": True,
    },
)
```

`attention_options` accepts the field names of `AttentionCacheConfig`,
`AttentionVariantConfig`, and `AttentionFeatureConfig` (for example
`window_size`, `chunk_size`, `moba_topk`, `use_mla`, `attention_matching`,
`matching_keep_ratio`). Unknown names raise when the config is built.
`config.attention_config()` returns the nested `AttentionConfig` the layers use.

### Attention backends

`attention=` selects the backend:

| Backend | Kind | Notes |
| --- | --- | --- |
| `standard` | Softmax | O(T²); best for short sequences |
| `sype`, `prob_sparse`, `frequency`, `sliding_window`, `dilated_window`, `moba`, `nsa`, `softpick`, `autocor`, `dwt` | Softmax variants | Tune via `attention_options` |
| `linear` | Linear attention | O(T) |
| `gla` | Gated linear attention | Gate-controlled recurrence; long sequences |
| `deltanet` | Delta rule | Delta-based state update; streaming |
| `gated_deltanet`, `gated_delta` | Gated delta rule | Improved stability for streaming |
| `kimi` | Learned linear recurrence | Competitive with quadratic attention |

Recurrent and linear backends are self-attention only; decoder
cross-attention uses standard attention for them.

### Mixing backends across depth

`attention_pattern` interleaves the chosen backend with standard attention:

```python
# layers 0-2 use GLA, layer 3 uses standard attention, and so on
encoder = TransformerEncoder(input_size=8, num_layers=8, attention="gla", attention_pattern="3to1")

# every layer but the last uses Kimi
encoder = TransformerEncoder(input_size=8, num_layers=6, attention="kimi", attention_pattern="hybrid")
```

### Positional encoding

`position="rope"` and `"alibi"` are applied inside attention;
`"sinusoidal"` and `"learnable"` are added to the input embeddings. Pass
`pos_encoder=` to the constructor to use your own input-level module.

---

## GateSkip: Residual Path Gating

### Motivation

Residual paths can saturate. GateSkip learns a per-token gate controlling residual magnitude.

### Enable GateSkip

```python
encoder = TransformerEncoder(
    input_size=1,
    d_model=256,
    num_layers=6,
    gate_budget=1.0,  # Initial gate magnitude (1.0 = no gating initially)
    gate_aux_weight=0.1,  # Auxiliary loss weight
    residual="gateskip"
)
```

To anneal the budget during training, pass a scheduler; it steps once per
training forward:
```python
from foreblocks.nn.routing.gateskip import BudgetScheduler

encoder = TransformerEncoder(
    input_size=1,
    residual="gateskip",
    gate_scheduler=BudgetScheduler(b_start=1.0, b_end=0.5, total_steps=10_000),
)
```

---

## Mixture-of-Depths (MoD): Dynamic Layer Skipping

### Motivation

Not all tokens need all layers. MoD routes tokens to a subset of layers based on router scores, saving compute.

### Enable MoD

```python
from foreblocks.nn.routing.mod import MoDBudgetScheduler

budget_scheduler = MoDBudgetScheduler(
    num_layers=6,
    start_keep=1.0,      # Full capacity initially
    end_keep=0.85,       # 15% token reduction at end
    warmup_steps=1000,   # Steps before annealing starts
    total_steps=50000,   # Steps for annealing
    layer_profile="deeper_more",  # Deeper layers keep more tokens
)

encoder = TransformerEncoder(
    input_size=1,
    d_model=256,
    num_layers=6,
    mod_aux_weight=0.05,       # Auxiliary loss weight
    mod_scheduler=budget_scheduler,
    residual="mod",
)
```

**layer_profile:**
- `"flat"`: identical keep-rate all layers
- `"deeper_more"`: deeper layers keep more tokens (save early compute)
- `"deeper_less"`: deeper layers keep fewer tokens

---

## mHC: Manifold-Constrained Hyper-Connections

### Motivation

Maintain N parallel residual streams per token, with learned mixing via doubly-stochastic matrix. Reduces feature collapse.

### Enable mHC

```python
encoder = TransformerEncoder(
    input_size=1,
    d_model=256,
    num_layers=6,
    mhc_streams=4,         # Number of parallel streams
    mhc_sinkhorn_iters=20,   # Sinkhorn iterations for doubly-stochastic projection
    mhc_collapse="first",    # "first" or "mean" (how to collapse streams to output)
    residual="mhc"
)
```

**Limitation:** mHC is disabled for incremental decoding (KV-cached forward_one_step). Use mHC for full-sequence encoding only.

---

## Attention Residuals: Per-Token Depth Weighting

### Motivation

Paper "DeepNet" style: weight residuals from different depths, allowing early-exit and depth-awareness.

### Enable Attention Residuals

```python
encoder = TransformerEncoder(
    input_size=1,
    d_model=256,
    num_layers=6,
    attention_residual_mode="full",  # or "block"
    attention_residual_block_size=8,
    residual="attention",
)
```

**attn_residual_type:**
- `"full"`: per-token residual weights
- `"block"`: block-level residual weights

---

## Patching: Token-Level Compression

### Encoder Patching (Recommended)

Compress long sequences into patch tokens, reducing internal computation:

```python
encoder = TransformerEncoder(
    input_size=1,
    d_model=256,
    patching="shared",
    patch_len=16,        # Patch length in timesteps
    patch_stride=8,      # Stride between patches
    patch_pad_end=True,  # Pad end to align patches
)
# Input: [B, T, 1] → Patches: [B, Np, 256] where Np = ceil((T - patch_len) / stride) + 1
```

### CT-PatchTST: Channel-Time Patching

Forecasting-specific: treat channels as a dimension, create channel-time patches:

```python
encoder = TransformerEncoder(
    input_size=8,  # 8 variables
    patching="channel",
    patch_len=16,
    patch_stride=8,
    channel_fuse="linear",  # or "mean" (fuse channels)
)
```

---

## Gradient Checkpointing & Weight Tying

### Gradient Checkpointing

Trade compute for memory: recompute activations during backward instead of storing them.

```python
encoder = TransformerEncoder(
    input_size=1,
    d_model=256,
    num_layers=12,
    gradient_checkpointing=True,
)
```

**Effect:** ~30% memory reduction, ~25% compute increase. Worth it for large models or long sequences.

### Shared Layers (Weight Tying)

Use a single layer multiple times:

```python
encoder = TransformerEncoder(
    input_size=1,
    d_model=256,
    num_layers=12,
    share_layers=True,  # All 12 layers use the same weights
)
```

**Effect:** 1/12 parameters, similar downstream performance (empirically).

---

## Mixture-of-Experts (MoE) Feedforward

### Basic MoE

Replace standard FFN with routed expert mixture:

```python
from foreblocks.nn.moe.feedforward import FeedForwardBlock

moe_ffn = FeedForwardBlock(
    d_model=256,
    dim_ff=1024,
    use_moe=True,
    num_experts=8,
    top_k=2,           # Route to top-2 experts per token
    use_swiglu=True,   # SwiGLU experts (recommended)
)

# In transformer:
encoder = TransformerEncoder(
    input_size=1,
    d_model=256,
    ff_dim=1024,
    moe_experts=8,
    moe_top_k=2,
)
```

### Load Balancing

Auxiliary losses prevent expert collapse:

```python
config = TrainingConfig(
    # In FeedForwardBlock / TransformerEncoder, 
    # moe_aux_weight on the transformer scales the MoE auxiliary loss
)

# Access auxiliary losses during training:
loss = criterion(outputs, targets)
# MoE block computes aux_loss internally
if hasattr(model, "moe_log") and model.moe_log:
    moe_loss = model.moe_log.aux_loss
    total_loss = loss + moe_loss
```

### Router Types

```python
router_type="noisy_topk"     # Add noise, sample top-k
router_type="straight_topk"  # Straight-through, deterministic top-k
router_type="soft_dense"     # Soft assignment, all experts
router_type="hash"           # Hash-based routing (fast approximation)
router_type="adaptive_k"     # Learned adaptive k per token
```

### Shared Experts & Latent Projection

```python
moe_ffn = FeedForwardBlock(
    num_experts=8,
    num_shared=1,              # Extra shared expert (always active)
    moe_use_latent=True,       # Project through latent before routing
    moe_latent_dim=128,        # Latent dimension
    moe_latent_d_ff=512,       # Hidden dimension in latent FFN
)
```

---

## Complete Example: Production Encoder-Decoder

```python
from foreblocks.training.config import TrainingConfig
from foreblocks.training.trainer import Trainer
from foreblocks.nn.transformer.encoder import TransformerEncoder
from foreblocks.nn.transformer.decoder import TransformerDecoder
from foreblocks.nn.routing.mod import (
    LayerDropoutSchedule,
    MoDBudgetScheduler,
)
import torch
from torch.utils.data import DataLoader

# ── Schedules ──
dropout_schedule = LayerDropoutSchedule(
    num_layers=6,
    base_dropout=0.05,
    max_dropout=0.15,
    profile="deeper_more",
)

mod_scheduler = MoDBudgetScheduler(
    num_layers=6,
    start_keep=1.0,
    end_keep=0.9,
    warmup_steps=1000,
    total_steps=50000,
    layer_profile="deeper_more",
)

# ── Encoder ──
encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    n_heads=8,
    num_layers=6,
    ff_dim=1024,
    attention="linear", attention_pattern="hybrid",
    moe_experts=8,
    moe_top_k=2,
    mod_scheduler=mod_scheduler,
    gate_aux_weight=0.1,
    gradient_checkpointing=True,
    patching="shared",
    patch_len=16,
    patch_stride=8,
    dropout_schedule=dropout_schedule,
    residual="gateskip"  # one policy; also: mod,
)

# ── Decoder ──
decoder = TransformerDecoder(
    input_size=8,
    output_size=1,
    d_model=256,
    n_heads=8,
    num_layers=4,
    ff_dim=1024,
    attention="standard",
    dropout_schedule=LayerDropoutSchedule(
        num_layers=4,
        base_dropout=0.03,
        max_dropout=0.1,
    ),
    residual="gateskip",
)

# ── Training Config ──
config = TrainingConfig(
    num_epochs=50,
    learning_rate=1e-3,
    weight_decay=0.01,
    batch_size=32,
    use_llrd=True,
    llrd_decay=0.9,
    scheduler_type="warmup_cosine",
    warmup_ratio=0.1,
    steps_per_epoch=2500,
    use_gradient_checkpointing=True,
    gradient_clip_val=1.0,
)

# ── Training ──
trainer = Trainer(encoder, config=config)
# (decoder goes to loss computation separately in your training loop)

train_dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=config.batch_size)

history = trainer.train(train_dl, val_dl, epochs=50)
```

---

## Summary: Feature Combinations

| Goal | Recommended Config |
|------|-------------------|
| **Long sequences** | `attention="gla"` or `"deltanet"` + `residual="mod"` + `patching="shared"` |
| **Large models** | `gradient_checkpointing=True` + `share_layers=True` + `moe_experts=16` |
| **Fine-tuning** | `use_llrd=True` + `llrd_decay=0.9` + `scheduler_type="warmup_cosine"` |
| **Low latency** | `attention="kimi"` or `"gated_delta"` + `moe_options={"router_type": "hash"}` |
| **Stable training** | `residual="gateskip"` (or `"mhc"`) + `dropout_schedule=` |

---

## Transformer Tuner Features

`TransformerTuner` provides auto-hyperparameter selection and advanced feature analysis:

- **Lempel-Ziv complexity**: Measures sequence compressibility and structural regularity
- **CWT (Continuous Wavelet Transform) energy**: Captures frequency domain characteristics and periodic patterns

## References

- LLRD: ULMFiT (Howard & Ruder, 2018)
- Warmup-Cosine: BERT (Devlin et al., 2018)
- Per-layer dropout: Stochastic Depth (Huang et al., 2016)
- MoD: Ritter et al. (2024) arXiv:2404.02258
- mHC: Based on Hyper-Connection architectures
- GateSkip: Residual scaling literature
- GLA: Gated Linear Attention mechanisms
- DeltaNet: Gated Delta Rule attention
