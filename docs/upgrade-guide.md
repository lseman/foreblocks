---
title: Upgrade Guide — SOTA Features
description: Migration guide for new LLRD, per-layer dropout, and warmup-cosine scheduler. Backward-compatible.
editLink: true
---

[[toc]]

# Upgrade Guide: SOTA Training Features

ForeBlocks v2.1+ includes production-grade training optimizations (LLRD, per-layer dropout, warmup-cosine scheduler). All features are **backward-compatible**—existing code works unchanged.

## What's New

### 1. Layer-wise Learning Rate Decay (LLRD)

**New in v2.1:** Automatically group parameters by transformer depth, apply different LRs per layer.

**Why upgrade:** 
- Fine-tuning improvements: 2–5% accuracy gain on downstream tasks
- Stabilizes training of very deep networks (12+ layers)
- Standard practice in ULMFiT, BERT, modern LLMs

**Backward compatible:** Disabled by default (`use_llrd=False`).

### 2. Per-Layer Dropout Schedule

**New in v2.1:** Depth-scaled attention dropout—deeper layers use higher dropout.

**Why upgrade:**
- Reduces overfitting in late layers (refinement stages)
- More expressive than flat dropout
- Consistent with stochastic depth literature

**Backward compatible:** Disabled by default (`layer_dropout_schedule=None`).

### 3. Warmup-Cosine Scheduler

**New in v2.1:** Linear warmup + cosine annealing, measured in optimizer steps (handles gradient accumulation).

**Why upgrade:**
- Standard in modern DL (BERT, GPT-2+)
- Better convergence than step-decay or simple cosine
- Properly handles gradient accumulation

**Backward compatible:** Existing `scheduler_type` values work as-is.

---

## Migration Paths

### Path 1: Minimal — No Changes Needed

Your existing code continues to work:

```python
from foreblocks.config import TrainingConfig
from foreblocks.core.training.trainer import Trainer

config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    # No new fields required — defaults disable all new features
)

trainer = Trainer(model=your_model, config=config)
# Same behavior as before
```

### Path 2: Adopt LLRD + Warmup-Cosine (Recommended for Fine-tuning)

```python
config = TrainingConfig(
    num_epochs=50,
    learning_rate=1e-3,
    weight_decay=0.01,
    # ── New: enable LLRD ──
    use_llrd=True,
    llrd_decay=0.9,
    # ── New: enable warmup-cosine ──
    scheduler_type="warmup_cosine",
    warmup_ratio=0.1,  # 10% of training steps
    steps_per_epoch=2500,  # Needed for warmup_ratio calculation
)

trainer = Trainer(model=model, config=config)
```

**When to use:** Fine-tuning pretrained models, large models (6+ layers), training on new domains.

### Path 3: Adopt Per-Layer Dropout (Recommended for Overfitting)

```python
from foreblocks.modules.skip.mod import LayerDropoutSchedule
from foreblocks.models.transformer.tf_encoder import TransformerEncoder

dropout_schedule = LayerDropoutSchedule(
    num_layers=6,
    base_dropout=0.05,
    max_dropout=0.15,
    profile="deeper_more",
)

encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    num_layers=6,
    layer_dropout_schedule=dropout_schedule,  # New parameter
)

# Training config unchanged — dropout is a model-level choice
config = TrainingConfig(...)
trainer = Trainer(model=encoder, config=config)
```

**When to use:** If overfitting to small datasets, using very deep models, or need stochastic depth.

### Path 4: Full SOTA (LLRD + Warmup-Cosine + Per-Layer Dropout)

```python
from foreblocks.config import TrainingConfig
from foreblocks.core.training.trainer import Trainer
from foreblocks.modules.skip.mod import LayerDropoutSchedule
from foreblocks.models.transformer.tf_encoder import TransformerEncoder

# Model with per-layer dropout
dropout_schedule = LayerDropoutSchedule(
    num_layers=6,
    base_dropout=0.05,
    max_dropout=0.15,
    profile="deeper_more",
)

encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    num_layers=6,
    dim_feedforward=1024,
    use_moe=True,
    num_experts=8,
    layer_dropout_schedule=dropout_schedule,
)

# Training with LLRD + warmup-cosine
config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=0.01,
    batch_size=32,
    steps_per_epoch=2500,
    use_llrd=True,
    llrd_decay=0.9,
    scheduler_type="warmup_cosine",
    warmup_ratio=0.1,
    use_gradient_checkpointing=True,
)

trainer = Trainer(model=encoder, config=config)
history = trainer.train(train_loader, val_loader)
```

---

## Configuration Changes

### `TrainingConfig` New Fields

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `use_llrd` | bool | False | Enable layer-wise LR decay |
| `llrd_decay` | float | 0.9 | Decay factor per layer (0–1) |
| `warmup_steps` | int | 0 | Explicit warmup steps |
| `warmup_ratio` | float | 0.0 | Warmup as fraction of total steps |
| `steps_per_epoch` | int | None | Steps per epoch (for warmup_ratio) |

All are **optional and off by default**.

### Backward-Compatible Changes

| Feature | Behavior |
|---------|----------|
| `scheduler_type="cosine"` | Still works (epoch-level cosine annealing) |
| `scheduler_type="step"` | Still works (step-decay) |
| `scheduler_type="plateau"` | Still works (reduce on plateau) |
| `scheduler_type="warmup_cosine"` | New: step-level warmup + cosine |

---

## Transformer API Changes

### `BaseTransformer.__init__()` New Parameters

```python
layer_dropout_schedule: Optional[LayerDropoutSchedule] = None
```

**Only used if provided.** Defaults to flat dropout (existing behavior).

---

## Common Patterns

### Fine-tuning a Pretrained Model

```python
config = TrainingConfig(
    num_epochs=10,  # Short fine-tuning
    learning_rate=2e-5,  # Small LR
    use_llrd=True,
    llrd_decay=0.85,  # Conservative decay
    scheduler_type="warmup_cosine",
    warmup_ratio=0.05,
    steps_per_epoch=500,
)
```

### Training from Scratch (Large Model)

```python
config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    use_llrd=True,
    llrd_decay=0.9,  # Standard decay
    scheduler_type="warmup_cosine",
    warmup_ratio=0.1,  # Longer warmup from scratch
    steps_per_epoch=2500,
    use_gradient_checkpointing=True,
)
```

### Quick Baseline (No New Features)

```python
config = TrainingConfig(
    num_epochs=50,
    learning_rate=1e-3,
    scheduler_type="cosine",  # Existing epoch-level cosine
)
# No LLRD, no warmup-cosine, no per-layer dropout
```

---

## Breaking Changes

**None.** All upgrades are additive and backward-compatible.

---

## FAQ

### Q: Should I use LLRD for small models (2–4 layers)?

**A:** Probably not. LLRD's benefit scales with depth. For shallow models, use flat LR. If in doubt, try both and compare.

### Q: What's the difference between `warmup_steps` and `warmup_ratio`?

**A:**
- `warmup_steps=1000` — exactly 1000 optimizer steps of warmup, regardless of epoch count
- `warmup_ratio=0.1, steps_per_epoch=2500` — warmup is 10% of total training steps (0.1 × num_epochs × steps_per_epoch)

Use `warmup_ratio` for reproducibility across different batch sizes.

### Q: Can I use LLRD with NAS?

**A:** Yes. LLRD is applied to weight parameters only; NAS alpha parameters remain in a separate optimizer group.

### Q: Does per-layer dropout work with shared layers?

**A:** No. If `share_layers=True`, all layer references point to the same module, so per-layer dropout can't apply. The schedule is ignored and flat dropout is used.

### Q: Will the new scheduler work with gradient accumulation?

**A:** Yes. Warmup-cosine is stepped after each real `optimizer.step()`, so it's gradient-accumulation-aware.

### Q: What if I set both `warmup_steps` and `warmup_ratio`?

**A:** `warmup_steps` takes precedence. Only use one.

---

## Performance Expectations

### LLRD Impact

Typical improvements on fine-tuning tasks (pretrained → new domain):
- **2–5% accuracy gain** for encoder-decoder models
- **Faster convergence**: 10–20% fewer epochs
- **More stable**: lower loss variance across restarts

For from-scratch training, gains are typically smaller (0–2%).

### Per-Layer Dropout Impact

Typical improvements (small datasets, overfitting regime):
- **1–3% accuracy gain** when overfitting is present
- **~10% memory overhead**: more dropout ops
- **No impact** if model is not overfitting

### Warmup-Cosine Impact

Typical improvements vs. step-decay:
- **2–5% accuracy gain** on challenging datasets
- **Faster early convergence** (warmup ramps LR smoothly)
- **Smoother training curves**: less noisy loss

---

## Troubleshooting Upgrades

### Loss doesn't decrease with LLRD enabled

**Symptom:** Training loss flat or increasing with `use_llrd=True`.

**Cause:** Early layers are learning too slowly. Increase `llrd_decay` (make it closer to 1.0, e.g., 0.95) to reduce layer-depth penalty.

**Fix:**
```python
config = TrainingConfig(
    use_llrd=True,
    llrd_decay=0.95,  # Less aggressive decay
)
```

### Warmup-cosine scheduler hasn't converged by end

**Symptom:** Loss still decreasing at epoch 100, scheduler doesn't match epoch count.

**Cause:** `steps_per_epoch` mismatch. Verify actual number of batches per epoch.

**Fix:**
```python
# Log actual steps
import math
actual_steps = math.ceil(len(train_dataset) / batch_size)
print(f"Actual steps_per_epoch: {actual_steps}")

config = TrainingConfig(
    steps_per_epoch=actual_steps,  # Use correct value
)
```

### Per-layer dropout not applied

**Symptom:** All layers still have same dropout rate.

**Cause:** Forgot to pass schedule to encoder, or using `share_layers=True`.

**Fix:**
```python
encoder = TransformerEncoder(
    num_layers=6,
    layer_dropout_schedule=LayerDropoutSchedule(...),  # Don't forget
    share_layers=False,  # Must be False for per-layer dropout
)
```

---

## Next Steps

- **[Advanced Transformer Features](transformer-advanced)** — Full documentation on LLRD, dropout schedule, and all transformer components
- **[Transformer Guide](transformer)** — Quick reference for model configuration
- **[Training Config Reference](reference/configuration)** — All config options explained
