---
title: Quick Reference Cheat Sheet
description: One-page architecture and training config reference for common scenarios.
editLink: true
---

# Quick Reference Cheat Sheet

Copy-paste starting configs for common forecasting tasks.

## Scenario 1: Baseline Transformer (Minimal)

```python
from foreblocks import TransformerDecoder, Trainer, TrainingConfig

model = TransformerDecoder(
    input_size=1, output_size=1,
    d_model=256, nhead=8, num_layers=4, dim_feedforward=1024,
)

config = TrainingConfig(
    num_epochs=100, learning_rate=1e-3, batch_size=32,
)

trainer = Trainer(model, config)
trainer.train(train_loader, val_loader)
```

**Use case:** First model, sanity-check, baseline comparison.

---

## Scenario 2: Fine-tuning Pretrained (SOTA)

```python
from foreblocks.modules.skip.mod import LayerDropoutSchedule
from foreblocks.models.transformer.core.encoder import TransformerEncoder
from foreblocks.models.transformer.core.decoder import TransformerDecoder

encoder = TransformerEncoder(
    input_size=8, d_model=256, num_layers=6, nhead=8,
    layer_dropout_schedule=LayerDropoutSchedule(
        num_layers=6, base_dropout=0.03, max_dropout=0.1, profile="deeper_more"
    ),
)

decoder = TransformerDecoder(
    input_size=8, output_size=1, d_model=256, num_layers=4, nhead=8,
    layer_dropout_schedule=LayerDropoutSchedule(
        num_layers=4, base_dropout=0.02, max_dropout=0.08,
    ),
)

config = TrainingConfig(
    num_epochs=20, learning_rate=2e-5, weight_decay=0.01,
    use_llrd=True, llrd_decay=0.85,
    scheduler_type="warmup_cosine", warmup_ratio=0.1,
    steps_per_epoch=500,
)

trainer = Trainer(encoder, config)  # or decoder
trainer.train(train_loader, val_loader)
```

**Use case:** Transfer learning, limited data, pretrained backbone.

---

## Scenario 3: Long Sequences + Efficiency

```python
from foreblocks.modules.skip.mod import MoDBudgetScheduler
from foreblocks.models.transformer.core.encoder import TransformerEncoder

mod_sched = MoDBudgetScheduler(
    num_layers=6, start_keep=1.0, end_keep=0.9,
    warmup_steps=1000, total_steps=50000, layer_profile="deeper_more",
)

encoder = TransformerEncoder(
    input_size=8, d_model=256, num_layers=6, nhead=8,
    patch_encoder=True, patch_len=16, patch_stride=8,
    use_mod=True, mod_budget_scheduler=mod_sched,
    use_gradient_checkpointing=True,
)

# Configure attention architecture via config
encoder.attention.architecture = "linear"  # O(T) attention

config = TrainingConfig(
    num_epochs=50, learning_rate=1e-3,
    use_llrd=True, llrd_decay=0.9,
    scheduler_type="warmup_cosine", warmup_ratio=0.1,
    steps_per_epoch=1000,
)

trainer = Trainer(encoder, config)
trainer.train(train_loader, val_loader)
```

**Use case:** Very long sequences (5000+ tokens), memory-constrained.

---

## Scenario 4: Capacity + MoE

```python
from foreblocks.models.transformer.core.encoder import TransformerEncoder

encoder = TransformerEncoder(
    input_size=8, d_model=384, num_layers=6, nhead=8,
    dim_feedforward=2048,
    use_moe=True, num_experts=16, num_shared=2, top_k=2,
    router_type="noisy_topk",
    load_balance_weight=0.02, z_loss_weight=0.001,
    moe_use_latent=True, moe_latent_dim=192,
    use_gradient_checkpointing=True,
)

config = TrainingConfig(
    num_epochs=100, learning_rate=1e-3, weight_decay=0.01,
    use_llrd=True, llrd_decay=0.9,
    scheduler_type="warmup_cosine", warmup_ratio=0.1,
    steps_per_epoch=2500,
)

trainer = Trainer(encoder, config)
trainer.train(train_loader, val_loader)
```

**Use case:** Large datasets, high accuracy requirement, compute budget available.

---

## Scenario 5: Unstable Training (Deep Model)

```python
from foreblocks.modules.skip.mod import LayerDropoutSchedule
from foreblocks.models.transformer.core.encoder import TransformerEncoder

encoder = TransformerEncoder(
    input_size=8, d_model=256, num_layers=12, nhead=8,
    use_gateskip=True, gate_lambda=0.1,
    use_mhc=True, mhc_n_streams=4,
    layer_dropout_schedule=LayerDropoutSchedule(
        num_layers=12, base_dropout=0.05, max_dropout=0.2,
    ),
    use_gradient_checkpointing=True,
)

config = TrainingConfig(
    num_epochs=100, learning_rate=1e-3, weight_decay=0.01,
    gradient_clip_val=1.0,
    use_llrd=True, llrd_decay=0.85,
    scheduler_type="warmup_cosine", warmup_ratio=0.15,
    steps_per_epoch=2500,
)

trainer = Trainer(encoder, config)
trainer.train(train_loader, val_loader)
```

**Use case:** Very deep models (12+ layers), training divergence, gradient explosions.

---

## Architecture Quick-Pick

| Feature | Enable | Default | Why |
|---------|--------|---------|-----|
| **Long sequences** | `attention.architecture="linear"` or `gla` or `deltanet` | "standard" | O(T) vs O(T²) |
| **Variable tokens** | `use_mod=True` | False | Skip layers for easy tokens |
| **Overfitting** | `layer_dropout_schedule` | None | Deeper layers → higher dropout |
| **Capacity** | `use_moe=True` | False | Router to 16+ experts |
| **Stability (deep)** | `use_gateskip=True` | False | Learn residual magnitude |
| **Redundancy** | `use_mhc=True` | False | Parallel streams, learned mixing |
| **Memory-bound** | `use_gradient_checkpointing=True` | False | Recompute to save memory |
| **Efficiency** | `share_layers=True` | False | Reuse weights (1/n params) |

---

## Training Quick-Pick

| Goal | Setting | Default | Why |
|------|---------|---------|-----|
| **Fine-tune** | `use_llrd=True, llrd_decay=0.85` | False | Early layers static, late refine |
| **From scratch** | `use_llrd=True, llrd_decay=0.9` | False | All layers decay equally |
| **Convergence** | `scheduler_type="warmup_cosine"` | None | Warmup + cosine > step-decay |
| **Stability** | `gradient_clip_val=1.0` | None | Prevent exploding gradients |
| **Fast training** | `use_gradient_checkpointing=False` | False | Save memory, slower (trade-off) |

---

## Attention Architecture Selector

```
Is sequence length > 5000?
├─ Yes → use attention.architecture="linear" or "gla" or "deltanet"
└─ No  → use attention.architecture="standard"

Do you need length generalization?
├─ Yes → use attention.position.encoding="alibi"
└─ No  → use attention.position.encoding="rope" (default)
```

---

## Router Selector (MoE)

```
Do you want fast inference?
├─ Yes → router_type="hash"
└─ No  → Do you want adaptive routing?
         ├─ Yes → router_type="adaptive_noisy_topk"
         └─ No  → router_type="noisy_topk" (recommended)
```

---

## Import Quick-Ref

```python
# Top-level (stable)
from foreblocks import (
    TransformerEncoder, TransformerDecoder,
    Trainer, TrainingConfig,
    ModelEvaluator,
)

# Schedules & modules
from foreblocks.modules.skip.mod import (
    LayerDropoutSchedule,
    MoDBudgetScheduler,
)

# Config-only
from foreblocks.config import TrainingConfig

# MoE details
from foreblocks.modules.moe.ff import FeedForwardBlock

# Advanced attention
from foreblocks.modules.attention.multi_att import MultiAttention
```

---

## Config Fields Summary

### Essential (almost always set)

```python
config = TrainingConfig(
    num_epochs=50,
    learning_rate=1e-3,
    batch_size=32,
    weight_decay=0.01,  # L2 regularization
)
```

### Training optimization (new)

```python
# LLRD (layer-wise LR decay)
use_llrd=True,
llrd_decay=0.9,  # Decay factor per layer

# Warmup-cosine scheduler
scheduler_type="warmup_cosine",
warmup_steps=1000,  # OR warmup_ratio=0.1
steps_per_epoch=2500,  # Required if using warmup_ratio

# Gradient clipping
gradient_clip_val=1.0,
```

### Attention configuration (new)

```python
from foreblocks.models.transformer.config import TransformerConfig
from foreblocks.modules.attention.config import (
    AttentionConfig, AttentionShapeConfig, AttentionPositionConfig,
    AttentionVariantConfig,
)

config = TransformerConfig(
    d_model=256, nhead=8,
    attention=AttentionConfig(
        shape=AttentionShapeConfig(d_model=256, n_heads=8, max_seq_len=4096),
        architecture="linear",  # or "standard", "gla", "deltanet", etc.
        position=AttentionPositionConfig(encoding="rope"),
        variant=AttentionVariantConfig(use_swiglu=True),
    ),
)
```

### Optional efficiency

```python
use_gradient_checkpointing=True,
gradient_accumulation_steps=2,
```

---

## Common Hyperparameter Ranges

| Param | Small Model | Large Model | Notes |
|-------|------------|------------|-------|
| d_model | 128–256 | 512–1024 | Embedding dim |
| nhead | 4–8 | 8–16 | Attention heads |
| num_layers (enc) | 2–4 | 6–12 | Encoder depth |
| num_layers (dec) | 2–4 | 4–6 | Decoder depth |
| dim_feedforward | 512–1024 | 2048–4096 | FFN hidden |
| dropout | 0.1–0.3 | 0.05–0.15 | Attention/residual |
| learning_rate | 5e-4–1e-3 | 1e-4–1e-3 | Use LLRD for large |
| batch_size | 16–32 | 32–128 | GPU memory permitting |

---

## Performance Tips

### Training speed

1. Increase `batch_size` (if GPU memory allows)
2. Use `attention_mode="linear"` for long sequences
3. Enable `use_gradient_checkpointing=False` to trade memory for speed
4. Use `router_type="hash"` with MoE (no learned routing)

### Convergence

1. Use `scheduler_type="warmup_cosine"` + `use_llrd=True`
2. Set `warmup_ratio=0.1` (10% of training steps)
3. Clip gradients: `gradient_clip_val=1.0`

### Memory efficiency

1. Enable `use_gradient_checkpointing=True`
2. Use `share_layers=True` (reuse weights)
3. Reduce `batch_size`
4. Use `patch_encoder=True` (compress sequences)

### Accuracy (after you have a baseline)

1. Try `layer_dropout_schedule` (stochastic depth)
2. Enable MoE: `use_moe=True, num_experts=16`
3. Try `attention_mode="hybrid"` (mixed standard/linear)
4. Increase model capacity: larger `d_model`, `dim_feedforward`

---

## Testing Checklist

- [ ] Train runs for 1 epoch without error
- [ ] Loss decreases consistently
- [ ] Validation metrics improve
- [ ] No NaN/Inf in loss or metrics
- [ ] Gradient magnitudes reasonable (0.01–10)
- [ ] Model trains same speed with/without new features

