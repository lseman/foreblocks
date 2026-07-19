---
title: Advanced Mixture-of-Experts (MoE)
description: Comprehensive MoE guide—routers, load-balancing, expert types, shared experts, latent projections, and production tuning.
editLink: true
---

[[toc]]

# Advanced Mixture-of-Experts (MoE) Guide

ForeBlocks implements a production-grade MoE stack with flexible routers, load-balancing strategies, expert types, and auxiliary mechanisms for stable training.

## Architecture Overview

```
Input: [B, T, D]
  ↓
Router (learns per-token scores)
  ↓
Top-k selector (selects top-k experts per token)
  ↓
Dispatcher (routes tokens to experts)
  ↓
Experts (parallel FFNs, only active experts compute)
  ↓
Combiner (weighted sum of expert outputs)
  ↓
Output: [B, T, D]
```

Auxiliary losses computed alongside:
- Load balancing loss (prevent expert collapse)
- Auxiliary expert loss (sparse routing)
- Auxiliary token loss (encourage specialization)

---

## Enabling MoE in Transformers

### Basic Enable

```python
encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    nhead=8,
    num_layers=6,
    use_moe=True,
    num_experts=8,     # Total number of experts
    top_k=2,           # Route to top-k experts per token
)
```

### Full Control (via FeedForwardBlock)

```python
from foreblocks.modules.moe.ff import FeedForwardBlock

ffn = FeedForwardBlock(
    d_model=256,
    dim_ff=1024,
    use_moe=True,
    # ── Expert config ──
    num_experts=16,
    num_shared=1,            # Additional shared expert (always active)
    top_k=2,
    use_swiglu=True,         # SwiGLU experts (recommended)
    # ── Router config ──
    router_type="noisy_topk",
    router_temperature=1.0,
    router_perturb_noise=0.01,
    # ── Routing mode ──
    routing_mode="token_choice",  # or "expert_choice"
    # ── Load balancing ──
    load_balance_weight=0.01,
    z_loss_weight=0.001,
    moe_aux_lambda=1.0,
    # ── Latent projection (optional) ──
    moe_use_latent=True,
    moe_latent_dim=128,
    moe_latent_d_ff=512,
    # ── Capacity control ──
    moe_capacity_factor=1.25,
)
```

---

## Router Types

### Noisy Top-K (Recommended Default)

```python
router_type="noisy_topk"
router_temperature=1.0          # Gumbel noise temperature
router_perturb_noise=0.01       # Gaussian noise std
```

**Behavior:**
1. Compute router logits: `logits = linear(x)`
2. Add Gumbel noise: `logits += -log(-log(uniform()))`
3. Scale by temperature: `logits /= temperature`
4. Select top-k (differentiable via Gumbel-max trick)

**Why:** Encourages exploration early, recovers to deterministic routing when trained.

### Straight-Through Top-K

```python
router_type="straight_topk"
```

Deterministic top-k with straight-through gradient (no noise).

**Pros:** Fast, deterministic, reproducible.
**Cons:** No exploration—can converge to poor local optima.

### Soft Dense

```python
router_type="soft_dense"
```

Soft assignment to all experts (no sparsity).

**Use case:** Debugging, baseline comparison. Not recommended for production (no compute savings).

### Hash Router

```python
router_type="hash"
router_hash_num_hashes=2
router_hash_num_buckets=64
router_hash_bucket_size=8
router_hash_seed=17
```

Hash-based expert assignment: O(1) routing, no learned router.

**Pros:** Fast, fixed compute, no training instability.
**Cons:** Fixed routing pattern, less adaptive.

**How it works:**
1. Hash input ID (or token position) → bucket
2. Map bucket → expert group
3. Route to experts in group

**Use case:** Inference-only, federated learning, extremely large scale.

### Adaptive Top-K

```python
router_type="adaptive_noisy_topk"
adaptive_k_head_dim=32
adaptive_k_tau=1.0
adaptive_k_baseline_momentum=0.99
adaptive_k_sparsity_lambda=0.0
```

Learn per-token k (number of experts to route to).

**Behavior:** Router outputs both logits (expert assignment) and k (number of experts).

**Auxiliary loss:** Encourages sparse k via sparsity_lambda.

**Use case:** Highly variable compute budgets.

### Auxiliary-Token Router

```python
router_type="auxiliary_token"
```

Expert routing via auxiliary token embeddings (learned per expert).

---

## Routing Modes

### Token Choice (Default)

```python
routing_mode="token_choice"
```

Each token independently selects its top-k experts. O(T*k) capacity.

**Pros:** Natural load balancing, each token gets its choice.
**Cons:** Can overload high-capacity experts.

### Expert Choice

```python
routing_mode="expert_choice"
expert_choice_tokens_per_expert=32  # Max tokens per expert
```

Each expert independently selects its top-m tokens. O(E*m) capacity.

**Pros:** Strict capacity control, predictable compute.
**Cons:** Tokens may not get their top-k experts.

---

## Load Balancing Mechanisms

MoE naturally suffers from **expert collapse**: most tokens route to the same expert, wasting capacity. ForeBlocks provides multiple balancing strategies.

### Auxiliary Losses (Standard)

#### Load Balance Loss

Encourage uniform expert usage:
```python
load_balance_weight=0.01
```

Loss: `sum((expert_capacity_ratios - ideal_ratio)^2)`

Where:
- `expert_capacity_ratios`: fraction of tokens routed to each expert
- `ideal_ratio`: 1 / num_experts

#### Z Loss (Gating Entropy)

Encourage uniform (soft) routing probabilities:
```python
z_loss_weight=0.001
```

Loss: `sum(log(total_router_probability))` (from GShard paper)

#### Auxiliary Expert Loss

Multi-head auxiliary loss (one head per expert):
```python
router_aux_num_aux=1
```

Encourages each expert to specialize.

### Router Bias Balance (Optional)

Learned per-expert router biases, annealed during training:

```python
moe_router_bias_balance=True
moe_router_bias_warmup_steps=1000
moe_router_bias_min_usage=0.01
moe_router_bias_update_rate=0.01
moe_router_bias_clip=2.0
```

**Behavior:** Dynamically adjust per-expert router bias to keep minimum usage above threshold.

### Soft Capacity (Optional)

Soft (differentiable) capacity constraint instead of hard dropping:

```python
moe_soft_capacity=True
moe_capacity_factor=1.25     # 125% of ideal capacity
moe_capacity_min=0.5         # Min capacity floor
moe_capacity_max=2.0         # Max capacity ceiling
```

**Effect:** Penalizes exceeding capacity but allows it (smooth, differentiable).

### Entropy Regularization (Optional)

Encourage diversity in token-to-expert assignments:

```python
moe_entropy_reg_weight=0.01
```

---

## Expert Types

### SwiGLU (Recommended)

```python
use_swiglu=True
```

Gated linear unit with learnable gate:
```
u = w1(x)
v = w2(x)
output = silu(u) * v
```

vs. standard FFN:
```
output = w2(gelu(w1(x)))
```

**Why SwiGLU:** More expressive, used in modern models (PaLM, LLaMA).

### Standard FFN

```python
use_swiglu=False
activation="gelu"  # or "relu", "elu", etc.
```

---

## Shared Experts

### Purpose

Some tokens/features are best handled by a shared expert (always active, not routed). Reduces variance in per-expert gradients.

### Enable

```python
num_shared=1  # Number of shared experts (0 = none)
```

**Effect:** 1 shared expert + 8 routed experts = 9 total expert computations.

### Shared Expert Combination

```python
shared_combine="add"  # or "concat"
shared_scale_init=1.0
```

How routed output combines with shared output:
- `"add"`: `output_routed + output_shared`
- `"concat"`: `[output_routed, output_shared] → linear`

---

## Latent Projection (Optional)

Project tokens through a latent space before routing. Encourages abstraction.

### Enable

```python
moe_use_latent=True
moe_latent_dim=128           # Latent dimension
moe_latent_d_ff=512          # Hidden dim in latent FFN
```

**Architecture:**
```
Input: [B, T, D]
  ↓
Project to latent: linear(D → L)
  ↓
Latent FFN: [L → latent_d_ff → L]
  ↓
Project back: linear(L → D)
  ↓
Router consumes latent features
  ↓
Experts still operate on original input
```

**Why:** Separate routing logic from expert computation, reduces overfitting of router.

---

## Capacity & Computation Control

### Capacity Factor

```python
moe_capacity_factor=1.25  # Default 125% of ideal
```

Ideal capacity per expert: `(B * T) / num_experts`
Actual capacity: `ideal * moe_capacity_factor`

Tokens exceeding capacity are **dropped** (or soft-penalized if `moe_soft_capacity=True`).

### Mixture-of-Tokens-Per-Expert (MTP)

Track per-expert token counts, predict optimal task difficulty:

```python
mtp_num_heads=2          # Number of MTP heads
mtp_loss_weight=0.01     # Auxiliary loss weight
mtp_init_scale=0.02      # Initialization scale
```

---

## Gradient Checkpointing

Reduce memory footprint for large MoE layers:

```python
use_gradient_checkpointing=True
```

Recomputes expert activations during backward. Trade: 20–30% more compute, 50–70% less memory.

---

## Production Tuning Guide

### For Stable Training

```python
config = {
    "num_experts": 8,
    "num_shared": 1,
    "top_k": 2,
    "router_type": "noisy_topk",
    "router_temperature": 1.0,
    "router_perturb_noise": 0.01,
    "load_balance_weight": 0.01,
    "z_loss_weight": 0.001,
    "moe_aux_lambda": 1.0,
    "use_swiglu": True,
    "use_gradient_checkpointing": True,
}
```

Baseline setting: works well across most datasets.

### For Compute Efficiency

```python
config = {
    "num_experts": 16,
    "top_k": 1,  # Single expert (most sparse)
    "router_type": "hash",  # O(1) routing
    "routing_mode": "expert_choice",
    "expert_choice_tokens_per_expert": 32,
    "use_gradient_checkpointing": True,
}
```

Minimize compute: hash routing + expert choice + k=1.

### For High Accuracy (Research)

```python
config = {
    "num_experts": 32,
    "num_shared": 2,
    "top_k": 4,  # More experts per token
    "router_type": "adaptive_noisy_topk",
    "adaptive_k_sparsity_lambda": 0.01,
    "load_balance_weight": 0.05,
    "z_loss_weight": 0.005,
    "moe_use_latent": True,
    "moe_latent_dim": 256,
    "moe_entropy_reg_weight": 0.01,
    "moe_soft_capacity": True,
}
```

Maximize expressivity: adaptive k + latent + high load balancing weight.

---

## Complete Example: MoE-Heavy Encoder

```python
from foreblocks.config import TrainingConfig
from foreblocks.core.training.trainer import Trainer
from foreblocks.models.transformer.core.encoder import TransformerEncoder

# ── High-capacity MoE encoder ──
encoder = TransformerEncoder(
    input_size=8,
    d_model=384,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    # ── MoE ──
    use_moe=True,
    num_experts=16,
    num_shared=2,
    top_k=2,
    router_type="noisy_topk",
    routing_mode="token_choice",
    load_balance_weight=0.02,
    z_loss_weight=0.001,
    moe_aux_lambda=1.0,
    # ── Latent routing ──
    moe_use_latent=True,
    moe_latent_dim=192,
    moe_latent_d_ff=768,
    # ── Capacity ──
    moe_capacity_factor=1.5,
    # ── Efficiency ──
    use_gradient_checkpointing=True,
    # ── Per-layer dropout (see transformer-advanced.md) ──
    layer_dropout_schedule=LayerDropoutSchedule(
        num_layers=6,
        base_dropout=0.05,
        max_dropout=0.15,
    ),
)

config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=0.01,
    use_llrd=True,
    llrd_decay=0.9,
    scheduler_type="warmup_cosine",
    warmup_ratio=0.1,
    batch_size=64,
)

trainer = Trainer(model=encoder, config=config)
history = trainer.train(train_loader, val_loader)
```

---

## Monitoring MoE Health

### Expert Utilization

Enable logging:
```python
from foreblocks.modules.moe.experts.moe_logging import MoELogger

moe_logger = MoELogger()
encoder = TransformerEncoder(
    ...,
    moe_logger=moe_logger,
    step_getter=lambda: trainer.global_step,
)
```

Access per-step metrics:
```python
if moe_logger:
    expert_counts = moe_logger.expert_token_counts  # [num_experts]
    load_balance = moe_logger.load_balance_loss
    z_loss = moe_logger.z_loss
```

### Debugging Collapse

If all tokens route to the same expert:
1. Increase `load_balance_weight` (0.05–0.1)
2. Enable `moe_router_bias_balance=True`
3. Reduce `router_temperature` (increase sharpness)
4. Increase `num_experts` (more targets)

### Avoiding Overflow

If loss explodes:
1. Reduce `router_perturb_noise`
2. Enable `moe_soft_capacity=True`
3. Reduce `top_k` (fewer experts per token)
4. Increase gradient clipping threshold

---

## Inference Optimization

### Fused Kernels

```python
use_grouped_kernel=True     # Grouped GEMM for dispatching
use_fused_router_topk=True  # Fused top-k operation
```

Reduces kernels launches, improves GPU utilization.

### Torchcompile

```python
compile_router=True
compile_experts=True
```

JIT-compile router and experts for faster execution (PyTorch 2.0+).

### Paged KV Cache

MoE output can be cached in paged KV cache (see Attention docs):
```python
# In transformer decoder with MoE
decoder = TransformerDecoder(
    ...,
    use_moe=True,
    # KV caching transparent to MoE layer
)
```

---

## References

- GShard (Lepikhin et al., 2020): Load balancing in large-scale MoE
- Switch Transformers (Lewis et al., 2021): Simplified MoE with k=1
- Base Layers (Lewis et al., 2023): Shared expert patterns
- Expert Choice Routing (Zhou et al., 2022): Expert-side token selection
