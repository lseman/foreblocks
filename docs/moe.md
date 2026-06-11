---
title: Mixture of Experts (MoE) Guide
description: MoE integration in the transformer stack — experts, routers, gating, and configuration.
editLink: true
---


[[toc]]
# MoE Guide

ForeBlocks integrates Mixture-of-Experts into the transformer feedforward path through `MoEFeedForwardDMoE`.

You typically do not instantiate this block directly. Instead, you enable MoE through transformer constructor arguments.

Related docs:

- [Documentation Overview](overview)
- [Getting Started](getting-started)
- [Transformer](transformer)
- [Custom Blocks](custom_blocks)

![MoE](imgs/moe_architecture_diagram.svg)

## How MoE is enabled

```python
from foreblocks import TransformerEncoder, TransformerDecoder

encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    nhead=8,
    num_layers=4,
    use_moe=True,
    num_experts=8,
    top_k=2,
)

decoder = TransformerDecoder(
    input_size=1,
    output_size=1,
    d_model=256,
    nhead=8,
    num_layers=4,
    use_moe=True,
    num_experts=8,
    top_k=2,
)
```text

## Recommended presets

### Stable baseline

```python
encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    nhead=8,
    num_layers=4,
    use_moe=True,
    num_experts=8,
    num_shared=1,
    top_k=2,
    router_type="noisy_topk",
    routing_mode="token_choice",
    z_loss_weight=1e-3,
    moe_aux_lambda=1.0,
)
```toml

### Higher-capacity experimental setup

```python
encoder = TransformerEncoder(
    input_size=8,
    d_model=384,
    nhead=8,
    num_layers=6,
    use_moe=True,
    num_experts=16,
    num_shared=2,
    top_k=2,
    routing_mode="expert_choice",
    moe_capacity_factor=1.5,
    z_loss_weight=1e-3,
    use_gradient_checkpointing=True,
)
```toml

## Advanced features in the current implementation

### Adaptive top-k

`adaptive_noisy_topk` can vary the effective number of experts selected per token.

This path also tracks per-token `k` statistics and supports a REINFORCE-style adaptive-k loss internally.

### Hash routers

`hash_topk` and `multi_hash_topk` are available when you want routing diversity without a standard learned dense router over all experts.

### Grouped expert kernel path

The implementation can use grouped expert kernels and fused top-k routing in favorable runtime conditions.

You usually do not need to tune these first. They are lower-level performance details rather than primary modeling controls.

### MTP heads inside MoE

The MoE block supports optional multi-token-prediction heads:

- `mtp_num_heads`
- `mtp_loss_weight`

This is an advanced decoder-side path and should be treated as research functionality, not a default production setting.

## Integration with `ForecastingModel`

```python
from foreblocks import ForecastingModel

model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    forecasting_strategy="transformer_seq2seq",
    model_type="transformer",
    target_len=24,
    output_size=1,
)
