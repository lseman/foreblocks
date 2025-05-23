# Transformer Architecture

## Transformer Encoder Layer (Pre-Norm)
```
                   Input
                     |
                     v
         +-------------------------+
         |      LayerNorm 1        |
         +-------------------------+
                     |
                     v
         +-------------------------+
         |    Self-Attention       |<------- [Attention Mask]
         |    (XFormerAttention    |<------- [Padding Mask]
         |     or FreqAttention)   |
         +-------------------------+
                     |
                     |     +----------------+
                     +---->|    Dropout     |
                     |     +----------------+
                     |              |
         +-----------+              |
         |                          v
         |                   +-----------+
         |                   |  Residual |
         |                   +-----------+
         |                          |
         |                          v
         |           +-------------------------+
         |           |      LayerNorm 2        |
         |           +-------------------------+
         |                          |
         |                          v
         |           +-------------------------+
         |           |    Feed-Forward Network |
         |           |    (SwiGLU or Standard) |
         |           +-------------------------+
         |                          |
         |                          |
         |                          v
         |                   +----------------+
         |                   |    Dropout     |
         |                   +----------------+
         |                          |
         +------------------------->+
                                    |
                                    v
                                  Output


```

## Transformer Decoder Layer (Pre-Norm)
```
                   Input (tgt)             Memory (from Encoder)
                      |                           |
                      v                           |
          +-------------------------+             |
          |      LayerNorm 1        |             |
          +-------------------------+             |
                      |                           |
                      v                           |
          +-------------------------+             |
          |     Self-Attention      |<------ [Tgt Mask]
          |    (XFormerAttention)   |<------ [Padding Mask]
          +-------------------------+             |
                      |                           |
                      |     +----------------+    |
                      +---->|    Dropout     |    |
                      |     +----------------+    |
                      |              |            |
          +-----------+              |            |
          |                          v            |
          |                    +-----------+      |
          |                    |  Residual |      |
          |                    +-----------+      |
          |                          |            |
          |                          v            |
          |           +-------------------------+ |
          |           |      LayerNorm 2        | |
          |           +-------------------------+ |
          |                          |            |
          |                          v            |
          |           +-------------------------+ |
          |           |    Cross-Attention      |<+
          |           |    (XFormerAttention)   |<------ [Memory Mask]
          |           +-------------------------+<------ [Memory Padding Mask]
          |                          |
          |                          |
          |                          v
          |                   +----------------+
          |                   |    Dropout     |
          |                   +----------------+
          |                          |
          +------------------------->+
                                     |
                                     v
                               +-----------+
                               |  Residual |
                               +-----------+
                                     |
                                     v
                        +-------------------------+
                        |      LayerNorm 3        |
                        +-------------------------+
                                     |
                                     v
                        +-------------------------+
                        |    Feed-Forward Network |
                        |    (SwiGLU or Standard) |
                        +-------------------------+
                                     |
                                     v
                                +----------------+
                                |    Dropout     |
                                +----------------+
                                     |
                                     v
                               +-----------+
                               |  Residual |
                               +-----------+
                                     |
                                     v
                                   Output
```

## Complete Transformer Architecture
```
                              Input Sequence
                                    |
                                    v
                        +----------------------+
                        |   Input Projection   |
                        +----------------------+
                                    |
                                    v
                        +----------------------+
                        | Positional Encoding  |
                        +----------------------+
                                    |
                                    v
                        +----------------------+
                        |       Dropout        |
                        +----------------------+
                                    |
                                    v
                     +-----------------------------+
                     |                             |
                     |    Transformer Encoder      |
                     |    (Multiple Layers)        |
                     |                             |
                     +-----------------------------+
                                    |
                                    |          Target Sequence
                                    |                |
                                    |                v
                                    |     +----------------------+
                                    |     |   Input Projection   |
                                    |     +----------------------+
                                    |                |
                                    |                v
                                    |     +----------------------+
                                    |     | Positional Encoding  |
                                    |     +----------------------+
                                    |                |
                                    |                v
                                    |     +----------------------+
                                    |     |       Dropout        |
                                    |     +----------------------+
                                    |                |
                                    v                v
                     +-----------------------------+
                     |                             |
                     |    Transformer Decoder      |
                     |    (Multiple Layers)        |
                     |                             |
                     +-----------------------------+
                                    |
                                    v
                        +----------------------+
                        |   Output Projection  |
                        +----------------------+
                                    |
                                    v
                             Output Sequence
```

## Modern Features in Implementation

```
╔══════════════════════════════════════════════╗
║   Modern Transformer Architecture Features   ║
╚══════════════════════════════════════════════╝

┌────────────────────────────────────────────┐
│ ◆ Normalization Strategy                   │
│   ├── Pre-Norm (better training stability) │
│   └── Post-Norm (original transformer)     │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ ◆ Feed-Forward Network Options             │
│   ├── SwiGLU (improved performance)        │
│   └── Standard MLP with GELU/ReLU          │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ ◆ Attention Mechanisms                     │
│   ├── XFormerAttention (efficient impl)    │
│   ├── FlashAttention support               │
│   └── FrequencyAttention (FEDFormer style) │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ ◆ Training Optimizations                   │
│   ├── Gradient checkpointing               │
│   └── Configurable layer norm epsilon      │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ ◆ Decoder Features                         │
│   ├── Incremental/autoregressive decoding  │
│   └── Informer-style non-autoregressive    │
└────────────────────────────────────────────┘
```