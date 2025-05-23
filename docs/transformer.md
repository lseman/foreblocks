# Transformer Architecture

## Transformer Encoder Layer (Pre-Norm)

```
                   Input
                     |
                     v
         +-------------------------+
         |     AdaptiveNorm 1      |   ← RMSNorm / LayerNorm
         +-------------------------+
                     |
                     v
         +-------------------------+
         |    Self-Attention       |  ← XFormer / Flash / Frequency
         +-------------------------+
                     |
         +-------------------------+
         |        Dropout          |
         +-------------------------+
                     |
         +-----------+-------------+
                     ↓
               Residual Add
                     ↓
         +-------------------------+
         |     AdaptiveNorm 2      |
         +-------------------------+
                     |
                     v
         +-------------------------+
         |  Feed-Forward Network   |  ← SwiGLU / MoE / MLP
         +-------------------------+
                     |
         +-------------------------+
         |        Dropout          |
         +-------------------------+
                     |
         +-----------+-------------+
                     ↓
               Residual Add
                     ↓
                  Output
```

---

## Transformer Decoder Layer (Pre-Norm)

```
                   Input (tgt)                Memory (enc_out)
                      |                             |
                      v                             |
          +-------------------------+               |
          |     AdaptiveNorm 1      |               |
          +-------------------------+               |
                      |                             |
                      v                             |
          +-------------------------+               |
          |     Self-Attention      |  ← causal      |
          +-------------------------+               |
                      |                             |
          +-------------------------+               |
          |        Dropout          |               |
          +-------------------------+               |
                      |                             |
              +-------+-------+                     |
                      ↓                             |
                Residual Add                        |
                      ↓                             |
          +-------------------------+               |
          |     AdaptiveNorm 2      |               |
          +-------------------------+               |
                      |                             |
                      v                             |
          +-------------------------+               |
          |     Cross-Attention     | <-------------+
          +-------------------------+
                      |
          +-------------------------+
          |        Dropout          |
          +-------------------------+
                      |
              +-------+-------+
                      ↓
                Residual Add
                      ↓
          +-------------------------+
          |     AdaptiveNorm 3      |
          +-------------------------+
                      |
                      v
          +-------------------------+
          |  Feed-Forward Network   |  ← SwiGLU / MoE / MLP
          +-------------------------+
                      |
          +-------------------------+
          |        Dropout          |
          +-------------------------+
                      |
              +-------+-------+
                      ↓
                Residual Add
                      ↓
                   Output
```

---

## Complete Transformer Workflow

```
                    Input Sequence [B, T_enc, input_size]
                                |
                                v
                  +-------------------------------+
                  |     Input Projection (Enc)     |
                  +-------------------------------+
                                |
                                v
                  +-------------------------------+
                  |     Positional Encoding (Enc)  |
                  +-------------------------------+
                                |
                                v
                  +-------------------------------+
                  |            Dropout             |
                  +-------------------------------+
                                |
                                v
                  +-------------------------------+
                  |     Transformer Encoder        |
                  |      (Stacked Layers)          |
                  +-------------------------------+
                                |
                                v
                    Encoded Memory [B, T_enc, D]
                                |
        ┌────────────────────────────────────────────────────────┐
        │                                                        │
        ▼                                                        ▼
  Last Encoder Input (Start Token)                     Target Length: pred_len
        │                                                        │
        ▼                                                        ▼
  +------------------+                                   +------------------+
  | Replicate Start  |         → dec_input: [B, T_pred, input_size]        |
  +------------------+                                   +------------------+
                                |
                                v
                  +-------------------------------+
                  |     Input Projection (Dec)     |
                  +-------------------------------+
                                |
                                v
                  +-------------------------------+
                  |     Positional Encoding (Dec)  |
                  +-------------------------------+
                                |
                                v
                  +-------------------------------+
                  |            Dropout             |
                  +-------------------------------+
                                |
                                v
                  +-------------------------------+
                  |     Transformer Decoder        |
                  |      (Stacked Layers)          |
                  +-------------------------------+
                                |
                                v
                  +-------------------------------+
                  |       Output Projection        |
                  +-------------------------------+
                                |
                                v
                     Output Sequence [B, T_pred, output_size]
```

---

## Modern Features in Implementation

```
╔══════════════════════════════════════════════╗
║        Modern Transformer Architecture       ║
╚══════════════════════════════════════════════╝

Normalization Strategies:
  • Pre-Norm (default) for stability
  • Adaptive LayerNorm or Adaptive RMSNorm

Feed-Forward Options:
  • SwiGLU for gated nonlinearity
  • MoE support with top-k experts and aux loss
  • Traditional GELU/ReLU MLPs

Attention Mechanisms:
  • Memory-efficient XFormerAttention
  • FlashAttention for speedup
  • Frequency Attention (FEDformer-style)

Decoder Modes:
  • Full autoregressive decoding
  • Informer-style decoding using repeated start token

Training Features:
  • Custom weight init per module
  • Gradient checkpointing support
  • LayerNorm epsilon configuration
```

---

Let me know if you want this exported to LaTeX diagrams, Mermaid.js, or Plotly/SVG for slide decks or documentation.

