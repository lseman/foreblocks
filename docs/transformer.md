# Transformer Guide

Current transformer stack in ForeBlocks (`TransformerEncoder`, `TransformerDecoder`) with configurable attention routing, normalization, patching, MoE, and dynamic layer skipping.

Related docs:
- [Custom Blocks](custom_blocks.md)
- [DARTS](darts.md)

---

## Import

```python
from foreblocks import TransformerEncoder, TransformerDecoder
```

---

## Encoder quick start

```python
encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    att_type="standard",
    norm_strategy="pre_norm",
    custom_norm="rms",
    patch_encoder=True,
    patch_len=16,
    patch_stride=8,
)
```

---

## Decoder quick start

```python
decoder = TransformerDecoder(
    input_size=1,
    output_size=1,
    label_len=12,
    informer_like=True,
    model_type="transformer",
    d_model=256,
    nhead=8,
    num_layers=4,
    patch_decoder=False,
)
```

---

## Key options (shared base)

### Model dimensions
- `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `dropout`

### Attention
- `att_type`: base layer attention type (e.g. `standard`, `linear`, `sype`, `kimi`, `gated_delta`)
- `attention_mode`: routing policy across layers (`standard`, `linear`, `sype`, `hybrid`, `kimi`, `hybrid_kimi`, `kimi_3to1`, `gated_delta`, `hybrid_gdn`, `gdn_3to1`)

### Normalization
- `norm_strategy`: `pre_norm` or `post_norm`
- `custom_norm`: `rms`, `layer`, or adaptive variants supported by the norm factory

### Patching
- `patch_encoder`, `patch_decoder`
- `patch_len`, `patch_stride`, `patch_pad_end`

### Efficiency
- `use_gradient_checkpointing`
- `share_layers`
- `use_final_norm`
- `use_swiglu`

### Optional advanced modules
- `use_moe`, `num_experts`, `top_k`
- `use_layer_skipping`, `layer_skip_mode`, `layer_skip_temperature`, `layer_skip_lambda`
- `use_mhc`, `mhc_n_streams`, `mhc_sinkhorn_iters`, `mhc_temperature`, `mhc_collapse`

---

## Informer-like mode

- Set `model_type="informer-like"` on encoder/decoder.
- Decoder enables informer-style behavior (`informer_like=True`) and time encoding defaults.
- Keep `label_len` consistent with your decoder prompt strategy.

---

## CT-PatchTST encoder mode

Encoder supports channel-token PatchTST-style tokenization:

```python
encoder = TransformerEncoder(
    input_size=8,
    ct_patchtst=True,
    ct_patch_len=16,
    ct_patch_stride=8,
    ct_patch_fuse="linear",   # or "mean"
    d_model=256,
)
```

Useful when long `T` would otherwise exceed `max_seq_len`.

---

## Integration with `ForecastingModel`

```python
from foreblocks import ForecastingModel, TransformerEncoder, TransformerDecoder

encoder = TransformerEncoder(input_size=8, d_model=128, nhead=4, num_layers=3)
decoder = TransformerDecoder(input_size=1, output_size=1, d_model=128, nhead=4, num_layers=3)

model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    forecasting_strategy="transformer_seq2seq",
    model_type="transformer",
    target_len=24,
    output_size=1,
)
```

---

## Troubleshooting

- Sequence-too-long errors: increase `max_seq_len` or enable patching.
- Decoder/encoder mask mismatch: keep patching strategy consistent and pass correct masks.
- OOM: reduce `d_model`/`num_layers`, or enable `use_gradient_checkpointing=True`.
