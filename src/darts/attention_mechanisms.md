# Mecanismos de Atenção do DARTS

O DARTS suporta duas famílias de atenção pesquisável: kernels de **auto-atenção** (*self-attention*) e kernels de **atenção cruzada** (*cross-attention*). Cada família possui seu próprio conjunto de modos, modos de codificação posicional e seleção diferenciável via DARTS por meio de `attn_alphas`.

---

## 1. Auto-Atenção (`SelfAttention` — `architecture/self_attention.py`)

Usada dentro dos blocos encoder/decoder de transformers como o submodule `self_attn`.

### Modos de Kernel de Atenção

| Modo | Descrição | Complexidade |
|---|---|---|
| `sdp` | Atenção de produto escalar escalonado (via FlashAttention / `F.scaled_dot_product_attention`) | O(T·S) |
| `linear` | Atenção kernelizada estilo Performer (kernel ELU+1, sem matriz T×S) | O(T+S) |
| `probsparse` | ProbSparse do Informer: seleciona top-u queries pelo escore de esparsidade, preenche o resto com média(V) | O(L log L) |
| `cosine` | CosFormer: Q/K normalizados em L2 + mapa de features ReLU + temperatura inversa aprendível | O(T+S) |
| `local` | Atenção de janela deslizante — tamanho derivado de `LOCAL_WINDOW_RATIO × seq_len` | O(T·W), W ≪ S |
| `auto` | Pesquisarável via DARTS: mistura todos os cinco acima via softmax sobre `attn_alphas` | — |

### Modos de Codificação Posicional

| Modo | Descrição |
|---|---|
| `rope` | Embeddings Posicionais Rotativos (padrão) |
| `alibi` | Bias Linear na Atenção (Press et al. 2021) |
| `none` | Sem codificação posicional |
| `seasonal` | Bias posicional periódico ajustado para sazonalidade de séries temporais (períodos: 4, 8, 16, 24, 48) |
| `sinusoidal` | Codificação posicional sinusoidal padrão (Vaswani et al. 2017) |
| `learned` | Embeddings posicionais aprendíveis |
| `relative` | Bias de posição relativa estilo DeiT (Touvron et al. 2021) |
| `auto` | Pesquisarável via DARTS: mistura todos os sete acima via `position_alphas` |

---

## 2. Ponte de Atenção Cruzada (`AttentionBridge` — `architecture/bridges.py`)

Usada nos blocos decoder como o submodule `cross_attn` (atenção encoder→decoder).

### Modos de Kernel de Atenção

| Modo | Descrição | Complexidade |
|---|---|---|
| `none` | Passagem direta (sem atenção cruzada) | O(1) |
| `sdp` | Atenção de produto escalar escalonado | O(T·S) |
| `linear` | Atenção kernelizada estilo Performer (ELU+1) | O(T+S) |
| `probsparse` | ProbSparse do Informer: seleção de top-u queries | O(L log L) |
| `cosine` | CosFormer: Q/K normalizados em L2 + mapa de features ReLU | O(T+S) |
| `local` | Atenção cruzada de janela deslizante (janela proporcional ao comprimento) | O(T·W) |
| `auto` | Pesquisarável via DARTS: mistura todos os seis acima | — |

### Modos de Codificação Posicional

| Modo | Descrição |
|---|---|
| `rope` | Embeddings Posicionais Rotativos (padrão) |
| `alibi` | Bias Linear na Atenção |
| `none` | Sem codificação posicional |
| `seasonal` | Bias posicional periódico de séries temporais |
| `auto` | Pesquisarável via DARTS: mistura os quatro acima |

---

## 3. Operações no Nível da Célula (`mixed_op.py`)

No nível da célula DARTS, duas operações focadas em atenção podem ser selecionadas como candidatas alternativas à célula:

| Operação | Descrição | Localização |
|---|---|---|
| `PatchEmbed` | Tokenização por patches (estilo ViT) | `architecture/fixed_ops.py` |
| `InvertedAttention` | Atenção estilo iTransformer (atenção sobre canais, não tokens) | `architecture/fixed_ops.py` |

São escolhidas via a família `"attention"` no espaço de busca de operações da célula.

---

## 4. Configuração Padrão

Em `config.py`:

```python
DEFAULT_ATTENTION_VARIANTS: list[str] = ["auto"]
DEFAULT_FFN_VARIANTS: list[str] = ["auto"]
```

Tanto auto-atenção quanto atenção cruzada usam `"auto"` por padrão, permitindo busca diferenciável DARTS sobre o conjunto completo de modos. A codificação posicional usa `"rope"` por padrão (não pesquisável), a menos que o módulo pai defina `"auto"`.

---

## 5. Mecanismo de Busca

Quando `attention_type="auto"`:
- Cada modo recebe um parâmetro aprendível `attn_alphas[i]`
- Durante o treino: softmax sobre os alphas (escalonado por temperatura); com `variant_gdas=True`, Gumbel-Softmax com estimador de gradiente direto (*straight-through*)
- Durante inferência/avaliação: argmax sobre as probabilidades softmax

Quando um modo específico (ex. `"sdp"`) é definido:
- O submodule correspondente é fixado; `attn_alphas` é descartado
- `_freeze_transformer_self_attention()` / `_freeze_transformer_cross_attention()` aplicam o fixamento na finalização

---

## 6. Tabela Resumo

```
                          Auto-Atenção               Ponte de Atenção Cruzada
  Kernel    O(T,S)        sdp, linear, probsparse  sdp, linear, probsparse
                                cosine, local        cosine, local, none
                                auto (pesquisarável) auto (pesquisarável)

  Codif.    O(S)            rope*, alibi, none
  Posicional              seasonal, sinusoidal, learned,
                                relative, auto
```

`*` = padrão para ambos os módulos.
