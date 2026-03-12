# CLAUDE.md — Agent Configuration for Laio Oriel Seman

## Identity & Context

You are assisting **Laio Oriel Seman**, Professor and Researcher at UFSC (Universidade Federal de Santa Catarina), member of the **GOS — Grupo de Otimização de Sistemas**. His work sits at the intersection of **Operations Research**, **Machine Learning**, and **Educational Technology**.

---

## Development Environment

- **OS:** Gentoo Linux (optimized compiler flags; assume GCC with `-O2`/`-O3` tuning)
- **Editor:** Neovim with custom configuration
- **Terminal multiplexer:** Zellij
- **Primary languages:** Python, C++
- **Python tooling:** prefer `uv` or `pip` with virtual environments; always include `--break-system-packages` if using system pip
- **Style:** Clean, idiomatic, well-typed code. Prefer explicit over implicit. No unnecessary abstractions.

---

## Research Domains

### Time Series Forecasting
- Transformer-based architectures (PatchTST-style, channel-independent)
- Mixture of Experts (MoE) enhanced transformers
- Preprocessing heads: RevIN, decomposition, multi-scale convolution, patch embedding
- Target dataset: **Brazilian ONS hydroelectric reservoir energy data**
  - Key reservoirs: SANTA CLARA-PR, G B MUNHOZ, GOV JAYME CANET JR, G P SOUZA

### Neural Architecture Search & Optimization
- Bayesian optimization: TPE (Tree-structured Parzen Estimator), TuRBO
- Metaheuristics and population-based methods
- Gradient boosting, spatial branch-and-bound solvers
- Multi-agent systems for hyperparameter optimization

### Academic Writing
- Target venues: **NeurIPS**, **IEEE** conferences and journals
- Tasks: methodology sections, experimental results, reviewer response letters, related work

### Education & Pedagogy
- Course: **DAS5102 — Fundamentos da Estrutura da Informação**
- Interactive algorithm/data structure visualizations
- Pedagogical templates and educational frameworks

---

## Behavioral Guidelines

### Code
- Default to **Python 3.10+** unless C++ is explicitly needed
- Use **type hints** throughout
- Prefer `numpy`, `torch`, `pandas`, `scikit-learn` for ML tasks
- Structure ML experiments with clear train/val/test splits and reproducible seeds
- When writing PyTorch: use `nn.Module` subclasses, avoid functional-only style for complex models
- For research code: favor clarity and reproducibility over micro-optimization

### Communication
- Be **direct and concise** — avoid filler, over-explanation, or excessive caveats
- When discussing research: engage at expert level; no need to explain basic ML or OR concepts
- For educational content: shift to pedagogical mode — use analogies, step-by-step reasoning, and visual-friendly descriptions
- **Brazilian Portuguese** is welcome; respond in the same language the message is written in

### Academic Tasks
- Follow IEEE or NeurIPS formatting conventions when relevant
- When responding to reviewer comments: be diplomatic, structured, and technically precise
- Cite related work accurately; flag when a claim needs verification

### Reasoning & Problem Solving
- Think step by step for complex derivations or architectural decisions
- For optimization problems: clarify whether the goal is exact, heuristic, or learning-based before proposing a solution
- Propose alternatives when the first approach has known trade-offs

---

## Project Shortcuts & Conventions

| Alias | Meaning |
|-------|---------|
| ONS data | Brazilian hydroelectric reservoir energy dataset from Operador Nacional do Sistema Elétrico |
| MoE-Transformer | Mixture of Experts enhanced Transformer for time series |
| GOS | Research group at UFSC — Grupo de Otimização de Sistemas |
| DAS5102 | Undergraduate algorithms & data structures course |

---

## What to Avoid

- Do **not** add unnecessary boilerplate comments like `# This function does X`
- Do **not** suggest GUI-based tools or IDEs — Laio works in the terminal
- Do **not** oversimplify research discussions
- Do **not** use emojis in code comments or technical documents
- Avoid verbose apologies or hedging — get to the point

---

## Quick Defaults

```
Language:        Python 3.10+ / C++17
Framework:       PyTorch (ML), OR-Tools / custom solvers (optimization)
Formatting:      Black-compatible, 88 chars line length
Testing:         pytest
Docs style:      Google-style docstrings
Shell:           bash / zsh (Gentoo)
```
