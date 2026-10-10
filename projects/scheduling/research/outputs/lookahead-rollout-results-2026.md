# Lookahead Rollout Search: Experimental Results

## Finding: LRBS-style lookahead does NOT improve over single-step PPO on ONTS

### Setup

- **Instances**: 6 development (seed 42, regimes balanced/energy/periodic) + 12 OOD
- **Training**: 64 episodes × 2 seeds, train budget 0.8s, eval budget 2.0s
- **Methods compared**:
  - `uniform` — random action selection (no policy)
  - `ppo_single` — single-step PPO trained and evaluated with single-step rollout
  - `ppo_lookahead_k4_s2` — lookahead-trained, lookahead-evaluated (k=4, steps=2)
  - `ppo_lookahead_k8_s2` — same, k=8, steps=2
  - `ppo_lookahead_k8_s3` — same, k=8, steps=3

### Development Results

| Method | Mean Improvement | Std | Median | N |
| --- | --- | --- | --- | --- |
| uniform | 0.287 | 0.130 | 0.278 | 6 |
| **ppo_single** | **0.331** | 0.104 | 0.362 | 6 |
| ppo_lookahead_k4_s2 | 0.276 | 0.098 | 0.288 | 6 |
| ppo_lookahead_k8_s2 | 0.259 | 0.079 | 0.288 | 6 |
| ppo_lookahead_k8_s3 | 0.246 | 0.077 | 0.251 | 6 |

### Paired Comparisons (Development)

| Comparison | Mean Diff | 95% CI | Significance |
| --- | --- | --- | --- |
| lookahead_k4 vs ppo_single | **-0.056** | [-0.099, -0.025] | **Significant** |
| lookahead_k8 vs ppo_single | **-0.072** | [-0.108, -0.031] | **Significant** |
| lookahead_k4 vs uniform | -0.011 | [-0.102, 0.185] | NS |
| lookahead_k8 vs uniform | -0.028 | [-0.105, 0.185] | NS |

### Key Observations

1. **Lookahead only makes 1-2 decisions** per instance (vs 18-47 for single-step)
   - The lookahead selection phase consumes ~95% of the budget running K rollouts
   - Each rollout evaluation requires MILP solves + policy forward passes
   
2. **Single-step PPO is SIGNIFICANTLY better than lookahead** (p<0.05 via paired bootstrap)
   - Mean improvement: 0.331 vs 0.276 (k=4), 0.259 (k=8)
   - The confidence intervals are entirely negative

3. **Increasing lookahead depth/k does not help** — k=8, steps=3 is WORSE than k=4, steps=2
   - More expensive selection → fewer total decisions → worse performance

4. **Both single-step and lookahead beat uniform**, confirming RL value
   - But the RL advantage disappears when you add lookahead overhead

### Interpretation

ONTS benefits from **many cheap repair decisions** rather than fewer expensive informed ones:
- Each MILP repair takes ~10-50ms, allowing ~20-100 repairs per episode
- Lookahead evaluation of a single candidate requires multiple MILP solves + rollouts (~300-800ms)
- The policy's learned knowledge is better used for rapid sequential decisions than for expensive pre-selection

This contradicts the Camerota et al. (AAAI 2025) findings on job-shop scheduling, where lookahead helped. The difference likely stems from:
- ONTS has a **much larger action space** (repair operators vs. job operations)
- ONTS MILP repairs are **cheap relative to rollout evaluation cost**
- ONTS problem structure makes **greedy single-step decisions nearly as good** as informed multi-step selection

### Conclusion for Paper

**LRBS-style lookahead rollout search is not beneficial for ONTS.** The overhead of evaluating multiple rollouts per candidate decision consumes the search budget, leaving too few actual repair actions. Single-step PPO with rapid sequential repairs outperforms all lookahead variants by a statistically significant margin.

This negative result helps bound the design space for RL-based ONTS solvers and suggests that future work should focus on:
- Better action representations (not more expensive selection)
- Parallel candidate evaluation (not sequential rollouts)
- Cheaper rollout approximations (not deeper lookaheads)
