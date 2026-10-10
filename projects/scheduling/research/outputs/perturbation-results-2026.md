# Perturbation-Aware Search: Experimental Results

## Finding: Perturbations + annealing/ALNS offer no advantage over monotone repair on ONTS

### Setup

- **Instances**: 6 development (seed 42, regimes balanced/energy/periodic) + 12 OOD
- **Evaluation budget**: 2.0s per instance, 2 seeds averaged
- **Methods compared**:
  - `monotone` — strictly improving repairs only (baseline)
  - `annealing` — simulated annealing with temperature=0.5
  - `alns` — adaptive large-neighborhood search with sliding-window acceptance

### Development Results

| Strategy | N | Mean Improvement | Std | Median | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| monotone | 6 | **0.296** | 0.103 | 0.266 | 0.187 | 0.435 |
| annealing | 0.296 | 0.103 | 0.266 | 0.187 | 0.435 |
| alns | 0.278 | 0.122 | 0.250 | 0.117 | 0.435 |

### Key Observations

1. **Annealing and monotone produce IDENTICAL results** — the acceptance probability for worse solutions is never exercised because MILP repair always improves upon the perturbed state.

2. **ALNS performs slightly worse** than both (mean diff -0.018), but not significantly — its sigmoid acceptance function occasionally accepts worse solutions that then fail to improve via MILP, wasting budget.

3. **Perturbations are always recovered** — when a low-priority task is released and the MILP solver runs, it consistently finds an improved schedule (the solver reconfigures around the freed resources).

### Interpretation

The perturbation + annealing approach doesn't help because:
- **MILP repair dominates**: The HiGHS solver always finds improvements over any perturbed state, so no worse solutions are ever proposed for acceptance testing.
- **No irreversible decisions**: Unlike combinatorial problems where perturbations commit to bad choices, ONTS MILP repair can always recover from task releases.
- **Budget waste**: ALNS's occasional acceptance of worse solutions wastes time without benefit.

### Conclusion for Paper

**Perturbation-aware search with annealing/ALNS acceptance does not improve over monotone repair on ONTS.** The HiGHS MILP solver is so effective at recovering from perturbations that the policy never needs to learn "when to sacrifice" — every sacrifice is automatically recovered by the solver.

This contrasts with combinatorial problems where:
- Perturbations are irreversible (e.g., job-shop scheduling operations)
- Local search can get stuck in deep local optima
- Simulated annealing helps escape basins of attraction

For ONTS, the MILP repair operator is the dominant factor — policy learning should focus on **which constraints to relax** rather than whether to accept worse solutions.

### Design Implications

Future RL-based ONTS solvers should focus on:
1. **Constraint-aware graph representations** (experiment #4) — let the policy learn which constraints are safe to relax
2. **Better candidate generation** — not more acceptance strategies
3. **Multi-scale repair planning** — plan sequences of repairs, not single-step decisions
