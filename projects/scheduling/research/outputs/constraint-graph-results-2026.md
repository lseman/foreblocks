# Constraint-Aware Graph Study — Results

## Objective

Test whether policies that can learn to relax constraints strategically outperform policies that only select tasks from fixed candidate sets. This experiment extends the bipartite MILP graph representation to include constraint relaxability features, allowing the policy to learn which constraints are "soft" vs "hard" and potentially relax them.

## Design

### Constraint-Aware Graph Builder

Extends `MILPGraphBuilder` from `graph_converter.py` with:

1. **Per-constraint metadata**: Each constraint node gets 5 extra features:
   - Relaxable flag (1.0 if constraint can be relaxed)
   - Normalized relaxation cost (penalty per unit violation)
   - Constraint tightness (0=loose, 1=tight)
   - Constraint type one-hot (hard/soft/preference)

2. **Global context features**: Added to problem-level summary:
   - Fraction of relaxable constraints
   - Average relaxation cost
   - Constraint type distribution (hard/soft/preference counts)

3. **Action space metadata**: Observation includes:
   - `constraint_relaxable`: Boolean mask [1, m] indicating which constraints can be relaxed
   - `constraint_relaxation_cost`: Float tensor [1, m] with per-constraint relaxation costs

### Constraint Classification

For ONTS instances, constraints are classified as:
- **Hard** (first 1/3): Energy balance, resource limits — cannot be relaxed
- **Soft** (middle 1/3): SOC margins, duty cycle — can be relaxed with penalty
- **Preference** (last 1/3): Task ordering preferences — low relaxation cost

## Results

### Graph Construction Success

All 18 evaluations (6 dev + 12 OOD × 2 seeds) completed successfully:

| Metric | Value |
| --- | --- |
| Total instances evaluated | 18 |
| Successful graph constructions | 18 (100%) |
| Average runtime per instance | ~3.5s |

### Graph Structure

| Property | Value |
| --- | --- |
| Node feature dimension | 22 (16 base + 6 extra) |
| Variable nodes | Varies by instance (e.g., 624 for seed=987001) |
| Constraint nodes | Varies by instance (e.g., 1752 for seed=987001) |
| Total nodes | e.g., 2373, 5284, 5617 |
| Global context dimension | 32 |
| Constraint relaxability metadata | Present in all graphs |

### Placeholder Improvements

Note: The current study script does not train or evaluate a policy — it only verifies that constraint-aware graphs can be constructed successfully. The "improvement" values are placeholders (random uniform [0, 0.1]).

| Split | N | Mean Improvement | Std |
| --- | --- | --- | --- |
| Development | 6 | 0.052 | 0.023 |
| OOD | 12 | 0.048 | 0.021 |

## Key Findings

1. **Constraint-aware graphs build successfully**: The extended graph representation with relaxability features works correctly for all ONTS instances tested.

2. **Feature dimension mismatch resolved**: By padding variable and constraint features to the same dimension before concatenation, the graph can be used with standard NCO policies (after policy retraining).

3. **Runtime overhead**: Graph construction takes ~1-6 seconds per instance due to the large number of constraints (1752+ for typical ONTS instances). This is acceptable for offline training but may be too slow for online inference.

## Limitations

1. **No policy training/evaluation**: The current study only verifies graph construction. A proper evaluation would require:
   - Training a policy that can select both tasks AND which constraints to relax
   - Comparing against baselines (uniform, PPO single-step, MILP-only)

2. **Simplified constraint classification**: The current classification (1/3 hard, 1/3 soft, 1/3 preference) is arbitrary. A better approach would use actual ONTS constraint metadata to determine which constraints are truly relaxable.

3. **No relaxation mechanism implemented**: The graph includes relaxability features but doesn't implement the actual constraint relaxation mechanism (i.e., removing constraints from the MILP during solving).

## Next Steps

1. **Implement constraint relaxation action**: Extend the policy's action space to include both task selection AND constraint relaxation decisions.

2. **Train constraint-aware policy**: Train a policy that learns when to relax constraints strategically, using the constraint-aware graph representation.

3. **Compare against baselines**: Evaluate the trained policy against uniform, PPO single-step, and MILP-only baselines on the same instances.

4. **Ablation study**: Test whether:
   - Removing relaxability features degrades performance
   - Different constraint classification strategies affect results
   - Relaxation cost sensitivity affects policy behavior

## Files Created

- `/home/seman/dev/foreblocks/projects/scheduling/constraint_aware_graph.py` — Constraint-aware graph builder
- `/home/seman/dev/foreblocks/projects/scheduling/research/rl_constraint_graph_study.py` — Study script
