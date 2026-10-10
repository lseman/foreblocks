# ONTS formulation and correctness arguments

## 1. Parameters and conventions

Let J be the number of tasks and T the horizon length, with slots t=0,...,T−1.
Let w_j be priority, p_j≥0 task power, P_t≥0 solar power, and V,Q,η>0 the
battery voltage, capacity and update efficiency. Define k=η/(60QV). A slot
lasts one minute. Let s_init∈[s_min,1], where 0≤s_min≤1. The instantaneous
battery contribution is limited to I_max V, where I_max≥0.

Startup counts satisfy a_j≤number_of_starts≤b_j. Runs have minimum d_j≥1
and maximum D_j≥d_j. Minimum startup separation is l_j≥1; maximum periodic
window length is L_j≥l_j. Execution is permitted for r_j≤t<h_j, including
windows extending beyond the finite horizon. Constraints apply to existing
slots. All integer parameters are integral and task arrays have length J.

The default model permits a run starting at t to be shorter than d_j only when
it reaches the horizon: its required length is min(d_j,T−t). The strict model
forbids starts with t+d_j>T. Maximum periodicity is imposed on full L_j-length
windows; if L_j>T it imposes no startup requirement. Minimum separation remains
active when l_j>T, permitting at most one start within the horizon.

These conventions are explicit corrections/normalizations of the reference:
JSON powers are used directly rather than `int(value)*10`; unaudited cuts are
excluded; the minimum-separation constraint remains active when its window is
longer than the horizon. Results refer to this specified model, not to literal
execution of the buggy script with its optional cuts.

## 2. Sparse MILP

Variables are x_jt,φ_jt∈{0,1} and s_t∈[s_min,1]. Let x_j,−1=0.
The objective is max Σ_j,t w_j x_jt. The constraints are:

1. φ_jt≥x_jt−x_j,t−1, φ_jt≤x_jt, and for t>0,
   φ_jt≤1−x_j,t−1. At t=0, the first two imply φ_j0=x_j0.
2. a_j≤Σ_t φ_jt≤b_j. Fix x_jt=0 outside [r_j,h_j).
3. For every full l_j-length window, Σ φ_jt≤1. If l_j>T, impose
   Σ_t φ_jt≤1. For every full L_j-length window, Σ φ_jt≥1.
4. For each permitted start t, Σ_{u=t}^{t+q−1} x_ju≥qφ_jt,
   where q=min(d_j,T−t) in the truncated model. In the strict model q=d_j
   and φ_jt=0 when t+d_j>T.
5. For every full (D_j+1)-length window, Σ x_jt≤D_j.
6. Σ_j p_j x_jt≤P_t+I_max V.
7. $s_t-s_{t-1} \le k(P_t-\sum_j p_j x_{jt})$, where $s_{-1}=s_{\mathrm{init}}$.

Constraint 7 permits discarding available net power. It reproduces the
reference's upper bound on battery current; it does not impose equality when
charging would exceed full capacity. The factor η is the symmetric update
factor used in the reference. A battery model with separate charge/discharge
efficiencies would require a different formulation and evaluation.

## 3. Proposition 1: discrete scheduling equivalence

For a binary execution schedule x, constraints 1–5 hold if and only if its
actual off-to-on startups, run lengths, window membership, startup counts and
periodicity satisfy the conventions in Section 1.

**Proof.** For binary x, if x_jt=0 then φ_jt≤x_jt forces φ_jt=0. If
x_jt=1 and x_j,t−1=0, the lower bound forces φ_jt=1. If both execution
variables are one, the previous-execution upper bound forces φ_jt=0. At the
first slot the initial off state yields the same result. Thus startup
indicators are uniquely determined. Constraints 2 directly impose counts and
windows.

Two starts separated by fewer than l_j slots lie in a common full
l_j-length window when l_j≤T. Conversely two starts in such a window differ
by at most l_j−1. When l_j>T, every pair is too close, so at most one start
is permitted. Maximum periodicity is exactly the stated full-window condition.

Because execution variables are binary and a sum of q variables cannot exceed
q, constraint 4 forces all q slots following a start to be active. This is
exactly the required minimum run, with the chosen boundary convention. A run
longer than D_j contains an all-active (D_j+1)-length window, violating
constraint 5. If no run is longer than D_j, every such window contains an off
slot. These arguments prove both directions. □

## 4. Proposition 2: energy feasibility by maximal state trajectory

For a fixed execution schedule, let u_t=Σ_j p_j x_jt and define

$$
m_{-1}=s_{\mathrm{init}},\qquad
m_t=\min\{1,m_{t-1}+k(P_t-u_t)\}.
$$

A state trajectory satisfying constraints 6–7 and the state bounds exists if
and only if the instantaneous power limits hold and m_t≥s_min for every t.

**Proof.** For any feasible trajectory s, induction gives s_t≤m_t. It holds
at the initial state. Given $s_{t-1}\le m_{t-1}$, constraint 7 and the upper state
bound imply
$$s_t\le\min\{1,s_{t-1}+k(P_t-u_t)\}\le m_t.$$
If any m_t<s_min, no feasible s_t can meet the lower bound. Conversely, if
all m_t≥s_min, the trajectory s=m lies in [s_min,1] and satisfies constraint
7 by construction. Constraint 6 is checked separately. Thus the condition is
necessary and sufficient. □

This argument explains why the validator caps charge at one but never clamps
an energy deficit upward to s_min. Such clamping would conceal infeasibility.

## 5. Corollary: schedule validation and model feasibility

Under Section 1's assumptions, in exact arithmetic, the production validator's
feasibility predicate is equivalent to the sparse base MILP's feasibility for
a fixed binary execution schedule. The objective values coincide by definition.
The implementation uses floating-point arithmetic and specified tolerances;
finite tests cannot remove numerical limitations.

The independent exhaustive oracle uses change-point runs and sentinel startup
gaps rather than the production checker's run scan and sliding windows. For
starts u_1<...<u_m, a startup-free full L-length window exists exactly when a
gap in the sequence (−1,u_1,...,u_m,T) exceeds L, for L≤T.

## 6. Proposition 3: incumbent preservation

Given a validated incumbent x with objective f(x), each accepted repair leaves
the objective unchanged or increases it, and leaves the returned incumbent
feasible, including when a solver times out without a new solution.

**Proof.** The solver result is accepted only after binary and complete-schedule
validation. The implementation compares it with the incumbent and retains the
incumbent whenever the new candidate is absent or has lower objective. Induction
over repair iterations yields the result. □

This is a non-degradation property, not a guarantee of strict improvement.
Restricted-problem bounds apply only to their neighborhoods; they do not prove
a global optimality gap. A time limit without an incumbent is unknown, not a
proof of infeasibility. Learned selection changes the search trajectory, not
the validity of these properties.

## 7. Empirical scope

The local exhaustive audit checks 120 parameterized tiny instances and all
33,280 binary schedules. It compares the independent oracle, production
validator, explicit sparse-row feasibility and the unrestricted solver's
optimum. The result file records every case, seed, boundary convention and
mismatch. Larger-instance performance claims require the separate frozen
benchmark, and cannot follow from these correctness propositions.

## 8. Reference cut counterexample

Consider one task over 12 slots with d=1, D=3, l=1, L=8, startup count bounds
[0,4], execution window [0,12), and zero task draw/solar power. The schedule

```text
x   = 1 0 1 1 0 0 0 0 1 0 1 1
phi = 1 0 1 0 0 0 0 0 1 0 1 0
```

satisfies the base model: run lengths are 1,2,1,2; startup gaps are 2,6,2;
every full 8-slot window contains a startup; the battery can remain at its
initial state. The sixth reference inequality is activated because
D < L−d, namely 3 < 7. At t=0 it imposes φ_0+x_3≤1, but both terms are one.
Thus this inequality is not valid for the base formulation. The saved JSON
counterexample includes all instance fields and the complete schedule.

The supplemental systematic audit contains 24 additional coupled-resource
cases and 73,728 schedules, including 9,640 feasible schedules, with zero
oracle/validator/matrix disagreement and zero optimal-objective mismatches.
Together with the random audit this is 107,008 schedules across 144 cases.
This numerical scope remains distinct from Propositions 1–3's exact-arithmetic
arguments and the larger-instance performance experiment.
