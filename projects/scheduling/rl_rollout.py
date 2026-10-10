"""Lookahead rollout search for ONTS repair policies.

Inspired by LRBS (Camerota et al., AAAI 2025): at each decision point, run K
short rollouts from the current state using the policy, then select the first
action of the rollout with the best total reward. This addresses the myopic
nature of single-step PPO selection where the policy picks an action without
seeing its downstream consequences.

This is an ONTS adaptation, not a reproduction of LRBS's FJSP beam search.
All repair costs, inference, and validation are charged to the same wall-clock
deadline as the controls.
"""
from __future__ import annotations

import copy
from time import perf_counter

import numpy as np
import torch

from .milp import ONTSMilp
from .ranking import SearchState, make_candidates


def _deep_clone_env(env):
    """Create a shallow-enough clone for isolated rollout execution.

    Copies the incumbent schedule, objective, state, log, and problem state.
    Does NOT copy the RNG (rollouts share the same exploration stream) or the
    wall-clock baseline (each rollout measures from its own start).
    """
    clone = object.__new__(type(env))
    clone.problem = env.problem  # Shared; read-only during rollouts
    clone.incumbent = env.incumbent.detach().cpu().clone()
    clone.objective = env.objective
    clone.initial_objective = env.initial_objective
    clone.scale = env.scale
    clone.state = SearchState(
        last_mask=env.state.last_mask.detach().cpu().clone(),
        changed=env.state.changed.detach().cpu().clone(),
        steps=env.state.steps,
        stagnation=env.state.stagnation,
        last_gain=env.state.last_gain,
        last_runtime=env.state.last_runtime,
    )
    clone.log = []
    clone.done = False
    clone.candidates = []
    clone.operator_pulls = env.operator_pulls.copy()
    clone.operator_gains = env.operator_gains.copy()
    clone.operator_durations = env.operator_durations.copy()
    clone.operator_last_used = env.operator_last_used.copy()
    clone.rng = env.rng  # Shared RNG for consistent exploration
    clone.deadline = env.deadline
    clone.clock = env.clock
    clone.started = perf_counter()  # New baseline per rollout
    clone.repair_time = getattr(env, 'repair_time', 0.1)
    clone.budget = getattr(env, 'budget', None)
    clone.reserve = getattr(env, 'reserve', 0.05)
    clone.stop_mode = getattr(env, 'stop_mode', 'optional')
    return clone


def _restore_env(env, clone):
    """Restore the primary environment from a rollout clone's best result."""
    env.incumbent = clone.incumbent.detach().cpu().clone()
    env.objective = clone.objective
    env.state = SearchState(
        last_mask=clone.state.last_mask.detach().cpu().clone(),
        changed=clone.state.changed.detach().cpu().clone(),
        steps=clone.state.steps,
        stagnation=clone.state.stagnation,
        last_gain=clone.state.last_gain,
        last_runtime=clone.state.last_runtime,
    )
    # Merge operator statistics from the best rollout
    env.operator_pulls += clone.operator_pulls
    env.operator_gains += clone.operator_gains
    env.operator_durations += clone.operator_durations
    for i in range(4):
        if clone.state.steps > 0:
            env.operator_last_used[i] = max(env.operator_last_used[i],
                                            clone.state.steps)


def _rollout_one(clone_env, policy, *, max_steps=3, greedy=False, mode="ppo"):
    """Run a single short rollout from clone_env; return (total_reward, actions).

    The rollout executes each action immediately and accumulates reward.
    Returns the total undiscounted return and the sequence of actions taken.
    """
    total_reward = 0.0
    actions = []
    while not clone_env.done:
        if mode == "ppo" and hasattr(policy, "observation"):
            rows, mask = policy.observation(clone_env)
        else:
            rows, mask = clone_env.observe()

        with torch.no_grad():
            if mode == "ppo":
                dist, _ = policy(rows, mask)
                action = dist.logits.argmax() if greedy else dist.sample()
                selected = int(action)
            elif mode == "uniform":
                choices = torch.where(mask)[0].tolist()
                repairs = [a for a in choices if a]
                if not repairs:
                    selected = 0
                else:
                    selected = int(clone_env.rng.choice(repairs))
            else:
                raise ValueError(f"Unknown rollout mode: {mode}")

        reward, done = clone_env.step(selected)
        total_reward += reward
        actions.append(selected)
        if len(actions) >= max_steps and not done:
            # Force stop after max_steps to bound lookahead depth
            break

    return total_reward, actions


def lookahead_select(env, policy, *, k=8, rollout_steps=2, greedy=False,
                     mode="ppo", min_budget_per_rollout=0.05):
    """Select the best first action by running K short rollouts from the current state.

    At each decision point:
    1. Generate candidate actions (same as the base search)
    2. For each candidate, run a short rollout starting with that action
    3. Select the candidate whose rollout achieves the highest total reward
    4. Execute only that first action on the real environment

    Args:
        env: The SearchEnv instance (modified in-place after selection).
        policy: The trained ActorCritic policy for rollout evaluation.
        k: Number of rollouts to run per candidate action. Total rollouts =
           min(num_candidates, top_k) * k.
        rollout_steps: Maximum depth of each short rollout (1-3 recommended).
        greedy: If True, use argmax instead of sampling during rollouts.
        mode: Policy evaluation mode ("ppo", "uniform").
        min_budget_per_rollout: Minimum time reserved for each rollout to avoid
           oversubscription.

    Returns:
        The selected action index (0=STOP, 1..N=candidate indices).
    """
    remaining = env.remaining()
    if remaining < min_budget_per_rollout:
        return 0  # Not enough time for any rollout

    # Generate candidates (same as base search)
    env.refresh()
    candidates = env.candidates
    num_candidates = len(candidates)

    if num_candidates == 0 or remaining < min_budget_per_rollout:
        return 0  # No affordable actions

    # Budget allocation: reserve time for selection + best rollout execution
    # Heuristic: split remaining budget among rollouts, with a ceiling
    max_total_rollout_time = remaining * 0.7  # Leave 30% for execution
    max_rollouts = int(max_total_rollout_time / (min_budget_per_rollout * rollout_steps))
    max_rollouts = min(max_rollouts, num_candidates * k)

    if max_rollouts < 1:
        return 0

    # Sample which candidates to evaluate (top candidates by a cheap heuristic)
    # Use random + utility diversity for initial candidate set
    eval_count = min(num_candidates, max(4, max_rollouts // max(k, 1)))
    if num_candidates > eval_count:
        # Weighted sampling: prefer diverse operators and higher-utility candidates
        weights = np.ones(num_candidates)
        # Slight bias toward utility/run operators (typically higher gain)
        for i, c in enumerate(candidates):
            if c.operator in ("run", "energy"):
                weights[i] *= 1.5
        weights /= weights.sum()
        eval_indices = env.rng.choice(num_candidates, size=eval_count, replace=False,
                                      p=weights)
    else:
        eval_indices = list(range(num_candidates))

    # Run rollouts: for each candidate, run k short trajectories
    best_reward = -float('inf')
    best_action = 0
    best_actions_taken = []
    total_rollout_time = 0.0

    for idx in eval_indices:
        for r in range(k):
            if total_rollout_time > max_total_rollout_time:
                break

            start_time = perf_counter()
            clone = _deep_clone_env(env)
            # Execute the candidate action on the clone first
            result = clone.problem.solve(
                time_limit=min(env.repair_time, env.remaining()),
                incumbent=clone.incumbent.clone(),
                free_mask=candidates[idx].mask,
                solver_threads=1,
            )

            if result.schedule is not None:
                report = clone.problem.config.validate(
                    clone.problem.instance, result.schedule)
                if report.feasible and report.objective >= clone.objective:
                    clone.incumbent = result.schedule.detach().cpu().bool().clone()
                    clone.objective = report.objective
                    clone.done = (env.clock() + (perf_counter() - start_time) >= env.deadline)

            # Continue the rollout from this state
            if not clone.done:
                # Populate candidates for the clone so observe() works
                clone.refresh()
                _, actions_after = _rollout_one(clone, policy, max_steps=max(1, rollout_steps - 1),
                                                greedy=greedy, mode=mode)
                total_reward = (clone.objective - env.initial_objective) / env.scale
            else:
                total_reward = (clone.objective - env.initial_objective) / env.scale
                actions_after = []

            total_rollout_time += perf_counter() - start_time

            if total_reward > best_reward:
                best_reward = total_reward
                best_action = idx + 1  # 1-indexed (0 is STOP)
                best_actions_taken = [idx] + actions_after

    # Execute the best action on the real environment
    if best_action > 0:
        selected_candidate = candidates[best_action - 1]
        reward, done = env.step(best_action)
        # Restore state from the best rollout clone for consistency tracking
        # (We already executed on env; just update operator stats)
        # Find which clone had the best result and merge its stats
        # For simplicity, we use the real env's results but track actions taken

    return best_action


def beam_search(env, policy, *, beam_width=4, rollout_depth=2, k_samples=4,
                greedy=False, mode="ppo", time_limit=None):
    """Beam search over repair trajectories using the trained policy.

    Maintains a beam of up to `beam_width` partial trajectories. At each step,
    expands each trajectory by sampling `k_samples` actions, keeping the top
    `beam_width` by cumulative reward. Selects the first action of the best
    final trajectory.

    This is more expensive than lookahead_select but explores a richer set of
    multi-step plans. Recommended for longer budgets (>= 5 seconds).

    Args:
        env: The SearchEnv instance.
        policy: The trained ActorCritic policy.
        beam_width: Number of parallel trajectories to maintain.
        rollout_depth: Maximum number of decision steps per trajectory.
        k_samples: Actions to sample from each trajectory at each step.
        greedy: Use argmax instead of sampling.
        mode: Policy evaluation mode.
        time_limit: Override the environment's default deadline.

    Returns:
        The final incumbent schedule after executing the best trajectory.
    """
    if time_limit is not None:
        env.deadline = env.clock() + time_limit - min(0.05, time_limit * 0.05)

    # Initialize beam with one trajectory per top candidate
    env.refresh()
    candidates = env.candidates
    num_candidates = len(candidates)

    if num_candidates == 0:
        return env.incumbent

    # Select initial candidates for the beam
    init_count = min(beam_width, num_candidates)
    if num_candidates > init_count:
        weights = np.ones(num_candidates)
        for i, c in enumerate(candidates):
            if c.operator in ("run", "energy"):
                weights[i] *= 1.5
        weights /= weights.sum()
        init_indices = env.rng.choice(num_candidates, size=init_count, replace=False,
                                      p=weights)
    else:
        init_indices = list(range(num_candidates))

    # Each beam entry: (env_clone, cumulative_reward, action_history)
    beam = []
    for idx in init_indices:
        clone = _deep_clone_env(env)
        reward, done = clone.step(idx + 1)
        cum_reward = sum(r for _, r, _ in [(clone.observe(), reward, done)])
        # Recalculate properly
        cum_reward = (clone.objective - env.initial_objective) / env.scale
        beam.append((clone, cum_reward, [idx + 1]))

    # Expand beam for rollout_depth steps
    for step in range(rollout_depth - 1):
        if not beam or all(b[0].done for b in beam):
            break

        new_beam = []
        for clone, cum_reward, actions in beam:
            if clone.done:
                new_beam.append((clone, cum_reward, actions))
                continue

            env.refresh()
            cands = env.candidates
            if not cands:
                new_beam.append((clone, cum_reward, actions))
                continue

            # Sample k_actions from this state
            rows, mask = clone.observe()
            with torch.no_grad():
                dist, _ = policy(rows, mask)
                if greedy:
                    sampled = [dist.logits.argmax().item()] * k_samples
                else:
                    sampled = dist.sample((k_samples,)).tolist()

            for action in sampled[:k_samples]:
                child = _deep_clone_env(clone)
                # Re-execute from clone's state with the new action
                child_incumbent = clone.incumbent.clone()
                result = child.problem.solve(
                    time_limit=min(env.repair_time, child.remaining()),
                    incumbent=child_incumbent,
                    free_mask=cands[action - 1].mask if action > 0 else torch.zeros_like(child_incumbent),
                    solver_threads=1,
                )

                if result.schedule is not None:
                    report = child.problem.config.validate(
                        child.problem.instance, result.schedule)
                    if report.feasible and report.objective >= child.objective:
                        child.incumbent = result.schedule.detach().cpu().bool().clone()
                        child.objective = report.objective

                child_reward = (child.objective - env.initial_objective) / env.scale
                child_actions = actions + ([action] if action > 0 else [])
                new_beam.append((child, child_reward, child_actions))

        # Keep top beam_width by cumulative reward
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    # Select the best trajectory and merge its result into env
    if beam:
        best_clone, _, _ = max(beam, key=lambda x: x[1])
        env.incumbent = best_clone.incumbent.detach().cpu().clone()
        env.objective = best_clone.objective
        env.state = SearchState(
            last_mask=best_clone.state.last_mask.detach().cpu().clone(),
            changed=best_clone.state.changed.detach().cpu().clone(),
            steps=best_clone.state.steps,
            stagnation=best_clone.state.stagnation,
            last_gain=best_clone.state.last_gain,
            last_runtime=best_clone.state.last_runtime,
        )

    return env.incumbent


def rollout_with_lookahead(env, policy, *, lookahead_k=8, lookahead_steps=2,
                           greedy=False, mode="ppo", **kwargs):
    """Main entry point: PPO rollout with LRBS-style lookahead selection.

    Replaces the standard single-step action selection with best-of-K rollout
    evaluation. All other search logic (deadline, validation, incumbent
    preservation) is unchanged.

    Args:
        env: The SearchEnv instance.
        policy: The trained ActorCritic policy.
        lookahead_k: Number of rollouts per candidate (total = candidates * k).
        lookahead_steps: Depth of each short rollout (default 2).
        greedy: Use deterministic selection during evaluation.
        mode: Policy evaluation mode.
        **kwargs: Forwarded to lookahead_select.

    Returns:
        The final trajectory (list of transitions).
    """
    trajectory = []

    while not env.done:
        # Check budget before lookahead selection
        if env.remaining() < getattr(env, 'repair_time', 0.1):
            env.done = True
            break

        # Use lookahead to select the best action
        action = lookahead_select(
            env, policy,
            k=lookahead_k,
            rollout_steps=lookahead_steps,
            greedy=greedy,
            mode=mode,
            **kwargs,
        )

        if action == 0:
            env.done = True
            rows, mask = env.observe()
            with torch.no_grad():
                _, value = policy(rows, mask)
                value = value.item()
            trajectory.append((rows, mask, 0, 0.0, value, 0.0))
            break

        # Check budget again after lookahead (rollouts consumed time)
        if env.remaining() < getattr(env, 'repair_time', 0.1):
            env.done = True
            rows, mask = env.observe()
            with torch.no_grad():
                _, value = policy(rows, mask)
                value = value.item()
            trajectory.append((rows, mask, action, 0.0, value, 0.0))
            break

        # Execute the selected action on the real environment
        rows, mask = env.observe()
        with torch.no_grad():
            dist, value = policy(rows, mask)
            value = value.item()

        start = env.clock()
        reward, done = env.step(action)
        elapsed = env.clock() - start

        trajectory.append((rows, mask, action, 0.0, value, reward))
        if not done:
            env.refresh()

    return trajectory


def evaluate_lookahead(instance, initial_schedule, policy, *,
                       budget=3.0, seed=0, lookahead_k=8, lookahead_steps=2,
                       mode="ppo", greedy=False, **kwargs):
    """Run a single episode with lookahead rollout search and return the result.

    Convenience wrapper for benchmarking against baselines.

    Args:
        instance: The ONTS instance.
        initial_schedule: A feasible starting schedule.
        policy: The trained ActorCritic policy.
        budget: Wall-clock time limit in seconds.
        seed: Random seed.
        lookahead_k: Rollouts per candidate.
        lookahead_steps: Depth of each rollout.
        mode: Policy evaluation mode.
        greedy: Deterministic rollout selection.
        **kwargs: Forwarded to SearchEnv and rollout_with_lookahead.

    Returns:
        dict with keys: incumbent, objective, trajectory, elapsed, improvements.
    """
    from .rl_search import SearchEnv

    env = SearchEnv(instance, initial_schedule, budget=budget, seed=seed, **kwargs)
    trajectory = rollout_with_lookahead(
        env, policy,
        lookahead_k=lookahead_k,
        lookahead_steps=lookahead_steps,
        greedy=greedy,
        mode=mode,
    )

    improvements = [t[5] for t in trajectory if t[5] > 0]
    return {
        "incumbent": env.incumbent.clone(),
        "objective": float(env.objective),
        "initial_objective": float(env.initial_objective),
        "improvement": float((env.objective - env.initial_objective) / max(1.0, env.scale)),
        "scale": float(env.scale),
        "trajectory": trajectory,
        "elapsed": env.clock() - env.started,
        "steps": len(trajectory),
        "improvements": improvements,
        "n_improvements": len(improvements),
    }
