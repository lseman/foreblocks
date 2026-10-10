"""Perturbation-aware search with current/best separation and acceptance strategies.

Addresses the limitation of strictly-monotone incumbent updates: a policy that can only
accept improvements gets stuck in local optima. This module adds:

1. **Feasible perturbations** — actions that intentionally release tasks (worsen objective)
   to enable better solutions in subsequent repairs.
2. **Current/best separation** — tracks both the current schedule and the best feasible
   schedule found so far; rewards are based on best-so-far improvement.
3. **Multiple acceptance strategies**:
   - `monotone`: only accept improvements (baseline)
   - `annealing`: simulated annealing with fixed temperature
   - `alns`: adaptive large-neighborhood search with sliding-window acceptance rate
4. **Richer observation context** — current/best gap, stagnation history, remaining time.

Design rationale: ONTS repair operators often get stuck when tasks are tightly coupled
through battery/power constraints. Releasing a low-value task can enable a much better
reconfiguration of the remaining tasks. The policy must learn WHEN to sacrifice and WHEN
to improve.
"""
from __future__ import annotations

import math
from collections import deque
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .milp import ONTSMilp
from .ranking import SearchState, make_candidates


class PerturbationEnv:
    """Search environment with perturbations and current/best separation.

    Args:
        instance: The ONTS problem instance.
        initial_schedule: A feasible initial schedule (witness).
        budget: Total wall-clock time for the search episode.
        repair_time: Time limit per MILP repair call.
        seed: Random seed for reproducibility.
        acceptance: One of 'monotone', 'annealing', 'alns'.
        temperature: Initial temperature for annealing (0 = monotone).
        alns_window: Sliding window size for ALNS acceptance rate tracking.
        alns_target_rate: Target acceptance rate for ALNS adjustment.
        config: Optional ONTS configuration override.
    """

    def __init__(self, instance, initial_schedule, *, budget=3.0, repair_time=0.1,
                 seed=0, acceptance='monotone', temperature=0.0, alns_window=20,
                 alns_target_rate=0.4, config=None):
        if not all(math.isfinite(v) and v > 0 for v in (budget, repair_time)):
            raise ValueError("Budgets must be positive and finite")
        if acceptance not in ('monotone', 'annealing', 'alns'):
            raise ValueError(f"Unknown acceptance: {acceptance}")

        self.acceptance = acceptance
        self.temperature = temperature
        self.alns_window = alns_window
        self.alns_target_rate = alns_target_rate
        self.clock = perf_counter
        self.started = self.clock()
        self.budget, self.repair_time = budget, repair_time
        self.reserve = min(0.05, budget * 0.05)
        self.deadline = self.started + budget - self.reserve
        self.problem = ONTSMilp(instance, config)

        report = self.problem.config.validate(instance, initial_schedule)
        if not report.feasible:
            raise ValueError("A feasible initial schedule is required")

        self.incumbent = initial_schedule.detach().cpu().bool().clone()
        self.current_objective = report.objective
        self.initial_objective = report.objective
        # Best-so-far tracking (separate from current)
        self.best_schedule = initial_schedule.detach().cpu().bool().clone()
        self.best_objective = report.objective
        self.scale = max(float(instance.priority.clamp(min=0).sum()) * instance.T, 1.0)

        self.rng = np.random.default_rng(seed)
        self.state = SearchState.empty(instance.J, instance.T)
        self.log = []
        self.done = False
        self.candidates = []

        # ALNS state
        self._alns_history = deque(maxlen=alns_window)
        self._alns_current_rate = 0.5  # Initial acceptance rate estimate

        # Operator statistics
        self.operator_pulls = np.zeros(4)
        self.operator_gains = np.zeros(4)
        self.operator_durations = np.zeros(4)
        self.operator_last_used = np.full(4, -1)

        self.refresh()

    def remaining(self):
        return max(0.0, self.deadline - self.clock())

    def refresh(self):
        """Generate candidate repairs AND perturbations."""
        # Standard repair candidates (count >= 4 required by make_candidates)
        repairs = make_candidates(self.problem, self.incumbent,
                                  self.state, self.rng, count=8)
        # Perturbation candidates: release 1-2 low-priority tasks
        perturbations = self._make_perturbations(count=1)
        self.candidates = repairs + perturbations

    def _make_perturbations(self, count=1):
        """Create perturbation candidates that release low-value tasks.

        A perturbation releases (sets to False) 1-2 tasks with the lowest
        priority, allowing the solver to reconfigure around them.
        """
        perturbations = []
        # Find currently scheduled tasks sorted by priority
        scheduled = self.incumbent.nonzero(as_tuple=True)[0].tolist()
        if len(scheduled) < 2:
            return perturbations  # Can't perturb if too few tasks

        priorities = self.problem.instance.priority[scheduled].float()

        # Release lowest-priority tasks (up to 2)
        release_count = min(2, max(1, len(scheduled) // 4))
        release_idx = priorities.topk(release_count, largest=False).indices.tolist()

        for idx in release_idx[:count]:
            task_id = scheduled[idx]
            mask = torch.zeros_like(self.incumbent)
            mask[task_id] = True  # Allow changing this task
            features = torch.zeros(10)  # Match make_candidates feature size
            features[0] = 1.0  # Perturbation flag
            perturbations.append(type('Candidate', (), {
                'mask': mask,
                'features': features,
                'operator': 'perturb',
            })())

        return perturbations

    def observe(self):
        """Observation with current/best gap and acceptance context."""
        report = self.problem.config.validate(self.problem.instance, self.incumbent)
        gap = (self.best_objective - self.current_objective) / max(1.0, self.scale)
        context = torch.tensor([
            self.current_objective / self.scale,       # 0: current objective (normalized)
            self.best_objective / self.scale,          # 1: best objective (normalized)
            gap,                                        # 2: current/best gap
            self.remaining() / self.budget,            # 3: remaining budget fraction
            min(self.state.stagnation, 20) / 20,       # 4: stagnation
            self.state.last_gain / self.scale,         # 5: last gain
            self.state.last_runtime / self.budget,     # 6: last runtime
            min(report.soc),                           # 7: min SOC
            sum(report.soc) / len(report.soc),         # 8: avg SOC
            self.incumbent.float().mean().item(),      # 9: task density
        ], dtype=torch.float32)

        rows = [torch.cat((torch.zeros(10), context))]
        rows.extend(torch.cat((c.features, context)) for c in self.candidates)
        affordable = not self.done and self.remaining() >= self.repair_time
        mask = torch.tensor([self.acceptance != 'alns' or not affordable] + [affordable] * len(self.candidates))
        return torch.stack(rows), mask

    def _accept_probability(self, delta, scale):
        """Compute acceptance probability for a worse solution."""
        if delta >= 0:
            return 1.0  # Improvement: always accept

        if self.acceptance == 'monotone':
            return 0.0  # Never accept worse

        if self.acceptance == 'annealing':
            if self.temperature <= 0:
                return 0.0
            # Simulated annealing: exp(delta / (T * scale))
            # delta is negative, so this is exp(-|delta| / (T * scale))
            t = max(self.temperature * self.scale, 1e-6)
            return float(np.exp(delta / t))

        if self.acceptance == 'alns':
            # Adaptive: use current estimated acceptance rate
            # Accept worse solutions with probability proportional to target rate
            normalized_delta = delta / max(1.0, self.scale)
            # Sigmoid mapping from delta to acceptance probability
            return float(1.0 / (1.0 + np.exp(-10 * (normalized_delta + self._alns_current_rate - 0.5))))

        return 0.0

    def step(self, action):
        """Execute an action with the configured acceptance strategy."""
        if self.done:
            raise RuntimeError("Episode has ended")
        if not isinstance(action, int) or not 0 <= action <= len(self.candidates):
            raise ValueError("Invalid action")

        if action == 0 or self.remaining() < self.repair_time:
            self.done = True
            reward = 0.0
            return reward, self.done

        candidate = self.candidates[action - 1]
        operator = candidate.operator
        before = self.current_objective
        start = self.clock()

        # For perturbations, execute the release first then solve without incumbent validation
        perturbed = None
        if operator == 'perturb':
            task_id = candidate.mask.nonzero(as_tuple=True)[0][0].item()
            perturbed = self.incumbent.clone()
            perturbed[task_id] = False
            # Solve on perturbed state: bypass assignment check since release is intentional
            from scipy.optimize import milp, Bounds as ScipyBounds
            import warnings
            from .milp import SolveResult
            lo, hi = self.problem.lower.copy(), self.problem.upper.copy()
            mask = torch.as_tensor(candidate.mask).detach().cpu()
            fixed = ~mask.numpy().ravel()
            values = perturbed.detach().cpu().double().numpy().ravel()
            lo[:self.problem.n_x][fixed] = values[fixed]
            hi[:self.problem.n_x][fixed] = values[fixed]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Unrecognized options.*")
                raw = milp(
                    c=self.problem.c,
                    integrality=self.problem.integrality,
                    bounds=ScipyBounds(lo, hi),
                    constraints=self.problem.constraints,
                    options={"time_limit": self.repair_time, "mip_rel_gap": 0.0, "random_seed": 0},
                )
            status = {0: 'optimal', 1: 'limit', 2: 'infeasible', 3: 'unbounded', 4: 'error'}.get(raw.status, 'error')
            if raw.x is not None and np.isfinite(raw.x).all():
                raw_x = raw.x[:self.problem.n_x]
                if np.max(np.abs(raw_x - raw_x.round())) <= 1e-5:
                    candidate_sched = torch.from_numpy(raw_x.round().reshape(self.problem.J, self.problem.T)).bool()
                    candidate_report = self.problem.config.validate(self.problem.instance, candidate_sched)
                    if candidate_report.feasible:
                        result = SolveResult(status, '', candidate_sched, candidate_report.objective,
                                             candidate_report, None, 0, None)
                    else:
                        result = SolveResult(status, '', None, None, None, None, 0, None)
                else:
                    result = SolveResult(status, '', None, None, None, None, 0, None)
            else:
                result = SolveResult(status, '', None, None, None, None, 0, None)
        else:
            previous = self.incumbent.clone()
            result = self.problem.solve(time_limit=self.repair_time,
                                        incumbent=previous, free_mask=candidate.mask,
                                        solver_threads=1)

        status = result.status
        accepted = False

        if result.schedule is not None:
            report = self.problem.config.validate(self.problem.instance, result.schedule)
            delta = report.objective - before

            if self.clock() > self.deadline:
                # Late result: no credit
                pass
            elif report.feasible:
                accept_prob = self._accept_probability(delta, self.scale)
                if self.rng.random() < accept_prob:
                    self.incumbent = result.schedule.detach().cpu().bool().clone()
                    self.current_objective = report.objective
                    accepted = True

                    # Update best-so-far
                    if report.objective > self.best_objective:
                        self.best_schedule = result.schedule.detach().cpu().bool().clone()
                        self.best_objective = report.objective

        # Track ALNS acceptance rate
        if operator != 'perturb':
            delta = (self.current_objective - before) / max(1.0, self.scale)
            was_accepted = delta >= 0 or (self.acceptance in ('annealing', 'alns') and accepted)
            self._alns_history.append(1.0 if was_accepted else 0.0)

            # Update ALNS current rate
            if len(self._alns_history) >= 5:
                self._alns_current_rate = float(np.mean(self._alns_history))

        # State advancement (use incumbent after acceptance decision)
        after = self.current_objective
        reward_raw = after - before
        self.state.advance(candidate.mask,
                           perturbed if perturbed is not None else self.incumbent.clone(),
                           self.incumbent, reward_raw, self.clock() - start)

        arm = int(candidate.features[:4].argmax()) if candidate.features[:4].sum() > 0 else 0
        self.operator_pulls[arm] += 1
        self.operator_gains[arm] += reward_raw / self.scale
        self.operator_durations[arm] += self.clock() - start
        self.operator_last_used[arm] = self.state.steps

        # Reward: best-so-far improvement (normalized)
        reward = (self.best_objective - before) / max(1.0, self.scale)

        self.log.append(dict(action=action, operator=operator, status=status,
                             accepted=accepted, reward=reward,
                             current_objective=self.current_objective,
                             best_objective=self.best_objective,
                             elapsed=self.clock() - self.started,
                             duration=self.clock() - start))

        if not self.done:
            self.refresh()
            self.done = self.remaining() < self.repair_time

        return reward, self.done


class PerturbationActorCritic(nn.Module):
    """Policy for perturbation-aware search with 10-dim context."""

    def __init__(self, width=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
        )
        self.actor = nn.Linear(width, 1)
        self.critic = nn.Linear(width, 1)

    def forward(self, rows, mask):
        hidden = self.encoder(rows)
        dist = Categorical(logits=self.actor(hidden).squeeze(-1).masked_fill(~mask, -torch.inf))
        return dist, self.critic(hidden[0]).squeeze(-1)


def rollout(env, policy=None, *, mode="ppo", greedy=False):
    """Rollout with perturbation-aware environment."""
    trajectory = []
    while not env.done:
        rows, mask = env.observe()
        with torch.no_grad():
            if mode == "ppo":
                dist, value = policy(rows, mask)
                action_tensor = dist.logits.argmax() if greedy else dist.sample()
                logp = dist.log_prob(action_tensor).item()
                value = value.item()
                action = int(action_tensor)
            else:
                choices = torch.where(mask)[0].tolist()
                repairs = [a for a in choices if a]
                if not repairs:
                    action = 0
                elif mode == "uniform":
                    action = int(env.rng.choice(repairs))
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                logp, value = 0.0, 0.0

        reward, done = env.step(action)
        trajectory.append((rows, mask, action, logp, value, reward))
    return trajectory


def ppo_batch_update(policy, optimizer, trajectories, *, batch_episodes=8, epochs=4,
                     minibatch_size=8, trace_lambda=1, entropy_coef=0.002, seed=0):
    """PPO update with GAE (same interface as rl_learning.ppo_batch_update)."""
    if not trajectories:
        return {}

    rng = np.random.default_rng(seed)
    all_transitions = []
    for traj in trajectories:
        returns, total = [], 0.0
        for transition in reversed(traj):
            total += transition[-1]
            returns.append(total)
        returns.reverse()
        values = [t[4].item() if hasattr(t[4], 'item') else t[4] for t in traj]
        advantages = torch.tensor(returns) - torch.tensor(values)

        for i, (rows, mask, action, old_logp, _, _) in enumerate(traj):
            all_transitions.append((rows, mask, action, old_logp, advantages[i], returns[i]))

    # Shuffle and batch
    indices = rng.permutation(len(all_transitions))
    total_loss = 0.0
    total_reward = 0.0
    n_steps = 0

    for epoch in range(epochs):
        for start in range(0, len(indices), minibatch_size):
            batch_idx = indices[start:start + minibatch_size]
            rows_list = [all_transitions[i][0] for i in batch_idx]
            mask_list = [all_transitions[i][1] for i in batch_idx]
            actions = torch.tensor([all_transitions[i][2] for i in batch_idx])
            old_logps = torch.tensor([all_transitions[i][3] for i in batch_idx])
            advs = [a.item() if hasattr(a, 'item') else float(a) for a in [all_transitions[i][4] for i in batch_idx]]
            rets = [r.item() if hasattr(r, 'item') else float(r) for r in [all_transitions[i][5] for i in batch_idx]]
            advantages_t = torch.tensor(advs)
            returns_t = torch.tensor(rets)

            rows = torch.stack(rows_list)
            mask = torch.stack(mask_list)

            dist, values = policy(rows, mask)
            # Extract critic value for each selected action index
            if values.ndim == 0:
                values = values.expand(len(batch_idx))
            # values is (batch_size,) - one per row; actions are already the indices
            value_for_action = values[torch.arange(len(batch_idx))]
            ratio = (dist.log_prob(actions) - old_logps).exp()
            surrogate1 = ratio * advantages_t
            surrogate2 = ratio.clamp(1 - 0.2, 1 + 0.2) * advantages_t
            clip_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = 0.5 * (value_for_action - returns_t).square().mean()
            entropy = dist.entropy().mean()

            loss = clip_loss + 0.5 * value_loss - entropy_coef * entropy
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            for i in batch_idx:
                r = all_transitions[i][5]
                total_reward += r if isinstance(r, float) else r.item()
                n_steps += 1

    return dict(loss=total_loss / max(1, n_steps), reward=total_reward / max(1, n_steps),
                steps=n_steps)
