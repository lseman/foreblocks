"""Common-incumbent semi-Markov search environment and categorical PPO.

Compact observations approximate search state; this is not a full-state MDP.
Action zero is STOP. Other actions share the existing MILP repair portfolio.
"""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .milp import ONTSMilp
from .ranking import SearchState, make_candidates


class SearchEnv:
    def __init__(self, instance, initial_schedule, *, budget=3.0,
                 repair_time=0.1, seed=0, config=None, clock=perf_counter,
                 stop_mode="optional"):
        if not all(math.isfinite(v) and v > 0 for v in (budget, repair_time)):
            raise ValueError("Budgets must be positive and finite")
        if stop_mode not in ("optional", "exhaustion"):
            raise ValueError("Unknown stop mode")
        self.stop_mode = stop_mode
        self.clock = clock
        self.started = clock()
        self.budget, self.repair_time = budget, repair_time
        self.reserve = min(0.05, budget * 0.05)
        self.deadline = self.started + budget - self.reserve
        self.problem = ONTSMilp(instance, config)
        report = self.problem.config.validate(instance, initial_schedule)
        if not report.feasible:
            raise ValueError("A feasible initial schedule is required")
        self.incumbent = initial_schedule.detach().cpu().bool().clone()
        self.objective = self.initial_objective = report.objective
        self.scale = max(float(instance.priority.clamp(min=0).sum()) * instance.T, 1.0)
        self.rng = np.random.default_rng(seed)
        self.state = SearchState.empty(instance.J, instance.T)
        self.log = []
        self.done = False
        self.candidates = []
        self.operator_pulls = np.zeros(4)
        self.operator_gains = np.zeros(4)
        self.operator_durations = np.zeros(4)
        self.operator_last_used = np.full(4, -1)
        self.refresh()

    def remaining(self):
        return max(0.0, self.deadline - self.clock())

    def refresh(self):
        self.candidates = make_candidates(self.problem, self.incumbent,
                                         self.state, self.rng, count=4)

    def observe(self):
        report = self.problem.config.validate(self.problem.instance, self.incumbent)
        context = torch.tensor([
            self.objective / self.scale, self.remaining() / self.budget,
            min(self.state.stagnation, 20) / 20, self.state.last_gain / self.scale,
            self.state.last_runtime / self.budget, min(report.soc),
            sum(report.soc) / len(report.soc), self.incumbent.float().mean().item(),
        ], dtype=torch.float32)
        rows = [torch.cat((torch.zeros(10), context))]
        rows.extend(torch.cat((c.features, context)) for c in self.candidates)
        affordable = not self.done and self.remaining() >= self.repair_time
        mask = torch.tensor([self.stop_mode == "optional" or not affordable] + [affordable] * 4)
        return torch.stack(rows), mask

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended")
        if not isinstance(action, int) or not 0 <= action <= len(self.candidates):
            raise ValueError("Invalid action")
        if action == 0 and self.stop_mode == "exhaustion" and self.remaining() >= self.repair_time:
            raise ValueError("STOP is masked while a repair is affordable")
        before = self.objective
        start = self.clock()
        status, accepted, operator = "stop", False, "stop"
        if action == 0 or self.remaining() < self.repair_time:
            self.done = True
        else:
            candidate = self.candidates[action - 1]
            operator = candidate.operator
            previous = self.incumbent.clone()
            result = self.problem.solve(time_limit=self.repair_time,
                                        incumbent=previous, free_mask=candidate.mask,
                                        solver_threads=1)
            status = result.status
            if result.schedule is not None:
                report = self.problem.config.validate(self.problem.instance, result.schedule)
                # Validation is part of the deadline. Late results receive no credit.
                if report.feasible and report.objective >= before and self.clock() <= self.deadline:
                    self.incumbent = result.schedule.detach().cpu().bool().clone()
                    self.objective = report.objective
                    accepted = True
            self.state.advance(candidate.mask, previous, self.incumbent,
                               self.objective - before, self.clock() - start)
            arm = int(candidate.features[:4].argmax())
            self.operator_pulls[arm] += 1
            self.operator_gains[arm] += (self.objective - before) / self.scale
            self.operator_durations[arm] += self.clock() - start
            self.operator_last_used[arm] = self.state.steps
            self.done = self.remaining() < self.repair_time
        reward = (self.objective - before) / self.scale
        self.log.append(dict(action=action, operator=operator, status=status,
                             accepted=accepted, reward=reward, objective=self.objective,
                             elapsed=self.clock() - self.started,
                             duration=self.clock() - start))
        if not self.done:
            self.refresh()
            self.done = self.remaining() < self.repair_time
        return reward, self.done


class ActorCritic(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(18, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh())
        self.actor = nn.Linear(width, 1)
        self.critic = nn.Linear(width, 1)

    def forward(self, rows, mask):
        hidden = self.encoder(rows)
        distribution = Categorical(logits=self.actor(hidden).squeeze(-1).masked_fill(~mask, -torch.inf))
        return distribution, self.critic(hidden[0]).squeeze(-1)


def rollout(env, policy=None, *, mode="ppo", bandit=None, greedy=False):
    """Return transitions; inference time is charged by the environment clock."""
    trajectory = []
    while not env.done:
        selection_started = env.clock()
        if mode == "ppo" and hasattr(policy, "observation"):
            rows, mask = policy.observation(env)
        else:
            rows, mask = env.observe()
        with torch.no_grad():
            if mode == "ppo":
                dist, value = policy(rows, mask)
                action = dist.logits.argmax() if greedy else dist.sample()
                logp = dist.log_prob(action).item()
                value = value.item()
                selected = int(action)
            else:
                choices = torch.where(mask)[0].tolist()
                repairs = [a for a in choices if a]
                if not repairs:
                    selected = 0
                elif mode == "uniform":
                    selected = int(env.rng.choice(repairs))
                elif mode == "utility":
                    selected = 2
                elif mode == "bandit":
                    # Linear contextual UCB; shared action features, online only.
                    matrix, vector = bandit
                    inverse = np.linalg.inv(matrix)
                    x = rows.numpy()
                    scores = x @ inverse @ vector + 0.1 * np.sqrt(np.einsum('ij,jk,ik->i', x, inverse, x))
                    selected = max(repairs, key=lambda a: scores[a])
                else:
                    raise ValueError("Unknown selector")
                logp, value = 0.0, 0.0
        selection_runtime = env.clock() - selection_started
        reward, done = env.step(selected)
        env.log[-1]["selection_runtime"] = selection_runtime
        if mode == "bandit":
            x = rows[selected].numpy().astype(float)
            bandit[0][:] += np.outer(x, x)
            bandit[1][:] += reward * x
        trajectory.append((rows, mask, selected, logp, value, reward))
    return trajectory


def ppo_update(policy, optimizer, trajectory, *, epochs=4, clip=0.2):
    """Undiscounted Monte Carlo advantages for terminal fixed-budget quality."""
    if not trajectory:
        return {}
    returns, total = [], 0.0
    for transition in reversed(trajectory):
        total += transition[-1]
        returns.append(total)
    returns.reverse()
    targets = torch.tensor(returns)
    advantages = targets - torch.tensor([t[4] for t in trajectory])
    # Keep the absolute reward scale; centering a one-step episode removes signal.
    for _ in range(epochs):
        losses = []
        for i, (rows, mask, action, old_logp, _, _) in enumerate(trajectory):
            dist, value = policy(rows, mask)
            ratio = (dist.log_prob(torch.tensor(action)) - old_logp).exp()
            surrogate = torch.minimum(ratio * advantages[i], ratio.clamp(1-clip, 1+clip) * advantages[i])
            losses.append(-surrogate + 0.5 * (value-targets[i]).square() - 0.01 * dist.entropy())
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    return dict(loss=float(loss.detach()), reward=total, steps=len(trajectory))
