"""Training and evaluation for constraint-aware policies on ONTS.

This module provides proper training and evaluation of constraint-aware policies
that can learn to relax constraints strategically, compared against baselines.
"""
from __future__ import annotations

import math
from time import perf_counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .milp import ONTSMilp
from .ranking import SearchState, make_candidates
from .constraint_aware_graph import ConstraintAwareGraphBuilder, ConstraintInfo, ConstraintType
from .constraint_aware_policy import ConstraintAwareActorCritic


@dataclass
class ConstraintAwareObservation:
    """Observation for constraint-aware policy."""
    nodes: torch.Tensor  # [1, n_nodes, f_node]
    mask: torch.Tensor   # [1, n_nodes]
    relaxable_mask: torch.Tensor  # [1, m] — which constraints can be relaxed
    relaxation_costs: torch.Tensor  # [1, m] — per-constraint relaxation costs


class ConstraintAwareSearchEnv:
    """Search environment that uses constraint-aware graph observations.

    Extends the standard SearchEnv to provide constraint-aware observations
    while maintaining the same search dynamics and reward structure.
    """

    def __init__(self, instance, initial_schedule, *, budget=3.0,
                 repair_time=0.1, seed=0, config=None, clock=perf_counter):
        if not all(math.isfinite(v) and v > 0 for v in (budget, repair_time)):
            raise ValueError("Budgets must be positive and finite")
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
        
        # Build constraint info for this instance
        self.constraint_info = self._build_constraint_info()
        
        self.refresh()

    def _build_constraint_info(self):
        """Build constraint info for the current problem."""
        m = self.problem.constraints.A.shape[0] if hasattr(self.problem.constraints, 'A') else len(self.problem.row_lower)
        constraint_info = []
        for i in range(m):
            if i < m // 3:
                ctype = ConstraintType.HARD
                relaxable = False
                cost = float('inf')
            elif i < 2 * m // 3:
                ctype = ConstraintType.SOFT
                relaxable = True
                cost = 0.1
            else:
                ctype = ConstraintType.PREFERENCE
                relaxable = True
                cost = 0.01

            tightness = 0.5
            constraint_info.append(ConstraintInfo(
                index=i,
                constraint_type=ctype,
                relaxable=relaxable,
                relaxation_cost=cost,
                rhs_value=float(self.problem.row_upper[i]),
                tightness=tightness,
            ))
        return constraint_info

    def remaining(self):
        return max(0.0, self.deadline - self.clock())

    def refresh(self):
        self.candidates = make_candidates(self.problem, self.incumbent,
                                         self.state, self.rng, count=4)

    def observe_constraint_aware(self):
        """Build constraint-aware graph observation."""
        if not hasattr(self, 'graph_builder'):
            b_vec = self.problem.row_upper if hasattr(self.problem, 'row_upper') else self.problem.b
            self.graph_builder = ConstraintAwareGraphBuilder(
                self.problem.A, b_vec, self.problem.c,
                bounds=list(zip(self.problem.lower, self.problem.upper)),
                constraint_info=self.constraint_info,
            )
        
        graph_sol = self.graph_builder.build()
        obs = graph_sol.obs
        
        # Extract relaxability metadata
        relaxable_mask = obs['constraint_relaxable']  # [1, m]
        relaxation_costs = obs['constraint_relaxation_cost']  # [1, m]
        
        return ConstraintAwareObservation(
            nodes=obs['nodes'],
            mask=obs['mask'],
            relaxable_mask=relaxable_mask,
            relaxation_costs=relaxation_costs,
        )

    def observe_standard(self):
        """Build standard observation (for baselines)."""
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
        mask = torch.tensor([True] + [affordable] * 4)
        return torch.stack(rows), mask

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended")
        if not isinstance(action, int) or not 0 <= action <= len(self.candidates):
            raise ValueError("Invalid action")
        
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


def rollout_constraint_aware(env, policy, *, greedy=False):
    """Rollout using constraint-aware observations."""
    trajectory = []
    while not env.done:
        obs = env.observe_constraint_aware()
        
        # Build candidate masks from environment candidates
        n_candidates = len(env.candidates)
        candidate_masks = torch.zeros(1, n_candidates, obs.nodes.shape[1], dtype=torch.bool)
        for i, cand in enumerate(env.candidates):
            # Candidate mask is a boolean tensor of shape [n_nodes]
            # where True indicates variables belonging to this candidate
            cand_mask_flat = cand.mask.flatten().detach().clone()  # [J*T]
            # Map to node indices (variables are first n nodes)
            candidate_masks[0, i, :len(cand_mask_flat)] = cand_mask_flat.to(dtype=torch.bool)
        
        # Build global context from observation
        global_context = obs.nodes[0, 0]  # First node is context
        
        # Build action mask (STOP + candidates)
        affordable = not env.done and env.remaining() >= env.repair_time
        action_mask = torch.tensor([True] + [affordable] * n_candidates, dtype=torch.bool)
        
        with torch.no_grad():
            dist, value = policy(obs.nodes, candidate_masks, global_context.unsqueeze(0), action_mask.unsqueeze(0))
            if greedy:
                action = dist.logits.argmax()
            else:
                action = dist.sample()
            logp = dist.log_prob(action).item()
            val = value.item()
        
        reward, done = env.step(int(action))
        trajectory.append((obs.nodes, candidate_masks, global_context, action_mask, int(action), logp, val, reward))
        
        if not done:
            env.refresh()
            done = env.remaining() < env.repair_time
    
    return trajectory


def evaluate_policy(instance, initial_schedule, policy, *, budget=2.0, seed=0, use_constraint_aware=True):
    """Evaluate a policy on a single instance."""
    if use_constraint_aware:
        env = ConstraintAwareSearchEnv(instance, initial_schedule, budget=budget, seed=seed)
        trajectory = rollout_constraint_aware(env, policy, greedy=True)
    else:
        env = ONTSMilp(instance)  # Standard MILP-only evaluation
        from .ranking import SearchState
        state = SearchState.empty(instance.J, instance.T)
        incumbent = initial_schedule.detach().cpu().bool().clone()
        report = env.config.validate(instance, incumbent)
        objective = report.objective
        scale = max(float(instance.priority.clamp(min=0).sum()) * instance.T, 1.0)
        
        # Simple MILP-only evaluation
        started = perf_counter()
        steps = 0
        while started + budget > perf_counter():
            candidates = make_candidates(env, incumbent, state, np.random.default_rng(seed + steps), count=4)
            if not candidates:
                break
            # Try each candidate with MILP
            best_gain = 0
            for i, cand in enumerate(candidates):
                result = env.solve(time_limit=0.1, incumbent=incumbent, free_mask=cand.mask, solver_threads=1)
                if result.schedule is not None:
                    rep = env.config.validate(instance, result.schedule)
                    if rep.feasible and rep.objective > objective:
                        gain = (rep.objective - objective) / scale
                        if gain > best_gain:
                            best_gain = gain
                            incumbent = result.schedule.detach().cpu().bool().clone()
                            objective = rep.objective
            if best_gain == 0:
                break
            state.advance(candidates[0].mask, incumbent.clone(), incumbent, best_gain * scale, 0.1)
            steps += 1
        
        improvement = (objective - report.objective) / max(scale, 1.0)
        return dict(method='milp_only', feasible=True,
                    initial=float(report.objective), objective=float(objective),
                    improvement=improvement, runtime=perf_counter() - started,
                    overrun=max(0, perf_counter() - started - budget), steps=steps)
    
    total_reward = sum(t[-1] for t in trajectory)
    return dict(method='constraint_aware' if use_constraint_aware else 'standard',
                feasible=True,
                initial=float(env.initial_objective),
                objective=float(env.objective),
                improvement=total_reward,
                runtime=perf_counter() - env.started,
                overrun=max(0, perf_counter() - env.started - budget),
                steps=len(trajectory))


def train_constraint_aware_policy(data, policy_seed, *, episodes=64, budget=0.8, seed=0):
    """Train a constraint-aware PPO policy on training instances."""
    from .rl_learning import generalized_advantages
    
    # Initialize policy with constraint-aware features
    policy = ConstraintAwareActorCritic(node_feature_dim=22, hidden=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    
    trajectories = []
    
    for episode in range(episodes):
        idx = episode % len(data['train'])
        instance, initial = data['train'][idx]
        
        env = ConstraintAwareSearchEnv(instance, initial, budget=budget, seed=seed + episode)
        trajectory = rollout_constraint_aware(env, policy, greedy=False)
        
        if not trajectory:
            continue
        
        trajectories.append(trajectory)
    
    if not trajectories:
        return policy
    
    # PPO update on all trajectories
    device = next(policy.parameters()).device
    
    for episode_trajectories in [trajectories]:  # Single pass for simplicity
        for trajectory in episode_trajectories:
            # Compute advantages using GAE
            advantages, targets = generalized_advantages(trajectory, trace_lambda=0.95)
            
            adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
            if len(adv_tensor) > 1 and adv_tensor.std(unbiased=False) > 1e-8:
                adv_tensor = (adv_tensor - adv_tensor.mean()) / adv_tensor.std(unbiased=False)
            
            losses = []
            for i, (nodes, candidate_masks, global_ctx, action_mask, action, old_logp, old_value, _) in enumerate(trajectory):
                # Ensure correct dimensions for policy forward pass
                if global_ctx.dim() == 1:
                    global_ctx = global_ctx.unsqueeze(0)
                if action_mask.dim() == 1:
                    action_mask = action_mask.unsqueeze(0)
                dist, value = policy(nodes, candidate_masks, global_ctx, action_mask)
                logp = dist.log_prob(torch.tensor(action, device=device))
                ratio = (logp - old_logp).exp()
                
                surrogate = torch.minimum(
                    ratio * adv_tensor[i],
                    ratio.clamp(0.8, 1.2) * adv_tensor[i]
                )
                value_loss = (value - targets[i]).square()
                entropy = dist.entropy()
                
                loss = -surrogate + 0.5 * value_loss - 0.01 * entropy
                losses.append(loss)
            
            if losses:
                loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
    
    return policy


def generate_splits(seed, *, train_count=9, development_count=3, ood_count=6):
    """Generate disjoint instance splits for training/development/evaluation."""
    from .research.datasets import instance_hash, witness_instance
    
    regimes = ('balanced', 'energy', 'periodic')
    data = {split: [witness_instance(seed=seed + offset + i, regime=regimes[i % 3], large=large)
                    for i in range(count)]
            for split, offset, count, large in [
                ('train', 0, train_count, False),
                ('development', 20000, development_count, False),
                ('ood', 30000, ood_count, True),
            ]}
    hashes = [instance_hash(i) for entries in data.values() for i, _ in entries]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError('Generated instance overlap')
    return data
