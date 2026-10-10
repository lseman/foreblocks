"""Semantic sparse-graph actor-critic, outcome warm starts and batched PPO.

ONTS-specific adaptations, not implementations of MIStar, ReSched or BOPO.
All observations are frozen before action selection for on-policy replay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from .ranking import ROW_FAMILIES, Candidate, RankingGraphBuilder, SparseAttention
from .rl_search import ActorCritic


@dataclass
class GraphObservation:
    state: tuple
    candidates: list[Candidate]
    rows: torch.Tensor
    operator_history: torch.Tensor


class GraphActorCritic(nn.Module):
    """Typed variable/constraint attention and whole-neighborhood readout.

    Real MILP energy-transition edges expose temporal battery coupling. Candidate
    scores are permutation equivariant. History is explicit, bounded and reset
    per episode; the observation is still an approximation to full search state.
    """
    def __init__(self, hidden=24, layers=1, *, semantics=True, memory=True):
        super().__init__()
        if hidden < 1 or layers < 1:
            raise ValueError("hidden and layers must be positive")
        self.hidden, self.layers = hidden, layers
        self.semantics, self.memory = semantics, memory
        self.variable_input = nn.Linear(16, hidden)
        self.constraint_input = nn.Linear(8, hidden)
        self.category = nn.Embedding(len(ROW_FAMILIES), hidden)
        self.vc = nn.ModuleList([SparseAttention(hidden) for _ in range(layers)])
        self.cv = nn.ModuleList([SparseAttention(hidden) for _ in range(layers)])
        self.global_input = nn.Sequential(nn.Linear(2 * hidden + 6, hidden), nn.GELU())
        self.actor = nn.Sequential(nn.Linear(2 * hidden + 14, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.stop = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def observation(self, env):
        # Lazy graph setup is charged to the online deadline.
        if not hasattr(env, 'graph_builder'):
            env.graph_builder = RankingGraphBuilder(env.problem)
        state = env.graph_builder.build_state(
            env.incumbent, env.state, remaining_fraction=env.remaining()/env.budget,
            repair_time=env.repair_time)
        pulls = np.maximum(env.operator_pulls, 1)
        history = np.stack([
            env.operator_gains / pulls,
            env.operator_durations / pulls / env.budget,
            env.operator_pulls / max(env.state.steps, 1),
            np.where(env.operator_last_used < 0, 1,
                     (env.state.steps-env.operator_last_used)/(1+env.state.steps)),
        ], axis=1)
        rows, mask = env.observe()
        return GraphObservation(state, list(env.candidates), rows,
                                torch.tensor(history, dtype=torch.float32)), mask

    def forward(self, observation: GraphObservation, mask):
        graph, extras, categories, context = observation.state
        device = next(self.parameters()).device
        graph = graph.to(device)
        extras, categories, context = extras.to(device), categories.to(device), context.to(device)
        history = observation.operator_history.to(device)
        if not self.memory:
            extras, context = extras.clone(), context.clone()
            extras[:, :2] = 0
            context[1:5] = 0
            history = torch.zeros_like(history)
        v = F.gelu(self.variable_input(torch.cat([graph.variables, extras], 1)))
        c = self.constraint_input(graph.constraints)
        if self.semantics:
            c = c + self.category(categories)
        c = F.gelu(c)
        edge = torch.stack([graph.coefficients, graph.coefficients.abs()], 1)
        for vc, cv in zip(self.vc, self.cv):
            c = vc(v, c, graph.col_index, graph.row_index, edge)
            v = cv(c, v, graph.row_index, graph.col_index, edge)
        global_state = self.global_input(torch.cat([v[:graph.n_x].mean(0), c.mean(0), context]))
        candidates = observation.candidates
        masks = torch.stack([candidate.mask.flatten() for candidate in candidates]).to(device=device, dtype=v.dtype)
        pooled = masks @ v[:graph.n_x] / masks.sum(1, keepdim=True).clamp(min=1)
        features = torch.stack([candidate.features for candidate in candidates]).to(device)
        if not self.memory:
            features = features.clone()
            features[:, 8] = 0
        arm_history = features[:, :4] @ history
        logits = self.actor(torch.cat([pooled, global_state.expand(len(candidates), -1), features, arm_history], 1)).squeeze(-1)
        logits = torch.cat([self.stop(global_state).view(1), logits])
        distribution = Categorical(logits=logits.masked_fill(~mask.to(device), -torch.inf))
        return distribution, self.critic(global_state).squeeze(-1)


def initialize_value_head(policy):
    """Avoid arbitrary initial critic offsets overwhelming small positive gains."""
    if hasattr(policy, 'critic') and isinstance(policy.critic, nn.Linear):
        nn.init.zeros_(policy.critic.weight)
        nn.init.zeros_(policy.critic.bias)


def load_rl_policy(path: str | Path):
    """Load an upgrade checkpoint without unpickling custom Python objects."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'onts-rl-upgrade-v1':
        raise ValueError('Unsupported RL checkpoint')
    architecture = checkpoint.get('architecture')
    if architecture == 'graph':
        policy = GraphActorCritic(**checkpoint['model_config'])
    elif architecture == 'compact':
        policy = ActorCritic(**checkpoint['model_config'])
    else:
        raise ValueError('Unsupported RL architecture')
    policy.load_state_dict(checkpoint['state_dict'])
    policy.training_metadata = checkpoint.get('plan', {})
    policy.search_method = checkpoint['method']
    policy.eval()
    return policy


def generalized_advantages(trajectory, *, trace_lambda=0.95):
    """Gamma=1 finite wall-budget objective; lambda controls estimator variance.

    Exhaustion and STOP terminate episodes, so their bootstrap value is zero.
    Each complete trajectory is processed independently; no cross-episode GAE.
    """
    if not 0 <= trace_lambda <= 1:
        raise ValueError("trace_lambda must be in [0,1]")
    advantages = [0.0] * len(trajectory)
    running, next_value = 0.0, 0.0
    for i in range(len(trajectory)-1, -1, -1):
        value, reward = trajectory[i][4:6]
        delta = reward + next_value - value
        running = delta + trace_lambda * running
        advantages[i] = running
        next_value = value
    targets = [advantage + transition[4] for advantage, transition in zip(advantages, trajectory)]
    return advantages, targets


def ppo_batch_update(policy, optimizer, trajectories, *, epochs=4, clip=0.2,
                     trace_lambda=0.95, entropy_coef=0.01, target_kl=0.03,
                     minibatch_size=32, seed=0):
    """Multi-episode on-policy PPO with clipped values and KL early stopping."""
    if epochs < 1 or minibatch_size < 1 or not 0 < clip < 1:
        raise ValueError("Invalid PPO update settings")
    if not all(math.isfinite(x) and x >= 0 for x in (entropy_coef, target_kl)):
        raise ValueError("Invalid entropy or KL settings")
    transitions, advantages, targets = [], [], []
    for trajectory in trajectories:
        a, y = generalized_advantages(trajectory, trace_lambda=trace_lambda)
        transitions.extend(trajectory)
        advantages.extend(a)
        targets.extend(y)
    if not transitions:
        return dict(steps=0, updates=0)
    device = next(policy.parameters()).device
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    targets = torch.tensor(targets, dtype=torch.float32, device=device)
    if len(advantages) > 1 and advantages.std(unbiased=False) > 1e-8:
        advantages = (advantages-advantages.mean())/advantages.std(unbiased=False)
    rng = np.random.default_rng(seed)
    metrics = []
    early_stop = False
    for _ in range(epochs):
        indices = rng.permutation(len(transitions))
        for offset in range(0, len(indices), minibatch_size):
            losses, kls, entropies, clipped = [], [], [], []
            for i in indices[offset:offset+minibatch_size]:
                observation, mask, action, old_logp, old_value, _ = transitions[i]
                dist, value = policy(observation, mask)
                logp = dist.log_prob(torch.tensor(action, device=device))
                logratio = logp-old_logp
                ratio = logratio.exp()
                kls.append(((ratio-1)-logratio).detach())
                clipped.append(((ratio-1).abs() > clip).float().detach())
                surrogate = torch.minimum(ratio*advantages[i], ratio.clamp(1-clip, 1+clip)*advantages[i])
                clipped_value = old_value + (value-old_value).clamp(-clip, clip)
                value_loss = torch.maximum((value-targets[i]).square(), (clipped_value-targets[i]).square())
                entropy = dist.entropy()
                losses.append(-surrogate + 0.5*value_loss - entropy_coef*entropy)
                entropies.append(entropy.detach())
            approximate_kl = torch.stack(kls).mean().item()
            if target_kl and approximate_kl > target_kl:
                early_stop = True
                break
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not torch.isfinite(gradient):
                raise FloatingPointError("Nonfinite PPO gradient")
            optimizer.step()
            metrics.append((loss.item(), approximate_kl, torch.stack(entropies).mean().item(), torch.stack(clipped).mean().item()))
        if early_stop:
            break
    with torch.no_grad():
        variance = targets.var(unbiased=False)
        old_values = torch.tensor([t[4] for t in transitions], device=device)
        explained = float(1-(targets-old_values).var(unbiased=False)/variance) if variance > 1e-9 else None
    averages = np.mean(metrics, axis=0) if metrics else np.zeros(4)
    return dict(steps=len(transitions), updates=len(metrics), early_stop=early_stop,
                loss=float(averages[0]), approximate_kl=float(averages[1]),
                entropy=float(averages[2]), clip_fraction=float(averages[3]),
                explained_variance=explained,
                mean_episode_reward=float(np.mean([sum(t[-1] for t in ts) for ts in trajectories])))


def preference_update(policy, optimizer, examples, *, epochs=4):
    """Best-anchored outcome preference loss over complete candidate actions.

    Each example is (frozen observation, action mask, normalized repair gains).
    Ties contribute no preference gradient. This is BOPO-inspired supervision,
    not BOPO's original sequence-likelihood objective.
    """
    device = next(policy.parameters()).device
    losses = []
    for _ in range(epochs):
        for observation, mask, gains in examples:
            gains = torch.as_tensor(gains, dtype=torch.float32, device=device)
            active = torch.where(mask.to(device))[0]
            if len(active) < 2:
                continue
            best = active[gains[active].argmax()]
            worse = active[gains[active] < gains[best]-1e-7]
            if not len(worse):
                continue
            dist, _ = policy(observation, mask)
            gaps = gains[best]-gains[worse]
            weights = gaps/gaps.sum()
            loss = (weights*F.softplus(-(dist.logits[best]-dist.logits[worse]))).sum()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())
    return dict(updates=len(losses), mean_loss=float(np.mean(losses)) if losses else None)
