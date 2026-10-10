"""Compact task-set LNS with an exact nonempty-subset PPO likelihood.

The factorized destroy action follows Wu et al. (NeurIPS 2021). Conditioning
on a nonempty set changes its normalization, which is retained in every update.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
from torch.nn import functional as F

from .rl_bipartite import bipartite_observation
from .rl_portfolio import PortfolioAction, PortfolioEnv


class NonemptySubset:
    """Independent Bernoulli bits conditioned on at least one selected task."""

    def __init__(self, logits, *, stopped=False):
        if logits.ndim != 1 or not len(logits) or not torch.isfinite(logits).all():
            raise ValueError('Expected a finite nonempty logits vector')
        self.logits, self.stopped = logits, stopped
        self.log_zero = F.logsigmoid(-logits).sum()
        # -expm1 is stable when the unconditioned empty-set probability is near 1.
        self.log_normalizer = torch.log(-torch.expm1(self.log_zero))
        if not torch.isfinite(self.log_normalizer):
            raise ValueError('Nonempty-set mass is below the numerical range')

    def sample(self):
        if self.stopped:
            return torch.zeros_like(self.logits, dtype=torch.long)
        # Sample the first selected task, then independent bits after it. This
        # avoids rejection loops even when the empty-set probability is large.
        log_q = F.logsigmoid(-self.logits)
        prefix = torch.cat([log_q.new_zeros(1), log_q.cumsum(0)[:-1]])
        first = Categorical(logits=prefix+F.logsigmoid(self.logits)).sample()
        bits = Bernoulli(logits=self.logits).sample().long()
        indices = torch.arange(len(bits), device=bits.device)
        return torch.where(indices < first, 0, torch.where(indices == first, 1, bits))

    def mode(self):
        bits = (self.logits >= 0).long()
        if self.stopped:
            return torch.zeros_like(bits)
        if not bits.any():
            bits[self.logits.argmax()] = 1
        return bits

    def log_prob(self, bits):
        bits = torch.as_tensor(bits, device=self.logits.device)
        if bits.shape != self.logits.shape or not ((bits == 0) | (bits == 1)).all():
            raise ValueError('Invalid task subset')
        if self.stopped:
            return self.logits.sum()*0 if not bits.any() else self.logits.new_tensor(-torch.inf)
        if not bits.any():
            return self.logits.new_tensor(-torch.inf)
        return Bernoulli(logits=self.logits).log_prob(bits.float()).sum()-self.log_normalizer

    def entropy(self):
        if self.stopped:
            return self.logits.sum()*0
        base = Bernoulli(logits=self.logits).entropy().sum()
        return (base+self.log_zero.exp()*self.log_zero)/(-torch.expm1(self.log_zero))+self.log_normalizer


class SubsetEnv(PortfolioEnv):
    """One generated group, the existing profile repair and complete validator.

    Unlike an enumerated portfolio, refresh constructs only a constant-size
    placeholder. It is replaced with the sampled group immediately before repair.
    """

    def refresh(self):
        self.actions = [PortfolioAction(4, torch.ones_like(self.incumbent),
                                        torch.zeros(10), tuple(range(self.problem.J)))]
        self.done = self.remaining() < self.profile_floor

    def step_subset(self, bits):
        bits = torch.as_tensor(bits)
        if bits.shape != (self.problem.J,) or not ((bits == 0) | (bits == 1)).all() or not bits.any():
            raise ValueError('Select at least one task')
        jobs = tuple(torch.where(bits)[0].tolist())
        mask = bits.bool()[:, None].expand_as(self.incumbent).clone()
        self.actions = [PortfolioAction(4, mask, torch.zeros(10), jobs)]
        return super().step(1)


class SubsetActorCritic(nn.Module):
    """Linear-size task--time message pooling, without candidate enumeration."""

    def __init__(self, hidden=24, graph=True):
        super().__init__()
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        self.hidden, self.graph = hidden, graph
        self.task = nn.Sequential(nn.Linear(15, hidden), nn.Tanh())
        self.time = nn.Sequential(nn.Linear(9, hidden), nn.Tanh())
        self.edge = nn.Sequential(nn.Linear(6, hidden), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(3*hidden+8, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(2*hidden+8, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def observation(self, env):
        return bipartite_observation(env)

    def forward(self, observation, mask):
        device = next(self.parameters()).device
        task = self.task(observation.tasks.to(device))
        time = self.time(observation.times.to(device))
        eligible = observation.eligible.to(device).float()
        if self.graph:
            edge = self.edge(observation.edges.to(device))
            message = ((time[None]+edge)*eligible[..., None]).sum(1)/eligible.sum(1).clamp(min=1)[:, None]
        else:
            message = time.mean(0).expand_as(task)
        global_state = torch.cat([task.mean(0), time.mean(0), observation.rows[0, 19:27].to(device)])
        logits = self.actor(torch.cat([task, message, global_state[self.hidden:].expand(len(task), -1)], 1)).squeeze(-1)
        value = self.critic(global_state).squeeze(-1)
        return NonemptySubset(logits.clamp(-15, 15), stopped=bool(mask[0])), value

    def save(self, path, metadata=None):
        torch.save(dict(format='onts-subset-v1', hidden=self.hidden, graph=self.graph,
                        state_dict=self.state_dict(), metadata=metadata or {}), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['format'] != 'onts-subset-v1':
            raise ValueError('Unsupported subset checkpoint')
        model = cls(data['hidden'], data['graph'])
        model.load_state_dict(data['state_dict'])
        return model.eval()


def subset_rollout(env, policy=None, *, greedy=False):
    trajectory = []
    while not env.done:
        started = env.clock()
        observation, mask = bipartite_observation(env)
        if mask[0]:
            env.step(0)
            env.log[-1]['selection_runtime'] = env.clock()-started
            break
        if policy is None:
            count = int(env.selection_rng.integers(1, env.problem.J+1))
            bits = torch.zeros(env.problem.J, dtype=torch.long)
            bits[env.selection_rng.choice(env.problem.J, count, replace=False)] = 1
            logp, value = 0., 0.
        else:
            with torch.no_grad():
                distribution, estimate = policy(observation, mask)
                bits = distribution.mode() if greedy else distribution.sample()
                logp, value = float(distribution.log_prob(bits)), float(estimate)
        selection_time = env.clock()-started
        reward, _ = env.step_subset(bits)
        env.log[-1]['selection_runtime'] = selection_time
        trajectory.append((observation, mask, bits.tolist(), logp, value, reward))
    return trajectory


def subset_imitation(policy, optimizer, examples, *, epochs=1, seed=0):
    """Quality-weighted exact subset NLL; labels must be validated repairs."""
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        for i in rng.permutation(len(examples)):
            observation, mask, bits, weight = examples[i]
            dist, _ = policy(observation, mask)
            loss = -float(weight)*dist.log_prob(torch.as_tensor(bits)) / len(bits)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite subset imitation loss')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    return dict(updates=len(losses), loss=float(np.mean(losses)) if losses else 0.)
