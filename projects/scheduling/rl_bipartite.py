"""Task--time graph policies for concrete ONTS repair neighborhoods."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from .job_profile import select_jobs
from .rl_portfolio import PortfolioAction, PortfolioEnv, PortfolioObservation, ROW_WIDTH


class TaskNeighborhoodEnv(PortfolioEnv):
    """Offer every singleton/pair of tasks, plus three triple proposals and MILP.

    Failed concrete profile subsets are masked until the incumbent changes.
    This is search diversification, not a claim that every failed repair proves
    local optimality. The underlying repair, validation and deadline rules are
    inherited unchanged. Enumeration has quadratic complexity in task count.
    """
    def __init__(self, *args, failure_memory=True, **kwargs):
        self.failure_memory = failure_memory
        self.failed_subsets = set()
        super().__init__(*args, **kwargs)

    def refresh(self):
        super().refresh()
        self.actions = self.actions[:4*len(self.repair_budgets)]
        groups = [(tuple(jobs), 4) for count in (1, 2)
                  for jobs in combinations(range(self.problem.J), count)]
        if self.problem.J >= 3:
            for rule in ('pressure', 'utility', 'random'):
                jobs = tuple(select_jobs(self.problem, self.incumbent, self.rng, 3, rule=rule))
                groups.append((jobs, 5 if rule != 'random' else 6))
        for jobs, family in groups:
            mask = torch.zeros_like(self.incumbent)
            mask[list(jobs)] = True
            chosen = max(int(mask.sum()), 1)
            features = torch.tensor([
                0., 0., 0., 0., mask.float().mean(),
                (mask & self.incumbent).sum()/chosen, len(jobs)/self.problem.J, 1.,
                (mask & self.state.last_mask).sum()/chosen, 0.,
            ], dtype=torch.float32)
            self.actions.append(PortfolioAction(family, mask, features, jobs))

    def action_mask(self):
        mask = super().action_mask()
        if self.failure_memory:
            for index, action in enumerate(self.actions, 1):
                if action.family >= 4 and tuple(sorted(action.jobs)) in self.failed_subsets:
                    mask[index] = False
            mask[0] = not bool(mask[1:].any())
        return mask

    def step(self, action):
        previous = self.incumbent.clone()
        selected = self.actions[action-1] if action else None
        reward, done = super().step(action)
        if not torch.equal(previous, self.incumbent):
            self.failed_subsets.clear()
        if selected is not None and selected.family >= 4:
            self.log[-1]['family'] = ('profile_direct' if selected.family == 4
                                     else 'profile_triple')
            if self.log[-1]['executed'] and reward == 0 and torch.equal(previous, self.incumbent):
                self.failed_subsets.add(tuple(sorted(selected.jobs)))
        return reward, done


@dataclass
class BipartiteObservation(PortfolioObservation):
    edges: torch.Tensor
    eligible: torch.Tensor


def bipartite_observation(env):
    """Freeze task--slot relations using the validated half-open task windows."""
    base, mask = env.observation()
    inst, x = env.problem.instance, env.incumbent
    slots = torch.arange(inst.T)
    eligible = (slots >= inst.win_min[:, None]) & (slots < inst.win_max[:, None])
    starts = x & ~torch.cat([torch.zeros(inst.J, 1, dtype=torch.bool), x[:, :-1]], 1)
    draw = inst.uso_p.float()[:, None]
    scale = max(float(inst.recurso_p.max()), float(inst.uso_p.max()), 1.)
    usage = inst.uso_p.float() @ x.float()
    # The residual load is descriptive; it is not a battery-feasibility oracle.
    edges = torch.stack([
        x.float(), starts.float(), env.state.last_mask.float(), env.state.changed.float(),
        (usage[None, :]-draw*x.float())/scale,
        draw.expand_as(x)/scale,
    ], -1).float()
    edges *= eligible[..., None]
    return BipartiteObservation(base.rows, base.tasks, base.times, base.released,
                               edges, eligible), mask


class EdgeAttention(nn.Module):
    """Masked multi-head cross attention with edge terms in keys and values."""
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads, self.width = heads, hidden//heads
        self.query, self.key, self.value = (nn.Linear(hidden, hidden) for _ in range(3))
        self.edge_key, self.edge_value = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, target, source, edges, adjacency):
        n, m, h, d = len(target), len(source), self.heads, self.width
        query = self.query(target).view(n, h, d)
        key = self.key(source).view(m, h, d)
        edge_key = self.edge_key(edges).view(n, m, h, d)
        score = ((query[:, None]*(key[None]+edge_key)).sum(-1)/(d**.5))
        score = score.masked_fill(~adjacency[..., None], -1e9)
        weights = score.softmax(1)*adjacency[..., None]
        # Isolated nodes receive zero messages; avoid an all--inf softmax.
        values = self.value(source).view(m, h, d)[None]+self.edge_value(edges).view(n, m, h, d)
        message = (weights[..., None]*values).sum(1).reshape(n, h*d)
        return self.norm(target+self.output(message))


class BipartiteActorCritic(nn.Module):
    """Score complete task groups from task/time/edge representations.

    The graph-off control has the same readout and parameter shapes, but zero
    edge features and no cross-node messages. Candidate scoring uses exact
    released task--slot pairs; time position remains in the original features.
    """
    def __init__(self, hidden=32, heads=4, layers=2, *, graph=True):
        super().__init__()
        if min(hidden, heads, layers) < 1 or hidden % heads:
            raise ValueError('Positive dimensions and hidden divisible by heads required')
        self.hidden, self.heads, self.layers, self.graph = hidden, heads, layers, graph
        self.task_input, self.time_input = nn.Linear(16, hidden), nn.Linear(10, hidden)
        self.edge_input = nn.Linear(6, hidden)
        self.to_task = nn.ModuleList([EdgeAttention(hidden, heads) for _ in range(layers)])
        self.to_time = nn.ModuleList([EdgeAttention(hidden, heads) for _ in range(layers)])
        self.pair = nn.Sequential(nn.Linear(3*hidden, hidden), nn.GELU())
        self.readout = nn.Sequential(nn.Linear(ROW_WIDTH+4*hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.actor = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(ROW_WIDTH+2*hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def observation(self, env):
        return bipartite_observation(env)

    def forward(self, observation, mask):
        device = next(self.parameters()).device
        rows, tasks, times, released, edges, eligible = (t.to(device) for t in (
            observation.rows, observation.tasks, observation.times, observation.released,
            observation.edges, observation.eligible))
        task_degree = eligible.float().mean(1, keepdim=True)
        time_degree = eligible.float().mean(0)[:, None]
        task = F.gelu(self.task_input(torch.cat([tasks, task_degree], -1)))
        time = F.gelu(self.time_input(torch.cat([times, time_degree], -1)))
        edge = F.gelu(self.edge_input(edges if self.graph else torch.zeros_like(edges)))
        if self.graph:
            for to_task, to_time in zip(self.to_task, self.to_time):
                task = to_task(task, time, edge, eligible)
                time = to_time(time, task, edge.transpose(0, 1), eligible.T)
        j, t = len(task), len(time)
        pair = self.pair(torch.cat([task[:, None].expand(j, t, -1),
                                   time[None].expand(j, t, -1), edge], -1))
        weights = (released & eligible[None]).float()
        joint = weights.flatten(1) @ pair.flatten(0, 1)/weights.sum((1, 2))[:, None].clamp(min=1)
        task_weights, time_weights = weights.any(2).float(), weights.any(1).float()
        task_pool = task_weights @ task/task_weights.sum(1, keepdim=True).clamp(min=1)
        time_pool = time_weights @ time/time_weights.sum(1, keepdim=True).clamp(min=1)
        global_state = torch.cat([task.mean(0), time.mean(0)])
        # Candidate context includes a shared average of both node types.
        pooled_global = .5*(task.mean(0)+time.mean(0))
        features = torch.cat([rows, joint, task_pool, time_pool,
                              pooled_global.expand(len(rows), -1)], -1)
        logits = self.actor(self.readout(features)).squeeze(-1).masked_fill(~mask.to(device), -torch.inf)
        value = self.critic(torch.cat([rows[0], global_state])).squeeze(-1)
        return Categorical(logits=logits), value

    def save(self, path: str | Path, *, metadata=None):
        torch.save(dict(format='onts-bipartite-v1', hidden=self.hidden, heads=self.heads,
                        layers=self.layers, graph=self.graph, state_dict=self.state_dict(),
                        metadata=metadata or {}), path)

    @classmethod
    def load(cls, path: str | Path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('format') != 'onts-bipartite-v1':
            raise ValueError('Unsupported bipartite checkpoint')
        model = cls(checkpoint['hidden'], checkpoint['heads'], checkpoint['layers'], graph=checkpoint['graph'])
        model.load_state_dict(checkpoint['state_dict'])
        model.training_metadata = checkpoint.get('metadata', {})
        return model.eval()


def listwise_update(policy, optimizer, examples, *, epochs=12, seed=0, temperature=.2):
    """Distill repaired action utilities; keep all tied/high-quality positives.

    Utilities are normalized within each state before a soft target is formed.
    This is outcome distillation inspired by contrastive LNS, not CL-LNS's loss.
    """
    if epochs < 1 or temperature <= 0:
        raise ValueError('Positive epochs and temperature required')
    generator = torch.Generator().manual_seed(seed)
    device = next(policy.parameters()).device
    losses = []
    for _ in range(epochs):
        for index in torch.randperm(len(examples), generator=generator).tolist():
            observation, mask, utility = examples[index]
            active = torch.where(mask.to(device))[0]
            utility = torch.as_tensor(utility, dtype=torch.float32, device=device)[active]
            span = utility.max()-utility.min()
            if len(active) < 2 or span < 1e-8:
                continue
            target = ((utility-utility.min())/(span*temperature)).softmax(0)
            dist, _ = policy(observation, mask)
            loss = -(target*dist.logits[active]).sum()
            optimizer.zero_grad()
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(policy.parameters(), .5)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError('Nonfinite distillation update')
            optimizer.step()
            losses.append(float(loss.detach()))
    return dict(updates=len(losses), mean_loss=sum(losses)/max(len(losses), 1))
