"""ReSched-inspired ONTS candidate-repair policy, not an FJSP reproduction.

Source: https://arxiv.org/html/2603.07020v1 (sections 4.2.1--4.2.2).
We adapt edge-conditioned query/key/value attention and learned soft self
connections to task--time entities. ONTS has no ordered operation chain: tasks
remain permutation equivariant and sinusoidal position is attached only to
physical time slots. The original four-feature construction MDP, operation
RoPE, makespan reward and REINFORCE training are not reproduced. This network
uses the shared ONTS observation and selects concrete LNS repair candidates.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.distributions import Categorical


class _RelationalBlock(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.task_attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        # ReSched shares edge projection across Q/K/V and heads.
        self.edge = nn.Linear(6, self.width)
        self.output = nn.Linear(hidden, hidden)
        self.task_norm = nn.LayerNorm(hidden)
        self.time_norm = nn.LayerNorm(hidden)
        self.task_ff = nn.Sequential(nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Linear(2*hidden, hidden))
        self.time_ff = nn.Sequential(nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Linear(2*hidden, hidden))
        self.task_ff_norm = nn.LayerNorm(hidden)
        self.time_ff_norm = nn.LayerNorm(hidden)

    def forward(self, task, time, edges, eligible):
        attended = self.task_attention(task[None], task[None], task[None], need_weights=False)[0][0]
        task = self.task_norm(task + attended)
        task = self.task_ff_norm(task + self.task_ff(task))
        # Time queries attend eligible tasks plus their own time representation.
        q = self.query(time).reshape(len(time), self.heads, self.width)
        k = self.key(task).reshape(len(task), self.heads, self.width)
        v = self.value(task).reshape(len(task), self.heads, self.width)
        e = self.edge(edges).permute(1, 0, 2)[:, :, None, :]
        scores = ((q[:, None] + e) * (k[None] + e)).sum(-1) / math.sqrt(self.width)
        scores = scores.masked_fill(~eligible.T[:, :, None], -torch.inf)
        self_key = self.key(time).reshape_as(q)
        self_value = self.value(time).reshape_as(q)
        self_scores = (q*self_key).sum(-1) / math.sqrt(self.width)
        weights = torch.cat([scores, self_scores[:, None]], 1).softmax(1)
        message = (weights[:, :-1, :, None]*(v[None]+e)).sum(1)
        message = message + weights[:, -1, :, None]*self_value
        time = self.time_norm(time + self.output(message.reshape(len(time), -1)))
        time = self.time_ff_norm(time + self.time_ff(time))
        return task, time


class ReschedActorCritic(nn.Module):
    """Task competition, edge-aware time aggregation and candidate scoring.

    No task index encoding is used. History is intentionally omitted from this
    architecture; MIStar-style memory is a separate comparison. Availability
    masks come from the environment and must contain at least one true entry.
    """
    def __init__(self, hidden=24, heads=2, layers=1):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (hidden, heads, layers)) or hidden % heads:
            raise ValueError('Positive integer dimensions and hidden divisible by heads required')
        self.hidden, self.heads, self.layers = hidden, heads, layers
        self.task = nn.Linear(15, hidden)
        self.time = nn.Linear(9, hidden)
        self.blocks = nn.ModuleList([_RelationalBlock(hidden, heads) for _ in range(layers)])
        self.actor = nn.Sequential(nn.Linear(3*hidden+22, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(2*hidden+10, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.zeros_(self.actor[-1].bias)

    def observation(self, env):
        return env.observe()

    def forward(self, obs, mask):
        device = next(self.parameters()).device
        groups = obs.groups.to(device=device, dtype=torch.bool)
        mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
        if mask.shape != (len(groups),) or not mask.any():
            raise ValueError('Expected a nonempty available-candidate mask')
        task = self.task(obs.tasks.to(device))
        time = self.time(obs.times.to(device))
        if not len(task) or not len(time) or groups.shape[1:] != (len(task),):
            raise ValueError('Expected nonempty tasks/time and candidate task masks')
        # Absolute physical time carries ONTS windows/periodicity; task identity
        # is exchangeable. This is sinusoidal time encoding, not operation RoPE.
        position = torch.arange(len(time), device=device, dtype=time.dtype)[:, None]
        freq = torch.exp(-math.log(10000)*torch.arange(0, self.hidden, 2, device=device, dtype=time.dtype)/self.hidden)
        encoding = torch.zeros_like(time)
        encoding[:, 0::2] = torch.sin(position*freq)
        encoding[:, 1::2] = torch.cos(position*freq[:self.hidden//2])
        time = time + encoding
        eligible = obs.eligible.to(device=device, dtype=torch.bool)
        edges = obs.edges.to(device)
        for block in self.blocks:
            task, time = block(task, time, edges, eligible)
        memberships = groups.to(task.dtype)
        selected = memberships @ task / memberships.sum(1).clamp(min=1)[:, None]
        context = obs.context.to(device)
        global_state = torch.cat([task.mean(0), time.mean(0), context])
        logits = self.actor(torch.cat([selected, global_state.expand(len(groups), -1),
                                      obs.action_features.to(device)], -1)).squeeze(-1)
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf)), self.critic(global_state).squeeze(-1)

    def save(self, path, metadata=None):
        torch.save(dict(format='onts-resched-v1', config=dict(hidden=self.hidden, heads=self.heads, layers=self.layers),
                        state_dict=self.state_dict(), metadata=metadata or {}), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('format') != 'onts-resched-v1':
            raise ValueError('Unsupported ReSched-adaptation checkpoint')
        model = cls(**data['config'])
        model.load_state_dict(data['state_dict'])
        return model.eval()
