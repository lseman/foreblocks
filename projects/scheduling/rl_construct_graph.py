"""Task--time bipartite attention with explicit temporal battery context.

Graph incidence uses task windows, with execution and residual power on edges.
No task-index embeddings: task permutations permute candidate logits. Time is
ordered; a bidirectional GRU sees the full offline solar/resource trajectory.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .job_profile import battery_trace


def graph_observation(env, choices):
    p, i = env.problem, env.problem.instance
    slot = np.arange(p.T)
    power = i.uso_p.cpu().double().numpy()
    solar = i.recurso_p.cpu().double().numpy()
    usage = power @ env.schedule
    scale = max(1., float(power.max()), float(solar.max()), p.config.bat_usage * p.config.v_bat)
    eligible = (slot[None] >= i.win_min.cpu().numpy()[:, None]) & (slot[None] < i.win_max.cpu().numpy()[:, None])
    remaining = np.array([j in env.remaining for j in range(p.J)])
    task = np.stack([
        i.priority.cpu().numpy() / max(1., float(i.priority.max())), power / scale,
        *[getattr(i, name).cpu().numpy() / p.T for name in
          ("min_statup", "max_statup", "min_cpu_time", "max_cpu_time", "min_periodo_job", "max_periodo_job", "win_min", "win_max")],
        remaining, env.schedule.mean(1),
    ], -1)
    charge = battery_trace(p, usage)
    mandatory = np.zeros(p.T)
    for j in env.remaining:
        width = max(1, int(eligible[j].sum()))
        mandatory += eligible[j] * float(i.min_statup[j] * i.min_cpu_time[j]) * power[j] / width
    times = np.stack([
        solar / scale, usage / scale, (solar + p.config.bat_usage * p.config.v_bat - usage) / scale,
        charge, slot / p.T, remaining.astype(float) @ eligible / max(1, len(env.remaining)),
        mandatory / scale, np.cumsum(solar - usage) / max(1., scale * p.T),
    ], -1)
    edges = np.stack([
        env.schedule, np.broadcast_to(power[:, None] / scale, eligible.shape),
        (solar[None] + p.config.bat_usage * p.config.v_bat - usage[None] - power[:, None]) / scale,
    ], -1)
    return dict(tasks=torch.tensor(task, dtype=torch.float32),
                times=torch.tensor(times, dtype=torch.float32),
                edges=torch.tensor(edges, dtype=torch.float32),
                eligible=torch.tensor(eligible), remaining=torch.tensor(remaining),
                task_indices=torch.tensor([c.task for c in choices]),
                profiles=torch.tensor(np.stack([c.row for c in choices]), dtype=torch.float32),
                features=torch.tensor([c.features for c in choices], dtype=torch.float32))


class RelationAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query, self.key, self.value = [nn.Linear(width, width) for _ in range(3)]
        self.edge_bias = nn.Linear(3, 1)
        self.edge_value = nn.Linear(3, width)
        self.norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.output_norm = nn.LayerNorm(width)

    def forward(self, target, source, edges, mask):
        score = self.query(target) @ self.key(source).T / target.shape[-1] ** .5
        score = score + self.edge_bias(edges).squeeze(-1)
        # Rows with no edges have a zero message, rather than NaN softmax.
        weights = score.masked_fill(~mask, -1e9).softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        message = weights @ self.value(source) + (weights[..., None] * self.edge_value(edges)).sum(1)
        state = self.norm(target + message)
        return self.output_norm(state + self.feedforward(state))


class GraphConstructPolicy(nn.Module):
    def __init__(self, width=48, layers=2):
        super().__init__()
        self.task_input = nn.Linear(12, width)
        self.time_input = nn.Linear(8, width)
        self.temporal = nn.GRU(width, width // 2, bidirectional=True, batch_first=True)
        self.to_task = nn.ModuleList(RelationAttention(width) for _ in range(layers))
        self.to_time = nn.ModuleList(RelationAttention(width) for _ in range(layers))
        self.actor = nn.Sequential(nn.Linear(16 + 4 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, observation):
        device = next(self.parameters()).device
        o = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)).to(device) for k, v in observation.items()}
        task = self.task_input(o["tasks"].float())
        time = self.time_input(o["times"].float())
        time, _ = self.temporal(time[None])
        time = time[0]
        edges, eligible = o["edges"].float(), o["eligible"].bool()
        for to_task, to_time in zip(self.to_task, self.to_time):
            task = to_task(task, time, edges, eligible)
            time = to_time(time, task, edges.transpose(0, 1), eligible.T)
        profile = o["profiles"].float()
        local_time = profile @ time / profile.sum(1, keepdim=True).clamp_min(1.)
        remaining = o["remaining"].float()
        global_task = (remaining[:, None] * task).sum(0) / remaining.sum().clamp_min(1.)
        count = profile.shape[0]
        features = torch.cat([o["features"].float(), task[o["task_indices"].long()], local_time,
                              global_task.expand(count, -1), time.mean(0).expand(count, -1)], -1)
        return self.actor(features).squeeze(-1)

    def score_choices(self, env, choices, features):
        return self(graph_observation(env, choices))
