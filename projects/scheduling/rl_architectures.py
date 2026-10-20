"""Comparable profile-selection policies sharing raw observations and decoder.

These are ONTS adaptations, not replications of published routing/branching
models. Pointer uses an ordered recurrent task encoder; graph uses task-time
incidence. No model sees a feasible witness or MILP output at inference.
"""
from __future__ import annotations

import torch
from torch import nn

from .rl_construct_graph import RelationAttention, graph_observation

ARCHITECTURES = ("mlp", "pointer", "bipartite", "transformer")


def observation(env, choices, features):
    result = graph_observation(env, choices)
    result["features"] = features
    return result


class ArchitecturePolicy(nn.Module):
    def __init__(self, architecture, width=32):
        super().__init__()
        if architecture not in ARCHITECTURES or width < 4 or width % 4:
            raise ValueError("Unknown architecture or invalid width")
        self.architecture = architecture
        self.task_input = nn.Sequential(nn.Linear(12, width), nn.GELU())
        self.time_input = nn.Sequential(nn.Linear(8, width), nn.GELU())
        # Every recipe receives identical raw data and profile-edge summaries.
        if architecture in {"pointer", "bipartite"}:
            self.time_encoder = nn.GRU(width, width//2, bidirectional=True, batch_first=True)
        if architecture == "pointer":
            self.task_encoder = nn.GRU(width, width//2, bidirectional=True, batch_first=True)
        if architecture == "bipartite":
            self.to_task = nn.ModuleList(RelationAttention(width) for _ in range(2))
            self.to_time = nn.ModuleList(RelationAttention(width) for _ in range(2))
        if architecture == "transformer":
            layer = nn.TransformerEncoderLayer(width, 4, 2*width, dropout=0., batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.node_type = nn.Parameter(torch.zeros(2, width))
        self.candidate = nn.Sequential(nn.Linear(23+4*width, 64), nn.GELU(), nn.Linear(64, width), nn.GELU())
        if architecture == "pointer":
            self.key = nn.Linear(width, width, bias=False)
            self.query = nn.Linear(2*width, width, bias=False)
            self.pointer = nn.Linear(width, 1, bias=False)
        else:
            self.actor = nn.Linear(width, 1)

    def forward(self, raw):
        device = next(self.parameters()).device
        o = {k: torch.as_tensor(v, device=device) for k,v in raw.items()}
        task, time = self.task_input(o["tasks"].float()), self.time_input(o["times"].float())
        if self.architecture in {"pointer", "bipartite"}:
            time = self.time_encoder(time[None])[0][0]
        if self.architecture == "pointer":
            task = self.task_encoder(task[None])[0][0]
        if self.architecture == "bipartite":
            for to_task,to_time in zip(self.to_task,self.to_time):
                task = to_task(task,time,o["edges"].float(),o["eligible"].bool())
                time = to_time(time,task,o["edges"].float().transpose(0,1),o["eligible"].bool().T)
        if self.architecture == "transformer":
            joint = self.encoder(torch.cat([task+self.node_type[0],time+self.node_type[1]])[None])[0]
            task,time = joint[:task.shape[0]],joint[task.shape[0]:]
        profiles = o["profiles"].float(); indices = o["task_indices"].long()
        normalized = profiles / profiles.sum(1,keepdim=True).clamp_min(1.)
        local_time = normalized @ time
        local_edge = (normalized[:,:,None]*o["edges"].float()[indices]).sum(1)
        remaining = o["remaining"].float()
        global_task = (remaining[:,None]*task).sum(0)/remaining.sum().clamp_min(1.)
        global_time = time.mean(0); count = profiles.shape[0]
        candidate = self.candidate(torch.cat([o["features"].float(),task[indices],local_time,local_edge,
                    global_task.expand(count,-1),global_time.expand(count,-1)],-1))
        if self.architecture == "pointer":
            query = self.query(torch.cat([global_task,global_time]))
            return self.pointer(torch.tanh(self.key(candidate)+query)).squeeze(-1)
        return self.actor(candidate).squeeze(-1)

    def score_choices(self, env, choices, features):
        return self(observation(env, choices, features))
