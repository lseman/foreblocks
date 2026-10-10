"""MIStar-inspired ONTS memory and multiple-proposal policy.

This is an adaptation, not an FJSP reproduction: task/time attention replaces
operation-machine hyper-arcs, task-group repair replaces Nopt2, and invariant
schedule agreement replaces order-index Frobenius retrieval. The environment
must evaluate the sampled proposals from the SAME current schedule, charging
all evaluations, and record the whole ordered proposal vector for PPO.
"""
from pathlib import Path
import math
import torch
from torch import nn
from torch.distributions import Categorical


class MultiProposal:
    """IID categorical proposals, including repetitions; exact joint likelihood."""
    def __init__(self, logits, proposal_count=4):
        if proposal_count < 1:
            raise ValueError('proposal_count must be positive')
        self.base = Categorical(logits=logits)
        self.proposal_count = proposal_count
        self.logits = self.base.logits

    def sample(self):
        return self.base.sample((self.proposal_count,))

    def mode(self):
        return self.logits.argmax().expand(self.proposal_count).clone()

    def log_prob(self, proposals):
        proposals = torch.as_tensor(proposals, device=self.logits.device)
        if proposals.shape != (self.proposal_count,):
            raise ValueError('Record every ordered proposal, including duplicates')
        return self.base.log_prob(proposals).sum()

    def entropy(self):
        return self.proposal_count * self.base.entropy()


class MistarActorCritic(nn.Module):
    def __init__(self, hidden=24, proposal_count=4, memory_topk=4, memory=True,
                 repetition_penalty=.2):
        super().__init__()
        if min(hidden, proposal_count, memory_topk) < 1 or repetition_penalty < 0:
            raise ValueError('Positive dimensions and nonnegative penalty required')
        self.hidden, self.proposal_count = hidden, proposal_count
        self.memory_topk, self.memory = memory_topk, memory
        self.repetition_penalty = repetition_penalty
        self.task_input, self.time_input = nn.Linear(15, hidden), nn.Linear(9, hidden)
        self.edge_input = nn.Linear(6, hidden)
        self.task_query, self.time_key = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.memory_input = nn.Sequential(nn.Linear(2*hidden+4, hidden), nn.GELU())
        self.actor = nn.Sequential(nn.Linear(4*hidden+22, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(3*hidden+10, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.last_retrieved_indices = torch.empty(0, dtype=torch.long)

    def _retrieve(self, obs, task, time):
        device = task.device
        empty = task.new_zeros(self.hidden)
        self.last_retrieved_indices = torch.empty(0, dtype=torch.long, device=device)
        history = obs.history_schedules.to(device).bool()
        groups = obs.groups.to(device).bool()
        if not self.memory or not len(history):
            return empty, task.new_zeros(len(groups)), task.new_zeros(())
        current = obs.current_schedule.to(device).bool()
        # Equality includes idle slots and is invariant to a joint task permutation.
        similarity = (history == current[None]).float().mean((1, 2))
        k = min(self.memory_topk, len(history))
        indices = torch.topk(similarity, k, sorted=True).indices
        self.last_retrieved_indices = indices.detach()
        selected = obs.history_groups.to(device)[indices].float()
        action_embed = selected @ task / selected.sum(1, keepdim=True).clamp(min=1)
        schedule = history[indices].float()
        slot_weights = schedule.sum(1)
        schedule_embed = slot_weights @ time / slot_weights.sum(1, keepdim=True).clamp(min=1)
        features = torch.stack([similarity[indices], obs.history_gains.to(device)[indices],
                                obs.history_objectives.to(device)[indices],
                                obs.history_durations.to(device)[indices]], 1)
        embedded = self.memory_input(torch.cat([action_embed, schedule_embed, features], 1))
        weights = torch.softmax(similarity[indices], 0)
        retrieved = weights @ embedded
        # A matching action in another schedule is distinct from exact revisitation.
        exact_state = (history[indices] == current[None]).flatten(1).all(1)
        exact_group = (groups[:, None] == selected.bool()[None]).all(-1)
        repetition = (exact_group & exact_state[None]).float().mean(1)
        return retrieved, repetition, similarity[indices].mean()

    def observation(self, env):
        return env.observe()

    def forward(self, observation, mask):
        device = next(self.parameters()).device
        task = torch.nn.functional.gelu(self.task_input(observation.tasks.to(device)))
        time = torch.nn.functional.gelu(self.time_input(observation.times.to(device)))
        edge = torch.nn.functional.gelu(self.edge_input(observation.edges.to(device)))
        eligible = observation.eligible.to(device).bool()
        scores = (self.task_query(task)[:, None] * (self.time_key(time)[None] + edge)).sum(-1) / math.sqrt(self.hidden)
        # Stable attention for tasks with no eligible slots: their message is zero.
        attention = torch.softmax(scores.masked_fill(~eligible, -1e9), -1) * eligible
        attention = attention / attention.sum(1, keepdim=True).clamp(min=1e-8)
        task = task + (attention[..., None] * (time[None]+edge)).sum(1)
        memory, repetition, _ = self._retrieve(observation, task, time)
        groups = observation.groups.to(device).float()
        pooled = groups @ task / groups.sum(1, keepdim=True).clamp(min=1)
        n = len(groups)
        context = observation.context.to(device)
        features = torch.cat([pooled, task.mean(0).expand(n, -1), time.mean(0).expand(n, -1),
                              memory.expand(n, -1), observation.action_features.to(device),
                              context.expand(n, -1)], 1)
        valid = mask.to(device).bool()
        if valid.shape != (n,) or not valid.any():
            raise ValueError('At least one valid candidate required')
        logits = self.actor(features).squeeze(-1) - self.repetition_penalty * repetition
        logits = logits.masked_fill(~valid, -torch.inf)
        value = self.critic(torch.cat([task.mean(0), time.mean(0), memory, context])).squeeze(-1)
        return MultiProposal(logits, self.proposal_count), value

    def save(self, path: str | Path, *, metadata=None):
        torch.save(dict(format='onts-mistar-v1', hidden=self.hidden,
                        proposal_count=self.proposal_count, memory_topk=self.memory_topk,
                        memory=self.memory, repetition_penalty=self.repetition_penalty,
                        state_dict=self.state_dict(), metadata=metadata or {}), path)

    @classmethod
    def load(cls, path: str | Path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('format') != 'onts-mistar-v1':
            raise ValueError('Unsupported MIStar checkpoint')
        model = cls(**{k: state[k] for k in ('hidden', 'proposal_count', 'memory_topk',
                                           'memory', 'repetition_penalty')})
        model.load_state_dict(state['state_dict'])
        model.training_metadata = state.get('metadata', {})
        return model.eval()
