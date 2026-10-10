"""RL-LNS adaptation with the actual ONTS variable--constraint graph.

Wu et al. (NeurIPS 2021) select individual IP variables independently. Here
x/phi embeddings are pooled into complete task blocks and the Bernoulli policy
is conditioned on a nonempty block subset; repair and training are supplied by
the common ONTS engine. This is an adaptation, not an original-paper replica.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .rl_subset import NonemptySubset


@dataclass(frozen=True)
class LNSObservation:
    variables: torch.Tensor
    constraints: torch.Tensor
    edge_index: torch.Tensor
    coefficients: torch.Tensor
    task_index: torch.Tensor
    context: torch.Tensor
    jobs: int


def _topology(problem):
    matrix = problem.A.tocoo(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    rows, columns = matrix.row.astype(np.int64), matrix.col.astype(np.int64)
    scale = np.maximum(1., np.bincount(rows, weights=np.abs(matrix.data),
                                      minlength=matrix.shape[0]))
    degree = np.bincount(rows, minlength=matrix.shape[0])
    lower, upper = np.asarray(problem.row_lower), np.asarray(problem.row_upper)
    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
    row_lower = np.where(finite_lower, lower / scale, 0.)
    row_upper = np.where(finite_upper, upper / scale, 0.)
    objective = np.asarray(problem.c)
    static_vars = np.column_stack([
        objective / max(1., float(np.abs(objective).max())),
        np.asarray(problem.lower), np.asarray(problem.upper),
        np.asarray(problem.integrality),
        np.arange(len(objective)) < problem.n_x,
        (np.arange(len(objective)) >= problem.n_x) &
        (np.arange(len(objective)) < 2 * problem.n_x),
        np.arange(len(objective)) >= 2 * problem.n_x,
        np.arange(len(objective)) % problem.T / max(1, problem.T - 1),
    ])
    return dict(edge_index=torch.tensor(np.stack([rows, columns]), dtype=torch.long),
                coefficients=torch.tensor(matrix.data / scale[rows], dtype=torch.float32)[:, None],
                variable_static=torch.tensor(static_vars, dtype=torch.float32),
                row_lower=torch.tensor(row_lower, dtype=torch.float32),
                row_upper=torch.tensor(row_upper, dtype=torch.float32),
                finite_lower=torch.tensor(finite_lower), finite_upper=torch.tensor(finite_upper),
                degree=torch.tensor(np.log1p(degree), dtype=torch.float32),
                task_index=torch.arange(2 * problem.n_x) % problem.n_x // problem.T)


def lns_observation(env):
    """Snapshot dynamic values; build/cache topology inside the paid search clock."""
    problem = env.problem
    cached = getattr(env, '_rl_lns_topology', None)
    if cached is None or cached[0] is not problem:
        cached = (problem, _topology(problem))
        env._rl_lns_topology = cached
    graph = cached[1]
    assignment = torch.tensor(problem.assignment(env.current), dtype=torch.float32)
    rows, columns = graph['edge_index']
    activity = torch.zeros(len(graph['row_lower']))
    activity.index_add_(0, rows, graph['coefficients'][:, 0] * assignment[columns])
    low, high = graph['row_lower'], graph['row_upper']
    has_low, has_high = graph['finite_lower'], graph['finite_upper']
    constraints = torch.stack([activity, low, high,
                               torch.where(has_low, activity - low, 0.),
                               torch.where(has_high, high - activity, 0.),
                               has_low.float(), has_high.float(), graph['degree']], 1)
    variables = torch.cat([graph['variable_static'][:, :3], assignment[:, None],
                           graph['variable_static'][:, 3:]], 1)
    context = torch.as_tensor(env.search_context(), dtype=torch.float32).detach().cpu().clone()
    if context.shape != (10,) or not torch.isfinite(context).all():
        raise ValueError('Expected ten finite search-context features')
    observation = LNSObservation(variables.detach().clone(), constraints.detach().clone(),
                                 graph['edge_index'].clone(), graph['coefficients'].clone(),
                                 graph['task_index'].clone(), context, problem.J)
    return observation, env.subset_action_mask().detach().cpu().clone()


class RLLNSActorCritic(nn.Module):
    """Signed coefficient messages in both directions; task-block Bernoulli actor."""

    def __init__(self, hidden=24, layers=1):
        super().__init__()
        if not isinstance(hidden, int) or hidden < 1 or not isinstance(layers, int) or layers < 1:
            raise ValueError('hidden and layers must be positive integers')
        self.hidden, self.layers = hidden, layers
        self.variable_encoder = nn.Sequential(nn.Linear(9, hidden), nn.Tanh())
        self.constraint_encoder = nn.Sequential(nn.Linear(8, hidden), nn.Tanh())
        self.to_constraint = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.Tanh())
                                          for _ in range(layers))
        self.to_variable = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.Tanh())
                                        for _ in range(layers))
        self.constraint_update = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh())
                                              for _ in range(layers))
        self.variable_update = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh())
                                            for _ in range(layers))
        self.actor = nn.Sequential(nn.Linear(3 * hidden + 10, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(2 * hidden + 10, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.zeros_(self.actor[-1].bias)

    def observation(self, env):
        return lns_observation(env)

    @staticmethod
    def _aggregate(messages, indices, count):
        result = messages.new_zeros((count, messages.shape[1]))
        result.index_add_(0, indices, messages)
        degree = torch.bincount(indices, minlength=count).to(messages.dtype).clamp(min=1)
        return result / degree[:, None]

    def forward(self, observation, mask):
        device = next(self.parameters()).device
        variable = self.variable_encoder(observation.variables.to(device))
        constraint = self.constraint_encoder(observation.constraints.to(device))
        rows, columns = observation.edge_index.to(device)
        coefficient = observation.coefficients.to(device)
        for vc, cv, cu, vu in zip(self.to_constraint, self.to_variable,
                                  self.constraint_update, self.variable_update):
            messages = vc(torch.cat([variable[columns], constraint[rows], coefficient], 1))
            pooled = self._aggregate(messages, rows, len(constraint))
            constraint = cu(torch.cat([constraint, pooled], 1))
            messages = cv(torch.cat([constraint[rows], variable[columns], coefficient], 1))
            pooled = self._aggregate(messages, columns, len(variable))
            variable = vu(torch.cat([variable, pooled], 1))
        task_index = observation.task_index.to(device)
        tasks = self._aggregate(variable[:len(task_index)], task_index, observation.jobs)
        global_state = torch.cat([variable.mean(0), constraint.mean(0), observation.context.to(device)])
        logits = self.actor(torch.cat([tasks, global_state.expand(observation.jobs, -1)], 1)).squeeze(-1)
        return NonemptySubset(logits.clamp(-15, 15), stopped=bool(mask[0])), self.critic(global_state).squeeze(-1)

    def save(self, path, metadata=None):
        torch.save(dict(format='onts-rl-lns-v1', hidden=self.hidden, layers=self.layers,
                        state_dict=self.state_dict(), metadata=metadata or {}), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('format') != 'onts-rl-lns-v1':
            raise ValueError('Unsupported RL-LNS checkpoint')
        model = cls(data['hidden'], data['layers'])
        model.load_state_dict(data['state_dict'])
        return model.eval()
