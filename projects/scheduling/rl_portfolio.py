"""Deadline-validated RL control of task-profile and MILP search actions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .job_profile import reprofile, select_jobs
from .milp import ONTSMilp
from .ranking import SearchState, make_candidates

FAMILIES = ('milp_random', 'milp_utility', 'milp_run', 'milp_energy',
            'profile_pressure', 'profile_utility', 'profile_random')
ROW_WIDTH = 32


@dataclass
class PortfolioAction:
    family: int
    mask: torch.Tensor
    features: torch.Tensor
    jobs: tuple[int, ...] = ()
    allowance: float = 0.0


@dataclass
class PortfolioObservation:
    rows: torch.Tensor
    tasks: torch.Tensor
    times: torch.Tensor
    released: torch.Tensor


class PortfolioEnv:
    """A common-incumbent episode with action-dependent runtime allowances.

    Profile actions choose a selection rule and one to three jobs. MILP actions
    choose a candidate family and a repair allowance. All candidates are checked
    before acceptance; late or decreasing results leave the incumbent unchanged.
    """
    def __init__(self, instance, initial_schedule, *, budget=1.2,
                 repair_budgets=(0.025, 0.1), seed=0, config=None,
                 clock=perf_counter, profile_floor=0.002):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('budget must be positive and finite')
        if not repair_budgets or any(not math.isfinite(v) or v <= 0 for v in repair_budgets):
            raise ValueError('repair allowances must be positive and finite')
        if not math.isfinite(profile_floor) or profile_floor <= 0:
            raise ValueError('profile_floor must be positive and finite')
        self.clock, self.started = clock, clock()
        self.budget, self.repair_budgets = float(budget), tuple(repair_budgets)
        self.profile_floor = profile_floor
        self.reserve = min(.05, budget*.05)
        self.deadline = self.started+budget-self.reserve
        self.problem = ONTSMilp(instance, config)
        report = self.problem.config.validate(instance, initial_schedule)
        if not report.feasible:
            raise ValueError('A feasible initial schedule is required')
        self.incumbent = initial_schedule.detach().cpu().bool().clone()
        self.objective = self.initial_objective = report.objective
        self.scale = max(float(instance.priority.clamp(min=0).sum())*instance.T, 1.)
        self.rng = np.random.default_rng(seed)
        # Action sampling must not consume the proposal-generation stream.
        self.selection_rng = np.random.default_rng(seed+982451653)
        self.state = SearchState.empty(instance.J, instance.T)
        self.pulls, self.gains, self.durations = [np.zeros(7) for _ in range(3)]
        self.last_used = np.full(7, -1)
        self.log, self.actions = [], []
        self.done = False
        self.refresh()

    def remaining(self):
        return max(0., self.deadline-self.clock())

    def refresh(self):
        p, x = self.problem, self.incumbent
        proposals = make_candidates(p, x, self.state, self.rng, count=4)
        self.actions = [PortfolioAction(family, c.mask.clone(), c.features.clone(),
                                        allowance=allowance)
                        for family, c in enumerate(proposals)
                        for allowance in self.repair_budgets]
        for rule_index, rule in enumerate(('pressure', 'utility', 'random')):
            for count in range(1, min(3, p.J)+1):
                jobs = tuple(select_jobs(p, x, self.rng, count, rule=rule))
                mask = torch.zeros_like(x, dtype=torch.bool)
                mask[list(jobs)] = True
                chosen = max(int(mask.sum()), 1)
                # Match the original ten candidate feature slots; action family
                # is separately represented by a seven-way one-hot vector.
                features = torch.tensor([
                    0., 0., 0., 0., mask.float().mean(),
                    (mask & x).sum()/chosen, count/p.J, 1.,
                    (mask & self.state.last_mask).sum()/chosen, 0.,
                ], dtype=torch.float32)
                self.actions.append(PortfolioAction(4+rule_index, mask, features, jobs))
        self.done = self.remaining() < self.profile_floor

    def action_mask(self):
        remaining = self.remaining()
        active = [not self.done and remaining >= (a.allowance if a.family < 4 else self.profile_floor)
                  for a in self.actions]
        return torch.tensor([not any(active), *active], dtype=torch.bool)

    def observe(self):
        p, x = self.problem, self.incumbent
        report = p.config.validate(p.instance, x)
        context = torch.tensor([
            self.objective/self.scale, self.remaining()/self.budget,
            self.state.stagnation/(1+self.state.stagnation),
            self.state.steps/(1+self.state.steps), self.state.last_gain/self.scale,
            self.state.last_runtime/self.budget, min(report.soc), np.mean(report.soc),
        ], dtype=torch.float32)
        stop = torch.cat([torch.zeros(19), context, torch.zeros(5)])
        rows = [stop]
        for a in self.actions:
            f, pulls = a.family, max(self.pulls[a.family], 1)
            onehot = torch.nn.functional.one_hot(torch.tensor(f), 7).float()
            history = torch.tensor([
                self.gains[f]/pulls, self.durations[f]/pulls/self.budget,
                self.pulls[f]/max(self.state.steps, 1),
                1. if self.last_used[f] < 0 else (self.state.steps-self.last_used[f])/(1+self.state.steps),
            ], dtype=torch.float32)
            estimated = (self.durations[f]/pulls if self.pulls[f]
                         else a.allowance if f < 4 else self.profile_floor*len(a.jobs))
            rows.append(torch.cat([onehot, a.features,
                                   torch.tensor([a.allowance/self.budget, len(a.jobs)/p.J]),
                                   context, history, torch.tensor([estimated/self.budget], dtype=torch.float32)]))
        return torch.stack(rows), self.action_mask()

    def observation(self):
        p, x, inst = self.problem, self.incumbent, self.problem.instance
        starts = x & ~torch.cat([torch.zeros(p.J, 1, dtype=torch.bool), x[:, :-1]], 1)
        count = starts.sum(1).float()
        last_start = torch.where(starts, torch.arange(p.T), -1).max(1).values.float()
        tasks = torch.stack([
            inst.priority.float()/inst.priority.max().clamp(min=1),
            inst.uso_p.float()/inst.uso_p.max().clamp(min=1),
            inst.min_statup.float()/inst.max_statup.clamp(min=1),
            inst.max_statup.float()/(1+inst.max_statup), count/inst.max_statup.clamp(min=1),
            inst.min_cpu_time.float()/p.T, inst.max_cpu_time.float()/p.T,
            inst.min_periodo_job.float()/p.T, inst.max_periodo_job.float()/p.T,
            inst.win_min.float()/p.T, inst.win_max.float()/p.T, x.float().mean(1),
            self.state.last_mask.float().mean(1), self.state.changed.float().mean(1),
            last_start/p.T,
        ], 1).float()
        solar = inst.recurso_p.float()
        usage = inst.uso_p.float() @ x.float()
        soc = torch.tensor(p.config.validate(inst, x).soc, dtype=torch.float32)
        solar_scale = solar.max().clamp(min=1)
        future = solar.flip(0).cumsum(0).flip(0)/torch.arange(p.T, 0, -1)
        times = torch.stack([
            solar/solar_scale, usage/solar_scale, soc,
            future/solar_scale, (soc-p.config.limite_inferior),
            torch.arange(p.T)/max(p.T-1, 1), x.float().mean(0),
            self.state.last_mask.float().mean(0), self.state.changed.float().mean(0),
        ], 1)
        rows, mask = self.observe()
        released = torch.stack([torch.zeros_like(x), *[a.mask for a in self.actions]])
        return PortfolioObservation(rows, tasks, times, released), mask

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode has ended')
        if not isinstance(action, int) or not 0 <= action <= len(self.actions):
            raise ValueError('Invalid action')
        available = self.action_mask()
        if action == 0 and not available[0]:
            raise ValueError('STOP is masked while an action is affordable')
        before, started = self.objective, self.clock()
        status, accepted, family, info = 'exhausted', False, 'stop', {}
        executed = False
        if action == 0 or not available[action]:
            self.done = action == 0 or not bool(available[1:].any())
            status = 'exhausted' if self.done else 'expired_action'
        else:
            a = self.actions[action-1]
            family = FAMILIES[a.family]
            previous = self.incumbent.clone()
            executed = True
            if a.family < 4:
                result = self.problem.solve(time_limit=a.allowance, incumbent=previous,
                                            free_mask=a.mask, solver_threads=1)
                candidate, status = result.schedule, result.status
            else:
                candidate, _, info = reprofile(self.problem, previous, a.jobs, self.rng,
                                               deadline=self.deadline)
                status = 'profile'
            if candidate is not None:
                report = self.problem.config.validate(self.problem.instance, candidate)
                if report.feasible and report.objective >= before and self.clock() <= self.deadline:
                    self.incumbent = candidate.detach().cpu().bool().clone()
                    self.objective = report.objective
                    accepted = True
            self.state.advance(a.mask, previous, self.incumbent,
                               self.objective-before, self.clock()-started)
            self.pulls[a.family] += 1
            self.gains[a.family] += (self.objective-before)/self.scale
            self.durations[a.family] += self.clock()-started
            self.last_used[a.family] = self.state.steps
        reward = (self.objective-before)/self.scale
        self.log.append(dict(action=action, family=family, executed=executed, status=status,
                             accepted=accepted, objective=self.objective, reward=reward,
                             duration=self.clock()-started, elapsed=self.clock()-self.started,
                             jobs=list(self.actions[action-1].jobs) if action else [],
                             allowance=self.actions[action-1].allowance if action else 0.,
                             **{k: v for k, v in info.items() if k not in ('gain', 'jobs')}))
        if not self.done:
            self.refresh()
        return reward, self.done


class PortfolioActorCritic(nn.Module):
    """Task encoding, temporal attention, and candidate attention."""
    def __init__(self, hidden=32, heads=4):
        super().__init__()
        if hidden < 1 or heads < 1 or hidden % heads:
            raise ValueError('hidden must be positive and divisible by heads')
        self.hidden, self.heads = hidden, heads
        self.row_input = nn.Linear(ROW_WIDTH, hidden)
        self.task_input = nn.Sequential(nn.Linear(15, hidden), nn.GELU())
        self.time_input = nn.Linear(9, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, 2*hidden, dropout=0.,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        layer = nn.TransformerEncoderLayer(hidden, heads, 2*hidden, dropout=0.,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.candidate = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.actor, self.critic = nn.Linear(hidden, 1), nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def observation(self, env):
        return env.observation()

    def forward(self, observation, mask):
        device = next(self.parameters()).device
        rows, tasks, times, released = (t.to(device) for t in
            (observation.rows, observation.tasks, observation.times, observation.released))
        task = self.task_input(tasks)
        time = self.temporal(self.time_input(times).unsqueeze(0)).squeeze(0)
        task_weight = released.any(2).float()
        time_weight = released.any(1).float()
        task_pool = task_weight @ task/task_weight.sum(1, keepdim=True).clamp(min=1)
        time_pool = time_weight @ time/time_weight.sum(1, keepdim=True).clamp(min=1)
        hidden = self.row_input(rows)+task_pool+time_pool+task.mean(0)+time.mean(0)
        available = mask.to(device)
        hidden = self.candidate(hidden.unsqueeze(0), src_key_padding_mask=(~available).unsqueeze(0)).squeeze(0)
        logits = self.actor(hidden).squeeze(-1).masked_fill(~available, -torch.inf)
        return Categorical(logits=logits), self.critic(hidden[0]).squeeze(-1)

    def save(self, path: str | Path, *, metadata=None):
        torch.save(dict(format='onts-rl-portfolio-v1', hidden=self.hidden, heads=self.heads,
                        state_dict=self.state_dict(), metadata=metadata or {}), path)

    @classmethod
    def load(cls, path: str | Path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('format') != 'onts-rl-portfolio-v1':
            raise ValueError('Unsupported portfolio checkpoint')
        model = cls(checkpoint['hidden'], checkpoint['heads'])
        model.load_state_dict(checkpoint['state_dict'])
        model.training_metadata = checkpoint.get('metadata', {})
        model.eval()
        return model


def portfolio_rollout(env, policy=None, *, mode='ppo', greedy=False):
    trajectory = []
    matrix, vector = np.eye(ROW_WIDTH), np.zeros(ROW_WIDTH)
    while not env.done:
        started = env.clock()
        if mode == 'ppo':
            observation, mask = policy.observation(env)
            with torch.no_grad():
                dist, value = policy(observation, mask)
                selected = dist.logits.argmax() if greedy else dist.sample()
                action, logp, value = int(selected), float(dist.log_prob(selected)), float(value)
        else:
            observation, mask = env.observe()
            choices = torch.where(mask)[0].tolist()
            if choices == [0]:
                action = 0
            elif mode == 'uniform':
                action = int(env.selection_rng.choice(choices))
            elif mode == 'bandit':
                x = observation.numpy()
                inverse = np.linalg.inv(matrix)
                scores = x @ inverse @ vector + .1*np.sqrt(np.maximum(
                    np.einsum('ij,jk,ik->i', x, inverse, x), 0))
                action = max(choices, key=lambda a: scores[a])
            elif mode == 'profile_only':
                profiles = [i for i in choices if i and env.actions[i-1].family >= 4]
                action = int(env.selection_rng.choice(profiles or choices))
            else:
                raise ValueError('Unknown portfolio selector')
            logp, value = 0., 0.
        selection_runtime = env.clock()-started
        reward, done = env.step(action)
        env.log[-1]['selection_runtime'] = selection_runtime
        if mode == 'bandit':
            x = observation[action].numpy().astype(float)
            # Rate-based myopic utility is a comparator; PPO rewards remain gains.
            utility = reward/max(env.log[-1]['duration']+selection_runtime, .002)
            matrix += np.outer(x, x)
            vector += min(1., utility)*x
        trajectory.append((observation, mask, action, logp, value, reward))
    return trajectory
