"""Shared ONTS improvement search and classical/learned adaptive LNS.

DR-ALNS-inspired search control, not a routing-paper reproduction. All current
states remain feasible; best-so-far output is separate from annealed working state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .job_profile import best_profile, job_rules, reprofile, select_jobs
from .milp import ONTSMilp


@dataclass(frozen=True)
class SearchAction:
    jobs: tuple[int, ...]
    operator: str
    allowance: float
    temperature: float
    rule: str


@dataclass
class SearchObservation:
    tasks: torch.Tensor
    times: torch.Tensor
    edges: torch.Tensor
    eligible: torch.Tensor
    groups: torch.Tensor
    action_features: torch.Tensor
    context: torch.Tensor
    current_schedule: torch.Tensor
    history_schedules: torch.Tensor
    history_groups: torch.Tensor
    history_gains: torch.Tensor
    history_objectives: torch.Tensor
    history_durations: torch.Tensor


class LiteratureSearchEnv:
    """One wall-clock budget, frozen candidate proposals, best-increment reward.

    Profile/MIP repairs are monotone from current. Random per-task DP profiles
    provide genuinely different feasible candidates, allowing temporary sacrifice.
    Candidate batches are repaired from one frozen current state. All sampled
    proposals remain part of the action even if later ones expire.
    """

    def __init__(self, instance, initial_schedule, *, budget=2., seed=0,
                 repair_budget=.05, temperatures=(0., .003, .015),
                 memory_size=32, config=None, clock=perf_counter):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('budget must be finite and positive')
        if not math.isfinite(repair_budget) or repair_budget <= 0:
            raise ValueError('repair_budget must be finite and positive')
        if not temperatures or any(not math.isfinite(t) or t < 0 for t in temperatures):
            raise ValueError('temperatures must be finite and nonnegative')
        if not isinstance(memory_size, int) or memory_size < 0:
            raise ValueError('memory_size must be a nonnegative integer')
        self.clock, self.started = clock, clock()
        self.budget, self.repair_budget = float(budget), float(repair_budget)
        self.reserve = min(.05, .05*budget)
        self.deadline = self.started+budget-self.reserve
        self.problem = ONTSMilp(instance, config)
        checked = self.problem.config.validate(instance, initial_schedule)
        if not checked.feasible:
            raise ValueError('A feasible initial schedule is required')
        self.current = initial_schedule.detach().cpu().bool().clone()
        self.best = self.current.clone()
        self.current_objective = self.best_objective = self.initial_objective = checked.objective
        self.scale = max(float(instance.priority.clamp(min=0).sum())*instance.T, 1.)
        self.temperatures = tuple(float(t) for t in temperatures)
        self.memory_size = memory_size
        self.rng = np.random.default_rng(seed)
        self.selection_rng = np.random.default_rng(seed+982451653)
        self.steps = self.stagnation = 0
        self.last_gain = self.last_runtime = 0.
        self.last_group = torch.zeros(instance.J, dtype=torch.bool)
        self.last_changed = torch.zeros_like(self.current)
        self.history, self.log, self.actions = [], [], []
        self.refresh()

    @property
    def incumbent(self):
        return self.best

    @property
    def objective(self):
        return self.best_objective

    @property
    def done(self):
        return self.remaining() < .002

    def remaining(self):
        return max(0., self.deadline-self.clock())

    def search_context(self):
        report = self.problem.config.validate(self.problem.instance, self.current)
        return torch.tensor([
            self.current_objective/self.scale, self.best_objective/self.scale,
            self.remaining()/self.budget, (self.best_objective-self.current_objective)/self.scale,
            self.stagnation/(1+self.stagnation), self.steps/(1+self.steps),
            self.last_gain, self.last_runtime/self.budget, min(report.soc), np.mean(report.soc),
        ], dtype=torch.float32)

    def refresh(self):
        J = self.problem.J
        counts = sorted({1, min(2,J), max(1,math.ceil(J/3)), max(1,math.ceil(J/2))})
        actions, seen = [], set()
        for rule in ('pressure', 'utility', 'random'):
            for count in counts:
                jobs = tuple(sorted(select_jobs(self.problem, self.current, self.rng, count, rule=rule)))
                for operator in ('profile', 'milp', 'perturb'):
                    for temperature in self.temperatures:
                        key = (jobs,operator,temperature)
                        if key in seen: continue
                        seen.add(key)
                        actions.append(SearchAction(jobs,operator,self.repair_budget if operator=='milp' else 0.,temperature,rule))
        self.actions = actions

    def action_mask(self):
        remaining = self.remaining()
        return torch.tensor([remaining >= (a.allowance if a.operator=='milp' else .002) for a in self.actions])

    def subset_action_mask(self):
        return torch.tensor([self.remaining() < self.repair_budget])

    def observe(self):
        inst, x, J, T = self.problem.instance, self.current, self.problem.J, self.problem.T
        starts = x & ~torch.cat([torch.zeros(J,1,dtype=torch.bool),x[:,:-1]],1)
        last_start = torch.where(starts,torch.arange(T),-1).max(1).values.float()
        tasks = torch.stack([
            inst.priority.float()/inst.priority.max().clamp(min=1),
            inst.uso_p.float()/inst.uso_p.max().clamp(min=1),
            inst.min_statup.float()/inst.max_statup.clamp(min=1),
            inst.max_statup.float()/(1+inst.max_statup),starts.sum(1).float()/inst.max_statup.clamp(min=1),
            inst.min_cpu_time.float()/T,inst.max_cpu_time.float()/T,
            inst.min_periodo_job.float()/T,inst.max_periodo_job.float()/T,
            inst.win_min.float()/T,inst.win_max.float()/T,x.float().mean(1),
            self.last_group.float(),self.last_changed.float().mean(1),last_start/T,
        ],1).float()
        solar, draw = inst.recurso_p.float(), inst.uso_p.float()
        usage = draw@x.float()
        soc = torch.tensor(self.problem.config.validate(inst,x).soc,dtype=torch.float32)
        scale = max(float(solar.max()),float(draw.max()),1.)
        future = solar.flip(0).cumsum(0).flip(0)/torch.arange(T,0,-1)
        times = torch.stack([solar/scale,usage/scale,soc,future/scale,
            soc-self.problem.config.limite_inferior,torch.arange(T)/max(T-1,1),
            x.float().mean(0),self.last_group.float().mean().expand(T),self.last_changed.float().mean(0)],1)
        slots = torch.arange(T)
        eligible = (slots>=inst.win_min[:,None])&(slots<inst.win_max[:,None])
        edges = torch.stack([x.float(),starts.float(),self.last_group.float()[:,None].expand(J,T),
            self.last_changed.float(),(usage[None]-draw[:,None]*x.float())/scale,
            draw[:,None].expand(J,T)/scale],-1)*eligible[...,None]
        groups = torch.zeros(len(self.actions),J,dtype=torch.bool)
        features = []
        for index,a in enumerate(self.actions):
            groups[index,list(a.jobs)] = True
            group = groups[index]
            features.append([
                float(a.operator=='profile'),float(a.operator=='milp'),float(a.operator=='perturb'),
                len(a.jobs)/J,float(x[group].float().mean()),float(tasks[group,0].mean()),
                float(tasks[group,1].mean()),a.allowance/self.budget,a.temperature,
                float(a.rule=='pressure'),float(a.rule=='utility'),float(a.rule=='random')])
        H = len(self.history)
        histories = [torch.stack([r[k] for r in self.history]) if H else empty for k,empty in (
            ('schedule',torch.empty(0,J,T,dtype=torch.bool)),('group',torch.empty(0,J,dtype=torch.bool)),
            ('gain',torch.empty(0)),('objective',torch.empty(0)),('duration',torch.empty(0)))]
        return SearchObservation(tasks,times,edges,eligible,groups,torch.tensor(features,dtype=torch.float32),
            self.search_context(),x.clone(),*histories), self.action_mask()

    def _repair(self, action, base):
        if action.operator=='milp':
            free = torch.zeros_like(base);free[list(action.jobs)] = True
            result = self.problem.solve(time_limit=min(action.allowance,self.remaining()),
                incumbent=base,free_mask=free,solver_threads=1)
            return result.schedule,result.status
        if action.operator=='profile':
            candidate,_,_ = reprofile(self.problem,base,action.jobs,self.rng,deadline=perf_counter()+self.remaining())
            return candidate,'profile'
        candidate = base.clone()
        # Per-job randomized objective preserves the exact run/start/window rules.
        # Full energy/battery feasibility is enforced after every replacement.
        for j in self.rng.permutation(action.jobs):
            if self.clock()>self.deadline: break
            reward = self.rng.normal(-.15,1.,self.problem.T)
            row,_ = best_profile(reward,job_rules(self.problem,int(j)),
                truncated_final_run=self.problem.config.allow_truncated_final_run)
            if row is None: continue
            trial = candidate.clone();trial[int(j)] = torch.from_numpy(row)
            if self.problem.config.validate(self.problem.instance,trial).feasible:
                candidate = trial
        return candidate,'perturb'

    def step_actions(self, indices):
        indices = torch.as_tensor(indices)
        if indices.ndim==0: indices = indices.reshape(1)
        if indices.ndim!=1 or not len(indices) or indices.dtype==torch.bool or indices.is_floating_point():
            raise ValueError('Expected integer proposal indices')
        if ((indices<0)|(indices>=len(self.actions))).any():
            raise ValueError('Invalid candidate index')
        return self._step([self.actions[int(i)] for i in indices],indices.tolist())

    def step_subset(self, bits):
        bits = torch.as_tensor(bits)
        if bits.shape!=(self.problem.J,) or not ((bits==0)|(bits==1)).all() or not bits.any():
            raise ValueError('Select a nonempty binary task set')
        action = SearchAction(tuple(torch.where(bits)[0].tolist()),'milp',self.repair_budget,0.,'learned')
        return self._step([action],bits.long().tolist())

    def _step(self, actions, sampled):
        if self.done: raise RuntimeError('Episode has ended')
        tick = self.clock()
        base, before = self.current.clone(), self.current_objective
        best_before = self.best_objective
        proposals, usable = [], []
        for action in actions:
            start = self.clock()
            minimum = action.allowance if action.operator=='milp' else .002
            if self.remaining()<minimum:
                proposals.append(dict(jobs=list(action.jobs),operator=action.operator,temperature=action.temperature,
                    executed=False,status='expired',schedule=None,objective=None,runtime=0.,eligible=False))
                continue
            candidate,status = self._repair(action,base)
            report = None if candidate is None else self.problem.config.validate(self.problem.instance,candidate)
            checked_at = self.clock()
            eligible = report is not None and report.feasible and checked_at<=self.deadline
            record = dict(jobs=list(action.jobs),operator=action.operator,temperature=action.temperature,
                executed=True,status=status,feasible=bool(report and report.feasible),
                schedule=candidate.int().tolist() if report and report.feasible else None,
                objective=report.objective if report and report.feasible else None,
                runtime=checked_at-start,checked_elapsed=checked_at-self.started,eligible=eligible)
            proposals.append(record)
            if eligible: usable.append((report.objective,candidate.clone(),action))
            group = torch.zeros(self.problem.J,dtype=torch.bool);group[list(action.jobs)]=True
            if self.memory_size:
                self.history.append(dict(schedule=base.clone(),group=group,
                    gain=torch.tensor((report.objective-before)/self.scale if eligible else 0.),
                    objective=torch.tensor(before/self.scale),duration=torch.tensor(record['runtime']/self.budget)))
                self.history = self.history[-self.memory_size:]
        accepted, acceptance_probability = False, 0.
        selected = None
        accepted_elapsed = None
        if usable:
            candidate_value,candidate,selected = max(usable,key=lambda r:r[0])
            # Every validated evaluation can improve the output, regardless of working-state acceptance.
            if candidate_value>self.best_objective and self.clock()<=self.deadline:
                self.best,self.best_objective = candidate.clone(),candidate_value
            delta = (candidate_value-before)/self.scale
            acceptance_probability = 1. if delta>=0 else (math.exp(delta/selected.temperature) if selected.temperature>0 else 0.)
            acceptance_draw = self.rng.random()
            acceptance_tick = self.clock()
            if acceptance_tick<=self.deadline and acceptance_draw<acceptance_probability:
                self.current,self.current_objective = candidate.clone(),candidate_value
                accepted,accepted_elapsed = True,acceptance_tick-self.started
                self.last_group.zero_();self.last_group[list(selected.jobs)] = True
        self.last_changed = base!=self.current
        self.steps += 1
        self.last_gain = (self.best_objective-best_before)/self.scale
        self.stagnation = 0 if self.last_gain>0 else self.stagnation+1
        self.last_runtime = self.clock()-tick
        self.log.append(dict(step=self.steps,action=sampled,proposals=proposals,
            before_current=before,before_best=best_before,current_objective=self.current_objective,
            best_objective=self.best_objective,reward=self.last_gain,accepted=accepted,
            accepted_elapsed=accepted_elapsed,acceptance_probability=acceptance_probability,
            current_schedule=self.current.int().tolist(),best_schedule=self.best.int().tolist(),
            elapsed=self.clock()-self.started))
        if not self.done:
            self.refresh()
        self.log[-1]['elapsed'] = self.clock()-self.started
        return self.last_gain,self.done


class DRALNSActorCritic(nn.Module):
    """Compact PPO scores operator/severity/temperature actions from search status."""
    def __init__(self, hidden=24):
        super().__init__()
        if not isinstance(hidden,int) or hidden<1: raise ValueError('hidden must be positive')
        self.hidden = hidden
        self.actor = nn.Sequential(nn.Linear(22,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.critic = nn.Sequential(nn.Linear(10,hidden),nn.Tanh(),nn.Linear(hidden,1))
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].weight);nn.init.zeros_(self.critic[-1].bias)

    def observation(self,env): return env.observe()

    def forward(self,observation,mask):
        device = next(self.parameters()).device
        context = observation.context.to(device)
        mask = mask.to(device)
        if not mask.any(): raise ValueError('No affordable actions')
        logits = self.actor(torch.cat([observation.action_features.to(device),context.expand(len(mask),-1)],1)).squeeze(-1)
        return Categorical(logits=logits.masked_fill(~mask,-torch.inf)),self.critic(context).squeeze(-1)

    def save(self,path,metadata=None):
        torch.save(dict(format='onts-dr-alns-v1',hidden=self.hidden,state_dict=self.state_dict(),metadata=metadata or {}),Path(path))

    @classmethod
    def load(cls,path):
        saved = torch.load(path,map_location='cpu',weights_only=True)
        if saved['format']!='onts-dr-alns-v1': raise ValueError('Unsupported checkpoint')
        model = cls(saved['hidden']);model.load_state_dict(saved['state_dict']);return model.eval()


class AdaptiveLNS:
    """Classical reactive roulette LNS with fixed cooling and segmented rewards.

    Weights track (selection rule, repair operator, destruction size); temperature
    is fixed by a cooling schedule, not selected to exploit neural controls.
    """
    def __init__(self, *, temperature=.015, reaction=.2, segment=8):
        if not math.isfinite(temperature) or temperature<0 or not 0<reaction<=1 or segment<1:
            raise ValueError('Invalid adaptive LNS settings')
        self.temperature,self.reaction,self.segment = temperature,reaction,segment
        self.weights,self.scores,self.pulls = {},{},{}
        self.iterations = 0

    @staticmethod
    def key(action): return action.rule,action.operator,len(action.jobs)

    def select(self,env,mask,*,adaptive=True):
        available = torch.where(mask)[0].tolist()
        if not available: raise ValueError('No affordable actions')
        target = self.temperature*max(.05,env.remaining()/env.budget)
        representatives = {}
        for i in available:
            key = self.key(env.actions[i])
            if key not in representatives or abs(env.actions[i].temperature-target)<abs(env.actions[representatives[key]].temperature-target):
                representatives[key]=i
        choices = list(representatives.values())
        weights = np.array([self.weights.get(self.key(env.actions[i]),1.) if adaptive else 1. for i in choices])
        index = int(env.selection_rng.choice(choices,p=weights/weights.sum()))
        # SearchAction is frozen; apply fixed classical cooling to this choice.
        a = env.actions[index]
        env.actions[index] = SearchAction(a.jobs,a.operator,a.allowance,target,a.rule)
        return index,self.key(a)

    def update(self,key,record):
        score = 5. if record['reward']>0 else 2. if record['current_objective']>record['before_current'] else 1. if record['accepted'] else 0.
        self.scores[key] = self.scores.get(key,0.)+score
        self.pulls[key] = self.pulls.get(key,0)+1
        self.iterations+=1
        if self.iterations%self.segment==0:
            for arm,count in self.pulls.items():
                self.weights[arm] = max(.05,(1-self.reaction)*self.weights.get(arm,1.)+self.reaction*self.scores[arm]/count)
            self.scores.clear();self.pulls.clear()


def literature_rollout(env,policy=None,*,subset=False,adaptive=None,max_steps=None,collect=True):
    """Record executed action likelihood, including every MIStar proposal draw."""
    trajectory = []
    while not env.done and (max_steps is None or env.steps<max_steps):
        start = env.clock()
        if policy is None:
            mask = env.action_mask()
            if not mask.any(): break
            controller = adaptive or AdaptiveLNS(temperature=0.)
            action,key = controller.select(env,mask,adaptive=adaptive is not None)
            selection = env.clock()-start
            if env.done: break
            reward,_ = env.step_actions(action)
            if adaptive is not None: adaptive.update(key,env.log[-1])
        else:
            observation,mask = policy.observation(env)
            if (subset and mask[0]) or (not subset and not mask.any()): break
            with torch.no_grad():
                dist,value = policy(observation,mask)
                action = dist.sample()
                logp = float(dist.log_prob(action)) if collect else None
                estimate = float(value) if collect else None
            selection = env.clock()-start
            if env.done: break
            reward,_ = env.step_subset(action) if subset else env.step_actions(action)
            if collect:
                trajectory.append((observation,mask.clone(),action.tolist(),logp,estimate,reward))
        env.log[-1]['selection_runtime'] = selection
    return trajectory
